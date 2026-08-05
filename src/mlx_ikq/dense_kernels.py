"""Metal kernels for the dense-tensor IQ_K members.

Two kernel families serve one dense ``[out_features, in_features]`` matrix,
for any of ``IQ4_KS``, ``IQ4_K``, ``IQ5_K``, ``IQ6_K``:

- ``dense_gemv_kernel``: the decode route. One threadgroup walks a fixed
  number of output rows of one token, streaming the k-contiguous relayout at
  its address rate. The structure is the routed decode GEMV's with the
  expert-select machinery removed: a token indexes its own activation row
  and the output row id is the storage row id directly.
- ``dense_dequant_kernel`` and ``dense_dequant_range_kernel``: the prefill
  weight producers. One thread decodes 32 consecutive weights of one row and
  stores fp16; the range form covers ``range_rows`` output rows from a
  runtime row offset, bounding in-flight buffer bytes exactly as the stacked
  range form does.

The alphabet handling differs from the 2-bit kernels, deliberately. Those
stage each sub-block's four table entries into registers because at 2.2 bits
per weight a dynamically indexed threadgroup load prices above the thin
address stream. At 16, 32, or 64 entries per alphabet that staging would be
a select tree per weight, and the address streams here (4.25 to 6.625 bits
per weight) hide far more decode: the measured 3.4375-bit decode with a
per-weight dynamically indexed table read on this relayout priced at 0.997x
its address floor. Each kernel therefore stages the member's whole value
table into threadgroup memory once and folds the per-sub-block alphabet
select into a register base offset, so the per-weight work is one shift, one
mask, one threadgroup load, one fma.

Table storage is ``half`` for the integer-valued alphabets (``IQ4_KS`` and
``IQ4_K`` share ``iq4k_values``; ``IQ5_K`` uses ``iq5nl_values``; every
entry is a small integer, exact in fp16) and ``float`` for ``IQ6_K``, whose
served table is the cubic reconstruction's float32 outputs and is not
fp16-exact.

Geometry is baked into the source, so an address computation is a shift and
never a runtime division. The sized input widths are exactly the dense
widths of the target model plus none: a decode threadgroup makes one pass
over the reduction axis, so an unsized width would leave part of a row
unread and return a wrong answer; the generator refuses it instead.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_ikq.format import SUB_WEIGHTS, check_dense_member, check_row_width
from mlx_ikq.kernels import (
    ROWS_PER_TG,
    WEIGHTS_PER_SIMDGROUP,
    IkqKernelError,
    _log2,
    _xload,
    simdgroups,
)

DENSE_SUPPORTED_IN_FEATURES = (1024, 2048, 4096, 8192)
"""Input widths with sized dense kernels: the dense projection widths."""

DENSE_DEQUANT_THREADS = 256
"""Threads per dense dequantization threadgroup."""

TABLE_ENTRIES = {"iq4_ks": 32, "iq4_k": 32, "iq5_k": 64, "iq6_k": 128}
"""Value-table entries per member, both alphabets included."""

_SELECT_SHIFT = {"iq4_ks": 4, "iq4_k": 4, "iq5_k": 5, "iq6_k": 6}
_TABLE_METAL_TYPE = {"iq4_ks": "half", "iq4_k": "half",
                     "iq5_k": "half", "iq6_k": "float"}


def check_dense_gemv_geometry(in_features: int, out_features: int) -> tuple[int, int]:
    """Reject any geometry the dense decode GEMV does not cover."""
    n = check_row_width(in_features)
    if n not in DENSE_SUPPORTED_IN_FEATURES:
        raise IkqKernelError(
            f"the dense decode gemv is sized per input width and covers "
            f"{DENSE_SUPPORTED_IN_FEATURES}, got {in_features}")
    out = int(out_features)
    if out <= 0 or out % ROWS_PER_TG:
        raise IkqKernelError(
            f"out_features {out_features} is not a positive multiple of {ROWS_PER_TG}")
    return n, out


def check_dense_dequant_geometry(in_features: int, out_features: int) -> tuple[int, int]:
    """Reject any geometry the dense dequantization does not cover."""
    n = check_row_width(in_features)
    if n not in DENSE_SUPPORTED_IN_FEATURES:
        raise IkqKernelError(
            f"the dense dequantization is sized per input width and covers "
            f"{DENSE_SUPPORTED_IN_FEATURES}, got {in_features}")
    out = int(out_features)
    if out <= 0:
        raise IkqKernelError(f"out_features {out_features} is not positive")
    groups = n // 32
    if DENSE_DEQUANT_THREADS % groups:
        raise IkqKernelError(
            f"in_features {in_features} does not tile {DENSE_DEQUANT_THREADS} threads")
    return n, out


def _table_stage(member: str, threads: int) -> str:
    """Stage the member's value table into threadgroup memory."""
    entries = TABLE_ENTRIES[member]
    ttype = _TABLE_METAL_TYPE[member]
    return (f"    threadgroup {ttype} tgV[{entries}];\n"
            f"    for (uint i_ = tid; i_ < {entries}u; i_ += {threads}u) "
            f"{{ tgV[i_] = vtab[i_]; }}\n"
            f"    threadgroup_barrier(mem_flags::mem_threadgroup);")


def _code_loads(member: str, n: int, row_expr: str, group_expr: str) -> str:
    """Load one 32-weight group's code words into ``q0_..`` and ``h0_..``."""
    qs_load = (
        f"        const device uint4* qp_ = (const device uint4*)\n"
        f"            (qs + {row_expr} * {n // 8}ul + (ulong)({group_expr} << 2u));\n"
        f"        uint4 qw_ = qp_[0];\n"
        f"        uint q0_ = qw_.x; uint q1_ = qw_.y; "
        f"uint q2_ = qw_.z; uint q3_ = qw_.w;")
    lines = [qs_load]
    if member == "iq5_k":
        lines.append(
            f"        uint h0_ = qh[{row_expr} * {n // 32}ul + (ulong){group_expr}];")
    elif member == "iq6_k":
        lines.append(
            f"        const device uint2* hp_ = (const device uint2*)\n"
            f"            (qh + {row_expr} * {n // 16}ul + (ulong)({group_expr} << 1u));\n"
            f"        uint2 hw_ = hp_[0];\n"
            f"        uint h0_ = hw_.x; uint h1_ = hw_.y;")
    return "\n".join(lines)


def _index_expr(member: str, j: int) -> str:
    """Table index of weight ``j`` in its 32-weight group, base offset aside."""
    code = f"((q{j // 8}_ >> {4 * (j % 8)}u) & 0xFu)"
    if member == "iq5_k":
        hi = f"(((h0_ >> {j}u) & 1u) << 4u)"
        return f"({code} | {hi})"
    if member == "iq6_k":
        hi = f"(((h{j // 16}_ >> {2 * (j % 16)}u) & 3u) << 4u)"
        return f"({code} | {hi})"
    return code


def _scale_block(member: str, n: int, row_expr: str, group_expr: str) -> str:
    """Per-32-weight scale, alphabet-offset, and sub-scale arithmetic.

    Defines ``dl0_``/``t0_`` and, for the 16-weight sub-block members,
    ``dl1_``/``t1_``. The arithmetic is the CPU dequantizer's: the signed
    scale payload converts to float once, and the alphabet select becomes a
    base offset into the staged table.
    """
    shift = _SELECT_SHIFT[member]
    if member == "iq4_ks":
        return f"""
        uint sc_ = uint(scl[{row_expr} * {n // 32}ul + (ulong){group_expr}]);
        float dv_ = dv[{row_expr}];
        float dl0_ = dv_ * float(int(sc_ & 254u) - 127);
        uint t0_ = (sc_ & 1u) << {shift}u;
"""
    if member == "iq6_k":
        return f"""
        uint sc0_ = uint(scl[{row_expr} * {n // 16}ul + (ulong)({group_expr} << 1u)]);
        uint sc1_ = uint(scl[{row_expr} * {n // 16}ul + (ulong)(({group_expr} << 1u) | 1u)]);
        uint ex_ = uint(sex[{row_expr} * {n // 128}ul + (ulong)({group_expr} >> 2u)])
                   >> (({group_expr} << 1u) & 7u);
        float dv_ = float(dv[{row_expr} * {n // 256}ul + (ulong)({group_expr} >> 3u)]);
        float dl0_ = dv_ * float((int(sc0_) ^ 0x80) - 0x80);
        float dl1_ = dv_ * float((int(sc1_) ^ 0x80) - 0x80);
        uint t0_ = (ex_ & 1u) << {shift}u;
        uint t1_ = ((ex_ >> 1u) & 1u) << {shift}u;
"""
    return f"""
        uint sc_ = uint(scl[{row_expr} * {n // 32}ul + (ulong){group_expr}]);
        uint sh_ = uint(sch[{row_expr} * {n // 64}ul + (ulong)({group_expr} >> 1u)])
                   >> (({group_expr} & 1u) << 2u);
        uint ex_ = uint(sex[{row_expr} * {n // 128}ul + (ulong)({group_expr} >> 2u)])
                   >> (({group_expr} << 1u) & 7u);
        float dv_ = float(dv[{row_expr} * {n // 256}ul + (ulong)({group_expr} >> 3u)]);
        float dl0_ = dv_ * float(int((sc_ & 0xFu) | ((sh_ & 3u) << 4u)) - 32);
        float dl1_ = dv_ * float(int((sc_ >> 4u) | (((sh_ >> 2u) & 3u) << 4u)) - 32);
        uint t0_ = (ex_ & 1u) << {shift}u;
        uint t1_ = ((ex_ >> 1u) & 1u) << {shift}u;
"""


def _gemv_decode(member: str) -> tuple[str, str, str]:
    """Accumulator declarations, decode chain, and combine for the GEMV."""
    dual = SUB_WEIGHTS[member] == 16
    if dual:
        decode = "\n".join(
            f"        {'acc0_' if j < 16 else 'acc1_'} = fma("
            f"float(tgV[{'t0_' if j < 16 else 't1_'} + {_index_expr(member, j)}]), "
            f"xv[{j}], {'acc0_' if j < 16 else 'acc1_'});"
            for j in range(32))
        accs = "        float acc0_ = 0.0f;\n        float acc1_ = 0.0f;"
        combine = "        float pv_ = fma(dl0_, acc0_, dl1_ * acc1_);"
    else:
        decode = "\n".join(
            f"        acc0_ = fma(float(tgV[t0_ + {_index_expr(member, j)}]), "
            f"xv[{j}], acc0_);"
            for j in range(32))
        accs = "        float acc0_ = 0.0f;"
        combine = "        float pv_ = dl0_ * acc0_;"
    return accs, decode, combine


# ---------------------------------------------------------------------------
# Decode: dense GEMV
# ---------------------------------------------------------------------------


def dense_gemv_source(member: str, in_features: int, out_features: int,
                      rows_per_tg: int = ROWS_PER_TG) -> str:
    """Source of the dense decode GEMV for one member and geometry."""
    check_dense_member(member)
    n, out = in_features, out_features
    sg = simdgroups(n)
    threads = 32 * sg
    blocks_per_mat = out // rows_per_tg
    accs, decode, combine = _gemv_decode(member)
    partials = " + ".join(f"tg_r[{i}][tid]" for i in range(sg))
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    uint blk = threadgroup_position_in_grid.x;

    uint mat = blk / {blocks_per_mat}u;
    uint row0 = (blk - mat * {blocks_per_mat}u) * {rows_per_tg}u;

    uint kbase = sg * {WEIGHTS_PER_SIMDGROUP}u + lane * 32u;
    uint kg_ = kbase >> 5u;

{_table_stage(member, threads)}

{_xload(f"x + (ulong)mat * {n}ul + kbase")}

    threadgroup float tg_r[{sg}][{rows_per_tg}];
    for (uint r = 0u; r < {rows_per_tg}u; r++) {{
        ulong rid = (ulong)(row0 + r);
{_code_loads(member, n, "rid", "kg_")}
{_scale_block(member, n, "rid", "kg_")}
{accs}
{decode}
{combine}
        float ssum_ = simd_sum(pv_);
        if (lane == 0u) {{ tg_r[sg][r] = ssum_; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {rows_per_tg}u) {{
        out[(ulong)mat * {out}ul + (ulong)(row0 + tid)] = half({partials});
    }}
"""


# ---------------------------------------------------------------------------
# Prefill: dense dequantization, full and out-row-range forms
# ---------------------------------------------------------------------------


def _dequant_body(member: str) -> str:
    """The 32-weight reconstruction chain shared by both dequant forms."""
    dual = SUB_WEIGHTS[member] == 16
    if dual:
        return "\n".join(
            f"    float wv{j}_ = {'dl0_' if j < 16 else 'dl1_'} * "
            f"float(tgV[{'t0_' if j < 16 else 't1_'} + {_index_expr(member, j)}]);"
            for j in range(32))
    return "\n".join(
        f"    float wv{j}_ = dl0_ * float(tgV[t0_ + {_index_expr(member, j)}]);"
        for j in range(32))


def dense_dequant_source(member: str, in_features: int, out_features: int) -> str:
    """Source of the dense dequantization for one geometry."""
    check_dense_member(member)
    n = in_features
    groups_per_row = n // 32
    stores = "\n".join(
        f"    op_[{q}] = half4({', '.join(f'half(wv{4 * q + i}_)' for i in range(4))});"
        for q in range(8))
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint gid = thread_position_in_grid.x;
    uint kg_ = gid & {groups_per_row - 1}u;
    ulong rid = (ulong)(gid >> {_log2(groups_per_row)}u);
    uint kbase_ = kg_ << 5u;

{_table_stage(member, DENSE_DEQUANT_THREADS)}

{_code_loads(member, n, "rid", "kg_")}
{_scale_block(member, n, "rid", "kg_")}
{_dequant_body(member)}

    device half4* op_ = (device half4*)(out + rid * {n}ul + (ulong)kbase_);
{stores}
"""


def dense_dequant_range_source(member: str, in_features: int, out_features: int,
                               range_rows: int) -> str:
    """Source of the out-row-range dense dequantization for one geometry.

    Identical per-weight arithmetic to :func:`dense_dequant_source`; the
    grid covers ``range_rows`` compact output rows and the stream reads add
    the caller's row offset (``obase``). A range output equals the same
    slice of the full dequantization, bit for bit.
    """
    check_dense_member(member)
    n = in_features
    groups_per_row = n // 32
    if range_rows <= 0 or range_rows & (range_rows - 1):
        raise IkqKernelError(
            f"range_rows {range_rows} is not a positive power of two")
    if range_rows > out_features:
        raise IkqKernelError(
            f"range_rows {range_rows} exceeds out_features {out_features}")
    stores = "\n".join(
        f"    op_[{q}] = half4({', '.join(f'half(wv{4 * q + i}_)' for i in range(4))});"
        for q in range(8))
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint gid = thread_position_in_grid.x;
    uint kg_ = gid & {groups_per_row - 1}u;
    ulong lin_ = (ulong)(gid >> {_log2(groups_per_row)}u);
    ulong rid = (ulong)obase[0] + lin_;
    uint kbase_ = kg_ << 5u;

{_table_stage(member, DENSE_DEQUANT_THREADS)}

{_code_loads(member, n, "rid", "kg_")}
{_scale_block(member, n, "rid", "kg_")}
{_dequant_body(member)}

    device half4* op_ = (device half4*)(out + lin_ * {n}ul + (ulong)kbase_);
{stores}
"""


# ---------------------------------------------------------------------------
# Compiled-kernel caches and input orders
# ---------------------------------------------------------------------------


_DENSE_GEMV_KERNELS: dict[tuple, object] = {}
_DENSE_DEQUANT_KERNELS: dict[tuple, object] = {}
_DENSE_DEQUANT_RANGE_KERNELS: dict[tuple, object] = {}

_STREAM_ORDER = {
    "iq4_ks": ["qs", "scl", "dv"],
    "iq4_k": ["qs", "scl", "sch", "sex", "dv"],
    "iq5_k": ["qs", "qh", "scl", "sch", "sex", "dv"],
    "iq6_k": ["qs", "qh", "scl", "sex", "dv"],
}


def dense_gemv_input_names(member: str) -> list[str]:
    """Kernel input order for one member's dense decode GEMV."""
    check_dense_member(member)
    return ["x"] + list(_STREAM_ORDER[member]) + ["vtab"]


def dense_dequant_input_names(member: str) -> list[str]:
    """Kernel input order for one member's dense dequantization."""
    check_dense_member(member)
    return list(_STREAM_ORDER[member]) + ["vtab"]


def dense_dequant_range_input_names(member: str) -> list[str]:
    """Kernel input order for one member's out-row-range dequantization."""
    return dense_dequant_input_names(member) + ["obase"]


def dense_gemv_threads(in_features: int) -> int:
    """Threads per dense decode threadgroup for this input width."""
    return 32 * simdgroups(in_features)


def dense_gemv_kernel(member: str, in_features: int, out_features: int,
                      rows_per_tg: int = ROWS_PER_TG):
    """Compiled dense decode GEMV for one member and geometry (cached)."""
    check_dense_member(member)
    n, out = check_dense_gemv_geometry(in_features, out_features)
    if out % rows_per_tg:
        raise IkqKernelError(
            f"out_features {out} is not a multiple of rows_per_tg {rows_per_tg}")
    key = (member, n, out, rows_per_tg)
    kernel = _DENSE_GEMV_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mlx_ikq_dense_gemv_{member}_{n}x{out}_r{rows_per_tg}",
            input_names=dense_gemv_input_names(member),
            output_names=["out"],
            source=dense_gemv_source(member, n, out, rows_per_tg),
        )
        _DENSE_GEMV_KERNELS[key] = kernel
    return kernel


def dense_dequant_kernel(member: str, in_features: int, out_features: int):
    """Compiled dense dequantization for one geometry (cached)."""
    check_dense_member(member)
    n, out = check_dense_dequant_geometry(in_features, out_features)
    key = (member, n, out)
    kernel = _DENSE_DEQUANT_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mlx_ikq_dense_dequant_{member}_{n}x{out}",
            input_names=dense_dequant_input_names(member),
            output_names=["out"],
            source=dense_dequant_source(member, n, out),
        )
        _DENSE_DEQUANT_KERNELS[key] = kernel
    return kernel


def dense_dequant_range_kernel(member: str, in_features: int, out_features: int,
                               range_rows: int):
    """Compiled out-row-range dense dequantization (cached)."""
    check_dense_member(member)
    n, out = check_dense_dequant_geometry(in_features, out_features)
    key = (member, n, out, int(range_rows))
    kernel = _DENSE_DEQUANT_RANGE_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mlx_ikq_dense_dequant_{member}_{n}x{out}_r{int(range_rows)}",
            input_names=dense_dequant_range_input_names(member),
            output_names=["out"],
            source=dense_dequant_range_source(member, n, out, int(range_rows)),
        )
        _DENSE_DEQUANT_RANGE_KERNELS[key] = kernel
    return kernel


def built_dense_gemv_kernels() -> list[tuple]:
    """Geometry keys of every dense decode kernel compiled in this process."""
    return sorted(_DENSE_GEMV_KERNELS)


def built_dense_dequant_kernels() -> list[tuple]:
    """Geometry keys of every dense dequantization compiled in this process."""
    return sorted(_DENSE_DEQUANT_KERNELS)


def built_dense_dequant_range_kernels() -> list[tuple]:
    """Geometry keys of every range dequantization compiled in this process."""
    return sorted(_DENSE_DEQUANT_RANGE_KERNELS)


__all__ = [
    "DENSE_DEQUANT_THREADS",
    "DENSE_SUPPORTED_IN_FEATURES",
    "TABLE_ENTRIES",
    "built_dense_dequant_kernels",
    "built_dense_dequant_range_kernels",
    "built_dense_gemv_kernels",
    "check_dense_dequant_geometry",
    "check_dense_gemv_geometry",
    "dense_dequant_input_names",
    "dense_dequant_kernel",
    "dense_dequant_range_input_names",
    "dense_dequant_range_kernel",
    "dense_dequant_range_source",
    "dense_dequant_source",
    "dense_gemv_input_names",
    "dense_gemv_kernel",
    "dense_gemv_source",
    "dense_gemv_threads",
]
