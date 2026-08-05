"""Metal kernels for the IQ_K relayout routed-expert format.

Two kernels serve one stacked-expert projection, for every member:

- ``gemv_kernel``: the decode route. One threadgroup walks a fixed number of
  output rows of one (token, expert) pair, streaming the k-contiguous
  relayout at its address rate. The eight-entry value table lives in
  threadgroup memory as ``half``, every stream offset is a shift of a
  compile-time constant, and the reduction is one ``simd_sum`` per row
  followed by a threadgroup add of the simdgroup partials. No integer
  multiplies in the inner loop, no device-memory gathers, no serial
  dependence between symbols.
- ``dequant_kernel``: the prefill route's weight producer. One thread decodes
  32 consecutive weights of one row and stores them as fp16, so a whole
  stacked projection materializes in one pass. The arithmetic is the CPU
  dequantizer's float32 product chain in the same order, so the fp16 result
  is bit-identical to the reference decode.

Three representation choices carry the decode down to its address floor and
none of them changes a reconstructed value or a stored byte.

The value table is held as ``half`` rather than ``int8_t``: a dynamically
indexed byte load out of threadgroup memory is not a single load on this
hardware, and the ``iq2nl_values`` levels are exact in fp16. The codes are
read from one contiguous 2-bit stream rather than from ik's
sub-block-interleaved bytes, which turns a per-weight assemble into a single
shift-and-mask. A byte-faithful decode keeping ik's own placement measures
1.735x its address floor at 3.4375 bits per weight; the same arithmetic on
the relayout measures 0.997x.

At 2 bits per weight the address stream is a third cheaper and stops hiding
the rest, so a third choice binds here that does not bind at 3 bits: the four
alphabet entries a sub-block uses are read into registers once per sub-block
and the per-weight lookup becomes a select between them. That is the same
entry the table index would have returned, and it moves the two 2-bit members
from 1.42 and 1.37 against their floors to 1.14 and 1.13.

``IQ1_S_R4`` decodes from the 2048-entry ternary grid instead of the
eight-entry alphabet: each 8-weight sub-group names one grid entry through
an 11-bit index, and the reconstruction adds a per-block ``+-0.125`` shift
before the scale multiply. The grid is 16 KiB, too large for the register
staging the 2-bit members use, so it stays in device memory as ``uint2``
words and each sub-group costs one 8-byte gather. The table is shared by
every dispatch and orders of magnitude smaller than the wire it decodes, so
the gathers are cache reads, not stream traffic. The shift is applied per
weight inside the fma chain (``float(value) + shift``), which keeps the
per-block scale factored out of the partial sum exactly as the 2-bit
members keep theirs.

Both kernels bake the projection geometry into their source, so an address
computation is a shift and never a runtime division.

The supported input widths are exactly the two the routed geometry uses. A
decode threadgroup makes one pass over the reduction axis, so a width the
generator has not sized leaves part of the row unread and returns a wrong
answer rather than an error. The width is a guarded contract, not a tuning
parameter.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_ikq.format import SUB_WEIGHTS, check_member, check_row_width

IQ2_MEMBERS = ("iq2_ks", "iq2_k")
"""The alphabet-decode members; the recorded sorted/union GEMV formulations
exist only for these."""

ROWS_PER_TG = 16
"""Output rows one decode threadgroup walks."""

WEIGHTS_PER_LANE = 32
"""Weights one lane of a decode simdgroup covers, one contiguous group."""

WEIGHTS_PER_SIMDGROUP = 32 * WEIGHTS_PER_LANE
"""Reduction span of one simdgroup in a decode threadgroup."""

SUPPORTED_IN_FEATURES = (2048, 4096)
"""Input widths with a sized decode kernel: the down and the gate/up widths."""

DEQUANT_THREADS = 256
"""Threads per dequantization threadgroup."""


class IkqKernelError(RuntimeError):
    pass


def _check_iq2_member(member: str) -> str:
    """Members the recorded sorted/union GEMV formulations cover."""
    check_member(member)
    if member not in IQ2_MEMBERS:
        raise IkqKernelError(
            f"the sorted and union decode GEMVs are recorded formulations "
            f"for {IQ2_MEMBERS}; {member!r} has no such variant")
    return member


def check_gemv_geometry(in_features: int, out_features: int) -> tuple[int, int]:
    """Reject any geometry the decode GEMV does not cover, before compiling."""
    n = check_row_width(in_features)
    if n not in SUPPORTED_IN_FEATURES:
        raise IkqKernelError(
            f"the decode gemv is sized per input width and covers "
            f"{SUPPORTED_IN_FEATURES}, got {in_features}")
    out = int(out_features)
    if out <= 0 or out % ROWS_PER_TG:
        raise IkqKernelError(
            f"out_features {out_features} is not a positive multiple of {ROWS_PER_TG}")
    return n, out


def check_dequant_geometry(in_features: int, out_features: int) -> tuple[int, int]:
    """Reject any geometry the stacked dequantization does not cover."""
    n = check_row_width(in_features)
    if n not in SUPPORTED_IN_FEATURES:
        raise IkqKernelError(
            f"the dequantization is sized per input width and covers "
            f"{SUPPORTED_IN_FEATURES}, got {in_features}")
    out = int(out_features)
    if out <= 0:
        raise IkqKernelError(f"out_features {out_features} is not positive")
    groups = n // 32
    if DEQUANT_THREADS % groups:
        raise IkqKernelError(
            f"in_features {in_features} does not tile {DEQUANT_THREADS} threads")
    return n, out


def simdgroups(in_features: int) -> int:
    """Simdgroups one decode threadgroup uses for this input width."""
    n = check_row_width(in_features)
    if n % WEIGHTS_PER_SIMDGROUP:
        raise IkqKernelError(
            f"in_features {in_features} is not a multiple of {WEIGHTS_PER_SIMDGROUP}")
    return n // WEIGHTS_PER_SIMDGROUP


def gemv_threads(in_features: int) -> int:
    """Threads per decode threadgroup for this input width."""
    return 32 * simdgroups(in_features)


def _log2(value: int) -> int:
    if value <= 0 or value & (value - 1):
        raise IkqKernelError(f"{value} is not a positive power of two")
    return value.bit_length() - 1


def _xload(base: str) -> str:
    """Load 32 fp16 activations into ``xv[0..31]`` with literal indices only.

    A runtime subscript on a thread-private array forces it to scratch
    memory, and holding the lane's activations in registers across the row
    loop is the whole point of loading them once.
    """
    lines = [f"    const device uint4* xu_ = (const device uint4*)({base});",
             "    float xv[32];", "    {"]
    for q in range(4):
        lines.append(f"        uint4 wq{q}_ = xu_[{q}];")
    for q in range(4):
        for e, comp in enumerate("xyzw"):
            i = q * 8 + e * 2
            lines.append(f"        half2 h{q}{e}_ = as_type<half2>(wq{q}_.{comp});")
            lines.append(f"        xv[{i}] = float(h{q}{e}_.x);")
            lines.append(f"        xv[{i + 1}] = float(h{q}{e}_.y);")
    lines.append("    }")
    return "\n".join(lines)


def _code_expr(j: int) -> str:
    """Code ``j`` of a lane's 32 weights, from the two loaded words."""
    word = "q0_" if j < 16 else "q1_"
    shift = 2 * (j % 16)
    return f"(({word} >> {shift}u) & 3u)" if shift else f"({word} & 3u)"


def _quad_load(halves: tuple[int, ...]) -> str:
    """Stage each sub-block's four alphabet entries into registers.

    Four threadgroup reads per sub-block replace one read per weight. A
    per-lane dynamic index into a small threadgroup table is not a single
    load on this hardware, and at 2 bits per weight the decode is exposed
    enough for that to bind: reading the quad once and selecting between
    four registers measures 1.14 against the address floor where the
    per-weight indexed read measures 1.42.
    """
    return "\n".join(
        f"        float g{h}0_ = float(tgV[t{h}_ + 0u]);\n"
        f"        float g{h}1_ = float(tgV[t{h}_ + 1u]);\n"
        f"        float g{h}2_ = float(tgV[t{h}_ + 2u]);\n"
        f"        float g{h}3_ = float(tgV[t{h}_ + 3u]);"
        for h in halves)


def _value_expr(half_index: int, j: int) -> str:
    """Alphabet value of code ``j``, selected between four registers.

    The selected value is the entry the table index would have read, so the
    reconstruction is unchanged bit for bit.
    """
    c = _code_expr(j)
    h = half_index
    return (f"(({c} & 2u) ? (({c} & 1u) ? g{h}3_ : g{h}2_)"
            f" : (({c} & 1u) ? g{h}1_ : g{h}0_))")


def _scale_block(member: str, in_features: int, row_expr: str,
                 group_expr: str) -> str:
    """Per-32-weight scale, alphabet, and sub-scale arithmetic for one member.

    Defines ``dl0_``/``t0_`` and, for the 16-weight member, ``dl1_``/``t1_``.
    """
    n = in_features
    if member == "iq2_ks":
        return f"""
        uint sc_ = uint(scl[{row_expr} * {n // 64}ul + (ulong)({group_expr} >> 1u)])
                   >> (({group_expr} & 1u) << 2u);
        uint hi_ = uint(sch[{row_expr} * {n // 256}ul + (ulong)({group_expr} >> 3u)])
                   >> ({group_expr} & 7u);
        uint ex_ = uint(sex[{row_expr} * {n // 256}ul + (ulong)({group_expr} >> 3u)])
                   >> ({group_expr} & 7u);
        float dv_ = float(dv[{row_expr}]);
        float dl0_ = dv_ * float(int((sc_ & 0xFu) | ((hi_ & 1u) << 4u)) - 16);
        uint t0_ = (ex_ & 1u) << 2u;
"""
    return f"""
        uint sc_ = uint(scl[{row_expr} * {n // 32}ul + (ulong){group_expr}]);
        uint ex_ = uint(sex[{row_expr} * {n // 128}ul + (ulong)({group_expr} >> 2u)])
                   >> (({group_expr} << 1u) & 7u);
        float dv_ = float(dv[{row_expr} * {n // 256}ul + (ulong)({group_expr} >> 3u)]);
        float dl0_ = dv_ * float(int(sc_ & 0xFu) - 8);
        float dl1_ = dv_ * float(int(sc_ >> 4u) - 8);
        uint t0_ = (ex_ & 1u) << 2u;
        uint t1_ = ((ex_ >> 1u) & 1u) << 2u;
"""


# ---------------------------------------------------------------------------
# IQ1_S_R4: grid-decode source fragments
# ---------------------------------------------------------------------------


def _iq1sr4_reads(in_features: int, row_expr: str, group_expr: str) -> str:
    """Stream reads and scale arithmetic of one 32-weight block of one row.

    Defines ``qw_`` (four low index bytes), ``hw_`` (the block's qh word),
    ``dl0_`` (the float32 block scale, the CPU chain's ``d * (2*ls + 1)``),
    and ``sh_`` (the signed shift).
    """
    n = in_features
    return f"""
        uint qw_ = qs[{row_expr} * {n // 32}ul + (ulong){group_expr}];
        uint hw_ = uint(qh[{row_expr} * {n // 32}ul + (ulong){group_expr}]);
        float dv_ = float(dv[{row_expr}]);
        float dl0_ = dv_ * float(int(2u * ((hw_ >> 12u) & 7u) + 1u));
        float sh_ = (hw_ & 0x8000u) ? -0.125f : 0.125f;
"""


def _iq1sr4_gathers() -> str:
    """The four grid gathers of one 32-weight block, into ``char4`` pairs."""
    lines = []
    for i in range(4):
        idx = f"((qw_ >> {8 * i}u) & 0xFFu) | (((hw_ >> {3 * i}u) & 7u) << 8u)"
        lines.append(f"        uint2 gw{i}_ = gp_[{idx}];")
        lines.append(f"        char4 ga{i}_ = as_type<char4>(gw{i}_.x);")
        lines.append(f"        char4 gb{i}_ = as_type<char4>(gw{i}_.y);")
    return "\n".join(lines)


def _iq1sr4_value(j: int) -> str:
    """Weight ``j``'s grid value plus the block shift, as float32."""
    i, lane = divmod(j, 8)
    vec = f"ga{i}_" if lane < 4 else f"gb{i}_"
    comp = "xyzw"[lane % 4]
    return f"(float({vec}.{comp}) + sh_)"


# ---------------------------------------------------------------------------
# Decode: fused GEMV over the selected experts
# ---------------------------------------------------------------------------


def _gemv_source_iq1_s_r4(in_features: int, out_features: int,
                          rows_per_tg: int) -> str:
    """Source of the ``iq1_s_r4`` decode GEMV for one projection geometry."""
    n, out = in_features, out_features
    sg = simdgroups(n)
    blocks_per_mat = out // rows_per_tg
    decode = "\n".join(
        f"        acc0_ = fma({_iq1sr4_value(j)}, xv[{j}], acc0_);"
        for j in range(32))
    partials = " + ".join(f"tg_r[{i}][tid]" for i in range(sg))
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    uint blk = threadgroup_position_in_grid.x;

    uint mat = blk / {blocks_per_mat}u;
    uint row0 = (blk - mat * {blocks_per_mat}u) * {rows_per_tg}u;
    uint e_ = sel[mat];
    uint tok = mat / dims[0];

    uint kbase = sg * {WEIGHTS_PER_SIMDGROUP}u + lane * 32u;
    uint kg_ = kbase >> 5u;

    const device uint2* gp_ = (const device uint2*)grid;

{_xload(f"x + (ulong)tok * {n}ul + kbase")}

    threadgroup float tg_r[{sg}][{rows_per_tg}];
    for (uint r = 0u; r < {rows_per_tg}u; r++) {{
        ulong rid = (ulong)e_ * {out}ul + (ulong)(row0 + r);
{_iq1sr4_reads(n, "rid", "kg_")}
{_iq1sr4_gathers()}
        float acc0_ = 0.0f;
{decode}
        float pv_ = dl0_ * acc0_;
        float ssum_ = simd_sum(pv_);
        if (lane == 0u) {{ tg_r[sg][r] = ssum_; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {rows_per_tg}u) {{
        out[(ulong)mat * {out}ul + (ulong)(row0 + tid)] = half({partials});
    }}
"""


def gemv_source(member: str, in_features: int, out_features: int,
                rows_per_tg: int = ROWS_PER_TG) -> str:
    """Source of the decode GEMV for one member and projection geometry."""
    check_member(member)
    if member == "iq1_s_r4":
        return _gemv_source_iq1_s_r4(in_features, out_features, rows_per_tg)
    n, out = in_features, out_features
    sg = simdgroups(n)
    blocks_per_mat = out // rows_per_tg
    dual = SUB_WEIGHTS[member] == 16

    halves = (0, 1) if dual else (0,)
    quads = _quad_load(halves)
    if dual:
        decode = "\n".join(
            f"        {'acc0_' if j < 16 else 'acc1_'} = fma("
            f"{_value_expr(0 if j < 16 else 1, j)}, xv[{j}], "
            f"{'acc0_' if j < 16 else 'acc1_'});"
            for j in range(32))
        accs = "        float acc0_ = 0.0f;\n        float acc1_ = 0.0f;"
        combine = "        float pv_ = fma(dl0_, acc0_, dl1_ * acc1_);"
    else:
        decode = "\n".join(
            f"        acc0_ = fma({_value_expr(0, j)}, xv[{j}], acc0_);"
            for j in range(32))
        accs = "        float acc0_ = 0.0f;"
        combine = "        float pv_ = dl0_ * acc0_;"

    partials = " + ".join(f"tg_r[{i}][tid]" for i in range(sg))
    scales = _scale_block(member, n, "rid", "kg_")
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    uint blk = threadgroup_position_in_grid.x;

    uint mat = blk / {blocks_per_mat}u;
    uint row0 = (blk - mat * {blocks_per_mat}u) * {rows_per_tg}u;
    uint e_ = sel[mat];
    uint tok = mat / dims[0];

    uint kbase = sg * {WEIGHTS_PER_SIMDGROUP}u + lane * 32u;
    uint kg_ = kbase >> 5u;

    threadgroup half tgV[8];
    if (tid < 8u) {{ tgV[tid] = vtab[tid]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

{_xload(f"x + (ulong)tok * {n}ul + kbase")}

    threadgroup float tg_r[{sg}][{rows_per_tg}];
    for (uint r = 0u; r < {rows_per_tg}u; r++) {{
        ulong rid = (ulong)e_ * {out}ul + (ulong)(row0 + r);
        const device uint2* qp_ = (const device uint2*)
            (qs + rid * {n // 16}ul + (ulong)(kg_ << 1u));
        uint2 qw_ = qp_[0];
        uint q0_ = qw_.x; uint q1_ = qw_.y;
{scales}
{quads}
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


def gemv_sorted_source(member: str, in_features: int, out_features: int,
                       rows_per_tg: int = ROWS_PER_TG) -> str:
    """Source of the sorted-adjacency decode GEMV for one member/geometry.

    The incumbent GEMV's text with the pair-to-activation-row mapping
    taken from a `toks` array instead of the `mat / dims[0]` quotient, so
    a caller can hand the pairs sorted by expert id. Every threadgroup
    still computes exactly its own pair (full parallelism, no dedupe
    logic); pairs selecting the same expert become adjacent in dispatch
    order, so their reads of the same expert rows land close together in
    time and the cache hierarchy can serve the repeats. Per output
    element the arithmetic is `gemv_source`'s text, bit for bit.
    """
    _check_iq2_member(member)
    n, out = in_features, out_features
    sg = simdgroups(n)
    blocks_per_mat = out // rows_per_tg
    dual = SUB_WEIGHTS[member] == 16

    halves = (0, 1) if dual else (0,)
    quads = _quad_load(halves)
    if dual:
        decode = "\n".join(
            f"        {'acc0_' if j < 16 else 'acc1_'} = fma("
            f"{_value_expr(0 if j < 16 else 1, j)}, xv[{j}], "
            f"{'acc0_' if j < 16 else 'acc1_'});"
            for j in range(32))
        accs = "        float acc0_ = 0.0f;\n        float acc1_ = 0.0f;"
        combine = "        float pv_ = fma(dl0_, acc0_, dl1_ * acc1_);"
    else:
        decode = "\n".join(
            f"        acc0_ = fma({_value_expr(0, j)}, xv[{j}], acc0_);"
            for j in range(32))
        accs = "        float acc0_ = 0.0f;"
        combine = "        float pv_ = dl0_ * acc0_;"

    partials = " + ".join(f"tg_r[{i}][tid]" for i in range(sg))
    scales = _scale_block(member, n, "rid", "kg_")
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    uint blk = threadgroup_position_in_grid.x;

    uint mat = blk / {blocks_per_mat}u;
    uint row0 = (blk - mat * {blocks_per_mat}u) * {rows_per_tg}u;
    uint e_ = sel[mat];
    uint tok = toks[mat];

    uint kbase = sg * {WEIGHTS_PER_SIMDGROUP}u + lane * 32u;
    uint kg_ = kbase >> 5u;

    threadgroup half tgV[8];
    if (tid < 8u) {{ tgV[tid] = vtab[tid]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

{_xload(f"x + (ulong)tok * {n}ul + kbase")}

    threadgroup float tg_r[{sg}][{rows_per_tg}];
    for (uint r = 0u; r < {rows_per_tg}u; r++) {{
        ulong rid = (ulong)e_ * {out}ul + (ulong)(row0 + r);
        const device uint2* qp_ = (const device uint2*)
            (qs + rid * {n // 16}ul + (ulong)(kg_ << 1u));
        uint2 qw_ = qp_[0];
        uint q0_ = qw_.x; uint q1_ = qw_.y;
{scales}
{quads}
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


def gemv_union_source(member: str, in_features: int, out_features: int,
                      rows_per_tg: int = ROWS_PER_TG) -> str:
    """Source of the expert-union decode GEMV for one member and geometry.

    Serves the multi-token verify shape, where several token-expert pairs
    select the same expert. The caller hands the pairs sorted by expert id
    (`sel`, equal ids adjacent) with a per-pair activation row index
    (`toks`). One threadgroup owns a row block of one distinct expert: the
    first pair of an equal-id run computes every pair in the run, and the
    threadgroups of the run's other pairs return at once. The expert's
    packed rows are therefore streamed from device memory once per run
    instead of once per pair.

    Per output element the arithmetic is `gemv_source`'s text: the same
    activation load, the same decode-and-fma chain in the same order, the
    same simd reduction and partial sum. The result equals the incumbent
    GEMV's output rows reordered by the caller's sort, bit for bit.
    """
    _check_iq2_member(member)
    n, out = in_features, out_features
    sg = simdgroups(n)
    blocks_per_mat = out // rows_per_tg
    dual = SUB_WEIGHTS[member] == 16

    halves = (0, 1) if dual else (0,)
    quads = _quad_load(halves)
    if dual:
        decode = "\n".join(
            f"        {'acc0_' if j < 16 else 'acc1_'} = fma("
            f"{_value_expr(0 if j < 16 else 1, j)}, xv[{j}], "
            f"{'acc0_' if j < 16 else 'acc1_'});"
            for j in range(32))
        accs = "        float acc0_ = 0.0f;\n        float acc1_ = 0.0f;"
        combine = "        float pv_ = fma(dl0_, acc0_, dl1_ * acc1_);"
    else:
        decode = "\n".join(
            f"        acc0_ = fma({_value_expr(0, j)}, xv[{j}], acc0_);"
            for j in range(32))
        accs = "        float acc0_ = 0.0f;"
        combine = "        float pv_ = dl0_ * acc0_;"

    partials = " + ".join(f"tg_r[{i}][tid]" for i in range(sg))
    scales = _scale_block(member, n, "rid", "kg_")
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid & 31u;
    uint sg = tid >> 5u;
    uint blk = threadgroup_position_in_grid.x;

    uint mat = blk / {blocks_per_mat}u;
    uint row0 = (blk - mat * {blocks_per_mat}u) * {rows_per_tg}u;
    uint e_ = sel[mat];
    if (mat > 0u && sel[mat - 1u] == e_) {{ return; }}
    uint npairs_ = dims[0];
    uint run_ = 1u;
    while (mat + run_ < npairs_ && sel[mat + run_] == e_) {{ run_++; }}

    uint kbase = sg * {WEIGHTS_PER_SIMDGROUP}u + lane * 32u;
    uint kg_ = kbase >> 5u;

    threadgroup half tgV[8];
    if (tid < 8u) {{ tgV[tid] = vtab[tid]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float tg_r[{sg}][{rows_per_tg}];
    for (uint q_ = 0u; q_ < run_; q_++) {{
    uint tok = toks[mat + q_];
{_xload(f"x + (ulong)tok * {n}ul + kbase")}
    for (uint r = 0u; r < {rows_per_tg}u; r++) {{
        ulong rid = (ulong)e_ * {out}ul + (ulong)(row0 + r);
        const device uint2* qp_ = (const device uint2*)
            (qs + rid * {n // 16}ul + (ulong)(kg_ << 1u));
        uint2 qw_ = qp_[0];
        uint q0_ = qw_.x; uint q1_ = qw_.y;
{scales}
{quads}
{accs}
{decode}
{combine}
        float ssum_ = simd_sum(pv_);
        if (lane == 0u) {{ tg_r[sg][r] = ssum_; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {rows_per_tg}u) {{
        out[(ulong)(mat + q_) * {out}ul + (ulong)(row0 + tid)] = half({partials});
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
"""


# ---------------------------------------------------------------------------
# Prefill: full stacked-expert dequantization
# ---------------------------------------------------------------------------


def _dequant_body_iq1_s_r4(n: int, row_expr: str, out_expr: str) -> str:
    """One thread's 32-weight ``iq1_s_r4`` reconstruction and stores.

    The float32 chain is the CPU dequantizer's own order,
    ``dl * (float(value) + shift)``, so the fp16 store is bit-identical to
    the reference decode.
    """
    body = "\n".join(
        f"    float wv{j}_ = dl0_ * {_iq1sr4_value(j)};" for j in range(32))
    stores = "\n".join(
        f"    op_[{q}] = half4({', '.join(f'half(wv{4 * q + i}_)' for i in range(4))});"
        for q in range(8))
    return f"""
    const device uint2* gp_ = (const device uint2*)grid;
{_iq1sr4_reads(n, row_expr, "kg_")}
{_iq1sr4_gathers()}
{body}

    device half4* op_ = (device half4*)(out + {out_expr} * {n}ul + (ulong)kbase_);
{stores}
"""


def _dequant_source_iq1_s_r4(in_features: int, out_features: int) -> str:
    n = in_features
    groups_per_row = n // 32
    return f"""
    uint gid = thread_position_in_grid.x;
    uint kg_ = gid & {groups_per_row - 1}u;
    ulong rid = (ulong)(gid >> {_log2(groups_per_row)}u);
    uint kbase_ = kg_ << 5u;
{_dequant_body_iq1_s_r4(n, "rid", "rid")}
"""


def _dequant_range_source_iq1_s_r4(in_features: int, out_features: int,
                                   range_rows: int) -> str:
    n = in_features
    groups_per_row = n // 32
    return f"""
    uint gid = thread_position_in_grid.x;
    uint kg_ = gid & {groups_per_row - 1}u;
    ulong lin_ = (ulong)(gid >> {_log2(groups_per_row)}u);
    ulong eid_ = lin_ >> {_log2(range_rows)}u;
    ulong lrow_ = lin_ & {range_rows - 1}ul;
    ulong srow_ = eid_ * {out_features}ul + (ulong)obase[0] + lrow_;
    uint kbase_ = kg_ << 5u;
{_dequant_body_iq1_s_r4(n, "srow_", "lin_")}
"""


def dequant_source(member: str, in_features: int, out_features: int) -> str:
    """Source of the stacked-expert dequantization for one geometry."""
    check_member(member)
    if member == "iq1_s_r4":
        return _dequant_source_iq1_s_r4(in_features, out_features)
    n = in_features
    groups_per_row = n // 32
    dual = SUB_WEIGHTS[member] == 16

    halves = (0, 1) if dual else (0,)
    quads = _quad_load(halves)
    if dual:
        body = "\n".join(
            f"    float wv{j}_ = {'dl0_' if j < 16 else 'dl1_'} * "
            f"{_value_expr(0 if j < 16 else 1, j)};"
            for j in range(32))
    else:
        body = "\n".join(
            f"    float wv{j}_ = dl0_ * {_value_expr(0, j)};"
            for j in range(32))
    stores = "\n".join(
        f"    op_[{q}] = half4({', '.join(f'half(wv{4 * q + i}_)' for i in range(4))});"
        for q in range(8))
    scales = _scale_block(member, n, "rid", "kg_")
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint gid = thread_position_in_grid.x;
    uint kg_ = gid & {groups_per_row - 1}u;
    ulong rid = (ulong)(gid >> {_log2(groups_per_row)}u);
    uint kbase_ = kg_ << 5u;

    threadgroup half tgV[8];
    if (tid < 8u) {{ tgV[tid] = vtab[tid]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device uint2* qp_ = (const device uint2*)
        (qs + rid * {n // 16}ul + (ulong)(kg_ << 1u));
    uint2 qw_ = qp_[0];
    uint q0_ = qw_.x; uint q1_ = qw_.y;
{scales}
{quads}
{body}

    device half4* op_ = (device half4*)(out + rid * {n}ul + (ulong)kbase_);
{stores}
"""


def dequant_range_source(member: str, in_features: int, out_features: int,
                         range_rows: int) -> str:
    """Source of the out-row-range stacked dequantization for one geometry.

    Identical per-weight arithmetic to :func:`dequant_source`; the grid
    covers ``num_experts x range_rows`` compact output rows and the stream
    reads add the caller's row offset (``obase``) inside each expert. The
    per-row reconstruction is the full kernel's, bit for bit, so a range
    output equals the same slice of the full dequantization.
    """
    check_member(member)
    n = in_features
    groups_per_row = n // 32
    if range_rows <= 0 or range_rows & (range_rows - 1):
        raise IkqKernelError(
            f"range_rows {range_rows} is not a positive power of two")
    if range_rows > out_features:
        raise IkqKernelError(
            f"range_rows {range_rows} exceeds out_features {out_features}")
    if member == "iq1_s_r4":
        return _dequant_range_source_iq1_s_r4(in_features, out_features,
                                              range_rows)
    dual = SUB_WEIGHTS[member] == 16

    halves = (0, 1) if dual else (0,)
    quads = _quad_load(halves)
    if dual:
        body = "\n".join(
            f"    float wv{j}_ = {'dl0_' if j < 16 else 'dl1_'} * "
            f"{_value_expr(0 if j < 16 else 1, j)};"
            for j in range(32))
    else:
        body = "\n".join(
            f"    float wv{j}_ = dl0_ * {_value_expr(0, j)};"
            for j in range(32))
    stores = "\n".join(
        f"    op_[{q}] = half4({', '.join(f'half(wv{4 * q + i}_)' for i in range(4))});"
        for q in range(8))
    scales = _scale_block(member, n, "srow_", "kg_")
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint gid = thread_position_in_grid.x;
    uint kg_ = gid & {groups_per_row - 1}u;
    ulong lin_ = (ulong)(gid >> {_log2(groups_per_row)}u);
    ulong eid_ = lin_ >> {_log2(range_rows)}u;
    ulong lrow_ = lin_ & {range_rows - 1}ul;
    ulong srow_ = eid_ * {out_features}ul + (ulong)obase[0] + lrow_;
    uint kbase_ = kg_ << 5u;

    threadgroup half tgV[8];
    if (tid < 8u) {{ tgV[tid] = vtab[tid]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device uint2* qp_ = (const device uint2*)
        (qs + srow_ * {n // 16}ul + (ulong)(kg_ << 1u));
    uint2 qw_ = qp_[0];
    uint q0_ = qw_.x; uint q1_ = qw_.y;
{scales}
{quads}
{body}

    device half4* op_ = (device half4*)(out + lin_ * {n}ul + (ulong)kbase_);
{stores}
"""


_GEMV_KERNELS: dict[tuple, object] = {}
_GEMV_SORTED_KERNELS: dict[tuple, object] = {}
_GEMV_UNION_KERNELS: dict[tuple, object] = {}
_DEQUANT_KERNELS: dict[tuple, object] = {}
_DEQUANT_RANGE_KERNELS: dict[tuple, object] = {}

_IQ2_KS_INPUTS = ["x", "qs", "scl", "sch", "sex", "dv", "vtab", "sel", "dims"]
_IQ2_K_INPUTS = ["x", "qs", "scl", "sex", "dv", "vtab", "sel", "dims"]
_IQ1_S_R4_INPUTS = ["x", "qs", "qh", "dv", "grid", "sel", "dims"]

TABLE_INPUT_NAMES = {"iq2_ks": "vtab", "iq2_k": "vtab", "iq1_s_r4": "grid"}
"""The shared-table input of each member's kernels: the fp16 eight-entry
alphabet for the 2-bit members, the packed uint2 grid for ``iq1_s_r4``."""


def gemv_input_names(member: str) -> list[str]:
    """Kernel input order for one member's decode GEMV."""
    check_member(member)
    if member == "iq2_ks":
        return list(_IQ2_KS_INPUTS)
    if member == "iq1_s_r4":
        return list(_IQ1_S_R4_INPUTS)
    return list(_IQ2_K_INPUTS)


def gemv_union_input_names(member: str) -> list[str]:
    """Kernel input order for one member's expert-union decode GEMV."""
    names = gemv_input_names(member)
    return names[:-1] + ["toks", "dims"]


def gemv_sorted_input_names(member: str) -> list[str]:
    """Kernel input order for one member's sorted-adjacency decode GEMV."""
    names = gemv_input_names(member)
    return names[:-1] + ["toks"]


def dequant_input_names(member: str) -> list[str]:
    """Kernel input order for one member's stacked dequantization."""
    return [n for n in gemv_input_names(member) if n not in ("x", "sel", "dims")]


def dequant_range_input_names(member: str) -> list[str]:
    """Kernel input order for one member's out-row-range dequantization."""
    return dequant_input_names(member) + ["obase"]


def gemv_kernel(member: str, in_features: int, out_features: int,
                rows_per_tg: int = ROWS_PER_TG):
    """Compiled decode GEMV for one member and geometry (cached)."""
    check_member(member)
    n, out = check_gemv_geometry(in_features, out_features)
    if out % rows_per_tg:
        raise IkqKernelError(
            f"out_features {out} is not a multiple of rows_per_tg {rows_per_tg}")
    key = (member, n, out, rows_per_tg)
    kernel = _GEMV_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mlx_ikq_gemv_{member}_{n}x{out}_r{rows_per_tg}",
            input_names=gemv_input_names(member),
            output_names=["out"],
            source=gemv_source(member, n, out, rows_per_tg),
        )
        _GEMV_KERNELS[key] = kernel
    return kernel


def gemv_sorted_kernel(member: str, in_features: int, out_features: int,
                       rows_per_tg: int = ROWS_PER_TG):
    """Compiled sorted-adjacency decode GEMV for one member and geometry."""
    check_member(member)
    n, out = check_gemv_geometry(in_features, out_features)
    if out % rows_per_tg:
        raise IkqKernelError(
            f"out_features {out} is not a multiple of rows_per_tg {rows_per_tg}")
    key = (member, n, out, rows_per_tg)
    kernel = _GEMV_SORTED_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mlx_ikq_gemv_sorted_{member}_{n}x{out}_r{rows_per_tg}",
            input_names=gemv_sorted_input_names(member),
            output_names=["out"],
            source=gemv_sorted_source(member, n, out, rows_per_tg),
        )
        _GEMV_SORTED_KERNELS[key] = kernel
    return kernel


def gemv_union_kernel(member: str, in_features: int, out_features: int,
                      rows_per_tg: int = ROWS_PER_TG):
    """Compiled expert-union decode GEMV for one member and geometry."""
    check_member(member)
    n, out = check_gemv_geometry(in_features, out_features)
    if out % rows_per_tg:
        raise IkqKernelError(
            f"out_features {out} is not a multiple of rows_per_tg {rows_per_tg}")
    key = (member, n, out, rows_per_tg)
    kernel = _GEMV_UNION_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mlx_ikq_gemv_union_{member}_{n}x{out}_r{rows_per_tg}",
            input_names=gemv_union_input_names(member),
            output_names=["out"],
            source=gemv_union_source(member, n, out, rows_per_tg),
        )
        _GEMV_UNION_KERNELS[key] = kernel
    return kernel


def dequant_kernel(member: str, in_features: int, out_features: int):
    """Compiled stacked-expert dequantization for one geometry (cached)."""
    check_member(member)
    n, out = check_dequant_geometry(in_features, out_features)
    key = (member, n, out)
    kernel = _DEQUANT_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mlx_ikq_dequant_{member}_{n}x{out}",
            input_names=dequant_input_names(member),
            output_names=["out"],
            source=dequant_source(member, n, out),
        )
        _DEQUANT_KERNELS[key] = kernel
    return kernel


def dequant_range_kernel(member: str, in_features: int, out_features: int,
                         range_rows: int):
    """Compiled out-row-range dequantization for one geometry (cached)."""
    check_member(member)
    n, out = check_dequant_geometry(in_features, out_features)
    key = (member, n, out, int(range_rows))
    kernel = _DEQUANT_RANGE_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mlx_ikq_dequant_{member}_{n}x{out}_r{int(range_rows)}",
            input_names=dequant_range_input_names(member),
            output_names=["out"],
            source=dequant_range_source(member, n, out, int(range_rows)),
        )
        _DEQUANT_RANGE_KERNELS[key] = kernel
    return kernel


def built_gemv_kernels() -> list[tuple]:
    """Geometry keys of every decode kernel compiled in this process."""
    return sorted(_GEMV_KERNELS)


def built_gemv_sorted_kernels() -> list[tuple]:
    """Geometry keys of every sorted decode kernel compiled in this process."""
    return sorted(_GEMV_SORTED_KERNELS)


def built_gemv_union_kernels() -> list[tuple]:
    """Geometry keys of every union decode kernel compiled in this process."""
    return sorted(_GEMV_UNION_KERNELS)


def built_dequant_kernels() -> list[tuple]:
    """Geometry keys of every dequantization kernel compiled in this process."""
    return sorted(_DEQUANT_KERNELS)


def built_dequant_range_kernels() -> list[tuple]:
    """Geometry keys of every range dequantization compiled in this process."""
    return sorted(_DEQUANT_RANGE_KERNELS)


__all__ = [
    "DEQUANT_THREADS",
    "IQ2_MEMBERS",
    "ROWS_PER_TG",
    "SUPPORTED_IN_FEATURES",
    "TABLE_INPUT_NAMES",
    "WEIGHTS_PER_LANE",
    "WEIGHTS_PER_SIMDGROUP",
    "IkqKernelError",
    "built_dequant_kernels",
    "built_dequant_range_kernels",
    "built_gemv_kernels",
    "built_gemv_sorted_kernels",
    "built_gemv_union_kernels",
    "check_dequant_geometry",
    "check_gemv_geometry",
    "dequant_input_names",
    "dequant_kernel",
    "dequant_range_input_names",
    "dequant_range_kernel",
    "dequant_range_source",
    "dequant_source",
    "gemv_input_names",
    "gemv_kernel",
    "gemv_sorted_input_names",
    "gemv_sorted_kernel",
    "gemv_sorted_source",
    "gemv_source",
    "gemv_threads",
    "gemv_union_input_names",
    "gemv_union_kernel",
    "gemv_union_source",
    "simdgroups",
]
