"""Deleted-decode address floors for the two kernel families.

A floor arm issues the decode arm's load set byte for byte, deletes the
decode, and folds every loaded word into a hardened integer accumulator whose
result reaches the output, so no load can be eliminated and no term can
cancel. Its time is what the address stream costs on its own; the ratio of a
decode arm to its own floor is what the pricing cells report.

The load sets are generated from the same expressions the served kernels use,
so an arm and its floor cannot drift apart. Every floor is generated per
member and per geometry, exactly as its decode arm is: a floor built for one
member would price a different set of planes.

``iq1_s_r4``'s floor keeps the grid gathers: the index arithmetic on the
loaded ``qs``/``qh`` words is addressing, not decode, and deleting the
gathers would price a cheaper address stream than the kernel issues. Every
gathered word folds into the accumulator with its own prime.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_iqk.dense_kernels import (
    DENSE_DEQUANT_THREADS,
    TABLE_ENTRIES,
    check_dense_dequant_geometry,
    check_dense_gemv_geometry,
    dense_dequant_input_names,
    dense_gemv_input_names,
)
from mlx_iqk.format import SUB_WEIGHTS, check_dense_member, check_member
from mlx_iqk.kernels import (
    DEQUANT_THREADS,
    ROWS_PER_TG,
    WEIGHTS_PER_SIMDGROUP,
    _log2,
    dequant_input_names,
    gemv_input_names,
    simdgroups,
)


def _iq1sr4_floor_loads(n: int, row_expr: str, group_expr: str) -> str:
    """The ``iq1_s_r4`` load set of one 32-weight block: streams plus grid."""
    lines = [
        f"        uint qw_ = qs[{row_expr} * {n // 32}ul + (ulong){group_expr}];",
        (f"        uint hw_ = uint(qh[{row_expr} * {n // 32}ul "
         f"+ (ulong){group_expr}]);"),
        "        iacc += qw_ + 59u * hw_;",
        f"        iacc += 67u * uint(as_type<ushort>(dv[{row_expr}]));",
    ]
    primes = ((79, 83), (89, 97), (101, 103), (107, 109))
    for i, (p0, p1) in enumerate(primes):
        idx = f"((qw_ >> {8 * i}u) & 0xFFu) | (((hw_ >> {3 * i}u) & 7u) << 8u)"
        lines.append(f"        uint2 gw{i}_ = gp_[{idx}];")
        lines.append(f"        iacc += {p0}u * gw{i}_.x + {p1}u * gw{i}_.y;")
    return "\n".join(lines)


def _plane_loads(member: str, n: int, row_expr: str, group_expr: str) -> str:
    """Every scale-plane load of one 32-weight group, folded into ``iacc``.

    The multipliers are distinct odd primes so no two loads can cancel and no
    load is dead.
    """
    lines = []
    if member == "iq2_ks":
        lines.append(f"        iacc += 59u * uint(scl[{row_expr} * {n // 64}ul "
                     f"+ (ulong)({group_expr} >> 1u)]);")
        lines.append(f"        iacc += 61u * uint(sch[{row_expr} * {n // 256}ul "
                     f"+ (ulong)({group_expr} >> 3u)]);")
        lines.append(f"        iacc += 71u * uint(sex[{row_expr} * {n // 256}ul "
                     f"+ (ulong)({group_expr} >> 3u)]);")
        lines.append(f"        iacc += 67u * uint(as_type<ushort>(dv[{row_expr}]));")
    else:
        lines.append(f"        iacc += 59u * uint(scl[{row_expr} * {n // 32}ul "
                     f"+ (ulong){group_expr}]);")
        lines.append(f"        iacc += 71u * uint(sex[{row_expr} * {n // 128}ul "
                     f"+ (ulong)({group_expr} >> 2u)]);")
        lines.append(f"        iacc += 67u * uint(as_type<ushort>(dv[{row_expr} "
                     f"* {n // 256}ul + (ulong)({group_expr} >> 3u)]));")
    return "\n".join(lines)


def gemv_floor_source(member: str, in_features: int, out_features: int,
                      rows_per_tg: int = ROWS_PER_TG) -> str:
    """Address floor of the decode GEMV: same loads, decode deleted."""
    check_member(member)
    n, out = in_features, out_features
    sg = simdgroups(n)
    blocks_per_mat = out // rows_per_tg
    partials = " + ".join(f"tg_r[{i}][tid]" for i in range(sg))
    if member == "iq1_s_r4":
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

    const device uint4* xu_ = (const device uint4*)(x + (ulong)tok * {n}ul + kbase);
    const device uint2* gp_ = (const device uint2*)grid;

    uint iacc = 0u;
    {{
        uint4 w0_ = xu_[0]; uint4 w1_ = xu_[1];
        uint4 w2_ = xu_[2]; uint4 w3_ = xu_[3];
        iacc += w0_.x + 3u * w0_.y + 5u * w0_.z + 7u * w0_.w
              + 11u * w1_.x + 13u * w1_.y + 17u * w1_.z + 19u * w1_.w
              + 23u * w2_.x + 29u * w2_.y + 31u * w2_.z + 37u * w2_.w
              + 41u * w3_.x + 43u * w3_.y + 47u * w3_.z + 53u * w3_.w;
    }}

    threadgroup float tg_r[{sg}][{rows_per_tg}];
    for (uint r = 0u; r < {rows_per_tg}u; r++) {{
        ulong rid = (ulong)e_ * {out}ul + (ulong)(row0 + r);
{_iq1sr4_floor_loads(n, "rid", "kg_")}
        float ssum_ = simd_sum(float(iacc & 0xFFu));
        if (lane == 0u) {{ tg_r[sg][r] = ssum_; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {rows_per_tg}u) {{
        out[(ulong)mat * {out}ul + (ulong)(row0 + tid)] = half({partials});
    }}
"""
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

    const device uint4* xu_ = (const device uint4*)(x + (ulong)tok * {n}ul + kbase);

    uint iacc = 0u;
    {{
        uint4 w0_ = xu_[0]; uint4 w1_ = xu_[1];
        uint4 w2_ = xu_[2]; uint4 w3_ = xu_[3];
        iacc += w0_.x + 3u * w0_.y + 5u * w0_.z + 7u * w0_.w
              + 11u * w1_.x + 13u * w1_.y + 17u * w1_.z + 19u * w1_.w
              + 23u * w2_.x + 29u * w2_.y + 31u * w2_.z + 37u * w2_.w
              + 41u * w3_.x + 43u * w3_.y + 47u * w3_.z + 53u * w3_.w;
        iacc += 73u * uint(as_type<ushort>(vtab[tid & 7u]));
    }}

    threadgroup float tg_r[{sg}][{rows_per_tg}];
    for (uint r = 0u; r < {rows_per_tg}u; r++) {{
        ulong rid = (ulong)e_ * {out}ul + (ulong)(row0 + r);
        const device uint2* qp_ = (const device uint2*)
            (qs + rid * {n // 16}ul + (ulong)(kg_ << 1u));
        uint2 qw_ = qp_[0];
        iacc += qw_.x + 3u * qw_.y;
{_plane_loads(member, n, "rid", "kg_")}
        float ssum_ = simd_sum(float(iacc & 0xFFu));
        if (lane == 0u) {{ tg_r[sg][r] = ssum_; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {rows_per_tg}u) {{
        out[(ulong)mat * {out}ul + (ulong)(row0 + tid)] = half({partials});
    }}
"""


def dequant_floor_source(member: str, in_features: int, out_features: int) -> str:
    """Address floor of the stacked dequantization: same loads, same stores."""
    check_member(member)
    n = in_features
    groups_per_row = n // 32
    stores = "\n".join(f"    op_[{q}] = half4(hv_, hv_, hv_, hv_);" for q in range(8))
    if member == "iq1_s_r4":
        body = _iq1sr4_floor_loads(n, "rid", "kg_").replace("        ", "    ")
        return f"""
    uint gid = thread_position_in_grid.x;
    uint kg_ = gid & {groups_per_row - 1}u;
    ulong rid = (ulong)(gid >> {_log2(groups_per_row)}u);
    uint kbase_ = kg_ << 5u;

    const device uint2* gp_ = (const device uint2*)grid;
    uint iacc = 0u;
{body}

    half hv_ = half(float(iacc & 0xFFu));
    device half4* op_ = (device half4*)(out + rid * {n}ul + (ulong)kbase_);
{stores}
"""
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint gid = thread_position_in_grid.x;
    uint kg_ = gid & {groups_per_row - 1}u;
    ulong rid = (ulong)(gid >> {_log2(groups_per_row)}u);
    uint kbase_ = kg_ << 5u;

    uint iacc = 0u;
    iacc += 73u * uint(as_type<ushort>(vtab[tid & 7u]));

    const device uint2* qp_ = (const device uint2*)
        (qs + rid * {n // 16}ul + (ulong)(kg_ << 1u));
    uint2 qw_ = qp_[0];
    iacc += qw_.x + 3u * qw_.y;
{_plane_loads(member, n, "rid", "kg_")}

    half hv_ = half(float(iacc & 0xFFu));
    device half4* op_ = (device half4*)(out + rid * {n}ul + (ulong)kbase_);
{stores}
"""


# ---------------------------------------------------------------------------
# Dense-member floors
# ---------------------------------------------------------------------------


def _dense_vtab_loads(member: str, threads: int) -> str:
    """The staged value-table loads, folded. Mirrors the decode arm's loop."""
    entries = TABLE_ENTRIES[member]
    bits = ("as_type<uint>(vtab[i_])" if member == "iq6_k"
            else "uint(as_type<ushort>(vtab[i_]))")
    return (f"    for (uint i_ = tid; i_ < {entries}u; i_ += {threads}u) "
            f"{{ iacc2 += 73u * {bits}; }}")


def _dense_plane_loads(member: str, n: int, row_expr: str,
                       group_expr: str) -> str:
    """Every load of one 32-weight group of one row, folded and hardened.

    Two independent accumulators mirror the decode arm's instruction-level
    parallelism (its per-half fma chains), so the floor's cost is the
    address stream rather than a serial integer dependency the decode arm
    does not have. Every load still reaches the output through one of the
    accumulators, so none is deletable.
    """
    qs_load = (
        f"        const device uint4* qp_ = (const device uint4*)\n"
        f"            (qs + {row_expr} * {n // 8}ul + (ulong)({group_expr} << 2u));\n"
        f"        uint4 qw_ = qp_[0];\n"
        f"        iacc += qw_.x + 3u * qw_.y;\n"
        f"        iacc2 += 5u * qw_.z + 7u * qw_.w;")
    lines = [qs_load]
    if member == "iq5_k":
        lines.append(f"        iacc += 11u * qh[{row_expr} * {n // 32}ul "
                     f"+ (ulong){group_expr}];")
    elif member == "iq6_k":
        lines.append(
            f"        const device uint2* hp_ = (const device uint2*)\n"
            f"            (qh + {row_expr} * {n // 16}ul + (ulong)({group_expr} << 1u));\n"
            f"        uint2 hw_ = hp_[0];\n"
            f"        iacc += 11u * hw_.x;\n"
            f"        iacc2 += 13u * hw_.y;")
    if member == "iq4_ks":
        lines.append(f"        iacc += 59u * uint(scl[{row_expr} * {n // 32}ul "
                     f"+ (ulong){group_expr}]);")
        lines.append(f"        iacc2 += 67u * as_type<uint>(dv[{row_expr}]);")
    elif member == "iq6_k":
        lines.append(f"        iacc += 59u * uint(scl[{row_expr} * {n // 16}ul "
                     f"+ (ulong)({group_expr} << 1u)]);")
        lines.append(f"        iacc2 += 61u * uint(scl[{row_expr} * {n // 16}ul "
                     f"+ (ulong)(({group_expr} << 1u) | 1u)]);")
        lines.append(f"        iacc += 71u * uint(sex[{row_expr} * {n // 128}ul "
                     f"+ (ulong)({group_expr} >> 2u)]);")
        lines.append(f"        iacc2 += 67u * uint(as_type<ushort>(dv[{row_expr} "
                     f"* {n // 256}ul + (ulong)({group_expr} >> 3u)]));")
    else:
        lines.append(f"        iacc += 59u * uint(scl[{row_expr} * {n // 32}ul "
                     f"+ (ulong){group_expr}]);")
        lines.append(f"        iacc2 += 61u * uint(sch[{row_expr} * {n // 64}ul "
                     f"+ (ulong)({group_expr} >> 1u)]);")
        lines.append(f"        iacc += 71u * uint(sex[{row_expr} * {n // 128}ul "
                     f"+ (ulong)({group_expr} >> 2u)]);")
        lines.append(f"        iacc2 += 67u * uint(as_type<ushort>(dv[{row_expr} "
                     f"* {n // 256}ul + (ulong)({group_expr} >> 3u)]));")
    return "\n".join(lines)


def dense_gemv_floor_source(member: str, in_features: int, out_features: int,
                            rows_per_tg: int = ROWS_PER_TG) -> str:
    """Address floor of the dense decode GEMV: same loads, decode deleted."""
    check_dense_member(member)
    n, out = check_dense_gemv_geometry(in_features, out_features)
    sg = simdgroups(n)
    threads = 32 * sg
    blocks_per_mat = out // rows_per_tg
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

    const device uint4* xu_ = (const device uint4*)(x + (ulong)mat * {n}ul + kbase);

    uint iacc = 0u;
    uint iacc2 = 0u;
    {{
        uint4 w0_ = xu_[0]; uint4 w1_ = xu_[1];
        uint4 w2_ = xu_[2]; uint4 w3_ = xu_[3];
        iacc += w0_.x + 3u * w0_.y + 5u * w0_.z + 7u * w0_.w
              + 11u * w1_.x + 13u * w1_.y + 17u * w1_.z + 19u * w1_.w;
        iacc2 += 23u * w2_.x + 29u * w2_.y + 31u * w2_.z + 37u * w2_.w
              + 41u * w3_.x + 43u * w3_.y + 47u * w3_.z + 53u * w3_.w;
    }}
{_dense_vtab_loads(member, threads)}

    threadgroup float tg_r[{sg}][{rows_per_tg}];
    for (uint r = 0u; r < {rows_per_tg}u; r++) {{
        ulong rid = (ulong)(row0 + r);
{_dense_plane_loads(member, n, "rid", "kg_")}
        float ssum_ = simd_sum(float((iacc & 0xFFu) + (iacc2 & 0xFFu)));
        if (lane == 0u) {{ tg_r[sg][r] = ssum_; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {rows_per_tg}u) {{
        out[(ulong)mat * {out}ul + (ulong)(row0 + tid)] = half({partials});
    }}
"""


def dense_dequant_floor_source(member: str, in_features: int,
                               out_features: int) -> str:
    """Address floor of the dense dequantization: same loads, same stores."""
    check_dense_member(member)
    n, _out = check_dense_dequant_geometry(in_features, out_features)
    groups_per_row = n // 32
    stores = "\n".join(f"    op_[{q}] = half4(hv_, hv_, hv_, hv_);" for q in range(8))
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint gid = thread_position_in_grid.x;
    uint kg_ = gid & {groups_per_row - 1}u;
    ulong rid = (ulong)(gid >> {_log2(groups_per_row)}u);
    uint kbase_ = kg_ << 5u;

    uint iacc = 0u;
    uint iacc2 = 0u;
{_dense_vtab_loads(member, DENSE_DEQUANT_THREADS)}

{_dense_plane_loads(member, n, "rid", "kg_")}

    half hv_ = half(float((iacc & 0xFFu) + (iacc2 & 0xFFu)));
    device half4* op_ = (device half4*)(out + rid * {n}ul + (ulong)kbase_);
{stores}
"""


_FLOOR_KERNELS: dict[tuple, object] = {}


def gemv_floor_kernel(member: str, in_features: int, out_features: int,
                      rows_per_tg: int = ROWS_PER_TG):
    key = ("gemv", member, in_features, out_features, rows_per_tg)
    kernel = _FLOOR_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mlx_iqk_gemv_floor_{member}_{in_features}x{out_features}",
            input_names=gemv_input_names(member),
            output_names=["out"],
            source=gemv_floor_source(member, in_features, out_features, rows_per_tg),
        )
        _FLOOR_KERNELS[key] = kernel
    return kernel


def dequant_floor_kernel(member: str, in_features: int, out_features: int):
    key = ("dequant", member, in_features, out_features)
    kernel = _FLOOR_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mlx_iqk_dequant_floor_{member}_{in_features}x{out_features}",
            input_names=dequant_input_names(member),
            output_names=["out"],
            source=dequant_floor_source(member, in_features, out_features),
        )
        _FLOOR_KERNELS[key] = kernel
    return kernel


def dense_gemv_floor_kernel(member: str, in_features: int, out_features: int,
                            rows_per_tg: int = ROWS_PER_TG):
    key = ("dense_gemv", member, in_features, out_features, rows_per_tg)
    kernel = _FLOOR_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mlx_iqk_dense_gemv_floor_{member}_{in_features}x{out_features}",
            input_names=dense_gemv_input_names(member),
            output_names=["out"],
            source=dense_gemv_floor_source(member, in_features, out_features,
                                           rows_per_tg),
        )
        _FLOOR_KERNELS[key] = kernel
    return kernel


def dense_dequant_floor_kernel(member: str, in_features: int, out_features: int):
    key = ("dense_dequant", member, in_features, out_features)
    kernel = _FLOOR_KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mlx_iqk_dense_dequant_floor_{member}_{in_features}x{out_features}",
            input_names=dense_dequant_input_names(member),
            output_names=["out"],
            source=dense_dequant_floor_source(member, in_features, out_features),
        )
        _FLOOR_KERNELS[key] = kernel
    return kernel


__all__ = [
    "DEQUANT_THREADS", "SUB_WEIGHTS", "dense_dequant_floor_kernel",
    "dense_dequant_floor_source", "dense_gemv_floor_kernel",
    "dense_gemv_floor_source", "dequant_floor_kernel",
    "dequant_floor_source", "gemv_floor_kernel", "gemv_floor_source",
]
