"""Module-level serving entry points for the dense-tensor IQ_K members.

The dense seam is functional rather than a module class: a dense projection
is one matrix with no routing state, so the caller holds the wire streams
and passes them per call. The functions mirror the ``nn.py`` conventions:
fail-closed geometry checks before any dispatch, fp16 activations at the
kernel seam, and a decode/prefill split with the same numerical split as
the routed module (the decode GEMV factors each sub-block scale out of its
own partial sum; the dequantized route reconstructs each weight exactly as
the CPU reference does and accumulates in the matmul's order).

Integration surface:

- :func:`dense_value_table` — the member's value table, one array per
  process.
- :func:`dense_gemv` — the decode route, ``[tokens, out]`` fp16.
- :func:`dense_dequantized` / :func:`dense_dequantized_range` — the prefill
  weight producers, bit-identical to the reference decode rounded to fp16.
- :func:`dense_linear` — the route chooser over the two.

``x`` may be fp16 or bfloat16; both meet the kernels at fp16. A bfloat16
activation inside fp16 range converts exactly (its mantissa is a prefix of
fp16's); the cast is at the call seam so the kernel set stays single-dtype.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_ikq.dense_kernels import (
    DENSE_DEQUANT_THREADS,
    ROWS_PER_TG,
    check_dense_dequant_geometry,
    check_dense_gemv_geometry,
    dense_dequant_input_names,
    dense_dequant_kernel,
    dense_dequant_range_kernel,
    dense_gemv_kernel,
    dense_gemv_threads,
)
from mlx_ikq.format import (
    IQ4K_VALUES,
    IQ5NL_VALUES,
    IQ6K_TABLE,
    check_dense_member,
    dense_component_dtypes,
    dense_component_shapes,
)

DENSE_DECODE_TOKEN_LIMIT = 16
"""Tokens at or below which :func:`dense_linear` takes the decode GEMV.

Provisional: the crossover against dequantize-and-matmul is a served
measurement that has not run. Callers that own the routing decision call
:func:`dense_gemv` or the dequantized route directly.
"""

_VALUE_TABLES: dict[str, mx.array] = {}


class IkqDenseError(RuntimeError):
    pass


def dense_value_table(member: str) -> mx.array:
    """The member's value table, one shared array per process.

    fp16 for the integer alphabets (exact by integrality), float32 for
    ``IQ6_K`` whose table is the cubic reconstruction's float32 outputs.
    """
    check_dense_member(member)
    table = _VALUE_TABLES.get(member)
    if table is None:
        if member == "iq6_k":
            table = mx.array(IQ6K_TABLE)
        elif member == "iq5_k":
            table = mx.array(IQ5NL_VALUES.astype("float16"))
        else:
            table = mx.array(IQ4K_VALUES.astype("float16"))
        mx.eval(table)
        _VALUE_TABLES[member] = table
    return table


def check_dense_streams(member: str, streams: dict[str, mx.array],
                        out_features: int, in_features: int) -> list[mx.array]:
    """Validate stream names, shapes, and dtypes; return kernel input order."""
    shapes = dense_component_shapes(member, out_features, in_features)
    dtypes = dense_component_dtypes(member)
    if set(streams) != set(shapes):
        raise IkqDenseError(
            f"{member} needs streams {sorted(shapes)}, got {sorted(streams)}")
    ordered = []
    for name in dense_dequant_input_names(member):
        if name == "vtab":
            continue
        value = streams[name]
        if tuple(value.shape) != shapes[name]:
            raise IkqDenseError(
                f"stream {name} must be {shapes[name]} for "
                f"{out_features}x{in_features}, got {tuple(value.shape)}")
        if str(value.dtype) != f"mlx.core.{dtypes[name].name}":
            raise IkqDenseError(
                f"stream {name} must be {dtypes[name].name}, got {value.dtype}")
        ordered.append(value)
    return ordered


def _tokens_view(x, in_features: int):
    xt = x.reshape(-1, in_features)
    if xt.dtype != mx.float16:
        xt = xt.astype(mx.float16)
    return xt


def dense_gemv(member: str, x, streams: dict[str, mx.array],
               out_features: int, in_features: int) -> mx.array:
    """Project ``[..., in]`` activations through the dense matrix, fused.

    Returns ``[..., out]`` fp16. Valid for any token count; the decode
    shapes are where it is the intended route.
    """
    check_dense_member(member)
    n, out = check_dense_gemv_geometry(in_features, out_features)
    ordered = check_dense_streams(member, streams, out, n)
    xt = _tokens_view(x, n)
    tokens = int(xt.shape[0])
    if tokens <= 0:
        raise IkqDenseError("no activation rows")
    blocks = tokens * (out // ROWS_PER_TG)
    threads = dense_gemv_threads(n)
    result = dense_gemv_kernel(member, n, out)(
        inputs=[xt] + ordered + [dense_value_table(member)],
        grid=(blocks * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(tokens * out,)],
        output_dtypes=[mx.float16],
    )[0]
    return result.reshape(list(x.shape[:-1]) + [out])


def dense_dequantized(member: str, streams: dict[str, mx.array],
                      out_features: int, in_features: int) -> mx.array:
    """Full fp16 dequantization ``[out, in]``.

    Bit-identical to the reference decode rounded to fp16: the kernel
    performs the same float32 ``d * payload`` then ``dl * value`` chain.
    """
    check_dense_member(member)
    n, out = check_dense_dequant_geometry(in_features, out_features)
    ordered = check_dense_streams(member, streams, out, n)
    threads = out * (n // 32)
    return dense_dequant_kernel(member, n, out)(
        inputs=ordered + [dense_value_table(member)],
        grid=(threads, 1, 1),
        threadgroup=(DENSE_DEQUANT_THREADS, 1, 1),
        output_shapes=[(out, n)],
        output_dtypes=[mx.float16],
    )[0]


def dense_dequantized_range(member: str, streams: dict[str, mx.array],
                            out_features: int, in_features: int,
                            start: int, rows: int) -> mx.array:
    """fp16 dequantization of out rows ``[start, start + rows)``: ``[rows, in]``.

    Bit-identical to the same slice of :func:`dense_dequantized`. The range
    form bounds in-flight buffer bytes; a shape that does not tile by one
    power of two (the lm_head's 129280 rows) is covered by mixing range
    sizes, each a compiled variant, over the caller's schedule.
    """
    check_dense_member(member)
    n, out = check_dense_dequant_geometry(in_features, out_features)
    ordered = check_dense_streams(member, streams, out, n)
    start, rows = int(start), int(rows)
    if rows <= 0 or rows & (rows - 1):
        raise IkqDenseError(f"range rows {rows} is not a positive power of two")
    if start < 0 or start + rows > out:
        raise IkqDenseError(
            f"range [{start}, {start + rows}) is outside out_features {out}")
    threads = rows * (n // 32)
    obase = mx.array([start], dtype=mx.uint32)
    return dense_dequant_range_kernel(member, n, out, rows)(
        inputs=ordered + [dense_value_table(member), obase],
        grid=(threads, 1, 1),
        threadgroup=(DENSE_DEQUANT_THREADS, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.float16],
    )[0]


def dense_linear(member: str, x, streams: dict[str, mx.array],
                 out_features: int, in_features: int,
                 token_limit: int = DENSE_DECODE_TOKEN_LIMIT) -> mx.array:
    """Project ``x`` through the dense matrix, choosing the route by shape.

    At or below ``token_limit`` activation rows the fused GEMV serves;
    above it the matrix dequantizes to fp16 in one pass and the projection
    runs as a plain matmul. The limit default is provisional (see
    :data:`DENSE_DECODE_TOKEN_LIMIT`).
    """
    check_dense_member(member)
    xt = _tokens_view(x, in_features)
    if int(xt.shape[0]) <= int(token_limit):
        return dense_gemv(member, x, streams, out_features, in_features)
    weights = dense_dequantized(member, streams, out_features, in_features)
    result = mx.matmul(xt, weights.swapaxes(0, 1))
    return result.reshape(list(x.shape[:-1]) + [int(out_features)])


__all__ = [
    "DENSE_DECODE_TOKEN_LIMIT",
    "IkqDenseError",
    "check_dense_streams",
    "dense_dequantized",
    "dense_dequantized_range",
    "dense_gemv",
    "dense_linear",
    "dense_value_table",
]
