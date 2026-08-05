"""Stacked-expert IQ_K module with the ``(x, indices)`` switch interface.

``IqkSwitchLinear`` fills the same per-projection seam as the other quantized
switch linears: ``__call__(x, indices)`` over stacked experts, with routing
and the shared expert staying on the caller's block. Two internal routes:

- decode (few token-expert pairs): the fused GEMV kernel streams the relayout
  directly.
- prefill (many pairs): the stacked experts dequantize to fp16 in one kernel
  pass and the projection runs through ``mx.gather_mm``. The caller's sort
  flag passes through, so a sorted prefill reads each expert's weights once.

The two routes are distinct numerical paths, as with the prefill/decode split
of the other expert formats: the dequantized route reconstructs each weight
exactly as the CPU reference does and then accumulates in ``gather_mm``'s
order, while the decode route factors each sub-block's scale out of its own
partial sum before the simdgroup reduction.

The member is per module instance, so one layer's gate, up, and down
projections can each carry whichever member the allocation assigned. Nothing
in the module or the kernels couples a layer to a single member.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mlx_iqk.format import (
    IQ2NL_VALUES,
    check_member,
    component_dtypes,
    component_shapes,
)
from mlx_iqk.iq1grid import grid_values
from mlx_iqk.kernels import (
    DEQUANT_THREADS,
    ROWS_PER_TG,
    check_dequant_geometry,
    check_gemv_geometry,
    dequant_kernel,
    dequant_range_kernel,
    gemv_kernel,
    gemv_sorted_kernel,
    gemv_threads,
    gemv_union_kernel,
)

DECODE_MAT_LIMIT = 63
"""Token-expert pairs at or below which the decode GEMV serves.

The stock SwitchGLU sorts at 64 gathered rows and larger runs are
prefill-shaped, so the dequantized gather route serves those.
"""

_VALUE_TABLE: mx.array | None = None
_GRID_TABLE: mx.array | None = None


def value_table() -> mx.array:
    """The fp16 ``iq2nl_values`` table, one shared array per process.

    Held outside the module so it is not a per-instance parameter: it is a
    format constant, identical for every tensor of both 2-bit members, and
    every entry is exact in fp16.
    """
    global _VALUE_TABLE
    if _VALUE_TABLE is None:
        _VALUE_TABLE = mx.array(IQ2NL_VALUES.astype("float16"))
        mx.eval(_VALUE_TABLE)
    return _VALUE_TABLE


def grid_table() -> mx.array:
    """The IQ1_S grid value table as uint32 words, one shared array per process.

    2048 entries of eight int8 ternary values, two little-endian uint32
    words per entry, in index order — the layout the ``iq1_s_r4`` kernels
    read through a ``uint2`` pointer. A format constant like the alphabet
    table, just 4096 words instead of 8 entries.
    """
    global _GRID_TABLE
    if _GRID_TABLE is None:
        import numpy as np
        words = np.ascontiguousarray(grid_values()).view(np.uint32).reshape(-1)
        _GRID_TABLE = mx.array(words)
        mx.eval(_GRID_TABLE)
    return _GRID_TABLE


def member_table(member: str) -> mx.array:
    """The shared-table kernel input of one member."""
    check_member(member)
    return grid_table() if member == "iq1_s_r4" else value_table()


class IqkRuntimeError(RuntimeError):
    pass


_MX_DTYPES = {"uint8": mx.uint8, "uint16": mx.uint16, "uint32": mx.uint32,
              "float16": mx.float16}


class IqkSwitchLinear(nn.Module):
    """Stacked IQ_K experts for one projection, ``(x, indices)`` interface."""

    mode = "iqk"

    def __init__(self, member: str, num_experts: int, out_features: int,
                 in_features: int):
        super().__init__()
        self.member = check_member(member)
        check_gemv_geometry(in_features, out_features)
        check_dequant_geometry(in_features, out_features)
        self.num_experts = int(num_experts)
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        if self.num_experts <= 0:
            raise IqkRuntimeError(f"num_experts {num_experts} is not positive")
        shapes = component_shapes(member, num_experts, out_features, in_features)
        dtypes = component_dtypes(member)
        for name, shape in shapes.items():
            setattr(self, name, mx.zeros(shape, dtype=_MX_DTYPES[dtypes[name].name]))

    # -- wire streams -------------------------------------------------------

    def stream_names(self) -> list[str]:
        """Relayout stream names of this member, in kernel input order."""
        return list(component_shapes(
            self.member, self.num_experts, self.out_features,
            self.in_features))

    def _streams(self) -> list[mx.array]:
        return [getattr(self, name) for name in self.stream_names()]

    def load_streams(self, streams: dict[str, mx.array]) -> None:
        """Install packed streams, checking every shape against the format."""
        shapes = component_shapes(self.member, self.num_experts,
                                  self.out_features, self.in_features)
        if set(streams) != set(shapes):
            raise IqkRuntimeError(
                f"{self.member} needs streams {sorted(shapes)}, got {sorted(streams)}")
        for name, want in shapes.items():
            got = tuple(streams[name].shape)
            if got != want:
                raise IqkRuntimeError(
                    f"stream {name} must be {want} for "
                    f"{self.num_experts}x{self.out_features}x{self.in_features}, got {got}")
        for name, value in streams.items():
            setattr(self, name, value)

    # -- dequantized route (prefill) ---------------------------------------

    def dequantized(self) -> mx.array:
        """Full fp16 dequantization ``[E, out, in]``.

        Bit-identical to the reference decode rounded to fp16: the kernel
        performs the same float32 ``d * index`` then ``dl * value`` chain.
        """
        kernel = dequant_kernel(self.member, self.in_features, self.out_features)
        threads = self.num_experts * self.out_features * (self.in_features // 32)
        return kernel(
            inputs=self._streams() + [member_table(self.member)],
            grid=(threads, 1, 1),
            threadgroup=(DEQUANT_THREADS, 1, 1),
            output_shapes=[(self.num_experts, self.out_features, self.in_features)],
            output_dtypes=[mx.float16],
        )[0]

    def dequantized_range(self, start: int, rows: int) -> mx.array:
        """fp16 dequantization of out rows ``[start, start + rows)``,
        every expert: ``[E, rows, in]``.

        Bit-identical to the same slice of :meth:`dequantized`; the range
        kernel runs the full kernel's per-row reconstruction with the
        caller's row offset added to the stream reads. The point of the
        range form is allocator liveness: buffers allocate at encode and
        free at command-buffer completion, so a full-stack fp16 buffer per
        projection puts three multi-GiB temporaries in flight per layer of
        a sorted prefill. Range-sized buffers cap that in-flight set.
        """
        start, rows = int(start), int(rows)
        if rows <= 0 or rows & (rows - 1):
            raise IqkRuntimeError(
                f"range rows {rows} is not a positive power of two")
        if start < 0 or start + rows > self.out_features:
            raise IqkRuntimeError(
                f"range [{start}, {start + rows}) is outside "
                f"out_features {self.out_features}")
        kernel = dequant_range_kernel(
            self.member, self.in_features, self.out_features, rows)
        threads = self.num_experts * rows * (self.in_features // 32)
        obase = mx.array([start], dtype=mx.uint32)
        return kernel(
            inputs=self._streams() + [member_table(self.member), obase],
            grid=(threads, 1, 1),
            threadgroup=(DEQUANT_THREADS, 1, 1),
            output_shapes=[(self.num_experts, rows, self.in_features)],
            output_dtypes=[mx.float16],
        )[0]

    def sorted_matmul_range(self, x, indices, start: int,
                            rows: int) -> mx.array:
        """The sorted dequantized projection over one out-row range.

        ``mx.gather_mm`` over the range dequantization, the same operand
        handling as the sorted branch of ``__call__``: activations meet the
        decoded weights at fp16 (a bfloat16 operand would promote the
        matmul to float32 and materialize a float32 copy of the decoded
        range). Concatenating the range outputs along the feature axis
        reproduces the unsplit call; each output element is one dot
        product either way.
        """
        weights = self.dequantized_range(start, rows)
        return mx.gather_mm(x.astype(mx.float16), weights.swapaxes(-1, -2),
                            rhs_indices=indices, sorted_indices=True)

    # -- call ---------------------------------------------------------------

    def __call__(self, x, indices, sorted_indices=False) -> mx.array:
        """Project ``x`` through the experts named by ``indices``.

        Shapes follow the stock switch-linear seam. Two operand shapes reach
        it: the gate and up projections pass ``[..., 1, 1, in_features]``, one
        activation row shared by that token's ``top_k`` experts, and the down
        projection passes ``[..., top_k, 1, in_features]``, one row per
        token-expert pair. Both are handled; ``indices`` is ``[..., top_k]``
        and the result is ``[..., top_k, 1, out_features]`` either way.
        """
        if indices.size <= DECODE_MAT_LIMIT and not sorted_indices:
            return self.gemv(x, indices)
        weights = self.dequantized()
        # Meet the decoded weights at fp16. A bfloat16 activation against an
        # fp16 weight promotes the whole matmul to float32, which materializes
        # a float32 copy of the entire decoded expert stack per projection.
        return mx.gather_mm(x.astype(mx.float16), weights.swapaxes(-1, -2),
                            rhs_indices=indices, sorted_indices=sorted_indices)

    def gemv(self, x, indices) -> mx.array:
        """Fused GEMV over the selected experts, packed bytes streamed.

        Valid for any pair count; a caller that owns its own routing decision
        calls this directly instead of going through the pair-count default in
        ``__call__``.
        """
        xt = x.reshape(-1, self.in_features).astype(mx.float16)
        sel = indices.reshape(-1).astype(mx.uint32)
        mats = int(sel.size)
        rows = int(xt.shape[0])
        # Token-expert pairs sharing one activation row. The gate and up
        # projections broadcast a token's row over its `top_k` experts, so
        # this is `top_k`; the down projection carries its own row per pair,
        # so it is 1. Deriving it from the operand rather than from
        # `indices.shape[-1]` is what lets one module serve both seams.
        if rows <= 0 or mats % rows:
            raise IqkRuntimeError(
                f"{mats} token-expert pairs do not tile {rows} activation "
                f"row(s); the operand and the routing disagree")
        dims = mx.array([mats // rows], dtype=mx.uint32)
        blocks = mats * (self.out_features // ROWS_PER_TG)
        threads = gemv_threads(self.in_features)
        out = gemv_kernel(self.member, self.in_features, self.out_features)(
            inputs=[xt] + self._streams() + [member_table(self.member), sel, dims],
            grid=(blocks * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(mats * self.out_features,)],
            output_dtypes=[mx.float16],
        )[0]
        out = out.reshape(list(indices.shape) + [self.out_features])
        return mx.expand_dims(out, -2)

    def gemv_sorted(self, x, sorted_ids, toks) -> mx.array:
        """Fused GEMV over expert-sorted pairs, one threadgroup per pair.

        The incumbent GEMV with the pair-to-activation-row mapping taken
        from `toks`; sorting puts same-expert pairs adjacent in dispatch
        order so the cache hierarchy can serve their repeated reads. Per
        output element the arithmetic equals `gemv`'s, so the output is
        the incumbent's rows reordered by the caller's sort, bit for bit.
        """
        xt = x.reshape(-1, self.in_features).astype(mx.float16)
        sel = sorted_ids.reshape(-1).astype(mx.uint32)
        tok = toks.reshape(-1).astype(mx.uint32)
        mats = int(sel.size)
        if int(tok.size) != mats:
            raise IqkRuntimeError(
                f"toks carries {int(tok.size)} rows for {mats} pairs")
        blocks = mats * (self.out_features // ROWS_PER_TG)
        threads = gemv_threads(self.in_features)
        out = gemv_sorted_kernel(
            self.member, self.in_features, self.out_features)(
            inputs=[xt] + self._streams() + [member_table(self.member), sel, tok],
            grid=(blocks * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(mats * self.out_features,)],
            output_dtypes=[mx.float16],
        )[0]
        return out.reshape(mats, self.out_features)

    def gemv_union(self, x, sorted_ids, toks) -> mx.array:
        """Fused GEMV over expert-sorted pairs, one read per distinct expert.

        Serves the multi-token verify shape, where several pairs select the
        same expert and the pair-per-threadgroup GEMV streams that expert's
        bytes once per pair. `sorted_ids` holds the selected expert ids with
        equal ids adjacent (the caller sorts once per verify call and shares
        the order across the three projections); `toks` maps each sorted
        pair to its activation row in `x`. Output row `p` is the projection
        of `x[toks[p]]` through expert `sorted_ids[p]`: the same per-element
        dot product in the same order as `gemv`, so the output equals the
        incumbent's rows reordered by the caller's sort, bit for bit. The
        caller applies the inverse permutation.
        """
        xt = x.reshape(-1, self.in_features).astype(mx.float16)
        sel = sorted_ids.reshape(-1).astype(mx.uint32)
        tok = toks.reshape(-1).astype(mx.uint32)
        mats = int(sel.size)
        if int(tok.size) != mats:
            raise IqkRuntimeError(
                f"toks carries {int(tok.size)} rows for {mats} pairs")
        dims = mx.array([mats], dtype=mx.uint32)
        blocks = mats * (self.out_features // ROWS_PER_TG)
        threads = gemv_threads(self.in_features)
        out = gemv_union_kernel(
            self.member, self.in_features, self.out_features)(
            inputs=[xt] + self._streams() + [member_table(self.member), sel, tok, dims],
            grid=(blocks * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(mats * self.out_features,)],
            output_dtypes=[mx.float16],
        )[0]
        return out.reshape(mats, self.out_features)


__all__ = ["DECODE_MAT_LIMIT", "IqkRuntimeError", "IqkSwitchLinear",
           "grid_table", "member_table", "value_table"]
