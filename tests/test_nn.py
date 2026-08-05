"""The switch module: both routes, the stream contract, and mixed members."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import wirepack
from mlx_ikq import format as fmt
from mlx_ikq.iq1grid import grid_values
from mlx_ikq.kernels import IQ2_MEMBERS, IkqKernelError
from mlx_ikq.nn import (
    DECODE_MAT_LIMIT,
    IkqRuntimeError,
    IkqSwitchLinear,
    grid_table,
    member_table,
    value_table,
)

pytestmark = pytest.mark.gpu

EXPERTS = 8
OUT_FEATURES = 32


def _module(member: str, n: int, seed: int, out_features: int = OUT_FEATURES):
    wire = wirepack.random_wire(member, EXPERTS * out_features, n, seed=seed,
                                scales="serving")
    streams = fmt.pack(member, wire, n)
    reference = fmt.decode(member, streams, n).reshape(EXPERTS, out_features, n)
    module = IkqSwitchLinear(member, EXPERTS, out_features, n)
    shapes = fmt.component_shapes(member, EXPERTS, out_features, n)
    module.load_streams({name: mx.array(np.ascontiguousarray(value).reshape(shapes[name]))
                         for name, value in streams.items()})
    return module, reference


def _reference_projection(reference, x, indices):
    """Reference for the gate/up seam: one activation row per token."""
    tokens, top_k = indices.shape
    out = np.zeros((tokens, top_k, reference.shape[1]), dtype=np.float64)
    for t in range(tokens):
        for k in range(top_k):
            out[t, k] = (reference[indices[t, k]].astype(np.float64)
                         @ x[t, 0, 0].astype(np.float64))
    return out


def _reference_paired_projection(reference, x, indices):
    """Reference for the down seam: one activation row per token-expert pair."""
    tokens, top_k = indices.shape
    out = np.zeros((tokens, top_k, reference.shape[1]), dtype=np.float64)
    for t in range(tokens):
        for k in range(top_k):
            out[t, k] = (reference[indices[t, k]].astype(np.float64)
                         @ x[t, k, 0].astype(np.float64))
    return out


def test_value_table_is_shared_and_exact_in_fp16():
    table = value_table()
    assert table.dtype == mx.float16
    assert np.array_equal(np.asarray(table).astype(np.int64),
                          fmt.IQ2NL_VALUES.astype(np.int64))
    assert value_table() is table


def test_grid_table_is_shared_and_carries_the_grid_values():
    table = grid_table()
    assert table.dtype == mx.uint32 and table.shape == (4096,)
    values = np.asarray(table).view(np.int8).reshape(2048, 8)
    assert np.array_equal(values, grid_values())
    assert grid_table() is table
    assert member_table("iq1_s_r4") is table
    assert member_table("iq2_ks") is value_table()


@pytest.mark.parametrize("member", fmt.MEMBERS)
def test_load_streams_rejects_a_wrong_shape(member):
    module, _ = _module(member, 2048, seed=1)
    shapes = fmt.component_shapes(member, EXPERTS, OUT_FEATURES, 2048)
    streams = {name: getattr(module, name) for name in shapes}
    name = "qs"
    streams[name] = mx.zeros((EXPERTS, OUT_FEATURES, 3), dtype=mx.uint32)
    with pytest.raises(IkqRuntimeError):
        module.load_streams(streams)


@pytest.mark.parametrize("member", fmt.MEMBERS)
def test_load_streams_rejects_a_missing_stream(member):
    module, _ = _module(member, 2048, seed=2)
    shapes = fmt.component_shapes(member, EXPERTS, OUT_FEATURES, 2048)
    streams = {name: getattr(module, name) for name in shapes}
    streams.pop(list(shapes)[1])
    with pytest.raises(IkqRuntimeError):
        module.load_streams(streams)


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", (2048, 4096))
def test_decode_route_serves_small_dispatches(member, n):
    module, reference = _module(member, n, seed=wirepack.seed_for(member, n))
    rng = np.random.default_rng(31)
    tokens, top_k = 4, 6
    assert tokens * top_k <= DECODE_MAT_LIMIT
    x = rng.standard_normal((tokens, 1, 1, n)).astype(np.float16)
    indices = rng.integers(0, EXPERTS, size=(tokens, top_k)).astype(np.uint32)
    got = np.asarray(module(mx.array(x), mx.array(indices)))
    assert got.shape == (tokens, top_k, 1, OUT_FEATURES)
    want = _reference_projection(reference, x, indices)
    rel = np.max(np.abs(got.reshape(want.shape).astype(np.float64) - want)) \
        / np.max(np.abs(want))
    assert rel < 1.5e-3, rel


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", (2048, 4096))
def test_prefill_route_serves_sorted_dispatches(member, n):
    module, reference = _module(member, n, seed=wirepack.seed_for(member, n, "prefill"))
    rng = np.random.default_rng(37)
    tokens, top_k = 24, 4
    x = rng.standard_normal((tokens, 1, 1, n)).astype(np.float16)
    indices = np.sort(rng.integers(0, EXPERTS, size=(tokens, top_k)), axis=None)
    indices = indices.reshape(tokens, top_k).astype(np.uint32)
    got = np.asarray(module(mx.array(x), mx.array(indices), sorted_indices=True))
    assert got.shape == (tokens, top_k, 1, OUT_FEATURES)
    want = _reference_projection(reference, x, indices)
    rel = np.max(np.abs(got.reshape(want.shape).astype(np.float64) - want)) \
        / np.max(np.abs(want))
    assert rel < 3e-3, rel


@pytest.mark.parametrize("member", fmt.MEMBERS)
def test_the_two_routes_agree_on_the_same_dispatch(member):
    """Different accumulation orders, so agreement is fp16-class, not exact."""
    n = 2048
    module, _ = _module(member, n, seed=41)
    rng = np.random.default_rng(43)
    tokens, top_k = 4, 4
    x = mx.array(rng.standard_normal((tokens, 1, 1, n)).astype(np.float16))
    indices = mx.array(rng.integers(0, EXPERTS, size=(tokens, top_k)).astype(np.uint32))
    decode = np.asarray(module.gemv(x, indices), dtype=np.float64)
    weights = module.dequantized()
    prefill = np.asarray(mx.gather_mm(x, weights.swapaxes(-1, -2),
                                      rhs_indices=indices), dtype=np.float64)
    rel = np.max(np.abs(decode - prefill)) / np.max(np.abs(prefill))
    assert rel < 3e-3, rel


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", (2048, 4096))
def test_decode_route_serves_the_down_operand_shape(member, n):
    """One activation row per token-expert pair, the down projection's seam.

    The gate and up projections hand the module one row per token and let
    every selected expert read it; the down projection hands it one row per
    pair. A module that assumed the first shape reads row 0 for every pair
    here and still returns finite, plausible numbers, so the anti-case below
    asserts the wrong mapping really would differ.
    """
    module, reference = _module(member, n, seed=wirepack.seed_for(member, n, "down"))
    rng = np.random.default_rng(59)
    tokens, top_k = 4, 4
    assert tokens * top_k <= DECODE_MAT_LIMIT
    x = rng.standard_normal((tokens, top_k, 1, n)).astype(np.float16)
    indices = rng.integers(0, EXPERTS, size=(tokens, top_k)).astype(np.uint32)
    got = np.asarray(module(mx.array(x), mx.array(indices)))
    assert got.shape == (tokens, top_k, 1, module.out_features)
    want = _reference_paired_projection(reference, x, indices)
    rel = np.max(np.abs(got.reshape(want.shape).astype(np.float64) - want)) \
        / np.max(np.abs(want))
    assert rel < 1.5e-3, rel

    shared_row = _reference_projection(reference, x[:, :1], indices)
    assert np.max(np.abs(shared_row - want)) > 1e-2 * np.max(np.abs(want))


def _union_operands(indices, top_k):
    """Sort pairs by expert and derive the union route's index operands."""
    flat = mx.array(indices).reshape(-1)
    order = mx.argsort(flat)
    sorted_ids = flat[order]
    toks = (order // top_k).astype(mx.uint32)
    inv = mx.argsort(order)
    return sorted_ids, toks, order, inv


@pytest.mark.parametrize("member", IQ2_MEMBERS)
@pytest.mark.parametrize("n", (2048, 4096))
def test_union_gemv_is_bit_identical_to_the_pair_gemv(member, n):
    """The union route's rows are the incumbent GEMV's, reordered.

    Per output element both kernels run the same activation load, the same
    decode-and-fma chain in the same order, and the same reduction, so the
    comparison is on raw fp16 bit patterns, not a tolerance. The selection
    is duplicate-heavy so equal-id runs form, including a run that starts
    at pair 0 and one that ends at the last pair after the sort.
    """
    module, _ = _module(member, n, seed=wirepack.seed_for(member, n, "union"))
    rng = np.random.default_rng(71)
    tokens, top_k = 6, 6
    indices = rng.integers(0, 5, size=(tokens, top_k)).astype(np.uint32)
    x = rng.standard_normal((tokens, 1, 1, n)).astype(np.float16)

    pair = module.gemv(mx.array(x), mx.array(indices))
    sorted_ids, toks, _order, inv = _union_operands(indices, top_k)
    got = module.gemv_union(mx.array(x), sorted_ids, toks)
    unsorted = got[inv].reshape(tokens, top_k, 1, module.out_features)
    assert np.array_equal(np.asarray(unsorted.view(mx.uint16)),
                          np.asarray(pair.view(mx.uint16)))


@pytest.mark.parametrize("member", IQ2_MEMBERS)
@pytest.mark.parametrize("n", (2048, 4096))
def test_union_gemv_serves_the_down_operand_shape(member, n):
    """One activation row per pair: `toks` is the sort order itself."""
    module, reference = _module(
        member, n, seed=wirepack.seed_for(member, n, "union_down"))
    rng = np.random.default_rng(73)
    tokens, top_k = 6, 6
    indices = rng.integers(0, 5, size=(tokens, top_k)).astype(np.uint32)
    x = rng.standard_normal((tokens, top_k, 1, n)).astype(np.float16)

    pair = module.gemv(mx.array(x), mx.array(indices))
    sorted_ids, _toks, order, inv = _union_operands(indices, top_k)
    got = module.gemv_union(mx.array(x), sorted_ids, order.astype(mx.uint32))
    unsorted = got[inv].reshape(tokens, top_k, 1, module.out_features)
    assert np.array_equal(np.asarray(unsorted.view(mx.uint16)),
                          np.asarray(pair.view(mx.uint16)))
    want = _reference_paired_projection(reference, x, indices)
    rel = np.max(np.abs(np.asarray(unsorted, dtype=np.float64).reshape(
        want.shape) - want)) / np.max(np.abs(want))
    assert rel < 1.5e-3, rel


@pytest.mark.parametrize("member", IQ2_MEMBERS)
def test_union_gemv_covers_the_run_extremes(member):
    """A single all-pairs run and an all-distinct selection both serve."""
    n = 2048
    module, _ = _module(member, n, seed=79)
    rng = np.random.default_rng(83)
    tokens, top_k = 4, 2
    x = rng.standard_normal((tokens, 1, 1, n)).astype(np.float16)

    for indices in (
        np.full((tokens, top_k), 3, dtype=np.uint32),
        np.arange(tokens * top_k, dtype=np.uint32).reshape(tokens, top_k),
    ):
        pair = module.gemv(mx.array(x), mx.array(indices))
        sorted_ids, toks, _order, inv = _union_operands(indices, top_k)
        got = module.gemv_union(mx.array(x), sorted_ids, toks)
        unsorted = got[inv].reshape(tokens, top_k, 1, module.out_features)
        assert np.array_equal(np.asarray(unsorted.view(mx.uint16)),
                              np.asarray(pair.view(mx.uint16)))


@pytest.mark.parametrize("member", IQ2_MEMBERS)
@pytest.mark.parametrize("n", (2048, 4096))
def test_sorted_gemv_is_bit_identical_to_the_pair_gemv(member, n):
    """The adjacency route's rows are the incumbent GEMV's, reordered."""
    module, _ = _module(member, n, seed=wirepack.seed_for(member, n, "adj"))
    rng = np.random.default_rng(91)
    tokens, top_k = 6, 6
    indices = rng.integers(0, 5, size=(tokens, top_k)).astype(np.uint32)
    x = rng.standard_normal((tokens, 1, 1, n)).astype(np.float16)

    pair = module.gemv(mx.array(x), mx.array(indices))
    sorted_ids, toks, _order, inv = _union_operands(indices, top_k)
    got = module.gemv_sorted(mx.array(x), sorted_ids, toks)
    unsorted = got[inv].reshape(tokens, top_k, 1, module.out_features)
    assert np.array_equal(np.asarray(unsorted.view(mx.uint16)),
                          np.asarray(pair.view(mx.uint16)))


@pytest.mark.parametrize("member", IQ2_MEMBERS)
def test_sorted_gemv_refuses_a_toks_pair_mismatch(member):
    module, _ = _module(member, 2048, seed=93)
    x = mx.zeros((2, 1, 1, 2048), dtype=mx.float16)
    sorted_ids = mx.zeros((8,), dtype=mx.uint32)
    toks = mx.zeros((6,), dtype=mx.uint32)
    with pytest.raises(IkqRuntimeError):
        module.gemv_sorted(x, sorted_ids, toks)


@pytest.mark.parametrize("member", IQ2_MEMBERS)
def test_union_gemv_refuses_a_toks_pair_mismatch(member):
    module, _ = _module(member, 2048, seed=89)
    x = mx.zeros((2, 1, 1, 2048), dtype=mx.float16)
    sorted_ids = mx.zeros((8,), dtype=mx.uint32)
    toks = mx.zeros((6,), dtype=mx.uint32)
    with pytest.raises(IkqRuntimeError):
        module.gemv_union(x, sorted_ids, toks)


@pytest.mark.parametrize("member", fmt.MEMBERS)
def test_gemv_refuses_a_pair_count_that_does_not_tile_the_rows(member):
    module, _ = _module(member, 2048, seed=61)
    x = mx.zeros((3, 1, 1, 2048), dtype=mx.float16)
    indices = mx.zeros((1, 4), dtype=mx.uint32)
    with pytest.raises(IkqRuntimeError):
        module.gemv(x, indices)


def test_one_layer_can_mix_members():
    """The three projections of one layer, each with its own member and width.

    Nothing couples a layer to a single member: the gate takes ``iq2_ks`` at
    the 4096 width, the up projection takes ``iq2_k`` at the same width, and
    the down projection takes ``iq1_s_r4`` at the 2048 width. Each is
    checked against its own reference in one dispatch sequence.
    """
    rng = np.random.default_rng(47)
    projections = [
        ("gate", *_module("iq2_ks", 4096, seed=51), 4096),
        ("up", *_module("iq2_k", 4096, seed=53), 4096),
        ("down", *_module("iq1_s_r4", 2048, seed=57, out_features=64), 2048),
    ]
    members = {module.member for _name, module, _ref, _n in projections}
    assert members == {"iq2_ks", "iq2_k", "iq1_s_r4"}

    tokens, top_k = 4, 4
    indices = rng.integers(0, EXPERTS, size=(tokens, top_k)).astype(np.uint32)
    im = mx.array(indices)
    for name, module, reference, n in projections:
        x = rng.standard_normal((tokens, 1, 1, n)).astype(np.float16)
        got = np.asarray(module(mx.array(x), im), dtype=np.float64)
        assert got.shape == (tokens, top_k, 1, module.out_features), name
        want = _reference_projection(reference, x, indices)
        rel = np.max(np.abs(got.reshape(want.shape) - want)) / np.max(np.abs(want))
        assert rel < 1.5e-3, (name, rel)


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", (2048, 4096))
def test_dequantized_range_matches_the_full_slice(member, n):
    """A range dequantization is the same bits as the full kernel's slice."""
    module, _ = _module(member, n, seed=wirepack.seed_for(member, n, "range"))
    full = np.asarray(module.dequantized().view(mx.uint16))
    for start, rows in ((0, 8), (8, 8), (16, 16), (0, 32), (24, 8)):
        got = np.asarray(module.dequantized_range(start, rows).view(mx.uint16))
        assert got.shape == (EXPERTS, rows, n), (start, rows)
        assert np.array_equal(got, full[:, start:start + rows]), (start, rows)


@pytest.mark.parametrize("member", fmt.MEMBERS)
def test_dequantized_range_refuses_a_bad_range(member):
    module, _ = _module(member, 2048, seed=61)
    for start, rows in ((0, 0), (0, 12), (0, -8), (-8, 8), (28, 8), (0, 64)):
        with pytest.raises(IkqRuntimeError):
            module.dequantized_range(start, rows)


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", (2048, 4096))
def test_sorted_matmul_range_concat_matches_the_sorted_call(member, n):
    """Range outputs concatenated along features equal the unsplit call.

    Each output element is one dot product over the full input width in
    both forms, so the comparison is on raw fp16 bit patterns.
    """
    module, _ = _module(member, n, seed=wirepack.seed_for(member, n, "nsplit"))
    rng = np.random.default_rng(67)
    tokens, top_k = 24, 4
    x = mx.array(rng.standard_normal((tokens, 1, 1, n)).astype(np.float16))
    indices = np.sort(rng.integers(0, EXPERTS, size=(tokens, top_k)), axis=None)
    indices = mx.array(indices.reshape(tokens, top_k).astype(np.uint32))
    full = module(x, indices, sorted_indices=True)
    half = OUT_FEATURES // 2
    parts = mx.concatenate(
        [module.sorted_matmul_range(x, indices, 0, half),
         module.sorted_matmul_range(x, indices, half, half)],
        axis=-1)
    got = np.asarray(parts.view(mx.uint16))
    want = np.asarray(full.view(mx.uint16)).reshape(got.shape)
    assert np.array_equal(got, want)


def test_iq1_s_r4_has_no_sorted_or_union_formulation():
    """The recorded sorted/union GEMVs cover the 2-bit members only."""
    module, _ = _module("iq1_s_r4", 2048, seed=97)
    x = mx.zeros((2, 1, 1, 2048), dtype=mx.float16)
    ids = mx.zeros((4,), dtype=mx.uint32)
    toks = mx.zeros((4,), dtype=mx.uint32)
    with pytest.raises(IkqKernelError):
        module.gemv_sorted(x, ids, toks)
    with pytest.raises(IkqKernelError):
        module.gemv_union(x, ids, toks)
