"""Dense kernel validation against the reference decode. No timing lives here.

The dense counterpart of ``tests/test_kernels.py``, resting on
``tests/test_dense_format.py`` (the reference decode is bit-identical to
ik's own dequantizers):

1. **The dense dequantization is bit-exact.** It performs the reference
   decode's float32 product chain in the same order and stores fp16;
   compared for exact equality, not to a tolerance. The range form must
   equal the same slice of the full form.
2. **The dense decode GEMV matches a float64 matmul** of the
   reference-decoded weights, at every sized width including 1024 and 8192
   and at the lm_head shape, for fp16 and bfloat16 activations at the call
   seam.
"""

from __future__ import annotations

import hashlib

import mlx.core as mx
import numpy as np
import pytest

import wirepack
from mlx_iqk import codec
from mlx_iqk import format as fmt
from mlx_iqk.dense import (
    check_dense_streams,
    dense_dequantized,
    dense_dequantized_range,
    dense_gemv,
    dense_linear,
)
from mlx_iqk.dense_kernels import (
    DENSE_SUPPORTED_IN_FEATURES,
    IqkKernelError,
    dense_dequant_input_names,
    dense_dequant_kernel,
    dense_gemv_input_names,
    dense_gemv_kernel,
    dense_gemv_source,
    dense_gemv_threads,
)

pytestmark = pytest.mark.gpu

OUT_FEATURES = 32
SAMPLE_TOKENS = 5

LM_HEAD_OUT, LM_HEAD_IN = 129280, 4096
LM_HEAD_MEMBERS = ("iq4_ks", "iq6_k")


def _streams(member: str, out_features: int, n: int, wire: np.ndarray):
    packed = fmt.pack(member, wire, n)
    shapes = fmt.dense_component_shapes(member, out_features, n)
    return {name: mx.array(np.ascontiguousarray(value).reshape(shapes[name]))
            for name, value in packed.items()}


def _random_case(member: str, out_features: int, n: int, seed: int):
    wire = wirepack.random_wire(member, out_features, n, seed=seed,
                                scales="serving")
    streams = _streams(member, out_features, n, wire)
    reference = fmt.decode_wire(member, wire, n)
    return streams, reference


def _fp16_ulp(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    def order(a):
        i = a.astype(np.float16).view(np.int16).astype(np.int64)
        return np.where(i < 0, np.int64(-32768) - i, i)
    return np.abs(order(got) - order(want))


# -- geometry guards ----------------------------------------------------------


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
@pytest.mark.parametrize("bad", [512, 1536, 2560, 3072, 4352, 16384])
def test_dense_gemv_refuses_unsized_input_widths(member, bad):
    with pytest.raises(IqkKernelError):
        dense_gemv_kernel(member, bad, OUT_FEATURES)


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
@pytest.mark.parametrize("bad", [512, 2560, 16384])
def test_dense_dequant_refuses_unsized_input_widths(member, bad):
    with pytest.raises(IqkKernelError):
        dense_dequant_kernel(member, bad, OUT_FEATURES)


@pytest.mark.parametrize("bad", [24, 40, 100])
def test_dense_gemv_refuses_out_features_that_do_not_tile(bad):
    with pytest.raises(IqkKernelError):
        dense_gemv_kernel("iq4_ks", 2048, bad)


def test_dense_gemv_refuses_a_row_width_that_is_not_a_block_multiple():
    with pytest.raises(fmt.IqkFormatError):
        dense_gemv_kernel("iq4_ks", 4000, OUT_FEATURES)


def test_every_dense_width_is_a_real_variant():
    assert DENSE_SUPPORTED_IN_FEATURES == (1024, 2048, 4096, 8192)
    assert dense_gemv_threads(1024) == 32
    assert dense_gemv_threads(8192) == 256


def test_kernel_sources_are_distinct_per_member_and_width():
    digests = {
        (member, n): hashlib.sha256(
            dense_gemv_source(member, n, OUT_FEATURES).encode()).hexdigest()
        for member in fmt.DENSE_MEMBERS for n in DENSE_SUPPORTED_IN_FEATURES
    }
    assert len(set(digests.values())) == len(digests)


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
def test_kernel_input_names_follow_the_format_stream_order(member):
    streams = list(fmt.dense_component_shapes(member, 1, 2048))
    assert dense_gemv_input_names(member) == ["x"] + streams + ["vtab"]
    assert dense_dequant_input_names(member) == streams + ["vtab"]


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
def test_stream_checks_fail_closed(member):
    from mlx_iqk.dense import IqkDenseError
    streams, _ = _random_case(member, OUT_FEATURES, 1024, seed=3)
    good = check_dense_streams(member, streams, OUT_FEATURES, 1024)
    assert len(good) == len(streams)
    broken = dict(streams)
    broken.pop("qs")
    with pytest.raises(IqkDenseError):
        check_dense_streams(member, broken, OUT_FEATURES, 1024)
    broken = dict(streams)
    broken["scl"] = broken["scl"][:, :-1]
    with pytest.raises(IqkDenseError):
        check_dense_streams(member, broken, OUT_FEATURES, 1024)


# -- the dense dequantization is bit-exact ------------------------------------


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
@pytest.mark.parametrize("n", DENSE_SUPPORTED_IN_FEATURES)
def test_dense_dequant_is_bit_identical_to_the_reference(member, n):
    streams, reference = _random_case(member, OUT_FEATURES, n,
                                      seed=wirepack.seed_for("dq", member, n))
    got = np.asarray(dense_dequantized(member, streams, OUT_FEATURES, n))
    want = reference.astype(np.float16)
    mismatches = int(np.sum(got.view(np.uint16) != want.view(np.uint16)))
    assert mismatches == 0, f"{mismatches} of {got.size}"


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
def test_dense_dequant_is_bit_identical_on_converted_rows(member):
    n = 2048
    rng = np.random.default_rng(4242)
    rows = (rng.standard_normal((OUT_FEATURES, n)) * 0.05).astype(np.float32)
    imatrix = rng.random(n).astype(np.float32)
    wire = codec.quantize(member, rows, imatrix)
    streams = _streams(member, OUT_FEATURES, n, wire)
    got = np.asarray(dense_dequantized(member, streams, OUT_FEATURES, n))
    want = fmt.decode_wire(member, wire, n).astype(np.float16)
    assert int(np.sum(got.view(np.uint16) != want.view(np.uint16))) == 0


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
@pytest.mark.parametrize("start,rows", [(0, 16), (16, 16), (8, 8)])
def test_dense_dequant_range_equals_the_full_slice(member, start, rows):
    n = 1024
    streams, _ = _random_case(member, OUT_FEATURES, n,
                              seed=wirepack.seed_for("rng", member, n))
    full = np.asarray(dense_dequantized(member, streams, OUT_FEATURES, n))
    part = np.asarray(dense_dequantized_range(
        member, streams, OUT_FEATURES, n, start, rows))
    assert np.array_equal(part.view(np.uint16),
                          full[start:start + rows].view(np.uint16))


# -- the dense decode GEMV matches a float64 reference ------------------------


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
@pytest.mark.parametrize("n", DENSE_SUPPORTED_IN_FEATURES)
def test_dense_gemv_matches_a_float64_reference(member, n):
    streams, reference = _random_case(member, OUT_FEATURES, n,
                                      seed=wirepack.seed_for("gemv", member, n))
    rng = np.random.default_rng(17)
    x = rng.standard_normal((SAMPLE_TOKENS, n)).astype(np.float16)
    got = np.asarray(dense_gemv(member, mx.array(x), streams, OUT_FEATURES, n))
    assert got.shape == (SAMPLE_TOKENS, OUT_FEATURES)

    want = x.astype(np.float64) @ reference.astype(np.float64).T
    rel = np.max(np.abs(got.astype(np.float64) - want)) / np.max(np.abs(want))
    # The kernel stores fp16, so the achievable relative error is one fp16
    # rounding of the result, not the float32 tree-reduction class.
    assert rel < 1.5e-3, rel
    ulp = _fp16_ulp(got, want.astype(np.float16))
    exact = float(np.mean(ulp == 0))
    assert exact > 0.99, (exact, int(ulp.max()))


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_dense_gemv_serves_both_activation_dtypes(member, dtype):
    """bfloat16 casts exactly to fp16 at the seam for in-range values."""
    n = 2048
    streams, reference = _random_case(member, OUT_FEATURES, n,
                                      seed=wirepack.seed_for("dt", member, n))
    rng = np.random.default_rng(29)
    x16 = rng.standard_normal((3, n)).astype(np.float16)
    x = mx.array(x16).astype(getattr(mx, dtype))
    # Values already representable in bfloat16 and fp16 make the cast the
    # identity, so both dtypes must produce the same bits.
    x = mx.array(np.asarray(x.astype(mx.float32)).astype(np.float16)).astype(
        getattr(mx, dtype))
    got = np.asarray(dense_gemv(member, x, streams, OUT_FEATURES, n))
    xr = np.asarray(x.astype(mx.float32)).astype(np.float64)
    want = xr @ reference.astype(np.float64).T
    rel = np.max(np.abs(got.astype(np.float64) - want)) / np.max(np.abs(want))
    assert rel < 1.5e-3, rel


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
def test_dense_gemv_reads_the_row_the_output_names(member):
    """A wrong row stride would still look plausible on one row."""
    n = 1024
    streams, reference = _random_case(member, OUT_FEATURES, n, seed=555)
    rng = np.random.default_rng(23)
    x = rng.standard_normal((1, n)).astype(np.float16)
    got = np.asarray(dense_gemv(member, mx.array(x), streams,
                                OUT_FEATURES, n)).reshape(-1)
    want = reference.astype(np.float64) @ x[0].astype(np.float64)
    for row in range(OUT_FEATURES):
        dev = abs(float(got[row]) - want[row])
        assert dev <= max(1.5e-3 * np.max(np.abs(want)), 1e-6), row


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
def test_dense_linear_routes_agree(member):
    """The GEMV and the dequantize-and-matmul route compute the same math.

    Different reduction orders, so agreement is numerical, not bitwise.
    """
    n = 1024
    streams, _ = _random_case(member, OUT_FEATURES, n,
                              seed=wirepack.seed_for("route", member, n))
    rng = np.random.default_rng(31)
    x = mx.array(rng.standard_normal((4, n)).astype(np.float16))
    via_gemv = np.asarray(dense_linear(member, x, streams, OUT_FEATURES, n,
                                       token_limit=64)).astype(np.float64)
    via_mm = np.asarray(dense_linear(member, x, streams, OUT_FEATURES, n,
                                     token_limit=0)).astype(np.float64)
    scale = max(np.max(np.abs(via_mm)), 1e-6)
    assert np.max(np.abs(via_gemv - via_mm)) / scale < 3e-3


# -- the lm_head shape ---------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("member", LM_HEAD_MEMBERS)
def test_dense_gemv_at_the_lm_head_shape(member):
    """One token through the full 129280 x 4096 head, sampled rows checked."""
    wire = wirepack.random_wire(member, LM_HEAD_OUT, LM_HEAD_IN,
                                seed=wirepack.seed_for("head", member),
                                scales="serving")
    streams = _streams(member, LM_HEAD_OUT, LM_HEAD_IN, wire)
    rng = np.random.default_rng(37)
    x = rng.standard_normal((1, LM_HEAD_IN)).astype(np.float16)
    got = np.asarray(dense_gemv(member, mx.array(x), streams,
                                LM_HEAD_OUT, LM_HEAD_IN)).reshape(-1)
    assert got.shape == (LM_HEAD_OUT,)

    rows = np.sort(rng.choice(LM_HEAD_OUT, 512, replace=False))
    weights = fmt.decode_wire(member, np.ascontiguousarray(wire[rows]), LM_HEAD_IN)
    want = weights.astype(np.float64) @ x[0].astype(np.float64)
    rel = np.max(np.abs(got[rows].astype(np.float64) - want)) / np.max(np.abs(want))
    assert rel < 1.5e-3, rel


@pytest.mark.slow
@pytest.mark.parametrize("member", LM_HEAD_MEMBERS)
def test_dense_dequant_range_covers_the_lm_head_tail(member):
    """129280 tiles as 126 x 1024-row ranges plus one 256-row tail."""
    assert 126 * 1024 + 256 == LM_HEAD_OUT
    n = 1024
    out = 1024 + 256    # the tail arithmetic at a testable height
    wire = wirepack.random_wire(member, out, n,
                                seed=wirepack.seed_for("tail", member),
                                scales="serving")
    streams = _streams(member, out, n, wire)
    tail_start = (out // 1024) * 1024
    part = np.asarray(dense_dequantized_range(
        member, streams, out, n, tail_start, out - tail_start))
    want = fmt.decode_wire(
        member, np.ascontiguousarray(wire[tail_start:]), n).astype(np.float16)
    assert np.array_equal(part.view(np.uint16), want.view(np.uint16))
