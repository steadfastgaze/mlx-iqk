"""Kernel validation against the reference decode. No timing lives here.

Two layers, in the order the pricing cells depend on them:

1. **The dequantization kernel is bit-exact.** It performs the reference
   decode's float32 product chain in the same order and stores fp16, so
   replacing the reference with the kernel must not move a single stored
   weight. Compared for exact equality, not to a tolerance.
2. **The decode GEMV matches a float64 matmul** of the reference-dequantized
   weights at the routed geometry. Its reduction order differs (per-lane
   fused multiply-adds, a simdgroup sum, then the simdgroup partials), and it
   stores fp16, so it is compared by relative error and by fp16 ULP distance
   from the correctly rounded result.

Both layers rest on ``tests/test_format.py``, which shows the reference decode
is itself bit-identical to ik's own dequantizers.
"""

from __future__ import annotations

import hashlib

import mlx.core as mx
import numpy as np
import pytest

import wirepack
from mlx_iqk import codec
from mlx_iqk import format as fmt
from mlx_iqk.kernels import (
    SUPPORTED_IN_FEATURES,
    TABLE_INPUT_NAMES,
    IqkKernelError,
    dequant_input_names,
    dequant_kernel,
    gemv_input_names,
    gemv_kernel,
    gemv_source,
    gemv_threads,
    simdgroups,
)
from mlx_iqk.nn import IqkSwitchLinear

pytestmark = pytest.mark.gpu

EXPERTS = 4
OUT_FEATURES = 32
SAMPLE_TOKENS = 8
TOP_K = 3


def _module(member: str, n: int, wire: np.ndarray) -> tuple[IqkSwitchLinear, np.ndarray]:
    streams = fmt.pack(member, wire, n)
    reference = fmt.decode(member, streams, n).reshape(EXPERTS, OUT_FEATURES, n)
    module = IqkSwitchLinear(member, EXPERTS, OUT_FEATURES, n)
    shapes = fmt.component_shapes(member, EXPERTS, OUT_FEATURES, n)
    module.load_streams({name: mx.array(np.ascontiguousarray(value).reshape(shapes[name]))
                         for name, value in streams.items()})
    return module, reference


def _random_module(member: str, n: int, seed: int):
    wire = wirepack.random_wire(member, EXPERTS * OUT_FEATURES, n, seed=seed,
                                scales="serving")
    return _module(member, n, wire)


def _fp16_ulp(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Distance in fp16 representations between two fp16 arrays."""
    def order(a):
        i = a.astype(np.float16).view(np.int16).astype(np.int64)
        return np.where(i < 0, np.int64(-32768) - i, i)
    return np.abs(order(got) - order(want))


# -- geometry guards --------------------------------------------------------


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("bad", [1024, 1536, 2560, 3072, 4352, 8192])
def test_gemv_refuses_unsized_input_widths(member, bad):
    with pytest.raises(IqkKernelError):
        gemv_kernel(member, bad, OUT_FEATURES)


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("bad", [1024, 2560, 8192])
def test_dequant_refuses_unsized_input_widths(member, bad):
    with pytest.raises(IqkKernelError):
        dequant_kernel(member, bad, OUT_FEATURES)


@pytest.mark.parametrize("bad", [24, 40, 100])
def test_gemv_refuses_out_features_that_do_not_tile_a_threadgroup(bad):
    with pytest.raises(IqkKernelError):
        gemv_kernel("iq2_ks", 2048, bad)


def test_gemv_refuses_a_row_width_that_is_not_a_block_multiple():
    with pytest.raises(fmt.IqkFormatError):
        gemv_kernel("iq2_ks", 4000, OUT_FEATURES)


def test_module_construction_refuses_unsized_widths():
    with pytest.raises(IqkKernelError):
        IqkSwitchLinear("iq2_k", EXPERTS, OUT_FEATURES, 1024)


def test_both_routed_widths_are_real_variants():
    assert SUPPORTED_IN_FEATURES == (2048, 4096)
    assert simdgroups(2048) == 2 and gemv_threads(2048) == 64
    assert simdgroups(4096) == 4 and gemv_threads(4096) == 128


def test_kernel_sources_are_distinct_per_member_and_width():
    digests = {
        (member, n): hashlib.sha256(
            gemv_source(member, n, OUT_FEATURES).encode()).hexdigest()
        for member in fmt.MEMBERS for n in SUPPORTED_IN_FEATURES
    }
    assert len(set(digests.values())) == len(digests)


@pytest.mark.parametrize("member", fmt.MEMBERS)
def test_kernel_input_names_follow_the_format_stream_order(member):
    streams = list(fmt.component_shapes(member, 1, 1, 2048))
    table = TABLE_INPUT_NAMES[member]
    assert gemv_input_names(member) == ["x"] + streams + [table, "sel", "dims"]
    assert dequant_input_names(member) == streams + [table]


# -- the dequantization kernel is bit-exact ---------------------------------


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", SUPPORTED_IN_FEATURES)
def test_dequant_kernel_is_bit_identical_to_the_reference(member, n):
    module, reference = _random_module(member, n, seed=wirepack.seed_for(member, n))
    got = np.asarray(module.dequantized())
    want = reference.astype(np.float16)
    mismatches = int(np.sum(got.view(np.uint16) != want.view(np.uint16)))
    assert mismatches == 0, f"{mismatches} of {got.size}"


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", SUPPORTED_IN_FEATURES)
def test_dequant_kernel_is_bit_identical_on_converted_rows(member, n):
    rng = np.random.default_rng(4242)
    rows = (rng.standard_normal((EXPERTS * OUT_FEATURES, n)) * 0.05).astype(np.float32)
    imatrix = rng.random(n).astype(np.float32)
    wire = codec.quantize(member, rows, imatrix)
    module, reference = _module(member, n, wire)
    got = np.asarray(module.dequantized())
    want = reference.astype(np.float16)
    assert int(np.sum(got.view(np.uint16) != want.view(np.uint16))) == 0


# -- the decode GEMV matches a float64 reference ----------------------------


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", SUPPORTED_IN_FEATURES)
def test_gemv_matches_a_float64_reference(member, n):
    module, reference = _random_module(member, n, seed=wirepack.seed_for(member, n, "gemv"))
    rng = np.random.default_rng(17)
    x = rng.standard_normal((SAMPLE_TOKENS, 1, 1, n)).astype(np.float16)
    indices = rng.integers(0, EXPERTS, size=(SAMPLE_TOKENS, TOP_K)).astype(np.uint32)
    got = np.asarray(module.gemv(mx.array(x), mx.array(indices)))
    assert got.shape == (SAMPLE_TOKENS, TOP_K, 1, OUT_FEATURES)
    got = got.reshape(SAMPLE_TOKENS, TOP_K, OUT_FEATURES)

    want = np.zeros((SAMPLE_TOKENS, TOP_K, OUT_FEATURES), dtype=np.float64)
    for t in range(SAMPLE_TOKENS):
        for k in range(TOP_K):
            want[t, k] = (reference[indices[t, k]].astype(np.float64)
                          @ x[t, 0, 0].astype(np.float64))
    rel = np.max(np.abs(got.astype(np.float64) - want)) / np.max(np.abs(want))
    # The kernel stores fp16, so the achievable relative error is one fp16
    # rounding of the result, not the float32 tree-reduction class.
    assert rel < 1.5e-3, rel
    # Almost every output is the correctly rounded fp16 result. Over 40 seeds
    # per geometry the share at zero units in the last place is 99.92 to 99.96
    # percent, the remainder is one unit, and a near-cancelling output can
    # reach six units while staying inside the relative bound above. The gate
    # is on the share, so it does not depend on which sample a run drew.
    ulp = _fp16_ulp(got, want.astype(np.float16))
    exact = float(np.mean(ulp == 0))
    assert exact > 0.99, (exact, int(ulp.max()))


@pytest.mark.parametrize("member", fmt.MEMBERS)
def test_gemv_reads_the_expert_the_index_names(member):
    """A wrong expert stride would still look plausible against one expert."""
    n = 2048
    module, reference = _random_module(member, n, seed=555)
    rng = np.random.default_rng(23)
    x = rng.standard_normal((1, 1, 1, n)).astype(np.float16)
    for e in range(EXPERTS):
        indices = np.array([[e]], dtype=np.uint32)
        got = np.asarray(module.gemv(mx.array(x), mx.array(indices))).reshape(-1)
        want = (reference[e].astype(np.float64) @ x[0, 0, 0].astype(np.float64))
        rel = np.max(np.abs(got.astype(np.float64) - want)) / np.max(np.abs(want))
        assert rel < 1.5e-3, (e, rel)
