"""The relayout is bit-identical to ik's own dequantizers.

This is the gate every other claim in the repository rests on. The reference
decode in :mod:`mlx_ikq.format` operates on the k-contiguous relayout, so
before it can validate a kernel it has itself to be shown equal to
``dequantize_row_iq2_k`` and ``dequantize_row_iq2_ks``. Two sweeps do that:
an enumeration of the whole decode value space, and a random sweep over the
whole byte space.
"""

from __future__ import annotations

import numpy as np
import pytest

import wirepack
from mlx_ikq import codec
from mlx_ikq import format as fmt

WIDTHS = (256, 512, 2048, 4096)


def _bit_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a.astype(np.float32).view(np.uint32)
                      != b.astype(np.float32).view(np.uint32)))


# -- geometry ---------------------------------------------------------------


def test_row_bytes_match_the_declared_budgets():
    assert fmt.ik_row_bytes("iq2_k", 256) == 76
    assert fmt.ik_row_bytes("iq2_ks", 256) == 72
    assert fmt.ik_row_bytes("iq2_k", 4096) == 76 * 16
    assert fmt.ik_row_bytes("iq2_ks", 4096) == 2 + 70 * 16
    assert fmt.ik_row_bytes("iq1_s_r4", 32) == 8
    assert fmt.ik_row_bytes("iq1_s_r4", 2048) == 386
    assert fmt.ik_row_bytes("iq1_s_r4", 4096) == 770


# Every registered member's ggml row-size facts, independent of the tables
# `ik_row_bytes` consults: (row meta bytes, bytes per block, weights per
# block). `iq1_s_r4` blocks four rows together, so its per-row share is a
# quarter of the 24-byte group block.
ROW_SIZE_FACTS = {
    "iq2_ks": (2, 70, 256),
    "iq2_k": (0, 76, 256),
    "iq1_s_r4": (2, 6, 32),
    "iq4_ks": (4, 136, 256),
    "iq4_k": (0, 144, 256),
    "iq5_k": (0, 176, 256),
    "iq6_k": (0, 212, 256),
}


def test_row_size_facts_cover_every_registered_member():
    """A new member must state its row-size facts, not inherit a default."""
    assert set(ROW_SIZE_FACTS) == set(fmt.MEMBERS) | set(fmt.DENSE_MEMBERS)


@pytest.mark.parametrize("member", sorted(ROW_SIZE_FACTS))
@pytest.mark.parametrize("n", (256, 1024, 2048, 4096, 8192))
def test_ik_row_bytes_reproduces_ggml_row_size_per_member(member, n):
    """Row bytes recomputed from the struct facts, not from the same tables.

    A row-byte expression that divides by a constant block width instead of
    the member's own is wrong only for the member whose block is not 256, and
    every byte offset downstream inherits the error silently. Recomputing
    here from independently stated facts is what makes that fail loudly.
    """
    meta, block_bytes, weights_per_block = ROW_SIZE_FACTS[member]
    expected = meta + block_bytes * (n // weights_per_block)
    assert fmt.ik_row_bytes(member, n) == expected


@pytest.mark.parametrize("member", sorted(ROW_SIZE_FACTS))
def test_bits_per_weight_follows_the_row_size(member):
    meta, block_bytes, weights_per_block = ROW_SIZE_FACTS[member]
    n = 4096
    expected = (meta + block_bytes * (n // weights_per_block)) * 8.0 / n
    assert fmt.bits_per_weight(member, n) == expected


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", WIDTHS)
def test_relayout_costs_exactly_what_the_member_costs(member, n):
    assert fmt.relayout_row_bytes(member, n) == fmt.ik_row_bytes(member, n)


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
@pytest.mark.parametrize("n", WIDTHS)
def test_dense_relayout_costs_exactly_what_the_member_costs(member, n):
    assert fmt.relayout_row_bytes(member, n) == fmt.ik_row_bytes(member, n)


def test_bits_per_weight():
    assert fmt.bits_per_weight("iq2_k", 4096) == 2.375
    assert fmt.bits_per_weight("iq2_ks", 4096) == 2.1875 + 16 / 4096
    assert fmt.bits_per_weight("iq2_ks", 2048) == 2.1875 + 16 / 2048
    assert fmt.bits_per_weight("iq1_s_r4", 4096) == 1.5 + 16 / 4096
    assert fmt.bits_per_weight("iq1_s_r4", 2048) == 1.5 + 16 / 2048


def test_iq1_s_r4_width_follows_the_32_weight_block():
    assert fmt.check_row_width(96, "iq1_s_r4") == 96
    assert fmt.check_row_width(4000, "iq1_s_r4") == 4000
    for bad in (0, -32, 31, 100, 4001):
        with pytest.raises(fmt.IkqFormatError):
            fmt.check_row_width(bad, "iq1_s_r4")


def test_iq1_s_r4_wire_rows_come_in_groups_of_four():
    assert fmt.check_wire_rows("iq1_s_r4", 8) == 8
    for bad in (0, -4, 2, 6, 37):
        with pytest.raises(fmt.IkqFormatError):
            fmt.check_wire_rows("iq1_s_r4", bad)
    assert fmt.check_wire_rows("iq2_ks", 7) == 7


@pytest.mark.parametrize("bad", [0, -256, 255, 300, 4095])
def test_row_width_guard_rejects_non_multiples_of_the_block(bad):
    with pytest.raises(fmt.IkqFormatError):
        fmt.check_row_width(bad)


def test_unknown_member_is_rejected():
    with pytest.raises(fmt.IkqFormatError):
        fmt.check_member("iq3_k")


@pytest.mark.parametrize("member", ("iq2_ks", "iq2_k"))
def test_component_shapes_cover_every_weight_once(member):
    shapes = fmt.component_shapes(member, 3, 32, 2048)
    assert shapes["qs"] == (3, 32, 2048 // 16)
    sub = fmt.SUB_WEIGHTS[member]
    assert shapes["sex"] == (3, 32, 2048 // (8 * sub))


def test_iq1_s_r4_component_shapes_cover_every_weight_once():
    shapes = fmt.component_shapes("iq1_s_r4", 3, 32, 2048)
    assert shapes["qs"] == (3, 32, 2048 // 32)
    assert shapes["qh"] == (3, 32, 2048 // 32)
    assert shapes["dv"] == (3, 32)
    dtypes = fmt.component_dtypes("iq1_s_r4")
    assert dtypes["qs"].itemsize == 4 and dtypes["qh"].itemsize == 2


@pytest.mark.parametrize("member", fmt.MEMBERS)
def test_pack_rejects_a_wire_row_of_the_wrong_length(member):
    wire = np.zeros((2, fmt.ik_row_bytes(member, 2048) + 1), dtype=np.uint8)
    with pytest.raises(fmt.IkqFormatError):
        fmt.pack(member, wire, 2048)


# -- bit exactness against ik's own dequantizers ----------------------------


def test_vendored_block_sizes_match_the_wire_spec():
    lib = codec.load()
    assert lib.ikq_block_size_iq2_k() == fmt.IQ2_K_BLOCK_BYTES
    assert lib.ikq_block_size_iq2_ks() == fmt.IQ2_KS_BLOCK_BYTES
    lib1 = codec.load_iq1sr4()
    assert lib1.ikq_block_size_iq1_s_r4() == fmt.IQ1_S_R4_BLOCK_BYTES
    assert lib1.ikq_row_size_iq1_s_r4(4096) == 770
    assert lib1.ikq_row_size_iq1_s_r4(4001) == 0


@pytest.mark.slow
def test_iq2_k_exhaustive_value_space_is_bit_identical():
    wire = wirepack.exhaustive_iq2_k(seed=1)
    ref = codec.dequantize("iq2_k", wire, 256)
    got = fmt.decode_wire("iq2_k", wire, 256)
    assert wire.shape[0] == 16 * 2 * 4
    assert _bit_mismatches(ref, got) == 0
    planes = fmt.decode_planes("iq2_k", fmt.pack("iq2_k", wire, 256), 256)
    assert set(np.unique(planes["codes"])) == {0, 1, 2, 3}
    assert set(np.unique(planes["scale_index"])) == set(range(-8, 8))
    assert set(np.unique(planes["values"])) == {int(v) for v in fmt.IQ2NL_VALUES}


@pytest.mark.slow
def test_iq2_ks_exhaustive_value_space_is_bit_identical():
    wire = wirepack.exhaustive_iq2_ks(seed=2)
    ref = codec.dequantize("iq2_ks", wire, 256)
    got = fmt.decode_wire("iq2_ks", wire, 256)
    assert wire.shape[0] == 32 * 2 * 4
    assert _bit_mismatches(ref, got) == 0
    planes = fmt.decode_planes("iq2_ks", fmt.pack("iq2_ks", wire, 256), 256)
    assert set(np.unique(planes["codes"])) == {0, 1, 2, 3}
    assert set(np.unique(planes["scale_index"])) == set(range(-16, 16))
    assert set(np.unique(planes["values"])) == {int(v) for v in fmt.IQ2NL_VALUES}


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", WIDTHS)
def test_random_byte_space_is_bit_identical(member, n):
    wire = wirepack.random_wire(member, 24, n, seed=wirepack.seed_for(member, n))
    ref = codec.dequantize(member, wire, n)
    got = fmt.decode_wire(member, wire, n)
    assert _bit_mismatches(ref, got) == 0


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", (2048, 4096))
@pytest.mark.parametrize("with_imatrix", (True, False))
def test_quantized_rows_relayout_bit_identically(member, n, with_imatrix):
    rng = np.random.default_rng(7)
    rows = rng.standard_normal((16, n)).astype(np.float32) * 0.05
    rows[3] = 0.0
    rows[5] *= 1e-4
    imatrix = rng.random(n).astype(np.float32) if with_imatrix else None
    if with_imatrix:
        imatrix[: n // 4] = 0.0     # a route-starved span, the uniform-collapse case
    wire = codec.quantize(member, rows, imatrix)
    ref = codec.dequantize(member, wire, n)
    got = fmt.decode_wire(member, wire, n)
    assert _bit_mismatches(ref, got) == 0


@pytest.mark.parametrize("member", fmt.MEMBERS)
def test_pack_result_is_independent_of_the_chunk_size(member):
    wire = wirepack.random_wire(member, 36, 2048, seed=13)
    whole = fmt.pack(member, wire, 2048, chunk=1024)
    chunked = fmt.pack(member, wire, 2048, chunk=8)
    assert set(whole) == set(chunked)
    for name in whole:
        assert np.array_equal(whole[name].view(np.uint8),
                              chunked[name].view(np.uint8)), name


@pytest.mark.parametrize("member", fmt.MEMBERS)
def test_pack_is_a_permutation_not_a_rewrite(member):
    """The relayout moves bits between planes and adds none."""
    wire = wirepack.random_wire(member, 4, 2048, seed=11)
    streams = fmt.pack(member, wire, 2048)
    packed = sum(int(np.asarray(v).nbytes) for v in streams.values())
    assert packed == 4 * fmt.ik_row_bytes(member, 2048)


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", WIDTHS)
def test_unpack_returns_the_ik_wire_bytes_it_was_packed_from(member, n):
    """Every stored byte survives the relayout, over the random byte space.

    A consumer that rearranges a built package's bytes proves the
    rearrangement lost nothing on the bytes it actually moved by running
    this inverse over them, rather than inferring it from the transform
    being correct in general.
    """
    wire = wirepack.random_wire(member, 24, n, seed=wirepack.seed_for(member, n, "rt"))
    back = fmt.unpack(member, fmt.pack(member, wire, n), n)
    assert back.dtype == np.uint8 and back.shape == wire.shape
    assert np.array_equal(back, wire)


@pytest.mark.parametrize("member", fmt.MEMBERS)
@pytest.mark.parametrize("n", (2048, 4096))
def test_unpack_round_trips_quantized_rows(member, n):
    rng = np.random.default_rng(17)
    rows = rng.standard_normal((12, n)).astype(np.float32) * 0.05
    rows[2] = 0.0
    wire = codec.quantize(member, rows, rng.random(n).astype(np.float32))
    assert np.array_equal(fmt.unpack(member, fmt.pack(member, wire, n), n), wire)


@pytest.mark.slow
def test_iq1_s_r4_exhaustive_value_space_is_bit_identical():
    """Every (grid index, block position, scale, shift) combination decodes
    bit-identically to ik's own group dequantizer."""
    wire = wirepack.exhaustive_iq1_s_r4(seed=3)
    assert wire.shape[0] == 8 * 2 * 4 * 64
    ref = codec.dequantize("iq1_s_r4", wire, 256)
    got = fmt.decode_wire("iq1_s_r4", wire, 256)
    assert _bit_mismatches(ref, got) == 0
    planes = fmt.decode_planes("iq1_s_r4", fmt.pack("iq1_s_r4", wire, 256), 256)
    assert set(np.unique(planes["codes"])) == set(range(2048))
    assert set(np.unique(planes["values"])) == {-1, 0, 1}
    assert set(np.unique(planes["scale_index"])) == set(range(1, 16, 2))
    assert set(np.unique(planes["shift"])) == {np.float32(-0.125),
                                               np.float32(0.125)}


def test_iq1_s_r4_pack_refuses_a_row_count_that_breaks_groups():
    wire = wirepack.random_wire("iq1_s_r4", 8, 2048, seed=23)
    with pytest.raises(fmt.IkqFormatError):
        fmt.pack("iq1_s_r4", wire[:6], 2048)
    with pytest.raises(fmt.IkqFormatError):
        fmt.pack("iq1_s_r4", wire, 2048, chunk=6)


@pytest.mark.parametrize("member", fmt.MEMBERS)
def test_unpack_result_is_independent_of_the_chunk_size(member):
    wire = wirepack.random_wire(member, 36, 2048, seed=19)
    streams = fmt.pack(member, wire, 2048)
    assert np.array_equal(fmt.unpack(member, streams, 2048, chunk=1024),
                          fmt.unpack(member, streams, 2048, chunk=8))
