"""The dense relayout is bit-identical to ik's own dequantizers.

The dense counterpart of ``tests/test_format.py``: before the reference
decode can validate a kernel it has itself to be shown equal to
``dequantize_row_iq4_ks``, ``dequantize_row_iq4_k``, ``dequantize_row_iq5_k``
and ``dequantize_row_iq6_k``. Two sweeps do that per member: an enumeration
of the decode value space and a random sweep over the byte space. The IQ6_K
gate additionally pins the embedded cubic-output table to the vendored
codec's own bits.
"""

from __future__ import annotations

import numpy as np
import pytest

import wirepack
from mlx_iqk import codec
from mlx_iqk import format as fmt

WIDTHS = (256, 512, 1024, 2048, 4096, 8192)

DENSE_ROW_BYTES = {"iq4_ks": 136, "iq4_k": 144, "iq5_k": 176, "iq6_k": 212}


def _bit_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a.astype(np.float32).view(np.uint32)
                      != b.astype(np.float32).view(np.uint32)))


# -- geometry ---------------------------------------------------------------


def test_row_bytes_match_the_declared_budgets():
    assert fmt.ik_row_bytes("iq4_ks", 256) == 4 + 136
    assert fmt.ik_row_bytes("iq4_k", 256) == 144
    assert fmt.ik_row_bytes("iq5_k", 256) == 176
    assert fmt.ik_row_bytes("iq6_k", 256) == 212
    assert fmt.ik_row_bytes("iq4_ks", 4096) == 4 + 136 * 16
    assert fmt.ik_row_bytes("iq6_k", 8192) == 212 * 32


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
@pytest.mark.parametrize("n", WIDTHS)
def test_relayout_costs_exactly_what_the_member_costs(member, n):
    assert fmt.relayout_row_bytes(member, n) == fmt.ik_row_bytes(member, n)


def test_bits_per_weight():
    assert fmt.bits_per_weight("iq4_ks", 4096) == 4.25 + 32 / 4096
    assert fmt.bits_per_weight("iq4_k", 4096) == 4.5
    assert fmt.bits_per_weight("iq5_k", 4096) == 5.5
    assert fmt.bits_per_weight("iq6_k", 4096) == 6.625


def test_dense_member_guard():
    with pytest.raises(fmt.IqkFormatError):
        fmt.check_dense_member("iq2_ks")
    with pytest.raises(fmt.IqkFormatError):
        fmt.check_dense_member("iq3_k")
    with pytest.raises(fmt.IqkFormatError):
        fmt.check_any_member("iq3_k")


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
def test_component_shapes_cover_every_byte_once(member):
    o, n = 32, 2048
    shapes = fmt.dense_component_shapes(member, o, n)
    dtypes = fmt.dense_component_dtypes(member)
    assert set(shapes) == set(dtypes)
    total = 0
    for name, shape in shapes.items():
        count = 1
        for dim in shape:
            count *= dim
        total += count * dtypes[name].itemsize
    assert total == o * fmt.ik_row_bytes(member, n)


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
def test_pack_rejects_a_wire_row_of_the_wrong_length(member):
    wire = np.zeros((2, fmt.ik_row_bytes(member, 2048) + 1), dtype=np.uint8)
    with pytest.raises(fmt.IqkFormatError):
        fmt.pack(member, wire, 2048)


def test_vendored_block_sizes_match_the_wire_spec():
    lib = codec.load_dense()
    for member, expect in DENSE_ROW_BYTES.items():
        assert getattr(lib, f"iqk_block_size_{member}")() == expect
        meta = 4 if member == "iq4_ks" else 0
        assert getattr(lib, f"iqk_row_size_{member}")(2048) == meta + expect * 8
        assert getattr(lib, f"iqk_row_size_{member}")(2047) == 0


# -- the IQ6_K table pin ------------------------------------------------------


def test_iq6k_table_matches_the_vendored_codec_bit_for_bit():
    """The embedded cubic-output table is the vendored dequantizer's own.

    A probe block with a unit fp16 scale and unit sub-block scales decodes
    to the reconstruction values themselves, so the decode output at codes
    0..63 under both alphabets is the table. A codec rebuild that changed
    the compiled polynomial's bits fails here rather than drifting silently.
    """
    w = np.arange(256)
    codes = (w % 64)[None, :]
    scale_bytes = np.ones((1, 16), dtype=np.int64)
    ebit = np.zeros((1, 16), dtype=np.int64)
    ebit[0, 8:] = 1
    wire = wirepack.pack_iq6_k(codes, scale_bytes, ebit,
                               np.ones(1, dtype=np.float16))
    decoded = codec.dequantize("iq6_k", wire, 256)[0]
    table = np.concatenate([decoded[0:64], decoded[128:192]]).astype(np.float32)
    assert np.array_equal(table.view(np.uint32), fmt.IQ6K_TABLE.view(np.uint32))


def test_iq6k_table_rounds_to_the_integer_table():
    rounded = np.round(fmt.IQ6K_TABLE).astype(np.int64)
    assert np.array_equal(rounded, fmt.IQ6NL_VALUES.astype(np.int64))
    off = np.abs(fmt.IQ6K_TABLE - rounded.astype(np.float32))
    assert 0.4 < float(off.max()) < 0.5


# -- bit exactness against ik's own dequantizers ----------------------------


@pytest.mark.slow
def test_iq4_ks_exhaustive_value_space_is_bit_identical():
    wire = wirepack.exhaustive_iq4_ks(seed=41)
    ref = codec.dequantize("iq4_ks", wire, 256)
    got = fmt.decode_wire("iq4_ks", wire, 256)
    assert wire.shape[0] == 256 * 4
    assert _bit_mismatches(ref, got) == 0
    planes = fmt.decode_planes("iq4_ks", fmt.pack("iq4_ks", wire, 256), 256)
    assert set(np.unique(planes["codes"])) == set(range(16))
    assert set(np.unique(planes["scale_index"])) == set(range(-127, 128, 2))
    assert set(np.unique(planes["values"])) == {int(v) for v in fmt.IQ4K_VALUES}


@pytest.mark.slow
def test_iq4_k_exhaustive_value_space_is_bit_identical():
    wire = wirepack.exhaustive_iq4_k(seed=42)
    ref = codec.dequantize("iq4_k", wire, 256)
    got = fmt.decode_wire("iq4_k", wire, 256)
    assert wire.shape[0] == 64 * 2 * 4
    assert _bit_mismatches(ref, got) == 0
    planes = fmt.decode_planes("iq4_k", fmt.pack("iq4_k", wire, 256), 256)
    assert set(np.unique(planes["codes"])) == set(range(16))
    assert set(np.unique(planes["scale_index"])) == set(range(-32, 32))
    assert set(np.unique(planes["values"])) == {int(v) for v in fmt.IQ4K_VALUES}


@pytest.mark.slow
def test_iq5_k_exhaustive_value_space_is_bit_identical():
    wire = wirepack.exhaustive_iq5_k(seed=43)
    ref = codec.dequantize("iq5_k", wire, 256)
    got = fmt.decode_wire("iq5_k", wire, 256)
    assert wire.shape[0] == 64 * 2 * 4
    assert _bit_mismatches(ref, got) == 0
    planes = fmt.decode_planes("iq5_k", fmt.pack("iq5_k", wire, 256), 256)
    assert set(np.unique(planes["codes"])) == set(range(32))
    assert set(np.unique(planes["scale_index"])) == set(range(-32, 32))
    assert set(np.unique(planes["values"])) == {int(v) for v in fmt.IQ5NL_VALUES}


@pytest.mark.slow
def test_iq6_k_exhaustive_value_space_is_bit_identical():
    wire = wirepack.exhaustive_iq6_k(seed=44)
    ref = codec.dequantize("iq6_k", wire, 256)
    got = fmt.decode_wire("iq6_k", wire, 256)
    assert wire.shape[0] == 256 * 2 * 4
    assert _bit_mismatches(ref, got) == 0
    planes = fmt.decode_planes("iq6_k", fmt.pack("iq6_k", wire, 256), 256)
    assert set(np.unique(planes["codes"])) == set(range(64))
    assert set(np.unique(planes["scale_index"])) == set(range(-128, 128))


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
@pytest.mark.parametrize("n", WIDTHS)
def test_random_byte_space_is_bit_identical(member, n):
    wire = wirepack.random_wire(member, 24, n, seed=wirepack.seed_for(member, n))
    ref = codec.dequantize(member, wire, n)
    got = fmt.decode_wire(member, wire, n)
    assert _bit_mismatches(ref, got) == 0


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
@pytest.mark.parametrize("n", (1024, 4096))
@pytest.mark.parametrize("with_imatrix", (True, False))
def test_quantized_rows_relayout_bit_identically(member, n, with_imatrix):
    rng = np.random.default_rng(7)
    rows = rng.standard_normal((16, n)).astype(np.float32) * 0.05
    rows[3] = 0.0
    rows[5] *= 1e-4
    imatrix = rng.random(n).astype(np.float32) if with_imatrix else None
    if with_imatrix:
        imatrix[: n // 4] = 0.0     # a dead span, the uniform-collapse case
    wire = codec.quantize(member, rows, imatrix)
    ref = codec.dequantize(member, wire, n)
    got = fmt.decode_wire(member, wire, n)
    assert _bit_mismatches(ref, got) == 0


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
def test_pack_result_is_independent_of_the_chunk_size(member):
    wire = wirepack.random_wire(member, 37, 2048, seed=13)
    whole = fmt.pack(member, wire, 2048, chunk=1024)
    chunked = fmt.pack(member, wire, 2048, chunk=8)
    assert set(whole) == set(chunked)
    for name in whole:
        assert np.array_equal(whole[name].view(np.uint8),
                              chunked[name].view(np.uint8)), name


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
def test_pack_is_a_permutation_not_a_rewrite(member):
    """The relayout moves bits between planes and adds none."""
    wire = wirepack.random_wire(member, 4, 2048, seed=11)
    streams = fmt.pack(member, wire, 2048)
    packed = sum(int(np.asarray(v).nbytes) for v in streams.values())
    assert packed == 4 * fmt.ik_row_bytes(member, 2048)


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
@pytest.mark.parametrize("n", WIDTHS)
def test_unpack_returns_the_ik_wire_bytes_it_was_packed_from(member, n):
    wire = wirepack.random_wire(member, 24, n, seed=wirepack.seed_for(member, n, "rt"))
    back = fmt.unpack(member, fmt.pack(member, wire, n), n)
    assert back.dtype == np.uint8 and back.shape == wire.shape
    assert np.array_equal(back, wire)


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
def test_unpack_round_trips_quantized_rows(member):
    n = 2048
    rng = np.random.default_rng(17)
    rows = rng.standard_normal((12, n)).astype(np.float32) * 0.05
    rows[2] = 0.0
    wire = codec.quantize(member, rows, rng.random(n).astype(np.float32))
    assert np.array_equal(fmt.unpack(member, fmt.pack(member, wire, n), n), wire)


# -- stream shapes agree with the pack output --------------------------------


@pytest.mark.parametrize("member", fmt.DENSE_MEMBERS)
def test_packed_streams_reshape_to_the_declared_dense_shapes(member):
    o, n = 24, 2048
    wire = wirepack.random_wire(member, o, n, seed=23)
    streams = fmt.pack(member, wire, n)
    shapes = fmt.dense_component_shapes(member, o, n)
    dtypes = fmt.dense_component_dtypes(member)
    assert list(streams) == list(shapes)
    for name, value in streams.items():
        assert value.dtype == dtypes[name], name
        assert value.reshape(shapes[name]).shape == shapes[name], name
