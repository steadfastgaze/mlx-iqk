"""IQ_K relayout wire format: constants, geometry, packing, reference decode.

This module is the single definition of the served wire format for the
IQ_K expert members this repository covers, ``IQ2_KS``, ``IQ2_K``, and
``IQ1_S_R4``. The encoder, the package writer, and the Metal kernels all
read the constants and the component shapes from here so the build side and
the serving side cannot drift apart.

Two layouts are in play and must never be confused.

**The ik wire layout** is what the ik quantizers emit. ``quantize_iq2_ks``
and ``quantize_iq2_k`` produce fixed-size super-blocks of 256 weights whose
code bytes interleave sub-blocks so that one byte carries codes from four
different 32-weight groups. ``quantize_iq1_s_r4`` interleaves across rows
instead: rows come in groups of four, each group opening with the four
rows' fp16 scales and continuing in 24-byte blocks that carry 32 weights of
each of the four rows side by side. A single ``IQ1_S_R4`` wire row is
therefore not self-contained; only whole four-row groups are addressable,
and :func:`pack` for that member consumes row counts in multiples of
:data:`WIRE_GROUP_ROWS`. The per-row information itself is self-contained
(the interleave is placement, not coding), which is what makes the per-row
relayout below possible.

**The relayout** is what this repository serves. It carries exactly the same
bits, re-laid so every stream is k-contiguous per output row:

``IQ2_KS`` (row of ``K`` weights, ``K`` a multiple of 256)

===========  ==================  =========================================
stream       granularity         content
===========  ==================  =========================================
``qs``       1 weight            2-bit code, little-endian in uint32 words
``scl``      32 weights          low 4 bits of the signed scale index
``sch``      32 weights          high bit of the signed scale index
``sex``      32 weights          alphabet-select bit
``dv``       1 row               fp16 row scale
===========  ==================  =========================================

``IQ2_K`` (row of ``K`` weights)

===========  ==================  =========================================
stream       granularity         content
===========  ==================  =========================================
``qs``       1 weight            2-bit code, little-endian in uint32 words
``scl``      16 weights          4-bit offset-8 signed scale index
``sex``      16 weights          alphabet-select bit
``dv``       256 weights         fp16 super-block scale
===========  ==================  =========================================

``IQ1_S_R4`` (row of ``K`` weights, ``K`` a multiple of 32)

===========  ==================  =========================================
stream       granularity         content
===========  ==================  =========================================
``qs``       8 weights           low 8 bits of the 11-bit grid index, four
                                 to a little-endian uint32 word
``qh``       32 weights          one uint16: four 3-bit grid-index high
                                 parts, the 3-bit block scale, the shift
                                 sign bit
``dv``       1 row               fp16 row scale
===========  ==================  =========================================

The relayout is byte-exact against the member it carries: ``IQ2_KS`` costs
``2.1875`` bits per weight plus 16 bits per row, ``IQ2_K`` costs ``2.375``
bits per weight, and ``IQ1_S_R4`` costs ``1.5`` bits per weight plus 16
bits per row, the same budgets ``block_iq2_ks``, ``block_iq2_k``, and
``block_iq1_s_r4`` spend. Nothing is added and nothing is dropped; only the
placement changes.

Reconstruction of one weight in the two 2-bit members is

``d * float(scale_index) * values[4 * alphabet + code]``

with ``values`` the eight-entry ``iq2nl_values`` table and ``scale_index``
signed. In ``IQ1_S_R4`` it is

``d * float(2 * ls + 1) * (float(grid_value) + shift)``

with ``grid_value`` a ternary value from the 2048-entry IQ1_S grid
(:mod:`mlx_ikq.iq1grid`), ``ls`` the 3-bit block scale, and ``shift`` a
per-block ``+-0.125``. Every arithmetic step is float32 in both chains.
That is the CPU dequantizer's own order of operations, which is the
bit-exactness reference for this repository. ik's Metal helpers hold the
scale in ``half`` and fold it into the alphabet before indexing, so they
are not bit-identical to the CPU path for these members and are not the
reference.

The value table is exact in fp16: every entry of ``iq2nl_values`` is a small
integer, so promoting the table to fp16 at build time changes no
reconstructed value while removing the dynamically indexed byte load that
prices two thirds of a byte-faithful decode's excess over its address floor.

The dense side of a package is covered by the ``DENSE_MEMBERS`` set below:
single-matrix tensors quantized as ``IQ4_KS``, ``IQ4_K``, ``IQ5_K``, or
``IQ6_K``, the degenerate single-expert case of the same relayout. Their
stream definitions, packers, and reference decoders live in the dense
section of this module; ``DESIGN.md`` at the repository root records the
geometry and the upstream citations.
"""

from __future__ import annotations

import numpy as np

from mlx_ikq.iq1grid import grid_values

QK_K = 256
"""Weights per ik super-block. Both 2-bit members declare this block size."""

IQ2NL_VALUES = np.array([-31, -13, 1, 17, -26, -8, 6, 22], dtype=np.int8)
"""``iq2nl_values``: the base alphabet in 0..3, the shifted one in 4..7.

The shifted alphabet is the base plus a uniform ``+5``, so the select bit
buys a scale-coupled offset rather than a constant bias.
"""

MEMBERS = ("iq2_ks", "iq2_k", "iq1_s_r4")
"""Routed-expert wire members with a relayout, an encoder shim, and kernels."""

DENSE_MEMBERS = ("iq4_ks", "iq4_k", "iq5_k", "iq6_k")
"""Dense-tensor wire members: the draft higher-bpw serving set."""

SUB_WEIGHTS = {
    "iq2_ks": 32, "iq2_k": 16, "iq1_s_r4": 32,
    "iq4_ks": 32, "iq4_k": 16, "iq5_k": 16, "iq6_k": 16,
}
"""Weights sharing one scale index (and, for the block members, one
alphabet-select bit; for ``iq1_s_r4``, one shift bit)."""

WIRE_GROUP_ROWS = {
    "iq2_ks": 1, "iq2_k": 1, "iq1_s_r4": 4,
    "iq4_ks": 1, "iq4_k": 1, "iq5_k": 1, "iq6_k": 1,
}
"""Rows one addressable ik-wire unit covers.

``iq1_s_r4`` interleaves four rows into one wire group, so wire byte
offsets are meaningful only at group boundaries and :func:`pack`,
:func:`unpack`, and the encoder shim consume row counts in multiples of
this. The relayout itself is per-row for every member.
"""

BLOCK_WEIGHTS = {
    "iq2_ks": QK_K, "iq2_k": QK_K, "iq1_s_r4": 32,
    "iq4_ks": QK_K, "iq4_k": QK_K, "iq5_k": QK_K, "iq6_k": QK_K,
}
"""Weights per wire block: the row-width granularity of each member.

``QK_K`` is the block size of every member but ``iq1_s_r4``, which blocks
32 weights; nothing here may assume the 256 constant per member.
"""

ALPHABET_SIZES = {"iq4_ks": 16, "iq4_k": 16, "iq5_k": 32, "iq6_k": 64}
"""Entries per alphabet of each dense member (two alphabets per member)."""

IQ4K_VALUES = np.array([
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117,
], dtype=np.int8)
"""``iq4k_values``: base alphabet in 0..15, shifted (base + 4) in 16..31.

Shared by ``IQ4_KS`` and ``IQ4_K``; every entry is exact in fp16.
"""

IQ5NL_VALUES = np.array([
    -126, -114, -103, -92, -83, -74, -65, -57, -50, -43, -36, -30, -24, -18,
    -12, -6, -1, 5, 11, 17, 23, 29, 36, 43, 51, 59, 68, 77, 87, 97, 109, 121,
    -124, -112, -101, -90, -81, -72, -63, -55, -48, -41, -34, -28, -22, -16,
    -10, -4, 1, 7, 13, 19, 25, 31, 38, 45, 53, 61, 70, 79, 89, 99, 111, 123,
], dtype=np.int8)
"""``iq5nl_values``: base alphabet in 0..31, shifted (base + 2) in 32..63."""

IQ6NL_VALUES = np.array([
    -127, -121, -115, -109, -104, -98, -93, -88, -84, -79, -74, -70, -66, -62,
    -58, -54, -51, -47, -44, -40, -37, -34, -31, -28, -25, -22, -19, -16, -13,
    -11, -8, -5, -2, 0, 3, 6, 9, 12, 14, 17, 20, 23, 27, 30, 33, 36, 40, 44,
    47, 51, 55, 59, 63, 68, 72, 77, 82, 87, 92, 98, 103, 109, 115, 121,
    -126, -120, -114, -108, -103, -97, -92, -87, -83, -78, -73, -69, -65, -61,
    -57, -53, -50, -46, -43, -39, -36, -33, -30, -27, -24, -21, -18, -15, -12,
    -10, -7, -4, -1, 1, 4, 7, 10, 13, 15, 18, 21, 24, 28, 31, 34, 37, 41, 45,
    48, 52, 56, 60, 64, 69, 73, 78, 83, 88, 93, 99, 104, 110, 116, 122,
], dtype=np.int8)
"""``iq6nl_values``: the integer table ik's quantizer and Metal path use.

Recorded for the geometry and for sanity checks. It is not the serving
reconstruction: see :data:`IQ6K_TABLE`.
"""

_IQ6K_TABLE_BITS = np.array([
    0xC2FE0000, 0xC2F1B557, 0xC2E5D9E1, 0xC2DA69EE, 0xC2CF61D2, 0xC2C4BDE0,
    0xC2BA7A69, 0xC2B093C1, 0xC2A70639, 0xC29DCE24, 0xC294E7D5, 0xC28C4F9F,
    0xC28401D2, 0xC277F587, 0xC2686D87, 0xC259644D, 0xC24AD27C, 0xC23CB0B6,
    0xC22EF7A6, 0xC2219FEE, 0xC214A232, 0xC207F718, 0xC1F72E90, 0xC1DEF6C0,
    0xC1C73814, 0xC1AFE3D6, 0xC198EB4A, 0xC1823FC0, 0xC157A4F8, 0xC12B299C,
    0xC0FDE002, 0xC0A5B558, 0xC01B3100, 0x3EAB16D8, 0x4046C6B4, 0x40BCB7EB,
    0x410B7516, 0x41391A72, 0x41676976, 0x418B3FC3, 0x41A33D0A, 0x41BBBB48,
    0x41D4C931, 0x41EE757D, 0x4204676E, 0x4211F208, 0x421FE1E1, 0x422E3E57,
    0x423D0EC6, 0x424C5A88, 0x425C28F9, 0x426C8170, 0x427D6B4F, 0x428776F5,
    0x42908850, 0x4299ED66, 0x42A3A9E2, 0x42ADC174, 0x42B837CC, 0x42C31092,
    0x42CE4F77, 0x42D9F828, 0x42E60E52, 0x42F295A4, 0xC2FC0000, 0xC2EFB557,
    0xC2E3D9E1, 0xC2D869EE, 0xC2CD61D2, 0xC2C2BDE0, 0xC2B87A69, 0xC2AE93C1,
    0xC2A50639, 0xC29BCE24, 0xC292E7D5, 0xC28A4F9F, 0xC28201D2, 0xC273F587,
    0xC2646D87, 0xC255644D, 0xC246D27C, 0xC238B0B6, 0xC22AF7A6, 0xC21D9FEE,
    0xC210A232, 0xC203F718, 0xC1EF2E90, 0xC1D6F6C0, 0xC1BF3814, 0xC1A7E3D6,
    0xC190EB4A, 0xC1747F80, 0xC147A4F8, 0xC11B299C, 0xC0DDE002, 0xC085B558,
    0xBFB66200, 0x3FAAC5B6, 0x4083635A, 0x40DCB7EB, 0x411B7516, 0x41491A72,
    0x41776976, 0x41933FC3, 0x41AB3D0A, 0x41C3BB48, 0x41DCC931, 0x41F6757D,
    0x4208676E, 0x4215F208, 0x4223E1E1, 0x42323E57, 0x42410EC6, 0x42505A88,
    0x426028F9, 0x42708170, 0x4280B5A8, 0x428976F5, 0x42928850, 0x429BED66,
    0x42A5A9E2, 0x42AFC174, 0x42BA37CC, 0x42C51092, 0x42D04F77, 0x42DBF828,
    0x42E80E52, 0x42F495A4,
], dtype=np.uint32)

IQ6K_TABLE = _IQ6K_TABLE_BITS.view(np.float32)
"""The 128 float32 values ``IQ6_K`` serving reconstructs, bit for bit.

ik's CPU dequantizer reconstructs a code through a cubic in the code index
rather than through :data:`IQ6NL_VALUES` (the cubic's outputs round to that
integer table but differ by up to 0.488 in float). These are the cubic's
float32 outputs as ik's own CPU dequantizer computes them, extracted from
the vendored codec by decoding a probe block with a unit scale, base
alphabet in 0..63 and shifted alphabet in 64..127. Embedded as bit patterns
because the compiled expression fuses multiply-adds, so no separately
written evaluation of the polynomial is guaranteed to reproduce it;
``tests/test_dense_format.py`` re-derives the table from the vendored codec
and asserts bit equality, so a codec rebuild that changed these bits would
surface as a failure rather than a silent drift.
"""

IQ2_KS_BLOCK_BYTES = 70
"""``sizeof(block_iq2_ks)``: extra 2 + scales 4 + qs 64."""

IQ2_KS_ROW_META_BYTES = 2
"""fp16 row scale carried ahead of the blocks of an ``IQ2_KS`` row."""

IQ2_K_BLOCK_BYTES = 76
"""``sizeof(block_iq2_k)``: d 2 + extra 2 + scales 8 + qs 64."""

IQ1_S_R4_BLOCK_BYTES = 24
"""``sizeof(block_iq1_s_r4)``: qs 16 + qh 8, covering 32 weights of each of
the four rows of one wire group."""

IQ1_S_R4_ROW_META_BYTES = 2
"""fp16 row scale, stored as a 4 x fp16 prefix on each four-row group and
accounted per row, exactly as ggml's ``row_meta_size`` charges it."""

IQ1S_DELTA = 0.125
"""The per-block reconstruction shift; the sign bit in ``qh`` selects it."""

IQ4_KS_BLOCK_BYTES = 136
"""``sizeof(block_iq4_ks)``: scales 8 + qs 128."""

IQ4_KS_ROW_META_BYTES = 4
"""fp32 row scale carried ahead of the blocks of an ``IQ4_KS`` row."""

IQ4_K_BLOCK_BYTES = 144
"""``sizeof(block_iq4_k)``: d 2 + extra 2 + scales_h 4 + scales_l 8 + qs 128."""

IQ5_K_BLOCK_BYTES = 176
"""``sizeof(block_iq5_k)``: IQ4_K's planes + qh 32."""

IQ6_K_BLOCK_BYTES = 212
"""``sizeof(block_iq6_k)``: d 2 + extra 2 + scales 16 + qs 128 + qh 64."""


class IkqFormatError(ValueError):
    pass


def check_member(member: str) -> str:
    if member not in MEMBERS:
        raise IkqFormatError(f"ikq routed member {member!r} not in {MEMBERS}")
    return member


def check_dense_member(member: str) -> str:
    if member not in DENSE_MEMBERS:
        raise IkqFormatError(f"ikq dense member {member!r} not in {DENSE_MEMBERS}")
    return member


def check_any_member(member: str) -> str:
    if member not in MEMBERS and member not in DENSE_MEMBERS:
        raise IkqFormatError(
            f"ikq member {member!r} not in {MEMBERS} or {DENSE_MEMBERS}")
    return member


def check_row_width(in_features: int, member: str = "iq2_ks") -> int:
    """Row widths the relayout defines. Not the kernels' guard, which is narrower.

    The block members, routed and dense alike, need a multiple of 256;
    ``iq1_s_r4`` needs a multiple of 32. The default keeps the historical
    one-argument call meaning the 256-block rule.
    """
    n = int(in_features)
    block = BLOCK_WEIGHTS[check_any_member(member)]
    if n <= 0 or n % block:
        raise IkqFormatError(
            f"in_features {in_features} is not a positive multiple of {block}; ik "
            "silently substitutes a different type for such a row rather than "
            "failing, so the width is rejected here instead")
    return n


def check_wire_rows(member: str, rows: int) -> int:
    """Row counts one member's ik wire can address.

    Every member but ``iq1_s_r4`` addresses single rows; the encoder shim
    calls this for routed and dense members alike.
    """
    group = WIRE_GROUP_ROWS[check_any_member(member)]
    r = int(rows)
    if r <= 0 or r % group:
        raise IkqFormatError(
            f"{member} wire rows come in groups of {group}; {rows} rows do "
            "not split into whole groups")
    return r


# Bytes each member spends per wire block, and the row-wide prefix ahead of
# them. `iq1_s_r4` blocks four rows together, so its per-row share is a
# quarter of the group block; charging the whole 24 here would read every
# row four times too long and no arithmetic downstream would catch it.
_BLOCK_BYTES = {
    "iq2_ks": IQ2_KS_BLOCK_BYTES,
    "iq2_k": IQ2_K_BLOCK_BYTES,
    "iq1_s_r4": IQ1_S_R4_BLOCK_BYTES // WIRE_GROUP_ROWS["iq1_s_r4"],
    "iq4_ks": IQ4_KS_BLOCK_BYTES,
    "iq4_k": IQ4_K_BLOCK_BYTES,
    "iq5_k": IQ5_K_BLOCK_BYTES,
    "iq6_k": IQ6_K_BLOCK_BYTES,
}
_ROW_META_BYTES = {
    "iq2_ks": IQ2_KS_ROW_META_BYTES,
    "iq1_s_r4": IQ1_S_R4_ROW_META_BYTES,
    "iq4_ks": IQ4_KS_ROW_META_BYTES,
}


def ik_row_bytes(member: str, in_features: int) -> int:
    """Bytes of one ik-layout row, matching ``ggml_row_size``.

    The block count divides by the member's own block width, never by the
    256 constant: ``iq1_s_r4`` blocks 32 weights.
    """
    check_any_member(member)
    n = check_row_width(in_features, member)
    blocks = n // BLOCK_WEIGHTS[member]
    return _ROW_META_BYTES.get(member, 0) + _BLOCK_BYTES[member] * blocks


def component_shapes(member: str, num_experts: int, out_features: int,
                     in_features: int) -> dict[str, tuple[int, ...]]:
    """Stacked-expert shapes of every relayout stream for one projection."""
    check_member(member)
    n = check_row_width(in_features, member)
    e, o = int(num_experts), int(out_features)
    if member == "iq2_ks":
        return {
            "qs": (e, o, n // 16),
            "scl": (e, o, n // 64),
            "sch": (e, o, n // 256),
            "sex": (e, o, n // 256),
            "dv": (e, o),
        }
    if member == "iq1_s_r4":
        return {
            "qs": (e, o, n // 32),
            "qh": (e, o, n // 32),
            "dv": (e, o),
        }
    return {
        "qs": (e, o, n // 16),
        "scl": (e, o, n // 32),
        "sex": (e, o, n // 128),
        "dv": (e, o, n // 256),
    }


def component_dtypes(member: str) -> dict[str, np.dtype]:
    """NumPy dtype of every relayout stream."""
    check_member(member)
    if member == "iq2_ks":
        names = ("qs", "scl", "sch", "sex", "dv")
    elif member == "iq1_s_r4":
        return {
            "qs": np.dtype(np.uint32),
            "qh": np.dtype(np.uint16),
            "dv": np.dtype(np.float16),
        }
    else:
        names = ("qs", "scl", "sex", "dv")
    return {n: (np.dtype(np.float16) if n == "dv" else
                np.dtype(np.uint32) if n == "qs" else np.dtype(np.uint8))
            for n in names}


def bits_per_weight(member: str, in_features: int) -> float:
    """Exact stored bits per weight of one relayout row, row meta charged."""
    return (8.0 * ik_row_bytes(member, in_features)
            / check_row_width(in_features, member))


def relayout_row_bytes(member: str, in_features: int) -> int:
    """Bytes of one relayout row. Equal to :func:`ik_row_bytes` by construction."""
    check_any_member(member)
    if member in DENSE_MEMBERS:
        shapes = dense_component_shapes(member, 1, in_features)
        dtypes = dense_component_dtypes(member)
    else:
        shapes = component_shapes(member, 1, 1, in_features)
        dtypes = component_dtypes(member)
    total = 0
    for name, shape in shapes.items():
        count = 1
        for dim in shape:
            count *= dim
        total += count * dtypes[name].itemsize
    return total


# ---------------------------------------------------------------------------
# ik wire layout -> relayout (build time, bit-exact)
# ---------------------------------------------------------------------------


def _bits_to_words(bits: np.ndarray) -> np.ndarray:
    """Pack a little-endian bit array of shape ``[rows, 32 * w]`` into words."""
    return np.packbits(bits, axis=1, bitorder="little").view(np.uint32)


def _codes_to_qs(codes: np.ndarray) -> np.ndarray:
    """Two-bit codes in k order to little-endian uint32 words."""
    rows, n = codes.shape
    bits = np.empty((rows, 2 * n), dtype=np.uint8)
    bits[:, 0::2] = codes & 1
    bits[:, 1::2] = (codes >> 1) & 1
    return _bits_to_words(bits)


def _qs_to_codes(qs: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`_codes_to_qs`."""
    rows = qs.shape[0]
    bits = np.unpackbits(np.ascontiguousarray(qs).view(np.uint8).reshape(rows, -1),
                         axis=1, bitorder="little")[:, : 2 * n].reshape(rows, n, 2)
    return (bits[:, :, 0] | (bits[:, :, 1] << 1)).astype(np.uint8)


def _ik_code_positions(member: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Byte index and bit shift of every weight of a row in the ik layout.

    ``IQ2_K``: weight ``w`` of a super-block sits in byte
    ``32*(ib32/4) + 16*half + j`` at shift ``2*(ib32%4)`` with ``ib32 = w/32``,
    ``half = (w%32)/16``, ``j = w%16``.

    ``IQ2_KS``: weight ``w`` sits in byte ``32*(ib64/2) + j`` at shift
    ``4*(ib64%2) + 2*half`` with ``ib64 = w/64``, ``half = (w%64)/32``,
    ``j = w%32``.
    """
    w = np.arange(QK_K, dtype=np.int64)
    if member == "iq2_k":
        ib32, rem = w // 32, w % 32
        half, j = rem // 16, rem % 16
        byte = 32 * (ib32 // 4) + 16 * half + j
        shift = 2 * (ib32 % 4)
    else:
        ib64, rem = w // 64, w % 64
        half, j = rem // 32, rem % 32
        byte = 32 * (ib64 // 2) + j
        shift = 4 * (ib64 % 2) + 2 * half
    nblocks = n // QK_K
    block = np.arange(nblocks, dtype=np.int64)[:, None]
    qs_bytes = 64
    return ((block * qs_bytes + byte[None, :]).reshape(-1),
            np.tile(shift, nblocks))


def _pack_iq2_ks_rows(wire: np.ndarray, in_features: int) -> dict[str, np.ndarray]:
    """Relayout one chunk of ``IQ2_KS`` rows from ik wire bytes."""
    n = check_row_width(in_features)
    rows = wire.shape[0]
    expect = ik_row_bytes("iq2_ks", n)
    if wire.shape[1] != expect:
        raise IkqFormatError(
            f"iq2_ks row of {n} weights is {expect} wire bytes, got {wire.shape[1]}")
    nblocks = n // QK_K
    dv = wire[:, :2].copy().view(np.float16).reshape(rows)
    body = wire[:, 2:].reshape(rows, nblocks, IQ2_KS_BLOCK_BYTES)

    extra = body[:, :, 0].astype(np.uint16) | (body[:, :, 1].astype(np.uint16) << 8)
    scales = np.ascontiguousarray(body[:, :, 2:6]).reshape(rows, -1)
    qs_bytes = np.ascontiguousarray(body[:, :, 6:]).reshape(rows, -1)

    byte_idx, shift = _ik_code_positions("iq2_ks", n)
    codes = (qs_bytes[:, byte_idx] >> shift[None, :]) & 3

    nsub = n // 32
    # Sub-block ``ib`` of a super-block reads its low four scale bits from
    # nibble ``ib`` of ``scales``, its high scale bit from ``extra`` bit
    # ``8 + ib``, and its alphabet-select bit from ``extra`` bit ``ib``. Since
    # ``ib`` is the k-order sub-block index, the low nibbles and both bit
    # planes are already k-contiguous within a super-block; the relayout
    # concatenates them across super-blocks and permutes only the codes.
    scl_lo = np.empty((rows, nsub), dtype=np.uint8)
    scl_lo[:, 0::2] = scales & 0xF
    scl_lo[:, 1::2] = scales >> 4
    sel = ((extra[:, :, None] >> np.arange(8, dtype=np.uint16)[None, None, :]) & 1)
    hi = ((extra[:, :, None] >> (8 + np.arange(8, dtype=np.uint16))[None, None, :]) & 1)
    return {
        "qs": _codes_to_qs(codes.astype(np.uint8)),
        "scl": (scl_lo[:, 0::2] | (scl_lo[:, 1::2] << 4)).astype(np.uint8),
        "sch": np.packbits(hi.reshape(rows, nsub).astype(np.uint8), axis=1,
                           bitorder="little"),
        "sex": np.packbits(sel.reshape(rows, nsub).astype(np.uint8), axis=1,
                           bitorder="little"),
        "dv": dv,
    }


def _pack_iq2_k_rows(wire: np.ndarray, in_features: int) -> dict[str, np.ndarray]:
    """Relayout one chunk of ``IQ2_K`` rows from ik wire bytes."""
    n = check_row_width(in_features)
    rows = wire.shape[0]
    expect = ik_row_bytes("iq2_k", n)
    if wire.shape[1] != expect:
        raise IkqFormatError(
            f"iq2_k row of {n} weights is {expect} wire bytes, got {wire.shape[1]}")
    nblocks = n // QK_K
    body = wire.reshape(rows, nblocks, IQ2_K_BLOCK_BYTES)

    dv = np.ascontiguousarray(body[:, :, 0:2]).reshape(rows, -1).copy()
    dv = dv.view(np.float16).reshape(rows, nblocks)
    extra = body[:, :, 2].astype(np.uint16) | (body[:, :, 3].astype(np.uint16) << 8)
    scales = np.ascontiguousarray(body[:, :, 4:12]).reshape(rows, -1)
    qs_bytes = np.ascontiguousarray(body[:, :, 12:]).reshape(rows, -1)

    byte_idx, shift = _ik_code_positions("iq2_k", n)
    codes = (qs_bytes[:, byte_idx] >> shift[None, :]) & 3

    nsub = n // 16
    # Sub-block ``ib`` reads nibble ``ib`` of ``scales`` and ``extra`` bit
    # ``ib``, and ``ib`` is the k-order sub-block index, so both planes are
    # already k-contiguous inside a super-block. Only the codes move.
    scl_lo = np.empty((rows, nsub), dtype=np.uint8)
    scl_lo[:, 0::2] = scales & 0xF
    scl_lo[:, 1::2] = scales >> 4
    sel = ((extra[:, :, None] >> np.arange(16, dtype=np.uint16)[None, None, :]) & 1)
    return {
        "qs": _codes_to_qs(codes.astype(np.uint8)),
        "scl": (scl_lo[:, 0::2] | (scl_lo[:, 1::2] << 4)).astype(np.uint8),
        "sex": np.packbits(sel.reshape(rows, nsub).astype(np.uint8), axis=1,
                           bitorder="little"),
        "dv": dv,
    }


def _pack_iq1_s_r4_rows(wire: np.ndarray, in_features: int) -> dict[str, np.ndarray]:
    """Relayout one chunk of ``IQ1_S_R4`` rows from ik wire bytes.

    ``wire`` is ``[rows, ik_row_bytes]`` uint8 with ``rows`` a multiple of
    four: the flat byte stream of consecutive four-row groups, reshaped at
    the per-row byte size. Row boundaries in that view are not meaningful;
    the group view recovered here is.
    """
    n = check_row_width(in_features, "iq1_s_r4")
    rows = check_wire_rows("iq1_s_r4", wire.shape[0])
    expect = ik_row_bytes("iq1_s_r4", n)
    if wire.shape[1] != expect:
        raise IkqFormatError(
            f"iq1_s_r4 row of {n} weights is {expect} wire bytes, got {wire.shape[1]}")
    nblock = n // 32
    groups = rows // 4
    g = np.ascontiguousarray(wire).reshape(groups, 4 * expect)

    dv = g[:, :8].copy().view(np.float16).reshape(rows)
    body = g[:, 8:].reshape(groups, nblock, IQ1_S_R4_BLOCK_BYTES)

    # qs[4*i + k] is the low index byte of 8-weight sub-group ``i`` of row
    # ``k``; qh[k] is row ``k``'s per-block word (index high bits, block
    # scale, shift sign). Both planes are self-contained per (row, block),
    # so the relayout only de-interleaves the rows.
    qs = np.ascontiguousarray(body[:, :, :16]).reshape(groups, nblock, 4, 4)
    qh = np.ascontiguousarray(body[:, :, 16:24]).reshape(
        groups, nblock, 8).view(np.uint16).reshape(groups, nblock, 4)
    qs_rows = np.ascontiguousarray(qs.transpose(0, 3, 1, 2)).reshape(
        rows, nblock * 4).view(np.uint32).reshape(rows, nblock)
    qh_rows = np.ascontiguousarray(qh.transpose(0, 2, 1)).reshape(rows, nblock)
    return {"qs": qs_rows, "qh": qh_rows, "dv": dv}


def _unpack_iq1_s_r4_rows(streams: dict[str, np.ndarray],
                          in_features: int) -> np.ndarray:
    n = check_row_width(in_features, "iq1_s_r4")
    qs = streams["qs"]
    rows = check_wire_rows("iq1_s_r4", qs.shape[0])
    nblock = n // 32
    groups = rows // 4
    expect = ik_row_bytes("iq1_s_r4", n)
    wire = np.empty((groups, 4 * expect), dtype=np.uint8)
    wire[:, :8] = np.ascontiguousarray(
        streams["dv"].astype(np.float16)).reshape(groups, 4).view(np.uint8)
    body = wire[:, 8:].reshape(groups, nblock, IQ1_S_R4_BLOCK_BYTES)
    qs_bytes = np.ascontiguousarray(qs).view(np.uint8).reshape(
        groups, 4, nblock, 4)
    body[:, :, :16] = qs_bytes.transpose(0, 2, 3, 1).reshape(groups, nblock, 16)
    qh = np.ascontiguousarray(streams["qh"]).reshape(groups, 4, nblock)
    body[:, :, 16:24] = np.ascontiguousarray(
        qh.transpose(0, 2, 1)).view(np.uint8).reshape(groups, nblock, 8)
    return wire.reshape(rows, expect)


PACK_CHUNK_ROWS = 2048
"""Rows one relayout pass holds at once.

The pack expands each row to one byte per weight while it moves the codes, so
a whole stacked expert tensor at once would cost far more than the tensor. The
chunk bounds that working set; the result is identical for any chunk size,
which ``tests/test_format.py`` asserts.
"""


def _pack_chunked(fn, wire: np.ndarray, in_features: int,
                  chunk: int, group: int = 1) -> dict[str, np.ndarray]:
    rows = wire.shape[0]
    if chunk <= 0 or chunk % group:
        raise IkqFormatError(
            f"pack chunk {chunk} is not a positive multiple of the "
            f"{group}-row wire group")
    if rows <= chunk:
        return fn(wire, in_features)
    parts = [fn(wire[start:start + chunk], in_features)
             for start in range(0, rows, chunk)]
    return {name: np.concatenate([p[name] for p in parts], axis=0)
            for name in parts[0]}


def pack_iq2_ks(wire: np.ndarray, in_features: int,
                chunk: int = PACK_CHUNK_ROWS) -> dict[str, np.ndarray]:
    """Relayout ``IQ2_KS`` rows from ik wire bytes.

    ``wire`` is ``[rows, ik_row_bytes]`` uint8. Returns the five relayout
    streams for those rows.
    """
    return _pack_chunked(_pack_iq2_ks_rows, wire, in_features, chunk)


def pack_iq2_k(wire: np.ndarray, in_features: int,
               chunk: int = PACK_CHUNK_ROWS) -> dict[str, np.ndarray]:
    """Relayout ``IQ2_K`` rows from ik wire bytes."""
    return _pack_chunked(_pack_iq2_k_rows, wire, in_features, chunk)


def pack_iq1_s_r4(wire: np.ndarray, in_features: int,
                  chunk: int = PACK_CHUNK_ROWS) -> dict[str, np.ndarray]:
    """Relayout ``IQ1_S_R4`` rows from ik wire bytes, whole groups only."""
    return _pack_chunked(_pack_iq1_s_r4_rows, wire, in_features, chunk,
                         group=WIRE_GROUP_ROWS["iq1_s_r4"])


_PACK_FNS = {"iq2_ks": pack_iq2_ks, "iq2_k": pack_iq2_k,
             "iq1_s_r4": pack_iq1_s_r4}


def pack(member: str, wire: np.ndarray, in_features: int,
         chunk: int = PACK_CHUNK_ROWS) -> dict[str, np.ndarray]:
    """Relayout ik wire rows of any member, routed or dense.

    The routed members dispatch through the table above; the dense packers
    are defined below it, so they resolve here instead.
    """
    check_any_member(member)
    fn = _PACK_FNS.get(member)
    if fn is not None:
        return fn(wire, in_features, chunk)
    if member == "iq4_ks":
        return _pack_chunked(_pack_iq4_ks_rows, wire, in_features, chunk)
    return _pack_chunked(
        lambda w, n: _pack_block_scale_rows(member, w, n), wire, in_features, chunk)


# ---------------------------------------------------------------------------
# relayout -> ik wire layout (the inverse, for round-trip gates)
# ---------------------------------------------------------------------------


def _codes_to_ik_qs(member: str, codes: np.ndarray, n: int) -> np.ndarray:
    """k-ordered codes back into ik's interleaved ``qs`` bytes.

    Each weight owns one two-bit field, so the scatter is a disjoint OR.
    Weights sharing a bit shift land in distinct bytes, which makes the
    scatter four vectorized writes rather than a loop over weights.
    """
    rows = codes.shape[0]
    byte_idx, shift = _ik_code_positions(member, n)
    out = np.zeros((rows, (n // QK_K) * 64), dtype=np.uint8)
    for s in np.unique(shift):
        sel = shift == s
        out[:, byte_idx[sel]] |= (codes[:, sel] << s).astype(np.uint8)
    return out


def _unpack_iq2_ks_rows(streams: dict[str, np.ndarray], in_features: int) -> np.ndarray:
    n = check_row_width(in_features)
    qs = streams["qs"]
    rows = qs.shape[0]
    nblocks = n // QK_K
    wire = np.empty((rows, ik_row_bytes("iq2_ks", n)), dtype=np.uint8)
    wire[:, :2] = np.ascontiguousarray(
        streams["dv"].astype(np.float16)).reshape(rows, 1).view(np.uint8)
    body = wire[:, 2:].reshape(rows, nblocks, IQ2_KS_BLOCK_BYTES)
    # ``extra`` is one uint16 per super-block: the alphabet-select bits in the
    # low byte and the high scale bits in the high byte, both in k order, so
    # each plane is one byte per super-block.
    body[:, :, 0] = streams["sex"].reshape(rows, nblocks)
    body[:, :, 1] = streams["sch"].reshape(rows, nblocks)
    # ik's scale nibbles are indexed by the k-order sub-block already, so the
    # relayout carries them unchanged.
    body[:, :, 2:6] = streams["scl"].reshape(rows, nblocks, 4)
    body[:, :, 6:] = _codes_to_ik_qs(
        "iq2_ks", _qs_to_codes(qs, n), n).reshape(rows, nblocks, 64)
    return wire


def _unpack_iq2_k_rows(streams: dict[str, np.ndarray], in_features: int) -> np.ndarray:
    n = check_row_width(in_features)
    qs = streams["qs"]
    rows = qs.shape[0]
    nblocks = n // QK_K
    wire = np.empty((rows, ik_row_bytes("iq2_k", n)), dtype=np.uint8)
    body = wire.reshape(rows, nblocks, IQ2_K_BLOCK_BYTES)
    body[:, :, 0:2] = np.ascontiguousarray(
        streams["dv"].astype(np.float16)).reshape(rows, nblocks, 1).view(np.uint8)
    body[:, :, 2:4] = streams["sex"].reshape(rows, nblocks, 2)
    body[:, :, 4:12] = streams["scl"].reshape(rows, nblocks, 8)
    body[:, :, 12:] = _codes_to_ik_qs(
        "iq2_k", _qs_to_codes(qs, n), n).reshape(rows, nblocks, 64)
    return wire


_UNPACK_FNS = {"iq2_ks": _unpack_iq2_ks_rows, "iq2_k": _unpack_iq2_k_rows,
               "iq1_s_r4": _unpack_iq1_s_r4_rows}


def unpack(member: str, streams: dict[str, np.ndarray], in_features: int,
           chunk: int = PACK_CHUNK_ROWS) -> np.ndarray:
    """Inverse of :func:`pack`: relayout streams back to ik wire rows.

    Exists so a consumer that rearranges stored bytes can prove the
    rearrangement lost nothing on the bytes it actually moved, rather than
    inferring it from the transform being correct in general.
    """
    check_any_member(member)
    fn = _UNPACK_FNS.get(member)
    if fn is None:
        if member == "iq4_ks":
            fn = _unpack_iq4_ks_rows
        else:
            def fn(s, n, _member=member):
                return _unpack_block_scale_rows(_member, s, n)
    group = WIRE_GROUP_ROWS[member]
    rows = streams["qs"].shape[0]
    if chunk <= 0 or chunk % group:
        raise IkqFormatError(
            f"unpack chunk {chunk} is not a positive multiple of the "
            f"{group}-row wire group")
    if rows <= chunk:
        return fn(streams, in_features)
    return np.concatenate(
        [fn({name: value[start:start + chunk] for name, value in streams.items()},
            in_features)
         for start in range(0, rows, chunk)],
        axis=0)


# ---------------------------------------------------------------------------
# Reference decode of the relayout
# ---------------------------------------------------------------------------


def _decode_planes_iq1_s_r4(streams: dict[str, np.ndarray],
                            in_features: int) -> dict[str, np.ndarray]:
    """The ``IQ1_S_R4`` decode planes, per relayout row.

    ``weights`` performs the CPU dequantizer's own float32 product chain in
    the same order: ``dl = d * float(2 * ls + 1)`` then
    ``y = dl * (float(grid_value) + shift)``.
    """
    n = check_row_width(in_features, "iq1_s_r4")
    qs = streams["qs"]
    rows = qs.shape[0]
    nblock = n // 32

    qs_bytes = np.ascontiguousarray(qs).view(np.uint8).reshape(
        rows, nblock, 4).astype(np.int64)
    qh = np.ascontiguousarray(streams["qh"]).astype(np.int64)
    hi = (qh[:, :, None] >> (3 * np.arange(4, dtype=np.int64))[None, None, :]) & 7
    indices = (qs_bytes | (hi << 8)).reshape(rows, n // 8)

    ls = (qh >> 12) & 7
    scale_index = 2 * ls + 1
    shift = np.where(qh & 0x8000, np.float32(-IQ1S_DELTA), np.float32(IQ1S_DELTA))
    d = streams["dv"].astype(np.float32).reshape(rows, 1)
    dl = d * scale_index.astype(np.float32)

    values = grid_values()[indices].reshape(rows, n)
    weights = (dl.repeat(32, axis=1).reshape(rows, n)
               * (values.astype(np.float32)
                  + shift.astype(np.float32).repeat(32, axis=1).reshape(rows, n)))
    return {
        "codes": indices,
        "values": values.astype(np.int64),
        "scale_index": scale_index,
        "sub_scale": dl,
        "shift": shift,
        "weights": weights,
    }


def decode_planes(member: str, streams: dict[str, np.ndarray],
                  in_features: int) -> dict[str, np.ndarray]:
    """Decode relayout rows into their integer, table, and float planes.

    Returned separately so the integer and table path can be checked for
    exact equality and only the closing float multiplies are checked against
    a tolerance. ``weights`` performs the CPU dequantizer's own float32
    product chain in the same order: for the 2-bit members
    ``dl = d * float(index)`` then ``y = dl * float(value)``; for
    ``iq1_s_r4`` see :func:`_decode_planes_iq1_s_r4` (its ``codes`` plane
    holds the 11-bit grid indices, one per 8 weights, and a ``shift`` plane
    joins the returned set).
    """
    check_any_member(member)
    if member == "iq1_s_r4":
        return _decode_planes_iq1_s_r4(streams, in_features)
    if member in DENSE_MEMBERS:
        return _dense_decode_planes(member, streams, in_features)
    n = check_row_width(in_features)
    qs = streams["qs"]
    rows = qs.shape[0]
    sub = SUB_WEIGHTS[member]
    nsub = n // sub

    codes = _qs_to_codes(qs, n).astype(np.int64)
    scl = streams["scl"]
    idx_lo = np.empty((rows, nsub), dtype=np.int64)
    idx_lo[:, 0::2] = scl & 0xF
    idx_lo[:, 1::2] = scl >> 4
    sex = np.unpackbits(streams["sex"], axis=1,
                        bitorder="little")[:, :nsub].astype(np.int64)

    if member == "iq2_ks":
        sch = np.unpackbits(streams["sch"], axis=1,
                            bitorder="little")[:, :nsub].astype(np.int64)
        scale_index = (idx_lo | (sch << 4)) - 16
        d = streams["dv"].astype(np.float32).reshape(rows, 1)
        dl = d * scale_index.astype(np.float32)
    else:
        scale_index = idx_lo - 8
        d = streams["dv"].astype(np.float32).repeat(QK_K // sub, axis=1)
        dl = d * scale_index.astype(np.float32)

    values = IQ2NL_VALUES.astype(np.int64)[
        (sex << 2).repeat(sub, axis=1).reshape(rows, n) + codes]
    weights = dl.repeat(sub, axis=1).reshape(rows, n) * values.astype(np.float32)
    return {
        "codes": codes,
        "values": values,
        "scale_index": scale_index,
        "sub_scale": dl,
        "weights": weights,
    }


def decode(member: str, streams: dict[str, np.ndarray],
           in_features: int) -> np.ndarray:
    """Reference float32 dequantization of relayout rows."""
    return decode_planes(member, streams, in_features)["weights"]


def decode_wire(member: str, wire: np.ndarray, in_features: int) -> np.ndarray:
    """Relayout ik wire rows and decode them, the whole build-time path."""
    return decode(member, pack(member, wire, in_features), in_features)


# ---------------------------------------------------------------------------
# Dense members: stream geometry
# ---------------------------------------------------------------------------
#
# A dense matrix is one ``[out_features, in_features]`` tensor, the
# degenerate single-expert case of the stacked relayout: every stream keeps
# the per-row layout and drops the expert axis. Only the code planes are
# permuted from ik's in-block placements to k-contiguous little-endian
# streams; the scale planes are already indexed by the k-order sub-block
# inside a super-block and concatenate unchanged. Byte counts are preserved
# plane for plane against ``ggml_row_size``.


def dense_component_shapes(member: str, out_features: int,
                           in_features: int) -> dict[str, tuple[int, ...]]:
    """Dense-matrix shapes of every relayout stream, in kernel input order."""
    check_dense_member(member)
    n = check_row_width(in_features)
    o = int(out_features)
    if o <= 0:
        raise IkqFormatError(f"out_features {out_features} is not positive")
    if member == "iq4_ks":
        return {"qs": (o, n // 8), "scl": (o, n // 32), "dv": (o,)}
    if member == "iq4_k":
        return {"qs": (o, n // 8), "scl": (o, n // 32), "sch": (o, n // 64),
                "sex": (o, n // 128), "dv": (o, n // 256)}
    if member == "iq5_k":
        return {"qs": (o, n // 8), "qh": (o, n // 32), "scl": (o, n // 32),
                "sch": (o, n // 64), "sex": (o, n // 128), "dv": (o, n // 256)}
    return {"qs": (o, n // 8), "qh": (o, n // 16), "scl": (o, n // 16),
            "sex": (o, n // 128), "dv": (o, n // 256)}


def dense_component_dtypes(member: str) -> dict[str, np.dtype]:
    """NumPy dtype of every dense relayout stream.

    ``dv`` is float32 for ``iq4_ks`` (the member's row scale is fp32 on the
    wire) and fp16 for the block-scale members. ``scl`` for ``iq6_k`` stores
    the direct signed int8 scale as its byte.
    """
    check_dense_member(member)
    names = tuple(dense_component_shapes(member, 1, QK_K))
    dv = np.dtype(np.float32) if member == "iq4_ks" else np.dtype(np.float16)
    return {name: (dv if name == "dv" else
                   np.dtype(np.uint32) if name in ("qs", "qh") else
                   np.dtype(np.uint8))
            for name in names}


# ---------------------------------------------------------------------------
# Dense members: ik wire layout <-> relayout
# ---------------------------------------------------------------------------


def _dense_code_positions(member: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Byte index and nibble shift of every weight's 4-bit code, ik layout.

    ``IQ4_KS``/``IQ4_K``: weight ``w`` of a super-block sits in ``qs`` byte
    ``16*(w/32) + w%16`` at shift ``4*((w%32)/16)``. ``IQ5_K``/``IQ6_K``:
    byte ``32*(w/64) + w%32`` at shift ``4*((w%64)/32)``.
    """
    w = np.arange(QK_K, dtype=np.int64)
    if member in ("iq4_ks", "iq4_k"):
        byte = 16 * (w // 32) + w % 16
        shift = 4 * ((w % 32) // 16)
    else:
        byte = 32 * (w // 64) + w % 32
        shift = 4 * ((w % 64) // 32)
    nblocks = n // QK_K
    block = np.arange(nblocks, dtype=np.int64)[:, None]
    return ((block * 128 + byte[None, :]).reshape(-1), np.tile(shift, nblocks))


def _dense_qh_positions(member: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Byte index and bit shift of every weight's high code bits, ik layout.

    ``IQ5_K``: one bit, byte ``w%32``, bit ``w/32`` (``qh`` is 32 bytes per
    super-block). ``IQ6_K``: two bits, byte ``32*(w/128) + w%32``, shift
    ``2*((w/32)%4)`` (``qh`` is 64 bytes per super-block).
    """
    w = np.arange(QK_K, dtype=np.int64)
    if member == "iq5_k":
        byte, shift, qh_bytes = w % 32, w // 32, 32
    else:
        byte, shift, qh_bytes = 32 * (w // 128) + w % 32, 2 * ((w // 32) % 4), 64
    nblocks = n // QK_K
    block = np.arange(nblocks, dtype=np.int64)[:, None]
    return ((block * qh_bytes + byte[None, :]).reshape(-1), np.tile(shift, nblocks))


def _codes4_to_qs(codes: np.ndarray) -> np.ndarray:
    """4-bit codes in k order to little-endian uint32 words."""
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return np.ascontiguousarray(packed).view(np.uint32)


def _qs_to_codes4(qs: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`_codes4_to_qs`."""
    rows = qs.shape[0]
    packed = np.ascontiguousarray(qs).view(np.uint8).reshape(rows, n // 2)
    codes = np.empty((rows, n), dtype=np.uint8)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    return codes


def _bits1_to_words(bits: np.ndarray) -> np.ndarray:
    """1-bit fields in k order to little-endian uint32 words."""
    packed = np.packbits(bits.astype(np.uint8), axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view(np.uint32)


def _words_to_bits1(words: np.ndarray, n: int) -> np.ndarray:
    rows = words.shape[0]
    return np.unpackbits(np.ascontiguousarray(words).view(np.uint8).reshape(rows, -1),
                         axis=1, bitorder="little")[:, :n]


def _fields2_to_words(fields: np.ndarray) -> np.ndarray:
    """2-bit fields in k order to little-endian uint32 words."""
    packed = (fields[:, 0::4] | (fields[:, 1::4] << 2)
              | (fields[:, 2::4] << 4) | (fields[:, 3::4] << 6)).astype(np.uint8)
    return np.ascontiguousarray(packed).view(np.uint32)


def _words_to_fields2(words: np.ndarray, n: int) -> np.ndarray:
    rows = words.shape[0]
    packed = np.ascontiguousarray(words).view(np.uint8).reshape(rows, n // 4)
    fields = np.empty((rows, n), dtype=np.uint8)
    fields[:, 0::4] = packed & 3
    fields[:, 1::4] = (packed >> 2) & 3
    fields[:, 2::4] = (packed >> 4) & 3
    fields[:, 3::4] = (packed >> 6) & 3
    return fields


def _pack_iq4_ks_rows(wire: np.ndarray, in_features: int) -> dict[str, np.ndarray]:
    """Relayout one chunk of ``IQ4_KS`` rows from ik wire bytes."""
    n = check_row_width(in_features)
    rows = wire.shape[0]
    expect = ik_row_bytes("iq4_ks", n)
    if wire.shape[1] != expect:
        raise IkqFormatError(
            f"iq4_ks row of {n} weights is {expect} wire bytes, got {wire.shape[1]}")
    nblocks = n // QK_K
    dv = wire[:, :4].copy().view(np.float32).reshape(rows)
    body = wire[:, 4:].reshape(rows, nblocks, IQ4_KS_BLOCK_BYTES)
    # The per-32 scale byte keeps its wire form: bit 0 is the alphabet
    # select, bits 1..7 the odd-scale payload. Splitting it would spend a
    # byte per plane; the kernel separates the two with one mask each.
    scl = np.ascontiguousarray(body[:, :, 0:8]).reshape(rows, n // 32)
    qs_bytes = np.ascontiguousarray(body[:, :, 8:]).reshape(rows, -1)
    byte_idx, shift = _dense_code_positions("iq4_ks", n)
    codes = (qs_bytes[:, byte_idx] >> shift[None, :]) & 0xF
    return {"qs": _codes4_to_qs(codes.astype(np.uint8)), "scl": scl, "dv": dv}


def _pack_block_scale_rows(member: str, wire: np.ndarray,
                           in_features: int) -> dict[str, np.ndarray]:
    """Relayout one chunk of ``IQ4_K``/``IQ5_K``/``IQ6_K`` rows.

    The three members share the block header order (``d``, ``extra``, then
    scale planes, then code planes); only the plane widths differ. ``extra``
    and the scale planes are already k-ordered per super-block, so they
    concatenate unchanged; the code planes are gathered per weight.
    """
    n = check_row_width(in_features)
    rows = wire.shape[0]
    expect = ik_row_bytes(member, n)
    if wire.shape[1] != expect:
        raise IkqFormatError(
            f"{member} row of {n} weights is {expect} wire bytes, got {wire.shape[1]}")
    nblocks = n // QK_K
    body = wire.reshape(rows, nblocks, _BLOCK_BYTES[member])

    dv = np.ascontiguousarray(body[:, :, 0:2]).reshape(rows, -1).copy()
    dv = dv.view(np.float16).reshape(rows, nblocks)
    sex = np.ascontiguousarray(body[:, :, 2:4]).reshape(rows, n // 128)

    if member == "iq6_k":
        scl = np.ascontiguousarray(body[:, :, 4:20]).reshape(rows, n // 16)
        qs_bytes = np.ascontiguousarray(body[:, :, 20:148]).reshape(rows, -1)
        qh_bytes = np.ascontiguousarray(body[:, :, 148:212]).reshape(rows, -1)
    else:
        sch = np.ascontiguousarray(body[:, :, 4:8]).reshape(rows, n // 64)
        scl = np.ascontiguousarray(body[:, :, 8:16]).reshape(rows, n // 32)
        qs_bytes = np.ascontiguousarray(body[:, :, 16:144]).reshape(rows, -1)
        if member == "iq5_k":
            qh_bytes = np.ascontiguousarray(body[:, :, 144:176]).reshape(rows, -1)

    byte_idx, shift = _dense_code_positions(member, n)
    codes = ((qs_bytes[:, byte_idx] >> shift[None, :]) & 0xF).astype(np.uint8)
    out = {"qs": _codes4_to_qs(codes)}
    if member == "iq5_k":
        hb, hs = _dense_qh_positions(member, n)
        out["qh"] = _bits1_to_words((qh_bytes[:, hb] >> hs[None, :]) & 1)
    elif member == "iq6_k":
        hb, hs = _dense_qh_positions(member, n)
        out["qh"] = _fields2_to_words(
            ((qh_bytes[:, hb] >> hs[None, :]) & 3).astype(np.uint8))
    out["scl"] = scl
    if member != "iq6_k":
        out["sch"] = sch
    out["sex"] = sex
    out["dv"] = dv
    return out


def _unpack_iq4_ks_rows(streams: dict[str, np.ndarray],
                        in_features: int) -> np.ndarray:
    n = check_row_width(in_features)
    qs = streams["qs"]
    rows = qs.shape[0]
    nblocks = n // QK_K
    wire = np.empty((rows, ik_row_bytes("iq4_ks", n)), dtype=np.uint8)
    wire[:, :4] = np.ascontiguousarray(
        streams["dv"].astype(np.float32)).reshape(rows, 1).view(np.uint8)
    body = wire[:, 4:].reshape(rows, nblocks, IQ4_KS_BLOCK_BYTES)
    body[:, :, 0:8] = streams["scl"].reshape(rows, nblocks, 8)
    body[:, :, 8:] = _codes_to_dense_qs(
        "iq4_ks", _qs_to_codes4(qs, n), n).reshape(rows, nblocks, 128)
    return wire


def _unpack_block_scale_rows(member: str, streams: dict[str, np.ndarray],
                             in_features: int) -> np.ndarray:
    n = check_row_width(in_features)
    qs = streams["qs"]
    rows = qs.shape[0]
    nblocks = n // QK_K
    wire = np.empty((rows, ik_row_bytes(member, n)), dtype=np.uint8)
    body = wire.reshape(rows, nblocks, _BLOCK_BYTES[member])
    body[:, :, 0:2] = np.ascontiguousarray(
        streams["dv"].astype(np.float16)).reshape(rows, nblocks, 1).view(np.uint8)
    body[:, :, 2:4] = streams["sex"].reshape(rows, nblocks, 2)
    if member == "iq6_k":
        body[:, :, 4:20] = streams["scl"].reshape(rows, nblocks, 16)
        body[:, :, 20:148] = _codes_to_dense_qs(
            member, _qs_to_codes4(qs, n), n).reshape(rows, nblocks, 128)
        body[:, :, 148:212] = _fields_to_dense_qh(
            member, _words_to_fields2(streams["qh"], n), n).reshape(rows, nblocks, 64)
    else:
        body[:, :, 4:8] = streams["sch"].reshape(rows, nblocks, 4)
        body[:, :, 8:16] = streams["scl"].reshape(rows, nblocks, 8)
        body[:, :, 16:144] = _codes_to_dense_qs(
            member, _qs_to_codes4(qs, n), n).reshape(rows, nblocks, 128)
        if member == "iq5_k":
            body[:, :, 144:176] = _fields_to_dense_qh(
                member, _words_to_bits1(streams["qh"], n), n).reshape(rows, nblocks, 32)
    return wire


def _codes_to_dense_qs(member: str, codes: np.ndarray, n: int) -> np.ndarray:
    """k-ordered 4-bit codes back into ik's in-block ``qs`` bytes."""
    rows = codes.shape[0]
    byte_idx, shift = _dense_code_positions(member, n)
    out = np.zeros((rows, (n // QK_K) * 128), dtype=np.uint8)
    for s in np.unique(shift):
        sel = shift == s
        out[:, byte_idx[sel]] |= (codes[:, sel] << s).astype(np.uint8)
    return out


def _fields_to_dense_qh(member: str, fields: np.ndarray, n: int) -> np.ndarray:
    """k-ordered high code bits back into ik's in-block ``qh`` bytes."""
    rows = fields.shape[0]
    byte_idx, shift = _dense_qh_positions(member, n)
    qh_bytes = 32 if member == "iq5_k" else 64
    out = np.zeros((rows, (n // QK_K) * qh_bytes), dtype=np.uint8)
    for s in np.unique(shift):
        sel = shift == s
        out[:, byte_idx[sel]] |= (fields[:, sel] << s).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Dense members: reference decode of the relayout
# ---------------------------------------------------------------------------


def _dense_decode_planes(member: str, streams: dict[str, np.ndarray],
                         in_features: int) -> dict[str, np.ndarray]:
    """Decode dense relayout rows into integer, table, and float planes.

    ``weights`` performs the CPU dequantizer's float32 product chain in the
    same order: ``dl = d * float(scale_payload)`` then ``y = dl * value``.
    The ``IQ6_K`` value plane reads :data:`IQ6K_TABLE`, the bit-exact float32
    outputs of ik's cubic reconstruction; the other members read their
    integer tables.
    """
    check_dense_member(member)
    n = check_row_width(in_features)
    qs = streams["qs"]
    rows = qs.shape[0]
    sub = SUB_WEIGHTS[member]
    nsub = n // sub

    codes = _qs_to_codes4(qs, n).astype(np.int64)
    if member == "iq5_k":
        codes |= _words_to_bits1(streams["qh"], n).astype(np.int64) << 4
    elif member == "iq6_k":
        codes |= _words_to_fields2(streams["qh"], n).astype(np.int64) << 4

    if member == "iq4_ks":
        scl = streams["scl"].astype(np.int64)
        scale_index = (scl & 254) - 127
        sel = scl & 1
        d = streams["dv"].astype(np.float32).reshape(rows, 1)
        dl = d * scale_index.astype(np.float32)
    else:
        if member == "iq6_k":
            scale_index = streams["scl"].view(np.int8).astype(np.int64)
        else:
            scl = streams["scl"]
            idx_lo = np.empty((rows, nsub), dtype=np.int64)
            idx_lo[:, 0::2] = scl & 0xF
            idx_lo[:, 1::2] = scl >> 4
            sch = streams["sch"].astype(np.int64)
            hi = np.empty((rows, nsub), dtype=np.int64)
            hi[:, 0::4] = sch & 3
            hi[:, 1::4] = (sch >> 2) & 3
            hi[:, 2::4] = (sch >> 4) & 3
            hi[:, 3::4] = (sch >> 6) & 3
            scale_index = (idx_lo | (hi << 4)) - 32
        sel = np.unpackbits(streams["sex"], axis=1,
                            bitorder="little")[:, :nsub].astype(np.int64)
        d = streams["dv"].astype(np.float32).repeat(QK_K // sub, axis=1)
        dl = d * scale_index.astype(np.float32)

    shift_bits = {"iq4_ks": 4, "iq4_k": 4, "iq5_k": 5, "iq6_k": 6}[member]
    table_index = (sel << shift_bits).repeat(sub, axis=1).reshape(rows, n) + codes
    if member == "iq6_k":
        values = IQ6K_TABLE[table_index]
        weights = dl.repeat(sub, axis=1).reshape(rows, n) * values
    else:
        table = IQ5NL_VALUES if member == "iq5_k" else IQ4K_VALUES
        values = table.astype(np.int64)[table_index]
        weights = (dl.repeat(sub, axis=1).reshape(rows, n)
                   * values.astype(np.float32))
    return {
        "codes": codes,
        "values": values,
        "scale_index": scale_index,
        "sub_scale": dl,
        "weights": weights,
    }


__all__ = [
    "ALPHABET_SIZES",
    "BLOCK_WEIGHTS",
    "DENSE_MEMBERS",
    "IQ1S_DELTA",
    "IQ1_S_R4_BLOCK_BYTES",
    "IQ1_S_R4_ROW_META_BYTES",
    "IQ2NL_VALUES",
    "IQ2_KS_BLOCK_BYTES",
    "IQ2_KS_ROW_META_BYTES",
    "IQ2_K_BLOCK_BYTES",
    "IQ4K_VALUES",
    "IQ4_KS_BLOCK_BYTES",
    "IQ4_KS_ROW_META_BYTES",
    "IQ4_K_BLOCK_BYTES",
    "IQ5NL_VALUES",
    "IQ5_K_BLOCK_BYTES",
    "IQ6K_TABLE",
    "IQ6NL_VALUES",
    "IQ6_K_BLOCK_BYTES",
    "MEMBERS",
    "QK_K",
    "SUB_WEIGHTS",
    "WIRE_GROUP_ROWS",
    "IkqFormatError",
    "bits_per_weight",
    "check_any_member",
    "check_dense_member",
    "check_member",
    "check_row_width",
    "check_wire_rows",
    "component_dtypes",
    "component_shapes",
    "decode",
    "decode_planes",
    "decode_wire",
    "dense_component_dtypes",
    "dense_component_shapes",
    "ik_row_bytes",
    "pack",
    "pack_iq1_s_r4",
    "pack_iq2_k",
    "pack_iq2_ks",
    "relayout_row_bytes",
    "unpack",
]
