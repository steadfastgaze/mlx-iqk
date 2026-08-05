"""Independent packers into the ik block layouts, for the bit-exactness gates.

These build ``block_iq2_k``, ``block_iq2_ks``, and ``block_iq1_s_r4`` bytes
straight from field values, using the struct layout as written in
``ggml-common.h`` and the index arithmetic as written in the encoders. They exist so a test can drive the
decode over its whole value space instead of over whatever a quantizer
happens to emit, and they are deliberately not written in terms of
:mod:`mlx_ikq.format`: a packer that shared code with the module under test
would agree with it for the wrong reason.
"""

from __future__ import annotations

import zlib

import numpy as np

QK_K = 256


def seed_for(*parts) -> int:
    """A stable seed from any label.

    Python's ``hash`` of a string is salted per process, so a seed derived
    from one changes between runs and a sampled tolerance stops being a
    reproducible claim.
    """
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31)


def _finite_half(rng: np.random.Generator, size) -> np.ndarray:
    """Random finite fp16 values, sign and exponent swept, no NaN or infinity."""
    bits = rng.integers(0, 1 << 16, size=size, dtype=np.uint16)
    mag = bits & 0x7FFF
    bad = mag >= 0x7C00
    bits = np.where(bad, bits & 0x3FFF, bits).astype(np.uint16)
    return bits.view(np.float16)


def pack_iq2_k(codes: np.ndarray, ls: np.ndarray, ebit: np.ndarray,
               d: np.ndarray) -> np.ndarray:
    """``[nb, 76]`` bytes: d, extra, scales[8], qs[64].

    ``codes`` is ``[nb, 256]`` in 0..3, ``ls`` is ``[nb, 16]`` in 0..15 (the
    offset-8 scale index), ``ebit`` is ``[nb, 16]`` in 0..1, ``d`` is
    ``[nb]`` fp16.
    """
    nb = codes.shape[0]
    out = np.zeros((nb, 76), dtype=np.uint8)
    out[:, 0:2] = np.ascontiguousarray(d.astype(np.float16)).view(np.uint8).reshape(nb, 2)
    for b in range(nb):
        extra = 0
        for ib in range(16):
            out[b, 4 + ib // 2] |= np.uint8(int(ls[b, ib]) << (4 * (ib % 2)))
            if ebit[b, ib]:
                extra |= 1 << ib
        out[b, 2] = extra & 0xFF
        out[b, 3] = (extra >> 8) & 0xFF
        for w in range(QK_K):
            ib32, rem = divmod(w, 32)
            half, j = divmod(rem, 16)
            byte = 32 * (ib32 // 4) + 16 * half + j
            out[b, 12 + byte] |= np.uint8(int(codes[b, w]) << (2 * (ib32 % 4)))
    return out


def pack_iq2_ks(codes: np.ndarray, ls: np.ndarray, ebit: np.ndarray,
                drow: np.float16) -> np.ndarray:
    """``[2 + 70 * nb]`` bytes of one row: the fp16 row scale then the blocks.

    ``codes`` is ``[nb, 256]`` in 0..3, ``ls`` is ``[nb, 8]`` in 0..31 (the
    offset-16 scale index), ``ebit`` is ``[nb, 8]`` in 0..1.
    """
    nb = codes.shape[0]
    out = np.zeros(2 + 70 * nb, dtype=np.uint8)
    out[0:2] = np.array([drow], dtype=np.float16).view(np.uint8)
    for b in range(nb):
        base = 2 + 70 * b
        extra = 0
        for ib in range(8):
            value = int(ls[b, ib])
            out[base + 2 + ib // 2] |= np.uint8((value & 0xF) << (4 * (ib % 2)))
            extra |= (value >> 4) << (8 + ib)
            if ebit[b, ib]:
                extra |= 1 << ib
        out[base + 0] = extra & 0xFF
        out[base + 1] = (extra >> 8) & 0xFF
        for w in range(QK_K):
            ib64, rem = divmod(w, 64)
            half, j = divmod(rem, 32)
            byte = 32 * (ib64 // 2) + j
            shift = 4 * (ib64 % 2) + 2 * half
            out[base + 6 + byte] |= np.uint8(int(codes[b, w]) << shift)
    return out


def exhaustive_iq2_k(seed: int = 0) -> np.ndarray:
    """Every (scale index, alphabet, code) triple at every weight position.

    128 super-blocks: the 16 scale indices crossed with the 2 alphabets, each
    at 4 code rotations so every one of the 256 positions carries every one of
    the 4 codes. One row per block, since the member's scale is per block.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    w = np.arange(QK_K)
    for s in range(16):
        for e in range(2):
            for r in range(4):
                codes = ((w + r) % 4)[None, :]
                ls = np.full((1, 16), s, dtype=np.int64)
                ebit = np.full((1, 16), e, dtype=np.int64)
                d = _finite_half(rng, 1)
                blocks.append(pack_iq2_k(codes, ls, ebit, d))
    return np.concatenate(blocks, axis=0)


def exhaustive_iq2_ks(seed: int = 0) -> np.ndarray:
    """Every (scale index, alphabet, code) triple at every weight position.

    256 rows of one super-block: the 32 scale indices crossed with the 2
    alphabets, each at 4 code rotations.
    """
    rng = np.random.default_rng(seed)
    rows = []
    w = np.arange(QK_K)
    for s in range(32):
        for e in range(2):
            for r in range(4):
                codes = ((w + r) % 4)[None, :]
                ls = np.full((1, 8), s, dtype=np.int64)
                ebit = np.full((1, 8), e, dtype=np.int64)
                rows.append(pack_iq2_ks(codes, ls, ebit, _finite_half(rng, 1)[0]))
    return np.stack(rows, axis=0)


def pack_iq1_s_r4(indices: np.ndarray, ls: np.ndarray, sneg: np.ndarray,
                  d: np.ndarray) -> np.ndarray:
    """``[rows, 2 + 6 * nb]`` bytes of ``rows`` rows, ``rows`` a multiple of 4.

    ``indices`` is ``[rows, 8 * nb]`` in 0..2047 (one 11-bit grid index per
    8 weights), ``ls`` is ``[rows, nb]`` in 0..7 (the block scale), ``sneg``
    is ``[rows, nb]`` in 0..1 (1 selects the negative shift), ``d`` is
    ``[rows]`` fp16. The bytes come out in the four-row-group wire order:
    each group of four rows is its 4 x f16 scale prefix followed by 24-byte
    blocks interleaving the four rows, and the flat result is that stream
    reshaped at the per-row byte size.
    """
    rows, nsub = indices.shape
    assert rows % 4 == 0
    nb = nsub // 4
    row_bytes = 2 + 6 * nb
    groups = rows // 4
    out = np.zeros((groups, 4 * row_bytes), dtype=np.uint8)
    out[:, :8] = np.ascontiguousarray(
        d.astype(np.float16)).reshape(groups, 4).view(np.uint8)
    for g in range(groups):
        body = out[g, 8:].reshape(nb, 24)
        for k in range(4):
            r = 4 * g + k
            for ib in range(nb):
                h = 0
                for i in range(4):
                    ind = int(indices[r, 4 * ib + i])
                    body[ib, 4 * i + k] = ind & 255
                    h |= (ind >> 8) << (3 * i)
                if sneg[r, ib]:
                    h |= 0x8000
                h |= int(ls[r, ib]) << 12
                body[ib, 16 + 2 * k] = h & 0xFF
                body[ib, 17 + 2 * k] = (h >> 8) & 0xFF
    return out.reshape(rows, row_bytes)


def exhaustive_iq1_s_r4(seed: int = 0) -> np.ndarray:
    """Every (grid index, block position, scale, shift) combination.

    16 scale-shift combos; inside each, the 2048 indices tile 64 rows of
    256 weights and four rotations move every index through every one of
    the four in-block positions (a rotation by ``rot`` moves the index at
    slot ``c`` to slot ``c + rot``, changing its ``qh`` bit field). 4096
    rows in total, group scales drawn from the finite fp16 sweep.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(2048, dtype=np.int64).reshape(64, 32)
    rows = []
    for s in range(8):
        for neg in range(2):
            for rot in range(4):
                block = np.roll(idx, rot, axis=1)
                ls = np.full((64, 8), s, dtype=np.int64)
                sneg = np.full((64, 8), neg, dtype=np.int64)
                rows.append(pack_iq1_s_r4(block, ls, sneg, _finite_half(rng, 64)))
    return np.concatenate(rows, axis=0)


def pack_iq4_ks(codes: np.ndarray, scale_bytes: np.ndarray,
                drow: np.float32) -> np.ndarray:
    """``[4 + 136 * nb]`` bytes of one row: the fp32 row scale, then blocks.

    ``codes`` is ``[nb, 256]`` in 0..15, ``scale_bytes`` is ``[nb, 8]`` full
    scale bytes (bit 0 the alphabet select, bits 1..7 the scale payload).
    """
    nb = codes.shape[0]
    out = np.zeros(4 + 136 * nb, dtype=np.uint8)
    out[0:4] = np.array([drow], dtype=np.float32).view(np.uint8)
    for b in range(nb):
        base = 4 + 136 * b
        for ib in range(8):
            out[base + ib] = scale_bytes[b, ib]
        for i in range(8):
            for j in range(16):
                out[base + 8 + 16 * i + j] = (int(codes[b, 32 * i + j])
                                              | (int(codes[b, 32 * i + 16 + j]) << 4))
    return out


def pack_iq4_k(codes: np.ndarray, uls: np.ndarray, ebit: np.ndarray,
               d: np.ndarray) -> np.ndarray:
    """``[nb, 144]`` bytes: d, extra, scales_h[4], scales_l[8], qs[128].

    ``codes`` is ``[nb, 256]`` in 0..15, ``uls`` is ``[nb, 16]`` in 0..63
    (the offset-32 scale index), ``ebit`` is ``[nb, 16]`` in 0..1, ``d`` is
    ``[nb]`` fp16.
    """
    nb = codes.shape[0]
    out = np.zeros((nb, 144), dtype=np.uint8)
    out[:, 0:2] = np.ascontiguousarray(d.astype(np.float16)).view(np.uint8).reshape(nb, 2)
    for b in range(nb):
        extra = 0
        for ib in range(16):
            value = int(uls[b, ib])
            out[b, 8 + ib // 2] |= np.uint8((value & 0xF) << (4 * (ib % 2)))
            out[b, 4 + ib // 4] |= np.uint8((value >> 4) << (2 * (ib % 4)))
            if ebit[b, ib]:
                extra |= 1 << ib
        out[b, 2] = extra & 0xFF
        out[b, 3] = (extra >> 8) & 0xFF
        for i in range(8):
            for j in range(16):
                out[b, 16 + 16 * i + j] = (int(codes[b, 32 * i + j])
                                           | (int(codes[b, 32 * i + 16 + j]) << 4))
    return out


def pack_iq5_k(codes: np.ndarray, uls: np.ndarray, ebit: np.ndarray,
               d: np.ndarray) -> np.ndarray:
    """``[nb, 176]`` bytes: d, extra, scales_h[4], scales_l[8], qs[128], qh[32].

    ``codes`` is ``[nb, 256]`` in 0..31, ``uls`` is ``[nb, 16]`` in 0..63.
    """
    nb = codes.shape[0]
    out = np.zeros((nb, 176), dtype=np.uint8)
    out[:, 0:2] = np.ascontiguousarray(d.astype(np.float16)).view(np.uint8).reshape(nb, 2)
    for b in range(nb):
        extra = 0
        for ib in range(16):
            value = int(uls[b, ib])
            out[b, 8 + ib // 2] |= np.uint8((value & 0xF) << (4 * (ib % 2)))
            out[b, 4 + ib // 4] |= np.uint8((value >> 4) << (2 * (ib % 4)))
            if ebit[b, ib]:
                extra |= 1 << ib
            ib32, offset = ib // 2, 16 * (ib % 2)
            for j in range(16):
                ibest = int(codes[b, 16 * ib + j])
                out[b, 16 + 32 * (ib32 // 2) + offset + j] |= np.uint8(
                    (ibest & 0xF) << (4 * (ib32 % 2)))
                out[b, 144 + 32 * (ib32 // 8) + offset + j] |= np.uint8(
                    (ibest >> 4) << (ib32 % 8))
        out[b, 2] = extra & 0xFF
        out[b, 3] = (extra >> 8) & 0xFF
    return out


def pack_iq6_k(codes: np.ndarray, scale_bytes: np.ndarray, ebit: np.ndarray,
               d: np.ndarray) -> np.ndarray:
    """``[nb, 212]`` bytes: d, extra, scales[16], qs[128], qh[64].

    ``codes`` is ``[nb, 256]`` in 0..63, ``scale_bytes`` is ``[nb, 16]`` raw
    int8 scale bytes in 0..255.
    """
    nb = codes.shape[0]
    out = np.zeros((nb, 212), dtype=np.uint8)
    out[:, 0:2] = np.ascontiguousarray(d.astype(np.float16)).view(np.uint8).reshape(nb, 2)
    for b in range(nb):
        extra = 0
        for ib in range(16):
            out[b, 4 + ib] = scale_bytes[b, ib]
            if ebit[b, ib]:
                extra |= 1 << ib
            ib32, offset = ib // 2, 16 * (ib % 2)
            for j in range(16):
                ibest = int(codes[b, 16 * ib + j])
                out[b, 20 + 32 * (ib32 // 2) + offset + j] |= np.uint8(
                    (ibest & 0xF) << (4 * (ib32 % 2)))
                out[b, 148 + 32 * (ib32 // 4) + offset + j] |= np.uint8(
                    (ibest >> 4) << (2 * (ib32 % 4)))
        out[b, 2] = extra & 0xFF
        out[b, 3] = (extra >> 8) & 0xFF
    return out


def exhaustive_iq4_ks(seed: int = 0) -> np.ndarray:
    """Every (scale byte, code) pair, each pair at four code rotations."""
    rng = np.random.default_rng(seed)
    rows = []
    w = np.arange(QK_K)
    for sb in range(256):
        for r in range(4):
            codes = ((w + r) % 16)[None, :]
            scale_bytes = np.full((1, 8), sb, dtype=np.int64)
            rows.append(pack_iq4_ks(codes, scale_bytes,
                                    np.float32(_finite_half(rng, 1)[0])))
    return np.stack(rows, axis=0)


def exhaustive_iq4_k(seed: int = 0) -> np.ndarray:
    """Every (scale index, alphabet, code) triple, four code rotations."""
    rng = np.random.default_rng(seed)
    rows = []
    w = np.arange(QK_K)
    for s in range(64):
        for e in range(2):
            for r in range(4):
                codes = ((w + r) % 16)[None, :]
                uls = np.full((1, 16), s, dtype=np.int64)
                ebit = np.full((1, 16), e, dtype=np.int64)
                rows.append(pack_iq4_k(codes, uls, ebit, _finite_half(rng, 1)))
    return np.concatenate(rows, axis=0)


def exhaustive_iq5_k(seed: int = 0) -> np.ndarray:
    """Every (scale index, alphabet, code) triple, four code rotations."""
    rng = np.random.default_rng(seed)
    rows = []
    w = np.arange(QK_K)
    for s in range(64):
        for e in range(2):
            for r in range(4):
                codes = ((w + r) % 32)[None, :]
                uls = np.full((1, 16), s, dtype=np.int64)
                ebit = np.full((1, 16), e, dtype=np.int64)
                rows.append(pack_iq5_k(codes, uls, ebit, _finite_half(rng, 1)))
    return np.concatenate(rows, axis=0)


def exhaustive_iq6_k(seed: int = 0) -> np.ndarray:
    """Every (scale byte, alphabet, code) triple, four code rotations.

    The scale byte sweeps all 256 values: the wire carries a raw int8 and
    the decode must handle every bit pattern, including -128, which the
    encoder itself never emits (it clamps to [-127, 127]).
    """
    rng = np.random.default_rng(seed)
    rows = []
    w = np.arange(QK_K)
    for sb in range(256):
        for e in range(2):
            for r in range(4):
                codes = ((w + r) % 64)[None, :]
                scale_bytes = np.full((1, 16), sb, dtype=np.int64)
                ebit = np.full((1, 16), e, dtype=np.int64)
                rows.append(pack_iq6_k(codes, scale_bytes, ebit, _finite_half(rng, 1)))
    return np.concatenate(rows, axis=0)


def _scales(rng: np.random.Generator, size, mode: str) -> np.ndarray:
    """fp16 super-block or row scales for a random wire row.

    ``sweep`` draws the whole finite fp16 space, which is what the CPU
    bit-exactness gate wants: the reference and ik agree over it or they do
    not. ``serving`` draws the magnitudes a converted expert row carries, so a
    reconstructed weight stays inside fp16 range and a GPU comparison measures
    arithmetic rather than saturation.
    """
    if mode == "sweep":
        return _finite_half(rng, size)
    if mode != "serving":
        raise ValueError(f"unknown scale mode {mode!r}")
    mag = rng.random(size) * 8e-4 + 2e-5
    sign = np.where(rng.random(size) < 0.5, -1.0, 1.0)
    return (mag * sign).astype(np.float16)


def random_wire(member: str, rows: int, in_features: int, seed: int,
                scales: str = "sweep") -> np.ndarray:
    """Uniformly random wire rows.

    Every field of both members is dense over its byte space: any code, any
    scale index, any alphabet bit is legal and the decode is branch-free over
    all of them, so random bytes are valid wire content and sweep the space
    the enumeration does not reach in combination. Only the fp16 scale is
    drawn rather than taken from the byte stream, so a caller can choose
    between the full fp16 sweep and serving magnitudes.
    """
    rng = np.random.default_rng(seed)
    if member == "iq1_s_r4":
        assert rows % 4 == 0, "iq1_s_r4 wire rows come in groups of four"
        row_bytes = 2 + 6 * (in_features // 32)
        out = rng.integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)
        groups = out.reshape(rows // 4, 4 * row_bytes)
        groups[:, :8] = _scales(rng, (rows // 4, 4), scales).view(
            np.uint8).reshape(rows // 4, 8)
        return out
    nb = in_features // QK_K
    block_scale_bytes = {"iq2_k": 76, "iq4_k": 144, "iq5_k": 176, "iq6_k": 212}
    if member in block_scale_bytes:
        width = block_scale_bytes[member]
        out = rng.integers(0, 256, size=(rows, width * nb), dtype=np.uint8)
        out = out.reshape(rows, nb, width)
        out[:, :, 0:2] = _scales(rng, (rows, nb), scales).view(np.uint8).reshape(
            rows, nb, 2)
        return out.reshape(rows, width * nb)
    if member == "iq4_ks":
        # The row scale is fp32 on the wire. It is drawn from the finite
        # fp16 value set promoted to fp32: sign and exponent breadth for the
        # bit-identity gates while every reconstruction (|d| * 127 * 127 at
        # most) stays finite in float32, so no comparison depends on how a
        # host materializes non-finite products.
        out = rng.integers(0, 256, size=(rows, 4 + 136 * nb), dtype=np.uint8)
        out[:, 0:4] = _scales(rng, rows, scales).astype(
            np.float32).view(np.uint8).reshape(rows, 4)
        return out
    out = rng.integers(0, 256, size=(rows, 2 + 70 * nb), dtype=np.uint8)
    out[:, 0:2] = _scales(rng, rows, scales).view(np.uint8).reshape(rows, 2)
    return out
