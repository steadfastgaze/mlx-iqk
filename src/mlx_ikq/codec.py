"""Encode shim and CPU reference decode over the vendored ik codecs.

Three shared libraries built from ``vendor/ik_llama`` carry ik_llama's own
quantizers and dequantizers: ``libikq_iq2.dylib`` for the routed pair
(IQ2_K, IQ2_KS), ``libikq_iq1sr4.dylib`` for IQ1_S_R4, and
``libikq_dense.dylib`` for the dense set (IQ4_KS, IQ4_K, IQ5_K, IQ6_K).
Two jobs run through them:

- **Encode.** Conversion turns float rows plus an imatrix into ik wire bytes,
  which :mod:`mlx_ikq.format` then relays out. The arithmetic is upstream's,
  so packages this repository serves carry the same bytes an ik conversion
  would produce for the same input on this instruction set.
- **Reference.** ``dequantize_row_iq2_k``, ``dequantize_row_iq2_ks``, and
  ``dequantize_row_iq1_s_r4`` are the bit-exactness reference for the
  relayout and for the kernels. ik's own Metal helpers are not: for these
  members they hold the scale in ``half`` and fold it into the alphabet
  before indexing, where the CPU path multiplies in float32.

``IQ1_S_R4`` is a four-row-group wire: its quantizer and dequantizer consume
rows four at a time, so this surface requires row counts in multiples of
four for that member and hands the group dequantizer whole groups. A
per-row loop over that member's wire decodes garbage, which is why no such
loop exists here.

IQ2_KS conversion is pinned to the portable quantizer. Upstream's AVX2 path
searches a different candidate set and closes with a different multiplier, so
a row converted on an AVX2 host does not match one converted here. The
vendored copy contains the portable path only, which makes the pin structural
rather than a runtime flag that could be flipped by accident.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

from mlx_ikq.format import (
    DENSE_MEMBERS,
    IQ1_S_R4_BLOCK_BYTES,
    IQ2_K_BLOCK_BYTES,
    IQ2_KS_BLOCK_BYTES,
    IQ4_K_BLOCK_BYTES,
    IQ4_KS_BLOCK_BYTES,
    IQ5_K_BLOCK_BYTES,
    IQ6_K_BLOCK_BYTES,
    IkqFormatError,
    check_any_member,
    check_row_width,
    check_wire_rows,
    ik_row_bytes,
)

_HERE = Path(__file__).resolve().parent
_VENDOR_CANDIDATES = (
    _HERE.parents[1] / "vendor" / "ik_llama",          # source checkout
    _HERE.parent / "mlx_ikq_vendor" / "ik_llama",      # installed distribution
)

_LIB = None
_LIB_IQ1 = None
_DENSE_LIB = None

_DENSE_BLOCK_BYTES = {
    "iq4_ks": IQ4_KS_BLOCK_BYTES,
    "iq4_k": IQ4_K_BLOCK_BYTES,
    "iq5_k": IQ5_K_BLOCK_BYTES,
    "iq6_k": IQ6_K_BLOCK_BYTES,
}


class IkqCodecError(RuntimeError):
    pass


def vendor_dir() -> Path:
    """Directory holding the vendored codec sources and their build scripts."""
    for path in _VENDOR_CANDIDATES:
        if (path / "ikq_iq2.cpp").is_file():
            return path
    raise IkqCodecError(
        "vendored codec sources not found in "
        + ", ".join(str(p) for p in _VENDOR_CANDIDATES))


def library_path() -> Path:
    """Shared library the 2-bit codec loads, built beside the vendored sources."""
    return vendor_dir() / "libikq_iq2.dylib"


def iq1sr4_library_path() -> Path:
    """Shared library the IQ1_S_R4 codec loads."""
    return vendor_dir() / "libikq_iq1sr4.dylib"


def dense_library_path() -> Path:
    """Shared library the dense-member codec loads."""
    return vendor_dir() / "libikq_dense.dylib"


def _build() -> None:
    subprocess.run(["/bin/zsh", str(vendor_dir() / "build.sh")], check=True,
                   capture_output=True)


def _build_iq1sr4() -> None:
    subprocess.run(["/bin/zsh", str(vendor_dir() / "build_iq1sr4.sh")],
                   check=True, capture_output=True)


def load(build_if_missing: bool = True):
    """Load the vendored 2-bit codec, building it once if needed."""
    global _LIB
    if _LIB is not None:
        return _LIB
    library = library_path()
    if not library.exists():
        if not build_if_missing:
            raise IkqCodecError(f"{library} not built; run {vendor_dir()}/build.sh")
        _build()
    lib = ctypes.CDLL(str(library))
    c_f32 = ctypes.POINTER(ctypes.c_float)
    for name in ("ikq_row_size_iq2_k", "ikq_row_size_iq2_ks"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_size_t
        fn.argtypes = [ctypes.c_int64]
    for name in ("ikq_quantize_iq2_k", "ikq_quantize_iq2_ks"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_size_t
        fn.argtypes = [c_f32, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, c_f32]
    for name in ("ikq_dequantize_iq2_k", "ikq_dequantize_iq2_ks"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, c_f32, ctypes.c_int64, ctypes.c_int64]
    for name in ("ikq_block_size_iq2_k", "ikq_block_size_iq2_ks"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_size_t
        fn.argtypes = []
    if lib.ikq_block_size_iq2_k() != IQ2_K_BLOCK_BYTES:
        raise IkqCodecError(
            f"vendored block_iq2_k is {lib.ikq_block_size_iq2_k()} bytes, "
            f"the wire spec says {IQ2_K_BLOCK_BYTES}")
    if lib.ikq_block_size_iq2_ks() != IQ2_KS_BLOCK_BYTES:
        raise IkqCodecError(
            f"vendored block_iq2_ks is {lib.ikq_block_size_iq2_ks()} bytes, "
            f"the wire spec says {IQ2_KS_BLOCK_BYTES}")
    _LIB = lib
    return lib


def load_iq1sr4(build_if_missing: bool = True):
    """Load the vendored IQ1_S_R4 codec, building it once if needed."""
    global _LIB_IQ1
    if _LIB_IQ1 is not None:
        return _LIB_IQ1
    library = iq1sr4_library_path()
    if not library.exists():
        if not build_if_missing:
            raise IkqCodecError(
                f"{library} not built; run {vendor_dir()}/build_iq1sr4.sh")
        _build_iq1sr4()
    lib = ctypes.CDLL(str(library))
    c_f32 = ctypes.POINTER(ctypes.c_float)
    lib.ikq_row_size_iq1_s_r4.restype = ctypes.c_size_t
    lib.ikq_row_size_iq1_s_r4.argtypes = [ctypes.c_int64]
    lib.ikq_quantize_iq1_s_r4.restype = ctypes.c_size_t
    lib.ikq_quantize_iq1_s_r4.argtypes = [
        c_f32, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, c_f32]
    lib.ikq_dequantize_iq1_s_r4.restype = ctypes.c_int
    lib.ikq_dequantize_iq1_s_r4.argtypes = [
        ctypes.c_void_p, c_f32, ctypes.c_int64, ctypes.c_int64]
    lib.ikq_block_size_iq1_s_r4.restype = ctypes.c_size_t
    lib.ikq_block_size_iq1_s_r4.argtypes = []
    if lib.ikq_block_size_iq1_s_r4() != IQ1_S_R4_BLOCK_BYTES:
        raise IkqCodecError(
            f"vendored block_iq1_s_r4 is {lib.ikq_block_size_iq1_s_r4()} bytes, "
            f"the wire spec says {IQ1_S_R4_BLOCK_BYTES}")
    _LIB_IQ1 = lib
    return lib


def load_dense(build_if_missing: bool = True):
    """Load the vendored dense-member codec, building it once if needed."""
    global _DENSE_LIB
    if _DENSE_LIB is not None:
        return _DENSE_LIB
    library = dense_library_path()
    if not library.exists():
        if not build_if_missing:
            raise IkqCodecError(f"{library} not built; run {vendor_dir()}/build.sh")
        _build()
    lib = ctypes.CDLL(str(library))
    c_f32 = ctypes.POINTER(ctypes.c_float)
    for member in _DENSE_BLOCK_BYTES:
        fn = getattr(lib, f"ikq_row_size_{member}")
        fn.restype = ctypes.c_size_t
        fn.argtypes = [ctypes.c_int64]
        fn = getattr(lib, f"ikq_quantize_{member}")
        fn.restype = ctypes.c_size_t
        fn.argtypes = [c_f32, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, c_f32]
        fn = getattr(lib, f"ikq_dequantize_{member}")
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, c_f32, ctypes.c_int64, ctypes.c_int64]
        fn = getattr(lib, f"ikq_block_size_{member}")
        fn.restype = ctypes.c_size_t
        fn.argtypes = []
    for member, expect in _DENSE_BLOCK_BYTES.items():
        got = getattr(lib, f"ikq_block_size_{member}")()
        if got != expect:
            raise IkqCodecError(
                f"vendored block_{member} is {got} bytes, the wire spec "
                f"says {expect}")
    _DENSE_LIB = lib
    return lib


def _member_fn(member: str, kind: str):
    """The codec entry point for one member, loading the right library.

    Three vendored libraries carry disjoint member sets, so the member name
    picks the library rather than the caller.
    """
    if member in DENSE_MEMBERS:
        return getattr(load_dense(), f"ikq_{kind}_{member}")
    if member == "iq1_s_r4":
        return getattr(load_iq1sr4(), f"ikq_{kind}_{member}")
    return getattr(load(), f"ikq_{kind}_{member}")


def _f32(array: np.ndarray):
    """Contiguous float32 view plus its pointer; the view must outlive the call."""
    buf = np.ascontiguousarray(array, dtype=np.float32)
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), buf


def quantize(member: str, weights: np.ndarray,
             imatrix: np.ndarray | None = None) -> np.ndarray:
    """Quantize ``[rows, in_features]`` float weights to ik wire bytes.

    ``imatrix`` is one vector of ``in_features`` importance values reused for
    every row, matching upstream's per-tensor binding. A ``None`` imatrix
    selects the member's unweighted fallback objective, which is a different
    objective, not a neutral one.
    """
    check_any_member(member)
    if weights.ndim != 2:
        raise IkqFormatError(f"weights must be [rows, in_features], got {weights.shape}")
    rows, n = weights.shape
    check_row_width(n, member)
    check_wire_rows(member, rows)
    src = np.ascontiguousarray(weights, dtype=np.float32)
    row_bytes = ik_row_bytes(member, n)
    out = np.zeros((rows, row_bytes), dtype=np.uint8)
    if imatrix is None:
        im_ptr, _im_buf = ctypes.cast(0, ctypes.POINTER(ctypes.c_float)), None
    else:
        if imatrix.shape != (n,):
            raise IkqFormatError(
                f"imatrix must be one vector of {n} values, got {imatrix.shape}")
        im_ptr, _im_buf = _f32(imatrix)
    fn = _member_fn(member, "quantize")
    written = fn(src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 out.ctypes.data_as(ctypes.c_void_p), rows, n, im_ptr)
    if written != rows * row_bytes:
        raise IkqCodecError(
            f"{member} quantize wrote {written} bytes, expected {rows * row_bytes}")
    return out


def dequantize(member: str, wire: np.ndarray, in_features: int) -> np.ndarray:
    """Dequantize ik wire rows through ik's own CPU dequantizer.

    For ``iq1_s_r4`` the rows must form whole four-row groups; the group
    dequantizer consumes them four at a time. For ``iq6_k`` this is the
    cubic-polynomial reconstruction, the serving reference for that member.
    """
    check_any_member(member)
    n = check_row_width(in_features, member)
    row_bytes = ik_row_bytes(member, n)
    if wire.ndim != 2 or wire.shape[1] != row_bytes:
        raise IkqFormatError(
            f"{member} rows of {n} weights are {row_bytes} wire bytes, got {wire.shape}")
    buf = np.ascontiguousarray(wire, dtype=np.uint8)
    rows = check_wire_rows(member, buf.shape[0])
    out = np.zeros((rows, n), dtype=np.float32)
    fn = _member_fn(member, "dequantize")
    rc = fn(buf.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), rows, n)
    if rc != 0:
        raise IkqCodecError(f"{member} dequantize rejected geometry {n}")
    return out


__all__ = [
    "IkqCodecError",
    "dense_library_path",
    "dequantize",
    "iq1sr4_library_path",
    "library_path",
    "load",
    "load_dense",
    "load_iq1sr4",
    "quantize",
]
