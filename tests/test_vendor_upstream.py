"""The vendored codec reproduces ik_llama's own bytes.

The rest of the suite treats ``vendor/ik_llama`` as the reference. That is
only sound if the vendored copy agrees with the tree it was copied from, so
this test drives the same inputs through ``libggml`` from a local ik_llama
checkout and compares byte for byte, in both directions:

- quantize: the same float rows and the same imatrix must produce the same
  wire bytes, which also pins the portable IQ2_KS path.
- dequantize: the same wire bytes must produce bit-identical float32.

The test skips unless ``IK_LLAMA_DIR`` names a checkout. It is the one part
of the suite that is not standalone.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import wirepack
from mlx_ikq import codec
from mlx_ikq import format as fmt

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "ikref_upstream"
BUILD = ROOT / "tools" / "build_ikref_upstream.sh"


@pytest.fixture(scope="module")
def upstream(tmp_path_factory):
    ik_llama_dir = os.environ.get("IK_LLAMA_DIR")
    if not ik_llama_dir:
        pytest.skip("IK_LLAMA_DIR is not set; skipping the upstream cross-check")
    ik_llama = Path(ik_llama_dir)
    if not ik_llama.is_dir():
        pytest.skip(f"IK_LLAMA_DIR is not a directory: {ik_llama}")
    if not TOOL.exists():
        if shutil.which("cmake") is None:
            pytest.skip("cmake not available to build the upstream cross-check")
        subprocess.run(["/bin/zsh", str(BUILD)], check=True, capture_output=True)
    return tmp_path_factory.mktemp("ikref")


def _run(work: Path, member: str, mode: str, payload: np.ndarray, out_shape,
         nrows: int, n: int, imatrix: np.ndarray | None) -> np.ndarray:
    src = work / "in.bin"
    dst = work / "out.bin"
    src.write_bytes(np.ascontiguousarray(payload).tobytes())
    args = [str(TOOL), member, mode, str(src), str(dst), str(nrows), str(n)]
    if imatrix is not None:
        im = work / "imatrix.f32"
        im.write_bytes(np.ascontiguousarray(imatrix, dtype=np.float32).tobytes())
        args.append(str(im))
    subprocess.run(args, check=True, capture_output=True, text=True)
    dtype = np.uint8 if mode == "quantize" else np.float32
    return np.fromfile(dst, dtype=dtype).reshape(out_shape)


ALL_MEMBERS = fmt.MEMBERS + fmt.DENSE_MEMBERS


@pytest.mark.parametrize("member", ALL_MEMBERS)
@pytest.mark.parametrize("n", (256, 2048))
def test_vendored_dequantize_matches_upstream(upstream, member, n):
    wire = wirepack.random_wire(member, 8, n, seed=101)
    ref = _run(upstream, member, "dequantize", wire, (8, n), 8, n, None)
    got = codec.dequantize(member, wire, n)
    mismatches = int(np.sum(ref.view(np.uint32) != got.view(np.uint32)))
    assert mismatches == 0


@pytest.mark.parametrize("member", ALL_MEMBERS)
@pytest.mark.parametrize("with_imatrix", (True, False))
def test_vendored_quantize_matches_upstream(upstream, member, with_imatrix):
    n, rows = 2048, 8
    rng = np.random.default_rng(202)
    weights = (rng.standard_normal((rows, n)) * 0.05).astype(np.float32)
    imatrix = rng.random(n).astype(np.float32) if with_imatrix else None
    row_bytes = fmt.ik_row_bytes(member, n)
    ref = _run(upstream, member, "quantize", weights, (rows, row_bytes), rows, n,
               imatrix)
    got = codec.quantize(member, weights, imatrix)
    assert int(np.sum(ref != got)) == 0


@pytest.mark.parametrize("member", ALL_MEMBERS)
def test_upstream_row_size_matches_the_wire_spec(upstream, member):
    """``ggml_quantize_chunk`` writes exactly ``ik_row_bytes`` per row.

    The row count is deliberately not a power of two, and it is taken in
    whole wire groups: ``iq1_s_r4`` addresses four rows at a time and
    upstream aborts on a partial group.
    """
    n, rows = 1024, 3 * fmt.WIRE_GROUP_ROWS[member]
    rng = np.random.default_rng(404)
    weights = (rng.standard_normal((rows, n)) * 0.05).astype(np.float32)
    row_bytes = fmt.ik_row_bytes(member, n)
    wire = _run(upstream, member, "quantize", weights, (rows, row_bytes),
                rows, n, None)
    assert wire.shape == (rows, row_bytes)


def test_upstream_relayout_end_to_end(upstream):
    """Upstream bytes, this repository's relayout, upstream's own decode."""
    n, rows = 4096, 4
    rng = np.random.default_rng(303)
    weights = (rng.standard_normal((rows, n)) * 0.03).astype(np.float32)
    imatrix = rng.random(n).astype(np.float32)
    for member in ALL_MEMBERS:
        row_bytes = fmt.ik_row_bytes(member, n)
        wire = _run(upstream, member, "quantize", weights, (rows, row_bytes),
                    rows, n, imatrix)
        ref = _run(upstream, member, "dequantize", wire, (rows, n), rows, n, None)
        got = fmt.decode_wire(member, wire, n)
        mismatches = int(np.sum(ref.view(np.uint32)
                                != got.astype(np.float32).view(np.uint32)))
        assert mismatches == 0, member
