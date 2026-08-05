"""Dense-member pricing cells: every kernel against its own deleted-decode floor.

The dense counterpart of ``bench/price.py``, same protocol and gates: one
process under one GPU lock, lazy dispatch chains with one ``mx.eval`` per
block, arms block-alternated with the order reversed on odd rounds, medians
with the full spread, validation before any timing row, and the deleted-load
signature check. Bytes are charged from the wire row model
(``ggml_row_size``-equal), never from an allocation.

The bar declared before the numbers: a decode arm within **1.15x** of its
own floor. GEMV floor arms must stream inside the 330-440 GB/s read band;
the dequantization family's floors are mixed read-plus-write streams (the
fp16 output is two to four times the wire bytes) and carry their own band,
250-440 GB/s, because a mixed DRAM stream runs below the pure-read rate.

The deleted-load signature takes both of its conditions: a decode arm
faster than its own floor by more than the round spread, at an implied
byte rate above the streaming band's top. The rate condition is the
defining one (a kernel that skips loads shows an impossible rate); the
floor comparison alone can misfire by the floor arm's own overhead when a
decode arm sits exactly at the DRAM limit.

Cells: the two lead members (``iq4_ks``, the deeper-savings target;
``iq6_k``, the q8_0 replacement candidate) at the dense shapes, including
the lm_head. The GEMV cells run one token, the decode shape.

Each cell cycles its dispatches over a pool of weight matrices sized past
the system-level cache. A single resident dense matrix is a few MiB and a
repeated-dispatch cell over it is served out of cache: floors measured 837
to 1295 GB/s that way, far above the DRAM band, which voids the floor as an
address-stream price. The serving model is DRAM-cold weights: a decode
token streams every dense tensor of every layer between repeats of any one
matrix. Pool entries are buffer copies (cache identity is by address, not
content), so the pool multiplies setup memory, not setup time. The lm_head
matrix exceeds the cache on its own and pools at one.

Usage:
    uv run --locked python bench/price_dense.py --label dense --json-out bench/raw/dense.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))

from price import powermode, run_cell, thermal_sample

import floors
import wirepack
from mlx_ikq import format as fmt
from mlx_ikq.dense import dense_value_table
from mlx_ikq.dense_kernels import (
    DENSE_DEQUANT_THREADS,
    ROWS_PER_TG,
    dense_dequant_kernel,
    dense_gemv_kernel,
    dense_gemv_threads,
)

SHAPES = {
    # name: (out_features, in_features)
    "shared_up": (2048, 4096),
    "shared_down": (4096, 2048),
    "indexer": (8192, 1024),
    "lm_head": (129280, 4096),
}

GEMV_SHAPES = ("shared_up", "shared_down", "indexer", "lm_head")
DEQUANT_SHAPES = ("shared_up", "shared_down")


# ---------------------------------------------------------------------------
# Cell arrays
# ---------------------------------------------------------------------------


POOL_TARGET_BYTES = 256 * 1024 * 1024
"""Weight bytes a cell's matrix pool must at least span."""

POOL_MAX_MATRICES = 64


def dense_arrays(member: str, shape: str, seed: int, variants: int) -> dict:
    """A pool of dense matrices in relayout form, plus activation variants."""
    out_features, in_features = SHAPES[shape]
    wire = wirepack.random_wire(member, out_features, in_features, seed=seed,
                                scales="serving")
    streams = fmt.pack(member, wire, in_features)
    shapes = fmt.dense_component_shapes(member, out_features, in_features)
    hosts = {name: np.ascontiguousarray(value).reshape(shapes[name])
             for name, value in streams.items()}
    row_bytes = fmt.ik_row_bytes(member, in_features)
    matrix_bytes = out_features * row_bytes
    copies = min(POOL_MAX_MATRICES,
                 max(1, -(-POOL_TARGET_BYTES // matrix_bytes)))
    pool = [{name: mx.array(value) for name, value in hosts.items()}
            for _ in range(copies)]
    rng = np.random.default_rng(seed + 1)
    xs = [mx.array(rng.standard_normal((1, in_features)).astype(np.float16))
          for _ in range(variants)]
    for entry in pool:
        mx.eval(list(entry.values()))
    mx.eval(xs)
    return {
        "pool": pool, "xs": xs, "member": member, "shape": shape,
        "out_features": out_features, "in_features": in_features,
        "row_bytes": row_bytes,
        "pool_matrices": copies,
        "weight_bytes": matrix_bytes,
        "read_bytes": matrix_bytes,
        "written_bytes": out_features * in_features * 2,
        "wire": wire,
    }


# ---------------------------------------------------------------------------
# Validation, before any timing row
# ---------------------------------------------------------------------------


def validate_gemv(arrays: dict, kernel, threads: int) -> dict:
    """Sampled rows of one dispatch against a float64 reference."""
    member = arrays["member"]
    n, out = arrays["in_features"], arrays["out_features"]
    x = arrays["xs"][0]
    got = kernel(
        inputs=[x] + list(arrays["pool"][0].values()) + [dense_value_table(member)],
        grid=((out // ROWS_PER_TG) * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(out,)],
        output_dtypes=[mx.float16],
    )[0]
    mx.eval(got)
    got = np.asarray(got, dtype=np.float64)

    rng = np.random.default_rng(99)
    rows = np.sort(rng.choice(out, 64, replace=False))
    weights = fmt.decode_wire(member, np.ascontiguousarray(arrays["wire"][rows]), n)
    ref = weights.astype(np.float64) @ np.asarray(x[0]).astype(np.float64)
    worst = float(np.max(np.abs(got[rows] - ref)))
    ref_mag = float(np.max(np.abs(ref)))
    return {"max_abs_dev": worst, "max_ref_mag": ref_mag,
            "rel_dev": worst / ref_mag, "rows_sampled": int(rows.size),
            "ok": bool(worst / ref_mag < 1.5e-3)}


def validate_dequant(arrays: dict, kernel) -> dict:
    member = arrays["member"]
    n, out = arrays["in_features"], arrays["out_features"]
    got = kernel(
        inputs=list(arrays["pool"][0].values()) + [dense_value_table(member)],
        grid=(out * (n // 32), 1, 1),
        threadgroup=(DENSE_DEQUANT_THREADS, 1, 1),
        output_shapes=[(out, n)],
        output_dtypes=[mx.float16],
    )[0]
    mx.eval(got)
    got = np.asarray(got)
    rng = np.random.default_rng(101)
    rows = np.sort(rng.choice(out, 256, replace=False))
    want = fmt.decode_wire(member, np.ascontiguousarray(arrays["wire"][rows]),
                           n).astype(np.float16)
    mismatches = int(np.sum(got[rows].view(np.uint16) != want.view(np.uint16)))
    return {"rows_sampled": int(rows.size), "bit_mismatches": mismatches,
            "ok": bool(mismatches == 0)}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def gemv_builder(arrays: dict, kernel, threads: int, variants: int):
    out = arrays["out_features"]
    blocks = out // ROWS_PER_TG
    pool = [list(entry.values()) for entry in arrays["pool"]]
    copies = len(pool)
    table = dense_value_table(arrays["member"])

    def build(i: int):
        return kernel(
            inputs=[arrays["xs"][i % variants]] + pool[i % copies] + [table],
            grid=(blocks * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(out,)],
            output_dtypes=[mx.float16],
        )[0]
    return build


def dequant_builder(arrays: dict, kernel):
    n, out = arrays["in_features"], arrays["out_features"]
    pool = [list(entry.values()) for entry in arrays["pool"]]
    copies = len(pool)
    table = dense_value_table(arrays["member"])
    threads = out * (n // 32)

    def build(i: int):
        return kernel(
            inputs=pool[i % copies] + [table],
            grid=(threads, 1, 1),
            threadgroup=(DENSE_DEQUANT_THREADS, 1, 1),
            output_shapes=[(out, n)],
            output_dtypes=[mx.float16],
        )[0]
    return build


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="iq4_ks,iq6_k")
    ap.add_argument("--gemv-shapes", default=",".join(GEMV_SHAPES))
    ap.add_argument("--dequant-shapes", default=",".join(DEQUANT_SHAPES))
    ap.add_argument("--kinds", default="gemv,dequant")
    ap.add_argument("--dispatches", type=int, default=200)
    ap.add_argument("--lm-head-dispatches", type=int, default=48,
                    help="dispatches per block for the lm_head cells, whose "
                         "single dispatch already streams hundreds of MiB")
    ap.add_argument("--dequant-dispatches", type=int, default=16)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--variants", type=int, default=16)
    ap.add_argument("--floor-band", default="330,440",
                    help="GB/s band every GEMV floor arm must fall inside")
    ap.add_argument("--dequant-floor-band", default="250,440",
                    help="GB/s band for the mixed read-plus-write dequant floors")
    ap.add_argument("--bar", type=float, default=1.15,
                    help="declared ratio bar for a decode arm against its floor")
    ap.add_argument("--label", default="dense")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    members = [m for m in args.members.split(",") if m]
    gemv_shapes = [s for s in args.gemv_shapes.split(",") if s]
    dequant_shapes = [s for s in args.dequant_shapes.split(",") if s]
    kinds = [k for k in args.kinds.split(",") if k]
    band = tuple(float(v) for v in args.floor_band.split(","))
    dequant_band = tuple(float(v) for v in args.dequant_floor_band.split(","))

    thermal_before = thermal_sample()
    print(f"thermal before: {thermal_before}", flush=True)

    result = {
        "label": args.label,
        "protocol": {
            "dispatches_per_block": args.dispatches,
            "lm_head_dispatches_per_block": args.lm_head_dispatches,
            "dequant_dispatches_per_block": args.dequant_dispatches,
            "blocks_per_cell": args.batches,
            "rounds": args.rounds,
            "input_variants": args.variants,
            "rows_per_tg": ROWS_PER_TG,
            "alternation": "arm order reversed on odd rounds",
            "tokens_per_gemv_dispatch": 1,
            "pool_target_bytes": POOL_TARGET_BYTES,
            "pool_max_matrices": POOL_MAX_MATRICES,
        },
        "bars": {"decode_vs_own_floor": args.bar, "floor_band_gb_s": list(band),
                 "dequant_floor_band_gb_s": list(dequant_band)},
        "cells": {}, "validation": {}, "rows": [], "thermal": [],
    }

    for member in members:
        if "gemv" in kinds:
            for shape in gemv_shapes:
                out_features, in_features = SHAPES[shape]
                name = f"dense_gemv_{member}_{shape}"
                print(f"[{name}] {out_features}x{in_features}", flush=True)
                arrays = dense_arrays(member, shape, seed=wirepack.seed_for(name),
                                      variants=args.variants)
                threads = dense_gemv_threads(in_features)
                decode = dense_gemv_kernel(member, in_features, out_features)
                floor = floors.dense_gemv_floor_kernel(member, in_features,
                                                       out_features)
                report = validate_gemv(arrays, decode, threads)
                result["validation"][name] = report
                print(f"  validated rel_dev {report['rel_dev']:.3e}", flush=True)
                if not report["ok"]:
                    raise SystemExit(f"{name} failed validation, no timing taken")
                builders = {
                    "decode": gemv_builder(arrays, decode, threads, args.variants),
                    "floor": gemv_builder(arrays, floor, threads, args.variants),
                }
                dispatches = (args.lm_head_dispatches if shape == "lm_head"
                              else args.dispatches)
                rows, summary = run_cell(name, builders, arrays["weight_bytes"],
                                         dispatches, args.batches, args.rounds)
                summary["geometry"] = {
                    "out_features": out_features, "in_features": in_features,
                    "row_bytes": arrays["row_bytes"],
                    "bpw": fmt.bits_per_weight(member, in_features),
                    "pool_matrices": arrays["pool_matrices"],
                }
                result["cells"][name] = summary
                result["rows"].extend(rows)
                del arrays, builders
                mx.clear_cache()
                result["thermal"].append({"after": name, **thermal_sample()})

        if "dequant" in kinds:
            for shape in dequant_shapes:
                out_features, in_features = SHAPES[shape]
                name = f"dense_dequant_{member}_{shape}"
                print(f"[{name}] {out_features}x{in_features}", flush=True)
                arrays = dense_arrays(member, shape, seed=wirepack.seed_for(name),
                                      variants=1)
                decode = dense_dequant_kernel(member, in_features, out_features)
                floor = floors.dense_dequant_floor_kernel(member, in_features,
                                                          out_features)
                report = validate_dequant(arrays, decode)
                result["validation"][name] = report
                print(f"  validated {report['bit_mismatches']} bit mismatches "
                      f"over {report['rows_sampled']} rows", flush=True)
                if not report["ok"]:
                    raise SystemExit(f"{name} failed validation, no timing taken")
                builders = {
                    "decode": dequant_builder(arrays, decode),
                    "floor": dequant_builder(arrays, floor),
                }
                cell_bytes = arrays["read_bytes"] + arrays["written_bytes"]
                rows, summary = run_cell(name, builders, cell_bytes,
                                         args.dequant_dispatches, args.batches,
                                         args.rounds)
                summary["geometry"] = {
                    "out_features": out_features, "in_features": in_features,
                    "read_bytes": arrays["read_bytes"],
                    "written_bytes": arrays["written_bytes"],
                    "pool_matrices": arrays["pool_matrices"],
                }
                result["cells"][name] = summary
                result["rows"].extend(rows)
                del arrays, builders
                mx.clear_cache()
                result["thermal"].append({"after": name, **thermal_sample()})

    result["host"] = {
        "thermal_before": thermal_before,
        "thermal_after": thermal_sample(),
        "powermode": powermode(),
        "mlx": mx.__version__,
    }

    gates = {}
    for name, summary in result["cells"].items():
        cell_band = dequant_band if name.startswith("dense_dequant") else band
        floor_rate = summary["floor"]["gb_per_s"]
        decode_rate = summary["decode"]["gb_per_s"]
        spread = summary["decode"]["spread"] + summary["floor"]["spread"]
        faster_than_floor = (summary["floor"]["median"]
                             - summary["decode"]["median"]) > spread
        impossible_rate = decode_rate > cell_band[1]
        gates[name] = {
            "floor_in_band": bool(cell_band[0] <= floor_rate <= cell_band[1]),
            "no_deleted_load_signature": bool(
                not (faster_than_floor and impossible_rate)),
            "within_bar": bool(summary["ratio_to_floor"] <= args.bar),
        }
    result["gates"] = gates
    result["all_gates_pass"] = all(all(g.values()) for g in gates.values())

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        path = Path(args.json_out)
        if not path.is_absolute():
            path = HERE / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        print(f"wrote {path}", flush=True)
    for name, summary in result["cells"].items():
        print(f"{name:32s} decode {summary['decode']['median']:9.2f} us "
              f"floor {summary['floor']['median']:9.2f} us "
              f"ratio {summary['ratio_to_floor']:.4f} "
              f"({summary['decode']['gb_per_s']:.1f} / "
              f"{summary['floor']['gb_per_s']:.1f} GB/s) "
              f"gates {gates[name]}")
    print(f"all gates pass: {result['all_gates_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
