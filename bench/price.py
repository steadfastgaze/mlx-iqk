"""Pricing cells: every kernel against its own deleted-decode address floor.

Protocol, the one the IQ_K decode precedent used:

- one process, one GPU lock, one thermal regime for the whole run;
- a lazy chain of independent dispatches per block, one ``mx.eval`` per block,
  best of several blocks per cell, so no per-call fence ever appears;
- arms block-alternated inside a cell with the order reversed on odd rounds;
- medians reported with the full spread;
- input activations and expert selections vary per dispatch from a pool, so a
  cell cannot be served out of one cached address stream.

Two gates run before any ratio is read:

- **Validation.** Every decode arm is checked against the reference decode
  before it is timed, so a fast arm that computes the wrong thing fails as a
  correctness failure rather than appearing as a result.
- **Deleted-load signature.** Every floor arm must stream inside the host's
  measured band, and no decode arm may beat its own floor by more than the
  round spread. A decode arm that appears faster than its address stream is
  reading fewer bytes than it claims.

The bar declared before the numbers: a decode arm within 10 to 15 percent of
its own floor puts the member in the same class as the formats designed for
this memory system. Materially above floor is a real cost and is reported as
one.

Usage:
    uv run --locked python bench/price.py --label headline --json-out bench/raw/headline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))

import floors
import wirepack
from mlx_ikq import format as fmt
from mlx_ikq.kernels import (
    DEQUANT_THREADS,
    ROWS_PER_TG,
    dequant_kernel,
    gemv_kernel,
    gemv_threads,
)

EXPERTS = 256
TOP_K = 6

GEOMETRIES = {
    # name: (out_features, in_features, projections per selected expert)
    "gate_up": (2048, 4096, 2),
    "down": (4096, 2048, 1),
}

DEQUANT_EXPERTS = 4
"""Experts in a dequantization cell.

The kernel is one thread per 32 weights with no cross-row state, so its rate
does not depend on the stack depth. The full 256-expert stack would be a
4 GB fp16 output per dispatch and a lazy chain of them would not fit, so the
cell prices the same per-row work over a shallow stack and reports the byte
volume it actually moves.
"""


# ---------------------------------------------------------------------------
# Host state
# ---------------------------------------------------------------------------


def thermal_sample() -> dict:
    try:
        raw = subprocess.run(
            ["mactop", "--headless", "--count", "1", "--format", "json"],
            capture_output=True, text=True, timeout=40, check=True).stdout
        d = json.loads(raw)[0]
        gpu = next(t for t in d["temperatures"] if t["group"] == "GPU")
        return {
            "thermal_state": d["thermal_state"],
            "gpu_avg_c": gpu["avg_celsius"],
            "gpu_max_c": gpu["max_celsius"],
            "on_ac_power": d["battery"]["on_ac_power"],
            "gpu_usage_pct": d["gpu_usage"],
        }
    except Exception as exc:  # instrument-local; never a silent pass
        return {"error": repr(exc)}


def powermode() -> str:
    try:
        out = subprocess.run(["pmset", "-g"], capture_output=True, text=True,
                             timeout=20, check=True).stdout
        for line in out.splitlines():
            if "powermode" in line:
                return line.strip()
    except Exception as exc:
        return repr(exc)
    return "unknown"


# ---------------------------------------------------------------------------
# Cell arrays
# ---------------------------------------------------------------------------


def gemv_arrays(member: str, geometry: str, seed: int, variants: int) -> dict:
    """Full-size stacked experts, plus a pool of activations and selections."""
    out_features, in_features, projections = GEOMETRIES[geometry]
    stacks = EXPERTS * projections
    wire = wirepack.random_wire(member, stacks * out_features, in_features,
                                seed=seed, scales="serving")
    streams = fmt.pack(member, wire, in_features)
    shapes = fmt.component_shapes(member, stacks, out_features, in_features)
    arrays = {name: mx.array(np.ascontiguousarray(value).reshape(shapes[name]))
              for name, value in streams.items()}
    rng = np.random.default_rng(seed + 1)
    tokens = projections
    mats = projections * TOP_K
    xs, sels = [], []
    for _ in range(variants):
        xs.append(mx.array(rng.standard_normal((tokens, in_features)).astype(np.float16)))
        picks = rng.choice(EXPERTS, TOP_K, replace=False).astype(np.uint32)
        sel = np.concatenate([picks + p * EXPERTS for p in range(projections)])
        sels.append(mx.array(sel.astype(np.uint32)))
    dims = mx.array([TOP_K], dtype=mx.uint32)
    mx.eval(list(arrays.values()) + xs + sels + [dims])
    return {
        "streams": arrays, "xs": xs, "sels": sels, "dims": dims,
        "member": member, "geometry": geometry, "mats": mats,
        "out_features": out_features, "in_features": in_features,
        "row_bytes": fmt.ik_row_bytes(member, in_features),
        "weight_bytes": mats * out_features * fmt.ik_row_bytes(member, in_features),
        "reference": None, "wire": wire,
    }


def dequant_arrays(member: str, geometry: str, seed: int) -> dict:
    out_features, in_features, _ = GEOMETRIES[geometry]
    rows = DEQUANT_EXPERTS * out_features
    wire = wirepack.random_wire(member, rows, in_features, seed=seed,
                                scales="serving")
    streams = fmt.pack(member, wire, in_features)
    shapes = fmt.component_shapes(member, DEQUANT_EXPERTS, out_features, in_features)
    arrays = {name: mx.array(np.ascontiguousarray(value).reshape(shapes[name]))
              for name, value in streams.items()}
    mx.eval(list(arrays.values()))
    read = rows * fmt.ik_row_bytes(member, in_features)
    written = rows * in_features * 2
    return {
        "streams": arrays, "member": member, "geometry": geometry,
        "out_features": out_features, "in_features": in_features,
        "weight_bytes": read + written, "read_bytes": read,
        "written_bytes": written, "wire": wire,
    }


# ---------------------------------------------------------------------------
# Validation, before any timing row
# ---------------------------------------------------------------------------


def validate_gemv(arrays: dict, kernel, threads: int) -> dict:
    """Sampled rows of one dispatch against a float64 reference."""
    member = arrays["member"]
    n, out = arrays["in_features"], arrays["out_features"]
    mats = arrays["mats"]
    x, sel = arrays["xs"][0], arrays["sels"][0]
    got = kernel(
        inputs=[x] + list(arrays["streams"].values())
        + [_member_table(member), sel, arrays["dims"]],
        grid=(mats * (out // ROWS_PER_TG) * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(mats * out,)],
        output_dtypes=[mx.float16],
    )[0]
    mx.eval(got)
    got = np.asarray(got, dtype=np.float64).reshape(mats, out)

    rng = np.random.default_rng(99)
    rows = _group_sample(member, out, rng.choice(out, 64, replace=False))
    sel_np = np.asarray(sel)
    xn = np.asarray(x)
    worst, ref_mag = 0.0, 0.0
    for mi in range(mats):
        absolute = int(sel_np[mi]) * out + rows
        weights = fmt.decode_wire(member, _slice_wire(arrays["wire"], absolute), n)
        ref = weights.astype(np.float64) @ xn[mi // TOP_K].astype(np.float64)
        worst = max(worst, float(np.max(np.abs(got[mi, rows] - ref))))
        ref_mag = max(ref_mag, float(np.max(np.abs(ref))))
    return {"max_abs_dev": worst, "max_ref_mag": ref_mag,
            "rel_dev": worst / ref_mag, "rows_sampled": int(rows.size),
            "ok": bool(worst / ref_mag < 1.5e-3)}


def validate_dequant(arrays: dict, kernel) -> dict:
    member = arrays["member"]
    n, out = arrays["in_features"], arrays["out_features"]
    got = kernel(
        inputs=list(arrays["streams"].values()) + [_member_table(arrays["member"])],
        grid=(DEQUANT_EXPERTS * out * (n // 32), 1, 1),
        threadgroup=(DEQUANT_THREADS, 1, 1),
        output_shapes=[(DEQUANT_EXPERTS, out, n)],
        output_dtypes=[mx.float16],
    )[0]
    mx.eval(got)
    got = np.asarray(got).reshape(-1, n)
    rng = np.random.default_rng(101)
    rows = _group_sample(member, got.shape[0],
                         rng.choice(got.shape[0], 256, replace=False))
    want = fmt.decode_wire(member, _slice_wire(arrays["wire"], rows),
                           n).astype(np.float16)
    mismatches = int(np.sum(got[rows].view(np.uint16) != want.view(np.uint16)))
    return {"rows_sampled": int(rows.size), "bit_mismatches": mismatches,
            "ok": bool(mismatches == 0)}


def _group_sample(member: str, count: int, picks: np.ndarray) -> np.ndarray:
    """Expand sampled row indices to whole wire groups, sorted.

    A grouped wire (`iq1_s_r4`, four rows per group) is addressable only at
    group boundaries, so the reference decode must receive complete groups;
    slicing single rows out of it would decode garbage and fail a correct
    kernel. Group 1 members pass through sorted.
    """
    group = fmt.WIRE_GROUP_ROWS[member]
    if group <= 1:
        return np.sort(np.asarray(picks, dtype=np.int64))
    starts = np.unique(np.asarray(picks, dtype=np.int64) // group) * group
    rows = (starts[:, None] + np.arange(group, dtype=np.int64)).reshape(-1)
    if rows.size and rows[-1] >= count:
        raise SystemExit(f"group sample {rows[-1]} exceeds {count} rows")
    return rows


def _slice_wire(wire: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(wire[rows])


def _member_table(member: str):
    from mlx_ikq.nn import member_table
    return member_table(member)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def measure(build, n_dispatch: int, batches: int) -> float:
    """Best-of-``batches`` blocks of ``n_dispatch`` lazily chained dispatches."""
    best = float("inf")
    for _ in range(batches):
        outs = [build(i) for i in range(n_dispatch)]
        t0 = time.perf_counter()
        mx.eval(outs)
        t1 = time.perf_counter()
        best = min(best, (t1 - t0) / n_dispatch)
        del outs
    return best * 1e6


def gemv_builder(arrays: dict, kernel, threads: int, variants: int):
    out, mats = arrays["out_features"], arrays["mats"]
    blocks = mats * (out // ROWS_PER_TG)
    streams = list(arrays["streams"].values())
    table = _member_table(arrays["member"])
    dims = arrays["dims"]

    def build(i: int):
        return kernel(
            inputs=[arrays["xs"][i % variants]] + streams
            + [table, arrays["sels"][i % variants], dims],
            grid=(blocks * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(mats * out,)],
            output_dtypes=[mx.float16],
        )[0]
    return build


def dequant_builder(arrays: dict, kernel):
    n, out = arrays["in_features"], arrays["out_features"]
    streams = list(arrays["streams"].values())
    table = _member_table(arrays["member"])
    threads = DEQUANT_EXPERTS * out * (n // 32)

    def build(_i: int):
        return kernel(
            inputs=streams + [table],
            grid=(threads, 1, 1),
            threadgroup=(DEQUANT_THREADS, 1, 1),
            output_shapes=[(DEQUANT_EXPERTS, out, n)],
            output_dtypes=[mx.float16],
        )[0]
    return build


def run_cell(name: str, builders: dict, weight_bytes: int, dispatches: int,
             batches: int, rounds: int) -> tuple[list, dict]:
    arms = list(builders)
    for arm in arms:
        mx.eval([builders[arm](i) for i in range(4)])
    rows = []
    for rnd in range(rounds):
        order = arms if rnd % 2 == 0 else list(reversed(arms))
        for arm in order:
            us = measure(builders[arm], dispatches, batches)
            rows.append({"cell": name, "round": rnd, "arm": arm,
                         "us_per_dispatch": us})
            print(f"  round {rnd} {arm:10s} {us:9.2f} us", flush=True)
    summary = {}
    for arm in arms:
        vals = [r["us_per_dispatch"] for r in rows if r["arm"] == arm]
        med = statistics.median(vals)
        summary[arm] = {
            "runs": [round(v, 3) for v in vals],
            "median": round(med, 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "spread": round(max(vals) - min(vals), 3),
            "bytes_per_dispatch": weight_bytes,
            "gb_per_s": round(weight_bytes / (med * 1e-6) / 1e9, 1),
        }
    summary["ratio_to_floor"] = round(
        summary["decode"]["median"] / summary["floor"]["median"], 4)
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="iq2_ks,iq2_k")
    ap.add_argument("--geometries", default="gate_up,down")
    ap.add_argument("--kinds", default="gemv,dequant")
    ap.add_argument("--dispatches", type=int, default=200)
    ap.add_argument("--dequant-dispatches", type=int, default=16)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--variants", type=int, default=16)
    ap.add_argument("--floor-band", default="330,440",
                    help="GB/s band every floor arm must fall inside")
    ap.add_argument("--bar", type=float, default=1.15,
                    help="declared ratio bar for a decode arm against its floor")
    ap.add_argument("--label", default="cell")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    members = [m for m in args.members.split(",") if m]
    geometries = [g for g in args.geometries.split(",") if g]
    kinds = [k for k in args.kinds.split(",") if k]
    band = tuple(float(v) for v in args.floor_band.split(","))

    thermal_before = thermal_sample()
    print(f"thermal before: {thermal_before}", flush=True)

    result = {
        "label": args.label,
        "protocol": {
            "dispatches_per_block": args.dispatches,
            "dequant_dispatches_per_block": args.dequant_dispatches,
            "blocks_per_cell": args.batches,
            "rounds": args.rounds,
            "input_variants": args.variants,
            "rows_per_tg": ROWS_PER_TG,
            "alternation": "arm order reversed on odd rounds",
            "experts": EXPERTS,
            "top_k": TOP_K,
            "dequant_experts": DEQUANT_EXPERTS,
        },
        "bars": {"decode_vs_own_floor": args.bar, "floor_band_gb_s": list(band)},
        "cells": {}, "validation": {}, "rows": [], "thermal": [],
    }

    for member in members:
        for geometry in geometries:
            out_features, in_features, _ = GEOMETRIES[geometry]
            if "gemv" in kinds:
                name = f"gemv_{member}_{geometry}"
                print(f"[{name}] {out_features}x{in_features}", flush=True)
                arrays = gemv_arrays(member, geometry, seed=wirepack.seed_for(name),
                                     variants=args.variants)
                threads = gemv_threads(in_features)
                decode = gemv_kernel(member, in_features, out_features)
                floor = floors.gemv_floor_kernel(member, in_features, out_features)
                report = validate_gemv(arrays, decode, threads)
                result["validation"][name] = report
                print(f"  validated rel_dev {report['rel_dev']:.3e}", flush=True)
                if not report["ok"]:
                    raise SystemExit(f"{name} failed validation, no timing taken")
                builders = {
                    "decode": gemv_builder(arrays, decode, threads, args.variants),
                    "floor": gemv_builder(arrays, floor, threads, args.variants),
                }
                rows, summary = run_cell(name, builders, arrays["weight_bytes"],
                                         args.dispatches, args.batches, args.rounds)
                summary["geometry"] = {
                    "out_features": out_features, "in_features": in_features,
                    "mats": arrays["mats"], "row_bytes": arrays["row_bytes"],
                    "bpw": fmt.bits_per_weight(member, in_features),
                }
                result["cells"][name] = summary
                result["rows"].extend(rows)
                del arrays, builders
                mx.clear_cache()
                result["thermal"].append({"after": name, **thermal_sample()})

            if "dequant" in kinds:
                name = f"dequant_{member}_{geometry}"
                print(f"[{name}] {out_features}x{in_features}", flush=True)
                arrays = dequant_arrays(member, geometry, seed=wirepack.seed_for(name))
                decode = dequant_kernel(member, in_features, out_features)
                floor = floors.dequant_floor_kernel(member, in_features, out_features)
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
                rows, summary = run_cell(name, builders, arrays["weight_bytes"],
                                         args.dequant_dispatches, args.batches,
                                         args.rounds)
                summary["geometry"] = {
                    "out_features": out_features, "in_features": in_features,
                    "experts": DEQUANT_EXPERTS,
                    "read_bytes": arrays["read_bytes"],
                    "written_bytes": arrays["written_bytes"],
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
        floor_rate = summary["floor"]["gb_per_s"]
        spread = summary["decode"]["spread"] + summary["floor"]["spread"]
        faster_than_floor = (summary["floor"]["median"]
                             - summary["decode"]["median"]) > spread
        gates[name] = {
            "floor_in_band": bool(band[0] <= floor_rate <= band[1]),
            "no_deleted_load_signature": bool(not faster_than_floor),
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
        print(f"{name:26s} decode {summary['decode']['median']:9.2f} us "
              f"floor {summary['floor']['median']:9.2f} us "
              f"ratio {summary['ratio_to_floor']:.4f} "
              f"({summary['decode']['gb_per_s']:.1f} / "
              f"{summary['floor']['gb_per_s']:.1f} GB/s) "
              f"gates {gates[name]}")
    print(f"all gates pass: {result['all_gates_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
