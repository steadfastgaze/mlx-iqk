# mlx-iqk

IQ_K relayout routed-expert kernels for MLX on Apple silicon, covering three
members of the IQ_K family: `IQ2_KS` (2.1875 bits per weight plus a 16-bit
row scale), `IQ2_K` (2.375 bits per weight), and `IQ1_S_R4` (1.5 bits per
weight plus a 16-bit row scale).

The bits are ik_llama's, but at build time each stream is rearranged so values
consumed consecutively along the reduction dimension (`k`, the input-feature
dimension) are contiguous in memory. The eight-entry value table is promoted
to fp16, codes are read from one contiguous 2-bit plane, and each sub-block's
four alphabet entries are loaded into registers once rather than once per
weight.

None of that changes a reconstructed value or a stored byte, and together they
take the decode from 1.42x its address floor to 1.13x. The prefill weight
producer runs at its floor. `IQ1_S_R4` decodes from the 2048-entry ternary
grid held in device memory; its wire arrives as four-row groups and the
relayout restores per-row addressability without moving a bit of coding.

On an M3 Max, over the routed geometry (256 experts, six selected, gate and up
at `2048x4096`, down at `4096x2048`):

| member | projection | decode vs own floor | dequantization vs own floor |
|---|---|---|---|
| `IQ2_KS` | gate/up | 1.125 | 0.994 |
| `IQ2_KS` | down | 1.131 | 1.005 |
| `IQ2_K` | gate/up | 1.109 | 1.001 |
| `IQ2_K` | down | 1.137 | 1.014 |

Every floor arm streams between 397 and 415 GB/s. See `bench/RESULTS.md`.

## Format

One logical weight tensor is `[out_features, in_features]`. Per row:

`IQ2_KS`

| stream | granularity | content |
|---|---|---|
| `qs`  | 1 weight    | 2-bit code, little-endian in uint32 words |
| `scl` | 32 weights  | low 4 bits of the signed scale index |
| `sch` | 32 weights  | high bit of the signed scale index |
| `sex` | 32 weights  | alphabet-select bit |
| `dv`  | 1 row       | fp16 row scale |

`IQ2_K`

| stream | granularity | content |
|---|---|---|
| `qs`  | 1 weight    | 2-bit code, little-endian in uint32 words |
| `scl` | 16 weights  | 4-bit offset-8 signed scale index |
| `sex` | 16 weights  | alphabet-select bit |
| `dv`  | 256 weights | fp16 super-block scale |

`IQ1_S_R4`

| stream | granularity | content |
|---|---|---|
| `qs`  | 8 weights   | low 8 bits of the 11-bit grid index, four to a uint32 word |
| `qh`  | 32 weights  | one uint16: index high bits, block scale, shift sign |
| `dv`  | 1 row       | fp16 row scale |

A 2-bit weight reconstructs as
`d * float(scale_index) * values[4*alphabet + code]` with `values` the
`iq2nl_values` table; an `IQ1_S_R4` weight as
`d * float(2*ls + 1) * (float(grid_value) + shift)` with `grid_value` a
ternary value from the 2048-entry IQ1_S grid and `shift` a per-block
`+-0.125`. Every step is float32 in both chains. That is the CPU
dequantizer's own order of operations, and the CPU dequantizer is this
repository's bit-exactness reference. ik's Metal helpers are not: they hold
the scale in `half` and fold it into the value table before indexing, so
they do not agree with the CPU path bit for bit.

The relayout is byte-exact against the member it carries. `IQ2_KS` rows cost
`2 + 70 * n/256` bytes, `IQ2_K` rows cost `76 * n/256` bytes, and
`IQ1_S_R4` rows cost `2 + 6 * n/32` bytes, the same budgets the ik block
structs spend. `IQ1_S_R4`'s ik wire interleaves four rows into one group
(only whole groups are addressable), so its `pack`/`unpack` and encoder shim
take row counts in multiples of four; the relayout rows themselves are
per-row addressable like every other member's.

The dense side of a package carries its own member set: `IQ4_KS`, `IQ4_K`,
`IQ5_K`, and `IQ6_K`, each with its own geometry, relayout, codec, and
decode kernels (`mlx_iqk.dense`, `mlx_iqk.dense_kernels`). Those members are
priced and gated but reach no serving default; a consumer selects them
explicitly. fp4 and q8_0 tensors serve through their own paths outside this
repository.

## Surface

```python
from mlx_iqk import IqkSwitchLinear

proj = IqkSwitchLinear("iq2_ks", num_experts, out_features, in_features)
proj.load_streams(streams)          # from mlx_iqk.format.pack
y = proj(x, indices, sorted_indices=True)
```

`IqkSwitchLinear` matches the `(x, indices)` switch-linear interface: a call
with few token-expert pairs runs the fused decode GEMV; a larger call
dequantizes the stacked experts in one kernel pass and runs `mx.gather_mm`,
honouring the caller's sort flag so each expert's weights are read once per
sorted prefill. The member is per instance, so one layer's projections can
carry different members.

Supported input widths are exactly `2048` and `4096`, the two widths of the
routed geometry. A decode threadgroup makes one pass over the reduction axis,
so an unsized width would leave part of a row unread and return a wrong
answer; the generator refuses it instead.

## Conversion

`mlx_iqk.codec` wraps a copy of ik_llama's own quantizers and
dequantizers, so the repository converts float rows plus an imatrix to wire
bytes without a checkout of that engine.

```python
from mlx_iqk import format as fmt
from mlx_iqk import codec

wire = codec.quantize("iq2_ks", rows, imatrix)   # ik wire bytes
streams = fmt.pack("iq2_ks", wire, in_features)  # the served relayout
```

`IQ2_KS` conversion is pinned to the portable quantizer. Upstream's AVX2 path
searches a different candidate set, adds a per-sub-block RMSE refinement, and
closes with a 1.000 multiplier where the portable path closes with 1.030, so
the two produce different bytes for the same row.

## Tests

```
uv run --locked --group dev pytest
```

runs the suite from the repository root. Kernel tests are marked `gpu`; the
exhaustive relayout sweeps are marked `slow`.

The suite is standalone: it needs this repository's own environment and a
build of `vendor/ik_llama`, nothing else. One test additionally cross-checks
the vendored codec against `libggml` from a local ik_llama checkout and skips
when that checkout is absent.

## Pricing

```
uv run --locked python bench/price.py --label headline --json-out bench/raw/headline.json
```

prices every kernel against a deleted-decode address floor: identical loads,
decode removed, a hardened integer accumulator reaching the output. Arms
alternate inside a cell, validation runs before any timing row, and the gates
are the declared ratio bar, a floor-rate band, and the absence of a
deleted-load signature. `bench/RESULTS.md` records the run.

## Licensing

MIT. `vendor/ik_llama` carries code copied from ik_llama.cpp under its own
MIT terms; see `NOTICE` and `ATTRIBUTION.md`.
