# Dense-tensor serving for the higher IQ_K members

Draft design for serving dense (non-expert) tensors quantized as `IQ4_KS`,
`IQ4_K`, `IQ5_K`, or `IQ6_K` through this repository's relayout-and-kernel
pattern. Nothing here lands in serving defaults; the routed 2-bit surface is
untouched. Source citations are to the ik_llama.cpp checkout at
`b341d0b58b24d69a635bb81cdb907f9369c73dec`.

## Deviations and decisions declared up front

1. **IQ6_K has two upstream reconstructions and this draft pins the CPU one.**
   ik's CPU dequantizer (`iqk_quantize.cpp:3448-3491`) reconstructs a code `q`
   through the cubic `A + q*(B + q*(-C + q*D)) (+ S if shifted)` with
   `A=-127, B=6.2568, C=0.11218, D=0.0011972, S=1` (`3442-3446`). ik's Metal
   kernels reconstruct the same code through the integer table
   `kvalues_iq6k_f` (`ggml-metal.metal:4119-4127`, helper `9364-9383`), which
   is the cubic rounded to integers; the two differ by up to 0.488 codebook
   units per weight (about 0.4 percent of the value range). The quantizer
   fits codes and the super-block scale against the integer table
   (`3790-3806`). This draft serves the CPU dequantizer's numbers: the
   128-entry float32 table of the cubic's outputs is extracted from the
   vendored codec once and embedded bit-exactly, so kernel, reference decode,
   and vendored CPU dequantizer agree to the bit. Serving therefore matches
   ik's CPU serving, not ik's Metal serving. The other three members have a
   single value set and no such fork.
2. **The alphabet register-quad staging of the 2-bit kernels does not carry
   over, deliberately.** See "Value handling" below for the replacement and
   the evidence.
3. **Tests and pricing use repository-root `uv` entry points.** Run the suite
   with `uv run --locked --group dev pytest`; run pricing with
   `uv run --locked python bench/price.py ...` or the corresponding dense
   command, writing scratch JSON under the ignored `bench/raw/` directory.
4. **IQ2_KS-style host pinning is not needed here.** None of the four
   quantizers has an AVX2-only variant; all are single-implementation and
   block-local or row-local as noted below, so vendored conversion is
   reproducible across hosts by construction.
5. **The dense pricing cells extend the floor protocol in two measured
   ways** (details in `bench/RESULTS.md`): each cell cycles a pool of
   weight-matrix buffer copies because a single resident dense matrix is
   cache-served and its floor prices at 837 to 1295 GB/s instead of the
   DRAM band; and the dequantization floors, being mixed read-plus-write
   streams, carry their own declared 250 to 440 GB/s band. The deleted-load
   signature takes both of its conditions (faster than floor beyond spread
   at an implied rate above the band top), because the dense decode arms
   sit at the DRAM limit where a floor arm's own overhead can put it
   fractionally behind the decode arm.

## 1. Per-member geometry

`QK_K = 256` throughout; all four members declare `blck_size = QK_K`. Byte
budgets verified from `sizeof` via the block structs, not from documentation;
the vendored library asserts the same sizes at build time.

| member | block struct | block bytes | row meta | bpw in-block |
|---|---|---|---|---|
| `IQ4_KS` | `scales[8], qs[128]` (`ggml-common.h:618-622`) | 136 | fp32, 4 bytes (`ggml.c:1364-1376`) | 4.25 |
| `IQ4_K` | `d, extra, scales_h[4], scales_l[8], qs[128]` (`719-726`) | 144 | none (`ggml.c:1754-1766`) | 4.5 |
| `IQ5_K` | `d, extra, scales_h[4], scales_l[8], qs[128], qh[32]` (`737-745`) | 176 | none (`ggml.c:1793-1809`) | 5.5 |
| `IQ6_K` | `d, extra, scales[16] (int8), qs[128], qh[64]` (`756-764`) | 212 | none (`ggml.c:1823-1836`) | 6.625 |

Row bytes (`ggml_row_size` semantics: row meta plus block bytes times
`n/256`) at the dense widths:

| member | n=1024 | n=2048 | n=4096 | n=8192 | lm_head row (4096) |
|---|---|---|---|---|---|
| `IQ4_KS` | 548 | 1092 | 2180 | 4356 | 2180 |
| `IQ4_K` | 576 | 1152 | 2304 | 4608 | 2304 |
| `IQ5_K` | 704 | 1408 | 2816 | 5632 | 2816 |
| `IQ6_K` | 848 | 1696 | 3392 | 6784 | 6784 x 129280 rows = 818.5 MiB at n=4096: 3392 |

### Alphabets and select bits

| member | table | entries per alphabet | shifted = base + | select granularity | select bit lives in |
|---|---|---|---|---|---|
| `IQ4_KS` | `iq4k_values` (`ggml-common.h:2227-2230`) | 16 | +4 | 32 weights | bit 0 of the per-32 scale byte (`iqk_quantize.cpp:4565-4566`) |
| `IQ4_K` | `iq4k_values` | 16 | +4 | 16 weights | `extra` bit `ib` (`2841-2843`) |
| `IQ5_K` | `iq5nl_values` (`2232-2235`) | 32 | +2 | 16 weights | `extra` bit `ib` (`3134-3137`) |
| `IQ6_K` | `iq6nl_values` (`2237-2246`) / the cubic | 64 | +1 (`S_IQ6K`) | 16 weights | `extra` bit `ib` (`3468-3471`) |

Every `iq4k_values` and `iq5nl_values` entry is a small integer, exact in
fp16. The IQ6_K cubic's outputs are not fp16-exact, so its served table is
float32.

### Scale mechanics

| member | sub-block | scale payload | range | zero representable |
|---|---|---|---|---|
| `IQ4_KS` | 32 | 7 bits, odd values `(byte&254)-127` (`4555-4575`) | +/-{1..127} | no |
| `IQ4_K` | 16 | 6 bits offset-32: low 4 in `scales_l` nibble, high 2 in `scales_h` (`3058-3063`, decode `2835-2837`) | [-32, 31] | yes |
| `IQ5_K` | 16 | 6 bits offset-32, same split (`3380-3383`, decode `3129-3133`) | [-32, 31] | yes |
| `IQ6_K` | 16 | direct signed int8 (`3742-3744`, decode `3464-3467`) | [-127, 127] | yes |

`scales_h` byte layout is identical for `IQ4_K` and `IQ5_K`: 2-bit field
`ib` at byte `ib/4`, shift `2*(ib%4)` (IQ4_K writes it through a `uint16_t`
alias at `3062`, IQ5_K byte-wise at `3383`; the bytes agree). All four
members fold the sign into the scale payload; the super-block (or row) `d`
carries its own sign from the `-max/…` normalization.

### Quantizer shape (for the codec vendoring)

All four use the shared `QHelper` (`iqk_quantize.cpp:46-92`) with
`block_size` 32 for `IQ4_KS` and 16 for the others, the seed-plus-sweep
search (seed `d = -max/values[0]` both orientations, then `itry` in
`[-ntry, ntry]` over base and shifted alphabets, both orientations), and a
closing `sumqx/sumq2` rescale with no empirical multiplier:

| member | ntry | first-level weights (imatrix / fallback) | second level | scope |
|---|---|---|---|---|
| `IQ4_KS` | 7 | `qw*sqrt(2*sumx2/256 + x^2)` / `x^2` (`4394-4398`) | `amax_scale/127`, unweighted, whole row (`4482-4500`) | row-local (row-wide amax and fp32 row scale) |
| `IQ4_K` | 7 | same / `x^2` (`2942-2946`) | `-max_scale/32`, unweighted (`3032-3040`) | block-local |
| `IQ5_K` | 5 | same / `0.25*sigma2 + x^2` (`3264-3268`) | `-max_scale/32`, unweighted (`3372-3383`) | block-local |
| `IQ6_K` | 5 | same / `0.25*sigma2 + x^2` (`3617-3621`) | `-max_scale/127`, direct int8 (`3733-3744`) | block-local |

The unweighted-fallback split is member-specific: the 4-bit pair falls back
to bare `x^2`, the 5/6-bit pair to `0.25*sigma2 + x^2`. A missing imatrix
therefore changes the objective differently per member; conversion here
treats the imatrix as required unless the caller explicitly opts out, as the
existing `codec.quantize` contract already states.

`IQ4_KS` is the one row-scope member: its second level normalizes by the
row-wide unweighted `amax_scale`, so one outlier sub-block anywhere in a
dense row sets the scale step for the whole row. At the dense widths this
exposure is 1024 to 8192 weights per row. Recorded as a quality flag for the
conversion side, not a serving concern.

### Metal presence upstream

All four members ship `mul_mv`, `mul_mv_id`, `mul_mm`, `mul_mm_id`, and
`get_rows` Metal kernels (`ggml-metal.m:830-841`, `873-884`), so the port
has a complete upstream reference for reconstruction semantics. The upstream
helpers hold scales in `float` and index `float` tables
(`ggml-metal.metal:9091-9109`, `9321-9339`, `9342-9362`, `9364-9384`); apart
from the IQ6_K table fork described above, their per-weight arithmetic
matches the CPU dequantizers.

## 2. Dense wire: relayout definition

A dense matrix is one `[out_features, in_features]` tensor: the degenerate
single-expert case of the stacked relayout. Every stream keeps the stacked
form's per-row layout and drops the expert axis; `format.pack` already
operates on `[rows, ik_row_bytes]` and is reused unchanged, with
`rows = out_features`. The relayout is byte-count preserving against
`ggml_row_size` for every member (asserted by tests, plane by plane).

Streams per member (`o = out_features`, `n = in_features`):

`IQ4_KS`

| stream | granularity | dtype/shape | content |
|---|---|---|---|
| `qs` | 1 weight | uint32 `(o, n/8)` | 4-bit code, little-endian nibble order |
| `scl` | 32 weights | uint8 `(o, n/32)` | the ik scale byte unchanged: bit 0 alphabet select, bits 1..7 scale |
| `dv` | 1 row | float32 `(o,)` | fp32 row scale |

`IQ4_K`

| stream | granularity | dtype/shape | content |
|---|---|---|---|
| `qs` | 1 weight | uint32 `(o, n/8)` | 4-bit code |
| `scl` | 16 weights | uint8 `(o, n/32)` | low 4 scale bits, two sub-blocks per byte |
| `sch` | 16 weights | uint8 `(o, n/64)` | high 2 scale bits, four sub-blocks per byte |
| `sex` | 16 weights | uint8 `(o, n/128)` | alphabet-select bit |
| `dv` | 256 weights | fp16 `(o, n/256)` | super-block scale |

`IQ5_K`: `IQ4_K`'s five streams plus

| stream | granularity | dtype/shape | content |
|---|---|---|---|
| `qh` | 1 weight | uint32 `(o, n/32)` | the fifth code bit, little-endian bit order |

`IQ6_K`

| stream | granularity | dtype/shape | content |
|---|---|---|---|
| `qs` | 1 weight | uint32 `(o, n/8)` | low 4 code bits |
| `qh` | 1 weight | uint32 `(o, n/16)` | high 2 code bits, little-endian field order |
| `scl` | 16 weights | uint8 `(o, n/16)` | direct signed int8 scale (stored as its byte) |
| `sex` | 16 weights | uint8 `(o, n/128)` | alphabet-select bit |
| `dv` | 256 weights | fp16 `(o, n/256)` | super-block scale |

Only the code planes move. The scale planes (`scales`/`scales_l`/`scales_h`
nibbles and fields, `extra` bits, int8 scales) are already indexed by the
k-order sub-block inside a super-block, so the relayout concatenates them
across super-blocks unchanged; the code planes are permuted from the
members' in-block placements to one k-contiguous little-endian stream per
plane. In-block code positions (weight `w` of a super-block):

- `IQ4_KS`/`IQ4_K` `qs`: byte `16*(w/32) + w%16`, nibble `(w%32)/16`
  (encoders `4510-4514`, `3067-3071`).
- `IQ5_K`/`IQ6_K` `qs`: byte `32*(w/64) + w%32`, nibble `(w%64)/32`
  (encoders `3397-3402`, `3757-3764`).
- `IQ5_K` `qh`: byte `w%32`, bit `w/32` (encoder `3403`).
- `IQ6_K` `qh`: byte `32*(w/128) + w%32`, field shift `2*((w/32)%4)`
  (encoder `3765`).

Reconstruction of one weight, every member, in the CPU dequantizer's own
float32 order:

```
dl = d * float(scale_payload)          # one fp32 multiply
y  = dl * table[select_offset + code]  # one fp32 multiply
```

with `d` the fp32 row scale (`IQ4_KS`) or the fp16 super-block scale
promoted to fp32, `scale_payload` the member's signed integer scale, and
`table` the member's value set (`iq4k_values`, `iq5nl_values`, or the
embedded IQ6_K cubic-output table).

## 3. Kernel plan

### Shared with the 2-bit kernels

The dense GEMV keeps the structure as a whole: one threadgroup
walks 16 output rows of one token; each lane owns 32 consecutive weights;
`simdgroups = n/1024`; activations are loaded once into registers via the
literal-index `half2` sequence; geometry is baked into the source so
addresses are shifts; the reduction is one `simd_sum` per row plus a
threadgroup add of the simdgroup partials; the stacked dequantization is one
thread per 32 weights storing fp16, with an out-row-range variant taking a
row-offset input for allocator liveness. Width is a guarded contract:
`(1024, 2048, 4096, 8192)` are the sized dense widths, anything else is
refused at kernel-build time.

What the dense forms drop: the expert-select input and the pair/activation
row mapping. A token indexes its own activation row, and the row id is the
output row directly. The union/sorted variants have no dense counterpart
because there is nothing to deduplicate: every token reads the same single
matrix.

### Bigger alphabets changes

The 2-bit kernels stage each sub-block's four alphabet entries into
registers and select between them per weight, because at 2.2 bits per weight
the address stream is thin enough that a dynamically indexed threadgroup
load prices above it. That staging does not generalize: 16, 32, or 64
entries per alphabet would cost a 15-to-63-select tree per weight or an
untenable register set.

The replacement is the per-weight dynamically indexed load out of a
threadgroup-staged table, with the alphabet select folded into a per-sub-
block register base offset (`t_ = select << log2(entries)`), so the select
costs one shift per sub-block and the per-weight work is shift, mask, one
threadgroup load, one fma. The evidence that this suffices at these
densities: the measured 3.4375-bpw decode with exactly this read pattern on
a k-contiguous relayout priced at 0.997x its address floor, and every dense
member here carries a heavier address stream than that (4.25 to 6.625 bpw),
which hides more decode, not less. The price cells exist to verify this
against the declared 1.15 bar rather than assume it.

Table dtype per member: half for `IQ4_KS`/`IQ4_K` (32 entries) and `IQ5_K`
(64), exact by integrality; float32 for `IQ6_K` (128 entries, 512 bytes of
threadgroup memory), because the cubic outputs are not fp16-exact.

Per-lane loads per 32-weight group: one `uint4` of `qs` (all members), plus
one `uint` of `qh` (`IQ5_K`) or one `uint2` of `qh` (`IQ6_K`), plus the
member's scale bytes. Sub-16 members compute two sub-scales per group and
run two accumulators, factoring each `dl` out of its half's partial sum
exactly as the `IQ2_K` kernel does; `IQ4_KS` runs one accumulator per group.

### Prefill bridge

Dense prefill reuses the stacked dequantization pattern at expert count one:
`dense_dequant` materializes `[out, in]` fp16 in one pass, bit-identical to
the reference decode, and the projection is a plain `mx.matmul`. The range
form (`dense_dequant_range`, power-of-two `range_rows`, runtime `obase`)
bounds in-flight buffer bytes; the lm_head (`129280 x 4096`) does not tile
by one power of two, so its bridge issues `126 x 1024`-row ranges plus one
`256`-row tail, which the two compiled range variants cover.

### GEMV grid and the lm_head

Decode grid: `tokens * (out/16)` threadgroups of `32*(n/1024)` threads. At
the lm_head shape one token dispatches 8080 threadgroups of 128 threads,
enough occupancy to stream the 418 MiB (`IQ4_KS`) at the memory system's
rate; `129280 % 16 == 0` so the row tiling is exact. All dense out-features
in the DS4 set (1024, 2048, 4096, 8192, 129280) are multiples of 16.

## 4. Codec and verification plan

- `vendor/ik_llama/ikq_dense.{h,cpp}`: byte-for-byte copies of the four
  quantizers, dequantizers, their `best_index` helpers and index tables,
  `QHelper`, and the value tables, with the same scaffolding changes as the
  existing vendored file (literal row sizes, geometry rejection instead of
  assertion). The upstream `user_data` parameter is dropped: all four
  quantizers mark it unused.
- Wire identity: the extended `tools/ikref_upstream.c` drives the same
  inputs through `libggml`'s `ggml_quantize_chunk` and type-trait
  `to_float`, and the suite asserts byte equality both ways, plus
  `ggml_row_size` agreement with `format.ik_row_bytes` on real quantize
  returns.
- Reference decode: `format.decode` on the relayout must be bit-identical to
  the vendored `dequantize_row_*` over random wire and over exhaustive
  (scale, select, code) enumerations, the existing two-sweep pattern.
- Kernels: the dequantization is compared bit-for-bit against the reference
  decode rounded to fp16; the GEMV against a float64 matmul of the
  reference-decoded weights, at widths including 1024 and 8192 and the
  lm_head shape, for fp16 and bfloat16 activations at the call seam.

## 5. Bench plan

Per the floor-arithmetic convention: every priced kernel gets a deleted-
decode sibling floor (same loads byte for byte, decode deleted, odd-prime
integer accumulate reaching the output, stores reproduced for the
dequantization family), both arms in one process under the GPU lock,
block-alternated, order reversed on odd rounds, medians with full spread,
bytes charged from the wire row model. Declared bar: decode within **1.15x**
of its own floor; floor band 330 to 440 GB/s. Lead members first: `IQ4_KS`
(the deeper-savings target) and `IQ6_K` (the q8_0 replacement candidate), at
the dense shapes `(2048, 4096)`, `(4096, 2048)`, `(8192, 1024)`, and the
lm_head `(129280, 4096)`.

## 6. Integration API surface

Module-level functions in `mlx_ikq.dense`, mirroring the `nn.py`
conventions; the member is an argument, streams are the wire components as
`mx.array`s in the format module's stream order:

- `dense_value_table(member) -> mx.array` — the member's table, one shared
  array per process.
- `dense_gemv(member, x, streams, out_features, in_features) -> [tokens, out]`
  fp16 — the decode route; accepts fp16 or bfloat16 `x` and casts at the
  seam.
- `dense_dequantized(member, streams, out_features, in_features) -> [out, in]`
  fp16 — the prefill weight producer, bit-identical to the reference decode.
- `dense_dequantized_range(member, streams, out_features, in_features,
  start, rows) -> [rows, in]` fp16 — the bounded-liveness form.
- `dense_linear(member, x, streams, out_features, in_features)` — the
  route chooser (GEMV at decode-shaped token counts, dequantize-and-matmul
  otherwise; the crossover is provisional until measured in context).

Build side: `format.pack(member, wire, in_features)` on `[out, row_bytes]`
wire from `codec.quantize(member, weights, imatrix)`, then
`format.dense_component_shapes(member, out_features, in_features)` names the
shapes the loader materializes.
