# Kernel pricing: every arm against its own address floor

Raw cell output: `results/headline.json`.

## Protocol

One process, one GPU lock held for the whole run, one thermal regime. Each
cell alternates its two arms with the order reversed on odd rounds, five
rounds, best of four blocks of independent lazily chained dispatches with one
`mx.eval` per block. No per-call fence is ever produced. Activations and
expert selections come from a pool of 16 variants, so a cell cannot be served
out of one cached address stream.

The floor arm issues the decode arm's load set byte for byte, deletes the
decode, and folds every loaded word into a hardened integer accumulator that
reaches the output. Nothing can be eliminated and no term can cancel.

Validation runs before any timing row in the same process. A decode GEMV arm
is checked against a float64 matmul of the reference-dequantized weights on
64 sampled rows of every matrix and lands at 2.6e-4 to 3.7e-4 relative; a
dequantization arm is checked for exact fp16 equality against the reference
decode on 256 sampled rows and shows zero bit mismatches in every cell. A cell
that fails validation raises instead of reporting a time.

Bar declared before the numbers: a decode arm within 10 to 15 percent of its
own floor.

## Geometry

The routed DS4 geometry. 256 experts, six selected per dispatch, gate and up
at `2048 x 4096` and down at `4096 x 2048`, `rows_per_tg` 16. The decode
cells select out of full-size stacked buffers: 512 stacks of 2048 rows for
gate/up, 256 stacks of 4096 rows for down.

The dequantization cells run a 4-expert stack. The kernel is one thread per
32 weights with no cross-row state, so its per-row rate does not depend on
the stack depth, and a full 256-expert fp16 output would be 4 GB per dispatch
with no way to hold a lazy chain of them. The byte volume those cells report
is read plus written, and the write dominates at roughly 7 to 1.

## Decode GEMV

| cell | bpw | bytes/dispatch | decode (us) | spread | floor (us) | spread | decode GB/s | floor GB/s | ratio |
|---|---|---|---|---|---|---|---|---|---|
| `IQ2_KS` gate/up `2048x4096` | 2.1914 | 27.57 MB | 78.13 | 0.14 | 69.43 | 0.14 | 352.9 | 397.2 | **1.125** |
| `IQ2_KS` down `4096x2048` | 2.1953 | 13.81 MB | 39.39 | 0.54 | 34.82 | 0.98 | 350.6 | 396.6 | **1.131** |
| `IQ2_K` gate/up `2048x4096` | 2.3750 | 29.88 MB | 82.85 | 0.38 | 74.73 | 0.46 | 360.7 | 399.9 | **1.109** |
| `IQ2_K` down `4096x2048` | 2.3750 | 14.94 MB | 40.94 | 0.24 | 36.00 | 0.28 | 364.9 | 415.0 | **1.137** |

An independent repeat of all eight cells on the same build, drawing different
synthetic wire data, gave 1.129, 1.130, 1.109 and 1.104. The decode medians
agree to 0.1 percent between the two runs; the spread between them is a floor
that moved, most of it in the `IQ2_K` down cell where the floor ran 37.10 us
in one session and 36.00 us in the other.

## Stacked dequantization

| cell | read | written | decode (us) | floor (us) | decode GB/s | floor GB/s | ratio |
|---|---|---|---|---|---|---|---|
| `IQ2_KS` gate/up | 9.19 MB | 67.11 MB | 191.28 | 192.41 | 398.9 | 396.6 | **0.994** |
| `IQ2_KS` down | 9.21 MB | 67.11 MB | 189.52 | 188.59 | 402.7 | 404.7 | **1.005** |
| `IQ2_K` gate/up | 9.96 MB | 67.11 MB | 187.69 | 187.56 | 410.6 | 410.9 | **1.001** |
| `IQ2_K` down | 9.96 MB | 67.11 MB | 193.34 | 190.77 | 398.6 | 404.0 | **1.014** |

## Gates

- **Bar.** Every decode GEMV arm sits between 1.109 and 1.137 of its own
  floor, inside the declared 10 to 15 percent. Every dequantization arm sits
  at its floor within 1.4 percent.
- **Floor band.** Every floor arm streams between 396.6 and 415.0 GB/s. Both
  kernel families and both members land in the same band, which is what a
  pure address stream should do.
- **No deleted-load signature.** No decode arm is faster than its own floor
  by more than the round spread. The dequantization arms that read a hair
  under their floor are inside a spread dominated by a single slow first
  round in each case, and the floor and decode arms move the same bytes by
  construction.

All gates pass on all eight cells.

## Host

`powermode 2`, on AC power throughout. GPU average temperature 35.8 C before
the run and 38.6 C after, the hottest single-core sample 46.3 C, thermal state
Nominal for every sample. MLX 0.31.2.

## Meaning

**The 2-bit members are not address-bound, and that is a property of the
rate, not a defect.** The decode arithmetic per weight is the same whatever
the code width, but at 2.19 bits per weight the address stream is a third
cheaper than at 3.4375, so what a 3-bit stream hides a 2-bit stream exposes.
The IQ3_K precedent priced identical arithmetic at 0.997x its floor; here the
same arithmetic, on a cheaper stream, prices at 1.11 to 1.14.

**The remaining excess is the value lookup, and cutting it was worth 20
percent.** With the two representation changes the IQ3_K cell established
(fp16 value table, one contiguous code plane), the 2-bit members priced at
1.42 and 1.37. Reading each sub-block's four alphabet entries into registers
once and selecting between them per weight, instead of indexing the
threadgroup table once per weight, moves them to 1.13 and 1.11 (measured on
the down geometry of each member in the tuning session). The selected
entry is the entry the index would have returned, so nothing about the
reconstruction changes.

Two alternatives were measured in the same session and rejected: a `float`
threadgroup table (1.44 and 1.38, no change) and a `simd_shuffle` table held
in lanes 0 to 7 (3.82 and 3.52, far worse). Splitting the inner accumulation
into 2, 4, or 8 independent chains measured flat to slightly negative, so the
shipped kernel keeps one accumulator chain per sub-block. `rows_per_tg` of 8
and 32 were measured against 16 and neither wins on both geometries.

**The prefill producer is write-bound and already at its floor.** It writes
fp16 for every weight it decodes, seven bytes out for every byte in, so the
decode disappears under the store stream in every cell.

---

# Dense-member pricing (draft)

Raw cell output: `results/dense_headline.json`. Same protocol, same 1.15
bar, one-token GEMV dispatches, `bench/price_dense.py`.

Two protocol elements are specific to the dense cells.

- **A matrix pool per cell.** One dense matrix is a few MiB; a
  repeated-dispatch cell over a single resident matrix is served out of
  cache (floors measured 837 to 1295 GB/s that way, far above DRAM), which
  voids the floor as an address-stream price. The serving model is
  DRAM-cold weights, since a decode token streams every dense tensor of
  every layer between repeats of any one matrix, so each cell cycles its
  dispatches over enough buffer copies to span 256 MiB. The lm_head exceeds
  that on its own.
- **Gate semantics at the DRAM limit.** The dense decode arms sit at or
  microscopically below their floors, so the deleted-load signature takes
  both of its conditions (faster than floor beyond spread, and an implied
  rate above the streaming band's top); the rate condition is the defining
  one. The dequantization floors are mixed read-plus-write streams (writes
  two to four times the wire bytes) and run 273 to 298 GB/s, below the
  pure-read band, so those cells carry their own declared 250 to 440 band.

## Dense decode GEMV, one token

| cell | bpw | bytes/dispatch | decode (us) | floor (us) | decode GB/s | floor GB/s | ratio |
|---|---|---|---|---|---|---|---|
| `IQ4_KS` `2048x4096` | 4.2578 | 4.46 MB | 12.93 | 12.86 | 345.3 | 347.3 | **1.006** |
| `IQ4_KS` `4096x2048` | 4.2656 | 4.47 MB | 12.65 | 12.68 | 353.7 | 352.8 | **0.998** |
| `IQ4_KS` `8192x1024` | 4.2813 | 4.49 MB | 12.84 | 12.78 | 349.6 | 351.2 | **1.005** |
| `IQ4_KS` `129280x4096` (lm_head) | 4.2578 | 281.83 MB | 701.71 | 706.75 | 401.6 | 398.8 | **0.993** |
| `IQ6_K` `2048x4096` | 6.625 | 6.95 MB | 18.99 | 19.05 | 365.9 | 364.7 | **0.997** |
| `IQ6_K` `4096x2048` | 6.625 | 6.95 MB | 18.98 | 19.08 | 366.0 | 364.0 | **0.995** |
| `IQ6_K` `8192x1024` | 6.625 | 6.95 MB | 19.14 | 19.14 | 363.0 | 363.0 | **1.000** |
| `IQ6_K` `129280x4096` (lm_head) | 6.625 | 438.52 MB | 1094.17 | 1100.84 | 400.8 | 398.3 | **0.994** |

## Dense dequantization (prefill bridge)

| cell | read | written | decode (us) | floor (us) | decode GB/s | floor GB/s | ratio |
|---|---|---|---|---|---|---|---|
| `IQ4_KS` `2048x4096` | 4.46 MB | 16.78 MB | 77.68 | 76.28 | 273.5 | 278.5 | **1.018** |
| `IQ4_KS` `4096x2048` | 4.47 MB | 16.78 MB | 75.96 | 76.17 | 279.8 | 279.0 | **0.997** |
| `IQ6_K` `2048x4096` | 6.95 MB | 16.78 MB | 82.56 | 81.49 | 287.4 | 291.1 | **1.013** |
| `IQ6_K` `4096x2048` | 6.95 MB | 16.78 MB | 79.53 | 80.56 | 298.3 | 294.5 | **0.987** |

## Gates

All twelve cells pass all three gates: floors in their declared bands,
ratios 0.987 to 1.018 against the 1.15 bar, no deleted-load signature
(every decode arm's implied rate is at or below the streaming band).

## What the dense numbers say

**The higher-bpw members are address-bound; the 2-bit exposure never
appears.** At 4.25 to 6.625 bits per weight, the per-weight dynamically
indexed threadgroup table read that the 2-bit kernels had to replace with
register quads is fully hidden under the address stream: every dense decode
arm prices at its floor, replicating the 3.4375-bit precedent (0.997x) at
every dense width including 1024, where a threadgroup is a single
simdgroup.

**Decode cost per dense matmul is its byte count.** One token through the
lm_head costs 702 us at `IQ4_KS` and 1094 us at `IQ6_K`, both at 400 GB/s;
the small dense matrices cost 13 to 19 us each DRAM-cold. Member choice on
the dense side is therefore a pure bytes-times-quality decision; no member
carries a kernel penalty.

---

# IQ1_S_R4 session (`results/iq1sr4_headline.json`, `results/iq1sr4_attrib.json`)

Same protocol, same geometries, the 1.5-bit grid member. Validation before
any timing row: GEMV rel_dev 3.8e-4 and 2.3e-4 against the float64
reference, dequantization 0 bit mismatches over 968 and 1004 sampled rows.

| cell | decode us | floor us | ratio | decode / floor GB/s | gates |
|---|---|---|---|---|---|
| gemv gate/up | 122.19 | 67.85 | **1.801** | 154.9 / 278.9 | floor below band, above bar |
| gemv down | 61.74 | 34.71 | **1.779** | 153.7 / 273.3 | floor below band, above bar |
| dequant gate/up | 213.76 | 210.37 | 1.016 | 343.5 / 349.0 | all pass |
| dequant down | 217.65 | 212.81 | 1.023 | 337.4 / 345.1 | all pass |

**The decode misses the declared 1.15 bar and the miss is recorded, not
tuned away.** The attribution arms name the mechanism
(`results/iq1sr4_attrib.json`, medians, gate/up then down):

| arm | us | GB/s | reading |
|---|---|---|---|
| floor_nogrid (streams only, gathers deleted) | 45.97 / 24.23 | 411.7 / 391.5 | the pure 1.5-bpw wire stream runs in the platform band |
| floor (streams + grid gathers) | 67.85 / 34.78 | 278.9 / 272.8 | the four dependent 8-byte table gathers per lane block add 44 to 48 percent to the address stream; they, not the stream, put the floor below band |
| decode_fixidx (gathers de-correlated from the stream, bit-exact:no, bound only) | 109.16 / 55.19 | 173.4 / 171.9 | breaking the load-to-gather dependency recovers only 13.1 / 6.5 us; latency chaining is a minor term |
| decode (served kernel) | 122.23 / 61.72 | 154.8 / 153.7 | the remaining +54.4 / +26.9 us over the full floor is the exposed per-weight chain (char-to-float conversion, shift add, fma) |

The family trend continues one step further: the IQ3_K-class arithmetic
hides behind a 3.4-bit stream (0.997x), is exposed at 2.2 bits (1.11 to
1.14), and at 1.5 bits even the table lookup itself no longer fits under
the stream. Unlike the 2-bit members, the lookup here is a 2048-entry
device-memory gather that register staging cannot replace. Candidate
levers if this ratio ever gets a declared campaign: a threadgroup-staged
grid with simdgroup-broadcast reads, and software pipelining of the next
row's stream loads across the row loop. Neither is measured; the ratio
above is the standing price. The prefill producer is at its floor, so the
serving prefill route pays no such tax.
