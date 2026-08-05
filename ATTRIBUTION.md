# Third-party provenance

The IQ_K formats and their quantizers are Iwan Kawrakow's work. Any release
of this repository names ik_llama.cpp as the source of both.

One directory carries some copied code, and this file is its inventory.

The rule followed here is that a
notice is reproduced only where code was actually copied, that the copied
region is named at symbol granularity, and that every change to a copied body
is stated.

## What is not copied

The Metal kernels, the relayout, the reference decode, the switch module, and
the pricing cells are original. ik's own Metal kernels were read for their
structure and are named in the design notes; no line of them is reproduced,
and their numerical behaviour is explicitly not the target — for both 2-bit
members they hold the sub-block scale in `half` and fold it into the value
table before indexing, where the CPU dequantizers this repository targets
multiply in float32.

## ik_llama.cpp

- **Upstream**: <https://github.com/ikawrakow/ik_llama.cpp>
- **Pin**: commit `b341d0b58b24d69a635bb81cdb907f9369c73dec`
- **License**: MIT (`LICENSE` in that repository, naming the ggml, llama.cpp,
  and ik_llama.cpp authors). Reproduced in this repository's `NOTICE`.
- **Where**: `vendor/ik_llama/iqk_iq2.cpp`, `vendor/ik_llama/iqk_iq1sr4.cpp`,
  `vendor/ik_llama/iqk_dense.cpp`

Copied symbols and their upstream locations (`iqk_iq2.cpp`):

| symbol | upstream file | upstream lines |
|---|---|---|
| `block_iq2_k`, `block_iq2_ks`, `iq2nl_values` | `ggml/src/ggml-common.h` | 635-641, 659-664, 2212-2214 |
| the NEON `fp16` conversion pair | `ggml/src/ggml-impl.h` | 431-450 |
| `nearest_int` | `ggml/src/iqk/iqk_quantize.cpp` | 37-42 |
| `QHelper` (`row_weights`, `quantize`) | `ggml/src/iqk/iqk_quantize.cpp` | 46-92 |
| `make_qx_quants` | `ggml/src/iqk/iqk_quantize.cpp` | 113-155 |
| `best_index_iq2nl` | `ggml/src/iqk/iqk_quantize.cpp` | 1185-1188 |
| `quantize_row_iq2_k_impl` | `ggml/src/iqk/iqk_quantize.cpp` | 1190-1333 |
| `dequantize_row_iq2_k` | `ggml/src/iqk/iqk_quantize.cpp` | 1356-1385 |
| `quantize_row_iq2_ks_impl` | `ggml/src/iqk/iqk_quantize.cpp` | 1692-1840 |
| `dequantize_row_iq2_ks` | `ggml/src/iqk/iqk_quantize.cpp` | 1877-1909 |

Changes to the copied bodies, complete:

1. `ggml_row_size` and the type-trait lookup become the two literal row-size
   expressions, `76 * n/256` and `2 + 70 * n/256`.
2. `GGML_ASSERT` on the row geometry becomes a return of zero, so a rejected
   geometry surfaces to the caller instead of aborting a long conversion.
   Upstream's own quantize driver does neither: it rewrites the type and
   carries on, which silently produces a differently shaped tensor.
3. `quantize_user_data` is dropped and the AVX2 `IQ2_KS` quantizer is not
   copied. The remaining path is upstream's portable one. This is a pin, not
   an omission: the two paths search different candidate sets, only the AVX2
   one applies a per-sub-block RMSE refinement, and they close with different
   multipliers (1.000 against 1.030), so they produce different bytes for the
   same row.
4. `GGML_FP32_TO_FP16` and `GGML_FP16_TO_FP32` become local functions with
   the same NEON body, falling back to `_Float16` where `__fp16` is absent.

Copied symbols and their upstream locations (`iqk_iq1sr4.cpp`):

| symbol | upstream file | upstream lines |
|---|---|---|
| `block_iq1_s_r4` | `ggml/src/ggml-common.h` | 528-532 |
| `IQ1S_DELTA` | `ggml/src/ggml-common.h` | 1434 |
| the NEON `fp16` conversion pair | `ggml/src/ggml-impl.h` | 431-450 |
| `nearest_int` | `ggml/src/iqk/iqk_quantize.cpp` | 37-42 |
| `GROUP_MAX_EPS_IQ1_S` | `ggml/src/ggml-quants.c` | 31 |
| `iq2_compare_func` | `ggml/src/ggml-quants.c` | 12583-12587 |
| `kgrid_1bit_2048` and the `IQ1_S` grid/map/neighbour construction of `iq2xs_init_impl` | `ggml/src/ggml-quants.c` | 12589-12943 |
| `iq1_sort_helper` | `ggml/src/ggml-quants.c` | 13155-13160 |
| `iq1_find_best_neighbour2` | `ggml/src/ggml-quants.c` | 14189-14244 |
| `iq1s_process_1block` | `ggml/src/ggml-quants.c` | 14246-14344 |
| `quantize_iq1_s_r4` | `ggml/src/iqk/iqk_quantize.cpp` | 8107-8193 |
| `dequantize_row_iq1_s_r4` | `ggml/src/iqk/iqk_quantize.cpp` | 8195-8216 |

Changes to those copied bodies, complete:

1. `ggml_row_size` becomes the literal row-size expression `2 + 6 * n/32`,
   and the row-count and width `GGML_ASSERT`s become error returns through
   the wrapper surface, so a rejected geometry surfaces to the caller.
2. The grid, map, and neighbour tables that upstream builds in
   `ggml_quantize_init` are built on first use by the same loops,
   specialised to the `IQ1_S` constants (`grid_size` 2048, `kmap_size`
   43692, `nwant` 3). The dequantizer's static `iq1s_grid` value table is
   derived from `kgrid_1bit_2048` (byte `i` of entry `k` is
   `((code >> 2*i) & 3) - 1`), a derivation that reproduces the shipped
   table entry for entry and is pinned against upstream's own library by
   the cross-check below.
3. `quantize_user_data` is dropped; the member has no fast-path branch, so
   the remaining path is upstream's only path.
4. `dequantize_row_iq1_s_r4` is wrapped in a loop over four-row groups; the
   group body is upstream's, and the wrapper refuses row counts that do not
   form whole groups (a per-row call over this wire decodes garbage).

Nothing else was touched. `tests/test_vendor_upstream.py` builds a checker
against `libggml` from a local ik_llama checkout and asserts the vendored
quantizers and dequantizers reproduce upstream's output bit for bit; it
skips when no checkout is present.

Copied symbols and their upstream locations (`iqk_dense.cpp`):

| symbol | upstream file | upstream lines |
|---|---|---|
| `block_iq4_ks` | `ggml/src/ggml-common.h` | 618-622 |
| `block_iq4_k` | `ggml/src/ggml-common.h` | 719-726 |
| `block_iq5_k` | `ggml/src/ggml-common.h` | 737-745 |
| `block_iq6_k` | `ggml/src/ggml-common.h` | 757-764 |
| `iq4k_values` | `ggml/src/ggml-common.h` | 2227-2230 |
| `iq5nl_values` | `ggml/src/ggml-common.h` | 2232-2235 |
| `iq6nl_values` | `ggml/src/ggml-common.h` | 2237-2246 |
| the NEON `fp16` conversion pair | `ggml/src/ggml-impl.h` | 431-450 |
| `nearest_int` | `ggml/src/iqk/iqk_quantize.cpp` | 37-42 |
| `QHelper` (`row_weights`, `quantize`) | `ggml/src/iqk/iqk_quantize.cpp` | 46-92 |
| `iq4nl_index`, `best_index_iq4nl` | `ggml/src/iqk/iqk_quantize.cpp` | 2901-2916 |
| `iq5nl_index`, `best_index_iq5nl` | `ggml/src/iqk/iqk_quantize.cpp` | 3219-3234 |
| the `IQ6_K` reconstruction constants (`A_IQ6K` through `S_IQ6K`) | `ggml/src/iqk/iqk_quantize.cpp` | 3442-3446 |
| `iq6nl_index`, `best_index_iq6nl` | `ggml/src/iqk/iqk_quantize.cpp` | 3572-3592 |
| `quantize_row_iq4_k_impl_bs16` | `ggml/src/iqk/iqk_quantize.cpp` | 2918-3072 |
| `quantize_row_iq4_k_impl_bs128` | `ggml/src/iqk/iqk_quantize.cpp` | 4369-4528 |
| `quantize_row_iq5_k_impl` | `ggml/src/iqk/iqk_quantize.cpp` | 3236-3416 |
| `quantize_row_iq6_k_impl` | `ggml/src/iqk/iqk_quantize.cpp` | 3594-3776 |
| the `IQ6_K` float search-table construction (in `iqk_quantize_iq6_k`) | `ggml/src/iqk/iqk_quantize.cpp` | 3790-3806 |
| `dequantize_row_iq4_k` | `ggml/src/iqk/iqk_quantize.cpp` | 2822-2849 |
| `dequantize_row_iq5_k` | `ggml/src/iqk/iqk_quantize.cpp` | 3112-3151 |
| `dequantize_row_iq6_k` | `ggml/src/iqk/iqk_quantize.cpp` | 3448-3491 |
| `dequantize_row_iq4_ks` | `ggml/src/iqk/iqk_quantize.cpp` | 4555-4575 |

Changes to those copied bodies, complete:

1. The ggml type-trait lookups become the literal row-size expressions:
   `4 + 136 * n/256` for `IQ4_KS` (the leading four bytes are its float32
   row scale), `144 * n/256` for `IQ4_K`, `176 * n/256` for `IQ5_K`, and
   `212 * n/256` for `IQ6_K`. The wrapper surface returns zero on a rejected
   geometry, so it surfaces to the caller instead of aborting a long
   conversion.
2. `quantize_user_data` is dropped: from `QHelper`'s constructor and members
   and from the four quantizer signatures. Upstream marks the parameter
   unused in all four, and none of the four members has an
   instruction-set-specific quantizer variant, so unlike `IQ2_KS` no path
   pinning is involved.
3. `GGML_FP32_TO_FP16` and `GGML_FP16_TO_FP32` become local functions with
   the same NEON body, falling back to `_Float16` where `__fp16` is absent.
4. Inside the copied bodies, `GGML_ASSERT` becomes plain `assert`
   (`quantize_row_iq4_k_impl_bs16`, `dequantize_row_iq4_ks`), the `MAX` and
   `MIN` macros become `std::max` and `std::min`, and the `static`
   qualifiers become anonymous-namespace placement.
5. Upstream's commented-out debug remnants are not carried: the disabled
   geometry assert and the `mse`/`printf` instrumentation in
   `quantize_row_iq4_k_impl_bs128`, the disabled `best_index` calls in
   `quantize_row_iq6_k_impl`, and the `//128` tail comment in
   `dequantize_row_iq4_ks`.
6. Declaration formatting is normalized (line wraps and spacing in the
   `quantize_row_iq5_k_impl` and `quantize_row_iq6_k_impl` signatures,
   `nearest_int`, and `QHelper`). No executable line differs beyond changes
   1-5.
7. The `IQ6_K` quantizer's float search table is built in the wrapper from
   `iq6nl_values` and `S_IQ6K` by the same loop upstream runs in
   `quantize_iq6_k` (`iqk_quantize.cpp:3790-3806`); the quantizer searches
   the float promotion of the integer table and only the dequantizer runs
   the cubic reconstruction.

`tests/test_vendor_upstream.py` covers the four
dense members through the same `libggml` cross-check as the routed members
and asserts the vendored quantizers and dequantizers reproduce upstream's
output bit for bit; it skips when no checkout is present.
