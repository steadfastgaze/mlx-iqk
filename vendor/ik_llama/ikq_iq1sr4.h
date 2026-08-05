// C entry points into the vendored ik_llama IQ1_S_R4 block codec.
//
// The functions below are thin extern "C" wrappers over code copied from
// ik_llama.cpp; see NOTICE and ATTRIBUTION.md at the repository root for the
// license and the copied-source inventory. Row geometry is asserted rather
// than adjusted: ik's own quantize driver rewrites the type when a row does
// not fit, which silently produces a differently shaped tensor, so this
// surface refuses such a row instead.
//
// The member's wire is four-row groups: rows are encoded and decoded four at
// a time, each group carrying a 4 x f16 row-scale prefix followed by 24-byte
// blocks that interleave 32 weights of each of the four rows. Row counts
// must therefore be multiples of 4 and widths multiples of 32.
//
// Sizes:
//   iq1_s_r4 row of n weights -> 2 + 6 * n/32 bytes (as ggml_row_size)

#ifndef IKQ_IQ1SR4_H
#define IKQ_IQ1SR4_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bytes one packed row occupies. Returns 0 for a rejected geometry.
size_t ikq_row_size_iq1_s_r4(int64_t n_per_row);

// Quantize nrows x n_per_row float32 weights. nrows must be a multiple of 4
// and n_per_row a multiple of 32. imatrix may be NULL (the unweighted
// fallback objective, which is a different objective, not a neutral one) or
// n_per_row floats reused per row. Returns bytes written, or 0 on a rejected
// geometry.
size_t ikq_quantize_iq1_s_r4(const float * src, void * dst, int64_t nrows,
                             int64_t n_per_row, const float * imatrix);

// Dequantize nrows packed rows into nrows * n_per_row float32 values.
// nrows must be a multiple of 4. Returns 0 on success, non-zero on a
// rejected geometry.
int ikq_dequantize_iq1_s_r4(const void * src, float * dst, int64_t nrows,
                            int64_t n_per_row);

// Struct size, so a caller can assert the build agrees with the wire spec.
size_t ikq_block_size_iq1_s_r4(void);

#ifdef __cplusplus
}
#endif

#endif  // IKQ_IQ1SR4_H
