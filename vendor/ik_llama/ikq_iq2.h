// C entry points into the vendored ik_llama IQ2_K / IQ2_KS block codec.
//
// The functions below are thin extern "C" wrappers over code copied from
// ik_llama.cpp; see NOTICE and ATTRIBUTION.md at the repository root for the
// license and the copied-source inventory. Row geometry is asserted rather
// than adjusted: ik's own quantize driver rewrites the type when a row is not
// a multiple of 256, which silently produces a differently shaped tensor, so
// this surface refuses such a row instead.
//
// Sizes:
//   iq2_k  row of n weights -> 76 * n/256 bytes
//   iq2_ks row of n weights -> 2 + 70 * n/256 bytes

#ifndef IKQ_IQ2_H
#define IKQ_IQ2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bytes one packed row occupies. Returns 0 for a rejected geometry.
size_t ikq_row_size_iq2_k(int64_t n_per_row);
size_t ikq_row_size_iq2_ks(int64_t n_per_row);

// Quantize nrows x n_per_row float32 weights. imatrix may be NULL (the
// member-specific unweighted fallback) or n_per_row floats reused per row.
// Returns bytes written, or 0 on a rejected geometry.
size_t ikq_quantize_iq2_k(const float * src, void * dst, int64_t nrows,
                          int64_t n_per_row, const float * imatrix);
size_t ikq_quantize_iq2_ks(const float * src, void * dst, int64_t nrows,
                           int64_t n_per_row, const float * imatrix);

// Dequantize nrows packed rows into nrows * n_per_row float32 values.
// Returns 0 on success, non-zero on a rejected geometry.
int ikq_dequantize_iq2_k(const void * src, float * dst, int64_t nrows,
                         int64_t n_per_row);
int ikq_dequantize_iq2_ks(const void * src, float * dst, int64_t nrows,
                          int64_t n_per_row);

// Struct sizes, so a caller can assert the build agrees with the wire spec.
size_t ikq_block_size_iq2_k(void);
size_t ikq_block_size_iq2_ks(void);

#ifdef __cplusplus
}
#endif

#endif  // IKQ_IQ2_H
