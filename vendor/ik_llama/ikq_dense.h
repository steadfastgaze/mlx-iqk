// C entry points into the vendored ik_llama dense-member block codecs:
// IQ4_KS, IQ4_K, IQ5_K, IQ6_K.
//
// The functions below are thin extern "C" wrappers over code copied from
// ik_llama.cpp; see NOTICE and ATTRIBUTION.md at the repository root for the
// license and the copied-source inventory. Row geometry is asserted rather
// than adjusted: ik's own quantize driver rewrites the type when a row is not
// a multiple of 256, which silently produces a differently shaped tensor, so
// this surface refuses such a row instead.
//
// Sizes:
//   iq4_ks row of n weights -> 4 + 136 * n/256 bytes (fp32 row scale)
//   iq4_k  row of n weights -> 144 * n/256 bytes
//   iq5_k  row of n weights -> 176 * n/256 bytes
//   iq6_k  row of n weights -> 212 * n/256 bytes

#ifndef IKQ_DENSE_H
#define IKQ_DENSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bytes one packed row occupies. Returns 0 for a rejected geometry.
size_t ikq_row_size_iq4_ks(int64_t n_per_row);
size_t ikq_row_size_iq4_k(int64_t n_per_row);
size_t ikq_row_size_iq5_k(int64_t n_per_row);
size_t ikq_row_size_iq6_k(int64_t n_per_row);

// Quantize nrows x n_per_row float32 weights. imatrix may be NULL (the
// member-specific unweighted fallback) or n_per_row floats reused per row.
// Returns bytes written, or 0 on a rejected geometry.
size_t ikq_quantize_iq4_ks(const float * src, void * dst, int64_t nrows,
                           int64_t n_per_row, const float * imatrix);
size_t ikq_quantize_iq4_k(const float * src, void * dst, int64_t nrows,
                          int64_t n_per_row, const float * imatrix);
size_t ikq_quantize_iq5_k(const float * src, void * dst, int64_t nrows,
                          int64_t n_per_row, const float * imatrix);
size_t ikq_quantize_iq6_k(const float * src, void * dst, int64_t nrows,
                          int64_t n_per_row, const float * imatrix);

// Dequantize nrows packed rows into nrows * n_per_row float32 values.
// Returns 0 on success, non-zero on a rejected geometry. The IQ6_K path is
// the cubic-polynomial CPU reconstruction, which is this repository's
// serving reference for that member.
int ikq_dequantize_iq4_ks(const void * src, float * dst, int64_t nrows,
                          int64_t n_per_row);
int ikq_dequantize_iq4_k(const void * src, float * dst, int64_t nrows,
                         int64_t n_per_row);
int ikq_dequantize_iq5_k(const void * src, float * dst, int64_t nrows,
                         int64_t n_per_row);
int ikq_dequantize_iq6_k(const void * src, float * dst, int64_t nrows,
                         int64_t n_per_row);

// Struct sizes, so a caller can assert the build agrees with the wire spec.
size_t ikq_block_size_iq4_ks(void);
size_t ikq_block_size_iq4_k(void);
size_t ikq_block_size_iq5_k(void);
size_t ikq_block_size_iq6_k(void);

#ifdef __cplusplus
}
#endif

#endif  // IKQ_DENSE_H
