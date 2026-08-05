// Vendored IQ2_K and IQ2_KS block codec from ik_llama.cpp.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024 The ggml authors
// Copyright (c) 2023-2024 The llama.cpp authors
// Copyright (c) 2024-2025 The ik_llama.cpp authors
//
// Source: https://github.com/ikawrakow/ik_llama.cpp at commit
// b341d0b58b24d69a635bb81cdb907f9369c73dec.
//   ggml/src/ggml-common.h    block_iq2_k, block_iq2_ks, iq2nl_values
//   ggml/src/iqk/iqk_quantize.cpp
//                             nearest_int, QHelper::row_weights,
//                             make_qx_quants, best_index_iq2nl,
//                             quantize_row_iq2_k_impl, quantize_iq2_k,
//                             dequantize_row_iq2_k,
//                             quantize_row_iq2_ks_impl, quantize_iq2_ks,
//                             dequantize_row_iq2_ks
//   ggml/src/ggml-impl.h      the NEON fp16 conversion pair
//
// The copied bodies are byte-for-byte the upstream ones. What changed, and
// only this, is the surrounding scaffolding: the ggml type-trait lookups
// become the two literal row-size expressions, GGML_ASSERT becomes a return
// of zero so a rejected geometry surfaces to the caller instead of aborting a
// long conversion, and the AVX2 IQ2_KS quantizer is deliberately absent.
//
// The absent fast path is a pinning decision, not an omission. Upstream's
// AVX2 IQ2_KS quantizer searches a different candidate set, applies a
// per-sub-block RMSE refinement the portable path does not have, and closes
// with a 1.000 multiplier where the portable path closes with 1.030. The two
// produce different bytes for the same row. Conversion here is pinned to the
// portable path so packed bytes are reproducible independently of the host's
// instruction set.

#include "iqk_iq2.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#define QK_K 256

typedef uint16_t iqk_half;

// ggml-impl.h, the NEON pair. On any host without __fp16 the same rounding is
// obtained through the C++ conversion of the compiler's _Float16.
#if defined(__ARM_NEON) && !defined(_MSC_VER)
typedef __fp16 iqk_half_internal;
#else
typedef _Float16 iqk_half_internal;
#endif

static inline float iqk_fp16_to_fp32(iqk_half h) {
    iqk_half_internal tmp;
    memcpy(&tmp, &h, sizeof(iqk_half));
    return (float)tmp;
}

static inline iqk_half iqk_fp32_to_fp16(float f) {
    iqk_half res;
    iqk_half_internal tmp = f;
    memcpy(&res, &tmp, sizeof(iqk_half));
    return res;
}

// ggml-common.h
typedef struct {
    iqk_half d;
    uint16_t extra;
    uint8_t  scales[QK_K / 32];
    uint8_t  qs[QK_K / 4];
} block_iq2_k;
static_assert(sizeof(block_iq2_k) == sizeof(iqk_half) + sizeof(uint16_t) + QK_K / 32 + QK_K / 4,
              "wrong iq2_k block size/padding");

typedef struct {
    uint16_t extra;
    uint8_t  scales[QK_K / 64];
    uint8_t  qs[QK_K / 4];
} block_iq2_ks;
static_assert(sizeof(block_iq2_ks) == sizeof(uint16_t) + QK_K / 64 + QK_K / 4,
              "wrong iq2_ks block size/padding");

static const int8_t iq2nl_values[8] = { -31, -13, 1, 17, -26, -8, 6, 22 };

namespace {

// iqk_quantize.cpp:37-42
inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// iqk_quantize.cpp:46-92, the row_weights half of QHelper. The degeneracy
// guard is per row and per block: a block whose weight mass, value mass, or
// cross term falls to the epsilon floor collapses to a uniform 1e-9 weight,
// which is a different objective from the no-imatrix branch rather than a
// fallback to it.
struct QHelper {
    QHelper(const float * imatrix, int n_per_row, int block_size)
        : m_imatrix(imatrix), m_n_per_row(n_per_row), m_block_size(block_size) {
        if (m_imatrix) {
            m_weight.resize(m_n_per_row);
        }
    }
    const float * row_weights(const float * x) {
        constexpr float kEps  = 1e-9f;
        constexpr float kEps2 = kEps * kEps;
        if (!m_imatrix) return m_imatrix;
        int nblock = m_n_per_row / m_block_size;
        for (int ib = 0; ib < nblock; ++ib) {
            auto wb_in = m_imatrix + ib * m_block_size;
            auto xb = x + ib * m_block_size;
            auto wb = m_weight.data() + ib * m_block_size;
            float sumw2 = 0, sumx2 = 0, sumwx = 0;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = wb_in[j];
                sumw2 += wb[j] * wb[j];
                sumx2 += xb[j] * xb[j];
                sumwx += wb[j] * std::abs(xb[j]);
            }
            if (sumw2 > m_block_size * kEps2 && sumx2 > m_block_size * kEps2 &&
                sumwx > m_block_size * kEps2) continue;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = kEps;
            }
        }
        return m_weight.data();
    }
    template <typename Func>
    void quantize(int nrows, const float * src, void * dst, int row_size, const Func & qfunc) {
        auto cdst = (char *)dst;
        for (int row = 0; row < nrows; ++row) {
            auto weights = row_weights(src);
            qfunc(src, cdst, m_n_per_row, weights);
            src  += m_n_per_row;
            cdst += row_size;
        }
    }
private:
    const float * m_imatrix;
    const int m_n_per_row;
    const int m_block_size;
    std::vector<float> m_weight;
};

// iqk_quantize.cpp:113-155
float make_qx_quants(int n, int nmax, const float * x, int8_t * L, const float * qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (!amax) { // all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax - 1, l));
        L[i] = l + nmax;
        sumlx += qw[i] * x[i] * l;
        suml2 += qw[i] * l * l;
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax - 1, l));
            sumlx += qw[i] * x[i] * l;
            suml2 += qw[i] * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + std::max(-nmax, std::min(nmax - 1, l));
            }
            scale = sumlx / suml2; best = scale * sumlx;
        }
    }
    return scale;
}

// iqk_quantize.cpp:1185-1188
inline int best_index_iq2nl(const int8_t * values, float x) {
    int idx = x < values[1] ? 0 : x > values[2] ? 2 : 1;
    return x - values[idx] < values[idx + 1] - x ? idx : idx + 1;
}

// iqk_quantize.cpp:1190-1333
void quantize_row_iq2_k_impl(const float * x, void * vy, int n_per_row,
                             const float * quant_weights) {

    constexpr int kBlockSize = 16;

    block_iq2_k * y = (block_iq2_k *)vy;

    float scales[QK_K / kBlockSize];
    float weight[kBlockSize];
    float sumx[kBlockSize + 1], sumw[kBlockSize + 1];
    float sw[QK_K / kBlockSize];
    int8_t Ls[QK_K / kBlockSize];

    std::array<std::pair<float, int>, kBlockSize> pairs;

    const int8_t * shifted_values = iq2nl_values + 4;

    for (int ibl = 0; ibl < n_per_row / QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_k));
        y[ibl].d = iqk_fp32_to_fp16(0.f);

        const float * xbl = x + ibl * QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j] * xbl[j];
        const float sigma2 = 1.5f * sumx2 / QK_K;

        uint16_t extra = 0;

        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K / kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize * ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl * QK_K + ib * kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f * sigma2 + xb[j] * xb[j];
            }
            sw[ib] = 0;
            float amax = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                sw[ib] += weight[j];
                pairs[j] = {xb[j], j};
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            std::sort(pairs.begin(), pairs.end());
            sumx[0] = sumw[0] = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                int jj = pairs[j].second;
                sumw[j + 1] = sumw[j] + weight[jj];
                sumx[j + 1] = sumx[j] + weight[jj] * xb[jj];
            }
            float best = 0, d = 0;
            bool is_shifted = false;
            float sumqx, sumq2;
            for (int i1 = 0; i1 < kBlockSize; ++i1) {
                for (int i2 = i1; i2 < kBlockSize; ++i2) {
                    for (int i3 = i2; i3 < kBlockSize; ++i3) {
                        sumqx = (sumx[i1] - sumx[ 0]) * iq2nl_values[0] + (sumx[i2] - sumx[i1]) * iq2nl_values[1]
                              + (sumx[i3] - sumx[i2]) * iq2nl_values[2] + (sumx[kBlockSize] - sumx[i3]) * iq2nl_values[3];
                        sumq2 = (sumw[i1] - sumw[ 0]) * iq2nl_values[0] * iq2nl_values[0] + (sumw[i2] - sumw[i1]) * iq2nl_values[1] * iq2nl_values[1]
                              + (sumw[i3] - sumw[i2]) * iq2nl_values[2] * iq2nl_values[2] + (sumw[kBlockSize] - sumw[i3]) * iq2nl_values[3] * iq2nl_values[3];
                        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                            d = sumqx / sumq2; best = d * sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0]) * shifted_values[0] + (sumx[i2] - sumx[i1]) * shifted_values[1]
                              + (sumx[i3] - sumx[i2]) * shifted_values[2] + (sumx[kBlockSize] - sumx[i3]) * shifted_values[3];
                        sumq2 = (sumw[i1] - sumw[ 0]) * shifted_values[0] * shifted_values[0] + (sumw[i2] - sumw[i1]) * shifted_values[1] * shifted_values[1]
                              + (sumw[i3] - sumw[i2]) * shifted_values[2] * shifted_values[2] + (sumw[kBlockSize] - sumw[i3]) * shifted_values[3] * shifted_values[3];
                        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                            d = sumqx / sumq2; best = d * sumqx; is_shifted = true;
                        }
                        sumqx = (sumx[i1] - sumx[ 0]) * iq2nl_values[3] + (sumx[i2] - sumx[i1]) * iq2nl_values[2]
                              + (sumx[i3] - sumx[i2]) * iq2nl_values[1] + (sumx[kBlockSize] - sumx[i3]) * iq2nl_values[0];
                        sumq2 = (sumw[i1] - sumw[ 0]) * iq2nl_values[3] * iq2nl_values[3] + (sumw[i2] - sumw[i1]) * iq2nl_values[2] * iq2nl_values[2]
                              + (sumw[i3] - sumw[i2]) * iq2nl_values[1] * iq2nl_values[1] + (sumw[kBlockSize] - sumw[i3]) * iq2nl_values[0] * iq2nl_values[0];
                        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                            d = sumqx / sumq2; best = d * sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0]) * shifted_values[3] + (sumx[i2] - sumx[i1]) * shifted_values[2]
                              + (sumx[i3] - sumx[i2]) * shifted_values[1] + (sumx[kBlockSize] - sumx[i3]) * shifted_values[0];
                        sumq2 = (sumw[i1] - sumw[ 0]) * shifted_values[3] * shifted_values[3] + (sumw[i2] - sumw[i1]) * shifted_values[2] * shifted_values[2]
                              + (sumw[i3] - sumw[i2]) * shifted_values[1] * shifted_values[1] + (sumw[kBlockSize] - sumw[i3]) * shifted_values[0] * shifted_values[0];
                        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                            d = sumqx / sumq2; best = d * sumqx; is_shifted = true;
                        }
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = std::max(max_abs_scale, abs_scale);
        }

        if (!max_abs_scale) continue;
        float d = make_qx_quants(QK_K / kBlockSize, 8, scales, Ls, sw);
        if (!d) continue;

        y[ibl].extra = extra;
        float id = 1 / d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K / kBlockSize; ++ib) {
            int ls = nearest_int(id * scales[ib]);
            ls = std::max(-8, std::min(7, ls));
            y[ibl].scales[ib / 2] |= ((ls + 8) << 4 * (ib % 2));
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
                const float * xb = xbl + kBlockSize * ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl * QK_K + ib * kBlockSize;
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
                } else {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f * sigma2 + xb[j] * xb[j];
                }
                float idl = 1 / dl;
                int ib32 = ib / 2;
                int offset = 16 * (ib % 2);
                uint8_t * qs = y[ibl].qs + 32 * (ib32 / 4) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl * xb[j];
                    int ibest = best_index_iq2nl(block_values, al);
                    qs[j] |= (ibest << 2 * (ib32 % 4));
                    float w = weight[j];
                    float q = block_values[ibest] * ls;
                    sumqx += w * q * xb[j];
                    sumq2 += w * q * q;
                }
            }
        }
        y[ibl].d = iqk_fp32_to_fp16(1.030f * (sumq2 > 0 ? sumqx / sumq2 : d));
    }
}

// iqk_quantize.cpp:1692-1840, the portable IQ2_KS quantizer.
void quantize_row_iq2_ks_impl(const float * x, void * vy, int n_per_row,
                              const float * quant_weights, float * all_scales,
                              float * all_sw, int8_t * all_Ls) {

    constexpr int kBlockSize = 32;
    constexpr int kMax_i1 = 3 * kBlockSize / 4;
    constexpr int kMin_i3 = kBlockSize / 4;

    iqk_half * dptr = (iqk_half *)vy;
    *dptr = iqk_fp32_to_fp16(0.f);

    block_iq2_ks * y = (block_iq2_ks *)(dptr + 1);

    float weight[kBlockSize];
    float sumx[kBlockSize + 1], sumw[kBlockSize + 1];

    std::array<std::pair<float, int>, kBlockSize> pairs;

    float val [4] = {float(iq2nl_values[0]), float(iq2nl_values[1]), float(iq2nl_values[2]), float(iq2nl_values[3])};
    float sval[4] = {float(iq2nl_values[4]), float(iq2nl_values[5]), float(iq2nl_values[6]), float(iq2nl_values[7])};

    const int8_t * shifted_values = iq2nl_values + 4;

    const int nblock = n_per_row / QK_K;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_ks));

        auto scales = all_scales + ibl * (QK_K / kBlockSize);
        auto sw = all_sw + ibl * (QK_K / kBlockSize);

        const float * xbl = x + ibl * QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j] * xbl[j];
        const float sigma2 = 1.5f * sumx2 / QK_K;

        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K / kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize * ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl * QK_K + ib * kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f * sigma2 + xb[j] * xb[j];
            }
            sw[ib] = 0;
            float amax = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                sw[ib] += weight[j];
                pairs[j] = {xb[j], j};
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            std::sort(pairs.begin(), pairs.end());
            sumx[0] = sumw[0] = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                int jj = pairs[j].second;
                sumw[j + 1] = sumw[j] + weight[jj];
                sumx[j + 1] = sumx[j] + weight[jj] * xb[jj];
            }
            float best = 0, d = 0;
            bool is_shifted = false;
            float sumqx, sumq2;
            for (int i1 = 0; i1 < kMax_i1; ++i1) {
                for (int i2 = i1; i2 < kBlockSize; ++i2) {
                    for (int i3 = std::max(i2, kMin_i3); i3 < kBlockSize; ++i3) {
                        sumqx = (sumx[i1] - sumx[ 0]) * val[0] + (sumx[i2] - sumx[i1]) * val[1]
                              + (sumx[i3] - sumx[i2]) * val[2] + (sumx[kBlockSize] - sumx[i3]) * val[3];
                        sumq2 = (sumw[i1] - sumw[ 0]) * val[0] * val[0] + (sumw[i2] - sumw[i1]) * val[1] * val[1]
                              + (sumw[i3] - sumw[i2]) * val[2] * val[2] + (sumw[kBlockSize] - sumw[i3]) * val[3] * val[3];
                        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                            d = sumqx / sumq2; best = d * sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0]) * sval[0] + (sumx[i2] - sumx[i1]) * sval[1]
                              + (sumx[i3] - sumx[i2]) * sval[2] + (sumx[kBlockSize] - sumx[i3]) * sval[3];
                        sumq2 = (sumw[i1] - sumw[ 0]) * sval[0] * sval[0] + (sumw[i2] - sumw[i1]) * sval[1] * sval[1]
                              + (sumw[i3] - sumw[i2]) * sval[2] * sval[2] + (sumw[kBlockSize] - sumw[i3]) * sval[3] * sval[3];
                        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                            d = sumqx / sumq2; best = d * sumqx; is_shifted = true;
                        }
                        sumqx = (sumx[i1] - sumx[ 0]) * val[3] + (sumx[i2        ] - sumx[i1]) * val[2]
                              + (sumx[i3] - sumx[i2]) * val[1] + (sumx[kBlockSize] - sumx[i3]) * val[0];
                        sumq2 = (sumw[i1] - sumw[ 0]) * val[3] * val[3] + (sumw[i2        ] - sumw[i1]) * val[2] * val[2]
                              + (sumw[i3] - sumw[i2]) * val[1] * val[1] + (sumw[kBlockSize] - sumw[i3]) * val[0] * val[0];
                        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                            d = sumqx / sumq2; best = d * sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0]) * sval[3] + (sumx[i2        ] - sumx[i1]) * sval[2]
                              + (sumx[i3] - sumx[i2]) * sval[1] + (sumx[kBlockSize] - sumx[i3]) * sval[0];
                        sumq2 = (sumw[i1] - sumw[ 0]) * sval[3] * sval[3] + (sumw[i2        ] - sumw[i1]) * sval[2] * sval[2]
                              + (sumw[i3] - sumw[i2]) * sval[1] * sval[1] + (sumw[kBlockSize] - sumw[i3]) * sval[0] * sval[0];
                        if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                            d = sumqx / sumq2; best = d * sumqx; is_shifted = true;
                        }
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
        }
        y[ibl].extra = extra;
    }

    float d = make_qx_quants(nblock * (QK_K / kBlockSize), 16, all_scales, all_Ls, all_sw);

    if (!d) return;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto xbl = x + ibl * QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j] * xbl[j];
        const float sigma2 = 1.5f * sumx2 / QK_K;
        auto Ls = all_Ls + ibl * (QK_K / kBlockSize);
        for (int ib = 0; ib < QK_K / kBlockSize; ++ib) {
            int ls = Ls[ib];
            y[ibl].scales[ib / 2] |= ((ls & 0xf) << 4 * (ib % 2));
            y[ibl].extra |= ((ls >> 4) << (8 + ib));
            ls -= 16;
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
                const float * xb = xbl + kBlockSize * ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl * QK_K + ib * kBlockSize;
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
                } else {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f * sigma2 + xb[j] * xb[j];
                }
                float idl = 1 / dl;
                uint8_t * qs = y[ibl].qs + 32 * (ib / 4);
                for (int j = 0; j < 32; ++j) {
                    const float al = idl * xb[j];
                    int ibest = best_index_iq2nl(block_values, al);
                    qs[j] |= (ibest << 2 * (ib % 4));
                    float w = weight[j];
                    float q = block_values[ibest] * ls;
                    sumqx += w * q * xb[j];
                    sumq2 += w * q * q;
                }
            }
        }
    }
    *dptr = iqk_fp32_to_fp16(1.030f * (sumq2 > 0 ? sumqx / sumq2 : d));
}

// iqk_quantize.cpp:1356-1385
void dequantize_row_iq2_k(const block_iq2_k * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = iqk_fp16_to_fp32(x[i].d);
        const uint8_t * qs = x[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            float dl1 = d * ((x[i].scales[ib32] & 0xf) - 8);
            float dl2 = d * ((x[i].scales[ib32] >>  4) - 8);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j +  0] = dl1 * values1[(qs[j +  0] >> shift) & 3];
                y[j + 16] = dl2 * values2[(qs[j + 16] >> shift) & 3];
            }
            y += 32;
            shift += 2;
            if (shift == 8) { qs += 32; shift = 0; }
        }
    }
}

// iqk_quantize.cpp:1877-1909
void dequantize_row_iq2_ks(const block_iq2_ks * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const iqk_half * dptr = (const iqk_half *)x;
    const float d = iqk_fp16_to_fp32(*dptr);
    x = (const block_iq2_ks *)(dptr + 1);

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K / 64; ++ib64) {
            float dl1 = d * (((x[i].scales[ib64] & 0xf) | ((extra >> 4) & 0x10)) - 16);
            float dl2 = d * (((x[i].scales[ib64] >>  4) | ((extra >> 5) & 0x10)) - 16);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            for (int j = 0; j < 32; ++j) {
                y[j +  0] = dl1 * values1[(qs[j] >> (shift + 0)) & 3];
                y[j + 32] = dl2 * values2[(qs[j] >> (shift + 2)) & 3];
            }
            y += 64;
            shift += 4;
            if (shift == 8) { qs += 32; shift = 0; }
        }
    }
}

inline bool geometry_ok(int64_t n_per_row) {
    return n_per_row > 0 && n_per_row % QK_K == 0;
}

}  // namespace

extern "C" {

size_t iqk_row_size_iq2_k(int64_t n_per_row) {
    if (!geometry_ok(n_per_row)) return 0;
    return sizeof(block_iq2_k) * (size_t)(n_per_row / QK_K);
}

size_t iqk_row_size_iq2_ks(int64_t n_per_row) {
    if (!geometry_ok(n_per_row)) return 0;
    return sizeof(iqk_half) + sizeof(block_iq2_ks) * (size_t)(n_per_row / QK_K);
}

size_t iqk_quantize_iq2_k(const float * src, void * dst, int64_t nrows,
                          int64_t n_per_row, const float * imatrix) {
    const size_t row_size = iqk_row_size_iq2_k(n_per_row);
    if (!row_size || nrows <= 0) return 0;
    QHelper helper(imatrix, (int)n_per_row, 16);
    helper.quantize((int)nrows, src, dst, (int)row_size, quantize_row_iq2_k_impl);
    return (size_t)nrows * row_size;
}

size_t iqk_quantize_iq2_ks(const float * src, void * dst, int64_t nrows,
                           int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    const size_t row_size = iqk_row_size_iq2_ks(n_per_row);
    if (!row_size || nrows <= 0) return 0;
    const int nblock = (int)(n_per_row / QK_K);
    std::vector<float> all_scales(nblock * (QK_K / kBlockSize));
    std::vector<float> all_sw(nblock * (QK_K / kBlockSize));
    std::vector<int8_t> all_Ls(nblock * (QK_K / kBlockSize));
    auto q_func = [&all_scales, &all_sw, &all_Ls](const float * x, void * vy,
                                                  int n, const float * qw) {
        quantize_row_iq2_ks_impl(x, vy, n, qw, all_scales.data(), all_sw.data(),
                                 all_Ls.data());
    };
    QHelper helper(imatrix, (int)n_per_row, kBlockSize);
    helper.quantize((int)nrows, src, dst, (int)row_size, q_func);
    return (size_t)nrows * row_size;
}

int iqk_dequantize_iq2_k(const void * src, float * dst, int64_t nrows,
                         int64_t n_per_row) {
    const size_t row_size = iqk_row_size_iq2_k(n_per_row);
    if (!row_size || nrows < 0) return 1;
    const char * s = (const char *)src;
    for (int64_t r = 0; r < nrows; ++r) {
        dequantize_row_iq2_k((const block_iq2_k *)(s + (size_t)r * row_size),
                             dst + r * n_per_row, n_per_row);
    }
    return 0;
}

int iqk_dequantize_iq2_ks(const void * src, float * dst, int64_t nrows,
                          int64_t n_per_row) {
    const size_t row_size = iqk_row_size_iq2_ks(n_per_row);
    if (!row_size || nrows < 0) return 1;
    const char * s = (const char *)src;
    for (int64_t r = 0; r < nrows; ++r) {
        dequantize_row_iq2_ks((const block_iq2_ks *)(s + (size_t)r * row_size),
                              dst + r * n_per_row, n_per_row);
    }
    return 0;
}

size_t iqk_block_size_iq2_k(void) { return sizeof(block_iq2_k); }
size_t iqk_block_size_iq2_ks(void) { return sizeof(block_iq2_ks); }

}  // extern "C"
