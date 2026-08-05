// Quantize and dequantize the vendored members through ik_llama's own
// libggml, so the vendored copies of those codecs can be checked against the
// source implementation rather than against themselves. No part of any codec
// is reimplemented here.
//
// usage: ikref_upstream <member> <mode> <in> <out> <nrows> <n_per_row> [imatrix]
//   member    iq2_k | iq2_ks | iq1_s_r4 | iq4_ks | iq4_k | iq5_k | iq6_k
//   mode      quantize | dequantize
//   quantize:   reads   nrows * n_per_row float32
//               writes  nrows * ggml_row_size bytes
//   dequantize: reads   nrows * ggml_row_size bytes
//               writes  nrows * n_per_row float32
//   imatrix   optional path to n_per_row float32 importance values

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"

static int read_all(const char *path, void *buf, size_t n) {
    FILE *fh = fopen(path, "rb");
    if (!fh) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    size_t got = fread(buf, 1, n, fh);
    fclose(fh);
    if (got != n) { fprintf(stderr, "short read %s\n", path); return 1; }
    return 0;
}

static int write_all(const char *path, const void *buf, size_t n) {
    FILE *fh = fopen(path, "wb");
    if (!fh) { fprintf(stderr, "cannot write %s\n", path); return 1; }
    size_t put = fwrite(buf, 1, n, fh);
    fclose(fh);
    if (put != n) { fprintf(stderr, "short write %s\n", path); return 1; }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 7 && argc != 8) {
        fprintf(stderr, "usage: %s <member> <mode> <in> <out> <nrows> <n_per_row>"
                        " [imatrix]\n", argv[0]);
        return 1;
    }
    enum ggml_type type;
    if (!strcmp(argv[1], "iq2_k")) {
        type = GGML_TYPE_IQ2_K;
    } else if (!strcmp(argv[1], "iq2_ks")) {
        type = GGML_TYPE_IQ2_KS;
    } else if (!strcmp(argv[1], "iq1_s_r4")) {
        type = GGML_TYPE_IQ1_S_R4;
    } else if (!strcmp(argv[1], "iq4_ks")) {
        type = GGML_TYPE_IQ4_KS;
    } else if (!strcmp(argv[1], "iq4_k")) {
        type = GGML_TYPE_IQ4_K;
    } else if (!strcmp(argv[1], "iq5_k")) {
        type = GGML_TYPE_IQ5_K;
    } else if (!strcmp(argv[1], "iq6_k")) {
        type = GGML_TYPE_IQ6_K;
    } else {
        fprintf(stderr, "unknown member %s\n", argv[1]);
        return 1;
    }
    const long nrows = atol(argv[5]);
    const long n_per_row = atol(argv[6]);
    const size_t row_size = ggml_row_size(type, n_per_row);
    const size_t nvals = (size_t)nrows * (size_t)n_per_row;

    float *imatrix = NULL;
    if (argc == 8) {
        imatrix = (float *)malloc((size_t)n_per_row * sizeof(float));
        if (read_all(argv[7], imatrix, (size_t)n_per_row * sizeof(float))) return 2;
    }

    if (!strcmp(argv[2], "quantize")) {
        float *src = (float *)malloc(nvals * sizeof(float));
        char *dst = (char *)malloc(row_size * (size_t)nrows);
        if (read_all(argv[3], src, nvals * sizeof(float))) return 2;
        // The portable IQ2_KS quantizer is what a NULL user_data selects, and
        // it is the only one this platform compiles anyway.
        size_t wrote = ggml_quantize_chunk(type, src, dst, 0, nrows, n_per_row,
                                           imatrix, NULL);
        if (wrote != row_size * (size_t)nrows) {
            fprintf(stderr, "quantize wrote %zu, expected %zu\n", wrote,
                    row_size * (size_t)nrows);
            return 2;
        }
        if (write_all(argv[4], dst, wrote)) return 2;
        printf("{\"row_size\": %zu, \"bytes\": %zu}\n", row_size, wrote);
        return 0;
    }
    if (!strcmp(argv[2], "dequantize")) {
        char *src = (char *)malloc(row_size * (size_t)nrows);
        float *dst = (float *)malloc(nvals * sizeof(float));
        if (read_all(argv[3], src, row_size * (size_t)nrows)) return 2;
        ggml_type_traits_t tr = ggml_internal_get_type_traits(type);
        // An _r4 member's to_float consumes a four-row group per call (its n
        // covers all four rows); a per-row loop over such a member decodes
        // garbage. Step by the member's row group.
        const long group = type == GGML_TYPE_IQ1_S_R4 ? 4 : 1;
        if (nrows % group) {
            fprintf(stderr, "nrows %ld is not a multiple of the %ld-row group\n",
                    nrows, group);
            return 2;
        }
        for (long r = 0; r < nrows; r += group) {
            tr.to_float(src + (size_t)r * row_size, dst + (size_t)r * n_per_row,
                        (int64_t)(group * n_per_row));
        }
        if (write_all(argv[4], dst, nvals * sizeof(float))) return 2;
        printf("{\"row_size\": %zu, \"values\": %zu}\n", row_size, nvals);
        return 0;
    }
    fprintf(stderr, "unknown mode %s\n", argv[2]);
    return 1;
}
