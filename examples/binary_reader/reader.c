/*
 * PyChebyshev .pcb binary format reader (reference implementation).
 *
 * Reads a v1 ChebyshevApproximation file and evaluates it at a query
 * point. Spline support is intentionally omitted — this exists to prove
 * the format is implementable from scratch in another language.
 *
 * Build:   make
 * Usage:   ./reader model.pcb x0 x1 ... xd-1
 */

#define _USE_MATH_DEFINES  /* MSVC */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAGIC "PCB\x00"
#define MAJOR 1
#define CLASS_TAG_APPROX 1

/* Fail with a message, free already-allocated memory, and exit 1. */
#define READ_OR_FAIL(call, msg) \
    do { if ((call) != 0) { fprintf(stderr, "EOF reading %s\n", (msg)); goto cleanup; } } while (0)

static int read_exact(FILE *f, void *buf, size_t n) {
    return fread(buf, 1, n, f) == n ? 0 : -1;
}

static int read_u32(FILE *f, uint32_t *out) {
    return read_exact(f, out, 4);
}

static double *read_f64_array(FILE *f, size_t count) {
    double *arr = (double *)malloc(count * sizeof(double));
    if (!arr) return NULL;
    if (read_exact(f, arr, count * sizeof(double)) != 0) {
        free(arr);
        return NULL;
    }
    return arr;
}

/* Chebyshev first-kind nodes scaled to [a, b], sorted ascending.
 * Matches PyChebyshev's _make_nodes_for_dim. */
static void make_nodes(double a, double b, uint32_t n, double *out) {
    /* chebpts1 (Type 1, first-kind nodes): x_k = cos((2k+1)pi / (2n)) */
    for (uint32_t k = 0; k < n; k++) {
        double theta = (2.0 * k + 1.0) * M_PI / (2.0 * n);
        double std = cos(theta);
        out[k] = 0.5 * (a + b) + 0.5 * (b - a) * std;
    }
    /* Insertion sort — n is small (typically <= 64). */
    for (uint32_t i = 1; i < n; i++) {
        double v = out[i];
        int j = (int)i - 1;
        while (j >= 0 && out[j] > v) { out[j+1] = out[j]; j--; }
        out[j+1] = v;
    }
}

/* Barycentric weights from node positions: w_k = 1 / prod(nodes[k] - nodes[j], j != k).
 * Matches PyChebyshev's compute_barycentric_weights. */
static void compute_weights(const double *nodes, uint32_t n, double *w) {
    for (uint32_t k = 0; k < n; k++) {
        double prod = 1.0;
        for (uint32_t j = 0; j < n; j++) {
            if (j != k) prod *= (nodes[k] - nodes[j]);
        }
        w[k] = 1.0 / prod;
    }
}

/* 1-D barycentric interpolation. */
static double bary_1d(double x, const double *nodes, const double *vals,
                      const double *w, uint32_t n) {
    /* Check exact-node match first to avoid 0/0. */
    for (uint32_t k = 0; k < n; k++) {
        if (x == nodes[k]) return vals[k];
    }
    double num = 0.0, den = 0.0;
    for (uint32_t k = 0; k < n; k++) {
        double diff = x - nodes[k];
        double t = w[k] / diff;
        num += t * vals[k];
        den += t;
    }
    return num / den;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s file.pcb x0 [x1 ...]\n", argv[0]);
        return 2;
    }

    /* Declare all pointers here so the cleanup label can free them safely.
     * Initialize to NULL so free(NULL) is harmless for any not-yet-allocated
     * pointer. */
    FILE *f = NULL;
    double *lo = NULL;
    double *hi = NULL;
    uint32_t *n_nodes = NULL;
    double *vals = NULL;
    double **nodes_arr = NULL;
    double **w_arr = NULL;
    double *query = NULL;
    double *current = NULL;
    uint32_t d = 0;

    f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }

    char magic[4];
    if (read_exact(f, magic, 4) != 0 || memcmp(magic, MAGIC, 4) != 0) {
        fprintf(stderr, "not a .pcb file\n"); fclose(f); return 1;
    }
    uint8_t major, minor;
    READ_OR_FAIL(read_exact(f, &major, 1), "major version");
    READ_OR_FAIL(read_exact(f, &minor, 1), "minor version");
    if (major != MAJOR) {
        fprintf(stderr, "unsupported major %u\n", major);
        goto cleanup;
    }
    uint16_t class_tag;
    READ_OR_FAIL(read_exact(f, &class_tag, 2), "class_tag");
    if (class_tag != CLASS_TAG_APPROX) {
        fprintf(stderr, "this reader handles ChebyshevApproximation only "
                        "(class_tag=%u)\n", class_tag);
        goto cleanup;
    }
    uint8_t reserved[4];
    READ_OR_FAIL(read_exact(f, reserved, 4), "reserved bytes");
    for (int i = 0; i < 4; i++) {
        if (reserved[i] != 0) {
            fprintf(stderr, "reserved bytes nonzero\n");
            goto cleanup;
        }
    }

    READ_OR_FAIL(read_u32(f, &d), "num_dimensions");
    if ((int)d != argc - 2) {
        fprintf(stderr, "file is %u-D but %d query coords given\n",
                d, argc - 2);
        goto cleanup;
    }

    lo = read_f64_array(f, d);
    if (!lo) { fprintf(stderr, "EOF reading domain_lo\n"); goto cleanup; }
    hi = read_f64_array(f, d);
    if (!hi) { fprintf(stderr, "EOF reading domain_hi\n"); goto cleanup; }

    n_nodes = (uint32_t *)malloc(d * sizeof(uint32_t));
    if (!n_nodes) { fprintf(stderr, "out of memory\n"); goto cleanup; }
    for (uint32_t i = 0; i < d; i++) {
        READ_OR_FAIL(read_u32(f, &n_nodes[i]), "n_nodes[i]");
    }

    {
        size_t total = 1;
        for (uint32_t i = 0; i < d; i++) total *= n_nodes[i];
        vals = read_f64_array(f, total);
        if (!vals) { fprintf(stderr, "EOF reading tensor_values\n"); goto cleanup; }

        /* Reconstruct nodes and weights per dim. */
        nodes_arr = (double **)calloc(d, sizeof(double *));
        w_arr = (double **)calloc(d, sizeof(double *));
        if (!nodes_arr || !w_arr) { fprintf(stderr, "out of memory\n"); goto cleanup; }
        for (uint32_t i = 0; i < d; i++) {
            nodes_arr[i] = (double *)malloc(n_nodes[i] * sizeof(double));
            w_arr[i] = (double *)malloc(n_nodes[i] * sizeof(double));
            if (!nodes_arr[i] || !w_arr[i]) {
                fprintf(stderr, "out of memory\n"); goto cleanup;
            }
            make_nodes(lo[i], hi[i], n_nodes[i], nodes_arr[i]);
            compute_weights(nodes_arr[i], n_nodes[i], w_arr[i]);
        }

        /* Parse query point. */
        query = (double *)malloc(d * sizeof(double));
        if (!query) { fprintf(stderr, "out of memory\n"); goto cleanup; }
        for (uint32_t i = 0; i < d; i++) query[i] = atof(argv[2 + i]);

        /* Evaluate by collapsing the highest dim down to scalar. */
        current = (double *)malloc(total * sizeof(double));
        if (!current) { fprintf(stderr, "out of memory\n"); goto cleanup; }
        memcpy(current, vals, total * sizeof(double));
        size_t cur_size = total;
        for (int dim = (int)d - 1; dim >= 0; dim--) {
            size_t nk = n_nodes[dim];
            size_t outer = cur_size / nk;
            double *next = (double *)malloc(outer * sizeof(double));
            if (!next) { fprintf(stderr, "out of memory\n"); goto cleanup; }
            double *slice_vals = (double *)malloc(nk * sizeof(double));
            if (!slice_vals) {
                fprintf(stderr, "out of memory\n");
                free(next);
                goto cleanup;
            }
            for (size_t o = 0; o < outer; o++) {
                for (size_t k = 0; k < nk; k++) {
                    slice_vals[k] = current[o * nk + k];
                }
                next[o] = bary_1d(query[dim], nodes_arr[dim], slice_vals,
                                  w_arr[dim], (uint32_t)nk);
            }
            free(slice_vals);
            free(current);
            current = next;
            cur_size = outer;
        }
    }

    printf("%.17g\n", current[0]);
    fclose(f);

    /* Normal cleanup. */
    free(current); free(query); free(vals); free(n_nodes);
    free(lo); free(hi);
    for (uint32_t i = 0; i < d; i++) { free(nodes_arr[i]); free(w_arr[i]); }
    free(nodes_arr); free(w_arr);
    return 0;

cleanup:
    if (f) fclose(f);
    free(current); free(query); free(vals); free(n_nodes);
    free(lo); free(hi);
    if (nodes_arr) {
        for (uint32_t i = 0; i < d; i++) { free(nodes_arr[i]); }
        free(nodes_arr);
    }
    if (w_arr) {
        for (uint32_t i = 0; i < d; i++) { free(w_arr[i]); }
        free(w_arr);
    }
    return 1;
}
