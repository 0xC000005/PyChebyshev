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

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }

    char magic[4];
    if (read_exact(f, magic, 4) || memcmp(magic, MAGIC, 4) != 0) {
        fprintf(stderr, "not a .pcb file\n"); fclose(f); return 1;
    }
    uint8_t major, minor;
    read_exact(f, &major, 1);
    read_exact(f, &minor, 1);
    if (major != MAJOR) {
        fprintf(stderr, "unsupported major %u\n", major);
        fclose(f); return 1;
    }
    uint16_t class_tag;
    read_exact(f, &class_tag, 2);
    if (class_tag != CLASS_TAG_APPROX) {
        fprintf(stderr, "this reader handles ChebyshevApproximation only "
                        "(class_tag=%u)\n", class_tag);
        fclose(f); return 1;
    }
    uint8_t reserved[4];
    read_exact(f, reserved, 4);
    for (int i = 0; i < 4; i++) {
        if (reserved[i] != 0) {
            fprintf(stderr, "reserved bytes nonzero\n");
            fclose(f); return 1;
        }
    }

    uint32_t d;
    read_u32(f, &d);
    if ((int)d != argc - 2) {
        fprintf(stderr, "file is %u-D but %d query coords given\n",
                d, argc - 2);
        fclose(f); return 1;
    }

    double *lo = read_f64_array(f, d);
    double *hi = read_f64_array(f, d);
    uint32_t *n_nodes = (uint32_t *)malloc(d * sizeof(uint32_t));
    for (uint32_t i = 0; i < d; i++) read_u32(f, &n_nodes[i]);

    size_t total = 1;
    for (uint32_t i = 0; i < d; i++) total *= n_nodes[i];
    double *vals = read_f64_array(f, total);

    /* Reconstruct nodes and weights per dim. */
    double **nodes_arr = (double **)malloc(d * sizeof(double *));
    double **w_arr = (double **)malloc(d * sizeof(double *));
    for (uint32_t i = 0; i < d; i++) {
        nodes_arr[i] = (double *)malloc(n_nodes[i] * sizeof(double));
        w_arr[i] = (double *)malloc(n_nodes[i] * sizeof(double));
        make_nodes(lo[i], hi[i], n_nodes[i], nodes_arr[i]);
        compute_weights(nodes_arr[i], n_nodes[i], w_arr[i]);
    }

    /* Parse query point. */
    double *query = (double *)malloc(d * sizeof(double));
    for (uint32_t i = 0; i < d; i++) query[i] = atof(argv[2 + i]);

    /* Evaluate by collapsing the highest dim down to scalar. */
    double *current = (double *)malloc(total * sizeof(double));
    memcpy(current, vals, total * sizeof(double));
    size_t cur_size = total;
    for (int dim = (int)d - 1; dim >= 0; dim--) {
        size_t nk = n_nodes[dim];
        size_t outer = cur_size / nk;
        double *next = (double *)malloc(outer * sizeof(double));
        for (size_t o = 0; o < outer; o++) {
            double slice_vals[64];  /* assumes n <= 64 — fine for v0.14 */
            for (size_t k = 0; k < nk; k++) {
                slice_vals[k] = current[o * nk + k];
            }
            next[o] = bary_1d(query[dim], nodes_arr[dim], slice_vals,
                              w_arr[dim], (uint32_t)nk);
        }
        free(current);
        current = next;
        cur_size = outer;
    }

    printf("%.17g\n", current[0]);

    /* Cleanup. */
    free(current); free(query); free(vals); free(n_nodes);
    free(lo); free(hi);
    for (uint32_t i = 0; i < d; i++) { free(nodes_arr[i]); free(w_arr[i]); }
    free(nodes_arr); free(w_arr);
    fclose(f);
    return 0;
}
