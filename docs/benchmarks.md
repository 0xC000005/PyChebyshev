# Benchmarks

## 5D Black-Scholes Performance

All benchmarks use 5D Black-Scholes: $V(S, K, T, \sigma, r)$ with 11 Chebyshev nodes per dimension ($11^5 = 161{,}051$ grid points).

### Accuracy

| Method | Price Error | Greek Error (max) |
|--------|-------------|-------------------|
| **Chebyshev Barycentric** | 0.000% | 1.980% |
| **MoCaX Standard (C++)** | 0.000% | 1.980% |
| **FDM (Crank-Nicolson)** | 0.803% | 2.234% |

Both Chebyshev methods achieve machine-precision price accuracy and identical Greek errors, as they compute the same unique interpolating polynomial.

### Timing

| Method | Build Time | Price Query | Price + 5 Greeks |
|--------|-----------|-------------|------------------|
| **Chebyshev Barycentric** | 0.35s | 0.065 ms | 0.29 ms |
| **MoCaX Standard (C++)** | 1.04s | 0.47 ms | 2.85 ms |
| **Analytical (direct)** | N/A | 0.01 ms | 0.06 ms |

### Evaluation Method Comparison

Within PyChebyshev, multiple evaluation paths exist:

| Method | Price Only | Price + 5 Greeks | Notes |
|--------|-----------|------------------|-------|
| `eval()` | ~45 ms | ~270 ms | Python loops, full validation |
| `fast_eval()` | ~10 ms | ~95 ms | **Deprecated** — JIT scalar loops |
| `vectorized_eval()` | 0.065 ms | 0.39 ms | **Recommended** — BLAS GEMV |
| `vectorized_eval_multi()` | — | 0.29 ms | Shared weights across derivatives |

`vectorized_eval()` is the recommended default. Use `vectorized_eval_multi()` when computing multiple derivatives at the same point.

!!! note "Why BLAS beats JIT"
    `fast_eval()` uses Numba JIT to compile scalar barycentric interpolation loops.
    But `vectorized_eval()` restructures the algorithm into matrix-vector products
    (BLAS GEMV), replacing 16,105 Python loop iterations with 5 BLAS calls for a 5D
    problem. Optimized BLAS (OpenBLAS/MKL) running a single GEMV is fundamentally
    faster than JIT-compiled scalar loops — the data access pattern is more
    cache-friendly and leverages SIMD vectorization at the hardware level.
    `fast_eval()` is deprecated and will be removed in a future version.

## Tensor Train (TT) vs MoCaX Extend

`ChebyshevTT` and MoCaX `MocaxExtend` both build Chebyshev interpolants in TT
format for the same 5D Black-Scholes problem. PyChebyshev uses TT-Cross (maxvol
pivoting); MoCaX uses rank-adaptive ALS on a random subgrid. Both use ~7,400--8,000
function evaluations.

### Build

| Metric | PyChebyshev TT | MoCaX TT |
|--------|----------------|----------|
| Build time | 0.35s | 5.73s |
| Function evaluations | 7,419 | 8,000 |
| TT ranks | [1, 11, 11, 11, 7, 1] | (not exposed) |
| Compression ratio | 43.4x | N/A |

### Price Accuracy (50 random test points)

| Metric | PyChebyshev TT | MoCaX TT |
|--------|----------------|----------|
| Mean error | 0.002% | 0.093% |
| Max error | 0.014% | 0.712% |
| Median error | 0.001% | 0.045% |

### Evaluation Speed (1000 random points)

| Method | PyChebyshev TT | MoCaX TT |
|--------|----------------|----------|
| Single eval | 0.065 ms | -- |
| Batch eval | 0.004 ms | 0.246 ms |

### Greeks Accuracy (10 scenarios, FD vs analytical)

| Greek | PyChebyshev avg error | MoCaX avg error |
|-------|-----------------------|-----------------|
| Delta | 0.029% | 0.379% |
| Gamma | 0.019% | 1.604% |

To reproduce: `uv run --with tqdm --with blackscholes python compare_tensor_train.py`
(requires MoCaX C++ library; PyChebyshev results are shown regardless).
