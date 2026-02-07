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

| Method | Price Only | Price + 5 Greeks |
|--------|-----------|------------------|
| `eval()` | ~45 ms | ~270 ms |
| `fast_eval()` | ~10 ms | ~95 ms |
| `vectorized_eval()` | 0.065 ms | 0.39 ms |
| `vectorized_eval_multi()` | — | 0.29 ms |

`vectorized_eval()` is the recommended default. Use `vectorized_eval_multi()` when computing multiple derivatives at the same point.
