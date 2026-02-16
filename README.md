# PyChebyshev

[![PyPI version](https://img.shields.io/pypi/v/pychebyshev)](https://pypi.org/project/pychebyshev/)
[![Python versions](https://img.shields.io/pypi/pyversions/pychebyshev)](https://pypi.org/project/pychebyshev/)
[![Tests](https://github.com/0xC000005/PyChebyshev/actions/workflows/test.yml/badge.svg)](https://github.com/0xC000005/PyChebyshev/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/0xC000005/PyChebyshev/graph/badge.svg)](https://codecov.io/gh/0xC000005/PyChebyshev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A fast Python library for multi-dimensional Chebyshev tensor interpolation with analytical derivatives**

PyChebyshev provides a **fully optimized Python implementation** of the Chebyshev tensor method, demonstrating that it achieves **comparable speed and accuracy to the state-of-the-art MoCaX C++ library** for European option pricing via Black-Scholes — with pure Python and NumPy only.

## Performance Comparison

Based on standardized 5D Black-Scholes tests (`test_5d_black_scholes()` with S, K, T, σ, r):

| Method | Price Error | Greek Error | Build Time | Query Time | Notes |
|--------|-------------|-------------|------------|------------|-------|
| **Analytical** | 0.000% | 0.000% | N/A | ~10μs | Ground truth (blackscholes library) |
| **Chebyshev Barycentric** | 0.000% | 1.980% | ~0.35s | ~0.065ms | Full tensor, analytical derivatives |
| **Chebyshev TT** | 0.014% | 0.029% | ~0.35s | ~0.004ms | TT-Cross, ~7,400 evals (vs 161k full) |
| **MoCaX Standard** | 0.000% | 1.980% | ~1.04s | ~0.47ms | Proprietary C++, full tensor (161k evals) |
| **MoCaX TT** | 0.712% | 1.604% | ~5.73s | ~0.25ms | Proprietary C++, ALS (8k evals) |
| **FDM** | 0.803% | 2.234% | N/A | ~0.5s/case | PDE solver, no pre-computation |

**Key Insights**:
- **Chebyshev Barycentric**: spectral accuracy (0.000% price error) with analytical derivatives
- **Chebyshev TT**: only 7,400 evaluations (vs 161,051 full tensor), 50x more accurate than MoCaX TT
- **Build time**: 0.35s (PyChebyshev) vs 1.04--5.73s (MoCaX)
- **Query time**: 0.004ms batch (TT) / 0.065ms single (barycentric) vs 0.25--0.47ms (MoCaX)

---

## Visual Comparison: Spectral Convergence

### Error Surface at 12×12 Nodes

**Chebyshev Barycentric (Python)**
<p align="center">
  <img src="https://raw.githubusercontent.com/0xC000005/PyChebyshev/main/barycentric_2d_error_n12.png" width="80%" alt="Chebyshev Barycentric Error Surface">
</p>

**MoCaX Standard (C++)**
<p align="center">
  <img src="https://raw.githubusercontent.com/0xC000005/PyChebyshev/main/mocax_2d_error_n12.png" width="80%" alt="MoCaX Error Surface">
</p>

Both methods achieve **spectral accuracy** (exponential error decay) with identical node configurations, demonstrating that the pure Python implementation successfully replicates the MoCaX algorithm's mathematical foundation.

### Convergence Analysis

**Chebyshev Barycentric**
<p align="center">
  <img src="https://raw.githubusercontent.com/0xC000005/PyChebyshev/main/barycentric_2d_convergence.png" width="80%" alt="Barycentric Convergence Plot">
</p>

**MoCaX Standard**
<p align="center">
  <img src="https://raw.githubusercontent.com/0xC000005/PyChebyshev/main/mocax_2d_convergence.png" width="80%" alt="MoCaX Convergence Plot">
</p>

The convergence plots demonstrate exponential error decay as node count increases, confirming the spectral accuracy predicted by the Bernstein Ellipse Theorem. Errors drop rapidly from 4×4 to 12×12 nodes, reaching near-machine precision for this analytic function.

## Features

- **Full tensor interpolation** via `ChebyshevApproximation` — spectral accuracy for up to ~5 dimensions
- **Piecewise Chebyshev (splines)** via `ChebyshevSpline` — knots at kinks/discontinuities restore spectral convergence for non-smooth functions
- **Tensor Train decomposition** via `ChebyshevTT` — TT-Cross builds from O(d·n·r²) evaluations for 5+ dimensions
- **Sliding technique** via `ChebyshevSlider` — additive decomposition for separable high-dimensional functions
- **Arithmetic operators** (`+`, `-`, `*`, `/`) — combine interpolants into portfolio-level proxies with no re-evaluation
- **Extrusion & slicing** — add or fix dimensions to combine interpolants across different risk-factor sets
- **Integration, rootfinding & optimization** — spectral calculus directly on interpolants, no re-evaluation needed
- **Analytical derivatives** via spectral differentiation matrices (no finite differences)
- **Vectorized evaluation** using BLAS matrix-vector products (~0.065ms/query)
- **Pure Python** — NumPy + SciPy only, no compiled extensions needed

## Acknowledgments

**Special thanks to MoCaX** for their high-quality [videos](https://www.youtube.com/@MoCaX) on YouTube explaining the mathematical foundations behind their library. These resources were invaluable for understanding and implementing the barycentric Chebyshev interpolation algorithm.

---

## Getting Started

### Installation

```bash
pip install pychebyshev
```

### Quick Example

```python
import math
from pychebyshev import ChebyshevApproximation

# Approximate any smooth function
def my_func(x, _):
    return math.sin(x[0]) * math.exp(-x[1])

cheb = ChebyshevApproximation(
    my_func,
    num_dimensions=2,
    domain=[[-1, 1], [0, 2]],
    n_nodes=[15, 15],
)
cheb.build()

# Evaluate function value
value = cheb.vectorized_eval([0.5, 1.0], [0, 0])

# Analytical first derivative w.r.t. x[0]
dfdx = cheb.vectorized_eval([0.5, 1.0], [1, 0])

# Price + all derivatives at once (shared weights, ~25% faster)
results = cheb.vectorized_eval_multi(
    [0.5, 1.0],
    [[0, 0], [1, 0], [0, 1], [2, 0]],
)
```

### High-Dimensional Functions (Tensor Train)

For 5+ dimensional functions where full tensor grids are too expensive, use `ChebyshevTT` with TT-Cross:

```python
from pychebyshev import ChebyshevTT

tt = ChebyshevTT(
    my_func, num_dimensions=5,
    domain=[[-1, 1]] * 5,
    n_nodes=[11] * 5,
    max_rank=10,
)
tt.build()
val = tt.eval([0.5] * 5)

# Batch evaluation (much faster per point)
import numpy as np
points = np.random.uniform(-1, 1, (1000, 5))
vals = tt.eval_batch(points)
```

> **Note:** `ChebyshevTT` uses finite differences for derivatives (not analytical spectral differentiation). First-order Greeks are typically accurate to ~0.03%; see the [docs](https://0xc000005.github.io/PyChebyshev/user-guide/tensor-train/#derivatives-via-finite-differences) for details.

### Separable Functions (Sliding Technique)

For functions that decompose additively, use `ChebyshevSlider`:

```python
from pychebyshev import ChebyshevSlider

slider = ChebyshevSlider(
    my_func, num_dimensions=10,
    domain=[[-1, 1]] * 10,
    n_nodes=[11] * 10,
    partition=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
    pivot_point=[0.0] * 10,
)
slider.build()
val = slider.eval([0.5] * 10, [0] * 10)
```

See the [documentation](https://0xc000005.github.io/PyChebyshev/) for full API reference and usage guides.

---

## How It Works

PyChebyshev uses **Chebyshev tensor interpolation** to pre-compute fast approximations of smooth functions. The approach generalizes to all analytic functions, as proven by [Gaß et al. (2018)](https://arxiv.org/abs/1505.04648), which establishes (sub)exponential convergence rates for Chebyshev interpolation of parametric conditional expectations.

### Four approaches for different scales

| Class | Approach | Build Cost | Eval Cost | Derivatives |
|-------|----------|-----------|-----------|-------------|
| `ChebyshevApproximation` | Full tensor + barycentric weights | $n^d$ evals | $O(d \cdot n)$ via BLAS GEMV | Analytical (spectral) |
| `ChebyshevSpline` | Piecewise Chebyshev at knots | $\text{pieces} \times n^d$ evals | $O(d \cdot n)$ via BLAS GEMV | Analytical (per piece) |
| `ChebyshevTT` | TT-Cross + maxvol pivoting | $O(d \cdot n \cdot r^2)$ evals | $O(d \cdot n \cdot r^2)$ via einsum | Finite differences |
| `ChebyshevSlider` | Additive decomposition into slides | Sum of slide grids | Sum of per-slide evals | Analytical (per-slide) |

### Key ideas

**Barycentric interpolation** (`ChebyshevApproximation`): The barycentric formula separates *node positions* from *function values* — weights $w_i = 1/\prod_{j \neq i}(x_i - x_j)$ depend only on nodes and can be fully pre-computed. This enables $O(N)$ evaluation for all dimensions (vs. $O(N \log N)$ for polynomial coefficient approaches) and **analytical derivatives** via spectral differentiation matrices — no finite differences needed.

**Tensor Train decomposition** (`ChebyshevTT`): The full $n^d$ tensor is never formed. Instead, TT-Cross builds a chain of small 3D cores from strategically selected function evaluations (maxvol pivot selection), reducing 161,051 evaluations to ~7,400 for 5D Black-Scholes.

**BLAS vectorization**: All evaluation paths route tensor contractions through optimized BLAS routines (`numpy.dot`, `numpy.einsum`), achieving ~150x speedup over scalar loops.

For the full mathematical foundation — Chebyshev nodes, Bernstein ellipse theorem, barycentric formula, spectral differentiation matrices, and TT-Cross algorithm — see the [documentation](https://0xc000005.github.io/PyChebyshev/user-guide/concepts/).

