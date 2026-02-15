# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyChebyshev is a pip-installable Python library for multi-dimensional Chebyshev tensor interpolation with analytical derivatives. It uses barycentric interpolation with full weight pre-computation and BLAS GEMV for fast evaluation. Originally developed as a research project comparing numerical methods for Black-Scholes option pricing against the MoCaX C++ library.

## Commands

```bash
# Setup
uv sync

# Run tests (~255 tests, ~120s due to 5D Black-Scholes builds)
uv run pytest tests/ -v

# Run a single test
uv run pytest tests/test_barycentric.py::TestSimple3D::test_price_accuracy -v

# Build package
uv build

# Build and preview docs locally
uv run mkdocs serve          # http://127.0.0.1:8000
uv run mkdocs build --strict # Verify docs build

# Deploy docs to GitHub Pages
uv run mkdocs gh-deploy --force

# Run benchmark scripts (not part of library)
uv run python chebyshev_barycentric.py
uv run python compare_methods_time_accuracy.py
```

## Architecture

### Library (`src/pychebyshev/`)

The installable package. Public classes: `ChebyshevApproximation`, `ChebyshevSpline`, `ChebyshevSlider`, and `ChebyshevTT`. Arithmetic operators (`+`, `-`, `*`, `/`) enable algebraic combination of interpolants sharing the same grid.

- **`barycentric.py`** — Core implementation. `ChebyshevApproximation` class with `build()`, `eval()`, `vectorized_eval()`, `vectorized_eval_multi()`. Key insight: barycentric weights depend only on node positions (not function values), enabling full pre-computation. `vectorized_eval()` uses a reshape trick to route N-D tensor contractions through BLAS GEMV (~0.065ms/query). `vectorized_eval_multi()` shares barycentric weight computation across price + derivatives (~0.29ms for 6 outputs). `fast_eval()` exists but is deprecated (JIT path, ~150x slower than BLAS).
- **`spline.py`** — `ChebyshevSpline` class for piecewise Chebyshev interpolation with user-specified knots at singularities. Partitions the domain into sub-intervals and builds an independent `ChebyshevApproximation` on each piece. Restores spectral convergence for functions with kinks or discontinuities.
- **`slider.py`** — `ChebyshevSlider` class for high-dimensional approximation via the Sliding Technique.
- **`tensor_train.py`** — `ChebyshevTT` class for Tensor Train Chebyshev interpolation. TT-Cross builds from O(d·n·r²) evaluations with maxvol pivoting, eval caching, and SVD-based adaptive rank. Vectorized batch eval via numpy einsum. FD derivatives.
- **`_algebra.py`** — Shared helpers for Chebyshev arithmetic operators (compatibility validation, operator dispatch).
- **`_jit.py`** — Deprecated Numba JIT kernel with pure NumPy fallback. Used only by deprecated `fast_eval()`.
- **`_version.py`** — Single source of truth for version string.

### Benchmark Scripts (project root)

Not part of the library. Compare Chebyshev barycentric against alternative methods:

- `chebyshev_barycentric.py` — Standalone version with embedded test suite
- `chebyshev_baseline.py` — NumPy polynomial coefficient approach
- `fdm_baseline.py` — Finite difference PDE solver
- `mocax_baseline.py`, `mocax_tt.py`, `mocax_sliding.py` — MoCaX C++ library tests (require `mocaxextend_lib/`)
- `compare_methods_time_accuracy.py` — Fair time/accuracy comparison across all methods
- `compare_tensor_train.py` — PyChebyshev TT vs MoCaX TT comparison (requires `mocaxextend_lib/`)
- `compare_spline.py` — PyChebyshev ChebyshevSpline vs MoCaX spine comparison (requires `mocaxextend_lib/`)
- `compare_algebra.py` — PyChebyshev Chebyshev algebra vs MoCaX comparison (requires `mocaxextend_lib/`)

### Tests (`tests/`)

- `conftest.py` — Shared fixtures (`cheb_sin_3d`, `cheb_bs_3d`, `cheb_bs_5d`, `tt_sin_3d`, `tt_bs_5d`, `spline_abs_1d`, `spline_bs_2d`, `algebra_cheb_f`, `algebra_cheb_g`, `algebra_spline_f`, `algebra_spline_g`, `algebra_slider_f`, `algebra_slider_g`) and analytical Black-Scholes functions (reimplemented via `scipy.stats.norm` to avoid external deps). Helper functions are imported as `from conftest import ...` (pytest auto-import, NOT `from tests.conftest`).
- `test_barycentric.py` — 48 tests: accuracy, derivatives, eval method consistency, node coincidence, error estimation, build-required guard.
- `test_slider.py` — 40 tests: additive/coupled functions, 5D, cross-group derivatives, error estimation, serialization.
- `test_spline.py` — 55 tests: construction validation, 1D/2D accuracy, batch eval, derivatives, knot boundary checks, multiple knots, error estimation, serialization.
- `test_tensor_train.py` — 35 tests: TT-Cross/TT-SVD accuracy, batch eval, FD derivatives, rank control, serialization, error estimation.
- `test_algebra.py` — 77 tests: arithmetic operators for ChebyshevApproximation, ChebyshevSpline, and ChebyshevSlider; batch/multi eval; compatibility error handling; portfolio use cases.

### CI/CD (`.github/workflows/`)

- `test.yml` — pytest on Python 3.10-3.13 (triggers on push/PR to main)
- `publish.yml` — `uv build && uv publish` via OIDC trusted publishing (triggers on GitHub release creation)

### Docs (`docs/`)

MkDocs + Material theme, deployed to GitHub Pages. KaTeX for math rendering, mkdocstrings for API autodoc from NumPy-style docstrings.

## Key Technical Constraints

- **Numba/JIT is deprecated**: `fast_eval()` and `_jit.py` are deprecated. `vectorized_eval()` via BLAS GEMV is ~150x faster and needs no optional deps.
- **Python >=3.10**: Uses `from __future__ import annotations` for modern type hints.
- **Core deps are only numpy and scipy**: pytest, matplotlib, blackscholes etc. belong in dev/optional groups only.
- **README images must use absolute GitHub URLs**: PyPI renders README but can't serve local files. Use `https://raw.githubusercontent.com/0xC000005/PyChebyshev/main/...` for all `<img src>`.
- **Version in two places**: `pyproject.toml` and `src/pychebyshev/_version.py` must stay in sync.

## Release Process

1. Bump version in `pyproject.toml` and `src/pychebyshev/_version.py`
2. Update `CHANGELOG.md`
3. Commit, push to main
4. `gh release create vX.Y.Z` — triggers publish workflow → PyPI
5. `uv run mkdocs gh-deploy --force` — updates docs site
