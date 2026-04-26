# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyChebyshev is a pip-installable Python library for multi-dimensional Chebyshev tensor interpolation with analytical derivatives. It uses barycentric interpolation with full weight pre-computation and BLAS GEMV for fast evaluation. Originally developed as a research project comparing numerical methods for Black-Scholes option pricing against the MoCaX C++ library.

## Commands

```bash
# Setup
uv sync

# Run tests (~838 tests, ~110s due to 5D Black-Scholes builds)
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

The installable package. Public classes: `ChebyshevApproximation`, `ChebyshevSpline`, `ChebyshevSlider`, and `ChebyshevTT`. Arithmetic operators (`+`, `-`, `*`, `/`) enable algebraic combination of interpolants sharing the same grid. `extrude()` and `slice()` enable dimension manipulation for combining interpolants across different risk-factor sets. `integrate()`, `roots()`, `minimize()`, `maximize()` provide spectral calculus operations. `nodes()` and `from_values()` enable a "nodes first, values later" workflow for HPC/distributed evaluation.

- **`barycentric.py`** — Core implementation. `ChebyshevApproximation` class with `build()`, `eval()`, `vectorized_eval()`, `vectorized_eval_multi()`, `integrate()`, `roots()`, `minimize()`, `maximize()`, `nodes()`, `from_values()`. Key insight: barycentric weights depend only on node positions (not function values), enabling full pre-computation. `vectorized_eval()` uses a reshape trick to route N-D tensor contractions through BLAS GEMV (~0.065ms/query). `vectorized_eval_multi()` shares barycentric weight computation across price + derivatives (~0.29ms for 6 outputs). `fast_eval()` exists but is deprecated (JIT path, ~150x slower than BLAS).
- **`spline.py`** — `ChebyshevSpline` class for piecewise Chebyshev interpolation with user-specified knots at singularities. Partitions the domain into sub-intervals and builds an independent `ChebyshevApproximation` on each piece. Restores spectral convergence for functions with kinks or discontinuities. Supports `integrate()`, `roots()`, `minimize()`, `maximize()`, `nodes()`, `from_values()` across pieces.
- **`slider.py`** — `ChebyshevSlider` class for high-dimensional approximation via the Sliding Technique.
- **`tensor_train.py`** — `ChebyshevTT` class for Tensor Train Chebyshev interpolation. TT-Cross builds from O(d·n·r²) evaluations with maxvol pivoting, eval caching, and SVD-based adaptive rank. Vectorized batch eval via numpy einsum. FD derivatives. v0.13 adds `method='als'` rank-adaptive ALS build, `run_completion()` to refine any built TT at fixed rank, `inner_product()` exact TT contraction, and in-place `orth_left`/`orth_right` canonicalization sweeps.
- **`_algebra.py`** — Shared helpers for Chebyshev arithmetic operators (compatibility validation, operator dispatch).
- **`_extrude_slice.py`** — Shared helpers for extrusion and slicing (parameter validation, tensor manipulation, barycentric contraction).
- **`_calculus.py`** — Shared helpers for Chebyshev calculus (Fejér-1 quadrature weights via DCT-III, sub-interval quadrature weights via Chebyshev antiderivatives, companion-matrix rootfinding, 1-D optimization). References: Waldvogel (2006), Trefethen (2013).
- **`_binary.py`** — Private. `.pcb` portable binary serialization (v0.14). Reading/writing for `ChebyshevApproximation` and `ChebyshevSpline`. Stdlib `struct` + NumPy only.
- **`_jit.py`** — Deprecated Numba JIT kernel with pure NumPy fallback. Used only by deprecated `fast_eval()`.
- **`_version.py`** — Single source of truth for version string.
- v0.15 adds `additional_data=` ctor kwarg, `set_descriptor`/`get_descriptor`,
  `is_construction_finished`/`get_constructor_type`/`get_used_ns`, and the
  `get_derivative_id` integer registry + `eval(..., derivative_id=...)` (the
  last one excluding `ChebyshevTT`).
- v0.16 adds `clone()`, instance getters (`get_max_derivative_order`,
  `get_error_threshold`, `get_special_points`, `get_evaluation_points`,
  `get_num_evaluation_points`), `peek_format_version()` static,
  `is_dimensionality_allowed()` static, `defer_build=True` +
  `set_original_function_values()`, and optional `Domain`/`Ns`/`SpecialPoints`
  typed helpers (constructors accept both forms).

### Benchmark Scripts (project root)

Not part of the library. Compare Chebyshev barycentric against alternative methods:

- `chebyshev_barycentric.py` — Standalone version with embedded test suite
- `chebyshev_baseline.py` — NumPy polynomial coefficient approach
- `fdm_baseline.py` — Finite difference PDE solver
- `mocax_baseline.py`, `mocax_tt.py`, `mocax_sliding.py` — MoCaX C++ library tests (require `mocaxextend_lib/`)
- `compare_methods_time_accuracy.py` — Fair time/accuracy comparison across all methods
- `compare_tensor_train.py` — PyChebyshev TT vs MoCaX TT comparison (requires `mocaxextend_lib/`)
- `compare_tensor_train_als.py` — PyChebyshev TT ALS + algebra (ALS build, `run_completion`, `inner_product`, `orth_left`/`orth_right`) vs MoCaX TT comparison (requires `mocaxextend_lib/`)
- `compare_spline.py` — PyChebyshev ChebyshevSpline vs MoCaX spine comparison (requires `mocaxextend_lib/`)
- `compare_algebra.py` — PyChebyshev Chebyshev algebra vs MoCaX comparison (requires `mocaxextend_lib/`)
- `compare_extrude_slice.py` — PyChebyshev extrusion/slicing vs MoCaX comparison (requires `mocaxextend_lib/`)
- `compare_from_values.py` — PyChebyshev nodes()/from_values() vs MoCaX Extend comparison (requires `mocaxextend_lib/`)
- `compare_special_points.py` — PyChebyshev special_points vs MoCaX MocaxSpecialPoints + MocaxNs comparison (requires `mocax_lib/`)
- `compare_v016_polish.py` — PyChebyshev v0.16 polish surface vs MoCaX 4.3.1 cosmetic API (requires `mocaxpy`; gracefully skips MoCaX side if not installed)

### Tests (`tests/`)

- `conftest.py` — Shared fixtures (`cheb_sin_3d`, `cheb_bs_3d`, `cheb_bs_5d`, `tt_sin_3d`, `tt_bs_5d`, `spline_abs_1d`, `spline_bs_2d`, `algebra_cheb_f`, `algebra_cheb_g`, `algebra_spline_f`, `algebra_spline_g`, `algebra_slider_f`, `algebra_slider_g`, plus extrude/slice fixtures) and analytical Black-Scholes functions (reimplemented via `scipy.stats.norm` to avoid external deps). Helper functions are imported as `from conftest import ...` (pytest auto-import, NOT `from tests.conftest`).
- `test_barycentric.py` — 48 tests: accuracy, derivatives, eval method consistency, node coincidence, error estimation, build-required guard.
- `test_slider.py` — 40 tests: additive/coupled functions, 5D, cross-group derivatives, error estimation, serialization.
- `test_spline.py` — 55 tests: construction validation, 1D/2D accuracy, batch eval, derivatives, knot boundary checks, multiple knots, error estimation, serialization.
- `test_tensor_train.py` — 71 tests: TT-Cross/TT-SVD accuracy, batch eval, FD derivatives, rank control, serialization, error estimation, plus v0.13 classes: `TestOrthogonalization` (8), `TestInnerProduct` (7), `TestALSInternals` (3), `TestALS` (7), `TestCompletion` (7), `TestCrossFeatureALS` (4).
- `test_algebra.py` — 77 tests: arithmetic operators for ChebyshevApproximation, ChebyshevSpline, and ChebyshevSlider; batch/multi eval; compatibility error handling; portfolio use cases.
- `test_extrude_slice.py` — 63 tests: extrusion and slicing for ChebyshevApproximation, ChebyshevSpline, and ChebyshevSlider; round-trip identity; derivatives; serialization; portfolio via extrude+algebra; edge cases (min nodes, boundary slicing, batch/multi eval, error estimates).
- `test_calculus.py` — 74 tests: integration (full-domain and sub-interval), rootfinding, and optimization for ChebyshevApproximation and ChebyshevSpline; 1-D and multi-D; partial integration; spline piece merging and overlap clipping; edge cases.
- `test_from_values.py` — 65 tests: nodes() and from_values() for ChebyshevApproximation and ChebyshevSpline; bit-identical equivalence with build(); derivatives, calculus, algebra, extrude/slice, save/load; edge cases (NaN/Inf, shape mismatch, 1-node dim, build guard, 4D, boundary eval, negative/wide/tight domains, duplicate knots, algebra chains, domain validation).
- `test_special_points.py` — 37 tests: `ChebyshevApproximation.__new__` dispatch to `ChebyshevSpline` when `special_points` declares any kink (option A, precedent `pathlib.Path`); validation of special_points shape + nested `n_nodes`; 1D/2D correctness (abs kink recovery to machine precision; plateau control); cross-feature (save/load, algebra, integrate, extrude/slice, from_values); edge cases.
- `test_error_threshold.py` — 37 tests: v0.11 auto-N doubling loop, max_n cap, get_optimal_n1, semi-auto mixed-N paths, verbose prints, spline per-piece doubling.
- `test_binary_format.py` — 76 tests: low-level helpers, header parsing, format detection, ChebyshevApproximation round-trip (incl. n=1 dim), ChebyshevSpline round-trip, save/load integration with `format=` kwarg + autodetect, golden vectors, corruption rejection, cross-feature (from_values, algebra, extrude, slice, 5D BS, min n_nodes).
- `test_ergonomics.py` — ~30 tests: descriptor, additional_data threading +
  binary rejection, derivative_id registry on Approximation/Spline/Slider,
  introspection trio (`is_construction_finished`, `get_constructor_type`,
  `get_used_ns`).
- `test_v016_polish.py` — ~74 tests: clone() on all four classes, instance getters
  (`get_max_derivative_order`, `get_error_threshold`, `get_special_points`,
  `get_evaluation_points`, `get_num_evaluation_points`), `peek_format_version`,
  `is_dimensionality_allowed`, `defer_build` + `set_original_function_values`,
  `Domain`/`Ns`/`SpecialPoints` typed helpers.

### CI/CD (`.github/workflows/`)

- `test.yml` — pytest on Python 3.10-3.13 (triggers on push/PR to main)
- `publish.yml` — `uv build && uv publish` via OIDC trusted publishing (triggers on GitHub release creation)
- `dependabot-automerge.yml` — auto-approves and merges Dependabot patch/minor version bumps

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
