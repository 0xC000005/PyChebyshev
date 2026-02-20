# Changelog

All notable changes to PyChebyshev will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.1] - 2026-02-20

### Fixed

- Corrected Good (1961) citation: journal is *The Quarterly Journal of
  Mathematics* 12(1):61-68, not *Quarterly Journal of Mechanics and Applied
  Mathematics* 14:195-196. Fixed in `_calculus.py`, `barycentric.py`, and
  `calculus.md`.
- Fixed Trefethen book year from "SIAM 2019" to "SIAM 2013" in `concepts.md`
  and `greeks.md` (2019 is the extended edition; all citations reference the
  2013 first edition).
- Fixed TT-Cross attribution in `tensor-train.md`: now correctly cites both
  Oseledets & Tyrtyshnikov (2010) and Savostyanov & Oseledets (2011).
- Added 30+ missing inline citations across 11 documentation pages, verified
  against original sources. Key additions: Runge (1901) for Runge's phenomenon,
  Berrut & Trefethen (2004) for barycentric formula and differentiation matrices,
  Trefethen (2013) chapter references for Lebesgue constant, Bernstein ellipse,
  spectral convergence of derivatives, and Chebyshev node generation.
- Added formal References sections to `concepts.md`, `spline.md`, `sliding.md`,
  and `tensor-train.md`.

## [0.10.0] - 2026-02-20

### Added

- `ChebyshevApproximation.nodes()` static method to generate Chebyshev grid
  points without evaluating any function. Returns per-dimension node arrays,
  the full Cartesian product grid, and the expected tensor shape.
- `ChebyshevApproximation.from_values()` class method to construct a fully
  functional interpolant from pre-computed function values. Supports all
  operations: evaluation, derivatives, integration, rootfinding, optimization,
  algebra, extrusion/slicing, and serialization.
- `ChebyshevSpline.nodes()` static method returning per-piece node information
  for piecewise Chebyshev interpolation with knots.
- `ChebyshevSpline.from_values()` class method to construct a spline from
  per-piece pre-computed values.
- Guard in `build()` for both classes: raises `RuntimeError` with a clear
  message when `function=None` (e.g., after `from_values()`, `load()`, or
  algebra operations).
- Input validation in `from_values()`: shape mismatch, NaN/Inf detection,
  dimension consistency, and domain ordering.
- 65 new tests covering core equivalence (bit-identical to `build()`),
  edge cases (NaN/Inf, shape mismatch, domain validation, duplicate knots,
  boundary evaluation, 4D, algebra chains), and cross-compatibility.
- Documentation page: "Pre-computed Values" with workflow, indexing convention,
  mathematical justification, API reference, and MoCaX Extend comparison.

## [0.9.5] - 2026-02-15

### Added

- Sub-interval integration via `bounds` parameter on `integrate()` for both
  `ChebyshevApproximation` and `ChebyshevSpline`. Uses generalized Fejér-1
  quadrature with sub-interval Chebyshev moments computed from closed-form
  antiderivatives (Trefethen 2013, Ch. 19).
- Spline sub-interval integration with automatic piece overlap clipping:
  pieces with no overlap are skipped, partial overlaps receive sub-interval
  bounds, fully contained pieces use standard Fejér-1 weights.
- 22 new tests for sub-interval integration (14 for ChebyshevApproximation,
  8 for ChebyshevSpline), including polynomial exactness, scipy cross-validation,
  additivity, and error handling.

## [0.9.0] - 2026-02-15

### Added

- `integrate()` method on `ChebyshevApproximation` and `ChebyshevSpline` for
  definite integration via Fejér-1 quadrature (Waldvogel 2006), with O(n log n)
  weight computation via DCT-III and O(n^d) multi-D contraction via BLAS.
- `roots()` method on `ChebyshevApproximation` and `ChebyshevSpline` for finding
  all real roots along a dimension via companion matrix eigenvalues
  (`numpy.polynomial.chebyshev.chebroots`).
- `minimize()` / `maximize()` methods on `ChebyshevApproximation` and
  `ChebyshevSpline` for global optimization via derivative rootfinding and
  endpoint evaluation.
- Partial integration returns a lower-dimensional `ChebyshevApproximation`
  (analogous to `slice()`).
- Internal `_calculus.py` module with shared helpers for Fejér-1 weights,
  1-D rootfinding, and 1-D optimization.
- New documentation page: Chebyshev Calculus (integration, roots, optimization).
- 52 new tests for calculus operations (test_calculus.py).

## [0.8.0] - 2026-02-15

### Added

- `extrude()` and `slice()` methods on `ChebyshevApproximation`, `ChebyshevSpline`, and `ChebyshevSlider`.
- Extrusion adds new dimensions where the function is constant, enabling portfolio combination across different risk-factor sets.
- Slicing fixes a dimension at a value, reducing dimensionality via barycentric interpolation (with fast path at exact nodes).
- Extrude-then-slice round-trip preserves the original interpolant exactly.
- Internal `_extrude_slice.py` module with shared validation and tensor manipulation helpers.
- New documentation page: Extrusion & Slicing (mathematical proofs, API reference, portfolio use case).
- `compare_extrude_slice.py` -- MoCaX comparison script (local only, not in CI).
- 63 new tests for extrusion and slicing across all three classes.

## [0.7.0] - 2026-02-14

### Added

- Arithmetic operators (`+`, `-`, `*`, `/`, unary `-`, `+=`, `-=`, `*=`, `/=`) for `ChebyshevApproximation`, `ChebyshevSpline`, and `ChebyshevSlider`.
- Portfolio-level combination of Chebyshev interpolants via weighted sums -- no re-evaluation needed.
- Derivatives and error estimates propagate correctly through algebraic operations.
- Internal `_algebra.py` module with shared compatibility validation.
- `_from_grid()` factory on `ChebyshevApproximation`, `_from_pieces()` on `ChebyshevSpline`, `_from_slides()` on `ChebyshevSlider`.

## [0.6.0] - 2026-02-14

### Added

- `ChebyshevSpline` class for piecewise Chebyshev interpolation with user-specified knots (Section 3.8, Ruiz & Zeron 2021)
- Knot-based domain partitioning: place knots at kinks or discontinuities to restore spectral convergence on each piece
- `eval()`, `eval_multi()`, `eval_batch()` with automatic piece routing via `np.searchsorted`
- Analytical derivatives within each piece via spectral differentiation matrices (inherited from `ChebyshevApproximation`)
- `ValueError` raised when requesting derivatives at knot boundaries (left/right derivatives differ)
- `error_estimate()` returning the maximum error across all disjoint pieces
- `save()` / `load()` serialization following the same pattern as other PyChebyshev classes
- `num_pieces`, `total_build_evals`, `build_time` properties
- New documentation page: Chebyshev Splines (mathematical justification, usage guide)
- `compare_spline.py` — MoCaX spine comparison script (local only, not in CI)
- ~34 new tests for spline construction, accuracy, derivatives, batch eval, serialization

## [0.5.0] - 2026-02-12

### Added

- `ChebyshevTT` class for Chebyshev interpolation in Tensor Train format (Chapter 6, Ruiz & Zeron 2021)
- TT-Cross approximation (`method='cross'`) for building from O(d * n * r^2) function evaluations instead of O(n^d)
- TT-SVD build option (`method='svd'`) for validation and moderate-dimension problems
- `eval()` for single-point evaluation via TT inner product with Chebyshev polynomial contraction
- `eval_batch()` for vectorized batch evaluation of multiple points simultaneously via numpy einsum
- `eval_multi()` for finite-difference derivatives (1st and 2nd order, including cross-derivatives)
- `error_estimate()` for approximate error estimation from Chebyshev coefficient cores
- `save()` / `load()` serialization with version compatibility checks (same pattern as other classes)
- `tt_ranks` and `compression_ratio` properties for inspecting the TT decomposition
- `_maxvol()` module-level function implementing the maximum-volume row selection algorithm
- `_tt_cross()` module-level function implementing TT-Cross with alternating left-right sweeps and maxvol pivot selection

## [0.4.0] - 2026-02-10

### Added

- `error_estimate()` method on `ChebyshevApproximation` — estimates supremum-norm interpolation error from Chebyshev expansion coefficients via DCT-II, without needing the true function (Ruiz & Zeron 2021, Section 3.4)
- `error_estimate()` method on `ChebyshevSlider` — returns sum of per-slide error estimates
- `_chebyshev_coefficients_1d()` static method for computing Chebyshev expansion coefficients from values at Type I nodes
- Error estimate shown in `__str__()` output for both classes when built
- New documentation page: Error Estimation (mathematical background, usage examples)
- 12 new tests for error estimation (8 for barycentric, 4 for slider)
- `compare_error_estimation.py` — local benchmarking script comparing PyChebyshev vs MoCaX error estimates

## [0.3.0] - 2026-02-10

### Deprecated

- `fast_eval()` — use `vectorized_eval()` instead, which is ~150x faster via BLAS GEMV
- `[jit]` optional dependency (Numba) — no longer needed since BLAS path outperforms JIT
- `_jit.py` module — will be removed in a future version

### Changed

- README, docs, and CLAUDE.md updated to reflect BLAS GEMV as the primary fast path
- Removed Numba JIT installation instructions from all documentation
- Removed `numba` from dev dependencies

## [0.2.1] - 2026-02-10

### Added

- `save()` and `load()` methods on `ChebyshevApproximation` and `ChebyshevSlider` for persisting built interpolants to disk (pickle-based)
- `__repr__` and `__str__` methods on both classes for human-readable printing
- Version compatibility check on load with warning for mismatched versions
- New documentation page: Saving & Loading Interpolants
- 21 new tests for serialization and printing

## [0.2.0] - 2026-02-09

### Added

- `ChebyshevSlider` class for high-dimensional approximation via the Sliding Technique (Ch. 7, Ruiz & Zeron 2021)
- Additive decomposition into low-dimensional slides around a pivot point
- Analytical derivatives per slide with correct cross-group mixed partial handling (returns 0)
- Documentation page for the Sliding Technique with usage examples and limitations
- 24 new tests for slider (additive, coupled, 5D, cross-group derivatives, validation)

### Changed

- README updated: repositioned as a library (was standalone educational script)
- Getting Started section now uses `pip install pychebyshev` with code examples
- Fixed repo URL in docs (`maxjingwezhang` → `0xC000005`)

## [0.1.1] - 2026-02-07

### Fixed

- README images now load on PyPI (use absolute GitHub URLs instead of relative paths)

## [0.1.0] - 2026-02-07

### Added

- `ChebyshevApproximation` class for multi-dimensional Chebyshev tensor interpolation
- Barycentric interpolation with full weight pre-computation
- Analytical derivatives via spectral differentiation matrices (1st and 2nd order)
- `vectorized_eval()` using BLAS matrix-vector products (~0.065 ms/query)
- `vectorized_eval_multi()` with shared barycentric weights (~0.29 ms for price + 5 Greeks)
- `fast_eval()` with Numba JIT compilation (optional)
- Node coincidence handling for all evaluation methods
- MkDocs + Material documentation with KaTeX math rendering
- pytest test suite (22 tests covering 3D/5D accuracy and method consistency)
- GitHub Actions CI/CD for testing and PyPI publishing
