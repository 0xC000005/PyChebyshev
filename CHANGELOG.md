# Changelog

All notable changes to PyChebyshev will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
