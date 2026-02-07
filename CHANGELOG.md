# Changelog

All notable changes to PyChebyshev will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
