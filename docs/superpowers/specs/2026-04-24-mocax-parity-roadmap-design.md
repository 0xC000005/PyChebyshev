# MoCaX Parity Roadmap — Design Document

**Date:** 2026-04-24
**Target:** Feature parity with MoCaX Intelligence 4.3.1 + MoCaXExtend
**Milestone:** v1.0.0 (PyChebyshev "parity achieved" release)
**Status:** Approved design — awaiting implementation plans per version

---

## 1. Motivation

PyChebyshev began as a research project reproducing MoCaX's Chebyshev tensor interpolation results in pure Python + numpy/scipy. As of v0.10.1, most core MoCaX behaviors are covered (barycentric eval, analytical derivatives, splines, sliding, tensor train, algebra, extrude/slice, calculus, deferred construction). A targeted audit against MoCaX's C/C++ source, Python wrappers, and official documentation identified 11 capability or ergonomic gaps across three tiers. This roadmap commits PyChebyshev to closing all of them and declaring feature parity at v1.0.0.

Non-goal: drop-in MoCaX API replacement. Equivalent capabilities, PyChebyshev-idiomatic APIs.

## 2. Scope — what lands in what version

| Version | Theme | Tier-1 | Tier-2 | Tier-3 | Risk |
|---|---|---|---|---|---|
| v0.11 | Error-driven construction | #1 error-threshold ctor | — | #10 `get_optimal_n1` | low |
| v0.12 | Special Points in core | #2 core special points | #5 per-sub-interval N | — | medium |
| v0.13 | TT algebra + TT-ALS | #3 TT-ALS build | #4 TT inner product, orthogonalization | — | **high** |
| v0.14 | Portable binary serialization | — | #6 `.pcb` format | — | medium |
| v0.15 | Ergonomics bundle | — | — | #7 additional_data, #8 derivative_id, #9 descriptor, #11 introspection | low |
| v1.0.0 | Parity announcement + stability | — | — | — | trivial |

Sequencing rationale: confidence-building first, hardest novel work in the middle, polish at the end. See §7 for risk assessment.

---

## 3. Per-version specifications

### 3.1 v0.11 — Error-Driven Construction

**New API:**
```python
# Existing behavior unchanged
ChebyshevApproximation(f, 3, domain, n_nodes=[11, 11, 11])

# All-dim auto-calibration
ChebyshevApproximation(f, 3, domain, error_threshold=1e-6)

# Semi-variable (mix of None/int + error_threshold)
ChebyshevApproximation(f, 3, domain, n_nodes=[None, 15, 15], error_threshold=1e-6)

# New classmethod
ChebyshevApproximation.get_optimal_n1(f, domain, error_threshold, max_n=64) -> int
```

**Algorithm:** doubling loop — start `n=3` on each `None` dim, build, call `error_estimate()`, double N on the worst dim until ε satisfied or `max_n` hit. Uses existing DCT-based `error_estimate()` machinery. No new math required.

**Edge cases:** `max_n` cap (warn, not raise), all-`None` with no `error_threshold` (raise), mutual exclusion when user passes both concrete Ns and ε on the same dim.

**Same for `ChebyshevSpline`** — error_threshold applies per piece.

**Deliverables:**
- `tests/test_error_threshold.py` (~20 tests, CI) — monotone convergence, cap warning, ε achieved, mixed spec validation, equivalence with manual N when result matches
- `compare_error_threshold.py` (root, local-only) — MoCaX semi-variable ctor value match
- `docs/user-guide/error-driven-construction.md` — math justification (why doubling converges for analytic functions via Bernstein ellipse), API reference, usage example (Black-Scholes with ε=1e-8), comparison note with existing fixed-N workflow
- CHANGELOG entry under `[0.11.0]` / `### Added`
- README: one-line bullet under "Features"

### 3.2 v0.12 — Special Points in Core

**New API:**
```python
# Unified API: special_points on core class routes to spline internally
ChebyshevApproximation(
    f, 2, domain=[[-1,1],[0,1]],
    n_nodes=[11, 11],
    special_points=[[0.0], []],  # kink at x[0]=0, none on x[1]
)

# Per-sub-interval N (requires special_points)
ChebyshevApproximation(
    f, 2, domain=[[-1,1],[0,1]],
    n_nodes=[[15, 12], [11]],  # dim-0 split at 0.0 gets 15+12 pts, dim-1 stays flat
    special_points=[[0.0], []],
)
```

**Mechanism:** when `special_points` has any non-empty entry, constructor delegates to a private `_build_as_spline()` that routes through `ChebyshevSpline`. `ChebyshevSpline` remains the direct API for power users. `n_nodes` shape validates against the partition induced by special points.

**Edge cases:** special points must lie strictly inside domain, sorted, unique; `n_nodes` shape-list-of-lists must match piece count per dim; interaction with `error_threshold` (from v0.11) — ε applies per piece.

**Deliverables:**
- `tests/test_special_points.py` (~25 tests, CI) — equivalence vs manual `ChebyshevSpline`, per-sub-interval N expansion, error at knot derivatives still raises, mixed empty/non-empty per-dim
- `compare_special_points.py` (root, local-only) — MoCaX `MocaxSpecialPoints` + `MocaxNs` value match
- `docs/user-guide/spline.md` extension: new section "Unified special-points API on `ChebyshevApproximation`"
- `docs/user-guide/concepts.md`: add paragraph on why special points restore spectral convergence (already partly covered in spline.md; cross-reference)
- CHANGELOG entry

### 3.3 v0.13 — TT Algebra + TT-ALS (HIGH RISK)

**New API:**
```python
# New build method
tt = ChebyshevTT(f, num_dimensions=5, domain=[[-1,1]]*5, n_nodes=[11]*5,
                 method='als', tolerance=1e-6, max_iters=50, max_rank=10)
tt.build()

# Existing method stays default
tt = ChebyshevTT(f, ..., method='cross')  # or method='svd'

# Fixed-rank refinement of existing TT
tt.run_completion(tolerance=1e-8, max_iters=20)

# Orthogonalization sweeps (public because users may want them after composition)
tt.orth_left(position=2)
tt.orth_right(position=0)
tt.orthogonalize(position=3)  # mixed canonical form

# TT-TT contraction
value = tt_a.inner_product(tt_b)  # scalar
```

**Implementation:** new internal module `src/pychebyshev/_tt_als.py`. Port algorithms from `mocaxextend_lib/mocaxextendpy/tt_cheb_utils.py` and `alt_least_squares.py` — do NOT wrap the MoCaX C shared libs. Use `numpy.linalg.qr` + `scipy.linalg.qr` for orthogonalization; use `np.linalg.lstsq` for ALS micro-solves.

**Math references:**
- Oseledets & Tyrtyshnikov (2010), Savostyanov & Oseledets (2011) — TT-Cross (already cited)
- Grasedyck, Kluge, Krämer (2015) "Variants of ALS for tensor completion in the Tensor Train format" — primary ALS reference
- Holtz, Rohwedder, Schneider (2012) — DMRG / rank-adaptive TT

**Pre-work (mandatory before committing to API):** spike branch `tt-als-spike` that runs ALS vs Cross on:
- 5D Black-Scholes (smooth)
- 5D function with a mild singularity
- 8D test function where Cross is known to stall

Only lock the API after the spike confirms ALS converges robustly.

**Edge cases:** ALS divergence (cap iterations, warn + fall back to Cross or raise), rank adaptation stopping criterion, zero-norm training subgrid.

**Deliverables:**
- `tests/test_tt_als.py` (~30 tests, CI) — ALS convergence on smooth functions, rank adaptation stops at tolerance, inner product exactness on TT-constructed tensors, orthogonalization preserves tensor value (evaluated at random points), method='als' vs method='cross' agreement on smooth functions
- `compare_tt_als.py` (root, local-only) — MoCaXExtend `run_rank_adaptive_algo` value match on 5D + 8D
- `docs/user-guide/tensor-train.md` extension: new section "Build methods: Cross vs SVD vs ALS — when to use which", decision matrix, ALS math section with Grasedyck et al. reference, inner product math section
- CHANGELOG entry with explicit callout that v0.13 is the largest TT update since v0.5
- README: TT section updated

### 3.4 v0.14 — Portable Binary Serialization

**New API:**
```python
cheb.save("model.pcb", format='binary')   # new
cheb.save("model.pkl")                    # existing pickle path, default
ChebyshevApproximation.load("model.pcb")  # auto-detects format by magic bytes
```

**Format spec (committed to docs):**
- Bytes 0–3: magic `b'PCB\x01'`
- Bytes 4–7: format version (uint32 LE)
- Bytes 8–11: class tag (uint32: 1=Approximation, 2=Spline, 3=Slider, 4=TT)
- Bytes 12–15: num_dimensions (uint32)
- Header extension per class (domain pairs, n_nodes, etc.) — length-prefixed
- Tensor payload: float64 little-endian, row-major, length-prefixed

Stdlib `struct` only. No HDF5, no external deps.

**Design rationale:** this is NOT MoCaX's `.mcx` format (we don't have the spec). It's *a* portable format designed to be cross-language-friendly (C, Rust, Julia consumers could read it). Pickle stays the default for Python-to-Python round-trips.

**Deliverables:**
- `tests/test_binary_serialization.py` (~15 tests, CI) — round-trip all four classes, magic byte detection, version mismatch warning, corrupted header rejection, cross-class load rejection, large-tensor round-trip
- Format spec document: `docs/user-guide/binary-format.md` — byte-level layout, endianness notes, extension rules
- Update `docs/user-guide/serialization.md`: table "pickle vs .pcb" — when to use each
- CHANGELOG entry
- No `compare_*.py` — there's nothing on the MoCaX side to compare against (different format by design)

### 3.5 v0.15 — Ergonomics Bundle

**New API (additive, no breaking changes):**
```python
# 1. additional_data — formalize the existing second-arg convention as a kwarg
cheb = ChebyshevApproximation(f, 3, domain, n_nodes, additional_data=my_ctx)

# 2. Stable derivative ID
did = cheb.get_derivative_id([1, 0, 2])  # returns cached int
val = cheb.eval(point, derivative_id=did)
# existing list syntax still works:
val = cheb.eval(point, [1, 0, 2])

# 3. Descriptor / metadata
cheb.set_descriptor("5D BS European call, S/K/T/sigma/r")
cheb.get_descriptor()

# 4. Introspection
cheb.get_used_ns()            # -> list[int] or list[list[int]] if special points
cheb.get_constructor_type()   # -> 'fixed' | 'error_threshold' | 'from_values' | 'algebra' | 'slice' | 'extrude'
cheb.is_construction_finished()  # -> bool
```

**Deliverables:**
- `tests/test_ergonomics.py` (~15 tests, CI) — straightforward unit tests
- No new `compare_*.py` (ergonomic wrappers, not capability)
- `docs/user-guide/usage.md` extension: small sections per new method
- CHANGELOG entry

### 3.6 v1.0.0 — Parity Announcement

**No new code.**
- CHANGELOG: summary entry listing all parity items achieved v0.11–v0.15
- `pyproject.toml`: bump status from `4 - Beta` → `5 - Production/Stable`
- `pyproject.toml`: optionally drop Python 3.10 if EOL timing aligns (check Python.org schedule at release time; default: keep 3.10)
- README: update lead paragraph to reflect parity, refresh performance table if benchmarks have shifted
- `docs/index.md`: "Parity achieved" note
- Tag + release + blog post (optional but recommended for visibility)

---

## 4. Conventions codified per release

Every release from v0.11 onward must include all of:

### 4.1 Testing
- **`tests/test_<feature>.py`** — Python correctness tests. Runs in GitHub Actions CI across Python 3.10, 3.11, 3.12, 3.13. Self-contained (no MoCaX imports). Covers: accuracy on analytic functions, edge cases, error paths, equivalence with existing APIs, serialization round-trip if applicable.
- **`compare_<feature>.py`** at repo root — MoCaX value-match test, manual local execution only. NOT in CI (MoCaX proprietary libs are gitignored). Demonstrates PyChebyshev matches MoCaX to reasonable tolerance on the feature's output. Required when the feature has a MoCaX counterpart; skippable for features where MoCaX has no equivalent (e.g., v0.14 `.pcb` binary format is PyChebyshev-only; v0.15 ergonomic shims have no meaningful numeric comparison).
- Target: overall test count grows by 15–30 tests per release.

### 4.2 Documentation
- **User guide page** at `docs/user-guide/<feature>.md` — structured as:
  1. Motivation (what problem this solves)
  2. Mathematical justification (with references — Trefethen 2013, Ruiz & Zeron 2021, or specific papers)
  3. API reference (or cross-reference to mkdocstrings autodoc)
  4. Usage example (at minimum: one small pedagogical example + one Black-Scholes/finance example where relevant)
  5. References section (author, title, journal/publisher, year — verify citation accuracy per v0.10.1 precedent)
- **CHANGELOG entry** under new `## [X.Y.Z] - YYYY-MM-DD` heading, `### Added` / `### Changed` / `### Fixed` / `### Deprecated` sections per Keep-a-Changelog convention
- **README update** if the feature is a major capability (Tier 1 items yes; Tier 3 items no)
- **`mkdocs.yml`** — add new user-guide page to the nav
- **Per-version design spec** at `docs/superpowers/specs/YYYY-MM-DD-v0.XX-<feature>-design.md`, committed with the first PR of the release cycle. Internal document, NOT added to public docs nav.

### 4.3 Release mechanics (unchanged from existing process)
1. Bump version in `pyproject.toml` AND `src/pychebyshev/_version.py`
2. Update `CHANGELOG.md`
3. Commit on `main`
4. `gh release create vX.Y.Z` — triggers publish workflow → PyPI
5. `uv run mkdocs gh-deploy --force` — updates docs site
6. Update `docs/roadmap.md` — mark version as released, link to release notes

---

## 5. User-facing artifacts

### 5.1 `docs/roadmap.md` (NEW — public)
Short (~200 lines), one section per version with status emoji (planned / in-progress / released), one-paragraph summary of what ships, links to CHANGELOG and user-guide pages once released. Added to `mkdocs.yml` nav under "Roadmap" at the bottom of the tree.

### 5.2 `ROADMAP.md` at repo root (NEW — stub)
Three lines: "The canonical roadmap lives at https://0xc000005.github.io/PyChebyshev/roadmap/ — this file exists only for GitHub discoverability." Committed to git, links out.

### 5.3 `docs/superpowers/specs/` (NEW — internal)
Per-version design docs, committed to git but not added to public docs nav. This document itself is the first entry.

---

## 6. Cross-cutting commitments

- **No breaking changes to existing public API until v1.0.0.** All additions are new kwargs/methods with safe defaults. If a breaking change becomes necessary during the v0.x series, it goes through a deprecation warning in one minor release before removal at v1.0.0.
- **Semantic versioning respected.** Minor version bumps for each addition; patch bumps only for bug fixes between planned releases.
- **`from __future__ import annotations`** on all new modules (existing convention).
- **Python 3.10+ support maintained through v1.0.0** (may drop 3.10 at v1.0 if EOL timing aligns).
- **Numba/JIT stays deprecated.** Do not introduce new Numba dependencies. BLAS GEMV remains the fast path.
- **No new runtime dependencies.** numpy + scipy only. Documentation may use any mkdocs extension as needed.

---

## 7. Risk assessment and contingencies

| Risk | Version | Severity | Mitigation |
|---|---|---|---|
| ALS stalls or diverges on real problem sizes | v0.13 | HIGH | Spike branch `tt-als-spike` before API lock; fall-back `method='cross'` remains default |
| Special points + error_threshold interaction (v0.11 × v0.12) | v0.12 | medium | Define semantics explicitly: ε applies per piece, max_n applies per piece |
| Binary format design creeps to HDF5 / full schema evolution | v0.14 | medium | Lock to stdlib `struct` only; HDF5 is explicitly out of scope |
| Python 3.10 EOL mid-roadmap forces scheduling decision | any | low | Keep matrix at 3.10–3.13 through v0.15; reassess at v1.0 |
| MoCaX license change affects our ability to run `compare_*.py` locally | any | low | `compare_*.py` are already local-only and optional; CI doesn't depend on them |

---

## 8. Open questions (defer to per-version specs)

1. **v0.11**: doubling schedule — `n → 2n` vs `n → n+2` vs `n → ceil(1.5*n)`. MoCaX's exact heuristic is not documented; pick empirically via v0.11 spike.
2. **v0.13**: when `method='als'` vs `method='cross'` — documented decision matrix, or let users discover? Lean toward a short decision matrix in the user guide.
3. **v0.15**: should `set_descriptor()` accept structured data (dict) or just `str`? Start with `str` only (matches MoCaX).
4. **v1.0.0**: blog post — worth the effort? Defer decision to release time.

---

## 9. Success criteria for the roadmap itself

- [ ] All 6 releases published to PyPI
- [ ] v1.0.0 CHANGELOG lists every Tier-1/2/3 item with release ref
- [ ] Every release has: `tests/test_<feature>.py`, `compare_<feature>.py` (where applicable), `docs/user-guide/<feature>.md`, CHANGELOG entry, design spec in `docs/superpowers/specs/`
- [ ] Total test count grows from 457 (v0.10.1) to ~550+ by v1.0.0
- [ ] `docs/roadmap.md` kept in sync with actual releases

---

## 10. Next step

Invoke the `superpowers:writing-plans` skill to produce a detailed implementation plan for **v0.11 (Error-Driven Construction)** — the first release in the sequence. Subsequent releases each get their own implementation plan at the start of their cycle, not pre-planned now (to avoid premature freeze of decisions).
