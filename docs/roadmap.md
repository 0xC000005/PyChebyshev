# Roadmap

PyChebyshev is on a path to **MoCaX Intelligence 4.3.1 feature parity** at **v1.0.0**.

The sequence below lands that parity across five focused minor releases. Each
release ships a coherent set of features, comprehensive tests, and full
documentation. Target: production-stable v1.0.0.

| Status | Meaning |
|---|---|
| :material-check: | Released |
| :material-progress-clock: | In progress |
| :material-clock-outline: | Planned |

---

## v0.11 — Error-Driven Construction :material-check:

Auto-calibrate node counts from a precision target instead of specifying
`n_nodes` by hand.

```python
# Instead of guessing N:
ChebyshevApproximation(f, 3, domain, n_nodes=[11, 11, 11])

# Target an error level directly:
ChebyshevApproximation(f, 3, domain, error_threshold=1e-6)

# Or mix: auto-pick dim 0, fix the rest
ChebyshevApproximation(f, 3, domain,
                       n_nodes=[None, 15, 15],
                       error_threshold=1e-6)
```

Also adds `ChebyshevApproximation.get_optimal_n1()` for pre-build capacity
estimates.

**Closes MoCaX gaps:** error-threshold ctor, semi-variable ctor,
`get_optimal_n1`.

## v0.12 — Special Points in the Core API :material-check:

Declare domain kinks at construction time on `ChebyshevApproximation` itself —
no need to reach for `ChebyshevSpline` manually. Optional per-sub-interval
node counts for fine-grained control.

```python
ChebyshevApproximation(
    f, 2, domain,
    n_nodes=[[15, 12], [11]],       # dim 0 split at 0.0 gets 15+12 nodes
    special_points=[[0.0], []],     # kink on dim 0 only
)
```

`ChebyshevSpline` remains available as the direct power-user API.

**Closes MoCaX gaps:** core `special_points` kwarg, per-sub-interval Ns.

## v0.13 — Tensor Train Algebra and ALS :material-check:

Alternating Least Squares as a new build method alongside TT-Cross and
TT-SVD, plus TT-level algebra primitives.

```python
tt = ChebyshevTT(f, 5, domain, n_nodes, tolerance=1e-6, max_rank=10)
tt.build(method='als')

# Refine an existing TT at fixed rank
tt.run_completion(tolerance=1e-8)

# TT inner product + orthogonalization sweeps
scalar = tt_a.inner_product(tt_b)
tt.orth_left(position=2)
```

Adds guidance on when to choose **Cross vs SVD vs ALS** as the build method.

**Closes MoCaX gaps:** rank-adaptive ALS, completion sweep, TT inner
product, TT orthogonalization.

## v0.14 — Portable Binary Serialization :material-check:

A language-agnostic `.pcb` binary format alongside the existing pickle-based
save/load. Consumers in C, Rust, or Julia can read PyChebyshev interpolants
without Python.

```python
cheb.save("model.pcb", format='binary')      # portable
cheb.save("model.pkl")                       # existing pickle, default
ChebyshevApproximation.load("model.pcb")     # auto-detects format
```

Format spec documented at `docs/user-guide/binary-format.md` — stdlib `struct`
only, no new dependencies.

**Closes MoCaX gaps:** cross-language binary serialization.

## v0.15 — Ergonomics Bundle :material-check:

Small quality-of-life additions aligned with MoCaX conventions.

- `additional_data=` constructor kwarg (formalizes the second-arg context
  pattern)
- `get_derivative_id(orders)` + `eval(point, derivative_id=...)` — stable
  integer IDs for repeated partials
- `set_descriptor(str)` / `get_descriptor()` — label your interpolants
- `get_used_ns()`, `get_constructor_type()`, `is_construction_finished()` —
  introspection

All additive, no breaking changes.

**Closes MoCaX gaps:** `additional_data`, `get_derivative_id`, descriptor,
introspection.

## v0.16 — Polish Bundle :material-clock-outline:

Final cosmetic mirror of the MoCaX 4.3.1 surface. Strictly additive, no
breaking changes.

- `clone()` — deep copy on all four classes
- `get_special_points()`, `get_max_derivative_order()`,
  `get_error_threshold()` — instance getters
- `get_evaluation_points()` / `get_num_evaluation_points()` — flat
  `(N, num_dim)` grid post-construction (MoCaX-style)
- `set_original_function_values(values)` — in-place deferred
  construction (alt to the `from_values()` factory)
- `peek_format_version(filename)` static — read `.pcb` version without
  deserializing
- `is_dimensionality_allowed(num_dim)` static — pre-flight capability check
- Optional typed helpers `Domain`, `Ns`, `SpecialPoints` (constructors
  accept both raw lists and these dataclasses)

**Closes MoCaX gaps:** the last cosmetic surface methods.

## v0.17 — Integrate Everywhere :material-clock-outline:

Extend the v0.9 calculus toolkit so every class supports integration.

- `ChebyshevSlider.integrate(dims=None, bounds=None)` — full and partial
  integration via the sliding-decomposition closed form
- `ChebyshevTT.integrate(dims=None, bounds=None)` — full and partial
  integration via Fejér-1 weight contraction into TT cores

After v0.17, `integrate()` is available on all four classes. Roots and
min/max on Slider/TT remain deferred to v0.21.

**Beyond MoCaX:** MoCaX has no `integrate()` API on any class.

## v0.18 — TT Feature Parity :material-clock-outline:

Bring `ChebyshevTT` up to parity with `ChebyshevApproximation` for the
non-calculus surface.

- TT `extrude()` — add a dim by appending a constant core
- TT `slice()` — fix a dim by contracting a core via barycentric weights
- TT `from_values()` classmethod — build from a precomputed full tensor
  (skip TT-Cross)
- TT `nodes()` static — generate the Chebyshev grid without function eval
- TT algebra `+`, `-`, `*` (scalar) — core stacking + rounding
- `to_dense()` / `from_dense()` — convert between TT and Approximation

**Beyond MoCaX:** richer TT primitives than MoCaXExtend exposes.

## v0.19 — Build & Diagnostics :material-clock-outline:

Ergonomics for users with slow `f` or large grids.

- Parallel function evaluation at build time (`n_workers=` ctor kwarg via
  `concurrent.futures.ProcessPoolExecutor`)
- `tqdm`-based progress bars during construction (opt-in, `verbose=2`)
- `plot_convergence(target_error)` helper — builds at increasing N,
  plots error decay
- Visualization helpers: `plot_1d`, `plot_2d_surface`, `plot_2d_contour`

New optional dependency group `pychebyshev[viz]` for matplotlib + tqdm.

**Beyond MoCaX:** MoCaX has neither parallel build nor visualization
helpers.

## v0.20 — Adaptive Refinement + Interop :material-clock-outline:

Smart node placement plus delivery of the v0.14 `.pcb` portability promise.

- Auto-knot detection for `ChebyshevSpline` — scan for kinks via
  curvature/derivative discontinuity, place knots automatically
- Sobol indices computed from spectral coefficients (cheap once the
  interpolant exists)
- Auto dimension reordering by importance (helps TT rank growth)
- Reference `.pcb` reader implementations in **Rust** and **Julia** as
  separate sub-repos (`0xC000005/pcb-readers`)

**Beyond MoCaX:** MoCaX is closed-source; `.pcb` ecosystem is
PyChebyshev-unique.

## v0.21 — Advanced Calculus :material-clock-outline:

Research-grade extensions to close the remaining calculus surface.

- N-D rootfinding via Möller–Stetter colleague matrices
  (Trefethen 2017 ch. 24) on `ChebyshevApproximation` and
  `ChebyshevSpline`
- `roots()`, `minimize()`, `maximize()` on `ChebyshevSlider` and
  `ChebyshevTT` (1-D-at-a-time + Brent's method bracket where no closed
  form exists)
- Higher-order partial derivatives on demand (currently capped at
  construction-time `max_derivative_order`)

**Beyond MoCaX:** spectral N-D rootfinding has no MoCaX analog.

## v1.0.0 — Parity Announcement :material-clock-outline:

Feature-complete against MoCaX Intelligence 4.3.1, plus the beyond-MoCaX
extensions from v0.16–v0.21. No new code — this release is:

- Status bump `Beta` → `Production/Stable`
- API stability commitment going forward
- Summary CHANGELOG entry covering the parity arc + extensions
- Refreshed README with the final performance comparison

---

## Conventions per release

Every release on this roadmap follows the same template:

- **Python correctness tests** in `tests/test_<feature>.py` — runs in CI on
  Python 3.10–3.13
- **MoCaX value-match benchmark** in `compare_<feature>.py` at the repo root —
  local-only, not in CI (MoCaX libraries are proprietary and gitignored)
- **User guide page** at `docs/user-guide/<feature>.md` — motivation,
  mathematical justification with citations, API reference, usage examples
- **CHANGELOG.md** entry under a new version heading
- **README update** for major capabilities (Tier 1 items)

Each release is preceded by an internal design spec and implementation plan
kept in the author's working tree (not published). The user-guide page and
CHANGELOG entry capture everything users need at release time.

---

## Non-goals

- **Not a drop-in MoCaX API replacement.** APIs are PyChebyshev-idiomatic;
  capabilities are equivalent.
- **Not pursuing** GPU acceleration, multi-threading, AAD, or log/reciprocal
  domain transforms — none are offered by MoCaX either, per the MoCaX 4.3.1
  manuals.
- **Not replacing pickle serialization** — `.pcb` is additive, pickle stays
  the Python-to-Python default.

---

## Post-v1.0.0

Open to community-driven additions. The roadmap above covers documented MoCaX
capabilities. Post-parity priorities will be user-driven — file an issue if
there's a feature you need.
