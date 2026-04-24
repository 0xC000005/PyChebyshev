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

## v0.14 — Portable Binary Serialization :material-clock-outline:

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

## v0.15 — Ergonomics Bundle :material-clock-outline:

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

## v1.0.0 — Parity Announcement :material-clock-outline:

Feature-complete against MoCaX Intelligence 4.3.1. No new code — this release
is:

- Status bump `Beta` → `Production/Stable`
- API stability commitment going forward
- Summary CHANGELOG entry covering the parity items
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
