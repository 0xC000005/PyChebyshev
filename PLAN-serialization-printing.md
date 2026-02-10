# Implementation Plan: Serialization & Printing for PyChebyshev

## Overview

Add save/load (serialization) and human-readable printing (`__repr__`/`__str__`) to
both `ChebyshevApproximation` and `ChebyshevSlider`. Include full test coverage
and documentation.

---

## 1. Serialization

### 1.1 Approach: pickle with `__getstate__` / `__setstate__`

**Why pickle?**

- `ChebyshevSlider` contains a list of `ChebyshevApproximation` objects, each with
  multiple numpy arrays. Pickle handles nested objects and mixed types natively.
- `.npz` is great for flat arrays but awkward for this nested structure.
- This is the standard pattern in scientific Python (scikit-learn, statsmodels, etc.).

**Core insight:** `self.function` (a callable, often a lambda/closure) is not reliably
picklable — but it is **not needed after `build()`**. All evaluation uses only
numerical data (`tensor_values`, `nodes`, `weights`, `diff_matrices`).

### 1.2 `ChebyshevApproximation` methods

#### `__getstate__(self) -> dict`

Returns `self.__dict__` copy with two modifications:
- `function` replaced with `None` (not reliably picklable, not needed for eval)
- `_eval_cache` excluded (reconstructed on load; it's just pre-allocated zero arrays
  derived from `n_nodes`, so serializing it wastes space)
- `_pychebyshev_version` key added with current `__version__` string

#### `__setstate__(self, state: dict)`

Restores all state from the dict, then:
- Sets `self.function = None`
- Reconstructs `self._eval_cache` from `self.n_nodes` (same logic as in `build()`,
  lines 244-247 of barycentric.py):
  ```python
  self._eval_cache = {}
  if self.tensor_values is not None:
      for d in range(self.num_dimensions - 1, 0, -1):
          shape = tuple(self.n_nodes[i] for i in range(d))
          self._eval_cache[d] = np.zeros(shape)
  ```
- Checks `_pychebyshev_version` against current `__version__` and emits
  `warnings.warn()` if they differ (does NOT error — just warns)

#### `save(self, path: str | pathlib.Path) -> None`

- Raises `RuntimeError` if the object has not been built (`self.tensor_values is None`)
- Writes `pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)` to `path`
- Accepts both `str` and `pathlib.Path` via `os.fspath()`

#### `load(cls, path: str | pathlib.Path) -> ChebyshevApproximation` (classmethod)

- Reads `pickle.load(f)` from `path`
- Validates the loaded object is a `ChebyshevApproximation` instance
- Accepts both `str` and `pathlib.Path` via `os.fspath()`
- Docstring includes a **security warning** that pickle can execute arbitrary code,
  so only load trusted files

**After loading:** `eval()`, `vectorized_eval()`, `fast_eval()`, and all other eval
methods work. `build()` raises because `function is None` — users can reassign
`.function` if they want to rebuild.

### 1.3 `ChebyshevSlider` methods

Same pattern as above, with these specifics:

#### `__getstate__(self) -> dict`

- `function` replaced with `None`
- `_pychebyshev_version` added
- The `slides` list contains `ChebyshevApproximation` objects — pickle automatically
  calls each slide's `__getstate__`, so nested serialization works naturally
- `_dim_to_slide` is a simple `{int: int}` dict, pickles fine

#### `__setstate__(self, state: dict)`

- Restores all state
- `self.function = None`
- Version mismatch warning (same as above)
- No `_eval_cache` to reconstruct (slider delegates to slides, which reconstruct
  their own caches)

#### `save(self, path) -> None`

- Raises `RuntimeError` if `self._built is False`
- Same pickle logic as `ChebyshevApproximation.save()`

#### `load(cls, path) -> ChebyshevSlider` (classmethod)

- Same pickle logic, validates instance type is `ChebyshevSlider`
- Security warning in docstring

---

## 2. Printing

### 2.1 `ChebyshevApproximation`

#### `__repr__(self) -> str`

Compact one-liner for REPL/debugger:
```
ChebyshevApproximation(dims=3, nodes=[11, 11, 11], built=True)
ChebyshevApproximation(dims=2, nodes=[15, 15], built=False)
```

#### `__str__(self) -> str`

Detailed multi-line summary via `print()`:

**Built state:**
```
ChebyshevApproximation (3D, built)
  Nodes:       [11, 11, 11] (1,331 total)
  Domain:      [80.0, 120.0] x [0.5, 2.0] x [0.01, 0.5]
  Build:       0.234s, 1,331 evaluations
  Derivatives: up to order 2
```

**Unbuilt state:**
```
ChebyshevApproximation (3D, not built)
  Nodes:       [11, 11, 11] (1,331 total)
  Domain:      [80.0, 120.0] x [0.5, 2.0] x [0.01, 0.5]
  Derivatives: up to order 2
```

**High-dimensional truncation (>6 dims):**
```
ChebyshevApproximation (10D, built)
  Nodes:       [11, 11, 11, 11, 11, 11, ...] (25,937,424,601 total)
  Domain:      [0.0, 1.0] x [0.0, 1.0] x [0.0, 1.0] x [0.0, 1.0] x [0.0, 1.0] x [0.0, 1.0] x ...
  Build:       12.345s, 25,937,424,601 evaluations
  Derivatives: up to order 2
```

### 2.2 `ChebyshevSlider`

#### `__repr__(self) -> str`

```
ChebyshevSlider(dims=5, slides=2, partition=[[0, 1, 2], [3, 4]], built=True)
ChebyshevSlider(dims=5, slides=2, partition=[[0, 1, 2], [3, 4]], built=False)
```

#### `__str__(self) -> str`

**Built state:**
```
ChebyshevSlider (5D, 2 slides, built)
  Partition: [[0, 1, 2], [3, 4]]
  Pivot:     [100.0, 1.0, 0.25, 0.05, 0.2]
  Nodes:     [11, 11, 11, 11, 11] (2,662 vs 161,051 full tensor)
  Domain:    [80.0, 120.0] x [0.5, 2.0] x [0.01, 0.5] x [0.01, 0.1] x [0.05, 0.5]
  Slides:
    [0] dims [0, 1, 2]: 1,331 evals, built in 0.189s
    [1] dims [3, 4]:     121 evals, built in 0.021s
```

**Unbuilt state:**
```
ChebyshevSlider (5D, 2 slides, not built)
  Partition: [[0, 1, 2], [3, 4]]
  Pivot:     [100.0, 1.0, 0.25, 0.05, 0.2]
  Nodes:     [11, 11, 11, 11, 11] (2,662 vs 161,051 full tensor)
  Domain:    [80.0, 120.0] x [0.5, 2.0] x [0.01, 0.5] x [0.01, 0.1] x [0.05, 0.5]
```

**High-dimensional truncation:** same >6 dims rule for nodes/domain lines.
Partition and pivot also truncated if >6 groups/dims.

---

## 3. Code Changes

All changes go into existing files — no new modules.

| File | Changes |
|------|---------|
| `src/pychebyshev/barycentric.py` | Add `import pickle, os, warnings, pathlib`; add `__getstate__`, `__setstate__`, `save`, `load`, `__repr__`, `__str__` to `ChebyshevApproximation` |
| `src/pychebyshev/slider.py` | Same imports; add same 6 methods to `ChebyshevSlider` |
| `src/pychebyshev/__init__.py` | No changes (classes already exported) |

---

## 4. Tests

Added to existing test files. Use pytest `tmp_path` fixture for temp file cleanup.

### 4.1 `test_barycentric.py` — new `TestSerialization` class

| Test | What it verifies |
|------|------------------|
| `test_save_load_roundtrip` | Build, save, load, then `vectorized_eval` at **5 diverse points** matches original within machine precision (`np.testing.assert_allclose`) |
| `test_fast_eval_after_load` | `fast_eval()` works after load (verifies `_eval_cache` reconstruction) |
| `test_function_is_none_after_load` | `loaded.function is None` |
| `test_loaded_state_attributes` | `tensor_values`, `weights`, `diff_matrices`, `nodes` are all present and correct shapes |
| `test_save_before_build_raises` | `save()` on unbuilt object raises `RuntimeError` |
| `test_version_mismatch_warning` | Monkey-patch version in saved state, load, verify `warnings.warn` fires |
| `test_pathlib_path` | `save(pathlib.Path(...))` and `load(pathlib.Path(...))` both work |

### 4.2 `test_barycentric.py` — new `TestRepr` class

| Test | What it verifies |
|------|------------------|
| `test_repr_unbuilt` | Contains `built=False`, dims, nodes |
| `test_repr_built` | Contains `built=True`, dims, nodes |
| `test_str_unbuilt` | Contains `not built`, nodes, domain; does NOT contain `Build:` line |
| `test_str_built` | Contains `built`, nodes, domain, build time, evaluations |

### 4.3 `test_slider.py` — new `TestSliderSerialization` class

| Test | What it verifies |
|------|------------------|
| `test_save_load_roundtrip` | Build, save, load, then `eval` at **5 diverse points** matches original |
| `test_derivative_after_load` | Derivatives work after load (tests `_dim_to_slide` routing) |
| `test_function_is_none_after_load` | `loaded.function is None` and `loaded.slides[i].function is None` |
| `test_slider_internal_state` | `loaded._dim_to_slide` matches original; `loaded.pivot_value` matches; `loaded.partition` matches |
| `test_save_before_build_raises` | `save()` on unbuilt slider raises `RuntimeError` |
| `test_pathlib_path` | Pathlib paths work for slider save/load |

### 4.4 `test_slider.py` — new `TestSliderRepr` class

| Test | What it verifies |
|------|------------------|
| `test_repr_unbuilt` | Contains `built=False`, dims, slides count, partition |
| `test_repr_built` | Contains `built=True` |
| `test_str_unbuilt` | Contains `not built`, partition, pivot, nodes, domain |
| `test_str_built` | Contains `built`, partition, pivot, slides detail lines |

---

## 5. Documentation

### 5.1 New page: `docs/user-guide/serialization.md`

Structure:
```
# Saving & Loading Interpolants

## Why Save?
  - Build is expensive (evaluates function at all grid points), eval is cheap
  - Build once, deploy/reuse everywhere
  - Share pre-built models across processes, machines, or team members

## Saving a Built Interpolant
  - Code example: build → save → file on disk
  - Works with both ChebyshevApproximation and ChebyshevSlider

## Loading an Interpolant
  - Code example: load → eval (no rebuild needed)
  - Loaded objects can evaluate but cannot rebuild (function is not saved)
  - Show reassigning .function if rebuild is needed

## Inspecting Objects
  - repr() for quick summary (show output)
  - print() for detailed view (show output)
  - Useful for verifying loaded objects match expectations

## Limitations
  - The original function is not saved (not needed for evaluation)
  - Calling build() on a loaded object raises RuntimeError unless .function is reassigned

## Security
  !!! warning
      `load()` uses Python's pickle module internally. Pickle can execute
      arbitrary code during deserialization. **Only load files you trust.**
      Do not load interpolants from untrusted or unverified sources.
```

### 5.2 Update `mkdocs.yml` nav

```yaml
nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - User Guide:
      - Mathematical Concepts: user-guide/concepts.md
      - Usage Patterns: user-guide/usage.md
      - Saving & Loading: user-guide/serialization.md      # NEW
      - Computing Greeks: user-guide/greeks.md
      - Sliding Technique: user-guide/sliding.md
  - API Reference: api/reference.md
  - Benchmarks: benchmarks.md
```

### 5.3 Update `docs/index.md`

Add a feature bullet to the "Key Features" list:
```
- **Save & load** — persist built interpolants to disk; rebuild-free deployment
```

### 5.4 Update `docs/getting-started.md`

Add a new section after "4. Evaluate price + all Greeks at once":

```
### 5. Save for later

Save the built interpolant to skip rebuilding next time:

    cheb.save("my_interpolant.pkl")

Load it back — no rebuild needed:

    from pychebyshev import ChebyshevApproximation
    cheb = ChebyshevApproximation.load("my_interpolant.pkl")
    value = cheb.vectorized_eval([1.0, 2.0], [0, 0])
```

### 5.5 Update `CHANGELOG.md`

Add under a new `[Unreleased]` section (or bump version if appropriate):

```markdown
## [Unreleased]

### Added

- `save()` and `load()` methods on `ChebyshevApproximation` and `ChebyshevSlider`
  for persisting built interpolants to disk (pickle-based)
- `__repr__` and `__str__` methods on both classes for human-readable printing
- Version compatibility check on load with warning for mismatched versions
- New documentation page: Saving & Loading Interpolants
- 18 new tests for serialization and printing
```

---

## 6. Implementation Order

1. `barycentric.py` — add 6 methods + imports to `ChebyshevApproximation`
2. `slider.py` — add 6 methods + imports to `ChebyshevSlider`
3. `test_barycentric.py` — add `TestSerialization` and `TestRepr` classes
4. `test_slider.py` — add `TestSliderSerialization` and `TestSliderRepr` classes
5. Run full test suite (`uv run pytest tests/ -v`) — verify all 44 existing + new tests pass
6. `docs/user-guide/serialization.md` — new doc page
7. `mkdocs.yml` — add nav entry
8. `docs/index.md` — add feature bullet
9. `docs/getting-started.md` — add step 5
10. `CHANGELOG.md` — add unreleased section
11. Commit and push

---

## 7. Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| Pickle over `.npz` | Handles nested `ChebyshevSlider` → `ChebyshevApproximation` naturally |
| Exclude `function` from pickle | Lambdas/closures not reliably picklable; not needed for eval |
| Exclude `_eval_cache` from pickle | Trivially reconstructed from `n_nodes`; saves file size |
| Embed version in state | Detect breaking changes across library versions |
| Warn (not error) on version mismatch | Internal state may not change between versions; don't block users unnecessarily |
| Raise on save-before-build | Prevents confusing loaded objects with no data |
| Truncate `__str__` at >6 dims | Keeps output readable for high-dimensional problems |
| Security warning in docstring + docs | Standard practice for pickle-based load APIs |
