# Ergonomics

PyChebyshev v0.15 added four small, additive features for managing
interpolant metadata and derivative-orders ergonomically. All are MoCaX-parity
conveniences; none change numerical behavior.

## `additional_data=` constructor kwarg

Thread context into the user function during build without writing a closure:

```python
def black_scholes(point, data):
    S, K, T = point
    r = data["rate"]
    sigma = data["sigma"]
    # ... pricing logic that uses r, sigma ...

cheb = ChebyshevApproximation(
    black_scholes, 3, domain, [11, 11, 11],
    additional_data={"rate": 0.05, "sigma": 0.2},
)
cheb.build()
```

The value is stored on `self.additional_data` (public attribute, mutable).
The function receives it as the second positional arg at every callsite during
`build()`. After build, mutating `additional_data` has no effect on the
already-baked tensor values.

### Persistence

| Path | Behavior |
|---|---|
| Pickle save/load | Preserved (free). |
| Binary `.pcb` save | Raises `NotImplementedError` if `additional_data is not None`. Pass `format='pickle'` for full persistence. |
| Binary `.pcb` load | `self.additional_data = None`. |

## `set_descriptor` / `get_descriptor`

Label your interpolants:

```python
cheb.set_descriptor("vega-rates-curve-A")
print(cheb.get_descriptor())  # "vega-rates-curve-A"
```

Default is `""`. Mutable any time. Pickle preserves; binary `.pcb` resets to
`""` on load.

## `get_derivative_id` registry

Stable session-local integer IDs for repeated partials:

```python
vega_id = cheb.get_derivative_id([0, 0, 1])    # 0
gamma_id = cheb.get_derivative_id([2, 0, 0])   # 1
cheb.get_derivative_id([0, 0, 1])              # 0 again — same orders → same id

price = cheb.eval(point, derivative_order=[0, 0, 0])
vega  = cheb.eval(point, derivative_id=vega_id)
```

Available on `ChebyshevApproximation`, `ChebyshevSpline`, and `ChebyshevSlider`
(not on `ChebyshevTT` — TT uses finite differences and has no
`derivative_order` arg in its `eval`).

`eval(..., derivative_order=..., derivative_id=...)` requires exactly one of
the two; both → `ValueError`, neither → `ValueError`, unknown id → `KeyError`.

IDs are session-local. Pickle preserves the registry; binary `.pcb` does not.

## Introspection trio

```python
cheb.is_construction_finished()    # bool — True after build / from_values / load / algebra
cheb.get_constructor_type()        # "ChebyshevApproximation" (class name string)
cheb.get_used_ns()                 # [11, 11, 11]  (post-build resolved n_nodes)
```

Available on all four classes.

- `is_construction_finished()` is False after bare `__init__` only; True after
  any successful `build()`, `from_values()`, `load()`, algebra, `extrude()`,
  `slice()`, or `integrate()` along a dim.
- `get_constructor_type()` returns the class name.
- `get_used_ns()` returns the per-dim node count list. For
  `ChebyshevSpline`, the nested vs flat shape per dim is preserved.

## Factory-derived interpolants

Operations that produce a new interpolant from existing ones — `extrude()`,
`slice()`, partial `integrate()`, algebra (`+`, `-`, `*`, `/`), and binary
`.pcb` `load()` — return a *fresh* object: empty `descriptor`, `additional_data
= None`, and an empty derivative-id registry. The source's metadata is not
inherited. Re-attach metadata on the result if needed.

```python
extruded = cheb.extrude((3, (-1.0, 1.0), 5))
assert extruded.get_descriptor() == ""           # not inherited
extruded.set_descriptor("derivative-of-source")  # re-attach if useful
```

Pickle `load()` is the exception — it preserves the saved object's metadata
(descriptor, additional_data, registry) faithfully.

## MoCaX correspondence

| MoCaX C++ | PyChebyshev v0.15 |
|---|---|
| `setAdditionalData(payload)` | `additional_data=` ctor kwarg |
| `getDescriptor()` / `setDescriptor(s)` | `get_descriptor()` / `set_descriptor(s)` |
| `getDerivativeId(orders)` | `get_derivative_id(orders)` |
| `eval(point, derivativeId)` | `eval(point, derivative_id=...)` |
| `getConstructorType()` | `get_constructor_type()` |
| `getUsedNs()` | `get_used_ns()` |
| `isConstructionFinished()` | `is_construction_finished()` |

---

## v0.16 polish surface

The v0.16 release adds the final cosmetic mirror of the MoCaX 4.3.1 API.
All additions are strictly additive.

### `clone()`

Return an independent deep copy of any interpolant:

```python
cheb2 = cheb1.clone()
cheb2.set_descriptor("variant")
# cheb1 is untouched
```

Available on `ChebyshevApproximation`, `ChebyshevSpline`, `ChebyshevSlider`,
`ChebyshevTT`. Like `save()`/`load()`, the source `function` callable is not
duplicated — the clone has `function = None`.

### Instance getters

| Method | Available on | Returns |
|---|---|---|
| `get_max_derivative_order()` | all four | `int` |
| `get_error_threshold()` | Approximation, Spline | `float \| None` |
| `get_special_points()` | Approximation, Spline | `list[list[float]] \| None` |
| `get_evaluation_points()` | all four | `np.ndarray` shape `(N, num_dim)` |
| `get_num_evaluation_points()` | all four | `int` |

`get_evaluation_points()` returns the flat `(N, num_dim)` grid (MoCaX style,
matches `len(get_evaluation_points()) == get_num_evaluation_points()`).

### Static helpers

```python
ChebyshevApproximation.peek_format_version("model.pcb")  # → 1

ChebyshevTT.is_dimensionality_allowed(5)  # → True
```

### Deferred construction (Approximation, Spline)

Construct a grid-only interpolant, then fill its tensor in place:

```python
cheb = ChebyshevApproximation(None, 2, [[-1,1],[-1,1]], [10,10],
                               defer_build=True)

# Compute values externally — e.g. on a distributed cluster
points = cheb.get_evaluation_points()  # (100, 2)
values = compute_function_on_cluster(points).reshape(10, 10)

cheb.set_original_function_values(values)
# now evaluable
result = cheb.eval([0.3, -0.4], [0, 0])
```

This is the in-place analog of the `from_values()` factory. Bit-identical
results in both paths.

### Optional typed helpers

```python
from pychebyshev import Domain, Ns, SpecialPoints, ChebyshevApproximation

cheb = ChebyshevApproximation(
    f, 2,
    Domain([(0.0, 1.0), (0.0, 1.0)]),  # equivalent to [[0,1],[0,1]]
    Ns([15, 15]),                       # equivalent to [15, 15]
)
```

Constructors of all four classes accept both raw lists and these frozen
dataclasses.
