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
