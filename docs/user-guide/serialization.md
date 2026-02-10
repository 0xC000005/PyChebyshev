# Saving & Loading Interpolants

## Why Save?

Building a Chebyshev interpolant is the expensive step — it evaluates your function
at every node in the tensor grid (e.g. $11^5 = 161{,}051$ evaluations for a 5-D
problem). Once built, evaluation takes microseconds.

Saving a built interpolant lets you:

- **Build once, evaluate forever** — skip the build step in production
- **Share models** — distribute pre-built interpolants to team members or across machines
- **Persist across sessions** — save your work and reload it later

## Saving a Built Interpolant

Both `ChebyshevApproximation` and `ChebyshevSlider` provide a `save()` method:

```python
import math
from pychebyshev import ChebyshevApproximation

def my_func(x, _):
    return math.sin(x[0]) * math.exp(-x[1])

cheb = ChebyshevApproximation(
    my_func, 2, [[-1, 1], [0, 2]], [15, 15]
)
cheb.build()

# Save to disk
cheb.save("interpolant.pkl")
```

For a `ChebyshevSlider`:

```python
from pychebyshev import ChebyshevSlider

slider = ChebyshevSlider(
    my_func, 2, [[-1, 1], [0, 2]], [15, 15],
    partition=[[0], [1]],
    pivot_point=[0.0, 1.0],
)
slider.build()

slider.save("slider.pkl")
```

## Loading an Interpolant

Use the `load()` class method — no rebuild needed:

```python
from pychebyshev import ChebyshevApproximation, ChebyshevSlider

# Load and evaluate immediately
cheb = ChebyshevApproximation.load("interpolant.pkl")
value = cheb.vectorized_eval([0.5, 1.0], [0, 0])

# Works the same for sliders
slider = ChebyshevSlider.load("slider.pkl")
value = slider.eval([0.5, 1.0], [0, 0])
```

The loaded `ChebyshevApproximation` supports all evaluation methods
(`vectorized_eval`, `fast_eval`, `vectorized_eval_multi`, `vectorized_eval_batch`).
The loaded `ChebyshevSlider` supports `eval` and `eval_multi`.

## Inspecting Objects

Use `repr()` for a compact summary and `print()` for a detailed view:

```python
cheb = ChebyshevApproximation.load("interpolant.pkl")

repr(cheb)
# ChebyshevApproximation(dims=2, nodes=[15, 15], built=True)

print(cheb)
# ChebyshevApproximation (2D, built)
#   Nodes:       [15, 15] (225 total)
#   Domain:      [-1, 1] x [0, 2]
#   Build:       0.002s, 225 evaluations
#   Derivatives: up to order 2
```

For a slider:

```python
print(slider)
# ChebyshevSlider (5D, 2 slides, built)
#   Partition: [[0, 1, 2], [3, 4]]
#   Pivot:     [100.0, 1.0, 0.25, 0.05, 0.2]
#   Nodes:     [11, 11, 11, 11, 11] (2,662 vs 161,051 full tensor)
#   Domain:    [80.0, 120.0] x [0.5, 2.0] x [0.01, 0.5] x [0.01, 0.1] x [0.05, 0.5]
#   Slides:
#     [0] dims [0, 1, 2]: 1,331 evals, built in 0.189s
#     [1] dims [3, 4]:     121 evals, built in 0.021s
```

This is useful for verifying that a loaded interpolant matches your expectations
before using it.

## Limitations

- **The original function is not saved.** Only the numerical data needed for
  evaluation (nodes, weights, tensor values, differentiation matrices) is persisted.
  After loading, `obj.function` is `None`.

- **Calling `build()` on a loaded object requires reassigning a function first:**

    ```python
    cheb = ChebyshevApproximation.load("interpolant.pkl")
    cheb.function = my_func  # reassign before rebuilding
    cheb.build()
    ```

- **Version compatibility.** If you load a file saved with a different version of
  PyChebyshev, a warning is emitted. Evaluation results should be identical unless
  internal data layout changed between versions.

!!! warning "Security"

    `load()` uses Python's `pickle` module internally. Pickle can execute arbitrary
    code during deserialization. **Only load files you trust.** Do not load
    interpolants from untrusted or unverified sources.
