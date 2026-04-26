# Build & Diagnostics (v0.19)

Three opt-in features to improve the experience of building large interpolants
with slow functions.

## Parallel build (Approximation, Spline)

Pass `n_workers=` to the constructor to evaluate `f` in parallel via
`concurrent.futures.ProcessPoolExecutor`:

```python
cheb = ChebyshevApproximation(
    expensive_f, 3, [[-1, 1]] * 3, [10, 10, 10],
    n_workers=4,           # 4 worker processes
)
cheb.build()

# Other accepted forms:
cheb = ChebyshevApproximation(..., n_workers=None)  # sequential (default)
cheb = ChebyshevApproximation(..., n_workers=-1)    # use os.cpu_count()
```

Constraints:

- `expensive_f` and `additional_data` must be picklable. Top-level functions
  and dicts of basic types work; closures over local variables don't.
- For functions that return in microseconds, the pool overhead can exceed
  parallelism gains — leave `n_workers=None` for cheap functions.

For `ChebyshevSpline`, `n_workers` propagates to every piece's underlying
`ChebyshevApproximation` build.

## Progress bars (`verbose=2`)

```python
cheb.build(verbose=2)  # opt-in tqdm progress bar
```

Available on all four classes. Existing `verbose=True/False` continues to
control text prints; `verbose=2` *adds* a tqdm progress bar.

If `tqdm` isn't installed, the build emits a one-time warning and proceeds
without a bar. Install via `pip install pychebyshev[viz]`.

## Plot helpers

```python
cheb.plot_1d()                          # 1-D source: plots f(x)
cheb.plot_1d(fixed={1: 0.5, 2: 0.0})    # multi-D: pin all but one
cheb.plot_2d_surface(n_points=50)       # 3-D surface plot
cheb.plot_2d_contour(n_levels=20)       # filled contour
cheb.plot_convergence(target_error=1e-6, max_n=64)  # Approximation only
```

All four classes support `plot_1d`, `plot_2d_surface`, `plot_2d_contour`.
`plot_convergence` is `ChebyshevApproximation`-only (Spline/Slider/TT have
different convergence regimes).

All return a matplotlib Axes — use `ax=` to compose with your own figures.

Requires `pip install pychebyshev[viz]`.
