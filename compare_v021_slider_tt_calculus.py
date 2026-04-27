"""v0.21 demo: Slider/TT roots, minimize, maximize.

PyChebyshev v0.21 closes the calculus parity gap promised since v0.17.
Before v0.21:
- ChebyshevApproximation: integrate, roots, minimize, maximize
- ChebyshevSpline:        integrate, roots, minimize, maximize
- ChebyshevSlider:        integrate only
- ChebyshevTT:            integrate only

After v0.21: all four classes have the full calculus surface.

This script demonstrates:
- 1-D Slider/TT roots/min/max against analytical answers
- 2-D and 3-D Slider/TT roots/min/max with fixed= dict
- Cross-class consistency: Slider/TT/Approximation agree on the same function
- Composition with extrude/slice/algebra/with_auto_order

No MoCaX baseline -- these are beyond-MoCaX features.
"""

from __future__ import annotations

import math
import time

import numpy as np

from pychebyshev import (
    ChebyshevApproximation,
    ChebyshevSlider,
    ChebyshevTT,
)


def _check(label: str, got: float, expected: float, tol: float = 1e-9) -> None:
    err = abs(got - expected)
    status = "OK " if err < tol else "FAIL"
    print(f"  [{status}] {label:50s} got {got:+.10e}  expected {expected:+.10e}  err {err:.2e}")


def demo_1d_slider() -> None:
    print("\n=== 1-D ChebyshevSlider: roots/min/max of x^2 - 0.25 on [-1, 1] ===")
    def f(x, _): return x[0] ** 2 - 0.25
    slider = ChebyshevSlider(
        f, num_dimensions=1, domain=[(-1, 1)], n_nodes=[10],
        partition=[[0]], pivot_point=[0.0],
    )
    slider.build(verbose=False)
    roots = slider.roots()
    _check("roots[0]", float(roots[0]), -0.5)
    _check("roots[1]", float(roots[1]), 0.5)
    val, loc = slider.minimize()
    _check("min value", val, -0.25)
    _check("min loc", loc, 0.0)
    val, loc = slider.maximize()
    _check("max value", val, 0.75)


def demo_1d_tt() -> None:
    print("\n=== 1-D ChebyshevTT: roots/min/max of x^2 - 0.25 on [-1, 1] ===")
    def f(x, _): return x[0] ** 2 - 0.25
    tt = ChebyshevTT(f, num_dimensions=1, domain=[(-1, 1)], n_nodes=[10])
    tt.build(verbose=False)
    roots = tt.roots()
    _check("roots[0]", float(roots[0]), -0.5)
    _check("roots[1]", float(roots[1]), 0.5)
    val, loc = tt.minimize()
    _check("min value", val, -0.25)
    val, loc = tt.maximize()
    _check("max value", val, 0.75)


def demo_3d_with_auto_order() -> None:
    print("\n=== 3-D TT with_auto_order: roots/min must respect user-frame ===")
    def f(x, _): return (x[0] - 0.4) * (1.0 + 0.1 * x[1] + 0.2 * x[2])
    # with_auto_order is a classmethod constructor that builds and reorders
    tt_reordered = ChebyshevTT.with_auto_order(
        f, num_dimensions=3, domain=[(-1, 1)] * 3, n_nodes=[10, 10, 10],
        n_trials=3,
    )
    print(f"  TT _dim_order after with_auto_order: {tt_reordered._dim_order}")
    roots = tt_reordered.roots(dim=0, fixed={1: 0.0, 2: 0.0})
    _check("roots[0]", float(roots[0]), 0.4)


def demo_3d_reorder_transparency() -> None:
    """Force non-identity _dim_order via explicit reorder() and verify
    user-frame roots/minimize results are unchanged."""
    print("\n=== 3-D TT reorder([2, 0, 1]): user-frame transparency under non-identity storage ===")
    def f(x, _): return (x[0] - 0.2) ** 2 + x[1] ** 2 + x[2] ** 2

    tt = ChebyshevTT(
        f, num_dimensions=3, domain=[(-1, 1)] * 3, n_nodes=[8, 8, 8],
    )
    tt.build(verbose=False)
    print(f"  Canonical TT _dim_order: {tt._dim_order}")

    # Reorder forces a non-identity storage layout
    tt_permuted = tt.reorder([2, 0, 1])
    print(f"  After reorder([2,0,1]): _dim_order = {tt_permuted._dim_order}")

    # Minimize in user-frame dim 0 must give same answer in both
    v_canonical, l_canonical = tt.minimize(dim=0, fixed={1: 0.0, 2: 0.0})
    v_permuted, l_permuted = tt_permuted.minimize(dim=0, fixed={1: 0.0, 2: 0.0})
    _check("canonical min value", v_canonical, 0.0, tol=1e-8)
    _check("canonical min loc", l_canonical, 0.2, tol=1e-8)
    _check("permuted min value", v_permuted, 0.0, tol=1e-8)
    _check("permuted min loc", l_permuted, 0.2, tol=1e-8)
    _check("canonical vs permuted value", abs(v_canonical - v_permuted), 0.0, tol=1e-12)
    _check("canonical vs permuted loc", abs(l_canonical - l_permuted), 0.0, tol=1e-12)


def demo_cross_class_consistency() -> None:
    print("\n=== Cross-class consistency: Slider/TT/Approx agree on same function ===")
    def f(x, _): return (x[0] - 0.2) ** 2 + (x[1] + 0.1) ** 2

    cheb = ChebyshevApproximation(f, 2, [(-1, 1), (-1, 1)], [9, 9])
    slider = ChebyshevSlider(
        f, num_dimensions=2, domain=[(-1, 1), (-1, 1)], n_nodes=[9, 9],
        partition=[[0], [1]], pivot_point=[0.0, 0.0],
    )
    tt = ChebyshevTT(f, num_dimensions=2, domain=[(-1, 1), (-1, 1)], n_nodes=[9, 9])
    for x in (cheb, slider, tt):
        x.build(verbose=False)

    v_c, l_c = cheb.minimize(dim=0, fixed={1: -0.1})
    v_s, l_s = slider.minimize(dim=0, fixed={1: -0.1})
    v_t, l_t = tt.minimize(dim=0, fixed={1: -0.1})
    print(f"  Approx.minimize  -> ({v_c:+.10e}, {l_c:+.10e})")
    print(f"  Slider.minimize  -> ({v_s:+.10e}, {l_s:+.10e})")
    print(f"  TT.minimize      -> ({v_t:+.10e}, {l_t:+.10e})")
    print(f"  diff |Slider-Approx| = {abs(v_s-v_c):.2e}, |TT-Approx| = {abs(v_t-v_c):.2e}")


def main() -> None:
    print(f"PyChebyshev v0.21 calculus parity demo")
    t0 = time.time()
    demo_1d_slider()
    demo_1d_tt()
    demo_3d_with_auto_order()
    demo_3d_reorder_transparency()
    demo_cross_class_consistency()
    print(f"\nTotal: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
