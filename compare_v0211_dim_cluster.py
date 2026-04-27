"""v0.21.1 demo: TT _dim_order cluster fixes.

Demonstrates each correctness fix with a non-uniform-domain TT under
explicit reorder([2, 0, 1]) — the case that v0.20.1 / v0.21.0 tests
masked with uniform domains.
"""

from __future__ import annotations

import time

import numpy as np

from pychebyshev import ChebyshevApproximation, ChebyshevTT


def _check(label: str, ok: bool, detail: str = "") -> None:
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {label}{(' — ' + detail) if detail else ''}")


def demo_inner_product_strict() -> None:
    print("\n=== inner_product strict _dim_order check (Item B) ===")
    def f(x, _): return x[0] ** 2 + x[1]
    tt = ChebyshevTT(f, num_dimensions=2, domain=[(-1, 1), (-1, 1)], n_nodes=[5, 5])
    tt.build(verbose=False)
    tt_p = tt.reorder([1, 0])
    try:
        tt.inner_product(tt_p)
        _check("ValueError raised on dim_order mismatch", False)
    except ValueError as e:
        _check("ValueError raised on dim_order mismatch", True, str(e)[:80])


def demo_get_evaluation_points_round_trip() -> None:
    print("\n=== get_evaluation_points user-frame round-trip (Item C) ===")
    def f(x, _): return 0.3 * x[0] + 0.7 * x[1] - 0.2 * x[2]
    tt = ChebyshevTT(
        f, num_dimensions=3,
        domain=[(-1, 1), (-2, 2), (-3, 3)], n_nodes=[5, 5, 5],
    )
    tt.build(verbose=False)
    tt_p = tt.reorder([2, 0, 1])
    points = tt_p.get_evaluation_points()
    max_err = 0.0
    for i in range(0, len(points), 25):
        pt = points[i]
        expected = f(pt, None)
        got = float(tt_p.eval(pt.tolist()))
        max_err = max(max_err, abs(got - expected))
    _check("round-trip eval == f for non-identity _dim_order",
           max_err < 1e-9, f"max_err={max_err:.2e}")


def demo_roots_user_frame_validation() -> None:
    print("\n=== roots/min/max validate against user-frame domain (Item A) ===")
    # User-frame dim 1 has domain [-2, 2]; storage-frame after reorder has different range
    def f(x, _): return (x[0] - 0.4) * (1.0 + 0.0 * x[1] + 0.0 * x[2])
    tt = ChebyshevTT(
        f, num_dimensions=3,
        domain=[(-1, 1), (-2, 2), (-3, 3)], n_nodes=[8, 8, 8],
    )
    tt.build(verbose=False)
    tt_p = tt.reorder([2, 0, 1])
    # fixed=1.5 is valid in user-frame dim 1, NOT in storage-frame after reorder
    try:
        roots = tt_p.roots(dim=0, fixed={1: 1.5, 2: 0.0})
        _check("roots accepts user-frame-valid fixed value",
               abs(float(roots[0]) - 0.4) < 1e-7,
               f"root={float(roots[0]):.4f}")
    except Exception as e:
        _check("roots accepts user-frame-valid fixed value", False,
               f"raised {type(e).__name__}: {e}")


def demo_integrate_error_user_frame() -> None:
    print("\n=== integrate error message uses user-frame dim (Item E / #20) ===")
    def f(x, _): return x[0] + x[1] + x[2]
    tt = ChebyshevTT(
        f, num_dimensions=3,
        domain=[(-1, 1), (-2, 2), (-3, 3)], n_nodes=[5, 5, 5],
    )
    tt.build(verbose=False)
    tt_p = tt.reorder([2, 0, 1])
    try:
        tt_p.integrate(dims=[1], bounds=[(5.0, 6.0)])
        _check("ValueError raised on out-of-domain bounds", False)
    except ValueError as e:
        msg = str(e)
        _check("error references user-frame dim 1",
               "dim 1" in msg, msg[:100])


def demo_eval_multi_no_mutation() -> None:
    print("\n=== eval_multi no longer mutates _dim_order (Item D / #19) ===")
    import inspect
    source = inspect.getsource(ChebyshevTT.eval_multi)
    _check("eval_multi source contains no 'self._dim_order = ' assignment",
           "self._dim_order = " not in source and "self._dim_order=" not in source)


def main() -> None:
    print("PyChebyshev v0.21.1 cluster fix demo")
    t0 = time.time()
    demo_inner_product_strict()
    demo_get_evaluation_points_round_trip()
    demo_roots_user_frame_validation()
    demo_integrate_error_user_frame()
    demo_eval_multi_no_mutation()
    print(f"\nTotal: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
