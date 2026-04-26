"""Side-by-side comparison: PyChebyshev v0.16 polish surface vs MoCaX 4.3.1.

Most v0.16 surface methods have direct MoCaX equivalents. This script
demonstrates the calls side-by-side. Run from the repo root:

    uv run python compare_v016_polish.py

Requires mocax_lib/ at the repo root (gitignored).
"""

from __future__ import annotations

import math

from pychebyshev import (
    ChebyshevApproximation,
    Domain,
    Ns,
)


def f(x, _):
    return math.sin(x[0]) + math.cos(x[1])


def main():
    # PyChebyshev: typed-helper construction
    py_cheb = ChebyshevApproximation(
        f, 2, Domain([(-1.0, 1.0), (-1.0, 1.0)]), Ns([10, 10]),
    )
    py_cheb.build(verbose=False)

    print("PyChebyshev v0.16 polish surface:")
    print(f"  clone()                         → {type(py_cheb.clone()).__name__}")
    print(f"  get_max_derivative_order()      → {py_cheb.get_max_derivative_order()}")
    print(f"  get_error_threshold()           → {py_cheb.get_error_threshold()}")
    print(f"  get_special_points()            → {py_cheb.get_special_points()}")
    print(f"  get_num_evaluation_points()     → {py_cheb.get_num_evaluation_points()}")
    print(f"  get_evaluation_points().shape   → {py_cheb.get_evaluation_points().shape}")
    print(f"  is_dimensionality_allowed(2)    → {ChebyshevApproximation.is_dimensionality_allowed(2)}")

    try:
        from mocaxpy import Mocax, MocaxDomain, MocaxNs
    except ImportError:
        print("\n(mocaxpy not installed; skipping MoCaX side)")
        return

    mc = Mocax(
        f,
        2,
        MocaxDomain([(-1.0, 1.0), (-1.0, 1.0)]),
        MocaxNs([10, 10]),
        None,  # error_threshold
    )
    print("\nMoCaX 4.3.1 equivalent surface:")
    print(f"  clone()                         → {type(mc.clone()).__name__}")
    print(f"  get_max_derivative_order()      → {mc.get_max_derivative_order()}")
    print(f"  get_num_evaluation_points()     → {mc.get_num_evaluation_points()}")


if __name__ == "__main__":
    main()
