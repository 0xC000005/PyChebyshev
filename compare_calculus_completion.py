"""Side-by-side comparison: PyChebyshev v0.17 integrate() on Slider/TT vs MoCaX 4.3.1.

MoCaX has no integrate() API on any class. This script demonstrates
PyChebyshev's spectral integration as a beyond-MoCaX feature.

    uv run python compare_calculus_completion.py
"""

from __future__ import annotations

import math

from pychebyshev import ChebyshevSlider, ChebyshevTT


def f(x, _):
    return math.sin(x[0]) + math.cos(x[1])


def main():
    domain = [[-1.0, 1.0], [-1.0, 1.0]]

    # PyChebyshev Slider integrate
    slider = ChebyshevSlider(
        f, 2, domain, [10, 10],
        partition=[[0], [1]], pivot_point=[0.0, 0.0],
    )
    slider.build(verbose=False)
    print("PyChebyshev v0.17 — integrate() on all four classes:")
    print(f"  Slider full integrate           → {slider.integrate():.6f}")
    print(f"  Slider partial (dims=[0])       → {type(slider.integrate(dims=[0])).__name__}")

    # PyChebyshev TT integrate
    tt = ChebyshevTT(f, 2, domain, [10, 10])
    tt.build(verbose=False)
    print(f"  TT full integrate               → {tt.integrate():.6f}")
    print(f"  TT partial (dims=[0])           → {type(tt.integrate(dims=[0])).__name__}")

    # Analytical answer for sin(x) + cos(y) over [-1,1]^2 = 4*sin(1)
    print(f"\n  Analytical: 4 * sin(1)         → {4.0 * math.sin(1.0):.6f}")

    # MoCaX equivalent: none. Document this clearly.
    print("\nMoCaX 4.3.1: no integrate() API on any class — PyChebyshev is unique here.")


if __name__ == "__main__":
    main()
