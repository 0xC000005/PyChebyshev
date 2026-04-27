"""Generate .pcb test fixtures used by Python, Rust, and Julia readers.

Run once; commit the resulting .pcb files. Re-run when the .pcb format
changes (which should never happen at the major-version level).
"""
from __future__ import annotations

import math
from pathlib import Path

from pychebyshev import ChebyshevApproximation, ChebyshevSpline


def f_2d(x, _):
    return x[0] * x[1]


def f_5d_bs(x, _):
    return math.sin(x[0]) + math.cos(x[1]) + x[2] ** 2 + x[3] * x[4]


def f_kink(x, _):
    return abs(x[0])


def main():
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    cheb = ChebyshevApproximation(f_2d, 2, [[-1, 1], [-1, 1]], [4, 4])
    cheb.build(verbose=False)
    cheb.save(str(fixtures_dir / "approx_2d_simple.pcb"), format="binary")

    cheb_5d = ChebyshevApproximation(f_5d_bs, 5, [[-1, 1]] * 5, [6] * 5)
    cheb_5d.build(verbose=False)
    cheb_5d.save(str(fixtures_dir / "approx_5d_bs.pcb"), format="binary")

    spl = ChebyshevSpline(f_kink, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[8])
    spl.build(verbose=False)
    spl.save(str(fixtures_dir / "spline_1d_kink.pcb"), format="binary")

    print("Generated fixtures:")
    for path in sorted(fixtures_dir.glob("*.pcb")):
        print(f"  {path.name} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
