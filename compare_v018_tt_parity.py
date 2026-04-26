"""Side-by-side: PyChebyshev v0.18 TT surface vs MoCaX 4.3.1.

MoCaX has TT add via MocaxExtend but lacks scalar mul, extrude, slice,
or to_dense on TT objects.

Run: uv run python compare_v018_tt_parity.py
"""
from __future__ import annotations

import math

import numpy as np

from pychebyshev import ChebyshevTT


def f(x, _):
    return math.sin(x[0]) + math.cos(x[1])


def main():
    domain = [[-1.0, 1.0], [-1.0, 1.0]]
    tt = ChebyshevTT(f, 2, domain, [10, 10])
    tt.build(verbose=False)

    print("PyChebyshev v0.18 TT surface:")
    print(
        f"  nodes() static                  → dict with "
        f"{len(ChebyshevTT.nodes(2, domain, [10, 10])['nodes_per_dim'])} per-dim arrays"
    )
    print(
        f"  from_values() classmethod       → "
        f"{type(ChebyshevTT.from_values(tt.to_dense(), 2, domain, [10, 10])).__name__}"
    )
    print(f"  extrude() (add dim)             → {type(tt.extrude((2, (0, 1), 4))).__name__}")
    print(f"  slice() (fix dim)               → {type(tt.slice((0, 0.5))).__name__}")
    print(f"  algebra (tt + tt)               → {type(tt + tt).__name__}")
    print(f"  scalar mul (tt * 2.5)           → {type(tt * 2.5).__name__}")
    print(f"  to_dense()                      → np.ndarray shape {tt.to_dense().shape}")

    try:
        from mocaxextendpy import MocaxTT  # noqa: F401
        print(
            "\nMoCaX 4.3.1: has TT add via MocaxExtend; lacks scalar mul, "
            "extrude, slice, to_dense on TT."
        )
    except ImportError:
        print("\n(mocaxextendpy not installed; MoCaX side skipped)")


if __name__ == "__main__":
    main()
