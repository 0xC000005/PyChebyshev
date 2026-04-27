"""Demonstrate v0.20.1 — full TT surface under non-identity ``_dim_order``.

PyChebyshev only. No MoCaX equivalent (MoCaX TT does not expose dim
reordering or the same surface methods).

Run: ``uv run python compare_v0201_dim_threading.py``
"""

from __future__ import annotations

import numpy as np

from pychebyshev import ChebyshevTT


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    rng = np.random.default_rng(0)

    # 5D rank-sensitive function: dim 4 entangles strongly with dim 0.
    def f(p, _):
        return (
            np.exp(-0.5 * p[0] * p[4])
            + np.sin(p[1])
            + p[2] ** 2
            + 0.3 * np.cos(p[3])
        )

    domain = [[-1.0, 1.0]] * 5
    n_nodes = [11] * 5

    banner("Build canonical TT vs auto-ordered TT")
    canonical = ChebyshevTT(f, 5, domain, n_nodes, tolerance=1e-8, max_rank=12)
    canonical.build()
    auto = ChebyshevTT.with_auto_order(
        f, 5, domain, n_nodes, tolerance=1e-8, max_rank=12,
        method="greedy_swap", n_trials=5,
    )
    print(f"canonical.dim_order  = {canonical.dim_order}")
    print(f"canonical.tt_ranks   = {canonical.tt_ranks}")
    print(f"auto.dim_order       = {auto.dim_order}")
    print(f"auto.tt_ranks        = {auto.tt_ranks}")
    print(
        f"rank sum: auto={sum(auto.tt_ranks)} vs canonical="
        f"{sum(canonical.tt_ranks)}"
    )

    banner("Eval agreement: canonical and auto-ordered TT")
    for _ in range(3):
        pt = rng.uniform(-0.9, 0.9, size=5).tolist()
        a = canonical.eval(pt)
        b = auto.eval(pt)
        coords = [f"{x:+.3f}" for x in pt]
        print(
            f"pt={coords}  canonical={a:+.6e}  auto={b:+.6e}  "
            f"diff={abs(a - b):.2e}"
        )

    banner("Full surface working on auto-ordered TT")
    pt5 = rng.uniform(-0.9, 0.9, size=5).tolist()
    print(f"eval(pt)            = {auto.eval(pt5):+.6e}")
    print(
        f"eval_multi(pt, [[0,0,0,0,0],[1,0,0,0,0]]) = "
        f"{auto.eval_multi(pt5, [[0,0,0,0,0],[1,0,0,0,0]])}"
    )
    sliced = auto.slice((2, 0.0))
    print(
        f"slice(orig dim 2)   -> num_dim={sliced.num_dimensions}, "
        f"dim_order={sliced.dim_order}"
    )
    extruded = auto.extrude((1, (-2, 2), 5))
    print(
        f"extrude pos=1       -> num_dim={extruded.num_dimensions}, "
        f"dim_order={extruded.dim_order}"
    )
    print(f"integrate(full)     = {auto.integrate():+.6e}")
    partial = auto.integrate(dims=[3])
    print(
        f"integrate(dims=[3]) -> num_dim={partial.num_dimensions}, "
        f"dim_order={partial.dim_order}"
    )
    print(f"to_dense().shape    = {auto.to_dense().shape}")

    banner("Unary algebra preserves dim_order")
    neg = -auto
    scaled = 2.5 * auto
    print(f"(-auto).dim_order        = {neg.dim_order}")
    print(f"(2.5 * auto).dim_order   = {scaled.dim_order}")
    print(
        f"verify (-auto).eval(pt) = {neg.eval(pt5):+.6e}  "
        f"≈ -auto.eval(pt) = {-auto.eval(pt5):+.6e}"
    )

    banner("Binary algebra: reorder() as the alignment escape hatch")
    other = ChebyshevTT.with_auto_order(
        f, 5, domain, n_nodes, tolerance=1e-8, max_rank=12,
        method="random", n_trials=5,
    )
    print(f"auto.dim_order   = {auto.dim_order}")
    print(f"other.dim_order  = {other.dim_order}")
    if auto.dim_order == other.dim_order:
        s = auto + other
        print("(orders matched on this run; direct add succeeded)")
        print(f"  s.dim_order      = {s.dim_order}")
    else:
        try:
            _ = auto + other
            print("(unexpected: direct add succeeded despite mismatch?)")
        except ValueError as e:
            err = str(e)
            print(f"direct add raises ValueError: {err[:120]}...")
            other_aligned = other.reorder(auto.dim_order)
            s = auto + other_aligned
            print(f"after reorder():  s.dim_order = {s.dim_order}")
            print(
                f"  s.eval(pt) = {s.eval(pt5):+.6e}  "
                f"(≈ 2*auto.eval = {2 * auto.eval(pt5):+.6e})"
            )

    banner("Done")


if __name__ == "__main__":
    main()
