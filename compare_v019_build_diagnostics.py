"""Demonstrate v0.19 parallel build + progress bars + viz helpers.

MoCaX has neither parallel build nor visualization helpers — this is a
beyond-MoCaX feature.
"""
import math
import time

from pychebyshev import ChebyshevApproximation


def slow_f(x, _):
    """Simulate a slow function (~1 ms per call)."""
    time.sleep(0.001)
    return math.sin(x[0]) * math.cos(x[1])


def main():
    domain = [[-1.0, 1.0], [-1.0, 1.0]]
    n = 12

    # Sequential build
    t0 = time.time()
    cheb_seq = ChebyshevApproximation(slow_f, 2, domain, [n, n])
    cheb_seq.build(verbose=False)
    t_seq = time.time() - t0

    # Parallel build (4 workers)
    t0 = time.time()
    cheb_par = ChebyshevApproximation(slow_f, 2, domain, [n, n], n_workers=4)
    cheb_par.build(verbose=False)
    t_par = time.time() - t0

    print(f"Sequential build:           {t_seq:.2f}s")
    print(f"Parallel build (4 workers): {t_par:.2f}s")
    if t_par > 0:
        print(f"Speedup:                    {t_seq / t_par:.2f}x")

    # Plot convergence
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = cheb_par.plot_convergence(target_error=1e-6, max_n=20)
        plt.savefig("convergence.png", dpi=80)
        plt.close()
        print("Convergence plot saved to convergence.png")
    except ImportError:
        print("(matplotlib not installed; skipping plot)")

    # Plot 2D surface
    try:
        import matplotlib.pyplot as plt
        ax = cheb_par.plot_2d_surface(n_points=30)
        plt.savefig("surface.png", dpi=80)
        plt.close()
        print("2D surface plot saved to surface.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
