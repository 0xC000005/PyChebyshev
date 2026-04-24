"""Compare PyChebyshev v0.13 TT ALS + algebra vs MoCaX 4.3.1.

Local-only; requires mocaxextend_lib/ which is gitignored.

Runs:
1. ALS build value match vs MocaxTT (method = MOCAX_CONSTRUCTOR_TT_ALS)
2. run_completion vs MocaxTT.runCompletion
3. inner_product vs MocaxTT.innerProduct
4. orth_left/orth_right: verify eval invariance (hard to cross-check MoCaX's
   exact core state, so we verify PyChebyshev's eval is unchanged pre/post)
"""
from __future__ import annotations
import ctypes
import os
import sys

import numpy as np

# --- MoCaX bootstrap (same pattern as compare_tensor_train.py) ---
HERE = os.path.dirname(os.path.abspath(__file__))
MOCAX_LIB_DIR = os.path.join(HERE, "mocaxextend_lib", "shared_libs")
if not os.path.isdir(MOCAX_LIB_DIR):
    print(f"MoCaX libs not found at {MOCAX_LIB_DIR} - skipping.")
    sys.exit(0)

# Force-load the hommat shim (same as compare_tensor_train.py)
try:
    _hommat = ctypes.CDLL(os.path.join(MOCAX_LIB_DIR, "libhommat.so"))
    _tensorvals = ctypes.CDLL(os.path.join(MOCAX_LIB_DIR, "libtensorvals.so"))
except OSError as e:
    print(f"MoCaX shared libs not loadable: {e} - skipping.")
    sys.exit(0)

sys.path.insert(0, os.path.join(HERE, "mocaxextend_lib"))

try:
    import mocaxextend as mx
except ImportError as e:
    print(f"mocaxextend not importable: {e}")
    sys.exit(0)

# Introspect the module API so we can tell, in the output, which of the
# plan's assumed symbols (MocaxTT, MOCAX_CONSTRUCTOR_TT_ALS, runCompletion,
# innerProduct, orthLeft, orthRight) are actually exposed. If any are
# missing, the individual compare sections below skip cleanly.
print("-" * 60)
print("mocaxextend API probe:")
print(f"  top-level symbols: {[s for s in dir(mx) if not s.startswith('_')]}")
if hasattr(mx, "MocaxTT"):
    print(
        f"  MocaxTT methods: "
        f"{[s for s in dir(mx.MocaxTT) if not s.startswith('_')]}"
    )
else:
    print("  MocaxTT: NOT exposed")
print("-" * 60)

from pychebyshev.tensor_train import ChebyshevTT


def black_scholes_5d(x, data=None):
    S, K, r, sigma, T = x
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def compare_als_build():
    print("=" * 60)
    print("1. ALS build value match")
    print("=" * 60)
    domain = [(80.0, 120.0), (90.0, 110.0), (0.02, 0.06), (0.15, 0.35), (0.5, 2.0)]
    n_nodes = [8, 8, 6, 6, 6]

    tt_py = ChebyshevTT(black_scholes_5d, 5, domain, n_nodes,
                        tolerance=1e-4, max_rank=8)
    tt_py.build(verbose=False, method="als", seed=42)

    if not hasattr(mx, "MocaxTT") or not hasattr(mx, "MOCAX_CONSTRUCTOR_TT_ALS"):
        print("  MoCaX MocaxTT / MOCAX_CONSTRUCTOR_TT_ALS not exposed - skipping.")
        return

    try:
        tt_mc = mx.MocaxTT(black_scholes_5d, 5, domain, n_nodes,
                           method=mx.MOCAX_CONSTRUCTOR_TT_ALS,
                           tolerance=1e-4, maxRank=8)
    except Exception as e:
        print(f"  MoCaX MocaxTT(method=ALS) construction failed: {e} - skipping.")
        return

    pts = [
        [100.0, 100.0, 0.04, 0.25, 1.0],
        [95.0, 105.0, 0.03, 0.2, 0.75],
        [110.0, 95.0, 0.05, 0.3, 1.5],
    ]
    max_diff = 0.0
    for p in pts:
        v_py = tt_py.eval(p)
        try:
            v_mc = tt_mc.eval(p)
        except Exception as e:
            print(f"  MocaxTT.eval failed: {e} - skipping.")
            return
        max_diff = max(max_diff, abs(v_py - v_mc))
        print(f"  {p}: py={v_py:.8f} mc={v_mc:.8f} diff={abs(v_py - v_mc):.2e}")
    print(f"\n  Max |py - mc| = {max_diff:.2e}")


def compare_completion():
    print("=" * 60)
    print("2. run_completion vs MocaxTT.runCompletion")
    print("=" * 60)
    domain = [(-1.0, 1.0)] * 3
    n_nodes = [10, 10, 10]

    def f(x, data=None):
        return np.exp(-0.5 * (x[0] ** 2 + x[1] ** 2)) * np.cos(x[2])

    tt_py = ChebyshevTT(f, 3, domain, n_nodes,
                        tolerance=1e-3, max_rank=5)
    tt_py.build(verbose=False, method="cross", seed=0)
    tt_py.run_completion(tolerance=1e-10, max_iter=20)

    if not hasattr(mx, "MocaxTT") or not hasattr(mx, "MOCAX_CONSTRUCTOR_TT_CROSS"):
        print("  MoCaX MocaxTT / MOCAX_CONSTRUCTOR_TT_CROSS not exposed - skipping.")
        return

    try:
        tt_mc = mx.MocaxTT(f, 3, domain, n_nodes,
                           method=mx.MOCAX_CONSTRUCTOR_TT_CROSS,
                           tolerance=1e-3, maxRank=5)
    except Exception as e:
        print(f"  MocaxTT construction failed: {e} - skipping.")
        return

    if not hasattr(tt_mc, "runCompletion"):
        print("  MocaxTT.runCompletion not exposed - skipping.")
        return

    try:
        tt_mc.runCompletion(tolerance=1e-10, maxIter=20)
    except Exception as e:
        print(f"  MocaxTT.runCompletion failed: {e} - skipping.")
        return

    max_diff = 0.0
    for p in [[0.1, -0.2, 0.3], [0.5, 0.5, -0.5]]:
        v_py = tt_py.eval(p)
        try:
            v_mc = tt_mc.eval(p)
        except Exception as e:
            print(f"  MocaxTT.eval failed: {e} - skipping.")
            return
        max_diff = max(max_diff, abs(v_py - v_mc))
    print(f"  Max diff after completion: {max_diff:.2e}")


def compare_inner_product():
    print("=" * 60)
    print("3. inner_product vs MocaxTT.innerProduct")
    print("=" * 60)
    domain = [(-1.0, 1.0)] * 2
    n_nodes = [10, 10]

    def f(x, data=None):
        return np.sin(x[0]) + 0.5 * x[1]

    def g(x, data=None):
        return np.cos(x[0]) * x[1]

    tt_a_py = ChebyshevTT(f, 2, domain, n_nodes,
                          tolerance=1e-8, max_rank=8)
    tt_a_py.build(verbose=False, method="cross")
    tt_b_py = ChebyshevTT(g, 2, domain, n_nodes,
                          tolerance=1e-8, max_rank=8)
    tt_b_py.build(verbose=False, method="cross")
    ip_py = tt_a_py.inner_product(tt_b_py)

    if not hasattr(mx, "MocaxTT"):
        print("  MoCaX MocaxTT not exposed - skipping.")
        print(f"  py inner_product: {ip_py:.12f}")
        return

    try:
        tt_a_mc = mx.MocaxTT(f, 2, domain, n_nodes, tolerance=1e-8, maxRank=8)
        tt_b_mc = mx.MocaxTT(g, 2, domain, n_nodes, tolerance=1e-8, maxRank=8)
    except Exception as e:
        print(f"  MocaxTT construction failed: {e} - skipping.")
        print(f"  py inner_product: {ip_py:.12f}")
        return

    if not hasattr(tt_a_mc, "innerProduct"):
        print("  MocaxTT.innerProduct not exposed - skipping.")
        print(f"  py inner_product: {ip_py:.12f}")
        return

    try:
        ip_mc = tt_a_mc.innerProduct(tt_b_mc)
    except Exception as e:
        print(f"  MocaxTT.innerProduct failed: {e} - skipping.")
        print(f"  py inner_product: {ip_py:.12f}")
        return

    print(f"  py: {ip_py:.12f}")
    print(f"  mc: {ip_mc:.12f}")
    print(f"  diff: {abs(ip_py - ip_mc):.2e}")


def compare_orth():
    print("=" * 60)
    print("4. orth_left/orth_right: eval invariance")
    print("=" * 60)
    domain = [(-1.0, 1.0)] * 3
    n_nodes = [8, 8, 8]

    def f(x, data=None):
        return np.cos(x[0]) * x[1] + x[2]

    tt_py = ChebyshevTT(f, 3, domain, n_nodes,
                        tolerance=1e-6, max_rank=5)
    tt_py.build(verbose=False, method="cross")
    before = tt_py.eval([0.2, -0.1, 0.3])
    tt_py.orth_left(position=2)
    after_left = tt_py.eval([0.2, -0.1, 0.3])
    tt_py.orth_right(position=0)
    after_right = tt_py.eval([0.2, -0.1, 0.3])
    print(f"  before:      {before:.12f}")
    print(f"  after L orth:{after_left:.12f}  diff={abs(before - after_left):.2e}")
    print(f"  after R orth:{after_right:.12f}  diff={abs(before - after_right):.2e}")


if __name__ == "__main__":
    compare_als_build()
    print()
    compare_completion()
    print()
    compare_inner_product()
    print()
    compare_orth()
