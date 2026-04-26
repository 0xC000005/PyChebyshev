"""Shared visualization helpers — implementations parameterized by interpolant.

These functions are private; they're called by instance methods on each
public class. Methods raise ImportError if matplotlib isn't installed.
"""
from __future__ import annotations

import numpy as np


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "this method requires matplotlib; install with "
            "`pip install pychebyshev[viz]`"
        )


def _eval_one(obj, point):
    """Evaluate obj at a single point, abstracting over class differences.

    ChebyshevTT uses eval(point); the others use eval(point, derivative_order).
    """
    from pychebyshev import ChebyshevTT
    if isinstance(obj, ChebyshevTT):
        return obj.eval(point)
    return obj.eval(point, [0] * obj.num_dimensions)


def _resolve_free_dims(obj, fixed, expected_free):
    """Validate fixed and return (free_dims_list, fixed_dict)."""
    fixed = dict(fixed) if fixed else {}
    for d in fixed:
        if not (0 <= d < obj.num_dimensions):
            raise ValueError(
                f"fixed dim {d} out of range [0, {obj.num_dimensions - 1}]"
            )
    free_dims = [d for d in range(obj.num_dimensions) if d not in fixed]
    if len(free_dims) != expected_free:
        raise ValueError(
            f"plot requires exactly {expected_free} free dim(s); got {len(free_dims)} "
            f"(use fixed= to constrain other dims)"
        )
    return free_dims, fixed


def _plot_1d_impl(obj, ax=None, n_points=200, fixed=None):
    plt = _import_matplotlib()
    free_dims, fixed = _resolve_free_dims(obj, fixed, expected_free=1)
    free = free_dims[0]
    lo, hi = obj.domain[free]
    xs = np.linspace(lo, hi, n_points)
    ys = []
    for x in xs:
        point = [fixed.get(d, 0.0) for d in range(obj.num_dimensions)]
        point[free] = float(x)
        ys.append(_eval_one(obj, point))
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_xlabel(f"x_{free}")
    ax.set_ylabel("f")
    ax.set_title(f"{type(obj).__name__} 1-D plot")
    return ax


def _plot_2d_surface_impl(obj, ax=None, n_points=50, fixed=None):
    plt = _import_matplotlib()
    free_dims, fixed = _resolve_free_dims(obj, fixed, expected_free=2)
    free_a, free_b = free_dims
    lo_a, hi_a = obj.domain[free_a]
    lo_b, hi_b = obj.domain[free_b]
    xs = np.linspace(lo_a, hi_a, n_points)
    ys = np.linspace(lo_b, hi_b, n_points)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            point = [fixed.get(d, 0.0) for d in range(obj.num_dimensions)]
            point[free_a] = float(X[i, j])
            point[free_b] = float(Y[i, j])
            Z[i, j] = _eval_one(obj, point)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel(f"x_{free_a}")
    ax.set_ylabel(f"x_{free_b}")
    try:
        ax.set_zlabel("f")
    except AttributeError:
        pass  # axes wasn't 3D
    ax.set_title(f"{type(obj).__name__} 2-D surface")
    return ax


def _plot_2d_contour_impl(obj, ax=None, n_points=50, n_levels=20, fixed=None):
    plt = _import_matplotlib()
    free_dims, fixed = _resolve_free_dims(obj, fixed, expected_free=2)
    free_a, free_b = free_dims
    lo_a, hi_a = obj.domain[free_a]
    lo_b, hi_b = obj.domain[free_b]
    xs = np.linspace(lo_a, hi_a, n_points)
    ys = np.linspace(lo_b, hi_b, n_points)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            point = [fixed.get(d, 0.0) for d in range(obj.num_dimensions)]
            point[free_a] = float(X[i, j])
            point[free_b] = float(Y[i, j])
            Z[i, j] = _eval_one(obj, point)
    if ax is None:
        _, ax = plt.subplots()
    cs = ax.contourf(X, Y, Z, levels=n_levels, cmap="viridis")
    plt.colorbar(cs, ax=ax)
    ax.set_xlabel(f"x_{free_a}")
    ax.set_ylabel(f"x_{free_b}")
    ax.set_title(f"{type(obj).__name__} 2-D contour")
    return ax
