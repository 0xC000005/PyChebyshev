"""Tests for pre-computed values: nodes() and from_values() on both classes."""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from pychebyshev import ChebyshevApproximation, ChebyshevSpline


# ======================================================================
# Helper: build via both paths and compare
# ======================================================================

def _build_both_ways_approx(func, ndim, domain, n_nodes, **kw):
    """Build same function via build() and via nodes()+from_values(), return both."""
    # Path A: traditional
    cheb_a = ChebyshevApproximation(func, ndim, domain, n_nodes, **kw)
    cheb_a.build(verbose=False)

    # Path B: pre-computed
    info = ChebyshevApproximation.nodes(ndim, domain, n_nodes)
    values = np.array([
        func([info["full_grid"][i, d] for d in range(ndim)], None)
        for i in range(info["full_grid"].shape[0])
    ]).reshape(info["shape"])
    cheb_b = ChebyshevApproximation.from_values(values, ndim, domain, n_nodes, **kw)

    return cheb_a, cheb_b


def _build_both_ways_spline(func, ndim, domain, n_nodes, knots, **kw):
    """Build same function via build() and via nodes()+from_values(), return both."""
    # Path A: traditional
    sp_a = ChebyshevSpline(func, ndim, domain, n_nodes, knots, **kw)
    sp_a.build(verbose=False)

    # Path B: pre-computed
    info = ChebyshevSpline.nodes(ndim, domain, n_nodes, knots)
    piece_values = []
    for piece_info in info["pieces"]:
        grid = piece_info["full_grid"]
        vals = np.array([
            func([grid[i, d] for d in range(ndim)], None)
            for i in range(grid.shape[0])
        ]).reshape(piece_info["shape"])
        piece_values.append(vals)
    sp_b = ChebyshevSpline.from_values(piece_values, ndim, domain, n_nodes, knots, **kw)

    return sp_a, sp_b


# ======================================================================
# TestNodesApprox
# ======================================================================

class TestNodesApprox:
    """Tests for ChebyshevApproximation.nodes()."""

    def test_nodes_returns_correct_keys(self):
        info = ChebyshevApproximation.nodes(2, [[-1, 1], [0, 2]], [5, 7])
        assert set(info.keys()) == {"nodes_per_dim", "full_grid", "shape"}

    def test_nodes_shape_1d(self):
        info = ChebyshevApproximation.nodes(1, [[-1, 1]], [10])
        assert info["shape"] == (10,)
        assert len(info["nodes_per_dim"]) == 1
        assert len(info["nodes_per_dim"][0]) == 10

    def test_nodes_shape_2d(self):
        info = ChebyshevApproximation.nodes(2, [[-1, 1], [0, 3]], [8, 12])
        assert info["shape"] == (8, 12)
        assert info["full_grid"].shape == (96, 2)

    def test_nodes_shape_3d(self):
        info = ChebyshevApproximation.nodes(3, [[-1, 1], [0, 1], [2, 4]], [5, 6, 7])
        assert info["shape"] == (5, 6, 7)
        assert info["full_grid"].shape == (210, 3)

    def test_nodes_full_grid_shape(self):
        info = ChebyshevApproximation.nodes(2, [[-1, 1], [0, 1]], [4, 5])
        assert info["full_grid"].shape == (20, 2)
        # All x-coords should be in [-1, 1]
        assert info["full_grid"][:, 0].min() >= -1.0
        assert info["full_grid"][:, 0].max() <= 1.0
        # All y-coords should be in [0, 1]
        assert info["full_grid"][:, 1].min() >= 0.0
        assert info["full_grid"][:, 1].max() <= 1.0

    def test_nodes_match_build_nodes(self):
        """nodes() returns same nodes as a built interpolant."""
        def f(x, _): return math.sin(x[0]) + x[1]
        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [0, 2]], [8, 6])
        cheb.build(verbose=False)

        info = ChebyshevApproximation.nodes(2, [[-1, 1], [0, 2]], [8, 6])
        for d in range(2):
            np.testing.assert_allclose(info["nodes_per_dim"][d], cheb.nodes[d])

    def test_nodes_domain_mismatch_raises(self):
        with pytest.raises(ValueError, match="num_dimensions"):
            ChebyshevApproximation.nodes(2, [[-1, 1]], [5, 5])

    def test_nodes_ndim_mismatch_raises(self):
        with pytest.raises(ValueError, match="num_dimensions"):
            ChebyshevApproximation.nodes(2, [[-1, 1], [0, 1]], [5])


# ======================================================================
# TestFromValuesApprox
# ======================================================================

class TestFromValuesApprox:
    """Tests for ChebyshevApproximation.from_values()."""

    # --- Core equivalence: bit-identical to build() ---

    def test_eval_matches_build_1d(self):
        def f(x, _): return math.sin(x[0])
        a, b = _build_both_ways_approx(f, 1, [[0, math.pi]], [20])
        pts = [0.5, 1.0, 2.0, 3.0]
        for p in pts:
            va = a.eval([p], [0])
            vb = b.eval([p], [0])
            assert va == vb, f"Mismatch at {p}: {va} vs {vb}"

    def test_eval_matches_build_2d(self):
        def f(x, _): return math.sin(x[0]) * math.exp(-x[1])
        a, b = _build_both_ways_approx(f, 2, [[-1, 1], [0, 2]], [12, 10])
        va = a.vectorized_eval([0.3, 0.7], [0, 0])
        vb = b.vectorized_eval([0.3, 0.7], [0, 0])
        assert va == vb

    def test_eval_matches_build_3d(self):
        def f(x, _): return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])
        a, b = _build_both_ways_approx(f, 3, [[-1, 1], [-1, 1], [1, 3]], [8, 8, 6])
        va = a.vectorized_eval([0.5, -0.3, 2.0], [0, 0, 0])
        vb = b.vectorized_eval([0.5, -0.3, 2.0], [0, 0, 0])
        assert va == vb

    def test_derivatives_1st(self):
        def f(x, _): return math.sin(x[0]) * math.cos(x[1])
        a, b = _build_both_ways_approx(f, 2, [[-1, 1], [-1, 1]], [15, 15])
        for deriv in ([1, 0], [0, 1]):
            da = a.vectorized_eval([0.3, 0.5], deriv)
            db = b.vectorized_eval([0.3, 0.5], deriv)
            assert da == db, f"Mismatch for deriv {deriv}"

    def test_derivatives_2nd(self):
        def f(x, _): return math.sin(x[0]) * math.cos(x[1])
        a, b = _build_both_ways_approx(f, 2, [[-1, 1], [-1, 1]], [15, 15])
        for deriv in ([2, 0], [0, 2], [1, 1]):
            da = a.vectorized_eval([0.3, 0.5], deriv)
            db = b.vectorized_eval([0.3, 0.5], deriv)
            assert da == db, f"Mismatch for deriv {deriv}"

    def test_integrate_full(self):
        def f(x, _): return math.sin(x[0])
        a, b = _build_both_ways_approx(f, 1, [[0, math.pi]], [20])
        assert a.integrate() == b.integrate()

    def test_integrate_partial(self):
        def f(x, _): return math.sin(x[0]) * math.exp(-x[1])
        a, b = _build_both_ways_approx(f, 2, [[-1, 1], [0, 2]], [15, 12])
        ra = a.integrate(dims=[1])
        rb = b.integrate(dims=[1])
        # Partial integration returns a ChebyshevApproximation
        pt = [0.3]
        assert ra.vectorized_eval(pt, [0]) == rb.vectorized_eval(pt, [0])

    def test_integrate_sub_interval(self):
        def f(x, _): return math.sin(x[0])
        a, b = _build_both_ways_approx(f, 1, [[0, math.pi]], [20])
        assert a.integrate(bounds=[[0, 1]]) == b.integrate(bounds=[[0, 1]])

    def test_roots(self):
        def f(x, _): return math.sin(x[0])
        a, b = _build_both_ways_approx(f, 1, [[0.5, 5.0]], [25])
        np.testing.assert_allclose(a.roots(), b.roots())

    def test_minimize(self):
        def f(x, _): return (x[0] - 0.3) ** 2
        a, b = _build_both_ways_approx(f, 1, [[-1, 1]], [15])
        xa, va = a.minimize()
        xb, vb = b.minimize()
        assert xa == xb and va == vb

    def test_maximize(self):
        def f(x, _): return -(x[0] - 0.3) ** 2
        a, b = _build_both_ways_approx(f, 1, [[-1, 1]], [15])
        xa, va = a.maximize()
        xb, vb = b.maximize()
        assert xa == xb and va == vb

    def test_algebra_add(self):
        def f(x, _): return math.sin(x[0])
        def g(x, _): return math.cos(x[0])
        _, fb = _build_both_ways_approx(f, 1, [[-1, 1]], [15])
        _, gb = _build_both_ways_approx(g, 1, [[-1, 1]], [15])
        result = fb + gb
        expected = math.sin(0.5) + math.cos(0.5)
        assert abs(result.vectorized_eval([0.5], [0]) - expected) < 1e-10

    def test_algebra_mul(self):
        def f(x, _): return math.sin(x[0])
        _, fb = _build_both_ways_approx(f, 1, [[-1, 1]], [15])
        result = fb * 3.0
        expected = 3.0 * math.sin(0.5)
        assert abs(result.vectorized_eval([0.5], [0]) - expected) < 1e-10

    def test_extrude(self):
        def f(x, _): return math.sin(x[0])
        _, b = _build_both_ways_approx(f, 1, [[-1, 1]], [15])
        extruded = b.extrude([(1, [0, 2], 5)])
        val = extruded.vectorized_eval([0.5, 1.0], [0, 0])
        assert abs(val - math.sin(0.5)) < 1e-10

    def test_slice(self):
        def f(x, _): return math.sin(x[0]) * math.exp(-x[1])
        _, b = _build_both_ways_approx(f, 2, [[-1, 1], [0, 2]], [15, 12])
        sliced = b.slice([(1, 1.0)])
        expected = math.sin(0.5) * math.exp(-1.0)
        assert abs(sliced.vectorized_eval([0.5], [0]) - expected) < 1e-10

    def test_save_load(self):
        def f(x, _): return math.sin(x[0])
        _, b = _build_both_ways_approx(f, 1, [[-1, 1]], [15])
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            path = tmp.name
        try:
            b.save(path)
            loaded = ChebyshevApproximation.load(path)
            assert loaded.vectorized_eval([0.5], [0]) == b.vectorized_eval([0.5], [0])
        finally:
            os.unlink(path)

    def test_error_estimate(self):
        def f(x, _): return math.sin(x[0])
        a, b = _build_both_ways_approx(f, 1, [[-1, 1]], [15])
        assert a.error_estimate() == b.error_estimate()

    def test_batch_eval(self):
        def f(x, _): return math.sin(x[0])
        a, b = _build_both_ways_approx(f, 1, [[-1, 1]], [15])
        points = [[0.1], [0.5], [0.9]]
        ra = a.vectorized_eval_multi(points, [[0]])
        rb = b.vectorized_eval_multi(points, [[0]])
        np.testing.assert_array_equal(ra, rb)

    def test_multi_eval(self):
        def f(x, _): return math.sin(x[0])
        a, b = _build_both_ways_approx(f, 1, [[-1, 1]], [15])
        ra = a.vectorized_eval_multi([[0.5]], [[0], [1]])
        rb = b.vectorized_eval_multi([[0.5]], [[0], [1]])
        np.testing.assert_array_equal(ra, rb)

    def test_end_to_end_workflow(self):
        """Full workflow: nodes() → external eval → from_values() → evaluate."""
        info = ChebyshevApproximation.nodes(2, [[-1, 1], [0, 2]], [12, 10])
        # Simulate external evaluation
        values = np.sin(info["full_grid"][:, 0]) * np.exp(-info["full_grid"][:, 1])
        values = values.reshape(info["shape"])
        cheb = ChebyshevApproximation.from_values(values, 2, [[-1, 1], [0, 2]], [12, 10])
        expected = math.sin(0.5) * math.exp(-1.0)
        assert abs(cheb.vectorized_eval([0.5, 1.0], [0, 0]) - expected) < 1e-10

    # --- Edge cases ---

    def test_shape_mismatch_raises(self):
        values = np.zeros((5, 5))
        with pytest.raises(ValueError, match="shape"):
            ChebyshevApproximation.from_values(values, 2, [[-1, 1], [0, 1]], [5, 7])

    def test_nan_raises(self):
        values = np.ones((5,))
        values[2] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            ChebyshevApproximation.from_values(values, 1, [[-1, 1]], [5])

    def test_inf_raises(self):
        values = np.ones((5,))
        values[2] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            ChebyshevApproximation.from_values(values, 1, [[-1, 1]], [5])

    def test_1_node_dimension(self):
        """n_nodes=[1, 15]: constant in first dimension."""
        def f(x, _): return math.sin(x[1])
        info = ChebyshevApproximation.nodes(2, [[-1, 1], [0, math.pi]], [1, 15])
        values = np.array([
            f([info["full_grid"][i, 0], info["full_grid"][i, 1]], None)
            for i in range(info["full_grid"].shape[0])
        ]).reshape(info["shape"])
        cheb = ChebyshevApproximation.from_values(
            values, 2, [[-1, 1], [0, math.pi]], [1, 15]
        )
        # Should be able to evaluate (dim 0 is constant)
        val = cheb.vectorized_eval([0.0, 1.0], [0, 0])
        assert abs(val - math.sin(1.0)) < 1e-10

    def test_max_derivative_order(self):
        """Custom max_derivative_order=3, verify 3rd derivative works."""
        def f(x, _): return math.sin(x[0])
        info = ChebyshevApproximation.nodes(1, [[0, math.pi]], [25])
        values = np.sin(info["full_grid"][:, 0]).reshape(info["shape"])
        cheb = ChebyshevApproximation.from_values(
            values, 1, [[0, math.pi]], [25], max_derivative_order=3
        )
        # 3rd derivative of sin(x) = -cos(x)
        d3 = cheb.vectorized_eval([1.0], [3])
        expected = -math.cos(1.0)
        assert abs(d3 - expected) < 1e-4, f"3rd deriv: {d3} vs {expected}"

    def test_build_on_from_values_raises(self):
        """build() on a from_values object raises RuntimeError."""
        values = np.ones((5,))
        cheb = ChebyshevApproximation.from_values(values, 1, [[-1, 1]], [5])
        with pytest.raises(RuntimeError, match="no function assigned"):
            cheb.build()

    def test_cross_compat_algebra(self):
        """Algebra between from_values and build objects works."""
        def f(x, _): return math.sin(x[0])
        def g(x, _): return math.cos(x[0])

        # f via from_values
        info = ChebyshevApproximation.nodes(1, [[-1, 1]], [15])
        vals_f = np.sin(info["full_grid"][:, 0]).reshape(info["shape"])
        cheb_f = ChebyshevApproximation.from_values(vals_f, 1, [[-1, 1]], [15])

        # g via build
        cheb_g = ChebyshevApproximation(g, 1, [[-1, 1]], [15])
        cheb_g.build(verbose=False)

        result = cheb_f + cheb_g
        expected = math.sin(0.5) + math.cos(0.5)
        assert abs(result.vectorized_eval([0.5], [0]) - expected) < 1e-10

    def test_str_output(self):
        """__str__ shows 'built' with build_time=0.000 and 0 evaluations."""
        values = np.ones((5,))
        cheb = ChebyshevApproximation.from_values(values, 1, [[-1, 1]], [5])
        s = str(cheb)
        assert "built" in s
        assert "0.000s" in s
        assert "0 evaluations" in s

    def test_repr(self):
        """__repr__ works without error."""
        values = np.ones((5,))
        cheb = ChebyshevApproximation.from_values(values, 1, [[-1, 1]], [5])
        r = repr(cheb)
        assert "ChebyshevApproximation" in r
        assert "built=True" in r


# ======================================================================
# TestNodesSpline
# ======================================================================

class TestNodesSpline:
    """Tests for ChebyshevSpline.nodes()."""

    def test_returns_correct_keys(self):
        info = ChebyshevSpline.nodes(1, [[-1, 1]], [10], [[0.0]])
        assert set(info.keys()) == {"pieces", "num_pieces", "piece_shape"}
        assert set(info["pieces"][0].keys()) == {
            "piece_index", "sub_domain", "nodes_per_dim", "full_grid", "shape"
        }

    def test_piece_count(self):
        info = ChebyshevSpline.nodes(1, [[-1, 1]], [10], [[0.0]])
        assert info["num_pieces"] == 2
        assert info["piece_shape"] == (2,)

    def test_sub_domains(self):
        info = ChebyshevSpline.nodes(1, [[-1, 1]], [10], [[0.0]])
        assert info["pieces"][0]["sub_domain"] == [(-1, 0.0)]
        assert info["pieces"][1]["sub_domain"] == [(0.0, 1)]

    def test_no_knots(self):
        """Empty knots → single piece per dimension."""
        info = ChebyshevSpline.nodes(1, [[-1, 1]], [10], [[]])
        assert info["num_pieces"] == 1
        assert info["piece_shape"] == (1,)

    def test_piece_ordering_matches_ndindex(self):
        """2D with knots: pieces follow np.ndindex order."""
        info = ChebyshevSpline.nodes(
            2, [[-1, 1], [0, 2]], [5, 5], [[0.0], [1.0]]
        )
        assert info["num_pieces"] == 4
        assert info["piece_shape"] == (2, 2)
        # Check piece indices match np.ndindex
        expected_indices = list(np.ndindex(2, 2))
        for i, piece in enumerate(info["pieces"]):
            assert piece["piece_index"] == expected_indices[i]


# ======================================================================
# TestFromValuesSpline
# ======================================================================

class TestFromValuesSpline:
    """Tests for ChebyshevSpline.from_values()."""

    def test_eval_matches_build(self):
        def f(x, _): return abs(x[0]) - 0.3
        a, b = _build_both_ways_spline(f, 1, [[-1, 1]], [15], [[0.0]])
        pts = [-0.7, -0.1, 0.1, 0.7]
        for p in pts:
            va = a.eval([p], [0])
            vb = b.eval([p], [0])
            assert va == vb, f"Mismatch at {p}: {va} vs {vb}"

    def test_derivatives(self):
        def f(x, _): return abs(x[0]) - 0.3
        a, b = _build_both_ways_spline(f, 1, [[-1, 1]], [15], [[0.0]])
        # Derivative away from knot
        da = a.eval([0.5], [1])
        db = b.eval([0.5], [1])
        assert da == db

    def test_integrate(self):
        def f(x, _): return abs(x[0]) - 0.3
        a, b = _build_both_ways_spline(f, 1, [[-1, 1]], [15], [[0.0]])
        assert a.integrate() == b.integrate()

    def test_roots(self):
        def f(x, _): return abs(x[0]) - 0.3
        a, b = _build_both_ways_spline(f, 1, [[-1, 1]], [15], [[0.0]])
        np.testing.assert_allclose(a.roots(), b.roots())

    def test_save_load(self):
        def f(x, _): return abs(x[0]) - 0.3
        _, b = _build_both_ways_spline(f, 1, [[-1, 1]], [15], [[0.0]])
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            path = tmp.name
        try:
            b.save(path)
            loaded = ChebyshevSpline.load(path)
            assert loaded.eval([0.5], [0]) == b.eval([0.5], [0])
        finally:
            os.unlink(path)

    def test_no_knots_matches_approx(self):
        """Spline with no knots ≈ ChebyshevApproximation."""
        def f(x, _): return math.sin(x[0])
        info_sp = ChebyshevSpline.nodes(1, [[-1, 1]], [15], [[]])
        info_ap = ChebyshevApproximation.nodes(1, [[-1, 1]], [15])

        vals = np.sin(info_ap["full_grid"][:, 0]).reshape(info_ap["shape"])
        cheb = ChebyshevApproximation.from_values(vals, 1, [[-1, 1]], [15])
        sp = ChebyshevSpline.from_values([vals], 1, [[-1, 1]], [15], [[]])

        pt = [0.5]
        assert abs(cheb.vectorized_eval(pt, [0]) - sp.eval(pt, [0])) < 1e-14

    def test_build_on_from_values_raises(self):
        """build() on a from_values spline raises RuntimeError."""
        info = ChebyshevSpline.nodes(1, [[-1, 1]], [10], [[0.0]])
        piece_values = [np.ones((10,)), np.ones((10,))]
        sp = ChebyshevSpline.from_values(piece_values, 1, [[-1, 1]], [10], [[0.0]])
        with pytest.raises(RuntimeError, match="no function assigned"):
            sp.build()

    def test_str_output(self):
        """__str__ shows 'built' for from_values spline."""
        info = ChebyshevSpline.nodes(1, [[-1, 1]], [10], [[0.0]])
        piece_values = [np.ones((10,)), np.ones((10,))]
        sp = ChebyshevSpline.from_values(piece_values, 1, [[-1, 1]], [10], [[0.0]])
        s = str(sp)
        assert "built" in s
        assert "0.000s" in s

    def test_2d_eval_matches_build(self):
        """2D spline with knots in both dims: from_values matches build."""
        def f(x, _):
            return abs(x[0]) + abs(x[1] - 1.0)

        a, b = _build_both_ways_spline(
            f, 2, [[-1, 1], [0, 2]], [10, 10], [[0.0], [1.0]]
        )
        pts = [(-0.5, 0.5), (0.3, 1.7), (-0.8, 0.2), (0.9, 1.9)]
        for p in pts:
            va = a.eval(list(p), [0, 0])
            vb = b.eval(list(p), [0, 0])
            assert va == vb, f"Mismatch at {p}: {va} vs {vb}"

    def test_multi_knot_1d(self):
        """1D spline with multiple knots: from_values matches build."""
        def f(x, _):
            v = x[0]
            return abs(v + 0.5) + abs(v) + abs(v - 0.5)

        a, b = _build_both_ways_spline(
            f, 1, [[-1, 1]], [12], [[-0.5, 0.0, 0.5]]
        )
        pts = [-0.8, -0.2, 0.3, 0.7]
        for p in pts:
            va = a.eval([p], [0])
            vb = b.eval([p], [0])
            assert va == vb, f"Mismatch at {p}: {va} vs {vb}"

    def test_spline_batch_eval(self):
        """Batch eval on from_values spline matches build."""
        def f(x, _): return abs(x[0]) - 0.3
        a, b = _build_both_ways_spline(f, 1, [[-1, 1]], [15], [[0.0]])
        pts = [[-0.7], [-0.1], [0.1], [0.7]]
        for pt in pts:
            va = a.eval(pt, [0])
            vb = b.eval(pt, [0])
            assert va == vb

    def test_spline_multi_eval(self):
        """Multi eval (value + derivative) on from_values spline matches build."""
        def f(x, _): return abs(x[0]) - 0.3
        a, b = _build_both_ways_spline(f, 1, [[-1, 1]], [15], [[0.0]])
        va = a.eval([0.5], [0])
        da = a.eval([0.5], [1])
        vb = b.eval([0.5], [0])
        db = b.eval([0.5], [1])
        assert va == vb
        assert da == db


# ======================================================================
# TestEdgeCases — Additional edge case tests from review
# ======================================================================

class TestEdgeCases:
    """Edge cases identified during code review."""

    def test_negative_domain(self):
        """Domain with negative bounds works correctly."""
        def f(x, _): return math.sin(x[0])
        info = ChebyshevApproximation.nodes(1, [[-10, -5]], [15])
        values = np.sin(info["full_grid"][:, 0]).reshape(info["shape"])
        cheb = ChebyshevApproximation.from_values(values, 1, [[-10, -5]], [15])
        expected = math.sin(-7.5)
        assert abs(cheb.vectorized_eval([-7.5], [0]) - expected) < 1e-10

    def test_wide_domain(self):
        """Very wide domain still works."""
        def f(x, _): return math.sin(x[0] / 100)
        info = ChebyshevApproximation.nodes(1, [[-1000, 1000]], [25])
        values = np.sin(info["full_grid"][:, 0] / 100).reshape(info["shape"])
        cheb = ChebyshevApproximation.from_values(values, 1, [[-1000, 1000]], [25])
        expected = math.sin(500 / 100)
        assert abs(cheb.vectorized_eval([500.0], [0]) - expected) < 1e-6

    def test_tight_domain(self):
        """Very narrow domain works."""
        def f(x, _): return x[0] ** 2
        info = ChebyshevApproximation.nodes(1, [[0.999, 1.001]], [10])
        values = (info["full_grid"][:, 0] ** 2).reshape(info["shape"])
        cheb = ChebyshevApproximation.from_values(values, 1, [[0.999, 1.001]], [10])
        expected = 1.0 ** 2
        assert abs(cheb.vectorized_eval([1.0], [0]) - expected) < 1e-12

    def test_boundary_evaluation(self):
        """Evaluation at exact domain boundaries."""
        def f(x, _): return math.sin(x[0])
        info = ChebyshevApproximation.nodes(1, [[0, math.pi]], [20])
        values = np.sin(info["full_grid"][:, 0]).reshape(info["shape"])
        cheb = ChebyshevApproximation.from_values(values, 1, [[0, math.pi]], [20])
        # Left boundary: sin(0) = 0
        assert abs(cheb.vectorized_eval([0.0], [0])) < 1e-10
        # Right boundary: sin(pi) ~ 0
        assert abs(cheb.vectorized_eval([math.pi], [0])) < 1e-10

    def test_algebra_chain(self):
        """Chained algebra operations on from_values objects."""
        info = ChebyshevApproximation.nodes(1, [[-1, 1]], [15])
        vals_f = np.sin(info["full_grid"][:, 0]).reshape(info["shape"])
        vals_g = np.cos(info["full_grid"][:, 0]).reshape(info["shape"])
        cheb_f = ChebyshevApproximation.from_values(vals_f, 1, [[-1, 1]], [15])
        cheb_g = ChebyshevApproximation.from_values(vals_g, 1, [[-1, 1]], [15])

        # (f + g) * 2 - g
        result = (cheb_f + cheb_g) * 2.0 - cheb_g
        # = 2*sin + 2*cos - cos = 2*sin + cos
        expected = 2.0 * math.sin(0.5) + math.cos(0.5)
        assert abs(result.vectorized_eval([0.5], [0]) - expected) < 1e-10

    def test_domain_lo_ge_hi_raises_approx(self):
        """from_values rejects domain with lo >= hi."""
        values = np.ones((5,))
        with pytest.raises(ValueError, match="lo=.*must be strictly less than hi"):
            ChebyshevApproximation.from_values(values, 1, [[1, 1]], [5])
        with pytest.raises(ValueError, match="lo=.*must be strictly less than hi"):
            ChebyshevApproximation.from_values(values, 1, [[2, 1]], [5])

    def test_domain_lo_ge_hi_raises_spline_nodes(self):
        """Spline nodes() rejects domain with lo >= hi."""
        with pytest.raises(ValueError, match="lo=.*must be strictly less than hi"):
            ChebyshevSpline.nodes(1, [[1, 1]], [10], [[]])

    def test_duplicate_knots_raises(self):
        """Spline rejects duplicate knots."""
        with pytest.raises(ValueError, match="duplicates"):
            ChebyshevSpline.nodes(1, [[-1, 1]], [10], [[0.0, 0.0]])
        with pytest.raises(ValueError, match="duplicates"):
            ChebyshevSpline.from_values(
                [np.ones((10,))], 1, [[-1, 1]], [10], [[0.0, 0.0]]
            )

    def test_4d_from_values(self):
        """4D from_values builds and evaluates correctly."""
        def f(x, _): return sum(xi ** 2 for xi in x)
        ndim = 4
        domain = [[-1, 1]] * ndim
        n_nodes = [5] * ndim
        info = ChebyshevApproximation.nodes(ndim, domain, n_nodes)
        values = np.array([
            f([info["full_grid"][i, d] for d in range(ndim)], None)
            for i in range(info["full_grid"].shape[0])
        ]).reshape(info["shape"])
        cheb = ChebyshevApproximation.from_values(values, ndim, domain, n_nodes)
        # f(0.5, 0.5, 0.5, 0.5) = 4 * 0.25 = 1.0
        expected = 1.0
        assert abs(cheb.vectorized_eval([0.5] * ndim, [0] * ndim) - expected) < 1e-10

    def test_spline_piece_count_mismatch_raises(self):
        """from_values rejects wrong number of piece values."""
        with pytest.raises(ValueError, match="Expected 2"):
            ChebyshevSpline.from_values(
                [np.ones((10,))],  # only 1, need 2
                1, [[-1, 1]], [10], [[0.0]],
            )

    def test_spline_piece_shape_mismatch_raises(self):
        """from_values rejects piece with wrong shape, includes piece index."""
        with pytest.raises(ValueError, match=r"piece_values\[1\]"):
            ChebyshevSpline.from_values(
                [np.ones((10,)), np.ones((8,))],  # second piece wrong shape
                1, [[-1, 1]], [10], [[0.0]],
            )
