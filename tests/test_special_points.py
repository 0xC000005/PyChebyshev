"""Tests for special_points in the core ChebyshevApproximation API (v0.12)."""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from pychebyshev import ChebyshevApproximation, ChebyshevSpline


def _abs1d(x, _):
    return abs(x[0])


def _abs_sum_2d(x, _):
    return abs(x[0]) + abs(x[1])


class TestDispatch:
    """ChebyshevApproximation.__new__ routes to ChebyshevSpline when
    special_points has any non-empty dim."""

    def test_special_points_none_returns_approximation(self):
        obj = ChebyshevApproximation(
            lambda x, _: x[0] ** 2, 1, [[-1, 1]], [11]
        )
        assert type(obj) is ChebyshevApproximation

    def test_all_empty_special_points_returns_approximation(self):
        obj = ChebyshevApproximation(
            lambda x, _: x[0] ** 2 + x[1] ** 2,
            2, [[-1, 1], [-1, 1]], [11, 11],
            special_points=[[], []],
        )
        assert type(obj) is ChebyshevApproximation

    def test_kink_returns_spline(self):
        obj = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            n_nodes=[[11, 11]],
            special_points=[[0.0]],
        )
        assert type(obj) is ChebyshevSpline

    def test_kink_2d_one_dim_returns_spline(self):
        obj = ChebyshevApproximation(
            _abs_sum_2d, 2, [[-1, 1], [-1, 1]],
            n_nodes=[[11, 11], [13]],
            special_points=[[0.0], []],
        )
        assert type(obj) is ChebyshevSpline
        assert obj.knots == [[0.0], []]

    def test_kink_both_dims_returns_spline(self):
        obj = ChebyshevApproximation(
            _abs_sum_2d, 2, [[-1, 1], [-1, 1]],
            n_nodes=[[11, 11], [11, 11]],
            special_points=[[0.0], [0.0]],
        )
        assert type(obj) is ChebyshevSpline
        assert obj._shape == (2, 2)

    def test_dispatch_passes_error_threshold(self):
        obj = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            special_points=[[0.0]],
            error_threshold=1e-6,
        )
        assert type(obj) is ChebyshevSpline
        assert obj.error_threshold == 1e-6

    def test_dispatch_passes_max_n(self):
        obj = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            special_points=[[0.0]],
            error_threshold=1e-6,
            max_n=32,
        )
        assert obj.max_n == 32

    def test_init_signature_accepts_special_points_when_none(self):
        # __init__ must accept special_points as a kwarg (via the
        # single-tensor path, Python calls __init__ with full kwargs).
        obj = ChebyshevApproximation(
            lambda x, _: x[0], 1, [[-1, 1]], [11],
            special_points=None,
        )
        assert type(obj) is ChebyshevApproximation


class TestValidation:
    """Validation errors for special_points + nested n_nodes."""

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must have 2 entries"):
            ChebyshevApproximation(
                _abs_sum_2d, 2, [[-1, 1], [-1, 1]],
                n_nodes=[[11, 11]],
                special_points=[[0.0]],
            )

    def test_unsorted_points_raises(self):
        with pytest.raises(ValueError, match="must be sorted"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[[11, 11, 11]],
                special_points=[[0.5, -0.5]],
            )

    def test_point_on_boundary_raises(self):
        with pytest.raises(ValueError, match="not strictly inside"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[[11, 11]],
                special_points=[[1.0]],
            )

    def test_point_outside_domain_raises(self):
        with pytest.raises(ValueError, match="not strictly inside"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[[11, 11]],
                special_points=[[2.0]],
            )

    def test_coinciding_points_raises(self):
        with pytest.raises(ValueError, match="[Cc]oinciding"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[[11, 11, 11]],
                special_points=[[0.3, 0.3]],
            )

    def test_flat_n_nodes_with_special_points_raises(self):
        with pytest.raises(ValueError, match="nested"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[11],
                special_points=[[0.0]],
            )

    def test_wrong_nested_inner_length_raises(self):
        with pytest.raises(ValueError, match="must have 2 entries"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[[11]],
                special_points=[[0.0]],
            )

    def test_mixed_nested_and_flat_raises(self):
        with pytest.raises(ValueError, match="fully nested"):
            ChebyshevApproximation(
                _abs_sum_2d, 2, [[-1, 1], [-1, 1]],
                n_nodes=[[11, 11], 13],
                special_points=[[0.0], []],
            )

    def test_non_list_inner_raises(self):
        with pytest.raises(ValueError, match="must be a list/tuple"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[[11, 11]],
                special_points=[0.0],   # forgot inner list
            )

    def test_none_inner_raises(self):
        with pytest.raises(ValueError, match="must be a list/tuple"):
            ChebyshevApproximation(
                _abs_sum_2d, 2, [[-1, 1], [-1, 1]],
                n_nodes=[11, 11],
                special_points=[None, [0.0]],
            )

    def test_outer_length_mismatch_all_empty_raises(self):
        # All-empty inner lists must still fail outer-length validation.
        with pytest.raises(ValueError, match="must have 2 entries"):
            ChebyshevApproximation(
                _abs_sum_2d, 2, [[-1, 1], [-1, 1]],
                n_nodes=[11, 11],
                special_points=[[], [], []],
            )


class TestAccuracy1D:
    """Special points restore spectral convergence at kinks."""

    def test_abs_kink_reaches_machine_precision(self):
        cheb = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            n_nodes=[[11, 11]],
            special_points=[[0.0]],
        )
        cheb.build(verbose=False)
        for x in np.linspace(-0.95, 0.95, 41):
            if abs(x) < 1e-8:
                continue
            assert cheb.eval([float(x)], [0]) == pytest.approx(abs(x), abs=1e-13)

    def test_abs_without_special_points_plateaus(self):
        # Control: abs(x) without a kink declaration — spectral methods
        # plateau at low-single-digit precision even with many nodes.
        cheb = ChebyshevApproximation(_abs1d, 1, [[-1, 1]], [31])
        cheb.build(verbose=False)
        max_err = 0.0
        for x in np.linspace(-0.95, 0.95, 41):
            max_err = max(max_err, abs(cheb.eval([float(x)], [0]) - abs(x)))
        assert max_err > 1e-3

    def test_abs_value_kink_off_origin(self):
        def f(x, _):
            return abs(x[0] - 0.3)
        cheb = ChebyshevApproximation(
            f, 1, [[-1, 1]],
            n_nodes=[[11, 11]],
            special_points=[[0.3]],
        )
        cheb.build(verbose=False)
        for x in np.linspace(-0.95, 0.95, 31):
            if abs(x - 0.3) < 1e-8:
                continue
            assert cheb.eval([float(x)], [0]) == pytest.approx(
                abs(x - 0.3), abs=1e-13
            )

    def test_multiple_kinks(self):
        def f(x, _):
            return abs(x[0] + 0.5) + abs(x[0]) + abs(x[0] - 0.5)
        cheb = ChebyshevApproximation(
            f, 1, [[-1, 1]],
            n_nodes=[[5, 5, 5, 5]],
            special_points=[[-0.5, 0.0, 0.5]],
        )
        cheb.build(verbose=False)
        for x in np.linspace(-0.9, 0.9, 25):
            kink_dists = [abs(x - k) for k in [-0.5, 0.0, 0.5]]
            if min(kink_dists) < 1e-8:
                continue
            expected = abs(x + 0.5) + abs(x) + abs(x - 0.5)
            assert cheb.eval([float(x)], [0]) == pytest.approx(
                expected, abs=1e-13
            )

    def test_error_threshold_with_kink(self):
        cheb = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            special_points=[[0.0]],
            error_threshold=1e-10,
        )
        cheb.build(verbose=False)
        for x in [-0.5, 0.5]:
            assert cheb.eval([x], [0]) == pytest.approx(abs(x), abs=1e-10)


class TestAccuracy2D:
    def test_kink_one_dim_only(self):
        def f(x, _):
            return abs(x[0]) * (1.0 + x[1] ** 2)
        cheb = ChebyshevApproximation(
            f, 2, [[-1, 1], [-1, 1]],
            n_nodes=[[9, 9], [11]],
            special_points=[[0.0], []],
        )
        cheb.build(verbose=False)
        for x in [-0.7, -0.2, 0.3, 0.8]:
            for y in [-0.9, -0.3, 0.5, 0.7]:
                expected = abs(x) * (1.0 + y ** 2)
                assert cheb.eval([x, y], [0, 0]) == pytest.approx(
                    expected, abs=1e-12
                )

    def test_kink_both_dims(self):
        cheb = ChebyshevApproximation(
            _abs_sum_2d, 2, [[-1, 1], [-1, 1]],
            n_nodes=[[7, 7], [7, 7]],
            special_points=[[0.0], [0.0]],
        )
        cheb.build(verbose=False)
        assert cheb._shape == (2, 2)
        for x in [-0.6, 0.4]:
            for y in [-0.8, 0.2]:
                expected = abs(x) + abs(y)
                assert cheb.eval([x, y], [0, 0]) == pytest.approx(
                    expected, abs=1e-13
                )

    def test_per_sub_interval_different_ns(self):
        def f(x, _):
            return abs(x[0]) + x[1] ** 4
        cheb = ChebyshevApproximation(
            f, 2, [[-1, 1], [-1, 1]],
            n_nodes=[[7, 9], [11]],
            special_points=[[0.2], []],
        )
        cheb.build(verbose=False)
        assert cheb._pieces[0].n_nodes == [7, 11]
        assert cheb._pieces[1].n_nodes == [9, 11]


class TestCrossFeature:
    """special_points results work with algebra, calculus, serialization,
    extrude/slice, and from_values — all via ChebyshevSpline's existing
    implementations."""

    def test_save_load_roundtrip(self, tmp_path):
        cheb = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            n_nodes=[[11, 11]],
            special_points=[[0.0]],
        )
        cheb.build(verbose=False)
        path = tmp_path / "sp.pkl"
        with open(path, "wb") as fh:
            pickle.dump(cheb, fh)
        with open(path, "rb") as fh:
            loaded = pickle.load(fh)
        assert type(loaded) is ChebyshevSpline
        for x in [-0.5, 0.2, 0.8]:
            assert loaded.eval([x], [0]) == cheb.eval([x], [0])

    def test_algebra_with_sibling(self):
        a = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            n_nodes=[[11, 11]],
            special_points=[[0.0]],
        )
        b = ChebyshevApproximation(
            lambda x, _: x[0] ** 2, 1, [[-1, 1]],
            n_nodes=[[11, 11]],
            special_points=[[0.0]],
        )
        a.build(verbose=False)
        b.build(verbose=False)
        c = a + b
        for x in [-0.5, 0.3, 0.7]:
            expected = abs(x) + x ** 2
            assert c.eval([x], [0]) == pytest.approx(expected, abs=1e-12)

    def test_integrate(self):
        cheb = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            n_nodes=[[11, 11]],
            special_points=[[0.0]],
        )
        cheb.build(verbose=False)
        assert cheb.integrate() == pytest.approx(1.0, abs=1e-12)

    def test_extrude_slice_roundtrip(self):
        cheb = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            n_nodes=[[9, 9]],
            special_points=[[0.0]],
        )
        cheb.build(verbose=False)
        extruded = cheb.extrude((1, (-2, 2), 5))
        sliced = extruded.slice((1, 0.7))
        for x in [-0.6, 0.4]:
            assert sliced.eval([x], [0]) == pytest.approx(
                cheb.eval([x], [0]), abs=1e-12
            )

    def test_from_values_with_knots(self):
        # ChebyshevSpline.nodes() / from_values() take flat n_nodes
        # (shared per-piece). This test exercises the spline from_values
        # path on a function with a kink — the cross-feature integration
        # that matters — using the flat form that those APIs support.
        # The kink is declared at x[0] = 0.3 and the function has its
        # kink there (so each piece is smooth).
        info = ChebyshevSpline.nodes(
            2, [[-1, 1], [-1, 1]],
            n_nodes=[7, 9],
            knots=[[0.3], []],
        )
        piece_values = []
        for piece in info["pieces"]:
            grid = piece["full_grid"]  # shape (prod, 2)
            vals = np.array([abs(p[0] - 0.3) + p[1] ** 2 for p in grid])
            piece_values.append(vals.reshape(piece["shape"]))
        sp = ChebyshevSpline.from_values(
            piece_values, 2, [[-1, 1], [-1, 1]], [7, 9], [[0.3], []],
        )
        for x in [-0.4, 0.5]:
            for y in [-0.8, 0.1]:
                expected = abs(x - 0.3) + y ** 2
                assert sp.eval([x, y], [0, 0]) == pytest.approx(
                    expected, abs=1e-10
                )


class TestEdgeCases:
    def test_get_optimal_n1_no_special_points_works(self):
        n = ChebyshevApproximation.get_optimal_n1(
            lambda x, _: math.sin(x[0]), (-1, 1), error_threshold=1e-6
        )
        assert n >= 3

    def test_get_optimal_n1_rejects_special_points_kwarg(self):
        with pytest.raises(TypeError):
            ChebyshevApproximation.get_optimal_n1(
                lambda x, _: abs(x[0]),
                (-1, 1),
                error_threshold=1e-6,
                special_points=[[0.0]],
            )

    def test_many_kinks_one_dim(self):
        knots = [-0.6, -0.2, 0.1, 0.5]
        n_per_piece = [5] * (len(knots) + 1)
        def f(x, _):
            return sum(abs(x[0] - k) for k in knots)
        cheb = ChebyshevApproximation(
            f, 1, [[-1, 1]],
            n_nodes=[n_per_piece],
            special_points=[knots],
        )
        cheb.build(verbose=False)
        for x in [-0.7, -0.4, 0.0, 0.3, 0.7]:
            expected = sum(abs(x - k) for k in knots)
            assert cheb.eval([x], [0]) == pytest.approx(expected, abs=1e-13)

    def test_all_empty_special_points_behaves_like_single_tensor(self):
        a = ChebyshevApproximation(
            lambda x, _: x[0] ** 2 + x[1] ** 2,
            2, [[-1, 1], [-1, 1]], [11, 11],
        )
        b = ChebyshevApproximation(
            lambda x, _: x[0] ** 2 + x[1] ** 2,
            2, [[-1, 1], [-1, 1]], [11, 11],
            special_points=[[], []],
        )
        a.build(verbose=False)
        b.build(verbose=False)
        for x in [-0.4, 0.6]:
            for y in [-0.2, 0.8]:
                assert a.eval([x, y], [0, 0]) == b.eval([x, y], [0, 0])

    def test_unbuilt_special_points_spline(self):
        cheb = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            n_nodes=[[11, 11]],
            special_points=[[0.0]],
        )
        assert type(cheb) is ChebyshevSpline
        assert cheb._built is False
