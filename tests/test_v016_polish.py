"""Tests for v0.16 Polish Bundle."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pychebyshev import (
    ChebyshevApproximation,
    ChebyshevSlider,
    ChebyshevSpline,
    ChebyshevTT,
)


# ============================================================================
# A3: get_max_derivative_order()
# ============================================================================

class TestGetMaxDerivativeOrder:
    def test_approximation_returns_default(self, cheb_sin_3d):
        assert cheb_sin_3d.get_max_derivative_order() == 2

    def test_approximation_returns_custom(self):
        def f(x, _):
            return x[0] ** 4

        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [8], max_derivative_order=4)
        cheb.build(verbose=False)
        assert cheb.get_max_derivative_order() == 4

    def test_spline_returns_value(self, spline_abs_1d):
        assert spline_abs_1d.get_max_derivative_order() == 2

    def test_slider_returns_value(self):
        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1])

        slider = ChebyshevSlider(
            f, 2, [[-1, 1], [-1, 1]], [8, 8], partition=[[0], [1]],
            pivot_point=[0.0, 0.0], max_derivative_order=3,
        )
        slider.build(verbose=False)
        assert slider.get_max_derivative_order() == 3

    def test_tt_returns_value(self, tt_sin_3d):
        assert tt_sin_3d.get_max_derivative_order() == 2

    def test_tt_returns_custom(self):
        def f(x, _):
            return math.sin(x[0]) + math.cos(x[1])

        tt = ChebyshevTT(
            f, 2, [[-1, 1], [-1, 1]], [8, 8], max_derivative_order=4,
        )
        tt.build(verbose=False)
        assert tt.get_max_derivative_order() == 4

    def test_tt_max_derivative_order_survives_pickle(self, tmp_path):
        def f(x, _):
            return math.sin(x[0])

        tt = ChebyshevTT(
            f, 1, [[-1, 1]], [8], max_derivative_order=3,
        )
        tt.build(verbose=False)
        path = tmp_path / "tt.pkl"
        tt.save(str(path))
        loaded = ChebyshevTT.load(str(path))
        assert loaded.get_max_derivative_order() == 3

    def test_tt_setstate_backfill_for_legacy_pickle(self):
        """Pre-v0.16 pickles lack max_derivative_order — verify __setstate__ backfills it."""
        def f(x, _):
            return math.sin(x[0])

        tt = ChebyshevTT(f, 1, [[-1, 1]], [8])
        tt.build(verbose=False)
        # Strip the v0.16 attribute to simulate a pre-v0.16 pickle
        state = tt.__getstate__()
        if "max_derivative_order" in state:
            del state["max_derivative_order"]

        # Reconstruct and verify backfill
        restored = ChebyshevTT.__new__(ChebyshevTT)
        restored.__setstate__(state)
        assert restored.get_max_derivative_order() == 2  # default backfill


# ============================================================================
# A4: get_error_threshold()
# ============================================================================

class TestGetErrorThreshold:
    def test_approximation_with_threshold(self):
        def f(x, _):
            return math.sin(x[0])

        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], error_threshold=1e-6)
        cheb.build(verbose=False)
        assert cheb.get_error_threshold() == 1e-6

    def test_approximation_without_threshold(self, cheb_sin_3d):
        assert cheb_sin_3d.get_error_threshold() is None

    def test_spline_with_threshold(self):
        def f(x, _):
            return abs(x[0])

        spl = ChebyshevSpline(
            f, 1, [[-1, 1]], knots=[[0.0]], error_threshold=1e-5,
        )
        spl.build(verbose=False)
        assert spl.get_error_threshold() == 1e-5

    def test_spline_without_threshold(self, spline_abs_1d):
        assert spline_abs_1d.get_error_threshold() is None


# ============================================================================
# A2: get_special_points()
# ============================================================================

class TestGetSpecialPoints:
    def test_approximation_no_special_points(self, cheb_sin_3d):
        assert cheb_sin_3d.get_special_points() is None

    def test_approximation_all_empty_special_points(self):
        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1])

        cheb = ChebyshevApproximation(
            f, 2, [[-1, 1], [-1, 1]], [8, 8], special_points=[[], []],
        )
        cheb.build(verbose=False)
        # All-empty: __new__ does NOT dispatch to Spline; we keep an Approximation
        assert isinstance(cheb, ChebyshevApproximation)
        assert cheb.get_special_points() == [[], []]

    def test_spline_returns_knots_per_dim(self, spline_abs_1d):
        # spline_abs_1d is built with knots=[[0.0]]
        sp = spline_abs_1d.get_special_points()
        assert sp == [[0.0]]

    def test_approximation_dispatches_to_spline_when_kink_declared(self):
        def f(x, _):
            return abs(x[0])

        # __new__ dispatch route: special_points with any non-empty list →
        # returns a ChebyshevSpline.
        # n_nodes must be nested when special_points is non-empty (per v0.12 API).
        obj = ChebyshevApproximation(
            f, 1, [[-1, 1]], [[8, 8]], special_points=[[0.0]],
        )
        assert isinstance(obj, ChebyshevSpline)
        assert obj.get_special_points() == [[0.0]]

    def test_round_trip_pickle(self, spline_abs_1d, tmp_path):
        path = tmp_path / "spl.pkl"
        spline_abs_1d.save(str(path))
        loaded = ChebyshevSpline.load(str(path))
        assert loaded.get_special_points() == spline_abs_1d.get_special_points()

    def test_round_trip_pcb(self, spline_abs_1d, tmp_path):
        path = tmp_path / "spl.pcb"
        spline_abs_1d.save(str(path), format="binary")
        loaded = ChebyshevSpline.load(str(path))
        assert loaded.get_special_points() == spline_abs_1d.get_special_points()


# ============================================================================
# A9: is_dimensionality_allowed() static
# ============================================================================

class TestIsDimensionalityAllowed:
    @pytest.mark.parametrize("cls", [
        ChebyshevApproximation, ChebyshevSpline, ChebyshevSlider, ChebyshevTT,
    ])
    def test_positive_dim_allowed(self, cls):
        assert cls.is_dimensionality_allowed(1) is True
        assert cls.is_dimensionality_allowed(2) is True
        assert cls.is_dimensionality_allowed(10) is True

    @pytest.mark.parametrize("cls", [
        ChebyshevApproximation, ChebyshevSpline, ChebyshevSlider, ChebyshevTT,
    ])
    def test_zero_or_negative_disallowed(self, cls):
        assert cls.is_dimensionality_allowed(0) is False
        assert cls.is_dimensionality_allowed(-1) is False

    @pytest.mark.parametrize("cls", [
        ChebyshevApproximation, ChebyshevSpline, ChebyshevSlider, ChebyshevTT,
    ])
    def test_callable_without_instance(self, cls):
        # Static method: callable on the class itself
        assert callable(cls.is_dimensionality_allowed)
        # Type signature
        result = cls.is_dimensionality_allowed(3)
        assert isinstance(result, bool)


# ============================================================================
# A6: get_num_evaluation_points()
# ============================================================================

class TestGetNumEvaluationPoints:
    def test_approximation_product_of_n_nodes(self, cheb_sin_3d):
        # Built with [10, 8, 4]
        assert cheb_sin_3d.get_num_evaluation_points() == 10 * 8 * 4

    def test_spline_sum_across_pieces(self, spline_abs_1d):
        """Spline returns sum of per-piece grid sizes (matching grid, not work)."""
        expected = sum(
            int(np.prod(piece.n_nodes)) for piece in spline_abs_1d._pieces
        )
        assert spline_abs_1d.get_num_evaluation_points() == expected

    def test_slider_matches_total_build_evals(self):
        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1])

        slider = ChebyshevSlider(
            f, 2, [[-1, 1], [-1, 1]], [8, 8], partition=[[0], [1]],
            pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        assert slider.get_num_evaluation_points() == slider.total_build_evals

    def test_tt_returns_full_grid_size(self, tt_sin_3d):
        """TT returns prod(n_nodes), the full Cartesian grid (not the sparse cross sample count)."""
        expected = int(np.prod(tt_sin_3d.n_nodes))
        assert tt_sin_3d.get_num_evaluation_points() == expected


# ============================================================================
# A5: get_evaluation_points()
# ============================================================================

class TestGetEvaluationPoints:
    def test_approximation_shape(self, cheb_sin_3d):
        pts = cheb_sin_3d.get_evaluation_points()
        assert pts.shape == (10 * 8 * 4, 3)
        assert pts.dtype == np.float64

    def test_approximation_within_domain(self, cheb_sin_3d):
        pts = cheb_sin_3d.get_evaluation_points()
        # Domain is [[-1,1], [-1,1], [1,3]]
        assert pts[:, 0].min() >= -1.0 and pts[:, 0].max() <= 1.0
        assert pts[:, 1].min() >= -1.0 and pts[:, 1].max() <= 1.0
        assert pts[:, 2].min() >= 1.0 and pts[:, 2].max() <= 3.0

    def test_approximation_count_matches_num_eval_points(self, cheb_sin_3d):
        pts = cheb_sin_3d.get_evaluation_points()
        assert len(pts) == cheb_sin_3d.get_num_evaluation_points()

    def test_approximation_unique_nodes_per_dim(self):
        def f(x, _):
            return x[0] + x[1]

        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [0, 1]], [4, 4])
        cheb.build(verbose=False)
        pts = cheb.get_evaluation_points()
        assert pts.shape == (16, 2)
        assert len(np.unique(pts[:, 0])) == 4
        assert len(np.unique(pts[:, 1])) == 4

    def test_spline_concatenates_pieces(self, spline_abs_1d):
        pts = spline_abs_1d.get_evaluation_points()
        assert pts.shape[1] == 1
        assert len(pts) == spline_abs_1d.get_num_evaluation_points()

    def test_slider_returns_2d_array(self):
        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1])

        slider = ChebyshevSlider(
            f, 2, [[-1, 1], [-1, 1]], [8, 8], partition=[[0], [1]],
            pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        pts = slider.get_evaluation_points()
        assert pts.ndim == 2
        assert pts.shape[1] == 2
        assert len(pts) == slider.get_num_evaluation_points()

    def test_tt_returns_2d_array(self, tt_sin_3d):
        pts = tt_sin_3d.get_evaluation_points()
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert len(pts) == tt_sin_3d.get_num_evaluation_points()

    def test_consistency_with_get_num_eval_points(self):
        """The fundamental contract: len(get_evaluation_points) == get_num_evaluation_points."""
        def f(x, _):
            return x[0] * x[1]

        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [5, 7])
        cheb.build(verbose=False)
        pts = cheb.get_evaluation_points()
        assert len(pts) == cheb.get_num_evaluation_points()
        assert len(pts) == 5 * 7


# ============================================================================
# A1: clone()
# ============================================================================

class TestClone:
    @pytest.mark.parametrize("fixture_name", [
        "cheb_sin_3d", "spline_abs_1d", "tt_sin_3d",
    ])
    def test_clone_returns_distinct_object(self, request, fixture_name):
        original = request.getfixturevalue(fixture_name)
        clone = original.clone()
        assert clone is not original

    def test_slider_clone(self):
        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1])

        slider = ChebyshevSlider(
            f, 2, [[-1, 1], [-1, 1]], [8, 8], partition=[[0], [1]],
            pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        clone = slider.clone()
        assert clone is not slider

    def test_clone_eval_matches_original(self, cheb_sin_3d):
        clone = cheb_sin_3d.clone()
        deriv = [0, 0, 0]
        for x in [[0.0, 0.0, 2.0], [0.5, -0.5, 1.5]]:
            assert clone.eval(x, deriv) == cheb_sin_3d.eval(x, deriv)

    def test_clone_descriptor_isolation(self, cheb_sin_3d):
        cheb_sin_3d.set_descriptor("original")
        clone = cheb_sin_3d.clone()
        clone.set_descriptor("modified")
        assert cheb_sin_3d.get_descriptor() == "original"
        assert clone.get_descriptor() == "modified"

    def test_clone_tensor_isolation(self, cheb_sin_3d):
        clone = cheb_sin_3d.clone()
        # Mutating the clone's tensor must not affect the original
        original_value = cheb_sin_3d.tensor_values[0, 0, 0]
        clone.tensor_values[0, 0, 0] = -999.0
        assert cheb_sin_3d.tensor_values[0, 0, 0] == original_value

    def test_clone_preserves_additional_data(self):
        sentinel = {"sentinel": 42}

        def f(x, ad):
            return ad["sentinel"] + x[0]

        cheb = ChebyshevApproximation(
            f, 1, [[-1, 1]], [4], additional_data=sentinel,
        )
        cheb.build(verbose=False)
        clone = cheb.clone()
        # additional_data is deepcopied
        assert clone.additional_data == sentinel
        assert clone.additional_data is not sentinel

    def test_clone_of_factory_built_object(self):
        """Object built via from_values has function=None; clone preserves that."""
        def f(x, _):
            return x[0] ** 2

        original = ChebyshevApproximation(f, 1, [[-1, 1]], [5])
        original.build(verbose=False)
        from_vals = ChebyshevApproximation.from_values(
            original.tensor_values, 1, [[-1, 1]], [5],
        )
        clone = from_vals.clone()
        assert clone.function is None
        assert clone.eval([0.5], [0]) == from_vals.eval([0.5], [0])

    def test_clone_preserves_derivative_id_registry(self, cheb_sin_3d):
        cheb_sin_3d.get_derivative_id([1, 0, 0])  # register
        clone = cheb_sin_3d.clone()
        # Same derivative order returns the same id on the clone
        original_id = cheb_sin_3d.get_derivative_id([1, 0, 0])
        clone_id = clone.get_derivative_id([1, 0, 0])
        assert original_id == clone_id

    def test_clone_strips_function_via_getstate(self, cheb_sin_3d):
        """clone() goes through __getstate__/__setstate__, which sets function=None."""
        clone = cheb_sin_3d.clone()
        assert clone.function is None
        # Original retains its function
        assert cheb_sin_3d.function is not None


# ============================================================================
# A8: peek_format_version()
# ============================================================================

class TestPeekFormatVersion:
    def test_peek_returns_v1_for_current_format(self, cheb_sin_3d, tmp_path):
        path = tmp_path / "model.pcb"
        cheb_sin_3d.save(str(path), format="binary")
        assert ChebyshevApproximation.peek_format_version(str(path)) == 1

    def test_peek_returns_v1_for_spline_pcb(self, spline_abs_1d, tmp_path):
        path = tmp_path / "spline.pcb"
        spline_abs_1d.save(str(path), format="binary")
        assert ChebyshevApproximation.peek_format_version(str(path)) == 1

    def test_peek_invalid_magic_raises(self, tmp_path):
        path = tmp_path / "fake.pcb"
        path.write_bytes(b"NOTPCB\x00\x00\x01\x00\x00\x00")
        with pytest.raises(ValueError, match="not a .pcb file|magic"):
            ChebyshevApproximation.peek_format_version(str(path))

    def test_peek_truncated_file_raises(self, tmp_path):
        path = tmp_path / "trunc.pcb"
        path.write_bytes(b"PCB\x00\x01")  # only 5 bytes, header is 12
        with pytest.raises((ValueError, IOError)):
            ChebyshevApproximation.peek_format_version(str(path))

    def test_peek_nonexistent_file_raises(self, tmp_path):
        path = tmp_path / "missing.pcb"
        with pytest.raises((FileNotFoundError, IOError)):
            ChebyshevApproximation.peek_format_version(str(path))

    def test_peek_does_not_load_full_file(self, cheb_sin_3d, tmp_path):
        """Sanity: peek must not call full deserialize. Truncate after the
        header — peek returns the version, but full load fails."""
        path = tmp_path / "head_only.pcb"
        cheb_sin_3d.save(str(path), format="binary")
        with open(path, "rb") as f:
            full = f.read()
        path.write_bytes(full[:12])
        # peek succeeds
        assert ChebyshevApproximation.peek_format_version(str(path)) == 1
        # full load fails
        with pytest.raises((ValueError, IOError, EOFError)):
            ChebyshevApproximation.load(str(path))


# ============================================================================
# A7: set_original_function_values() + defer_build flag
# ============================================================================

class TestSetOriginalFunctionValues:
    def test_defer_build_creates_empty_object(self):
        cheb = ChebyshevApproximation(
            None, 2, [[-1, 1], [-1, 1]], [4, 5], defer_build=True,
        )
        assert cheb.tensor_values is None
        assert cheb.is_construction_finished() is False

    def test_set_values_then_evaluable(self):
        def f(x, _):
            return math.sin(x[0]) + math.cos(x[1])

        # Reference: standard build
        ref = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [6, 7])
        ref.build(verbose=False)
        ref_vals = ref.tensor_values

        # Test: defer_build + set_original_function_values
        deferred = ChebyshevApproximation(
            None, 2, [[-1, 1], [-1, 1]], [6, 7], defer_build=True,
        )
        deferred.set_original_function_values(ref_vals)
        assert deferred.is_construction_finished() is True
        assert deferred.eval([0.3, -0.2], [0, 0]) == pytest.approx(ref.eval([0.3, -0.2], [0, 0]))

    def test_bit_identical_to_from_values(self):
        def f(x, _):
            return x[0] * x[1]

        ref = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [4, 4])
        ref.build(verbose=False)

        via_factory = ChebyshevApproximation.from_values(
            ref.tensor_values, 2, [[-1, 1], [-1, 1]], [4, 4],
        )
        via_in_place = ChebyshevApproximation(
            None, 2, [[-1, 1], [-1, 1]], [4, 4], defer_build=True,
        )
        via_in_place.set_original_function_values(ref.tensor_values)

        np.testing.assert_array_equal(
            via_factory.tensor_values, via_in_place.tensor_values
        )

    def test_double_call_rejected(self):
        cheb = ChebyshevApproximation(
            None, 1, [[-1, 1]], [4], defer_build=True,
        )
        vals = np.zeros(4)
        cheb.set_original_function_values(vals)
        with pytest.raises(RuntimeError, match="already"):
            cheb.set_original_function_values(vals)

    def test_call_on_built_object_rejected(self, cheb_sin_3d):
        with pytest.raises(RuntimeError, match="already"):
            cheb_sin_3d.set_original_function_values(cheb_sin_3d.tensor_values)

    def test_shape_mismatch_rejected(self):
        cheb = ChebyshevApproximation(
            None, 2, [[-1, 1], [-1, 1]], [4, 5], defer_build=True,
        )
        with pytest.raises(ValueError, match="shape"):
            cheb.set_original_function_values(np.zeros((3, 5)))

    def test_nan_inf_rejected(self):
        cheb = ChebyshevApproximation(
            None, 1, [[-1, 1]], [4], defer_build=True,
        )
        bad = np.array([0.0, float("nan"), 0.0, 0.0])
        with pytest.raises(ValueError, match="NaN|Inf|finite"):
            cheb.set_original_function_values(bad)

    def test_descriptor_preserved_through_defer_set(self):
        cheb = ChebyshevApproximation(
            None, 1, [[-1, 1]], [4], defer_build=True,
        )
        cheb.set_descriptor("deferred")
        cheb.set_original_function_values(np.zeros(4))
        assert cheb.get_descriptor() == "deferred"

    def test_spline_defer_and_set(self):
        def f(x, _):
            return abs(x[0])

        ref = ChebyshevSpline(f, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[8])
        ref.build(verbose=False)

        deferred = ChebyshevSpline(
            None, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[8], defer_build=True,
        )
        per_piece = [piece.tensor_values for piece in ref._pieces]
        deferred.set_original_function_values(per_piece)
        assert deferred.eval([-0.5], [0]) == pytest.approx(ref.eval([-0.5], [0]))

    def test_function_attribute_is_none_after_defer_set(self):
        cheb = ChebyshevApproximation(
            None, 1, [[-1, 1]], [4], defer_build=True,
        )
        cheb.set_original_function_values(np.zeros(4))
        assert cheb.function is None

    def test_special_points_dispatches_to_deferred_spline(self):
        """defer_build=True + special_points kink → __new__ dispatches to a deferred Spline."""
        obj = ChebyshevApproximation(
            None, 1, [[-1, 1]], n_nodes=[[5, 5]],
            special_points=[[0.0]], defer_build=True,
        )
        assert isinstance(obj, ChebyshevSpline)
        assert obj.is_construction_finished() is False
        # Fill per-piece values
        per_piece = [
            np.zeros(5),  # left piece
            np.ones(5),   # right piece
        ]
        obj.set_original_function_values(per_piece)
        assert obj.is_construction_finished() is True
        # Eval works: left half returns 0, right half returns 1
        assert obj.eval([-0.5], [0]) == pytest.approx(0.0)
        assert obj.eval([0.5], [0]) == pytest.approx(1.0)

    def test_spline_set_values_atomic_on_validation_failure(self):
        """Spline set_original_function_values must not partially mutate on per-piece failure."""
        deferred = ChebyshevSpline(
            None, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[8],
            defer_build=True,
        )
        # Bad: piece 1 has wrong shape
        with pytest.raises(ValueError, match="shape"):
            deferred.set_original_function_values([np.zeros(8), np.zeros(3)])
        # Spline must still be in deferred state — retry with correct shapes succeeds
        assert deferred.is_construction_finished() is False
        deferred.set_original_function_values([np.zeros(8), np.ones(8)])
        assert deferred.is_construction_finished() is True

    def test_spline_defer_threads_additional_data_to_pieces(self):
        """defer_build path must thread additional_data to pieces, matching build()."""
        sentinel = {"x": 7}
        deferred = ChebyshevSpline(
            None, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[4],
            additional_data=sentinel, defer_build=True,
        )
        for piece in deferred._pieces:
            assert piece.additional_data == sentinel


# ============================================================================
# A10: typed helpers Domain, Ns, SpecialPoints
# ============================================================================

class TestTypedHelpers:
    def test_domain_dataclass_is_frozen(self):
        from pychebyshev import Domain
        d = Domain([(0.0, 1.0), (-1.0, 1.0)])
        with pytest.raises(Exception):  # FrozenInstanceError
            d.bounds = []

    def test_ns_dataclass_is_frozen(self):
        from pychebyshev import Ns
        n = Ns([10, 12])
        with pytest.raises(Exception):
            n.counts = []

    def test_approximation_accepts_typed_domain(self):
        from pychebyshev import Domain

        def f(x, _):
            return x[0] ** 2

        cheb = ChebyshevApproximation(f, 1, Domain([(-1.0, 1.0)]), [4])
        cheb.build(verbose=False)
        assert cheb.eval([0.5], [0]) == pytest.approx(0.25, abs=1e-3)

    def test_approximation_accepts_typed_ns(self):
        from pychebyshev import Ns

        def f(x, _):
            return x[0] + x[1]

        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], Ns([4, 5]))
        cheb.build(verbose=False)
        assert cheb.n_nodes == [4, 5]

    def test_approximation_mixed_typed_and_raw(self):
        from pychebyshev import Domain, Ns

        def f(x, _):
            return math.sin(x[0])

        cheb = ChebyshevApproximation(f, 1, Domain([(-1.0, 1.0)]), Ns([6]))
        cheb.build(verbose=False)
        assert cheb.eval([0.0], [0]) == pytest.approx(0.0, abs=1e-6)

    def test_approximation_accepts_typed_special_points(self):
        from pychebyshev import SpecialPoints

        def f(x, _):
            return abs(x[0])

        # SpecialPoints with a kink → __new__ dispatches to Spline
        obj = ChebyshevApproximation(
            f, 1, [[-1, 1]], n_nodes=[[8, 8]],
            special_points=SpecialPoints([[0.0]]),
        )
        assert isinstance(obj, ChebyshevSpline)

    def test_typed_and_raw_produce_identical_interpolants(self):
        from pychebyshev import Domain, Ns

        def f(x, _):
            return x[0] * x[1]

        a = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [5, 5])
        a.build(verbose=False)
        b = ChebyshevApproximation(
            f, 2, Domain([(-1.0, 1.0), (-1.0, 1.0)]), Ns([5, 5]),
        )
        b.build(verbose=False)
        np.testing.assert_array_equal(a.tensor_values, b.tensor_values)

    def test_slider_accepts_typed_domain(self):
        from pychebyshev import Domain

        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1])

        slider = ChebyshevSlider(
            f, 2, Domain([(-1.0, 1.0), (-1.0, 1.0)]), [8, 8],
            partition=[[0], [1]], pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        assert slider.num_dimensions == 2

    # ------------------------------------------------------------------
    # Gap 5: spline.py:122, 124 — Domain/Ns on Spline
    # ------------------------------------------------------------------

    def test_spline_accepts_typed_domain_and_ns(self):
        from pychebyshev import Domain, Ns

        def f(x, _):
            return abs(x[0])

        # n_nodes=[8] means 8 nodes per piece (1D spline, 1 knot → 2 pieces)
        spl = ChebyshevSpline(
            f, 1, Domain([(-1.0, 1.0)]), knots=[[0.0]], n_nodes=Ns([8]),
        )
        spl.build(verbose=False)
        assert spl.eval([0.5], [0]) == pytest.approx(0.5, abs=1e-3)

    # ------------------------------------------------------------------
    # Gap 6: tensor_train.py:1047, 1049 — Domain/Ns on TT
    # ------------------------------------------------------------------

    def test_tt_accepts_typed_domain_and_ns(self):
        from pychebyshev import Domain, Ns

        def f(x, _):
            return math.sin(x[0]) * math.cos(x[1])

        tt = ChebyshevTT(
            f, 2, Domain([(-1.0, 1.0), (-1.0, 1.0)]), Ns([8, 8]),
        )
        tt.build(verbose=False)
        assert tt.num_dimensions == 2
        assert tt.n_nodes == [8, 8]

    # ------------------------------------------------------------------
    # Gap 7: slider.py:95 — Ns on Slider
    # ------------------------------------------------------------------

    def test_slider_accepts_typed_ns(self):
        from pychebyshev import Ns

        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1])

        slider = ChebyshevSlider(
            f, 2, [[-1, 1], [-1, 1]], Ns([8, 8]),
            partition=[[0], [1]], pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        assert slider.n_nodes == [8, 8]

    # ------------------------------------------------------------------
    # Gap 4: barycentric.py:360 — SpecialPoints typed helper; all-empty
    # knots stay in __init__ (line 360) since __new__ only dispatches
    # when at least one dimension has a kink.
    # ------------------------------------------------------------------

    def test_special_points_typed_all_empty_keeps_approximation(self):
        """SpecialPoints with all-empty knots: __new__ does NOT dispatch,
        so __init__ runs and unwraps the typed helper at line 360."""
        from pychebyshev import SpecialPoints

        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1])

        cheb = ChebyshevApproximation(
            f, 2, [[-1, 1], [-1, 1]], [8, 8],
            special_points=SpecialPoints([[], []]),
        )
        cheb.build(verbose=False)
        assert isinstance(cheb, ChebyshevApproximation)
        assert cheb.get_special_points() == [[], []]


# ============================================================================
# Gap 1: barycentric.py:414, 422 — defer_build rejection paths
# (appended to TestSetOriginalFunctionValues above, duplicated here as a
#  separate class to avoid ordering issues with the file structure)
# ============================================================================

class TestDeferBuildRejections:
    def test_defer_build_with_function_rejected(self):
        """defer_build=True must reject a non-None function (would be ignored)."""
        def f(x, _):
            return x[0]

        with pytest.raises(ValueError, match="function=None|requires function"):
            ChebyshevApproximation(
                f, 1, [[-1, 1]], [4], defer_build=True,
            )

    def test_defer_build_with_auto_n_rejected(self):
        """defer_build=True must reject auto-N (error_threshold) construction."""
        with pytest.raises(ValueError, match="positive int n_nodes|auto-N"):
            ChebyshevApproximation(
                None, 1, [[-1, 1]], n_nodes=None, error_threshold=1e-6,
                defer_build=True,
            )

    def test_defer_build_with_invalid_n_nodes_none_entry_rejected(self):
        """defer_build=True must reject n_nodes with None entries (even with error_threshold)."""
        # Must supply error_threshold to bypass the earlier "None entry without error_threshold"
        # guard and reach the defer_build validation at line 422.
        with pytest.raises(ValueError, match="positive int n_nodes"):
            ChebyshevApproximation(
                None, 2, [[-1, 1], [-1, 1]], n_nodes=[5, None],
                error_threshold=1e-6, defer_build=True,
            )

    def test_defer_build_with_invalid_n_nodes_zero_entry_rejected(self):
        """defer_build=True must reject n_nodes with non-positive entries."""
        with pytest.raises(ValueError, match="positive int n_nodes"):
            ChebyshevApproximation(
                None, 2, [[-1, 1], [-1, 1]], n_nodes=[5, 0], defer_build=True,
            )


# ============================================================================
# Gap 2: spline.py:286, 295, 297, 309 — set_original_function_values paths
# ============================================================================

class TestSplineSetValuesPaths:
    def test_spline_set_values_length_mismatch_rejected(self):
        """Passing wrong number of piece arrays must raise ValueError."""
        deferred = ChebyshevSpline(
            None, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[8, 8],
            defer_build=True,
        )
        # Spline has 2 pieces; pass 1 array
        with pytest.raises(ValueError, match="expected 2|piece tensors"):
            deferred.set_original_function_values([np.zeros(8)])

    def test_spline_set_values_already_built_rejected(self):
        """Calling set_original_function_values on a built spline must reject."""
        def f(x, _):
            return abs(x[0])

        spl = ChebyshevSpline(f, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[8, 8])
        spl.build(verbose=False)
        with pytest.raises(RuntimeError, match="already constructed|defer_build"):
            spl.set_original_function_values([np.zeros(8), np.zeros(8)])

    def test_spline_set_values_nan_rejected(self):
        """NaN in any piece tensor must raise ValueError."""
        # 1D spline with 1 knot (2 pieces), n_nodes=[4] → each piece has shape (4,)
        deferred = ChebyshevSpline(
            None, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[4],
            defer_build=True,
        )
        bad = np.array([0.0, float("nan"), 0.0, 0.0])
        with pytest.raises(ValueError, match="NaN|Inf|finite"):
            deferred.set_original_function_values([bad, np.zeros(4)])


# ============================================================================
# Gap 3: barycentric.py:1355 — __setstate__ backfill for special_points
# ============================================================================

class TestSetStateBackfill:
    def test_setstate_backfill_for_legacy_pickle(self):
        """Pre-v0.16 pickles lack special_points — verify __setstate__ backfills it."""
        def f(x, _):
            return math.sin(x[0])

        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [8])
        cheb.build(verbose=False)
        # Strip special_points from state to simulate a pre-v0.16 pickle
        state = cheb.__getstate__()
        if "special_points" in state:
            del state["special_points"]
        restored = ChebyshevApproximation.__new__(ChebyshevApproximation)
        restored.__setstate__(state)
        # Backfill should set special_points to None
        assert restored.get_special_points() is None
