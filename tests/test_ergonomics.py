"""Tests for v0.15 ergonomics bundle.

Currently covers: descriptor.
Will be extended for: additional_data, derivative_id, introspection trio.
"""

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


def _f3d(point, data):
    """3-D smooth function for unit tests; ignores data unless asked."""
    x, y, z = point
    return math.sin(x) + math.cos(y) * z


def _f2d(point, data):
    x, y = point
    return math.sin(x) + math.cos(y)


def _build_approx_3d():
    cheb = ChebyshevApproximation(_f3d, 3, [(-1, 1)] * 3, [9, 9, 9])
    cheb.build(verbose=False)
    return cheb


def _build_spline_2d():
    spl = ChebyshevSpline(_f2d, 2, [(-1, 1), (-1, 1)], [7, 7], [[0.0], []])
    spl.build(verbose=False)
    return spl


def _build_slider_3d():
    sld = ChebyshevSlider(
        _f3d, 3, [(-1, 1)] * 3, [7, 7, 7],
        partition=[[0], [1, 2]], pivot_point=[0.0, 0.0, 0.0],
    )
    sld.build(verbose=False)
    return sld


def _build_tt_3d():
    tt = ChebyshevTT(_f3d, 3, [(-1, 1)] * 3, [7, 7, 7], tolerance=1e-6, max_rank=8)
    tt.build(verbose=False)
    return tt


class TestDescriptor:
    """set_descriptor / get_descriptor across all four classes."""

    @pytest.mark.parametrize("builder", [
        _build_approx_3d, _build_spline_2d, _build_slider_3d, _build_tt_3d,
    ])
    def test_default_is_empty_string(self, builder):
        obj = builder()
        assert obj.get_descriptor() == ""

    @pytest.mark.parametrize("builder", [
        _build_approx_3d, _build_spline_2d, _build_slider_3d, _build_tt_3d,
    ])
    def test_set_get_round_trip(self, builder):
        obj = builder()
        obj.set_descriptor("vega-rates-curve-A")
        assert obj.get_descriptor() == "vega-rates-curve-A"

    @pytest.mark.parametrize("builder", [
        _build_approx_3d, _build_spline_2d, _build_slider_3d, _build_tt_3d,
    ])
    def test_non_string_raises_type_error(self, builder):
        obj = builder()
        with pytest.raises(TypeError, match="descriptor must be str"):
            obj.set_descriptor(123)

    @pytest.mark.parametrize("builder", [
        _build_approx_3d, _build_spline_2d, _build_slider_3d, _build_tt_3d,
    ])
    def test_pickle_round_trip_preserves(self, builder, tmp_path):
        obj = builder()
        obj.set_descriptor("label-X")
        path = tmp_path / "obj.pkl"
        obj.save(str(path))
        restored = type(obj).load(str(path))
        assert restored.get_descriptor() == "label-X"

    @pytest.mark.parametrize("builder", [_build_approx_3d, _build_spline_2d])
    def test_binary_save_load_resets_descriptor(self, builder, tmp_path):
        obj = builder()
        obj.set_descriptor("label-X")
        path = tmp_path / "obj.pcb"
        obj.save(str(path), format="binary")
        restored = type(obj).load(str(path))
        assert restored.get_descriptor() == ""

    @pytest.mark.parametrize("builder", [
        _build_approx_3d, _build_spline_2d, _build_slider_3d, _build_tt_3d,
    ])
    def test_descriptor_mutable_after_build(self, builder):
        obj = builder()
        obj.set_descriptor("first")
        obj.set_descriptor("second")
        assert obj.get_descriptor() == "second"


class TestIsConstructionFinished:
    """is_construction_finished() across classes and construction paths."""

    def test_false_after_bare_init_approx(self):
        cheb = ChebyshevApproximation(_f3d, 3, [(-1, 1)] * 3, [9, 9, 9])
        assert cheb.is_construction_finished() is False

    def test_true_after_build_all_classes(self):
        for builder in (_build_approx_3d, _build_spline_2d,
                        _build_slider_3d, _build_tt_3d):
            obj = builder()
            assert obj.is_construction_finished() is True

    def test_true_after_from_values_approx(self):
        nodes = ChebyshevApproximation.nodes(3, [(-1, 1)] * 3, [9, 9, 9])
        nodes_per_dim = nodes["nodes_per_dim"]
        values = np.zeros((9, 9, 9))
        for idx in np.ndindex(9, 9, 9):
            point = [nodes_per_dim[d][idx[d]] for d in range(3)]
            values[idx] = _f3d(point, None)
        cheb2 = ChebyshevApproximation.from_values(
            values, 3, [(-1, 1)] * 3, [9, 9, 9]
        )
        assert cheb2.is_construction_finished() is True

    def test_true_after_algebra(self):
        a = _build_approx_3d()
        b = _build_approx_3d()
        c = a + b
        assert c.is_construction_finished() is True


class TestGetConstructorType:
    """get_constructor_type() returns the class name string."""

    def test_approx(self):
        assert _build_approx_3d().get_constructor_type() == "ChebyshevApproximation"

    def test_spline(self):
        assert _build_spline_2d().get_constructor_type() == "ChebyshevSpline"

    def test_slider(self):
        assert _build_slider_3d().get_constructor_type() == "ChebyshevSlider"

    def test_tt(self):
        assert _build_tt_3d().get_constructor_type() == "ChebyshevTT"


class TestGetUsedNs:
    """get_used_ns() returns the resolved per-dim node count."""

    def test_approx_flat(self):
        cheb = _build_approx_3d()
        assert cheb.get_used_ns() == [9, 9, 9]

    def test_approx_auto_n_resolved_post_build(self):
        cheb = ChebyshevApproximation(
            _f3d, 3, [(-1, 1)] * 3, error_threshold=1e-6, max_n=64,
        )
        assert any(n is None for n in cheb.get_used_ns())  # pre-build
        cheb.build(verbose=False)
        used = cheb.get_used_ns()
        assert all(isinstance(n, int) for n in used)

    def test_spline_preserves_nested(self):
        # Use nested n_nodes (one piece per dim) to exercise the nested path.
        spl = ChebyshevSpline(_f2d, 2, [(-1, 1), (-1, 1)], [[7], [7]], [[], []])
        spl.build(verbose=False)
        used = spl.get_used_ns()
        assert used[0] == [7]  # nested for dim 0 (one piece)
        assert used[1] == [7]  # also nested

    def test_slider_flat(self):
        sld = _build_slider_3d()
        assert sld.get_used_ns() == [7, 7, 7]

    def test_tt_flat(self):
        tt = _build_tt_3d()
        assert tt.get_used_ns() == [7, 7, 7]


class TestAdditionalDataApprox:
    """additional_data on ChebyshevApproximation."""

    def test_default_is_none(self):
        cheb = ChebyshevApproximation(_f3d, 3, [(-1, 1)] * 3, [9, 9, 9])
        assert cheb.additional_data is None

    def test_attribute_stores_value(self):
        payload = {"strike": 100.0}
        cheb = ChebyshevApproximation(
            _f3d, 3, [(-1, 1)] * 3, [9, 9, 9], additional_data=payload
        )
        assert cheb.additional_data is payload

    def test_threaded_into_function_during_build(self):
        captured = []

        def f_records(point, data):
            captured.append(data)
            return point[0] + point[1] + point[2]

        payload = {"strike": 100.0}
        cheb = ChebyshevApproximation(
            f_records, 3, [(-1, 1)] * 3, [5, 5, 5], additional_data=payload
        )
        cheb.build(verbose=False)
        assert all(d is payload for d in captured)
        assert len(captured) == 5 * 5 * 5

    def test_pickle_round_trip_preserves(self, tmp_path):
        payload = {"strike": 100.0, "rate": 0.05}
        cheb = ChebyshevApproximation(
            _f3d, 3, [(-1, 1)] * 3, [9, 9, 9], additional_data=payload
        )
        cheb.build(verbose=False)
        path = tmp_path / "cheb.pkl"
        cheb.save(str(path))
        restored = ChebyshevApproximation.load(str(path))
        assert restored.additional_data == payload

    def test_binary_save_with_additional_data_raises(self, tmp_path):
        payload = {"strike": 100.0}
        cheb = ChebyshevApproximation(
            _f3d, 3, [(-1, 1)] * 3, [9, 9, 9], additional_data=payload
        )
        cheb.build(verbose=False)
        path = tmp_path / "cheb.pcb"
        with pytest.raises(NotImplementedError, match="cannot store additional_data"):
            cheb.save(str(path), format="binary")

    def test_binary_save_with_none_succeeds(self, tmp_path):
        cheb = _build_approx_3d()
        assert cheb.additional_data is None
        path = tmp_path / "cheb.pcb"
        cheb.save(str(path), format="binary")
        restored = ChebyshevApproximation.load(str(path))
        assert restored.additional_data is None


class TestAdditionalDataSpline:
    """additional_data on ChebyshevSpline propagates to pieces.

    Fixture shape: n_nodes=[[7, 7], [7]], knots=[[0.0], []]
    Dim 0 has 1 knot at 0.0 → 2 pieces, each with 7 nodes.
    Dim 1 has 0 knots → 1 piece with 7 nodes.
    Total pieces: 2 x 1 = 2.
    """

    def test_default_is_none(self):
        spl = ChebyshevSpline(_f2d, 2, [(-1, 1), (-1, 1)], [[7, 7], [7]],
                              [[0.0], []])
        assert spl.additional_data is None

    def test_propagated_to_each_piece(self):
        payload = {"strike": 100.0}
        spl = ChebyshevSpline(_f2d, 2, [(-1, 1), (-1, 1)], [[7, 7], [7]],
                              [[0.0], []], additional_data=payload)
        spl.build(verbose=False)
        # Internal pieces stored on self._pieces (list of ChebyshevApproximation)
        for piece in np.asarray(spl._pieces).flat:
            assert piece.additional_data is payload

    def test_threaded_into_function_during_build(self):
        captured = []

        def f_records(point, data):
            captured.append(data)
            return point[0] + point[1]

        payload = {"strike": 100.0}
        spl = ChebyshevSpline(f_records, 2, [(-1, 1), (-1, 1)],
                              [[5, 5], [5]], [[0.0], []],
                              additional_data=payload)
        spl.build(verbose=False)
        assert len(captured) > 0
        assert all(d is payload for d in captured)

    def test_pickle_round_trip_preserves(self, tmp_path):
        payload = {"strike": 100.0}
        spl = ChebyshevSpline(_f2d, 2, [(-1, 1), (-1, 1)], [[7, 7], [7]],
                              [[0.0], []], additional_data=payload)
        spl.build(verbose=False)
        path = tmp_path / "spl.pkl"
        spl.save(str(path))
        restored = ChebyshevSpline.load(str(path))
        assert restored.additional_data == payload

    def test_binary_save_with_additional_data_raises(self, tmp_path):
        payload = {"strike": 100.0}
        # Use flat n_nodes so the nested-n_nodes guard is not triggered first.
        spl = ChebyshevSpline(_f2d, 2, [(-1, 1), (-1, 1)], [7, 7],
                              [[0.0], []], additional_data=payload)
        spl.build(verbose=False)
        path = tmp_path / "spl.pcb"
        with pytest.raises(NotImplementedError, match="cannot store additional_data"):
            spl.save(str(path), format="binary")


class TestAdditionalDataSliderTT:
    """additional_data on ChebyshevSlider and ChebyshevTT."""

    def test_slider_default_is_none(self):
        sld = ChebyshevSlider(
            _f3d, 3, [(-1, 1)] * 3, [7, 7, 7],
            partition=[[0], [1, 2]], pivot_point=[0.0, 0.0, 0.0],
        )
        assert sld.additional_data is None

    def test_slider_threaded_into_pivot_and_slides(self):
        captured = []

        def f_records(point, data):
            captured.append(data)
            return point[0] + point[1] + point[2]

        payload = {"strike": 100.0}
        sld = ChebyshevSlider(
            f_records, 3, [(-1, 1)] * 3, [5, 5, 5],
            partition=[[0], [1, 2]], pivot_point=[0.0, 0.0, 0.0],
            additional_data=payload,
        )
        sld.build(verbose=False)
        assert len(captured) > 0
        assert all(d is payload for d in captured)

    def test_slider_pickle_round_trip(self, tmp_path):
        payload = {"strike": 100.0}
        sld = ChebyshevSlider(
            _f3d, 3, [(-1, 1)] * 3, [7, 7, 7],
            partition=[[0], [1, 2]], pivot_point=[0.0, 0.0, 0.0],
            additional_data=payload,
        )
        sld.build(verbose=False)
        path = tmp_path / "sld.pkl"
        sld.save(str(path))
        restored = ChebyshevSlider.load(str(path))
        assert restored.additional_data == payload

    def test_tt_default_is_none(self):
        tt = ChebyshevTT(_f3d, 3, [(-1, 1)] * 3, [7, 7, 7],
                         tolerance=1e-6, max_rank=8)
        assert tt.additional_data is None

    def test_tt_threaded_into_function(self):
        captured = []

        def f_records(point, data):
            captured.append(data)
            return point[0] + point[1] + point[2]

        payload = {"strike": 100.0}
        tt = ChebyshevTT(f_records, 3, [(-1, 1)] * 3, [7, 7, 7],
                         tolerance=1e-6, max_rank=8,
                         additional_data=payload)
        tt.build(verbose=False)
        assert len(captured) > 0
        assert all(d is payload for d in captured)

    def test_tt_pickle_round_trip(self, tmp_path):
        payload = {"strike": 100.0}
        tt = ChebyshevTT(_f3d, 3, [(-1, 1)] * 3, [7, 7, 7],
                         tolerance=1e-6, max_rank=8,
                         additional_data=payload)
        tt.build(verbose=False)
        path = tmp_path / "tt.pkl"
        tt.save(str(path))
        restored = ChebyshevTT.load(str(path))
        assert restored.additional_data == payload


class TestDerivativeIdApprox:
    """get_derivative_id() registry + eval(derivative_id=...) on ChebyshevApproximation."""

    def test_first_call_returns_zero(self):
        cheb = _build_approx_3d()
        assert cheb.get_derivative_id([0, 0, 0]) == 0

    def test_sequential_ids(self):
        cheb = _build_approx_3d()
        a = cheb.get_derivative_id([0, 0, 0])
        b = cheb.get_derivative_id([1, 0, 0])
        c = cheb.get_derivative_id([0, 1, 0])
        assert (a, b, c) == (0, 1, 2)

    def test_same_orders_returns_same_id(self):
        cheb = _build_approx_3d()
        first = cheb.get_derivative_id([1, 0, 0])
        second = cheb.get_derivative_id([1, 0, 0])
        assert first == second

    def test_eval_via_id_matches_eval_via_orders(self):
        cheb = _build_approx_3d()
        orders = [1, 0, 0]
        d_id = cheb.get_derivative_id(orders)
        point = [0.3, -0.2, 0.5]
        a = cheb.eval(point, derivative_order=orders)
        b = cheb.eval(point, derivative_id=d_id)
        assert a == b

    def test_eval_both_kwargs_raises(self):
        cheb = _build_approx_3d()
        d_id = cheb.get_derivative_id([0, 0, 0])
        with pytest.raises(ValueError, match="not both"):
            cheb.eval([0.0, 0.0, 0.0], derivative_order=[0, 0, 0], derivative_id=d_id)

    def test_eval_neither_kwarg_raises(self):
        cheb = _build_approx_3d()
        with pytest.raises(ValueError, match="must provide"):
            cheb.eval([0.0, 0.0, 0.0])

    def test_eval_unknown_id_raises(self):
        cheb = _build_approx_3d()
        with pytest.raises(KeyError, match="unknown derivative_id"):
            cheb.eval([0.0, 0.0, 0.0], derivative_id=999)

    def test_pickle_preserves_registry(self, tmp_path):
        cheb = _build_approx_3d()
        d_id = cheb.get_derivative_id([1, 0, 0])
        path = tmp_path / "cheb.pkl"
        cheb.save(str(path))
        restored = ChebyshevApproximation.load(str(path))
        assert restored.get_derivative_id([1, 0, 0]) == d_id
        a = restored.eval([0.1, 0.2, 0.3], derivative_id=d_id)
        b = restored.eval([0.1, 0.2, 0.3], derivative_order=[1, 0, 0])
        assert a == b


class TestDerivativeIdSpline:
    """derivative_id registry on ChebyshevSpline."""

    def test_basic_registry(self):
        spl = _build_spline_2d()
        a = spl.get_derivative_id([0, 0])
        b = spl.get_derivative_id([1, 0])
        assert (a, b) == (0, 1)
        assert spl.get_derivative_id([0, 0]) == 0  # stable

    def test_eval_via_id_matches_eval_via_orders(self):
        spl = _build_spline_2d()
        orders = [1, 0]
        d_id = spl.get_derivative_id(orders)
        point = [0.3, -0.2]
        a = spl.eval(point, derivative_order=orders)
        b = spl.eval(point, derivative_id=d_id)
        assert a == b

    def test_eval_unknown_id_raises(self):
        spl = _build_spline_2d()
        with pytest.raises(KeyError, match="unknown derivative_id"):
            spl.eval([0.1, 0.2], derivative_id=999)

    def test_eval_both_kwargs_raises(self):
        spl = _build_spline_2d()
        d_id = spl.get_derivative_id([0, 0])
        with pytest.raises(ValueError, match="not both"):
            spl.eval([0.1, 0.2], derivative_order=[0, 0], derivative_id=d_id)

    def test_pickle_preserves_registry(self, tmp_path):
        spl = _build_spline_2d()
        d_id = spl.get_derivative_id([1, 0])
        path = tmp_path / "spl.pkl"
        spl.save(str(path))
        restored = ChebyshevSpline.load(str(path))
        # Same orders should still map to same id
        assert restored.get_derivative_id([1, 0]) == d_id
        # Eval via id matches eval via orders
        a = restored.eval([0.1, 0.2], derivative_id=d_id)
        b = restored.eval([0.1, 0.2], derivative_order=[1, 0])
        assert a == b


class TestDerivativeIdSlider:
    """derivative_id registry on ChebyshevSlider."""

    def test_basic_registry(self):
        sld = _build_slider_3d()
        a = sld.get_derivative_id([0, 0, 0])
        b = sld.get_derivative_id([1, 0, 0])
        assert (a, b) == (0, 1)
        assert sld.get_derivative_id([0, 0, 0]) == 0

    def test_eval_via_id_matches_eval_via_orders(self):
        sld = _build_slider_3d()
        orders = [1, 0, 0]
        d_id = sld.get_derivative_id(orders)
        point = [0.3, -0.2, 0.5]
        a = sld.eval(point, derivative_order=orders)
        b = sld.eval(point, derivative_id=d_id)
        assert a == b

    def test_eval_unknown_id_raises(self):
        sld = _build_slider_3d()
        with pytest.raises(KeyError, match="unknown derivative_id"):
            sld.eval([0.1, 0.2, 0.3], derivative_id=999)

    def test_eval_both_kwargs_raises(self):
        sld = _build_slider_3d()
        d_id = sld.get_derivative_id([0, 0, 0])
        with pytest.raises(ValueError, match="not both"):
            sld.eval([0.1, 0.2, 0.3], derivative_order=[0, 0, 0], derivative_id=d_id)

    def test_pickle_preserves_registry(self, tmp_path):
        sld = _build_slider_3d()
        d_id = sld.get_derivative_id([1, 0, 0])
        path = tmp_path / "sld.pkl"
        sld.save(str(path))
        restored = ChebyshevSlider.load(str(path))
        assert restored.get_derivative_id([1, 0, 0]) == d_id
        a = restored.eval([0.1, 0.2, 0.3], derivative_id=d_id)
        b = restored.eval([0.1, 0.2, 0.3], derivative_order=[1, 0, 0])
        assert a == b


class TestFactoryPathResets:
    """Factory-path operations (extrude, slice, algebra) reset v0.15 metadata.

    Derived interpolants start fresh — they don't inherit descriptor, additional_data,
    or the derivative_id registry from the source. This is by design: derived objects
    are mathematically a new function (algebra) or new domain (extrude/slice), and
    inheriting metadata could mislead users about provenance.
    """

    def test_algebra_resets_descriptor(self):
        a = _build_approx_3d()
        a.set_descriptor("source")
        b = _build_approx_3d()
        result = a + b
        assert result.get_descriptor() == ""

    def test_algebra_resets_additional_data(self):
        a = _build_approx_3d()
        a.additional_data = {"key": "source"}
        b = _build_approx_3d()
        result = a + b
        assert result.additional_data is None

    def test_algebra_resets_derivative_id_registry(self):
        a = _build_approx_3d()
        a.get_derivative_id([1, 0, 0])  # register an id
        a.get_derivative_id([0, 1, 0])
        b = _build_approx_3d()
        result = a + b
        # Result starts fresh — first call should return 0, not inherit
        assert result.get_derivative_id([1, 0, 0]) == 0

    def test_extrude_resets_metadata_approx(self):
        cheb = _build_approx_3d()
        cheb.set_descriptor("source")
        cheb.additional_data = {"key": "source"}
        cheb.get_derivative_id([1, 0, 0])
        ext = cheb.extrude((3, (-1.0, 1.0), 5))
        assert ext.get_descriptor() == ""
        assert ext.additional_data is None
        assert ext.get_derivative_id([1, 0, 0, 0]) == 0  # fresh registry

    def test_slice_resets_metadata_approx(self):
        cheb = _build_approx_3d()
        cheb.set_descriptor("source")
        cheb.additional_data = {"key": "source"}
        cheb.get_derivative_id([1, 0, 0])
        sl = cheb.slice((0, 0.3))
        assert sl.get_descriptor() == ""
        assert sl.additional_data is None
        assert sl.get_derivative_id([1, 0]) == 0  # fresh registry, lower dim


class TestBackwardCompatPickle:
    """v0.15 attributes are backfilled on __setstate__ for pre-v0.15 pickles."""

    def test_approx_setstate_backfills_v0_15_fields(self):
        """A pre-v0.15 ChebyshevApproximation pickle (missing v0.15 fields) loads cleanly."""
        cheb = _build_approx_3d()
        # Simulate a pre-v0.15 pickle: pop the v0.15 fields from the state dict
        state = cheb.__getstate__() if hasattr(cheb, "__getstate__") else cheb.__dict__.copy()
        for field in ("descriptor", "additional_data",
                      "_derivative_id_registry", "_derivative_id_to_orders"):
            state.pop(field, None)
        # Restore via __setstate__ on a fresh instance
        restored = ChebyshevApproximation.__new__(ChebyshevApproximation)
        restored.__setstate__(state)
        assert restored.get_descriptor() == ""
        assert restored.additional_data is None
        # Registry starts fresh
        assert restored.get_derivative_id([1, 0, 0]) == 0

    def test_spline_setstate_backfills_v0_15_fields(self):
        """A pre-v0.15 ChebyshevSpline pickle (missing v0.15 fields) loads cleanly."""
        spl = _build_spline_2d()
        state = spl.__getstate__() if hasattr(spl, "__getstate__") else spl.__dict__.copy()
        for field in ("descriptor", "additional_data",
                      "_derivative_id_registry", "_derivative_id_to_orders"):
            state.pop(field, None)
        restored = ChebyshevSpline.__new__(ChebyshevSpline)
        restored.__setstate__(state)
        assert restored.get_descriptor() == ""
        assert restored.additional_data is None
        assert restored.get_derivative_id([1, 0]) == 0

    def test_slider_setstate_backfills_v0_15_fields(self):
        """A pre-v0.15 ChebyshevSlider pickle loads cleanly (already covered by impl, regression guard)."""
        sld = _build_slider_3d()
        state = sld.__getstate__() if hasattr(sld, "__getstate__") else sld.__dict__.copy()
        for field in ("descriptor", "additional_data",
                      "_derivative_id_registry", "_derivative_id_to_orders"):
            state.pop(field, None)
        restored = ChebyshevSlider.__new__(ChebyshevSlider)
        restored.__setstate__(state)
        assert restored.get_descriptor() == ""
        assert restored.additional_data is None
        assert restored.get_derivative_id([1, 0, 0]) == 0

    def test_tt_setstate_backfills_v0_15_fields(self):
        """A pre-v0.15 ChebyshevTT pickle loads cleanly (already covered by impl, regression guard)."""
        tt = _build_tt_3d()
        state = tt.__getstate__() if hasattr(tt, "__getstate__") else tt.__dict__.copy()
        for field in ("descriptor", "additional_data"):
            state.pop(field, None)
        restored = ChebyshevTT.__new__(ChebyshevTT)
        restored.__setstate__(state)
        assert restored.get_descriptor() == ""
        assert restored.additional_data is None


class TestDerivativeIdValidation:
    """Coverage for get_derivative_id validation paths and the eval xor guard.

    These paths are byte-identical across ChebyshevApproximation, ChebyshevSpline,
    and ChebyshevSlider, so the test parametrizes over the three builders.
    """

    @pytest.mark.parametrize("builder,wrong_length", [
        (_build_approx_3d, [1, 0]),       # 3-D needs len 3, this is len 2
        (_build_spline_2d, [1]),          # 2-D needs len 2, this is len 1
        (_build_slider_3d, [1, 0]),       # 3-D needs len 3, this is len 2
    ])
    def test_get_derivative_id_wrong_length_raises(self, builder, wrong_length):
        obj = builder()
        with pytest.raises(ValueError, match="does not match num_dimensions"):
            obj.get_derivative_id(wrong_length)

    @pytest.mark.parametrize("builder,bad_orders", [
        (_build_approx_3d, [1.0, 0, 0]),
        (_build_spline_2d, ["1", 0]),
        (_build_slider_3d, [1, 0, None]),
    ])
    def test_get_derivative_id_non_int_raises(self, builder, bad_orders):
        obj = builder()
        with pytest.raises(ValueError, match="must be int"):
            obj.get_derivative_id(bad_orders)

    @pytest.mark.parametrize("builder,out_of_range", [
        (_build_approx_3d, [-1, 0, 0]),
        (_build_approx_3d, [99, 0, 0]),     # exceeds max_derivative_order=2
        (_build_spline_2d, [-5, 0]),
        (_build_spline_2d, [0, 99]),
        (_build_slider_3d, [-1, 0, 0]),
        (_build_slider_3d, [99, 0, 0]),
    ])
    def test_get_derivative_id_out_of_range_raises(self, builder, out_of_range):
        obj = builder()
        with pytest.raises(ValueError, match="out of range"):
            obj.get_derivative_id(out_of_range)

    @pytest.mark.parametrize("builder,point", [
        (_build_spline_2d, [0.1, 0.2]),
        (_build_slider_3d, [0.1, 0.2, 0.3]),
    ])
    def test_eval_neither_kwarg_raises_on_spline_and_slider(self, builder, point):
        """Approximation already has this test; this covers Spline and Slider."""
        obj = builder()
        with pytest.raises(ValueError, match="must provide"):
            obj.eval(point)

    @pytest.mark.parametrize("builder,point", [
        (_build_approx_3d, [0.1, 0.2, 0.3]),
        (_build_spline_2d, [0.1, 0.2]),
        (_build_slider_3d, [0.1, 0.2, 0.3]),
    ])
    def test_eval_negative_derivative_id_raises(self, builder, point):
        """Negative IDs should hit the upper-bound check and raise KeyError."""
        obj = builder()
        with pytest.raises(KeyError, match="unknown derivative_id"):
            obj.eval(point, derivative_id=-1)
