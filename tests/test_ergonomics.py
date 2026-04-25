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
