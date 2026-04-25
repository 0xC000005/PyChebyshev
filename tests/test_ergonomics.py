"""Tests for v0.15 ergonomics bundle.

Currently covers: descriptor.
Will be extended for: additional_data, derivative_id, introspection trio.
"""

from __future__ import annotations

import math

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
