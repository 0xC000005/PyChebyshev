"""Tests for v0.17 — integrate() on ChebyshevSlider and ChebyshevTT."""
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
from pychebyshev._calculus import _slider_partition_intersect


# ============================================================================
# T1: _slider_partition_intersect() helper
# ============================================================================

class TestSliderPartitionIntersect:
    def test_full_intersection_returns_full(self):
        # group [0, 1], integrating dims [0, 1] → "full"
        kind, kept = _slider_partition_intersect(group_dims=[0, 1], integrate_dims=[0, 1])
        assert kind == "full"
        assert kept == []

    def test_no_intersection_returns_none(self):
        # group [2], integrating dims [0, 1] → "none"
        kind, kept = _slider_partition_intersect(group_dims=[2], integrate_dims=[0, 1])
        assert kind == "none"
        assert kept == [2]

    def test_partial_intersection_returns_partial(self):
        # group [0, 1, 2], integrating dims [1] → "partial", kept [0, 2]
        kind, kept = _slider_partition_intersect(group_dims=[0, 1, 2], integrate_dims=[1])
        assert kind == "partial"
        assert kept == [0, 2]

    def test_empty_integrate_dims_returns_none(self):
        kind, kept = _slider_partition_intersect(group_dims=[0, 1], integrate_dims=[])
        assert kind == "none"
        assert kept == [0, 1]

    def test_subset_group_full(self):
        # integrate_dims is a superset; group fully contained → "full"
        kind, kept = _slider_partition_intersect(group_dims=[1], integrate_dims=[0, 1, 2])
        assert kind == "full"
        assert kept == []
