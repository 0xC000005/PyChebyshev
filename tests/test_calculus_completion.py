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


from pychebyshev._calculus import _integrate_tt_along_dim


# ============================================================================
# T2: _integrate_tt_along_dim() helper
# ============================================================================

class TestIntegrateTTAlongDim:
    def test_contract_single_core(self):
        """Contracting a (1, n, 1) core along its node axis with weights
        returns a (1, 1) matrix."""
        # Rank-1 core, n=4 nodes, values [1, 2, 3, 4]
        core = np.array([[[1.0], [2.0], [3.0], [4.0]]])  # shape (1, 4, 1)
        weights = np.array([0.25, 0.25, 0.25, 0.25])  # uniform
        result = _integrate_tt_along_dim(core, weights)
        # Expected: 1*0.25 + 2*0.25 + 3*0.25 + 4*0.25 = 2.5
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result, [[2.5]])

    def test_contract_higher_rank_core(self):
        """Contracting a (2, 3, 2) core preserves the rank dimensions."""
        rng = np.random.default_rng(42)
        core = rng.standard_normal((2, 3, 2))
        weights = np.array([0.5, 0.25, 0.25])
        result = _integrate_tt_along_dim(core, weights)
        assert result.shape == (2, 2)
        # Manual check
        expected = (
            core[:, 0, :] * 0.5 + core[:, 1, :] * 0.25 + core[:, 2, :] * 0.25
        )
        np.testing.assert_allclose(result, expected)

    def test_contract_n_nodes_one(self):
        """A core with n=1 (singleton dim) integrates to weights[0] * core[:,0,:]."""
        core = np.array([[[3.0, 4.0]]])  # shape (1, 1, 2)
        weights = np.array([2.0])
        result = _integrate_tt_along_dim(core, weights)
        np.testing.assert_allclose(result, [[6.0, 8.0]])
