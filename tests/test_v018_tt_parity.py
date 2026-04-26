"""Tests for v0.18 TT Feature Parity."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pychebyshev import (
    ChebyshevApproximation,
    ChebyshevTT,
    Domain,
    Ns,
)


# ============================================================================
# T1: TT nodes() static method
# ============================================================================

class TestTTNodes:
    def test_static_method_callable_without_instance(self):
        result = ChebyshevTT.nodes(2, [[-1, 1], [-1, 1]], [4, 4])
        assert isinstance(result, dict)

    def test_returns_dict_with_nodes_per_dim(self):
        result = ChebyshevTT.nodes(2, [[-1, 1], [-1, 1]], [4, 5])
        assert "nodes_per_dim" in result
        assert len(result["nodes_per_dim"]) == 2
        assert len(result["nodes_per_dim"][0]) == 4
        assert len(result["nodes_per_dim"][1]) == 5

    def test_node_count_matches_input(self):
        result = ChebyshevTT.nodes(3, [[-1, 1]] * 3, [3, 5, 7])
        assert [len(n) for n in result["nodes_per_dim"]] == [3, 5, 7]

    def test_nodes_within_domain(self):
        result = ChebyshevTT.nodes(2, [[0, 2], [-3, 3]], [6, 6])
        for d, (lo, hi) in enumerate([[0, 2], [-3, 3]]):
            nodes = result["nodes_per_dim"][d]
            assert nodes.min() >= lo - 1e-12
            assert nodes.max() <= hi + 1e-12

    def test_consistency_with_approximation_nodes(self):
        tt_result = ChebyshevTT.nodes(2, [[-1, 1], [-1, 1]], [5, 5])
        cheb_result = ChebyshevApproximation.nodes(2, [[-1, 1], [-1, 1]], [5, 5])
        for d in range(2):
            np.testing.assert_array_equal(
                tt_result["nodes_per_dim"][d], cheb_result["nodes_per_dim"][d]
            )
