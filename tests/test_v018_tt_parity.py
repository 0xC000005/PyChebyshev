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


# ============================================================================
# T2: TT from_values() classmethod
# ============================================================================

class TestTTFromValues:
    def test_round_trip_via_explicit_tensor(self):
        n = 8
        nodes_x = ChebyshevTT.nodes(2, [[-1, 1], [-1, 1]], [n, n])["nodes_per_dim"][0]
        nodes_y = nodes_x.copy()
        X, Y = np.meshgrid(nodes_x, nodes_y, indexing="ij")
        dense = np.sin(X) * np.cos(Y)

        tt_from = ChebyshevTT.from_values(dense, 2, [[-1, 1], [-1, 1]], [n, n])
        assert isinstance(tt_from, ChebyshevTT)
        assert tt_from.num_dimensions == 2
        # Eval at a node should match dense entry
        assert tt_from.eval([float(nodes_x[2]), float(nodes_y[3])]) == pytest.approx(
            dense[2, 3], abs=1e-10
        )

    def test_constant_function_recovers_to_machine_precision(self):
        dense = np.full((5, 5), 7.0)
        tt = ChebyshevTT.from_values(dense, 2, [[0, 1], [0, 1]], [5, 5])
        assert tt.eval([0.3, 0.4]) == pytest.approx(7.0, abs=1e-10)

    def test_eval_after_from_values(self):
        # Build a known TT via dense tensor
        n = 8
        nodes_x = ChebyshevTT.nodes(2, [[-1, 1], [-1, 1]], [n, n])["nodes_per_dim"][0]
        nodes_y = nodes_x.copy()
        X, Y = np.meshgrid(nodes_x, nodes_y, indexing="ij")
        dense = np.sin(X) + np.cos(Y)

        tt = ChebyshevTT.from_values(dense, 2, [[-1, 1], [-1, 1]], [n, n])
        # Check eval at a node matches the dense tensor entry
        assert tt.eval([float(nodes_x[2]), float(nodes_y[3])]) == pytest.approx(
            dense[2, 3], abs=1e-10
        )

    def test_validates_tensor_shape(self):
        bad = np.zeros((4, 5))
        with pytest.raises(ValueError, match="shape"):
            ChebyshevTT.from_values(bad, 2, [[-1, 1], [-1, 1]], [5, 5])

    def test_validates_nan_inf(self):
        bad = np.zeros((5, 5))
        bad[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN|Inf|finite"):
            ChebyshevTT.from_values(bad, 2, [[-1, 1], [-1, 1]], [5, 5])

    def test_max_rank_caps_rank(self):
        rng = np.random.default_rng(42)
        dense = rng.standard_normal((6, 6, 6))
        tt = ChebyshevTT.from_values(
            dense, 3, [[-1, 1]] * 3, [6, 6, 6], max_rank=3
        )
        assert all(r <= 3 for r in tt.tt_ranks)

    def test_descriptor_default_empty(self):
        dense = np.zeros((4, 4))
        tt = ChebyshevTT.from_values(dense, 2, [[-1, 1], [-1, 1]], [4, 4])
        assert tt.get_descriptor() == ""

    def test_additional_data_kwarg_threaded(self):
        sentinel = {"x": 1}
        dense = np.zeros((4, 4))
        tt = ChebyshevTT.from_values(
            dense, 2, [[-1, 1], [-1, 1]], [4, 4], additional_data=sentinel
        )
        assert tt.additional_data == sentinel

    def test_function_is_none_after_from_values(self):
        dense = np.zeros((4, 4))
        tt = ChebyshevTT.from_values(dense, 2, [[-1, 1], [-1, 1]], [4, 4])
        assert tt.function is None


# ============================================================================
# T3: TT to_dense() instance method
# ============================================================================

class TestTTToDense:
    def test_returns_ndarray(self):
        def f(x, _):
            return x[0] * x[1]

        tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [4, 4])
        tt.build(verbose=False)
        dense = tt.to_dense()
        assert isinstance(dense, np.ndarray)

    def test_to_dense_shape_matches_n_nodes(self):
        def f(x, _):
            return x[0] + x[1] + x[2]

        tt = ChebyshevTT(f, 3, [[-1, 1]] * 3, [4, 5, 6])
        tt.build(verbose=False)
        dense = tt.to_dense()
        assert dense.shape == (4, 5, 6)

    def test_to_dense_values_match_eval_at_nodes(self):
        def f(x, _):
            return math.sin(x[0]) + math.cos(x[1])

        n = 5
        tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [n, n])
        tt.build(verbose=False)
        dense = tt.to_dense()
        nodes_x = ChebyshevTT.nodes(2, [[-1, 1], [-1, 1]], [n, n])["nodes_per_dim"][0]
        nodes_y = nodes_x.copy()
        for i in range(n):
            for j in range(n):
                expected = tt.eval([float(nodes_x[i]), float(nodes_y[j])])
                np.testing.assert_allclose(dense[i, j], expected, atol=1e-10)

    def test_to_dense_round_trip_via_from_values(self):
        def f(x, _):
            return x[0] * math.sin(x[1])

        tt_a = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [8, 8])
        tt_a.build(verbose=False)
        tt_b = ChebyshevTT.from_values(
            tt_a.to_dense(), 2, [[-1, 1], [-1, 1]], [8, 8]
        )
        # Eval should match to TT-SVD truncation precision
        x_test = [0.3, -0.4]
        assert tt_a.eval(x_test) == pytest.approx(tt_b.eval(x_test), abs=1e-8)

    def test_to_dense_constant_function(self):
        def f(x, _):
            return 3.0

        tt = ChebyshevTT(f, 2, [[0, 1], [0, 1]], [4, 4])
        tt.build(verbose=False)
        dense = tt.to_dense()
        np.testing.assert_allclose(dense, 3.0, atol=1e-10)
