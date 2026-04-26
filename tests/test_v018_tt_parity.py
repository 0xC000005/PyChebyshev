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


# ============================================================================
# T4: TT extrude(params)
# ============================================================================

class TestTTExtrude:
    def test_single_dim_extrude_returns_tt(self):
        def f(x, _):
            return x[0] ** 2

        tt = ChebyshevTT(f, 1, [[-1, 1]], [5])
        tt.build(verbose=False)
        result = tt.extrude((1, (0.0, 1.0), 4))
        assert isinstance(result, ChebyshevTT)
        assert result.num_dimensions == 2

    def test_extrude_preserves_eval_at_existing_dims(self):
        def f(x, _):
            return math.sin(x[0])

        tt = ChebyshevTT(f, 1, [[-1, 1]], [10])
        tt.build(verbose=False)
        result = tt.extrude((1, (0.0, 1.0), 5))
        # f(x) was sin(x); extruded to dim 1 with constant 1: result(x, y) = sin(x)
        for x_test in [-0.5, 0.0, 0.3]:
            for y_test in [0.1, 0.5, 0.9]:
                assert result.eval([x_test, y_test]) == pytest.approx(
                    math.sin(x_test), abs=1e-6
                )

    def test_extrude_constant_value_in_new_dim(self):
        def f(x, _):
            return 7.0

        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        result = tt.extrude((1, (0.0, 5.0), 4))
        # Result should still be 7.0 everywhere
        assert result.eval([0.5, 2.5]) == pytest.approx(7.0, abs=1e-10)

    def test_extrude_validates_dim_idx_in_range(self):
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        with pytest.raises(ValueError):
            tt.extrude((5, (0, 1), 4))  # dim_idx 5 is way out of range

    def test_extrude_descriptor_preserved(self):
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        tt.set_descriptor("source")
        result = tt.extrude((1, (0, 1), 4))
        assert result.get_descriptor() == "source"

    def test_extrude_then_integrate_consistency(self):
        """Extrude over [0, 1] (vol=1) then integrate over the new dim → original × 1 = original."""
        def f(x, _):
            return math.sin(x[0])

        tt = ChebyshevTT(f, 1, [[-1, 1]], [10])
        tt.build(verbose=False)
        extruded = tt.extrude((1, (0.0, 1.0), 5))
        integrated = extruded.integrate(dims=[1])  # integrate over new dim
        # ∫_0^1 sin(x) dy = sin(x) * 1 = sin(x), so integrated(x) ≈ sin(x)
        assert integrated.eval([0.3]) == pytest.approx(
            math.sin(0.3), abs=1e-6
        )


# ============================================================================
# T5: TT slice(params)
# ============================================================================

class TestTTSlice:
    def test_single_dim_slice_returns_tt(self):
        def f(x, _):
            return x[0] + x[1]

        tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [6, 6])
        tt.build(verbose=False)
        result = tt.slice((0, 0.5))
        assert isinstance(result, ChebyshevTT)
        assert result.num_dimensions == 1

    def test_slice_at_node_uses_fast_path(self):
        """Slicing at a Chebyshev node should give exact result."""
        def f(x, _):
            return math.sin(x[0]) * math.cos(x[1])

        n = 6
        tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [n, n])
        tt.build(verbose=False)
        nodes_x = ChebyshevTT.nodes(2, [[-1, 1], [-1, 1]], [n, n])["nodes_per_dim"][0]
        # Slice dim 0 at the third node
        result = tt.slice((0, float(nodes_x[2])))
        # Compare against tt.eval([nodes_x[2], y]) for various y
        for y in [-0.5, 0.0, 0.5]:
            assert result.eval([y]) == pytest.approx(
                tt.eval([float(nodes_x[2]), y]), abs=1e-10
            )

    def test_slice_at_interior_value_matches_eval(self):
        def f(x, _):
            return math.sin(x[0]) + math.cos(x[1])

        tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [10, 10])
        tt.build(verbose=False)
        result = tt.slice((1, 0.3))
        # f(x, 0.3) = sin(x) + cos(0.3)
        for x_test in [-0.5, 0.0, 0.4]:
            assert result.eval([x_test]) == pytest.approx(
                math.sin(x_test) + math.cos(0.3), abs=1e-6
            )

    def test_slice_endpoint_dim_left(self):
        """Slice dim 0 (no left neighbor)."""
        def f(x, _):
            return x[0] * x[1]

        tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [4, 6])
        tt.build(verbose=False)
        result = tt.slice((0, 0.5))
        # f(0.5, y) = 0.5 * y
        assert result.eval([0.6]) == pytest.approx(0.3, abs=1e-8)

    def test_slice_endpoint_dim_right(self):
        """Slice the last dim (no right neighbor)."""
        def f(x, _):
            return x[0] * x[1]

        tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [6, 4])
        tt.build(verbose=False)
        result = tt.slice((1, 0.5))
        # f(x, 0.5) = 0.5 * x
        assert result.eval([0.3]) == pytest.approx(0.15, abs=1e-8)

    def test_slice_validates_value_within_domain(self):
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        with pytest.raises(ValueError):
            tt.slice((0, 5.0))  # outside [-1, 1]

    def test_slice_descriptor_preserved(self):
        def f(x, _):
            return x[0] + x[1]

        tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [4, 4])
        tt.build(verbose=False)
        tt.set_descriptor("source")
        result = tt.slice((0, 0.0))
        assert result.get_descriptor() == "source"


# ============================================================================
# T6: TT addition (__add__, __sub__, __neg__)
# ============================================================================

class TestTTAddition:
    def test_add_two_tts_returns_tt(self):
        def f(x, _):
            return x[0] + x[1]

        def g(x, _):
            return math.sin(x[0]) + math.cos(x[1])

        tt_f = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [10, 10])
        tt_f.build(verbose=False)
        tt_g = ChebyshevTT(g, 2, [[-1, 1], [-1, 1]], [10, 10])
        tt_g.build(verbose=False)
        result = tt_f + tt_g
        assert isinstance(result, ChebyshevTT)

    def test_add_eval_matches_sum_of_evals(self):
        def f(x, _):
            return x[0] + x[1]

        def g(x, _):
            return math.sin(x[0])

        tt_f = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [10, 10])
        tt_f.build(verbose=False)
        tt_g = ChebyshevTT(g, 2, [[-1, 1], [-1, 1]], [10, 10])
        tt_g.build(verbose=False)
        result = tt_f + tt_g
        for x_test in [[0.3, 0.4], [-0.2, 0.5], [0.0, 0.0]]:
            assert result.eval(x_test) == pytest.approx(
                tt_f.eval(x_test) + tt_g.eval(x_test), abs=1e-6
            )

    def test_add_incompatible_domain_raises(self):
        def f(x, _):
            return x[0]

        tt_f = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt_f.build(verbose=False)
        tt_g = ChebyshevTT(f, 1, [[0, 2]], [4])
        tt_g.build(verbose=False)
        with pytest.raises(ValueError):
            tt_f + tt_g

    def test_add_incompatible_n_nodes_raises(self):
        def f(x, _):
            return x[0]

        tt_f = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt_f.build(verbose=False)
        tt_g = ChebyshevTT(f, 1, [[-1, 1]], [6])
        tt_g.build(verbose=False)
        with pytest.raises(ValueError):
            tt_f + tt_g

    def test_subtract_returns_tt(self):
        def f(x, _):
            return x[0] + x[1]

        tt_a = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [6, 6])
        tt_a.build(verbose=False)
        tt_b = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [6, 6])
        tt_b.build(verbose=False)
        result = tt_a - tt_b
        # tt_a - tt_b should be approx 0 everywhere
        assert result.eval([0.3, 0.4]) == pytest.approx(0.0, abs=1e-6)

    def test_negation(self):
        def f(x, _):
            return math.sin(x[0])

        tt = ChebyshevTT(f, 1, [[-1, 1]], [10])
        tt.build(verbose=False)
        neg = -tt
        assert neg.eval([0.5]) == pytest.approx(-tt.eval([0.5]), abs=1e-10)

    def test_add_function_is_none_on_result(self):
        def f(x, _):
            return x[0]

        tt_a = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt_a.build(verbose=False)
        tt_b = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt_b.build(verbose=False)
        result = tt_a + tt_b
        assert result.function is None

    def test_chained_adds_respect_max_rank(self):
        """Sum of multiple TTs should be rounded to max_rank to prevent rank explosion."""
        def make_tt(coef):
            def f(x, _):
                return coef * (x[0] + x[1])

            tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [6, 6], max_rank=4)
            tt.build(verbose=False)
            return tt

        # Three TTs added together
        result = make_tt(1.0) + make_tt(2.0) + make_tt(3.0)
        assert all(r <= 4 for r in result.tt_ranks), \
            f"max_rank=4 violated: ranks={result.tt_ranks}"


# ============================================================================
# T7: TT scalar __mul__ + in-place operators
# ============================================================================

class TestTTScalarMul:
    def test_scalar_mul_returns_tt(self):
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        result = tt * 2.5
        assert isinstance(result, ChebyshevTT)

    def test_scalar_mul_eval_scales(self):
        def f(x, _):
            return math.sin(x[0])

        tt = ChebyshevTT(f, 1, [[-1, 1]], [10])
        tt.build(verbose=False)
        result = tt * 3.0
        for x_test in [-0.5, 0.0, 0.5]:
            assert result.eval([x_test]) == pytest.approx(
                3.0 * tt.eval([x_test]), abs=1e-10
            )

    def test_rmul_works(self):
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        result = 2.5 * tt
        assert isinstance(result, ChebyshevTT)

    def test_truediv_scalar(self):
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        result = tt / 2.0
        assert result.eval([0.4]) == pytest.approx(0.2, abs=1e-10)

    def test_iadd_in_place(self):
        def f(x, _):
            return x[0]

        tt_a = ChebyshevTT(f, 1, [[-1, 1]], [6])
        tt_a.build(verbose=False)
        tt_b = ChebyshevTT(f, 1, [[-1, 1]], [6])
        tt_b.build(verbose=False)
        tt_a += tt_b
        # __iadd__ may return new object; just verify eval is sum
        assert tt_a.eval([0.5]) == pytest.approx(1.0, abs=1e-6)

    def test_imul_scalar_in_place(self):
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        tt *= 2.0
        assert tt.eval([0.3]) == pytest.approx(0.6, abs=1e-10)


# ============================================================================
# T8: Cross-feature tests (algebra + integrate, extrude/slice + v0.16/v0.17)
# ============================================================================

class TestTTCrossFeatures:
    def test_algebra_then_integrate(self):
        """(tt_a + tt_b).integrate() == tt_a.integrate() + tt_b.integrate()."""
        def f(x, _):
            return math.sin(x[0])

        def g(x, _):
            return math.cos(x[0])

        tt_a = ChebyshevTT(f, 1, [[-1, 1]], [10])
        tt_a.build(verbose=False)
        tt_b = ChebyshevTT(g, 1, [[-1, 1]], [10])
        tt_b.build(verbose=False)
        sum_int = (tt_a + tt_b).integrate()
        a_int = tt_a.integrate()
        b_int = tt_b.integrate()
        assert sum_int == pytest.approx(a_int + b_int, abs=1e-6)

    def test_clone_after_algebra(self):
        """v0.16 clone() works on algebra results."""
        def f(x, _):
            return x[0]

        tt_a = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt_a.build(verbose=False)
        tt_b = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt_b.build(verbose=False)
        result = tt_a + tt_b
        clone = result.clone()
        assert clone is not result
        assert clone.eval([0.3]) == pytest.approx(result.eval([0.3]), abs=1e-10)

    def test_extrude_then_get_evaluation_points(self):
        """v0.16 get_evaluation_points() works on extruded TT."""
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        extruded = tt.extrude((1, (0, 1), 5))
        pts = extruded.get_evaluation_points()
        assert pts.shape == (4 * 5, 2)

    def test_slice_then_to_dense(self):
        def f(x, _):
            return x[0] * x[1]

        tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [5, 5])
        tt.build(verbose=False)
        sliced = tt.slice((0, 0.5))
        dense = sliced.to_dense()
        assert dense.shape == (5,)

    def test_from_values_then_eval(self):
        n = 5
        nodes = ChebyshevTT.nodes(2, [[-1, 1], [-1, 1]], [n, n])["nodes_per_dim"]
        X, Y = np.meshgrid(nodes[0], nodes[1], indexing="ij")
        dense = X ** 2 + Y ** 2

        tt = ChebyshevTT.from_values(dense, 2, [[-1, 1], [-1, 1]], [n, n])
        # Eval at a grid point should match dense entry
        assert tt.eval([float(nodes[0][2]), float(nodes[1][3])]) == pytest.approx(
            dense[2, 3], abs=1e-8
        )

    def test_serialization_round_trip_algebra(self, tmp_path):
        def f(x, _):
            return x[0]

        tt_a = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt_a.build(verbose=False)
        tt_b = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt_b.build(verbose=False)
        result = tt_a + tt_b
        path = tmp_path / "result.pkl"
        result.save(str(path))
        loaded = ChebyshevTT.load(str(path))
        assert loaded.eval([0.5]) == pytest.approx(result.eval([0.5]), abs=1e-10)
