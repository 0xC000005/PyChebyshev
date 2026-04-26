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


# ============================================================================
# T3: TT full integration (returns scalar)
# ============================================================================

class TestTTFullIntegrate:
    def test_separable_function(self):
        """f(x, y) = sin(x) * cos(y) over [-1, 1]^2.

        ∫∫ sin(x)cos(y) dx dy = [−cos(x)]_{-1}^{1} · [sin(y)]_{-1}^{1}
                              = (−cos(1) + cos(−1)) · (sin(1) − sin(−1))
                              = 0 · (2 sin(1)) = 0
        """
        def f(x, _):
            return math.sin(x[0]) * math.cos(x[1])

        tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [12, 12])
        tt.build(verbose=False)
        result = tt.integrate()  # full integration over all dims
        assert result == pytest.approx(0.0, abs=1e-8)

    def test_constant_function(self):
        """f(x) = 7 over [0, 2] × [0, 3] integrates to 7 × 6 = 42."""
        def f(x, _):
            return 7.0

        tt = ChebyshevTT(f, 2, [[0, 2], [0, 3]], [4, 4])
        tt.build(verbose=False)
        result = tt.integrate()
        assert result == pytest.approx(42.0, abs=1e-10)

    def test_against_scipy_nquad(self):
        """5-D separable function vs scipy.integrate.nquad."""
        from scipy.integrate import nquad

        def f_scalar(x, _):
            return math.exp(-(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2))

        domain = [[-1, 1]] * 5
        tt = ChebyshevTT(f_scalar, 5, domain, [10] * 5)
        tt.build(verbose=False)
        cheb_result = tt.integrate()

        def f_nquad(*args):
            return math.exp(-sum(a ** 2 for a in args))

        scipy_result, _ = nquad(f_nquad, domain)
        assert cheb_result == pytest.approx(scipy_result, rel=1e-4)


# ============================================================================
# T4: TT partial integration (returns ChebyshevTT)
# ============================================================================

class TestTTPartialIntegrate:
    def test_returns_tt_with_correct_dim(self):
        def f(x, _):
            return x[0] + x[1] + x[2]

        tt = ChebyshevTT(f, 3, [[-1, 1]] * 3, [6, 6, 6])
        tt.build(verbose=False)
        result = tt.integrate(dims=[1])
        assert isinstance(result, ChebyshevTT)
        assert result.num_dimensions == 2
        assert result.n_nodes == [6, 6]

    def test_partial_consistent_with_consecutive(self):
        """integrate([0, 1]) should equal integrate([1]).integrate([0])."""
        def f(x, _):
            return math.sin(x[0]) * math.cos(x[1]) * (1 + x[2] ** 2)

        tt1 = ChebyshevTT(f, 3, [[-1, 1]] * 3, [10, 10, 10])
        tt1.build(verbose=False)
        tt2 = ChebyshevTT(f, 3, [[-1, 1]] * 3, [10, 10, 10])
        tt2.build(verbose=False)

        # Two-shot
        joint = tt1.integrate(dims=[0, 1])
        # One-at-a-time (note dim re-indexing after first integrate)
        step1 = tt2.integrate(dims=[1])  # surviving dims = [0, 2]; new dim 1 was original dim 2
        step2 = step1.integrate(dims=[0])  # integrate over original dim 0

        # joint and step2 should both be functions of original dim 2
        x_test = 0.3
        np.testing.assert_allclose(
            joint.eval([x_test]),
            step2.eval([x_test]),
            atol=1e-8,
        )

    def test_endpoint_dim_left(self):
        """Integrate over dim 0 (no left neighbor)."""
        def f(x, _):
            return x[0] * x[1]

        tt = ChebyshevTT(f, 2, [[-1, 1]] * 2, [4, 6])
        tt.build(verbose=False)
        result = tt.integrate(dims=[0])
        assert isinstance(result, ChebyshevTT)
        # ∫_{-1}^{1} x dx = 0, so result(y) ≈ 0 for all y
        assert result.eval([0.5]) == pytest.approx(0.0, abs=1e-10)

    def test_endpoint_dim_right(self):
        """Integrate over last dim (no right neighbor)."""
        def f(x, _):
            return x[0] * x[1]

        tt = ChebyshevTT(f, 2, [[-1, 1]] * 2, [6, 4])
        tt.build(verbose=False)
        result = tt.integrate(dims=[1])
        # ∫_{-1}^{1} y dy = 0, so result(x) ≈ 0
        assert result.eval([0.5]) == pytest.approx(0.0, abs=1e-10)

    def test_descriptor_preserved(self):
        def f(x, _):
            return x[0] * x[1]

        tt = ChebyshevTT(f, 2, [[-1, 1]] * 2, [4, 4])
        tt.build(verbose=False)
        tt.set_descriptor("source")
        result = tt.integrate(dims=[0])
        assert result.get_descriptor() == "source"

    def test_additional_data_preserved(self):
        sentinel = {"k": 42}

        def f(x, ad):
            return ad["k"] * x[0]

        tt = ChebyshevTT(f, 2, [[-1, 1]] * 2, [4, 4], additional_data=sentinel)
        tt.build(verbose=False)
        result = tt.integrate(dims=[0])
        assert result.additional_data == sentinel


# ============================================================================
# T5: TT bounds + validation
# ============================================================================

class TestTTIntegrateBoundsAndValidation:
    def test_with_sub_interval_bounds(self):
        """∫_0^1 x dx = 0.5 over [-1, 1]."""
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        result = tt.integrate(dims=[0], bounds=[(0.0, 1.0)])
        assert result == pytest.approx(0.5, abs=1e-10)

    def test_validation_dims_oob(self):
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 2, [[-1, 1]] * 2, [4, 4])
        tt.build(verbose=False)
        with pytest.raises(ValueError, match="out-of-range"):
            tt.integrate(dims=[5])

    def test_validation_bounds_outside_domain(self):
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        with pytest.raises(ValueError):
            tt.integrate(dims=[0], bounds=[(-2.0, 2.0)])

    def test_validation_bounds_length_mismatch(self):
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 2, [[-1, 1]] * 2, [4, 4])
        tt.build(verbose=False)
        with pytest.raises(ValueError):
            tt.integrate(dims=[0], bounds=[(0, 1), (0, 1)])  # 2 bounds, 1 dim

    def test_dims_order_invariance(self):
        """integrate(dims=[0, 1]) == integrate(dims=[1, 0])."""
        def f(x, _):
            return math.sin(x[0]) + math.cos(x[1])

        tt = ChebyshevTT(f, 2, [[-1, 1]] * 2, [10, 10])
        tt.build(verbose=False)
        a = tt.integrate(dims=[0, 1])
        # Re-build to get an independent TT
        tt2 = ChebyshevTT(f, 2, [[-1, 1]] * 2, [10, 10])
        tt2.build(verbose=False)
        b = tt2.integrate(dims=[1, 0])
        assert a == pytest.approx(b, abs=1e-10)

    def test_works_after_method_svd(self):
        def f(x, _):
            return x[0] * x[1]

        tt = ChebyshevTT(f, 2, [[-1, 1]] * 2, [6, 6])
        tt.build(verbose=False, method="svd")
        result = tt.integrate()
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_works_after_method_als(self):
        def f(x, _):
            return x[0] * x[1] + math.sin(x[0])

        tt = ChebyshevTT(f, 2, [[-1, 1]] * 2, [8, 8])
        tt.build(verbose=False, method="als")
        result = tt.integrate()
        # ∫ x*y dx dy = 0; ∫ sin(x) dx dy = 2 * 0 = 0 over [-1,1]^2
        assert result == pytest.approx(0.0, abs=1e-8)

    def test_serialization_round_trip(self, tmp_path):
        def f(x, _):
            return x[0] * x[1]

        tt = ChebyshevTT(f, 2, [[-1, 1]] * 2, [6, 6])
        tt.build(verbose=False)
        result = tt.integrate(dims=[0])
        path = tmp_path / "result.pkl"
        result.save(str(path))
        loaded = ChebyshevTT.load(str(path))
        assert loaded.eval([0.5]) == pytest.approx(result.eval([0.5]), abs=1e-12)


# ============================================================================
# T6: Slider full integration (returns scalar)
# ============================================================================

class TestSliderFullIntegrate:
    def test_pivot_only_function(self):
        """f(x, y) = constant; integral = constant * vol(D)."""
        def f(x, _):
            return 5.0

        slider = ChebyshevSlider(
            f, 2, [[0, 2], [0, 3]], [4, 4], partition=[[0], [1]],
            pivot_point=[1.0, 1.5],
        )
        slider.build(verbose=False)
        result = slider.integrate()
        # f = 5 everywhere, so integrate = 5 * 2 * 3 = 30
        assert result == pytest.approx(30.0, abs=1e-10)

    def test_additive_function_sum_of_x(self):
        """f(x, y) = x + y over [-1, 1]^2 → ∫ = 0."""
        def f(x, _):
            return x[0] + x[1]

        slider = ChebyshevSlider(
            f, 2, [[-1, 1], [-1, 1]], [8, 8], partition=[[0], [1]],
            pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        result = slider.integrate()
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_separable_function_against_analytical(self):
        """f(x, y) = sin(x) + cos(y) over [-1, 1]^2.
        ∫∫ sin(x) dx dy = 0; ∫∫ cos(y) dx dy = 4 sin(1)."""
        def f(x, _):
            return math.sin(x[0]) + math.cos(x[1])

        slider = ChebyshevSlider(
            f, 2, [[-1, 1], [-1, 1]], [10, 10], partition=[[0], [1]],
            pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        result = slider.integrate()
        expected = 4.0 * math.sin(1.0)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_5d_against_nquad(self):
        """5-D additive function vs scipy nquad."""
        from scipy.integrate import nquad

        def f_scalar(x, _):
            return sum(math.sin(xi) for xi in x)

        domain = [[-1, 1]] * 5
        slider = ChebyshevSlider(
            f_scalar, 5, domain, [8] * 5,
            partition=[[i] for i in range(5)],
            pivot_point=[0.0] * 5,
        )
        slider.build(verbose=False)
        cheb_result = slider.integrate()

        def f_nquad(*args):
            return sum(math.sin(a) for a in args)

        scipy_result, _ = nquad(f_nquad, domain)
        assert cheb_result == pytest.approx(scipy_result, abs=1e-6)
