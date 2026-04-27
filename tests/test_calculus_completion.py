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


# ============================================================================
# T7: Slider partial integration (returns ChebyshevSlider)
# ============================================================================

class TestSliderPartialIntegrate:
    def test_returns_slider(self):
        def f(x, _):
            return math.sin(x[0]) + math.cos(x[1]) + x[2]

        slider = ChebyshevSlider(
            f, 3, [[-1, 1]] * 3, [8, 8, 8],
            partition=[[0], [1], [2]], pivot_point=[0.0] * 3,
        )
        slider.build(verbose=False)
        result = slider.integrate(dims=[1])
        assert isinstance(result, ChebyshevSlider)
        assert result.num_dimensions == 2

    def test_partial_disjoint_slide_passes_through(self):
        """A slide whose group is disjoint from integrate dims must pass through."""
        def f(x, _):
            return math.sin(x[0]) + x[1] ** 2

        slider = ChebyshevSlider(
            f, 2, [[-1, 1]] * 2, [8, 8],
            partition=[[0], [1]], pivot_point=[0.0] * 2,
        )
        slider.build(verbose=False)
        # Integrate over dim 1; slide 0 (group [0]) is disjoint
        result = slider.integrate(dims=[1])
        # Result should reflect ∫(sin(x) + y^2 - pv) dy + pv*vol(y) for slide 1
        # Specifically: at x=0, ∫_{-1}^{1} (sin(0) + y^2) dy = 0 + 2/3 = 2/3
        np.testing.assert_allclose(result.eval([0.0], [0]), 2.0 / 3.0, atol=1e-3)

    def test_full_partial_consistency(self):
        """integrate(dims=[0,1,2]) == integrate(dims=[0,1]).integrate(dims=[0])."""
        def f(x, _):
            return math.sin(x[0]) + math.cos(x[1]) + x[2] ** 2

        slider = ChebyshevSlider(
            f, 3, [[-1, 1]] * 3, [10, 10, 10],
            partition=[[0], [1], [2]], pivot_point=[0.0] * 3,
        )
        slider.build(verbose=False)
        joint = slider.integrate(dims=[0, 1, 2])
        # build a fresh one for chaining
        slider2 = ChebyshevSlider(
            f, 3, [[-1, 1]] * 3, [10, 10, 10],
            partition=[[0], [1], [2]], pivot_point=[0.0] * 3,
        )
        slider2.build(verbose=False)
        step1 = slider2.integrate(dims=[0, 1])  # Slider over original dim 2 only
        step2 = step1.integrate(dims=[0])  # scalar
        assert step2 == pytest.approx(joint, abs=1e-6)

    def test_descriptor_preserved(self):
        def f(x, _):
            return x[0] + x[1]

        slider = ChebyshevSlider(
            f, 2, [[-1, 1]] * 2, [4, 4],
            partition=[[0], [1]], pivot_point=[0.0] * 2,
        )
        slider.build(verbose=False)
        slider.set_descriptor("source")
        result = slider.integrate(dims=[0])
        assert result.get_descriptor() == "source"

    def test_partial_with_multi_dim_group(self):
        """Multi-dim slide group, partially integrated → exercises the 'partial' classification.

        partition=[[0, 1]] is one 2D slide; integrating dim 0 leaves a 1D slide over dim 1.
        """
        def f(x, _):
            return math.sin(x[0]) * math.cos(x[1])

        slider = ChebyshevSlider(
            f, 2, [[-1, 1], [-1, 1]], [10, 10],
            partition=[[0, 1]], pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        result = slider.integrate(dims=[0])
        assert isinstance(result, ChebyshevSlider)
        assert result.num_dimensions == 1

        # ∫_{-1}^{1} sin(x) cos(y) dx = cos(y) * [-cos(x)]_{-1}^{1} = 0
        # So the integrated function should be ≈ 0 for all y
        assert result.eval([0.5], [0]) == pytest.approx(0.0, abs=1e-6)

    def test_partial_with_3d_group_partial_integration(self):
        """3D slide group with one dim integrated → 2D reduced slide.

        partition=[[0, 1, 2]] is a 3D slide; integrate=[1] reduces it to 2D over (0, 2).
        """
        def f(x, _):
            return math.sin(x[0]) + math.cos(x[1]) + x[2] ** 2

        slider = ChebyshevSlider(
            f, 3, [[-1, 1]] * 3, [8, 8, 8],
            partition=[[0, 1, 2]], pivot_point=[0.0, 0.0, 0.0],
        )
        slider.build(verbose=False)
        result = slider.integrate(dims=[1])
        assert isinstance(result, ChebyshevSlider)
        assert result.num_dimensions == 2

        # ∫_{-1}^{1} (sin(x) + cos(y) + z^2) dy = 2*sin(x) + 2*sin(1) + 2*z^2
        # At x=0, z=0: 0 + 2*sin(1) + 0 ≈ 1.683
        expected = 2.0 * math.sin(1.0)
        assert result.eval([0.0, 0.0], [0, 0]) == pytest.approx(expected, abs=1e-3)

    def test_partial_mixed_classifications(self):
        """Two slides where one undergoes 'partial' and another 'none'.

        partition=[[0, 1], [2]]; integrate=[0] →
        - slide 0 (group [0,1]): "partial" — reduces to 1D over dim 1
        - slide 1 (group [2]): "none" — passes through unchanged
        """
        def f(x, _):
            return math.sin(x[0]) * math.cos(x[1]) + x[2]

        slider = ChebyshevSlider(
            f, 3, [[-1, 1]] * 3, [8, 8, 8],
            partition=[[0, 1], [2]], pivot_point=[0.0, 0.0, 0.0],
        )
        slider.build(verbose=False)
        result = slider.integrate(dims=[0])
        assert isinstance(result, ChebyshevSlider)
        assert result.num_dimensions == 2

        # ∫_{-1}^{1} (sin(x) cos(y) + z) dx = 0 + 2z = 2z
        # So result(y, z) ≈ 2z for any y
        assert result.eval([0.0, 0.5], [0, 0]) == pytest.approx(1.0, abs=1e-6)
        assert result.eval([0.5, 0.5], [0, 0]) == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# T8: Cross-class consistency
# ============================================================================

class TestCrossClassIntegrateConsistency:
    def test_approx_vs_tt_separable(self):
        """Same separable function fit by Approximation and TT integrate equally."""
        def f(x, _):
            return math.sin(x[0]) * math.cos(x[1])

        domain = [[-1, 1]] * 2
        cheb = ChebyshevApproximation(f, 2, domain, [12, 12])
        cheb.build(verbose=False)
        tt = ChebyshevTT(f, 2, domain, [12, 12])
        tt.build(verbose=False)
        a = cheb.integrate()
        b = tt.integrate()
        assert a == pytest.approx(b, abs=1e-8)

    def test_approx_vs_slider_additive(self):
        """Additive function: Approximation and Slider integrate equally."""
        def f(x, _):
            return math.sin(x[0]) + math.cos(x[1])

        domain = [[-1, 1]] * 2
        cheb = ChebyshevApproximation(f, 2, domain, [10, 10])
        cheb.build(verbose=False)
        slider = ChebyshevSlider(
            f, 2, domain, [10, 10], partition=[[0], [1]], pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        a = cheb.integrate()
        b = slider.integrate()
        assert a == pytest.approx(b, abs=1e-6)

    def test_unit_volume_normalization_all_classes(self):
        """Constant 1 integrates to vol(D) on all four classes."""
        def f(x, _):
            return 1.0

        domain = [[0, 2], [0, 3]]
        expected = 2.0 * 3.0  # = 6.0

        cheb = ChebyshevApproximation(f, 2, domain, [4, 4])
        cheb.build(verbose=False)
        assert cheb.integrate() == pytest.approx(expected, abs=1e-10)

        spl = ChebyshevSpline(f, 2, domain, [4, 4])
        spl.build(verbose=False)
        assert spl.integrate() == pytest.approx(expected, abs=1e-10)

        slider = ChebyshevSlider(
            f, 2, domain, [4, 4], partition=[[0], [1]], pivot_point=[1.0, 1.5],
        )
        slider.build(verbose=False)
        assert slider.integrate() == pytest.approx(expected, abs=1e-10)

        tt = ChebyshevTT(f, 2, domain, [4, 4])
        tt.build(verbose=False)
        assert tt.integrate() == pytest.approx(expected, abs=1e-10)

    def test_partial_integrate_then_eval(self):
        """Partial integrate result is evaluable."""
        def f(x, _):
            return x[0] * x[1] + x[0]

        tt = ChebyshevTT(f, 2, [[-1, 1]] * 2, [6, 6])
        tt.build(verbose=False)
        # ∫_{-1}^{1} (x*y + x) dx = 0 + 0 = 0 → result is 0 for all y
        result_tt = tt.integrate(dims=[0])
        assert result_tt.eval([0.5]) == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# T9: Slider integrate() validation
# ============================================================================

class TestSliderIntegrateValidation:
    def test_integrate_on_unbuilt_slider_raises(self):
        def f(x, _):
            return x[0]
        slider = ChebyshevSlider(
            f, 1, [[-1, 1]], [4],
            partition=[[0]], pivot_point=[0.0],
        )
        # Don't build
        with pytest.raises(RuntimeError, match="not.*built|build"):
            slider.integrate()

    def test_integrate_dims_oob(self):
        def f(x, _):
            return x[0]
        slider = ChebyshevSlider(
            f, 1, [[-1, 1]], [4],
            partition=[[0]], pivot_point=[0.0],
        )
        slider.build(verbose=False)
        with pytest.raises(ValueError, match="out-of-range|negative"):
            slider.integrate(dims=[5])

    def test_integrate_negative_dim(self):
        def f(x, _):
            return x[0]
        slider = ChebyshevSlider(
            f, 1, [[-1, 1]], [4],
            partition=[[0]], pivot_point=[0.0],
        )
        slider.build(verbose=False)
        with pytest.raises(ValueError):
            slider.integrate(dims=[-1])

    def test_integrate_bounds_outside_domain(self):
        def f(x, _):
            return x[0]
        slider = ChebyshevSlider(
            f, 1, [[-1, 1]], [4],
            partition=[[0]], pivot_point=[0.0],
        )
        slider.build(verbose=False)
        with pytest.raises(ValueError):
            slider.integrate(dims=[0], bounds=[(-2.0, 2.0)])


# ======================================================================
# v0.21 — Slider/TT roots, minimize, maximize
# ======================================================================

class TestSliderTo1DChebyshev:
    """Slider._to_1d_chebyshev: build a 1-D ChebyshevApproximation from
    a 1-D Slider via eval at Chebyshev nodes."""

    def test_to_1d_chebyshev_recovers_function(self):
        """1-D Slider built from f(x) = x^3, _to_1d_chebyshev returns
        a 1-D Approximation with the same values."""
        def f(x, _): return x[0] ** 3
        slider = ChebyshevSlider(
            f, num_dimensions=1, domain=[(-1, 1)], n_nodes=[7],
            partition=[[0]], pivot_point=[0.0],
        )
        slider.build(verbose=False)
        cheb_1d = slider._to_1d_chebyshev(slider)
        # Compare on a fine grid
        for x in np.linspace(-0.9, 0.9, 11):
            expected = x ** 3
            got = float(cheb_1d.eval([float(x)], derivative_order=[0]))
            assert abs(got - expected) < 1e-10, f"x={x}: got {got}, expected {expected}"

    def test_to_1d_chebyshev_preserves_domain_and_n_nodes(self):
        """The 1-D Approximation has the same domain and n_nodes as input."""
        def f(x, _): return x[0]
        slider = ChebyshevSlider(
            f, num_dimensions=1, domain=[(-2.5, 3.5)], n_nodes=[9],
            partition=[[0]], pivot_point=[0.0],
        )
        slider.build(verbose=False)
        cheb_1d = slider._to_1d_chebyshev(slider)
        assert cheb_1d.num_dimensions == 1
        assert cheb_1d.domain[0][0] == -2.5
        assert cheb_1d.domain[0][1] == 3.5
        assert cheb_1d.n_nodes[0] == 9


class TestSliderRoots:
    """Tests for ChebyshevSlider.roots() — mirrors test_calculus.py::TestRootsApprox."""

    def test_roots_quadratic_1d(self):
        """1-D Slider: roots of x^2 - 0.25 on [-1,1] are {-0.5, 0.5}."""
        def f(x, _): return x[0] ** 2 - 0.25
        slider = ChebyshevSlider(
            f, num_dimensions=1, domain=[(-1, 1)], n_nodes=[10],
            partition=[[0]], pivot_point=[0.0],
        )
        slider.build(verbose=False)
        roots = slider.roots()
        assert len(roots) == 2, f"Expected 2 roots, got {len(roots)}: {roots}"
        for r, e in zip(roots, [-0.5, 0.5]):
            assert abs(r - e) < 1e-10, f"Root {r} != {e}"

    def test_roots_no_roots_1d(self):
        """1-D Slider: exp(x) on [0,1] has no roots."""
        def f(x, _): return math.exp(x[0])
        slider = ChebyshevSlider(
            f, num_dimensions=1, domain=[(0, 1)], n_nodes=[10],
            partition=[[0]], pivot_point=[0.5],
        )
        slider.build(verbose=False)
        roots = slider.roots()
        assert len(roots) == 0, f"Expected no roots, got {roots}"

    def test_roots_2d_fixed(self):
        """2-D Slider: f(x,y) = x - y, with y fixed at 0.3, root at x=0.3."""
        def f(x, _): return x[0] - x[1]
        slider = ChebyshevSlider(
            f, num_dimensions=2, domain=[(-1, 1), (-1, 1)], n_nodes=[5, 5],
            partition=[[0], [1]], pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        roots = slider.roots(dim=0, fixed={1: 0.3})
        assert len(roots) == 1, f"Expected 1 root, got {len(roots)}: {roots}"
        assert abs(roots[0] - 0.3) < 1e-10, f"Root {roots[0]} != 0.3"

    def test_roots_3d_fixed(self):
        """3-D Slider: f(x,y,z) = x*y - z, with y=2, z=0.5, root at x=0.25."""
        def f(x, _): return x[0] * x[1] - x[2]
        slider = ChebyshevSlider(
            f, num_dimensions=3,
            domain=[(-1, 1), (1, 3), (-1, 1)], n_nodes=[7, 7, 7],
            partition=[[0], [1], [2]], pivot_point=[0.0, 2.0, 0.0],
        )
        slider.build(verbose=False)
        roots = slider.roots(dim=0, fixed={1: 2.0, 2: 0.5})
        assert len(roots) == 1, f"Expected 1 root, got {len(roots)}: {roots}"
        assert abs(roots[0] - 0.25) < 1e-8

    def test_roots_missing_fixed_raises(self):
        """Multi-D without full fixed dict raises ValueError."""
        def f(x, _): return x[0] + x[1]
        slider = ChebyshevSlider(
            f, num_dimensions=2, domain=[(-1, 1), (-1, 1)], n_nodes=[5, 5],
            partition=[[0], [1]], pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        with pytest.raises(ValueError):
            slider.roots(dim=0)

    def test_roots_fixed_out_of_domain_raises(self):
        """Fixed value outside domain raises ValueError."""
        def f(x, _): return x[0] + x[1]
        slider = ChebyshevSlider(
            f, num_dimensions=2, domain=[(-1, 1), (-1, 1)], n_nodes=[5, 5],
            partition=[[0], [1]], pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        with pytest.raises(ValueError):
            slider.roots(dim=0, fixed={1: 5.0})

    def test_roots_before_build_raises(self):
        """roots() before build() raises RuntimeError."""
        def f(x, _): return x[0]
        slider = ChebyshevSlider(
            f, num_dimensions=1, domain=[(-1, 1)], n_nodes=[5],
            partition=[[0]], pivot_point=[0.0],
        )
        with pytest.raises(RuntimeError, match="build"):
            slider.roots()
