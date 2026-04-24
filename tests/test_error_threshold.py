"""Tests for error-threshold-driven construction (v0.11)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from pychebyshev import ChebyshevApproximation


def _sin2d(x, _):
    return math.sin(x[0]) + math.sin(x[1])


def _sin3d(x, _):
    return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])


class TestConstructorValidation:
    """Signature-level validation — no build required."""

    def test_explicit_n_unchanged(self):
        """Existing API still works: all-int n_nodes, no error_threshold."""
        cheb = ChebyshevApproximation(_sin2d, 2, [[-1, 1], [-1, 1]], [11, 11])
        assert cheb.n_nodes == [11, 11]
        assert cheb.error_threshold is None

    def test_error_threshold_only(self):
        """Auto-N mode: error_threshold without n_nodes."""
        cheb = ChebyshevApproximation(
            _sin2d, 2, [[-1, 1], [-1, 1]], error_threshold=1e-6,
        )
        assert cheb.error_threshold == 1e-6
        # n_nodes populated with sentinel Nones
        assert cheb.n_nodes == [None, None]

    def test_semi_variable(self):
        """Mixed: some None, some int, plus error_threshold."""
        cheb = ChebyshevApproximation(
            _sin3d, 3, [[-1, 1]] * 3,
            n_nodes=[None, 15, 15],
            error_threshold=1e-6,
        )
        assert cheb.n_nodes == [None, 15, 15]
        assert cheb.error_threshold == 1e-6

    def test_none_without_threshold_raises(self):
        """None in n_nodes without error_threshold → ValueError."""
        with pytest.raises(ValueError, match="error_threshold"):
            ChebyshevApproximation(
                _sin2d, 2, [[-1, 1], [-1, 1]], n_nodes=[None, 11],
            )

    def test_neither_n_nor_threshold_raises(self):
        """Omitting both n_nodes and error_threshold → ValueError."""
        with pytest.raises(ValueError, match="n_nodes.*error_threshold"):
            ChebyshevApproximation(_sin2d, 2, [[-1, 1], [-1, 1]])

    def test_all_none_all_dim_auto(self):
        """All-None n_nodes is equivalent to omitting n_nodes."""
        cheb = ChebyshevApproximation(
            _sin2d, 2, [[-1, 1], [-1, 1]],
            n_nodes=[None, None], error_threshold=1e-6,
        )
        assert cheb.n_nodes == [None, None]

    def test_max_n_default(self):
        cheb = ChebyshevApproximation(
            _sin2d, 2, [[-1, 1], [-1, 1]], error_threshold=1e-6,
        )
        assert cheb.max_n == 64

    def test_max_n_custom(self):
        cheb = ChebyshevApproximation(
            _sin2d, 2, [[-1, 1], [-1, 1]],
            error_threshold=1e-6, max_n=128,
        )
        assert cheb.max_n == 128


class TestDoublingLoop:
    """Tests that auto-N build actually achieves the target ε."""

    def test_1d_converges(self):
        """Simple 1-D function should converge well under max_n."""
        cheb = ChebyshevApproximation(
            lambda x, _: math.sin(x[0]),
            1, [[-1, 1]], error_threshold=1e-8,
        )
        cheb.build(verbose=False)
        assert cheb.n_nodes[0] is not None
        assert cheb.n_nodes[0] <= 64
        assert cheb.error_estimate() <= 1e-8

    def test_2d_auto_converges(self):
        cheb = ChebyshevApproximation(
            _sin2d, 2, [[-1, 1], [-1, 1]], error_threshold=1e-6,
        )
        cheb.build(verbose=False)
        assert all(n is not None for n in cheb.n_nodes)
        assert cheb.error_estimate() <= 1e-6

    def test_3d_auto_converges(self):
        cheb = ChebyshevApproximation(
            _sin3d, 3, [[-1, 1]] * 3, error_threshold=1e-6,
        )
        cheb.build(verbose=False)
        assert cheb.error_estimate() <= 1e-6

    def test_semi_variable_respects_fixed_dims(self):
        """Fixed dims must not be grown by the doubling loop."""
        cheb = ChebyshevApproximation(
            _sin3d, 3, [[-1, 1]] * 3,
            n_nodes=[None, 15, 15], error_threshold=1e-6,
        )
        cheb.build(verbose=False)
        assert cheb.n_nodes[1] == 15
        assert cheb.n_nodes[2] == 15
        assert cheb.n_nodes[0] is not None

    def test_already_accurate_stops_immediately(self):
        """A low-frequency function should stop at the minimum N (3)."""
        cheb = ChebyshevApproximation(
            lambda x, _: x[0] + x[1],  # linear — exact at N=3
            2, [[-1, 1], [-1, 1]],
            error_threshold=1e-6,
        )
        cheb.build(verbose=False)
        assert cheb.n_nodes[0] == 3
        assert cheb.n_nodes[1] == 3

    def test_tight_threshold_eventual(self):
        """Tight ε on a smooth function hits a reasonable N."""
        cheb = ChebyshevApproximation(
            lambda x, _: math.exp(-x[0]**2),
            1, [[-2, 2]], error_threshold=1e-12,
        )
        cheb.build(verbose=False)
        assert cheb.error_estimate() <= 1e-12

    def test_rebuild_with_tighter_threshold_rebuilds_auto_dims(self):
        """Second build() with a tighter threshold must re-run the doubling loop."""
        cheb = ChebyshevApproximation(
            lambda x, _: math.sin(x[0]),
            1, [[-1, 1]], error_threshold=1e-4,
        )
        cheb.build(verbose=False)
        n_first = cheb.n_nodes[0]
        err_first = cheb.error_estimate()
        assert err_first <= 1e-4

        # Tighten and rebuild
        cheb.error_threshold = 1e-10
        cheb.build(verbose=False)
        assert cheb.error_estimate() <= 1e-10
        # Tighter threshold should force at least as many nodes as before
        assert cheb.n_nodes[0] >= n_first


class TestMaxNCap:
    """Tests for max_n cap behavior in the auto-N doubling loop."""

    def test_cap_warns_and_returns_usable_object(self):
        """Deliberately unreachable ε with small max_n → warn, still usable."""
        # Oscillatory function + tight ε + tiny cap → cap kicks in.
        # Use sin(20x) + cos(17x) rather than sin(50x): the latter
        # aliases to antisymmetric values at n=3, causing the
        # last-coefficient error estimate to be 0 and the doubling
        # loop to exit before the cap is reached.
        def wiggly(x, _):
            return math.sin(20 * x[0]) + math.cos(17 * x[0])

        cheb = ChebyshevApproximation(
            wiggly, 1, [[-1, 1]],
            error_threshold=1e-12, max_n=16,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cheb.build(verbose=False)
        assert any(
            issubclass(w.category, RuntimeWarning)
            and "max_n" in str(w.message)
            for w in caught
        ), f"Expected RuntimeWarning mentioning 'max_n', got: {[str(w.message) for w in caught]}"

        # Object still usable — eval returns finite value
        value = cheb.vectorized_eval([0.1], [0])
        assert np.isfinite(value)

        # Final N respects cap
        assert cheb.n_nodes[0] <= 16

    def test_no_warning_when_threshold_met(self):
        """Successful auto-N build should not emit any RuntimeWarning."""
        cheb = ChebyshevApproximation(
            lambda x, _: math.sin(x[0]) + math.sin(x[1]),
            2, [[-1, 1], [-1, 1]], error_threshold=1e-6,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cheb.build(verbose=False)
        runtime_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
        ]
        assert not runtime_warnings, (
            f"Expected no RuntimeWarnings, got: "
            f"{[str(w.message) for w in runtime_warnings]}"
        )


class TestGetErrorThreshold:
    """Tests for the get_error_threshold() accessor."""

    def test_returns_threshold_when_set(self):
        cheb = ChebyshevApproximation(
            _sin2d, 2, [[-1, 1], [-1, 1]], error_threshold=1e-6,
        )
        cheb.build(verbose=False)
        assert cheb.get_error_threshold() == 1e-6

    def test_returns_none_when_not_set(self):
        cheb = ChebyshevApproximation(
            _sin2d, 2, [[-1, 1], [-1, 1]], n_nodes=[11, 11],
        )
        cheb.build(verbose=False)
        assert cheb.get_error_threshold() is None


class TestGetOptimalN1:
    """Tests for the get_optimal_n1() 1-D capacity estimator classmethod."""

    def test_returns_int_above_minimum(self):
        n = ChebyshevApproximation.get_optimal_n1(
            lambda x, _: math.sin(x[0]),
            domain_1d=[-1, 1],
            error_threshold=1e-8,
        )
        assert isinstance(n, int)
        assert 3 <= n <= 64

    def test_smooth_low_freq_small_n(self):
        """Linear function is exact at N=3."""
        n = ChebyshevApproximation.get_optimal_n1(
            lambda x, _: x[0],
            domain_1d=[-1, 1],
            error_threshold=1e-10,
        )
        assert n == 3

    def test_high_freq_larger_n(self):
        """Higher frequency needs more nodes."""
        # sin(kx) alone is odd about 0 — on the symmetric Chebyshev grid
        # its highest-frequency DCT coefficient aliases to 0, so the
        # error estimate stops the doubling loop at n=3 regardless of k
        # (same aliasing issue called out for test_respects_max_n).
        # sin(kx) + cos(kx) breaks the antisymmetry and exercises the
        # "higher frequency needs more nodes" invariant cleanly.
        n_low = ChebyshevApproximation.get_optimal_n1(
            lambda x, _: math.sin(x[0]) + math.cos(x[0]),
            domain_1d=[-1, 1],
            error_threshold=1e-8,
        )
        n_high = ChebyshevApproximation.get_optimal_n1(
            lambda x, _: math.sin(10 * x[0]) + math.cos(10 * x[0]),
            domain_1d=[-1, 1],
            error_threshold=1e-8,
        )
        assert n_high > n_low

    def test_respects_max_n(self):
        """Unreachable ε → returns max_n with a warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            n = ChebyshevApproximation.get_optimal_n1(
                lambda x, _: math.sin(50 * x[0]) + math.cos(43 * x[0]),
                domain_1d=[-1, 1],
                error_threshold=1e-14,
                max_n=8,
            )
        assert n == 8
        assert any(
            issubclass(w.category, RuntimeWarning) and "max_n" in str(w.message)
            for w in caught
        )


class TestSplineErrorThreshold:
    """error_threshold applied per-piece in ChebyshevSpline."""

    def test_1d_with_knot(self):
        from pychebyshev import ChebyshevSpline

        def abs_fn(x, _):
            return abs(x[0])

        spl = ChebyshevSpline(
            abs_fn, 1, [[-1, 1]],
            n_nodes=[None],
            knots=[[0.0]],
            error_threshold=1e-6,
        )
        spl.build(verbose=False)
        # Each piece should have fully-resolved Ns and hit the threshold
        for piece in spl._pieces:
            assert all(n is not None for n in piece.n_nodes)
            assert piece.error_estimate() <= 1e-6

    def test_2d_no_knots_matches_flat(self):
        """Spline with empty knots lists should behave like a single ChebyshevApproximation."""
        from pychebyshev import ChebyshevSpline

        spl = ChebyshevSpline(
            _sin2d, 2, [[-1, 1], [-1, 1]],
            n_nodes=[None, None],
            knots=[[], []],
            error_threshold=1e-6,
        )
        spl.build(verbose=False)
        # Single piece (flat list)
        assert len(spl._pieces) == 1
        assert spl._pieces[0].error_estimate() <= 1e-6

    def test_explicit_n_still_works(self):
        """Backward compat: existing fixed-N spline builds unchanged."""
        from pychebyshev import ChebyshevSpline

        def abs_fn(x, _):
            return abs(x[0])

        spl = ChebyshevSpline(
            abs_fn, 1, [[-1, 1]],
            n_nodes=[15],
            knots=[[0.0]],
        )
        spl.build(verbose=False)
        # Pieces should use the explicit N
        for piece in spl._pieces:
            assert piece.n_nodes == [15]
