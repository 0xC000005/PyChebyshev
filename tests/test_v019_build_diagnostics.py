"""Tests for v0.19 Build & Diagnostics."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from pychebyshev import (
    ChebyshevApproximation,
    ChebyshevSlider,
    ChebyshevSpline,
    ChebyshevTT,
)
from pychebyshev._progress import _maybe_progress


# ============================================================================
# T2: _maybe_progress() helper
# ============================================================================

class TestMaybeProgress:
    def test_passthrough_when_verbose_false(self):
        items = list(range(5))
        result = list(_maybe_progress(items, desc="test", verbose=False))
        assert result == items

    def test_passthrough_when_verbose_true(self):
        items = list(range(5))
        result = list(_maybe_progress(items, desc="test", verbose=True))
        assert result == items

    def test_wraps_with_tqdm_when_verbose_2(self):
        items = list(range(3))
        result = list(_maybe_progress(items, desc="test", verbose=2))
        assert result == items


# ============================================================================
# T3: Parallel build (Approximation)
# ============================================================================

# Module-level functions (picklable for ProcessPoolExecutor)
def _t3_f_simple(x, _):
    return math.sin(x[0]) + math.cos(x[1])


def _t3_f_with_ad(x, ad):
    return ad["k"] * x[0]


class TestParallelBuildApproximation:
    def test_parallel_matches_sequential(self):
        seq = ChebyshevApproximation(_t3_f_simple, 2, [[-1, 1], [-1, 1]], [10, 10])
        seq.build(verbose=False)
        par = ChebyshevApproximation(
            _t3_f_simple, 2, [[-1, 1], [-1, 1]], [10, 10], n_workers=2
        )
        par.build(verbose=False)
        np.testing.assert_allclose(seq.tensor_values, par.tensor_values, rtol=1e-12)

    def test_n_workers_minus_one_uses_cpu_count(self):
        cheb = ChebyshevApproximation(
            _t3_f_simple, 2, [[-1, 1], [-1, 1]], [4, 4], n_workers=-1
        )
        cheb.build(verbose=False)
        assert cheb.is_construction_finished()

    def test_n_workers_zero_rejected(self):
        with pytest.raises(ValueError, match="n_workers"):
            ChebyshevApproximation(
                _t3_f_simple, 2, [[-1, 1], [-1, 1]], [4, 4], n_workers=0
            )

    def test_n_workers_default_is_none(self):
        cheb = ChebyshevApproximation(_t3_f_simple, 2, [[-1, 1], [-1, 1]], [4, 4])
        assert cheb.n_workers is None

    def test_n_workers_negative_below_minus_one_rejected(self):
        with pytest.raises(ValueError, match="n_workers"):
            ChebyshevApproximation(
                _t3_f_simple, 2, [[-1, 1], [-1, 1]], [4, 4], n_workers=-5
            )

    def test_parallel_with_additional_data(self):
        sentinel = {"k": 7}
        cheb = ChebyshevApproximation(
            _t3_f_with_ad, 1, [[-1, 1]], [4],
            additional_data=sentinel, n_workers=2,
        )
        cheb.build(verbose=False)
        # f(x, ad) = 7 * x; eval at 0.5 should be 3.5
        assert cheb.eval([0.5], [0]) == pytest.approx(3.5, abs=1e-10)
