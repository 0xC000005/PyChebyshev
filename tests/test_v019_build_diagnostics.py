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
