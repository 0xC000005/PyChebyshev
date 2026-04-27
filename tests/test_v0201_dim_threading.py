"""Tests for v0.20.1 — TT _dim_order full threading + reorder()."""

from __future__ import annotations

import numpy as np
import pytest

from pychebyshev import ChebyshevTT
from pychebyshev._algebra import _tt_swap_adjacent


class TestTtSwapAdjacent:
    """Unit tests for the _tt_swap_adjacent helper."""

    def test_swap_preserves_function_2d(self):
        """Swapping the only swap-position in a 2D TT preserves the function."""
        f = lambda p, _: p[0] + 2 * p[1]
        tt = ChebyshevTT(f, 2, [[-1, 1], [-2, 2]], [5, 5], tolerance=1e-10)
        tt.build()
        cores = [c.copy() for c in tt._coeff_cores]

        swapped = _tt_swap_adjacent(
            cores, i=0, max_rank=10, tolerance=1e-10
        )

        from pychebyshev.tensor_train import _coeff_core_to_value_core

        def to_dense_storage(core_list):
            value_cores = [_coeff_core_to_value_core(c) for c in core_list]
            r = value_cores[0]
            for k in range(1, len(value_cores)):
                r = np.einsum("...r,rjs->...js", r, value_cores[k])
            return r.reshape([c.shape[1] for c in value_cores])

        orig = to_dense_storage(cores)
        new = to_dense_storage(swapped)
        assert np.allclose(new, orig.transpose(1, 0), atol=1e-9)

    def test_swap_3d_interior(self):
        """Swapping interior axes (1,2) in 3D preserves the function tensor."""
        f = lambda p, _: p[0] + 2 * p[1] + 3 * p[2] + p[0] * p[1]
        tt = ChebyshevTT(
            f, 3, [[-1, 1], [-1, 1], [-1, 1]], [5, 5, 5], tolerance=1e-10
        )
        tt.build()
        cores = [c.copy() for c in tt._coeff_cores]

        swapped = _tt_swap_adjacent(cores, i=1, max_rank=10, tolerance=1e-10)

        from pychebyshev.tensor_train import _coeff_core_to_value_core

        def to_dense_storage(core_list):
            value_cores = [_coeff_core_to_value_core(c) for c in core_list]
            r = value_cores[0]
            for k in range(1, len(value_cores)):
                r = np.einsum("...r,rjs->...js", r, value_cores[k])
            return r.reshape([c.shape[1] for c in value_cores])

        orig = to_dense_storage(cores)
        new = to_dense_storage(swapped)
        assert np.allclose(new, orig.transpose(0, 2, 1), atol=1e-9)

    def test_swap_invalid_index(self):
        """Out-of-range i raises ValueError."""
        f = lambda p, _: p[0]
        tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [3, 3])
        tt.build()
        with pytest.raises(ValueError, match="out of range"):
            _tt_swap_adjacent(list(tt._coeff_cores), i=5, max_rank=10)
        with pytest.raises(ValueError, match="out of range"):
            _tt_swap_adjacent(list(tt._coeff_cores), i=-1, max_rank=10)
