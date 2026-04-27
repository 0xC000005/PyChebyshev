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


class TestReorder:
    """Tests for the public ChebyshevTT.reorder() method."""

    def _build_3d(self):
        f = lambda p, _: np.sin(p[0]) + p[1] ** 2 + np.cos(p[2])
        tt = ChebyshevTT(
            f, 3, [[-1, 1], [-1, 1], [-1, 1]], [9, 9, 9],
            tolerance=1e-10, max_rank=12,
        )
        tt.build()
        return tt

    def test_reorder_to_same_is_clone(self):
        """reorder(self.dim_order) returns an independent clone with same eval."""
        tt = self._build_3d()
        canonical = list(range(tt.num_dimensions))
        assert tt.dim_order == canonical
        new = tt.reorder(canonical)
        assert new is not tt
        assert new.dim_order == canonical
        pt = [0.3, -0.7, 0.5]
        assert abs(new.eval(pt) - tt.eval(pt)) < 1e-10

    def test_reorder_eval_invariant(self):
        """reorder() preserves eval at random points."""
        tt = self._build_3d()
        rng = np.random.default_rng(0)
        new = tt.reorder([2, 0, 1])
        assert new.dim_order == [2, 0, 1]
        for _ in range(20):
            pt = rng.uniform(-1, 1, size=3).tolist()
            assert abs(new.eval(pt) - tt.eval(pt)) < 1e-7

    def test_reorder_invalid_permutation(self):
        """Non-permutation input raises ValueError."""
        tt = self._build_3d()
        with pytest.raises(ValueError, match="permutation"):
            tt.reorder([0, 0, 1])
        with pytest.raises(ValueError, match="permutation"):
            tt.reorder([0, 1])

    def test_reorder_storage_swap(self):
        """reorder() swaps n_nodes and domain into storage order."""
        f = lambda p, _: p[0] + 2 * p[1] + 3 * p[2]
        tt = ChebyshevTT(
            f, 3, [[-1, 1], [-2, 2], [-3, 3]], [3, 5, 7], tolerance=1e-10
        )
        tt.build()
        new = tt.reorder([2, 0, 1])
        # storage now: position 0 = orig dim 2, position 1 = orig dim 0,
        # position 2 = orig dim 1.
        assert new.n_nodes == [7, 3, 5]
        assert new.domain == [[-3, 3], [-1, 1], [-2, 2]]


def _build_non_identity_tt(num_dim=3):
    """Helper: build a TT then reorder it to deterministic non-identity permutation."""
    if num_dim == 2:
        f = lambda p, _: np.sin(p[0]) + p[1] ** 2
    else:
        f = lambda p, _: np.sin(p[0]) + p[1] ** 2 + np.cos(p[2])
    tt = ChebyshevTT(
        f, num_dim, [[-1, 1]] * num_dim, [9] * num_dim,
        tolerance=1e-10, max_rank=12,
    )
    tt.build()
    return tt


class TestEvalMultiThreading:
    """Tests for _dim_order threading through ChebyshevTT.eval_multi()."""

    def test_eval_multi_value_only_under_permutation(self):
        """eval_multi() value-only ([0,0,0]) on a permuted TT agrees with eval()."""
        tt = _build_non_identity_tt()
        permuted = tt.reorder([2, 0, 1])
        rng = np.random.default_rng(1)
        points = rng.uniform(-1, 1, size=(8, 3))
        for pt in points:
            pt_list = pt.tolist()
            out = permuted.eval_multi(pt_list, [[0, 0, 0]])
            assert abs(out[0] - tt.eval(pt_list)) < 1e-7

    def test_eval_multi_derivatives_under_permutation(self):
        """User-frame derivatives must be invariant under reorder()."""
        tt = _build_non_identity_tt()
        permuted = tt.reorder([2, 0, 1])
        rng = np.random.default_rng(2)
        derivs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for _ in range(5):
            pt = rng.uniform(-0.8, 0.8, size=3).tolist()
            ref = tt.eval_multi(pt, derivs)
            got = permuted.eval_multi(pt, derivs)
            for r, g in zip(ref, got):
                assert abs(r - g) < 1e-3  # FD has limited precision

    def test_eval_multi_second_derivative_under_permutation(self):
        """Second-order derivatives (one dim) invariant under reorder()."""
        tt = _build_non_identity_tt()
        permuted = tt.reorder([2, 0, 1])
        rng = np.random.default_rng(3)
        derivs = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        for _ in range(5):
            pt = rng.uniform(-0.8, 0.8, size=3).tolist()
            ref = tt.eval_multi(pt, derivs)
            got = permuted.eval_multi(pt, derivs)
            for r, g in zip(ref, got):
                assert abs(r - g) < 1e-2

    def test_eval_multi_cross_derivative_under_permutation(self):
        """Cross-derivative (mixed partial) invariant under reorder()."""
        tt = _build_non_identity_tt()
        permuted = tt.reorder([2, 0, 1])
        rng = np.random.default_rng(4)
        derivs = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        for _ in range(3):
            pt = rng.uniform(-0.8, 0.8, size=3).tolist()
            ref = tt.eval_multi(pt, derivs)
            got = permuted.eval_multi(pt, derivs)
            for r, g in zip(ref, got):
                assert abs(r - g) < 1e-2
