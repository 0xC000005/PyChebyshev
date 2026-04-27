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


class TestPartialIntegrateThreading:
    def test_partial_integrate_matches_canonical(self):
        """Partial integrate over original-dim 0 gives same scalar/eval as canonical TT."""
        tt_canon = _build_non_identity_tt()  # identity dim_order
        tt_perm = tt_canon.reorder([2, 0, 1])

        # Integrate out original dim 0.
        result_canon = tt_canon.integrate(dims=[0])
        result_perm = tt_perm.integrate(dims=[0])

        # Both are 2D TTs in (orig dim 1, orig dim 2) ordering — surviving
        # dims renumbered to 0..1 in original-order order.
        assert result_canon.num_dimensions == 2
        assert result_perm.num_dimensions == 2
        rng = np.random.default_rng(2)
        for _ in range(5):
            pt = rng.uniform(-1, 1, size=2).tolist()
            assert abs(result_canon.eval(pt) - result_perm.eval(pt)) < 1e-7

    def test_partial_integrate_full_still_works(self):
        """Full integrate (dims=None) is dim_order-invariant."""
        tt_canon = _build_non_identity_tt()
        tt_perm = tt_canon.reorder([1, 2, 0])
        a = tt_canon.integrate()
        b = tt_perm.integrate()
        assert abs(a - b) < 1e-7

    def test_partial_integrate_multiple_dims(self):
        """Integrating two dims (orig 0 and 2) leaves a 1D TT over orig dim 1."""
        tt_canon = _build_non_identity_tt()
        tt_perm = tt_canon.reorder([2, 0, 1])
        a = tt_canon.integrate(dims=[0, 2])
        b = tt_perm.integrate(dims=[0, 2])
        assert a.num_dimensions == 1
        assert b.num_dimensions == 1
        rng = np.random.default_rng(3)
        for _ in range(5):
            pt = rng.uniform(-1, 1, size=1).tolist()
            assert abs(a.eval(pt) - b.eval(pt)) < 1e-7


class TestToDenseThreading:
    def test_to_dense_matches_canonical(self):
        """to_dense returns the original-axis-order tensor regardless of dim_order."""
        tt_canon = _build_non_identity_tt()
        tt_perm = tt_canon.reorder([2, 0, 1])

        dense_canon = tt_canon.to_dense()
        dense_perm = tt_perm.to_dense()

        # Both should be in original axis order with same shape.
        assert dense_canon.shape == dense_perm.shape
        assert np.allclose(dense_canon, dense_perm, atol=1e-7)

    def test_to_dense_shape_in_original_axis_order(self):
        """dense.shape matches n_nodes remapped to original-dim order."""
        tt = _build_non_identity_tt().reorder([1, 0, 2])
        dense = tt.to_dense()
        # Build the expected shape in original-dim order from storage order.
        n_per_orig = [
            tt.n_nodes[tt._dim_order.index(d)] for d in range(tt.num_dimensions)
        ]
        assert dense.shape == tuple(n_per_orig)

    def test_to_dense_eval_grid_consistency(self):
        """A few corner grid points: dense[i] == eval at the corresponding point."""
        f = lambda p, _: np.sin(p[0]) + p[1] ** 2 + np.cos(p[2])
        tt = ChebyshevTT(
            f, 3, [[-1, 1], [-2, 2], [-1, 1]], [5, 5, 5],
            tolerance=1e-10, max_rank=12,
        )
        tt.build()
        permuted = tt.reorder([2, 0, 1])
        dense = permuted.to_dense()
        # Build node lists in ORIGINAL-dim order from canonical TT for clarity.
        from pychebyshev._extrude_slice import _make_nodes_for_dim
        n0 = _make_nodes_for_dim(-1, 1, 5)
        n1 = _make_nodes_for_dim(-2, 2, 5)
        n2 = _make_nodes_for_dim(-1, 1, 5)
        for (i, j, k) in [(0, 0, 0), (4, 4, 4), (2, 2, 2), (0, 4, 2)]:
            pt = [n0[i], n1[j], n2[k]]
            assert abs(dense[i, j, k] - permuted.eval(pt)) < 1e-7


class TestSliceThreading:
    def test_slice_one_dim_matches_canonical(self):
        """slice(orig_dim=0, value=v) on permuted TT agrees with same slice on canonical."""
        tt_canon = _build_non_identity_tt()
        tt_perm = tt_canon.reorder([2, 0, 1])

        sliced_canon = tt_canon.slice((0, 0.3))
        sliced_perm = tt_perm.slice((0, 0.3))

        assert sliced_canon.num_dimensions == 2
        assert sliced_perm.num_dimensions == 2
        rng = np.random.default_rng(3)
        for _ in range(10):
            pt = rng.uniform(-1, 1, size=2).tolist()
            assert abs(sliced_canon.eval(pt) - sliced_perm.eval(pt)) < 1e-7

    def test_slice_multiple_dims(self):
        """Slicing two dims (orig 0 and 2) produces a 1D TT."""
        tt_canon = _build_non_identity_tt()
        tt_perm = tt_canon.reorder([2, 0, 1])
        sliced_canon = tt_canon.slice([(0, 0.2), (2, -0.4)])
        sliced_perm = tt_perm.slice([(0, 0.2), (2, -0.4)])
        assert sliced_canon.num_dimensions == 1
        assert sliced_perm.num_dimensions == 1
        rng = np.random.default_rng(4)
        for _ in range(10):
            pt = rng.uniform(-1, 1, size=1).tolist()
            assert abs(sliced_canon.eval(pt) - sliced_perm.eval(pt)) < 1e-7

    def test_slice_dim_index_out_of_domain_user_frame(self):
        """Out-of-domain value uses original-dim's domain (user frame)."""
        f = lambda p, _: p[0] + p[1] + p[2]
        tt = ChebyshevTT(
            f, 3, [[-1, 1], [-2, 2], [-3, 3]], [3, 3, 3], tolerance=1e-10
        )
        tt.build()
        permuted = tt.reorder([2, 0, 1])
        # Original dim 1 has domain [-2, 2]; storage pos 2 also has domain [-2, 2]
        # because reorder swapped n_nodes/domain. But the error message should
        # reference dim 1 (user frame).
        with pytest.raises(ValueError, match="outside"):
            permuted.slice((1, 5.0))  # 5.0 is outside [-2, 2]
        # Within original dim 1's domain, this should succeed.
        sliced = permuted.slice((1, 1.0))
        assert sliced.num_dimensions == 2


class TestExtrudeThreading:
    def test_extrude_one_dim_matches_canonical(self):
        """Extruding a permuted TT and a canonical TT yield the same function."""
        tt_canon = _build_non_identity_tt()
        tt_perm = tt_canon.reorder([2, 0, 1])

        ext_canon = tt_canon.extrude((1, (-2, 2), 5))
        ext_perm = tt_perm.extrude((1, (-2, 2), 5))

        assert ext_canon.num_dimensions == 4
        assert ext_perm.num_dimensions == 4

        rng = np.random.default_rng(5)
        for _ in range(10):
            pt = rng.uniform(-1, 1, size=4).tolist()
            assert abs(ext_canon.eval(pt) - ext_perm.eval(pt)) < 1e-7

    def test_extrude_then_slice_round_trip(self):
        """Extrude then slice the new dim → identity (function-equality)."""
        tt = _build_non_identity_tt().reorder([2, 0, 1])
        ext = tt.extrude((1, (-2, 2), 5))
        rt = ext.slice((1, 0.0))
        assert rt.num_dimensions == 3
        rng = np.random.default_rng(6)
        for _ in range(10):
            pt = rng.uniform(-1, 1, size=3).tolist()
            assert abs(rt.eval(pt) - tt.eval(pt)) < 1e-7

    def test_extrude_at_position_zero(self):
        """Extruding at position 0 prepends a new dim at the start of original-order."""
        tt = _build_non_identity_tt().reorder([2, 0, 1])
        ext = tt.extrude((0, (0, 1), 5))
        assert ext.num_dimensions == 4
        # Slicing the new dim 0 (which is the prepended new dim) at any value
        # should give back the original 3D TT.
        rt = ext.slice((0, 0.5))
        assert rt.num_dimensions == 3
        rng = np.random.default_rng(7)
        for _ in range(5):
            pt = rng.uniform(-1, 1, size=3).tolist()
            assert abs(rt.eval(pt) - tt.eval(pt)) < 1e-7


class TestUnaryAlgebraThreading:
    def test_neg_preserves_dim_order(self):
        tt = _build_non_identity_tt().reorder([2, 0, 1])
        neg = -tt
        assert neg.dim_order == [2, 0, 1]
        rng = np.random.default_rng(7)
        for _ in range(8):
            pt = rng.uniform(-1, 1, size=3).tolist()
            assert abs(neg.eval(pt) + tt.eval(pt)) < 1e-7

    def test_mul_preserves_dim_order(self):
        tt = _build_non_identity_tt().reorder([1, 0, 2])
        scaled = tt * 3.5
        assert scaled.dim_order == [1, 0, 2]
        rng = np.random.default_rng(8)
        for _ in range(5):
            pt = rng.uniform(-1, 1, size=3).tolist()
            assert abs(scaled.eval(pt) - 3.5 * tt.eval(pt)) < 1e-7

    def test_truediv_preserves_dim_order(self):
        tt = _build_non_identity_tt().reorder([1, 0, 2])
        scaled = tt / 2.0
        assert scaled.dim_order == [1, 0, 2]
        rng = np.random.default_rng(9)
        for _ in range(5):
            pt = rng.uniform(-1, 1, size=3).tolist()
            assert abs(scaled.eval(pt) - tt.eval(pt) / 2.0) < 1e-7

    def test_rmul_preserves_dim_order(self):
        tt = _build_non_identity_tt().reorder([2, 0, 1])
        scaled = 2.0 * tt
        assert scaled.dim_order == [2, 0, 1]
        rng = np.random.default_rng(10)
        for _ in range(5):
            pt = rng.uniform(-1, 1, size=3).tolist()
            assert abs(scaled.eval(pt) - 2.0 * tt.eval(pt)) < 1e-7

    def test_imul_preserves_dim_order(self):
        tt = _build_non_identity_tt().reorder([2, 0, 1])
        ref_eval = tt.eval([0.1, 0.2, 0.3])
        tt *= 2.0
        assert tt.dim_order == [2, 0, 1]
        assert abs(tt.eval([0.1, 0.2, 0.3]) - 2.0 * ref_eval) < 1e-7

    def test_itruediv_preserves_dim_order(self):
        tt = _build_non_identity_tt().reorder([2, 0, 1])
        ref_eval = tt.eval([0.1, 0.2, 0.3])
        tt /= 2.0
        assert tt.dim_order == [2, 0, 1]
        assert abs(tt.eval([0.1, 0.2, 0.3]) - ref_eval / 2.0) < 1e-7


class TestBinaryAlgebraStrict:
    def test_add_matching_dim_order_works(self):
        tt = _build_non_identity_tt().reorder([1, 0, 2])
        s = tt + tt
        assert s.dim_order == [1, 0, 2]
        rng = np.random.default_rng(11)
        for _ in range(5):
            pt = rng.uniform(-1, 1, size=3).tolist()
            assert abs(s.eval(pt) - 2 * tt.eval(pt)) < 1e-7

    def test_sub_matching_dim_order_works(self):
        tt = _build_non_identity_tt().reorder([1, 0, 2])
        scaled = 2.0 * tt
        diff = scaled - tt
        assert diff.dim_order == [1, 0, 2]
        rng = np.random.default_rng(12)
        for _ in range(5):
            pt = rng.uniform(-1, 1, size=3).tolist()
            assert abs(diff.eval(pt) - tt.eval(pt)) < 1e-7

    def test_add_mismatched_dim_order_raises_with_hint(self):
        a = _build_non_identity_tt().reorder([1, 0, 2])
        b = _build_non_identity_tt().reorder([2, 0, 1])
        with pytest.raises(ValueError, match=r"reorder\("):
            _ = a + b

    def test_sub_mismatched_dim_order_raises(self):
        a = _build_non_identity_tt().reorder([1, 0, 2])
        b = _build_non_identity_tt().reorder([2, 0, 1])
        with pytest.raises(ValueError, match=r"reorder\("):
            _ = a - b

    def test_realign_via_reorder_then_add(self):
        a = _build_non_identity_tt().reorder([1, 0, 2])
        b = _build_non_identity_tt().reorder([2, 0, 1])
        b_aligned = b.reorder(a.dim_order)
        s = a + b_aligned
        assert s.dim_order == a.dim_order
        rng = np.random.default_rng(13)
        for _ in range(5):
            pt = rng.uniform(-1, 1, size=3).tolist()
            assert abs(s.eval(pt) - 2 * a.eval(pt)) < 1e-6

    def test_iadd_matching_dim_order_works(self):
        a = _build_non_identity_tt().reorder([1, 0, 2])
        b = _build_non_identity_tt().reorder([1, 0, 2])
        ref_eval = a.eval([0.1, 0.2, 0.3]) + b.eval([0.1, 0.2, 0.3])
        a += b
        assert a.dim_order == [1, 0, 2]
        assert abs(a.eval([0.1, 0.2, 0.3]) - ref_eval) < 1e-6

    def test_iadd_mismatched_dim_order_raises(self):
        a = _build_non_identity_tt().reorder([1, 0, 2])
        b = _build_non_identity_tt().reorder([2, 0, 1])
        with pytest.raises(ValueError, match=r"reorder\("):
            a += b


class TestCrossFeatureStress:
    def test_workflow_auto_order_then_full_surface(self):
        """Build → reorder → slice → partial integrate → pickle round-trip."""
        import pickle
        f = lambda p, _: np.sin(p[0]) + p[1] ** 2 + 0.5 * p[2]
        tt = ChebyshevTT(
            f, 3, [[-1, 1]] * 3, [9] * 3, tolerance=1e-10, max_rank=12
        )
        tt.build()

        # Reorder, slice, integrate.
        permuted = tt.reorder([2, 0, 1])
        sliced = permuted.slice((1, 0.0))  # original dim 1 → 2D TT
        integ = sliced.integrate(dims=[0])  # partial integrate over original dim 0
        # integ is a 1D TT over original dim 2.
        assert integ.num_dimensions == 1

        # Pickle round-trip.
        blob = pickle.dumps(integ)
        restored = pickle.loads(blob)
        assert restored.dim_order == integ.dim_order
        rng = np.random.default_rng(42)
        for _ in range(5):
            pt = rng.uniform(-1, 1, size=1).tolist()
            assert abs(restored.eval(pt) - integ.eval(pt)) < 1e-9

    def test_to_dense_axis_order_invariant(self):
        """to_dense from a permuted TT == to_dense from canonical TT."""
        tt_canon = _build_non_identity_tt()
        tt_perm = tt_canon.reorder([2, 0, 1])
        dense_canon = tt_canon.to_dense()
        dense_perm = tt_perm.to_dense()
        assert dense_canon.shape == dense_perm.shape
        assert np.allclose(dense_canon, dense_perm, atol=1e-7)

    def test_algebra_chain_via_reorder(self):
        """Add two TTs from different orderings via explicit reorder()."""
        f = lambda p, _: np.exp(-p[0] * p[1] - 0.3 * p[2])
        a = ChebyshevTT(
            f, 3, [[-1, 1]] * 3, [9] * 3, tolerance=1e-10, max_rank=12
        )
        a.build()
        a_perm = a.reorder([2, 0, 1])
        b_perm = a.reorder([1, 0, 2])

        # Mismatched dim_orders: explicit ValueError, hint at reorder().
        with pytest.raises(ValueError, match=r"reorder\("):
            _ = a_perm + b_perm

        # Realign and add.
        b_aligned = b_perm.reorder(a_perm.dim_order)
        s = a_perm + b_aligned
        assert s.dim_order == a_perm.dim_order

        rng = np.random.default_rng(11)
        for _ in range(5):
            pt = rng.uniform(-0.9, 0.9, size=3).tolist()
            # a, b are built from the same f; both equal a as functions.
            # Sum ≈ 2*a.
            assert abs(s.eval(pt) - 2 * a.eval(pt)) < 1e-5

    def test_extrude_slice_integrate_chain(self):
        """Chain extrude → slice → partial integrate on a permuted TT."""
        tt = _build_non_identity_tt().reorder([2, 0, 1])
        # Extrude a new dim at position 1.
        ext = tt.extrude((1, (-2, 2), 5))
        assert ext.num_dimensions == 4
        # Slice the original dim 0 (now at user-frame index 0 in the
        # 4-dim result).
        sliced = ext.slice((0, 0.5))
        assert sliced.num_dimensions == 3
        # Partial integrate over user-frame dim 0 (the new extruded dim).
        # Result is 2D.
        result = sliced.integrate(dims=[0])
        assert result.num_dimensions == 2
        # Sanity: eval is finite.
        rng = np.random.default_rng(20)
        for _ in range(3):
            pt = rng.uniform(-1, 1, size=2).tolist()
            v = result.eval(pt)
            assert np.isfinite(v)

    def test_save_load_pcb_not_supported_for_tt(self):
        """ChebyshevTT save/load goes via pickle, not .pcb (v0.14 limitation).

        v0.20.1 does not change this; just confirm the pickle round-trip
        preserves dim_order on a reordered TT.
        """
        import pickle
        tt = _build_non_identity_tt().reorder([2, 0, 1])
        blob = pickle.dumps(tt)
        restored = pickle.loads(blob)
        assert restored.dim_order == [2, 0, 1]
        assert abs(restored.eval([0.1, -0.2, 0.3]) - tt.eval([0.1, -0.2, 0.3])) < 1e-9


class TestGetEvaluationPointsFrame:
    """v0.21.1: get_evaluation_points must return columns in user-frame
    order so that eval(get_evaluation_points()[i]) round-trips."""

    def test_round_trip_canonical(self):
        """Canonical (identity _dim_order) — round-trip works."""
        def f(x, _): return 0.3 * x[0] + 0.7 * x[1] - 0.2 * x[2]
        tt = ChebyshevTT(
            f, num_dimensions=3,
            domain=[(-1, 1), (-2, 2), (-3, 3)], n_nodes=[5, 5, 5],
        )
        tt.build(verbose=False)
        points = tt.get_evaluation_points()
        # Sanity: shape and column count
        assert points.shape == (5 * 5 * 5, 3)
        # Round-trip a sample of points
        for i in [0, 7, 31, 50, 124]:
            pt = points[i]
            expected = f(pt, None)
            got = float(tt.eval(pt.tolist()))
            assert abs(got - expected) < 1e-9, (
                f"point {i}={pt}: got {got}, expected {expected}"
            )

    def test_round_trip_after_reorder(self):
        """Non-identity _dim_order via reorder([2, 0, 1]) — round-trip
        must still work in user-frame coordinates."""
        def f(x, _): return 0.3 * x[0] + 0.7 * x[1] - 0.2 * x[2]
        tt = ChebyshevTT(
            f, num_dimensions=3,
            domain=[(-1, 1), (-2, 2), (-3, 3)], n_nodes=[5, 5, 5],
        )
        tt.build(verbose=False)
        tt_reordered = tt.reorder([2, 0, 1])
        # _dim_order is now non-identity
        assert tt_reordered._dim_order != [0, 1, 2]
        points = tt_reordered.get_evaluation_points()
        assert points.shape == (5 * 5 * 5, 3)
        # Round-trip in user-frame
        for i in [0, 7, 31, 50, 124]:
            pt = points[i]
            expected = f(pt, None)
            got = float(tt_reordered.eval(pt.tolist()))
            assert abs(got - expected) < 1e-9, (
                f"point {i}={pt}: got {got}, expected {expected}"
            )

    def test_columns_match_user_frame_domain_after_reorder(self):
        """After reorder, column d's range must match the user-frame
        domain[d], not the storage-frame domain."""
        def f(x, _): return x[0] + x[1] + x[2]
        tt = ChebyshevTT(
            f, num_dimensions=3,
            domain=[(-1, 1), (-2, 2), (-3, 3)], n_nodes=[5, 5, 5],
        )
        tt.build(verbose=False)
        tt_reordered = tt.reorder([2, 0, 1])
        points = tt_reordered.get_evaluation_points()
        # Column 0 must be in user-frame dim 0 range: [-1, 1]
        assert points[:, 0].min() >= -1.0 - 1e-12
        assert points[:, 0].max() <= 1.0 + 1e-12
        # Column 1 must be in user-frame dim 1 range: [-2, 2]
        assert points[:, 1].min() >= -2.0 - 1e-12
        assert points[:, 1].max() <= 2.0 + 1e-12
        # Column 2 must be in user-frame dim 2 range: [-3, 3]
        assert points[:, 2].min() >= -3.0 - 1e-12
        assert points[:, 2].max() <= 3.0 + 1e-12


class TestEvalMultiNoDimOrderMutation:
    """Issue #19: eval_multi must not mutate self._dim_order via try/finally
    (race-prone under concurrent calls). The fix is structural: introduce
    _eval_storage_frame and call it from eval_multi after permuting once."""

    def test_eval_multi_source_does_not_assign_dim_order(self):
        """Source-level check: the body of eval_multi contains no
        'self._dim_order = ' assignment."""
        import inspect
        from pychebyshev.tensor_train import ChebyshevTT

        source = inspect.getsource(ChebyshevTT.eval_multi)
        forbidden = [
            "self._dim_order = ",
            "self._dim_order=",
        ]
        for pat in forbidden:
            assert pat not in source, (
                f"eval_multi must not contain assignment '{pat}' "
                f"after issue #19 fix"
            )

    def test_eval_multi_correctness_under_reorder(self):
        """Behavioral check: eval_multi still returns correct values
        for a reordered TT."""
        def f(x, _): return x[0] ** 2 + x[1] - x[2]
        tt = ChebyshevTT(
            f, num_dimensions=3, domain=[(-1, 1)] * 3, n_nodes=[7, 7, 7],
        )
        tt.build(verbose=False)
        tt_reordered = tt.reorder([2, 0, 1])
        # eval_multi for value + first-derivative-wrt-x0 (in user frame)
        results = tt_reordered.eval_multi(
            point=[0.3, 0.4, 0.5],
            derivative_orders=[[0, 0, 0], [1, 0, 0]],
        )
        # f(0.3, 0.4, 0.5) = 0.09 + 0.4 - 0.5 = -0.01
        # df/dx0(0.3, 0.4, 0.5) = 2*0.3 = 0.6
        assert abs(results[0] - (-0.01)) < 1e-9
        assert abs(results[1] - 0.6) < 1e-9
