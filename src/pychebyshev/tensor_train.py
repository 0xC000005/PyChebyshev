"""Chebyshev interpolation in Tensor Train (TT) format.

Implements TT-Cross approximation for building Chebyshev interpolants
of high-dimensional functions from O(d * n * r^2) function evaluations
instead of O(n^d). Evaluation uses vectorized TT inner products via
numpy einsum.

References
----------
- Oseledets & Tyrtyshnikov (2010), "TT-cross approximation for
  multidimensional arrays", Linear Algebra and its Applications
- Oseledets (2011), "Tensor-Train Decomposition", SIAM J. Sci. Comput.
- Ruiz & Zeron (2021), "Machine Learning for Risk Calculations",
  Wiley Finance, Chapter 6: Tensor Train Format
- Goreinov & Tyrtyshnikov (2001), "The maximal-volume concept in
  approximation by low-rank matrices"
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from typing import Callable, List, Tuple

import numpy as np
from numpy.polynomial.chebyshev import chebpts1


# ======================================================================
# Module-level helpers
# ======================================================================

def _maxvol(A: np.ndarray, tol: float = 1.05, max_iters: int = 100) -> np.ndarray:
    """Find rows of a tall matrix with approximately maximal volume.

    Given a matrix A of shape (m, r) with m >= r, finds r row indices
    such that the submatrix A[idx] has approximately maximal |det|.
    This is a key subroutine in TT-Cross: it selects which cross points
    to use for the next dimension, picking the most informative samples.

    The algorithm has two phases:

    1. **Initialization via column-pivoted QR on A^T.**
       ``Q, R, piv = qr(A.T, pivoting=True)`` selects the r most
       linearly independent rows of A (= columns of A^T) as the
       starting index set.

    2. **Iterative row-swapping refinement.**
       Computes the coefficient matrix B = A @ inv(A[idx]) of shape
       (m, r). Each entry B[i,j] measures how much |det(A[idx])| would
       change if row j in the index set were replaced with row i.
       While max|B[i,j]| > tol (default 1.05):
         - Swap row j for row i in the index set.
         - Apply a rank-1 update to B (O(m*r), avoids re-inverting).
       Each swap strictly increases |det(A[idx])|, so the algorithm
       converges in O(r^2) iterations in practice.

    Parameters
    ----------
    A : ndarray of shape (m, r)
        Tall matrix with m >= r.
    tol : float, optional
        Stop when max |B[i,j]| <= tol. Default is 1.05.
    max_iters : int, optional
        Maximum number of row-swap iterations. Default is 100.

    Returns
    -------
    ndarray of shape (r,)
        Row indices forming the maximum-volume submatrix.

    References
    ----------
    Goreinov, Tyrtyshnikov & Zamarashkin (1997), "A theory of
    pseudoskeleton approximations", Linear Algebra Appl.
    """
    from scipy.linalg import qr as scipy_qr

    m, r = A.shape
    if m <= r:
        return np.arange(m, dtype=np.intp)

    # Phase 1: Initialize via column-pivoted QR on A^T.
    # A^T has shape (r, m). Column pivoting selects the m columns of A^T
    # (= rows of A) that are most linearly independent — a good starting
    # set for iterative refinement.
    _, _, piv = scipy_qr(A.T, pivoting=True)
    idx = piv[:r].copy().astype(np.intp)

    # Phase 2: Iterative refinement.
    # B = A @ inv(A[idx]) is the coefficient matrix, shape (m, r).
    # B[i,j] represents how well row i can be expressed as a linear
    # combination of the selected rows. If |B[i,j]| > 1, swapping
    # row j for row i increases |det(A[idx])|.
    try:
        B = np.linalg.solve(A[idx].T, A.T).T
    except np.linalg.LinAlgError:
        return idx

    for _ in range(max_iters):
        # Find the entry with largest magnitude — the most beneficial swap.
        i, j = np.unravel_index(np.argmax(np.abs(B)), B.shape)
        if np.abs(B[i, j]) <= tol:
            break
        # Swap: replace idx[j] with row i.
        idx[j] = i
        # Rank-1 update of B: avoids recomputing A @ inv(A[idx]) from scratch.
        # After swapping, the new B differs by a rank-1 correction.
        bij = B[i, j]
        col_j = B[:, j].copy()
        row_i = B[i, :].copy()
        B -= np.outer(col_j, row_i) / bij
        B[:, j] = col_j / bij

    return idx


def _tt_cross(
    func: Callable,
    grids: List[np.ndarray],
    max_rank: int,
    tol: float,
    max_sweeps: int,
    verbose: bool,
    seed: int | None = None,
) -> Tuple[List[np.ndarray], int]:
    """Build TT cores from a callable via alternating TT-Cross.

    Uses the DMRG-style cross approximation algorithm with maxvol pivot
    selection (Oseledets & Tyrtyshnikov 2010). The algorithm alternates
    left-to-right and right-to-left sweeps, building TT cores via cross
    interpolation and refining the index sets via maxvol.

    Optimizations over a naive implementation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **Eval caching** (10x fewer evaluations):
        Function evaluations are cached in a dictionary keyed by the grid
        index tuple ``(i_0, i_1, ..., i_{d-1})``. Many grid points are
        visited multiple times across sweeps and between L→R / R→L passes.
        For 5D Black-Scholes with ``max_rank=15``, caching reduces unique
        evaluations from ~85,000 to ~7,400.

    **Per-mode rank caps** (prevents over-ranking):
        At bond k, the theoretical maximum TT rank is
        ``min(prod(n[:k]), prod(n[k:]))``. For example, with 5 dims of 11
        nodes, the first bond can have rank at most min(11, 14641) = 11.
        Requesting ``max_rank=15`` won't exceed this — the algorithm caps
        automatically.

    **SVD-based adaptive rank** (smaller cores where possible):
        Instead of always using ``max_rank`` columns from a QR factorization,
        the cross matrix is decomposed via SVD. Only singular values above
        ``1e-12 * sigma_max`` are kept, so dimensions where the function
        has low effective rank naturally get smaller cores. The interest rate
        dimension in Black-Scholes, for example, often needs lower rank
        than the spot price dimension.

    **Half-sweep convergence** (skip unnecessary R→L):
        Error is checked after the L→R half-sweep. If the TT already
        reproduces the function within ``tolerance``, the R→L sweep is
        skipped entirely. For separable functions, this means convergence
        in a single L→R pass.

    **Best-cores tracking** (robust to oscillation):
        TT-Cross error can oscillate between L→R and R→L sweeps. The
        algorithm keeps a copy of the best cores (lowest test error) seen
        across all convergence checks, and returns those when stopping.
        Stops early after 3 consecutive checks without ≥10% improvement.

    Parameters
    ----------
    func : callable
        Function f(point, data) -> float where point is a list of floats.
    grids : list of 1D ndarray
        Chebyshev nodes per dimension, in the original domain.
    max_rank : int
        Maximum TT rank.
    tol : float
        Convergence tolerance (relative error at random test points).
    max_sweeps : int
        Maximum number of full (L→R + R→L) sweeps.
    verbose : bool
        Print convergence info.
    seed : int or None
        Random seed for reproducibility of index initialization and
        convergence test points.

    Returns
    -------
    cores : list of ndarray
        TT cores, each of shape (r_{k-1}, n_k, r_k). These are
        **value** cores (function values at Chebyshev nodes); the
        caller (``build()``) converts them to coefficient cores via
        DCT-II.
    total_evals : int
        Number of **unique** function evaluations performed (cache size).
    """
    rng = np.random.default_rng(seed)
    d = len(grids)
    n = [len(g) for g in grids]

    # --- Eval cache ---
    # Function evaluations are cached by grid index tuple. When the same
    # grid point is requested again (e.g., during the R→L sweep, or in
    # a subsequent sweep), the cached value is returned immediately.
    # This is the single largest optimization: for 5D BS with max_rank=15,
    # caching reduces evaluations from ~85K to ~7.4K.
    _cache: dict[tuple, float] = {}

    def _eval_func(grid_indices):
        """Evaluate func at grid point, using cache to avoid redundant calls."""
        key = tuple(int(grid_indices[dim]) for dim in range(d))
        if key not in _cache:
            point = [float(grids[dim][key[dim]]) for dim in range(d)]
            _cache[key] = func(point, None)
        return _cache[key]

    def _eval_tt(cores, grid_indices):
        """Evaluate TT at grid indices via chain of matrix multiplications."""
        v = np.ones((1, 1))
        for dim in range(d):
            v = v @ cores[dim][:, grid_indices[dim], :]
        return v[0, 0]

    # --- Per-mode rank caps ---
    # At bond k (between dims k-1 and k), the theoretical maximum TT rank
    # equals the rank of the k-th unfolding matrix, which is bounded by
    # min(prod(n[:k]), prod(n[k:])). For 5D with n=11, the boundary bonds
    # can have rank at most 11 (not 14641), so setting max_rank > 11 has
    # no effect.
    rank_caps = [1] * (d + 1)
    for k in range(1, d):
        left_size = int(np.prod(n[:k]))
        right_size = int(np.prod(n[k:]))
        rank_caps[k] = min(max_rank, left_size, right_size)

    # --- Initialize ranks and index sets ---
    # r[k] = current rank at bond k. Starts conservative, updated each sweep.
    r = [1] * (d + 1)
    for k in range(1, d):
        r[k] = min(rank_caps[k], n[k - 1], n[k])

    # J_right[k] holds right multi-indices for dimension k. Shape (r_{k+1}, d-k-1).
    # Each row is a tuple of grid indices for dimensions k+1, ..., d-1.
    # These define which "columns" of the tensor unfolding the cross
    # approximation uses.
    J_right: List[np.ndarray | None] = [None] * d
    for k in range(d - 1):
        rk = r[k + 1]
        n_right = d - k - 1
        if n_right == 0:
            J_right[k] = np.zeros((1, 0), dtype=np.intp)
        else:
            J_right[k] = np.column_stack([
                rng.integers(0, n[k + 1 + j], size=rk)
                for j in range(n_right)
            ])
    J_right[d - 1] = np.zeros((1, 0), dtype=np.intp)

    # J_left[k] holds left multi-indices for dimension k. Shape (r_k, k).
    # Each row is a tuple of grid indices for dimensions 0, ..., k-1.
    # These define which "rows" of the tensor unfolding the cross
    # approximation uses.
    J_left: List[np.ndarray | None] = [None] * d
    J_left[0] = np.zeros((1, 0), dtype=np.intp)

    # --- Best-cores tracking ---
    # TT-Cross error can oscillate between L→R and R→L sweeps (a good
    # L→R result may be partially degraded by R→L rebalancing). We keep
    # a copy of whichever cores gave the lowest test error, and return
    # those when the algorithm stops.
    best_error = float("inf")
    best_cores = None
    stale_checks = 0  # consecutive checks without ≥10% improvement

    # --- Convergence check ---
    # Evaluate the TT at random grid points and compare against exact
    # function values. Uses the eval cache, so test-point function
    # evaluations are free if those points were already evaluated.
    n_test = min(20, max(5, d))

    def _check_error(cores_list):
        """Relative error of TT vs exact function at random grid points."""
        pts = np.column_stack([
            rng.integers(0, n[dim], size=n_test) for dim in range(d)
        ])
        tt_v = np.array([_eval_tt(cores_list, pts[t]) for t in range(n_test)])
        ex_v = np.array([_eval_func(pts[t]) for t in range(n_test)])
        ref = np.linalg.norm(ex_v)
        if ref > 0:
            return float(np.linalg.norm(tt_v - ex_v) / ref)
        return float(np.linalg.norm(tt_v - ex_v))

    # --- Sweep loop ---
    cores = [None] * d

    for sweep in range(max_sweeps):
        # ============================================================
        # Left-to-right half-sweep (k = 0, ..., d-2)
        #
        # For each dimension k, we:
        #   1. Build a cross matrix C from function evaluations
        #   2. Determine the effective rank via SVD
        #   3. Select pivot rows via maxvol
        #   4. Form the TT core via cross interpolation
        #   5. Update the left index set for the next dimension
        # ============================================================
        for k in range(d - 1):
            left = J_left[k]      # (r_left, k) — left multi-indices
            right = J_right[k]    # (r_right, d-k-1) — right multi-indices
            rl = left.shape[0]    # left rank
            rr = right.shape[0]   # right rank
            nk = n[k]             # nodes in this dimension
            cap = rank_caps[k + 1]  # theoretical rank bound at bond k+1

            # Step 1: Build cross matrix C of shape (rl * nk, rr).
            # Each row corresponds to a (left multi-index, node j_k) pair.
            # Each column corresponds to a right multi-index.
            # Entry C[a*nk + i, b] = f(left[a], j_k=i, right[b]).
            C = np.empty((rl * nk, rr))
            for a in range(rl):
                for i in range(nk):
                    for b in range(rr):
                        idx = list(left[a]) + [i] + list(right[b])
                        C[a * nk + i, b] = _eval_func(idx)

            # Step 2: SVD-based adaptive rank selection.
            # Decompose C = U @ diag(S) @ Vt. Keep singular values above
            # 1e-12 * sigma_max (only dropping numerically zero modes).
            # The effective rank is further capped by per-mode bounds.
            U, S, Vt = np.linalg.svd(C, full_matrices=False)
            if S[0] > 0:
                effective = int(np.sum(S > 1e-12 * S[0]))
            else:
                effective = 1
            rank = max(1, min(cap, effective, U.shape[1]))
            U = U[:, :rank]

            # Step 3: Maxvol pivot selection.
            # Find the `rank` rows of U whose submatrix has approximately
            # maximal |det|. These pivots identify the most "informative"
            # (left, node) index pairs for cross interpolation.
            if U.shape[0] > U.shape[1]:
                pivots = _maxvol(U)
            else:
                pivots = np.arange(U.shape[0], dtype=np.intp)
            pivots = pivots[:rank]

            # Step 4: Form TT core via cross interpolation.
            # C_hat = U @ inv(U[pivots]) satisfies C_hat[pivots] = I,
            # meaning exact interpolation at the selected cross points.
            # Reshape to 3D core (r_left, n_k, rank).
            try:
                C_hat = U @ np.linalg.inv(U[pivots])
            except np.linalg.LinAlgError:
                C_hat = U  # fallback if pivot submatrix is singular
            cores[k] = C_hat.reshape(rl, nk, rank)

            # Step 5: Update left index set for dimension k+1.
            # Each pivot row p encodes a (left_index a, node_index i_k) pair.
            # The new left multi-index is left[a] extended with i_k.
            new_left = np.empty((rank, k + 1), dtype=np.intp)
            for p_idx, prow in enumerate(pivots):
                a, ik = divmod(int(prow), nk)
                a = min(a, rl - 1)
                if k == 0:
                    new_left[p_idx] = [ik]
                else:
                    new_left[p_idx] = list(J_left[k][a]) + [ik]
            J_left[k + 1] = new_left
            r[k + 1] = rank

        # ============================================================
        # Last core (from L→R): direct evaluation
        # ============================================================
        k = d - 1
        left = J_left[k]
        rl = left.shape[0]
        nk = n[k]
        C_last = np.empty((rl, nk))
        for a in range(rl):
            for i in range(nk):
                idx = list(left[a]) + [i]
                C_last[a, i] = _eval_func(idx)
        cores[d - 1] = C_last[:, :, np.newaxis]

        # ============================================================
        # Half-sweep convergence check (after L→R, before R→L)
        # ============================================================
        rel_error_lr = _check_error(cores)

        if verbose:
            ranks_str = str([1] + [c.shape[2] for c in cores])
            print(f"    Sweep {sweep + 1} L->R: rel error = {rel_error_lr:.2e}, "
                  f"unique evals = {len(_cache):,}, ranks = {ranks_str}")

        # Track best
        if rel_error_lr < best_error * 0.9:  # 10% improvement threshold
            best_error = rel_error_lr
            best_cores = [c.copy() for c in cores]
            stale_checks = 0
        else:
            stale_checks += 1

        if rel_error_lr < tol:
            if verbose:
                print(f"    Converged after {sweep + 1} sweeps (L->R)")
            cores = best_cores
            break

        # Stale check: stop if no improvement in 3 consecutive checks
        if stale_checks >= 3 and best_error < 1e-3:
            if verbose:
                print(f"    No improvement in {stale_checks} checks "
                      f"(best = {best_error:.2e}) — stopping")
            cores = best_cores
            break

        # ============================================================
        # Right-to-left half-sweep (k = d-1, ..., 1)
        #
        # Analogous to L→R but processes dimensions in reverse.
        # The cross matrix is transposed (rows = right indices,
        # cols = left indices), and maxvol pivots update the
        # *right* index sets (used by the next L→R sweep).
        # ============================================================
        for k in range(d - 1, 0, -1):
            left = J_left[k]      # (r_left, k) — left multi-indices
            right = J_right[k]    # (r_right, d-k-1) — right multi-indices
            rl = left.shape[0]
            rr = right.shape[0]
            nk = n[k]
            cap = rank_caps[k]

            # Build cross matrix C of shape (rl, nk * rr).
            # Rows = left multi-indices, columns = (node j_k, right multi-index).
            C = np.empty((rl, nk * rr))
            for a in range(rl):
                for i in range(nk):
                    for b in range(rr):
                        idx = list(left[a]) + [i] + list(right[b])
                        C[a, i * rr + b] = _eval_func(idx)

            # Transpose to get shape (nk * rr, rl), then SVD.
            # Working with C^T lets us apply maxvol to the (node, right)
            # index space, selecting which right multi-indices to keep.
            Ct = C.T
            U, S, Vt = np.linalg.svd(Ct, full_matrices=False)
            if S[0] > 0:
                effective = int(np.sum(S > 1e-12 * S[0]))
            else:
                effective = 1
            rank = max(1, min(cap, effective, U.shape[1]))
            U = U[:, :rank]

            # Maxvol on the transposed space.
            if U.shape[0] > U.shape[1]:
                pivots = _maxvol(U)
            else:
                pivots = np.arange(U.shape[0], dtype=np.intp)
            pivots = pivots[:rank]

            # Cross interpolation on the transposed matrix.
            # C_hat^T = U @ inv(U[pivots]), then transpose back.
            try:
                C_hat_t = U @ np.linalg.inv(U[pivots])
            except np.linalg.LinAlgError:
                C_hat_t = U
            cores[k] = C_hat_t.T.reshape(rank, nk, rr)

            # Update right index set for dimension k-1.
            # Each pivot row p encodes a (node_index i_k, right_index b) pair.
            # The new right multi-index is [i_k] + right[b].
            n_right_new = d - k
            new_right = np.empty((rank, n_right_new), dtype=np.intp)
            for p_idx, prow in enumerate(pivots):
                ik, b = divmod(int(prow), max(rr, 1))
                ik = min(ik, nk - 1)
                b = min(b, max(rr, 1) - 1)
                if right.shape[1] == 0:
                    new_right[p_idx] = [ik]
                else:
                    new_right[p_idx] = [ik] + list(right[b])
            J_right[k - 1] = new_right
            r[k] = rank

        # ============================================================
        # First core (from R→L): direct evaluation
        # ============================================================
        right = J_right[0]
        rr = right.shape[0]
        nk = n[0]
        C_first = np.empty((nk, rr))
        for i in range(nk):
            for b in range(rr):
                idx = [i] + list(right[b])
                C_first[i, b] = _eval_func(idx)
        cores[0] = C_first[np.newaxis, :, :]  # (1, n_0, rr)

        # ============================================================
        # Full convergence check after R→L
        # ============================================================
        rel_error = _check_error(cores)

        if verbose:
            print(f"    Sweep {sweep + 1} R->L: rel error = {rel_error:.2e}, "
                  f"unique evals = {len(_cache):,}")

        # Track best
        if rel_error < best_error * 0.9:
            best_error = rel_error
            best_cores = [c.copy() for c in cores]
            stale_checks = 0
        else:
            stale_checks += 1

        if rel_error < tol:
            if verbose:
                print(f"    Converged after {sweep + 1} sweeps")
            cores = best_cores
            break

        # Stale check
        if stale_checks >= 3 and best_error < 1e-3:
            if verbose:
                print(f"    No improvement in {stale_checks} checks "
                      f"(best = {best_error:.2e}) — stopping")
            cores = best_cores
            break
    else:
        # max_sweeps exhausted — use best cores found
        if best_cores is not None:
            cores = best_cores

    return cores, len(_cache)


def _tt_svd(
    func: Callable,
    grids: List[np.ndarray],
    max_rank: int,
    tol: float,
    verbose: bool,
) -> Tuple[List[np.ndarray], int]:
    """Build TT cores via SVD of the full tensor.

    Evaluates the function at **all** grid points ($O(n^d)$), constructs
    the full $d$-dimensional tensor, then sequentially decomposes it
    via truncated SVD. This produces optimal TT ranks (up to the SVD
    truncation tolerance) and is useful for:

    - Validating TT-Cross accuracy against the best possible TT.
    - Confirming the function's intrinsic TT rank structure.
    - Moderate-dimension problems ($d \\leq 6$) where the full tensor
      fits in memory.

    The algorithm reshapes the tensor into a matrix at each unfolding,
    computes the SVD, truncates small singular values, and absorbs
    the remainder into the next unfolding.

    Parameters
    ----------
    func : callable
        Function f(point, data) -> float.
    grids : list of 1D ndarray
        Chebyshev nodes per dimension, in the original domain.
    max_rank : int
        Maximum TT rank for truncation.
    tol : float
        Singular value truncation tolerance. Singular values below
        ``tol * sigma_max`` are discarded at each unfolding.
    verbose : bool
        Print progress info.

    Returns
    -------
    cores : list of ndarray
        TT cores, each of shape (r_{k-1}, n_k, r_k). These are
        **value** cores (function values); the caller converts to
        coefficient cores via DCT-II.
    total_evals : int
        Total number of function evaluations (= full tensor size).
    """
    d = len(grids)
    n = [len(g) for g in grids]
    full_size = int(np.prod(n))

    if verbose:
        print(f"  Building full tensor ({full_size:,} evaluations)...")

    # Build full tensor
    T = np.empty(n)
    total_evals = 0
    for idx in np.ndindex(*n):
        point = [float(grids[dim][idx[dim]]) for dim in range(d)]
        T[idx] = func(point, None)
        total_evals += 1

    # Sequential SVD decomposition
    cores = []
    C = T  # will be reshaped progressively
    r_prev = 1

    for k in range(d - 1):
        C = C.reshape(r_prev * n[k], -1)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)

        # Determine rank: cap at max_rank, drop near-zero singular values
        rank = min(max_rank, len(S))
        if S[0] > 0:
            # Keep singular values above tol * sigma_max
            effective = np.sum(S > tol * S[0])
            rank = max(1, min(rank, int(effective)))

        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]

        cores.append(U.reshape(r_prev, n[k], rank))
        C = np.diag(S) @ Vt
        r_prev = rank

    # Last core
    cores.append(C.reshape(r_prev, n[d - 1], 1))

    if verbose:
        ranks = [1] + [c.shape[2] for c in cores]
        print(f"  TT-SVD ranks: {ranks}")

    return cores, total_evals


# ======================================================================
# ChebyshevTT class
# ======================================================================

class ChebyshevTT:
    """Chebyshev interpolation in Tensor Train format.

    For functions of 5+ dimensions where full tensor interpolation
    is infeasible. Uses TT-Cross to build from O(d * n * r^2) function
    evaluations instead of O(n^d), then evaluates via TT inner product.

    Parameters
    ----------
    function : callable
        Function to approximate. Signature: ``f(point, data) -> float``
        where ``point`` is a list of floats and ``data`` is arbitrary
        additional data (can be None).
    num_dimensions : int
        Number of input dimensions.
    domain : list of (float, float)
        Bounds [(lo, hi), ...] for each dimension.
    n_nodes : list of int
        Number of Chebyshev nodes per dimension.
    max_rank : int, optional
        Maximum TT rank. Higher = more accurate, more expensive.
        Default is 10.
    tolerance : float, optional
        Convergence tolerance for TT-Cross. Default is 1e-6.
    max_sweeps : int, optional
        Maximum number of TT-Cross sweeps. Default is 10.

    Examples
    --------
    >>> import math
    >>> def f(x, _):
    ...     return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])
    >>> tt = ChebyshevTT(f, 3, [[-1, 1], [-1, 1], [-1, 1]], [11, 11, 11])
    >>> tt.build(verbose=False)  # doctest: +SKIP
    >>> tt.eval([0.5, 0.3, 0.1])  # doctest: +SKIP
    0.8764...
    """

    def __init__(
        self,
        function: Callable,
        num_dimensions: int,
        domain: List[Tuple[float, float]],
        n_nodes: List[int],
        max_rank: int = 10,
        tolerance: float = 1e-6,
        max_sweeps: int = 10,
    ):
        # Validate inputs
        if len(domain) != num_dimensions:
            raise ValueError(
                f"domain has {len(domain)} entries but num_dimensions={num_dimensions}"
            )
        if len(n_nodes) != num_dimensions:
            raise ValueError(
                f"n_nodes has {len(n_nodes)} entries but num_dimensions={num_dimensions}"
            )

        self.function = function
        self.num_dimensions = num_dimensions
        self.domain = domain
        self.n_nodes = n_nodes
        self.max_rank = max_rank
        self.tolerance = tolerance
        self.max_sweeps = max_sweeps

        # Build-time state
        self._coeff_cores: List[np.ndarray] | None = None
        self._built: bool = False
        self._tt_ranks: List[int] | None = None
        self._build_time: float = 0.0
        self._total_build_evals: int = 0
        self._cached_error_estimate: float | None = None

    def build(
        self,
        verbose: bool = True,
        seed: int | None = None,
        method: str = "cross",
    ) -> None:
        """Build TT approximation and convert to Chebyshev coefficient cores.

        The build process has three stages:

        1. **Generate Chebyshev grids.** Compute Type I Chebyshev nodes
           in each dimension, scaled to the specified domain.
        2. **Build value cores.** Either TT-Cross (evaluating at
           $O(d \\cdot n \\cdot r^2)$ strategically selected points) or
           TT-SVD (evaluating the full $O(n^d)$ tensor, then decomposing
           via sequential SVD).
        3. **Convert to coefficient cores.** Apply DCT-II along the node
           axis of each core to convert from function values at Chebyshev
           nodes to Chebyshev expansion coefficients. This enables
           evaluation at arbitrary (non-grid) points via the Chebyshev
           polynomial inner product.

        Parameters
        ----------
        verbose : bool, optional
            If True, print build progress. Default is True.
        seed : int or None, optional
            Random seed for TT-Cross initialization. Default is None.
            Ignored when ``method='svd'``.
        method : ``'cross'`` or ``'svd'``, optional
            Build algorithm. ``'cross'`` (default) uses TT-Cross to
            evaluate the function at $O(d \\cdot n \\cdot r^2)$ strategically
            selected points. ``'svd'`` builds the full tensor and decomposes
            via truncated SVD -- only feasible for moderate dimensions
            ($d \\leq 6$) but useful for validation.
        """
        from scipy.fft import dct

        if method not in ("cross", "svd"):
            raise ValueError(f"method must be 'cross' or 'svd', got {method!r}")

        start = time.time()
        self._cached_error_estimate = None

        full_tensor_size = int(np.prod(self.n_nodes))
        if verbose:
            print(
                f"Building {self.num_dimensions}D ChebyshevTT "
                f"(max_rank={self.max_rank}, method={method!r})..."
            )
            print(f"  Full tensor would need {full_tensor_size:,} evaluations")

        # Step 1: Generate Chebyshev Type I nodes per dimension (in original domain)
        grids = []
        for d in range(self.num_dimensions):
            nodes_std = chebpts1(self.n_nodes[d])  # [-1, 1], ascending
            a, b = self.domain[d]
            nodes_scaled = 0.5 * (a + b) + 0.5 * (b - a) * nodes_std
            grids.append(np.sort(nodes_scaled))

        # Step 2: Build value cores
        if method == "cross":
            if verbose:
                print("  Running TT-Cross...")
            value_cores, n_evals = _tt_cross(
                self.function,
                grids,
                max_rank=self.max_rank,
                tol=self.tolerance,
                max_sweeps=self.max_sweeps,
                verbose=verbose,
                seed=seed,
            )
        else:  # svd
            value_cores, n_evals = _tt_svd(
                self.function,
                grids,
                max_rank=self.max_rank,
                tol=self.tolerance,
                verbose=verbose,
            )
        self._total_build_evals = n_evals

        # Step 3: Convert value cores to Chebyshev coefficient cores via DCT-II.
        #
        # Each value core has shape (r_{k-1}, n_k, r_k), where the middle
        # axis contains function values at Chebyshev Type I nodes. We need
        # Chebyshev expansion coefficients for evaluation at arbitrary points.
        #
        # The transform for each core is:
        #   1. Reverse along the node axis ([:, ::-1, :]) to match DCT-II
        #      convention with Type I nodes.
        #   2. Apply DCT-II along axis=1 and normalize by n_k.
        #   3. Halve the zeroth coefficient (DCT-II convention for
        #      Chebyshev series).
        #
        # This is the same transform used by ChebyshevApproximation's
        # _chebyshev_coefficients_1d(), extended to 3D cores.
        coeff_cores = []
        for k, core in enumerate(value_cores):
            n_k = core.shape[1]
            coeff = dct(core[:, ::-1, :], type=2, axis=1) / n_k
            coeff[:, 0, :] /= 2
            coeff_cores.append(coeff)

        self._coeff_cores = coeff_cores

        # Step 4: Extract TT ranks
        self._tt_ranks = [1]
        for core in self._coeff_cores:
            self._tt_ranks.append(core.shape[2])

        self._build_time = time.time() - start
        self._built = True

        if verbose:
            tt_storage = sum(c.size for c in self._coeff_cores)
            print(f"  Built in {self._build_time:.3f}s "
                  f"({n_evals:,} function evaluations)")
            print(f"  TT ranks: {self._tt_ranks}")
            print(f"  Compression: {full_tensor_size:,} -> {tt_storage:,} elements "
                  f"({full_tensor_size / tt_storage:.1f}x)")

    def _check_built(self) -> None:
        """Raise RuntimeError if build() has not been called."""
        if not self._built:
            raise RuntimeError("Call build() before using this method.")

    def eval(self, point: List[float]) -> float:
        """Evaluate at a single point via TT inner product.

        Computes the Chebyshev interpolant value at an arbitrary point by
        contracting the pre-computed coefficient cores with Chebyshev
        polynomial values. For each dimension $k$:

        1. Scale the query coordinate to $[-1, 1]$.
        2. Evaluate all Chebyshev polynomials $T_0, \\ldots, T_{n_k-1}$.
        3. Contract with the coefficient core:
           $v = \\sum_j q_j \\cdot \\text{core}[:, j, :]$

        The chain of contractions reduces to a scalar.
        Cost: $O(d \\cdot n \\cdot r^2)$ per point.

        Parameters
        ----------
        point : list of float
            Query point, one coordinate per dimension.

        Returns
        -------
        float
            Interpolated value at the query point.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        self._check_built()

        result = np.ones((1, 1))
        for d in range(self.num_dimensions):
            # Map point[d] from [a, b] to [-1, 1]
            a, b = self.domain[d]
            scaled = 2.0 * (point[d] - a) / (b - a) - 1.0
            # Evaluate all Chebyshev polynomials T_0..T_{n-1} at scaled point
            q = np.polynomial.chebyshev.chebval(
                scaled, np.eye(self.n_nodes[d])
            )  # shape (n_d,)
            # Contract polynomial values with coefficient core:
            # v[i,k] = sum_j q[j] * core[i,j,k]  →  shape (r_{d-1}, r_d)
            v = np.einsum("j,ijk->ik", q, self._coeff_cores[d])
            # Chain multiply: result = result @ v
            result = result @ v
        return float(result[0, 0])

    def eval_batch(self, points: np.ndarray) -> np.ndarray:
        """Evaluate at multiple points simultaneously.

        Vectorizes the TT inner product over all N points using
        ``np.einsum`` for batched matrix contractions. For each
        dimension, all N polynomial vectors are contracted with the
        coefficient core in a single einsum call, then all N chain
        multiplications proceed in parallel. Typical speedup is
        15--20x over calling :meth:`eval` in a loop.

        Parameters
        ----------
        points : ndarray of shape (N, num_dimensions)
            Query points.

        Returns
        -------
        ndarray of shape (N,)
            Interpolated values at each query point.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        self._check_built()

        points = np.asarray(points)
        N = points.shape[0]
        result = np.ones((N, 1, 1))

        for d in range(self.num_dimensions):
            a, b = self.domain[d]
            scaled = 2.0 * (points[:, d] - a) / (b - a) - 1.0  # (N,)
            # Evaluate T_0..T_{n-1} at all N scaled points simultaneously.
            # chebval with identity matrix gives polynomials as columns.
            eye = np.eye(self.n_nodes[d])
            Q = np.polynomial.chebyshev.chebval(scaled, eye)  # (n_d, N)
            Q = Q.T  # (N, n_d)
            # Batched contraction: V[n,i,k] = sum_j Q[n,j] * core[i,j,k]
            V = np.einsum("nj,ijk->nik", Q, self._coeff_cores[d])
            # Batched chain multiply: result[n,i,k] = sum_j result[n,i,j] * V[n,j,k]
            result = np.einsum("nij,njk->nik", result, V)

        return result[:, 0, 0]

    def eval_multi(
        self, point: List[float], derivative_orders: List[List[int]]
    ) -> List[float]:
        """Evaluate with finite-difference derivatives at a single point.

        Uses central finite differences. The first entry in
        ``derivative_orders`` is typically ``[0, 0, ..., 0]`` for the
        function value; subsequent entries specify derivative orders
        per dimension.

        Parameters
        ----------
        point : list of float
            Evaluation point in the full n-dimensional space.
        derivative_orders : list of list of int
            Each inner list specifies derivative order per dimension.
            Supports 0 (value), 1 (first derivative), and 2 (second
            derivative).

        Returns
        -------
        list of float
            One result per derivative order specification.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        self._check_built()

        results = []
        for deriv_order in derivative_orders:
            if all(d == 0 for d in deriv_order):
                results.append(self.eval(point))
            else:
                results.append(self._fd_derivative(point, deriv_order))
        return results

    def _fd_derivative(self, point: List[float], deriv_order: List[int]) -> float:
        """Compute a single derivative via central finite differences.

        Parameters
        ----------
        point : list of float
            Evaluation point.
        deriv_order : list of int
            Derivative order per dimension.

        Returns
        -------
        float
            Finite-difference derivative estimate.
        """
        # Identify which dimensions have nonzero derivative order
        active_dims = [
            (d, order) for d, order in enumerate(deriv_order) if order > 0
        ]

        if len(active_dims) == 1:
            d, order = active_dims[0]
            return self._fd_single_dim(point, d, order)
        elif len(active_dims) == 2:
            (d1, o1), (d2, o2) = active_dims
            if o1 == 1 and o2 == 1:
                return self._fd_cross_deriv(point, d1, d2)
            else:
                # Higher-order cross derivatives: nest single-dim FD
                # Approximate by applying FD sequentially
                return self._fd_nested(point, active_dims)
        else:
            return self._fd_nested(point, active_dims)

    def _fd_step(self, d: int) -> float:
        """Compute FD step size for dimension d."""
        a, b = self.domain[d]
        return (b - a) * 1e-4

    def _nudge_point(self, point: List[float], d: int, h: float) -> List[float]:
        """Nudge point away from domain boundary if needed."""
        pt = list(point)
        a, b = self.domain[d]
        needed = h * 1.5
        if pt[d] - a < needed:
            pt[d] = a + needed
        if b - pt[d] < needed:
            pt[d] = b - needed
        return pt

    def _fd_single_dim(self, point: List[float], d: int, order: int) -> float:
        """Central FD for a single dimension."""
        h = self._fd_step(d)
        pt = self._nudge_point(point, d, h)

        if order == 1:
            pt_plus = list(pt)
            pt_minus = list(pt)
            pt_plus[d] += h
            pt_minus[d] -= h
            return (self.eval(pt_plus) - self.eval(pt_minus)) / (2.0 * h)
        elif order == 2:
            pt_plus = list(pt)
            pt_minus = list(pt)
            pt_plus[d] += h
            pt_minus[d] -= h
            f_plus = self.eval(pt_plus)
            f_center = self.eval(pt)
            f_minus = self.eval(pt_minus)
            return (f_plus - 2.0 * f_center + f_minus) / (h * h)
        else:
            raise ValueError(f"Derivative order {order} not supported (use 1 or 2)")

    def _fd_cross_deriv(self, point: List[float], d1: int, d2: int) -> float:
        """Central FD for mixed partial d^2f / dx_{d1} dx_{d2}."""
        h1 = self._fd_step(d1)
        h2 = self._fd_step(d2)
        pt = self._nudge_point(point, d1, h1)
        pt = self._nudge_point(pt, d2, h2)

        def make_pt(delta1: float, delta2: float) -> List[float]:
            p = list(pt)
            p[d1] += delta1
            p[d2] += delta2
            return p

        f_pp = self.eval(make_pt(+h1, +h2))
        f_pm = self.eval(make_pt(+h1, -h2))
        f_mp = self.eval(make_pt(-h1, +h2))
        f_mm = self.eval(make_pt(-h1, -h2))
        return (f_pp - f_pm - f_mp + f_mm) / (4.0 * h1 * h2)

    def _fd_nested(
        self, point: List[float], active_dims: List[Tuple[int, int]]
    ) -> float:
        """Nested FD for higher-order or multi-dimensional derivatives."""
        # Apply FD one dimension at a time
        if len(active_dims) == 0:
            return self.eval(point)

        d, order = active_dims[0]
        remaining = active_dims[1:]
        h = self._fd_step(d)
        pt = self._nudge_point(point, d, h)

        if order == 1:
            pt_plus = list(pt)
            pt_minus = list(pt)
            pt_plus[d] += h
            pt_minus[d] -= h
            f_plus = self._fd_nested(pt_plus, remaining)
            f_minus = self._fd_nested(pt_minus, remaining)
            return (f_plus - f_minus) / (2.0 * h)
        elif order == 2:
            pt_plus = list(pt)
            pt_minus = list(pt)
            pt_plus[d] += h
            pt_minus[d] -= h
            f_plus = self._fd_nested(pt_plus, remaining)
            f_center = self._fd_nested(pt, remaining)
            f_minus = self._fd_nested(pt_minus, remaining)
            return (f_plus - 2.0 * f_center + f_minus) / (h * h)
        else:
            raise ValueError(f"Derivative order {order} not supported (use 1 or 2)")

    # ------------------------------------------------------------------
    # Error estimation
    # ------------------------------------------------------------------

    def error_estimate(self) -> float:
        """Estimate interpolation error from Chebyshev coefficient cores.

        For each dimension d, takes the maximum magnitude of the last
        Chebyshev coefficient across all "rows" and "columns" of the
        core (i.e., max over left-rank and right-rank indices of
        ``|core[:, -1, :]|``). Returns the sum across dimensions.

        This is an approximate analog of the ex ante error estimation
        from Ruiz & Zeron (2021), Section 3.4, adapted for TT format.

        Returns
        -------
        float
            Estimated interpolation error.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        self._check_built()

        if self._cached_error_estimate is not None:
            return self._cached_error_estimate

        total_error = 0.0
        for d in range(self.num_dimensions):
            core = self._coeff_cores[d]
            # Last coefficient along the Chebyshev axis
            last_coeff_slice = core[:, -1, :]  # shape (r_{d-1}, r_d)
            max_last = np.max(np.abs(last_coeff_slice))
            total_error += max_last

        self._cached_error_estimate = total_error
        return total_error

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tt_ranks(self) -> List[int]:
        """TT ranks [1, r_1, r_2, ..., r_{d-1}, 1].

        Returns
        -------
        list of int
            The TT rank vector. Only available after :meth:`build`.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        self._check_built()
        return list(self._tt_ranks)

    @property
    def compression_ratio(self) -> float:
        """Ratio of full tensor elements to TT storage elements.

        Returns
        -------
        float
            Compression ratio (> 1 means TT is more compact).

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        self._check_built()
        full_size = int(np.prod(self.n_nodes))
        tt_size = sum(c.size for c in self._coeff_cores)
        return full_size / tt_size

    @property
    def total_build_evals(self) -> int:
        """Total number of function evaluations used during build.

        Returns
        -------
        int
            Number of function evaluations. Only meaningful after
            :meth:`build`.
        """
        return self._total_build_evals

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Return picklable state, excluding the original function."""
        from pychebyshev._version import __version__

        state = self.__dict__.copy()
        state["function"] = None
        state["_pychebyshev_version"] = __version__
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state from a pickled dict."""
        from pychebyshev._version import __version__

        saved_version = state.pop("_pychebyshev_version", None)
        if saved_version is not None and saved_version != __version__:
            warnings.warn(
                f"This object was saved with pychebyshev {saved_version}, "
                f"but you are loading it with {__version__}. "
                f"Evaluation results may differ if internal data layout changed.",
                UserWarning,
                stacklevel=2,
            )

        self.__dict__.update(state)
        self.function = None

        # Ensure fields added in later versions exist (backward compat)
        if not hasattr(self, "_cached_error_estimate"):
            self._cached_error_estimate = None

    def save(self, path: str | os.PathLike) -> None:
        """Save the built TT interpolant to a file.

        The original function is **not** saved -- only the numerical
        data needed for evaluation. The saved file can be loaded with
        :meth:`load` without access to the original function.

        Parameters
        ----------
        path : str or path-like
            Destination file path.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        self._check_built()
        with open(os.fspath(path), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ChebyshevTT":
        """Load a previously saved TT interpolant from a file.

        The loaded object can evaluate immediately; no rebuild is needed.
        The ``function`` attribute will be ``None``. Assign a new function
        before calling ``build()`` again if a rebuild is desired.

        Parameters
        ----------
        path : str or path-like
            Path to the saved file.

        Returns
        -------
        ChebyshevTT
            The restored TT interpolant.

        Warns
        -----
        UserWarning
            If the file was saved with a different PyChebyshev version.

        .. warning::

            This method uses :mod:`pickle` internally. Pickle can execute
            arbitrary code during deserialization. **Only load files you
            trust.**
        """
        with open(os.fspath(path), "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a {cls.__name__} instance, got {type(obj).__name__}"
            )
        return obj

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ChebyshevTT("
            f"dims={self.num_dimensions}, "
            f"nodes={self.n_nodes}, "
            f"max_rank={self.max_rank}, "
            f"built={self._built})"
        )

    def __str__(self) -> str:
        status = "built" if self._built else "not built"
        full_tensor_size = int(np.prod(self.n_nodes))

        # Truncate display for high-dimensional objects (>6 dims)
        max_display = 6
        if self.num_dimensions > max_display:
            nodes_str = (
                "[" + ", ".join(str(n) for n in self.n_nodes[:max_display])
                + ", ...]"
            )
            domain_str = (
                " x ".join(
                    f"[{lo}, {hi}]"
                    for lo, hi in self.domain[:max_display]
                )
                + " x ..."
            )
        else:
            nodes_str = str(self.n_nodes)
            domain_str = " x ".join(
                f"[{lo}, {hi}]" for lo, hi in self.domain
            )

        lines = [
            f"ChebyshevTT ({self.num_dimensions}D, {status})",
            f"  Nodes:       {nodes_str}",
        ]

        if self._built:
            tt_storage = sum(c.size for c in self._coeff_cores)
            lines.append(f"  TT ranks:    {self._tt_ranks}")
            lines.append(
                f"  Compression: {full_tensor_size:,} -> "
                f"{tt_storage:,} elements "
                f"({full_tensor_size / tt_storage:.1f}x)"
            )
            lines.append(
                f"  Build:       {self._build_time:.3f}s "
                f"({self._total_build_evals:,} function evals)"
            )
            lines.append(f"  Domain:      {domain_str}")
            lines.append(f"  Error est:   {self.error_estimate():.2e}")
        else:
            lines.append(f"  Domain:      {domain_str}")

        return "\n".join(lines)
