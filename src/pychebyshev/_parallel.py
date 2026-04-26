"""Parallel function evaluation at build time via ProcessPoolExecutor."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _normalize_n_workers(n_workers):
    """Validate and normalize n_workers ctor arg.

    Returns
    -------
    int | None
        None for sequential; positive int for parallel pool size.

    Raises
    ------
    ValueError
        If n_workers is 0, < -1, or not an int.
    """
    if n_workers is None:
        return None
    if not isinstance(n_workers, int) or isinstance(n_workers, bool):
        raise ValueError(f"n_workers must be int or None, got {type(n_workers).__name__}")
    if n_workers == 0:
        raise ValueError("n_workers must be >= 1, -1 for cpu_count, or None for sequential")
    if n_workers == -1:
        return os.cpu_count() or 1
    if n_workers < -1:
        raise ValueError(f"n_workers={n_workers} not allowed (use -1, 1, or positive int)")
    return n_workers


def _evaluate_in_parallel(function, points, additional_data, n_workers):
    """Evaluate ``function(point, additional_data)`` at every point.

    Parameters
    ----------
    function : callable
        Picklable callable taking (point, additional_data) -> scalar.
    points : iterable of points
        Iterable of point lists or arrays.
    additional_data : object
        Picklable second-arg context.
    n_workers : int | None
        Effective worker count (already normalized via _normalize_n_workers).

    Returns
    -------
    np.ndarray
        Shape (N,) float64 array of results.
    """
    points_list = [list(p) for p in points]
    if n_workers is None or n_workers == 1:
        return np.array(
            [float(function(p, additional_data)) for p in points_list],
            dtype=np.float64,
        )
    worker = _Worker(function, additional_data)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(worker, points_list))
    return np.array(results, dtype=np.float64)


class _Worker:
    """Picklable wrapper that calls ``function(point, additional_data)``."""

    def __init__(self, function, additional_data):
        self.function = function
        self.additional_data = additional_data

    def __call__(self, point):
        return float(self.function(point, self.additional_data))
