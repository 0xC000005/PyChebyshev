"""Optional tqdm-based progress bars. Activated by ``verbose=2``."""
from __future__ import annotations

import warnings

try:
    from tqdm import tqdm  # type: ignore[import-untyped]
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    tqdm = None  # type: ignore[assignment]


def _maybe_progress(iterable, *, desc: str, verbose):
    """Wrap ``iterable`` in a tqdm progress bar when ``verbose == 2`` and
    tqdm is available; otherwise return the iterable unchanged.

    A one-time warning is emitted if ``verbose == 2`` but tqdm is missing,
    pointing the user at the ``pychebyshev[viz]`` install extra.
    """
    if verbose != 2:
        return iterable
    if not _HAS_TQDM:
        warnings.warn(
            "verbose=2 requires tqdm; install with `pip install pychebyshev[viz]`",
            stacklevel=2,
        )
        return iterable
    return tqdm(iterable, desc=desc, leave=False)
