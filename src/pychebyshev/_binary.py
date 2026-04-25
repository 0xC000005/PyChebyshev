"""Portable .pcb binary serialization format (v0.14).

This is a private module. Public access is via
``ChebyshevApproximation.save/load`` and ``ChebyshevSpline.save/load`` with
``format='binary'``.

The full format specification lives at
``docs/user-guide/binary-format.md``.

The layout uses fixed little-endian byte order, ``f64`` for all floats,
``uint32`` for all integers, and length-prefixed sections. No padding.

Reading uses ``numpy.frombuffer`` to map raw bytes to ``ndarray`` without
copying when possible. Writing uses ``ndarray.tobytes()`` after asserting
dtype and C-contiguity.
"""

from __future__ import annotations

import os
import struct
from typing import BinaryIO

import numpy as np

# --- Format constants ----------------------------------------------------

MAGIC = b"PCB\x00"
MAJOR = 1
MINOR = 0
CLASS_TAG_APPROX = 1
CLASS_TAG_SPLINE = 2

_HEADER_SIZE = 12  # 4 magic + 1 major + 1 minor + 2 class_tag + 4 reserved


# --- Low-level helpers ---------------------------------------------------


def _write_u32(f: BinaryIO, n: int) -> None:
    """Write a little-endian uint32."""
    f.write(struct.pack("<I", n))


def _read_u32(f: BinaryIO) -> int:
    """Read a little-endian uint32. Raises ValueError on EOF."""
    raw = f.read(4)
    if len(raw) != 4:
        raise ValueError("unexpected EOF reading uint32")
    return struct.unpack("<I", raw)[0]


def _write_u32_array(f: BinaryIO, arr) -> None:
    """Write a 1-D array as little-endian uint32 values.

    Raises TypeError if the array's dtype is not uint32 — silent dtype
    coercion would mask call-site bugs (e.g. an int64 array of node
    counts being truncated). Same policy as :func:`_write_f64_array`.
    """
    a = np.asarray(arr)
    if a.dtype != np.uint32:
        raise TypeError(
            f"binary format requires uint32 arrays, got dtype={a.dtype}"
        )
    a = np.ascontiguousarray(a, dtype="<u4")
    f.write(a.tobytes())


def _read_u32_array(f: BinaryIO, count: int) -> np.ndarray:
    """Read ``count`` little-endian uint32 values into a 1-D ndarray."""
    nbytes = count * 4
    raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise ValueError(
            f"unexpected EOF reading uint32 array (wanted {nbytes} bytes, "
            f"got {len(raw)})"
        )
    return np.frombuffer(raw, dtype="<u4").astype(np.uint32, copy=True)


def _write_f64_array(f: BinaryIO, arr) -> None:
    """Write a numeric array as little-endian f64 values, C-contiguous.

    Raises TypeError if the array's dtype is not f64. We do not silently
    upcast f32 -> f64 because the format spec is fixed at f64 and a
    silent cast would mask shape/dtype bugs at the call site.
    """
    a = np.asarray(arr)
    if a.dtype != np.float64:
        raise TypeError(
            f"binary format requires float64 arrays, got dtype={a.dtype}"
        )
    a = np.ascontiguousarray(a, dtype="<f8")
    f.write(a.tobytes())


def _read_f64_array(f: BinaryIO, count: int) -> np.ndarray:
    """Read ``count`` little-endian f64 values into a 1-D ndarray (copy)."""
    nbytes = count * 8
    raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise ValueError(
            f"unexpected EOF reading f64 array (wanted {nbytes} bytes, "
            f"got {len(raw)})"
        )
    return np.frombuffer(raw, dtype="<f8").astype(np.float64, copy=True)


# --- Header --------------------------------------------------------------


def _write_header(f: BinaryIO, class_tag: int) -> None:
    """Write the 12-byte header."""
    f.write(MAGIC)
    f.write(struct.pack("<BB", MAJOR, MINOR))
    f.write(struct.pack("<H", class_tag))
    f.write(b"\x00\x00\x00\x00")  # reserved


def _read_header(f: BinaryIO) -> int:
    """Read and validate the header. Returns the class tag."""
    raw = f.read(_HEADER_SIZE)
    if len(raw) != _HEADER_SIZE:
        raise ValueError(
            f"unexpected EOF reading header (wanted {_HEADER_SIZE} bytes, "
            f"got {len(raw)})"
        )
    if raw[:4] != MAGIC:
        raise ValueError("not a PyChebyshev binary file (bad magic)")
    major, minor = struct.unpack("<BB", raw[4:6])
    if major != MAJOR:
        raise ValueError(
            f"unsupported .pcb major version {major} "
            f"(this build reads major {MAJOR})"
        )
    class_tag = struct.unpack("<H", raw[6:8])[0]
    reserved = raw[8:12]
    if reserved != b"\x00\x00\x00\x00":
        raise ValueError("reserved header bytes nonzero — file may be corrupt")
    return class_tag


# --- Format detection ----------------------------------------------------


def detect_format(path) -> str:
    """Return ``'binary'`` if the file starts with ``MAGIC``, else ``'pickle'``.

    Files shorter than 4 bytes are reported as ``'pickle'`` — the pickle
    loader will then raise its own clear error.
    """
    p = os.fspath(path)
    with open(p, "rb") as f:
        head = f.read(4)
    if head == MAGIC:
        return "binary"
    return "pickle"


# --- ChebyshevApproximation ---------------------------------------------


def write_approx(f: BinaryIO, cheb) -> None:
    """Write a built ``ChebyshevApproximation`` to a binary stream.

    Raises RuntimeError if the interpolant has not been built.
    """
    if cheb.tensor_values is None:
        raise RuntimeError("Cannot save an unbuilt ChebyshevApproximation")

    _write_header(f, CLASS_TAG_APPROX)

    d = int(cheb.num_dimensions)
    _write_u32(f, d)

    domain_lo = np.array([cheb.domain[i][0] for i in range(d)], dtype=np.float64)
    domain_hi = np.array([cheb.domain[i][1] for i in range(d)], dtype=np.float64)
    _write_f64_array(f, domain_lo)
    _write_f64_array(f, domain_hi)

    n_nodes = np.array(cheb.n_nodes, dtype=np.uint32)
    _write_u32_array(f, n_nodes)

    tensor_values = np.ascontiguousarray(cheb.tensor_values, dtype=np.float64)
    _write_f64_array(f, tensor_values.ravel(order="C"))


def read_approx(f: BinaryIO):
    """Read a ``ChebyshevApproximation`` from a binary stream.

    Reconstructs the object via ``ChebyshevApproximation.from_values`` so
    weights, differentiation matrices, and the eval cache are recomputed
    consistently with every other code path.
    """
    from pychebyshev import ChebyshevApproximation

    tag = _read_header(f)
    if tag != CLASS_TAG_APPROX:
        raise ValueError(
            f"file contains class_tag {tag}, expected "
            f"{CLASS_TAG_APPROX} (ChebyshevApproximation)"
        )

    d = _read_u32(f)
    if d < 1:
        raise ValueError(f"num_dimensions must be >= 1, got {d}")

    domain_lo = _read_f64_array(f, count=d)
    domain_hi = _read_f64_array(f, count=d)
    domain = [[float(domain_lo[i]), float(domain_hi[i])] for i in range(d)]
    for i, (lo, hi) in enumerate(domain):
        if lo >= hi:
            raise ValueError(
                f"domain[{i}]: lo ({lo}) must be < hi ({hi})"
            )

    n_nodes_arr = _read_u32_array(f, count=d)
    n_nodes = [int(n) for n in n_nodes_arr]
    for i, n in enumerate(n_nodes):
        if n < 2:
            raise ValueError(f"n_nodes[{i}] must be >= 2, got {n}")

    total = int(np.prod(n_nodes))
    flat_vals = _read_f64_array(f, count=total)
    tensor_values = flat_vals.reshape(tuple(n_nodes), order="C")

    return ChebyshevApproximation.from_values(
        tensor_values=tensor_values,
        num_dimensions=d,
        domain=domain,
        n_nodes=n_nodes,
    )


# --- ChebyshevSpline -----------------------------------------------------


def _spline_uses_nested_n_nodes(spline) -> bool:
    """True iff any per-dim entry in ``spline.n_nodes`` is itself a list."""
    return any(isinstance(n, (list, tuple)) for n in spline.n_nodes)


def write_spline(f: BinaryIO, spline) -> None:
    """Write a built ``ChebyshevSpline`` to a binary stream.

    Raises RuntimeError if the spline has not been built.
    Raises NotImplementedError if the spline uses nested per-piece n_nodes
    (the binary format requires shared n_nodes across pieces).
    """
    if any(p is None for p in spline._pieces):
        raise RuntimeError("Cannot save an unbuilt ChebyshevSpline")

    if _spline_uses_nested_n_nodes(spline):
        raise NotImplementedError(
            "binary format requires flat n_nodes (shared across pieces); "
            "use format='pickle' for nested-n_nodes splines"
        )

    _write_header(f, CLASS_TAG_SPLINE)

    d = int(spline.num_dimensions)
    _write_u32(f, d)

    domain_lo = np.array([spline.domain[i][0] for i in range(d)], dtype=np.float64)
    domain_hi = np.array([spline.domain[i][1] for i in range(d)], dtype=np.float64)
    _write_f64_array(f, domain_lo)
    _write_f64_array(f, domain_hi)

    n_nodes = np.array(spline.n_nodes, dtype=np.uint32)
    _write_u32_array(f, n_nodes)

    num_knots = np.array(
        [len(spline.knots[i]) for i in range(d)], dtype=np.uint32
    )
    _write_u32_array(f, num_knots)

    knots_concat_parts = []
    for i in range(d):
        if len(spline.knots[i]) > 0:
            knots_concat_parts.append(
                np.asarray(spline.knots[i], dtype=np.float64)
            )
    if knots_concat_parts:
        _write_f64_array(f, np.concatenate(knots_concat_parts))

    num_pieces = len(spline._pieces)
    _write_u32(f, num_pieces)

    for piece in spline._pieces:
        flat = np.ascontiguousarray(
            piece.tensor_values, dtype=np.float64
        ).ravel(order="C")
        _write_f64_array(f, flat)


def read_spline(f: BinaryIO):
    """Read a ``ChebyshevSpline`` from a binary stream.

    Reconstructs via ``ChebyshevSpline.from_values``.
    """
    from pychebyshev import ChebyshevSpline

    tag = _read_header(f)
    if tag != CLASS_TAG_SPLINE:
        raise ValueError(
            f"file contains class_tag {tag}, expected "
            f"{CLASS_TAG_SPLINE} (ChebyshevSpline)"
        )

    d = _read_u32(f)
    if d < 1:
        raise ValueError(f"num_dimensions must be >= 1, got {d}")

    domain_lo = _read_f64_array(f, count=d)
    domain_hi = _read_f64_array(f, count=d)
    domain = [[float(domain_lo[i]), float(domain_hi[i])] for i in range(d)]
    for i, (lo, hi) in enumerate(domain):
        if lo >= hi:
            raise ValueError(f"domain[{i}]: lo ({lo}) must be < hi ({hi})")

    n_nodes_arr = _read_u32_array(f, count=d)
    n_nodes = [int(n) for n in n_nodes_arr]
    for i, n in enumerate(n_nodes):
        if n < 2:
            raise ValueError(f"n_nodes[{i}] must be >= 2, got {n}")

    num_knots_arr = _read_u32_array(f, count=d)
    num_knots = [int(k) for k in num_knots_arr]
    total_knot_floats = sum(num_knots)
    if total_knot_floats > 0:
        flat_knots = _read_f64_array(f, count=total_knot_floats)
    else:
        flat_knots = np.array([], dtype=np.float64)
    knots: list[list[float]] = []
    offset = 0
    for i in range(d):
        k = num_knots[i]
        knots_i = [float(x) for x in flat_knots[offset : offset + k]]
        offset += k
        if k > 1 and any(
            knots_i[j] >= knots_i[j + 1] for j in range(k - 1)
        ):
            raise ValueError(f"knots in dim {i} not strictly ascending")
        knots.append(knots_i)

    num_pieces = _read_u32(f)
    expected_pieces = 1
    for k in num_knots:
        expected_pieces *= k + 1
    if num_pieces != expected_pieces:
        raise ValueError(
            f"num_pieces={num_pieces} does not match prod(num_knots+1)"
            f"={expected_pieces}"
        )

    per_piece_floats = int(np.prod(n_nodes))
    piece_values = []
    for _ in range(num_pieces):
        flat = _read_f64_array(f, count=per_piece_floats)
        piece_values.append(flat.reshape(tuple(n_nodes), order="C"))

    return ChebyshevSpline.from_values(
        piece_values=piece_values,
        num_dimensions=d,
        domain=domain,
        n_nodes=n_nodes,
        knots=knots,
    )
