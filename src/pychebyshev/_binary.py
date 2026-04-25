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
