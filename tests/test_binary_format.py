"""Tests for the .pcb binary serialization format (v0.14)."""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from pychebyshev import _binary


class TestLowLevelHelpers:
    def test_magic_constant(self):
        assert _binary.MAGIC == b"PCB\x00"
        assert len(_binary.MAGIC) == 4

    def test_version_constants(self):
        assert _binary.MAJOR == 1
        assert _binary.MINOR == 0

    def test_class_tags(self):
        assert _binary.CLASS_TAG_APPROX == 1
        assert _binary.CLASS_TAG_SPLINE == 2

    def test_write_read_u32_round_trip(self):
        buf = io.BytesIO()
        _binary._write_u32(buf, 0)
        _binary._write_u32(buf, 1)
        _binary._write_u32(buf, 2**32 - 1)
        buf.seek(0)
        assert _binary._read_u32(buf) == 0
        assert _binary._read_u32(buf) == 1
        assert _binary._read_u32(buf) == 2**32 - 1

    def test_write_read_u32_array_round_trip(self):
        buf = io.BytesIO()
        arr = np.array([3, 5, 7, 11], dtype=np.uint32)
        _binary._write_u32_array(buf, arr)
        buf.seek(0)
        out = _binary._read_u32_array(buf, count=4)
        assert out.dtype == np.uint32
        assert np.array_equal(out, arr)

    def test_write_read_f64_array_round_trip(self):
        buf = io.BytesIO()
        arr = np.array([1.5, -2.25, np.pi, 1e-300, 1e300], dtype=np.float64)
        _binary._write_f64_array(buf, arr)
        buf.seek(0)
        out = _binary._read_f64_array(buf, count=5)
        assert out.dtype == np.float64
        assert np.array_equal(out, arr)

    def test_write_f64_array_rejects_non_f64(self):
        buf = io.BytesIO()
        arr = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises((TypeError, ValueError)):
            _binary._write_f64_array(buf, arr)

    def test_read_u32_truncated_raises(self):
        buf = io.BytesIO(b"\x01\x02")  # only 2 bytes
        with pytest.raises(ValueError, match="unexpected EOF"):
            _binary._read_u32(buf)

    def test_read_f64_array_truncated_raises(self):
        buf = io.BytesIO(b"\x00" * 7)  # less than one f64
        with pytest.raises(ValueError, match="unexpected EOF"):
            _binary._read_f64_array(buf, count=1)

    def test_read_u32_array_truncated_raises(self):
        buf = io.BytesIO(b"\x01\x02\x03")  # 3 bytes, less than one u32
        with pytest.raises(ValueError, match="unexpected EOF"):
            _binary._read_u32_array(buf, count=1)

    def test_write_u32_array_rejects_non_u32(self):
        buf = io.BytesIO()
        arr = np.array([1, 2, 3], dtype=np.int64)
        with pytest.raises(TypeError, match="uint32"):
            _binary._write_u32_array(buf, arr)


class TestHeader:
    def test_write_read_header_round_trip_approx(self):
        buf = io.BytesIO()
        _binary._write_header(buf, _binary.CLASS_TAG_APPROX)
        buf.seek(0)
        tag = _binary._read_header(buf)
        assert tag == _binary.CLASS_TAG_APPROX

    def test_write_read_header_round_trip_spline(self):
        buf = io.BytesIO()
        _binary._write_header(buf, _binary.CLASS_TAG_SPLINE)
        buf.seek(0)
        tag = _binary._read_header(buf)
        assert tag == _binary.CLASS_TAG_SPLINE

    def test_header_size_is_12_bytes(self):
        buf = io.BytesIO()
        _binary._write_header(buf, _binary.CLASS_TAG_APPROX)
        assert len(buf.getvalue()) == 12

    def test_header_starts_with_magic(self):
        buf = io.BytesIO()
        _binary._write_header(buf, _binary.CLASS_TAG_APPROX)
        assert buf.getvalue()[:4] == _binary.MAGIC

    def test_read_header_rejects_bad_magic(self):
        buf = io.BytesIO(b"XXXX" + b"\x01\x00\x01\x00\x00\x00\x00\x00")
        with pytest.raises(ValueError, match="not a PyChebyshev binary file"):
            _binary._read_header(buf)

    def test_read_header_rejects_future_major_version(self):
        # major=99 in byte 4
        bad = _binary.MAGIC + bytes([99, 0]) + bytes([1, 0]) + b"\x00\x00\x00\x00"
        with pytest.raises(ValueError, match="unsupported .pcb major version 99"):
            _binary._read_header(io.BytesIO(bad))

    def test_read_header_rejects_nonzero_reserved(self):
        bad = (
            _binary.MAGIC
            + bytes([1, 0])
            + bytes([1, 0])
            + b"\xff\x00\x00\x00"  # reserved nonzero
        )
        with pytest.raises(ValueError, match="reserved header bytes nonzero"):
            _binary._read_header(io.BytesIO(bad))

    def test_read_header_accepts_higher_minor(self):
        # major=1 minor=99 -> still acceptable
        ok = (
            _binary.MAGIC
            + bytes([1, 99])
            + bytes([1, 0])
            + b"\x00\x00\x00\x00"
        )
        tag = _binary._read_header(io.BytesIO(ok))
        assert tag == _binary.CLASS_TAG_APPROX


class TestDetectFormat:
    def test_detect_binary_from_magic(self, tmp_path):
        path = tmp_path / "x.pcb"
        path.write_bytes(_binary.MAGIC + b"\x00" * 8)
        assert _binary.detect_format(path) == "binary"

    def test_detect_pickle_from_pickle_stream(self, tmp_path):
        import pickle
        path = tmp_path / "x.pkl"
        with open(path, "wb") as f:
            pickle.dump({"hello": 1}, f)
        assert _binary.detect_format(path) == "pickle"

    def test_detect_pickle_for_short_or_empty_file(self, tmp_path):
        path = tmp_path / "tiny.bin"
        path.write_bytes(b"\x00\x01")
        assert _binary.detect_format(path) == "pickle"

    def test_detect_pickle_for_random_garbage(self, tmp_path):
        path = tmp_path / "junk.dat"
        path.write_bytes(b"GIF89a..." + b"\x00" * 100)
        assert _binary.detect_format(path) == "pickle"
