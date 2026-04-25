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


class TestApproxWriteRead:
    @staticmethod
    def _make_simple_approx():
        from pychebyshev import ChebyshevApproximation
        cheb = ChebyshevApproximation(
            function=lambda pt, _: pt[0] + pt[1],
            num_dimensions=2,
            domain=[(-1.0, 1.0), (-1.0, 1.0)],
            n_nodes=[3, 3],
        )
        cheb.build(verbose=False)
        return cheb

    def test_write_then_read_round_trip_in_memory(self):
        cheb = self._make_simple_approx()
        buf = io.BytesIO()
        _binary.write_approx(buf, cheb)
        buf.seek(0)
        loaded = _binary.read_approx(buf)
        assert loaded.num_dimensions == cheb.num_dimensions
        assert [list(b) for b in loaded.domain] == [list(b) for b in cheb.domain]
        assert list(loaded.n_nodes) == list(cheb.n_nodes)
        assert np.array_equal(loaded.tensor_values, cheb.tensor_values)

    def test_loaded_function_attr_is_none(self):
        cheb = self._make_simple_approx()
        buf = io.BytesIO()
        _binary.write_approx(buf, cheb)
        buf.seek(0)
        loaded = _binary.read_approx(buf)
        assert loaded.function is None

    def test_loaded_evaluates_to_same_values(self):
        cheb = self._make_simple_approx()
        buf = io.BytesIO()
        _binary.write_approx(buf, cheb)
        buf.seek(0)
        loaded = _binary.read_approx(buf)
        for x in [-0.5, 0.0, 0.3]:
            for y in [-0.7, 0.1, 0.9]:
                expected = cheb.eval([x, y], [0, 0])
                got = loaded.eval([x, y], [0, 0])
                assert abs(expected - got) < 1e-12

    def test_loaded_supports_derivatives(self):
        cheb = self._make_simple_approx()
        buf = io.BytesIO()
        _binary.write_approx(buf, cheb)
        buf.seek(0)
        loaded = _binary.read_approx(buf)
        # df/dx of x+y is 1
        assert abs(loaded.eval([0.0, 0.0], [1, 0]) - 1.0) < 1e-12
        # df/dy of x+y is 1
        assert abs(loaded.eval([0.0, 0.0], [0, 1]) - 1.0) < 1e-12

    def test_byte_size_matches_spec(self):
        cheb = self._make_simple_approx()
        buf = io.BytesIO()
        _binary.write_approx(buf, cheb)
        # 12 header + 4 num_dim + 16 domain_lo + 16 domain_hi + 8 n_nodes + 72 vals = 128
        assert len(buf.getvalue()) == 128

    def test_read_approx_rejects_spline_class_tag(self):
        # Forge a file with a spline header but pass to read_approx
        buf = io.BytesIO()
        _binary._write_header(buf, _binary.CLASS_TAG_SPLINE)
        buf.write(b"\x00" * 64)  # garbage body
        buf.seek(0)
        with pytest.raises(ValueError, match="expected"):
            _binary.read_approx(buf)

    def test_read_approx_rejects_zero_dimensions(self):
        buf = io.BytesIO()
        _binary._write_header(buf, _binary.CLASS_TAG_APPROX)
        _binary._write_u32(buf, 0)  # num_dimensions = 0
        buf.seek(0)
        with pytest.raises(ValueError, match="num_dimensions must be"):
            _binary.read_approx(buf)

    def test_read_approx_rejects_inverted_domain(self):
        buf = io.BytesIO()
        _binary._write_header(buf, _binary.CLASS_TAG_APPROX)
        _binary._write_u32(buf, 1)
        _binary._write_f64_array(buf, np.array([1.0]))   # lo
        _binary._write_f64_array(buf, np.array([0.0]))   # hi (inverted)
        _binary._write_u32_array(buf, np.array([3], dtype=np.uint32))
        _binary._write_f64_array(buf, np.zeros(3))
        buf.seek(0)
        with pytest.raises(ValueError, match="lo .* must be"):
            _binary.read_approx(buf)

    def test_read_approx_rejects_zero_n_nodes(self):
        buf = io.BytesIO()
        _binary._write_header(buf, _binary.CLASS_TAG_APPROX)
        _binary._write_u32(buf, 1)
        _binary._write_f64_array(buf, np.array([0.0]))
        _binary._write_f64_array(buf, np.array([1.0]))
        _binary._write_u32_array(buf, np.array([0], dtype=np.uint32))  # n=0
        buf.seek(0)
        with pytest.raises(ValueError, match="n_nodes\\[0\\] must be >= 1"):
            _binary.read_approx(buf)

    def test_read_approx_accepts_n_nodes_1(self):
        """A 1-node dim is valid (constant in that dim) — must round-trip."""
        from pychebyshev import ChebyshevApproximation
        info = ChebyshevApproximation.nodes(
            num_dimensions=2, domain=[(0.0, 1.0), (-1.0, 1.0)], n_nodes=[3, 1],
        )
        grid = info["full_grid"]
        vals = (grid[:, 0] ** 2).reshape(info["shape"])
        cheb = ChebyshevApproximation.from_values(
            vals, num_dimensions=2,
            domain=[(0.0, 1.0), (-1.0, 1.0)], n_nodes=[3, 1],
        )
        buf = io.BytesIO()
        _binary.write_approx(buf, cheb)
        buf.seek(0)
        loaded = _binary.read_approx(buf)
        assert list(loaded.n_nodes) == [3, 1]
        assert np.array_equal(loaded.tensor_values, cheb.tensor_values)


class TestSplineWriteRead:
    @staticmethod
    def _make_simple_spline():
        from pychebyshev import ChebyshevSpline
        s = ChebyshevSpline(
            function=lambda pt, _: abs(pt[0]),
            num_dimensions=1,
            domain=[(-1.0, 1.0)],
            n_nodes=[3],
            knots=[[0.0]],
        )
        s.build(verbose=False)
        return s

    def test_write_then_read_round_trip_in_memory(self):
        s = self._make_simple_spline()
        buf = io.BytesIO()
        _binary.write_spline(buf, s)
        buf.seek(0)
        loaded = _binary.read_spline(buf)
        assert loaded.num_dimensions == s.num_dimensions
        assert [list(b) for b in loaded.domain] == [list(b) for b in s.domain]
        assert list(loaded.n_nodes) == list(s.n_nodes)
        assert [list(k) for k in loaded.knots] == [list(k) for k in s.knots]

    def test_loaded_pieces_evaluate_correctly(self):
        s = self._make_simple_spline()
        buf = io.BytesIO()
        _binary.write_spline(buf, s)
        buf.seek(0)
        loaded = _binary.read_spline(buf)
        for x in [-0.7, -0.3, 0.0, 0.4, 0.8]:
            expected = s.eval([x], [0])
            got = loaded.eval([x], [0])
            assert abs(expected - got) < 1e-12

    def test_byte_size_matches_spec(self):
        s = self._make_simple_spline()
        buf = io.BytesIO()
        _binary.write_spline(buf, s)
        # 12 header + 4 num_dim + 8 lo + 8 hi + 4 n_nodes + 4 num_knots
        # + 8 knots + 4 num_pieces + 2*24 piece values = 100
        assert len(buf.getvalue()) == 100

    def test_loaded_function_attr_is_none(self):
        s = self._make_simple_spline()
        buf = io.BytesIO()
        _binary.write_spline(buf, s)
        buf.seek(0)
        loaded = _binary.read_spline(buf)
        assert loaded.function is None

    def test_2d_spline_round_trip(self):
        from pychebyshev import ChebyshevSpline
        s = ChebyshevSpline(
            function=lambda pt, _: abs(pt[0]) + abs(pt[1]),
            num_dimensions=2,
            domain=[(-1.0, 1.0), (-1.0, 1.0)],
            n_nodes=[5, 5],
            knots=[[0.0], [0.0]],
        )
        s.build(verbose=False)
        buf = io.BytesIO()
        _binary.write_spline(buf, s)
        buf.seek(0)
        loaded = _binary.read_spline(buf)
        for pt in [[-0.5, 0.5], [0.3, -0.3], [0.0, 0.0]]:
            assert abs(s.eval(pt, [0, 0]) - loaded.eval(pt, [0, 0])) < 1e-12

    def test_write_spline_rejects_nested_n_nodes(self):
        from pychebyshev import ChebyshevSpline
        s = ChebyshevSpline(
            function=lambda pt, _: abs(pt[0]),
            num_dimensions=1,
            domain=[(-1.0, 1.0)],
            n_nodes=[[3, 5]],   # nested per-piece
            knots=[[0.0]],
        )
        s.build(verbose=False)
        buf = io.BytesIO()
        with pytest.raises(NotImplementedError, match="binary format requires flat n_nodes"):
            _binary.write_spline(buf, s)

    def test_read_spline_rejects_approx_class_tag(self):
        buf = io.BytesIO()
        _binary._write_header(buf, _binary.CLASS_TAG_APPROX)
        buf.write(b"\x00" * 64)
        buf.seek(0)
        with pytest.raises(ValueError, match="expected"):
            _binary.read_spline(buf)

    def test_read_spline_rejects_unsorted_knots(self):
        buf = io.BytesIO()
        _binary._write_header(buf, _binary.CLASS_TAG_SPLINE)
        _binary._write_u32(buf, 1)                                       # d
        _binary._write_f64_array(buf, np.array([-1.0]))                  # lo
        _binary._write_f64_array(buf, np.array([1.0]))                   # hi
        _binary._write_u32_array(buf, np.array([3], dtype=np.uint32))    # n_nodes
        _binary._write_u32_array(buf, np.array([2], dtype=np.uint32))    # num_knots
        _binary._write_f64_array(buf, np.array([0.5, 0.2]))              # knots — unsorted
        _binary._write_u32(buf, 3)                                       # num_pieces
        for _ in range(3):
            _binary._write_f64_array(buf, np.zeros(3))
        buf.seek(0)
        with pytest.raises(ValueError, match="not strictly ascending"):
            _binary.read_spline(buf)


class TestApproxSaveLoadIntegration:
    @staticmethod
    def _make():
        from pychebyshev import ChebyshevApproximation
        cheb = ChebyshevApproximation(
            function=lambda pt, _: pt[0] + pt[1],
            num_dimensions=2,
            domain=[(-1.0, 1.0), (-1.0, 1.0)],
            n_nodes=[3, 3],
        )
        cheb.build(verbose=False)
        return cheb

    def test_save_default_is_pickle(self, tmp_path):
        cheb = self._make()
        path = tmp_path / "default.pkl"
        cheb.save(path)
        # First byte of pickle protocol 2+ stream is 0x80
        assert path.read_bytes()[:1] == b"\x80"

    def test_save_format_binary_writes_magic(self, tmp_path):
        cheb = self._make()
        path = tmp_path / "model.pcb"
        cheb.save(path, format="binary")
        assert path.read_bytes()[:4] == _binary.MAGIC

    def test_save_format_pickle_writes_pickle(self, tmp_path):
        cheb = self._make()
        path = tmp_path / "model.pcb"  # extension is advisory
        cheb.save(path, format="pickle")
        assert path.read_bytes()[:1] == b"\x80"

    def test_save_unknown_format_raises(self, tmp_path):
        cheb = self._make()
        with pytest.raises(ValueError, match="format must be"):
            cheb.save(tmp_path / "x.pcb", format="json")

    def test_load_autodetect_binary(self, tmp_path):
        from pychebyshev import ChebyshevApproximation
        cheb = self._make()
        path = tmp_path / "model.pcb"
        cheb.save(path, format="binary")
        loaded = ChebyshevApproximation.load(path)
        assert loaded.function is None
        assert np.array_equal(loaded.tensor_values, cheb.tensor_values)

    def test_load_autodetect_pickle(self, tmp_path):
        from pychebyshev import ChebyshevApproximation
        cheb = self._make()
        path = tmp_path / "model.pkl"
        cheb.save(path)  # default pickle
        loaded = ChebyshevApproximation.load(path)
        assert np.array_equal(loaded.tensor_values, cheb.tensor_values)

    def test_save_binary_unbuilt_raises(self, tmp_path):
        from pychebyshev import ChebyshevApproximation
        cheb = ChebyshevApproximation(
            function=lambda pt, _: pt[0],
            num_dimensions=1,
            domain=[(0.0, 1.0)],
            n_nodes=[3],
        )
        # NOT built
        with pytest.raises(RuntimeError, match="unbuilt"):
            cheb.save(tmp_path / "x.pcb", format="binary")

    def test_round_trip_eval_matches_machine_precision(self, tmp_path):
        from pychebyshev import ChebyshevApproximation
        cheb = self._make()
        path = tmp_path / "model.pcb"
        cheb.save(path, format="binary")
        loaded = ChebyshevApproximation.load(path)
        for pt in [[-0.5, 0.5], [0.0, 0.0], [0.7, -0.3]]:
            assert abs(cheb.eval(pt, [0, 0]) - loaded.eval(pt, [0, 0])) < 1e-14


class TestSplineSaveLoadIntegration:
    @staticmethod
    def _make():
        from pychebyshev import ChebyshevSpline
        s = ChebyshevSpline(
            function=lambda pt, _: abs(pt[0]),
            num_dimensions=1,
            domain=[(-1.0, 1.0)],
            n_nodes=[3],
            knots=[[0.0]],
        )
        s.build(verbose=False)
        return s

    def test_save_default_is_pickle(self, tmp_path):
        s = self._make()
        path = tmp_path / "default.pkl"
        s.save(path)
        assert path.read_bytes()[:1] == b"\x80"

    def test_save_format_binary_writes_magic(self, tmp_path):
        s = self._make()
        path = tmp_path / "model.pcb"
        s.save(path, format="binary")
        assert path.read_bytes()[:4] == _binary.MAGIC

    def test_save_unknown_format_raises(self, tmp_path):
        s = self._make()
        with pytest.raises(ValueError, match="format must be"):
            s.save(tmp_path / "x.pcb", format="zip")

    def test_load_autodetect_binary(self, tmp_path):
        from pychebyshev import ChebyshevSpline
        s = self._make()
        path = tmp_path / "model.pcb"
        s.save(path, format="binary")
        loaded = ChebyshevSpline.load(path)
        for x in [-0.7, -0.1, 0.3, 0.8]:
            assert abs(s.eval([x], [0]) - loaded.eval([x], [0])) < 1e-12

    def test_load_autodetect_pickle(self, tmp_path):
        from pychebyshev import ChebyshevSpline
        s = self._make()
        path = tmp_path / "model.pkl"
        s.save(path)
        loaded = ChebyshevSpline.load(path)
        for x in [-0.7, -0.1, 0.3, 0.8]:
            assert abs(s.eval([x], [0]) - loaded.eval([x], [0])) < 1e-12

    def test_save_binary_nested_n_nodes_raises(self, tmp_path):
        from pychebyshev import ChebyshevSpline
        s = ChebyshevSpline(
            function=lambda pt, _: abs(pt[0]),
            num_dimensions=1,
            domain=[(-1.0, 1.0)],
            n_nodes=[[3, 5]],   # nested
            knots=[[0.0]],
        )
        s.build(verbose=False)
        with pytest.raises(NotImplementedError, match="binary format requires flat"):
            s.save(tmp_path / "x.pcb", format="binary")


class TestGoldenVectors:
    """Lock down the exact byte layout against accidental drift."""

    def test_approx_2d_xy_exact_bytes(self):
        """f(x,y) = x+y on [-1,1]², n=[3,3] -> exactly 128 bytes."""
        from pychebyshev import ChebyshevApproximation
        cheb = ChebyshevApproximation(
            function=lambda pt, _: pt[0] + pt[1],
            num_dimensions=2,
            domain=[(-1.0, 1.0), (-1.0, 1.0)],
            n_nodes=[3, 3],
        )
        cheb.build(verbose=False)

        buf = io.BytesIO()
        _binary.write_approx(buf, cheb)
        data = buf.getvalue()

        # Header (12)
        assert data[:4] == _binary.MAGIC
        assert data[4] == 1   # major
        assert data[5] == 0   # minor
        assert data[6:8] == struct.pack("<H", _binary.CLASS_TAG_APPROX)
        assert data[8:12] == b"\x00\x00\x00\x00"
        # num_dimensions
        assert struct.unpack_from("<I", data, 12)[0] == 2
        # domain_lo (16 bytes), domain_hi (16 bytes)
        lo = np.frombuffer(data[16:32], dtype="<f8")
        hi = np.frombuffer(data[32:48], dtype="<f8")
        assert np.array_equal(lo, [-1.0, -1.0])
        assert np.array_equal(hi, [1.0, 1.0])
        # n_nodes (8 bytes)
        n = np.frombuffer(data[48:56], dtype="<u4")
        assert np.array_equal(n, [3, 3])
        # tensor_values (72 bytes)
        vals = np.frombuffer(data[56:128], dtype="<f8").reshape(3, 3)
        assert np.allclose(vals, cheb.tensor_values)
        assert len(data) == 128

    def test_approx_round_trip_from_exact_bytes(self):
        """Bytes -> object -> bytes must be identical."""
        from pychebyshev import ChebyshevApproximation
        cheb = ChebyshevApproximation(
            function=lambda pt, _: pt[0] + pt[1],
            num_dimensions=2,
            domain=[(-1.0, 1.0), (-1.0, 1.0)],
            n_nodes=[3, 3],
        )
        cheb.build(verbose=False)
        buf1 = io.BytesIO()
        _binary.write_approx(buf1, cheb)
        bytes1 = buf1.getvalue()

        buf1.seek(0)
        loaded = _binary.read_approx(buf1)

        buf2 = io.BytesIO()
        _binary.write_approx(buf2, loaded)
        assert buf2.getvalue() == bytes1

    def test_spline_1d_abs_exact_size(self):
        """|x| on [-1,1] with knots=[[0.0]], n=[3] -> exactly 100 bytes."""
        from pychebyshev import ChebyshevSpline
        s = ChebyshevSpline(
            function=lambda pt, _: abs(pt[0]),
            num_dimensions=1,
            domain=[(-1.0, 1.0)],
            n_nodes=[3],
            knots=[[0.0]],
        )
        s.build(verbose=False)
        buf = io.BytesIO()
        _binary.write_spline(buf, s)
        data = buf.getvalue()
        assert len(data) == 100
        # Sanity-check the header section
        assert data[:4] == _binary.MAGIC
        assert struct.unpack_from("<H", data, 6)[0] == _binary.CLASS_TAG_SPLINE
        # num_dim, domain_lo, domain_hi
        assert struct.unpack_from("<I", data, 12)[0] == 1
        assert np.frombuffer(data[16:24], dtype="<f8")[0] == -1.0
        assert np.frombuffer(data[24:32], dtype="<f8")[0] == 1.0
        # n_nodes, num_knots
        assert struct.unpack_from("<I", data, 32)[0] == 3
        assert struct.unpack_from("<I", data, 36)[0] == 1
        # knots, num_pieces
        assert np.frombuffer(data[40:48], dtype="<f8")[0] == 0.0
        assert struct.unpack_from("<I", data, 48)[0] == 2

    def test_spline_round_trip_from_exact_bytes(self):
        from pychebyshev import ChebyshevSpline
        s = ChebyshevSpline(
            function=lambda pt, _: abs(pt[0]),
            num_dimensions=1,
            domain=[(-1.0, 1.0)],
            n_nodes=[3],
            knots=[[0.0]],
        )
        s.build(verbose=False)
        buf1 = io.BytesIO()
        _binary.write_spline(buf1, s)
        bytes1 = buf1.getvalue()

        buf1.seek(0)
        loaded = _binary.read_spline(buf1)

        buf2 = io.BytesIO()
        _binary.write_spline(buf2, loaded)
        assert buf2.getvalue() == bytes1


class TestCorruption:
    """Each test simulates a specific failure mode."""

    @staticmethod
    def _make_approx_bytes():
        from pychebyshev import ChebyshevApproximation
        cheb = ChebyshevApproximation(
            function=lambda pt, _: pt[0] + pt[1],
            num_dimensions=2,
            domain=[(-1.0, 1.0), (-1.0, 1.0)],
            n_nodes=[3, 3],
        )
        cheb.build(verbose=False)
        buf = io.BytesIO()
        _binary.write_approx(buf, cheb)
        return buf.getvalue()

    def test_truncated_header(self):
        data = self._make_approx_bytes()
        with pytest.raises(ValueError, match="unexpected EOF"):
            _binary.read_approx(io.BytesIO(data[:8]))

    def test_truncated_at_tensor_values(self):
        data = self._make_approx_bytes()
        # Cut off in the middle of tensor_values
        with pytest.raises(ValueError, match="unexpected EOF"):
            _binary.read_approx(io.BytesIO(data[:64]))

    def test_wrong_magic(self):
        data = b"WRNG" + self._make_approx_bytes()[4:]
        with pytest.raises(ValueError, match="not a PyChebyshev binary file"):
            _binary.read_approx(io.BytesIO(data))

    def test_future_major_version(self):
        data = self._make_approx_bytes()
        forged = data[:4] + bytes([99]) + data[5:]
        with pytest.raises(ValueError, match="unsupported .pcb major version 99"):
            _binary.read_approx(io.BytesIO(forged))

    def test_nonzero_reserved(self):
        data = self._make_approx_bytes()
        forged = data[:8] + b"\xff\x00\x00\x00" + data[12:]
        with pytest.raises(ValueError, match="reserved header bytes nonzero"):
            _binary.read_approx(io.BytesIO(forged))

    def test_class_tag_mismatch_approx_to_spline(self):
        data = self._make_approx_bytes()
        # Flip class tag from APPROX (1) to SPLINE (2)
        forged = data[:6] + struct.pack("<H", _binary.CLASS_TAG_SPLINE) + data[8:]
        with pytest.raises(ValueError, match="expected"):
            _binary.read_approx(io.BytesIO(forged))

    def test_inverted_domain(self):
        # Build a fresh forged file with lo > hi
        buf = io.BytesIO()
        _binary._write_header(buf, _binary.CLASS_TAG_APPROX)
        _binary._write_u32(buf, 1)
        _binary._write_f64_array(buf, np.array([2.0]))   # lo
        _binary._write_f64_array(buf, np.array([1.0]))   # hi (inverted)
        _binary._write_u32_array(buf, np.array([3], dtype=np.uint32))
        _binary._write_f64_array(buf, np.zeros(3))
        buf.seek(0)
        with pytest.raises(ValueError, match="must be < hi"):
            _binary.read_approx(buf)

    def test_zero_n_nodes(self):
        buf = io.BytesIO()
        _binary._write_header(buf, _binary.CLASS_TAG_APPROX)
        _binary._write_u32(buf, 1)
        _binary._write_f64_array(buf, np.array([0.0]))
        _binary._write_f64_array(buf, np.array([1.0]))
        _binary._write_u32_array(buf, np.array([0], dtype=np.uint32))
        buf.seek(0)
        with pytest.raises(ValueError, match="n_nodes\\[0\\] must be >= 1"):
            _binary.read_approx(buf)

    def test_unknown_format_kwarg_raises(self, tmp_path):
        from pychebyshev import ChebyshevApproximation
        cheb = ChebyshevApproximation(
            function=lambda pt, _: pt[0],
            num_dimensions=1,
            domain=[(0.0, 1.0)],
            n_nodes=[3],
        )
        cheb.build(verbose=False)
        with pytest.raises(ValueError, match="format must be"):
            cheb.save(tmp_path / "x.pcb", format="zip")

    def test_unbuilt_save_binary_raises(self, tmp_path):
        from pychebyshev import ChebyshevApproximation
        cheb = ChebyshevApproximation(
            function=lambda pt, _: pt[0],
            num_dimensions=1,
            domain=[(0.0, 1.0)],
            n_nodes=[3],
        )
        with pytest.raises(RuntimeError, match="unbuilt"):
            cheb.save(tmp_path / "x.pcb", format="binary")


class TestCrossFeature:
    """Binary format must work with from_values, algebra, extrude, slice."""

    def test_from_values_built_round_trips(self, tmp_path):
        from pychebyshev import ChebyshevApproximation
        info = ChebyshevApproximation.nodes(
            num_dimensions=2,
            domain=[(0.0, 1.0), (0.0, 1.0)],
            n_nodes=[5, 5],
        )
        grid = info["full_grid"]
        vals = (grid[:, 0] * grid[:, 1]).reshape(info["shape"])
        cheb = ChebyshevApproximation.from_values(
            vals, num_dimensions=2,
            domain=[(0.0, 1.0), (0.0, 1.0)],
            n_nodes=[5, 5],
        )
        path = tmp_path / "fv.pcb"
        cheb.save(path, format="binary")
        loaded = ChebyshevApproximation.load(path)
        assert loaded.function is None
        for pt in [[0.3, 0.4], [0.7, 0.2]]:
            assert abs(cheb.eval(pt, [0, 0]) - loaded.eval(pt, [0, 0])) < 1e-12

    def test_algebra_derived_round_trips(self, tmp_path):
        from pychebyshev import ChebyshevApproximation
        f = ChebyshevApproximation(
            function=lambda pt, _: pt[0],
            num_dimensions=1, domain=[(-1.0, 1.0)], n_nodes=[5],
        )
        f.build(verbose=False)
        g = ChebyshevApproximation(
            function=lambda pt, _: pt[0]**2,
            num_dimensions=1, domain=[(-1.0, 1.0)], n_nodes=[5],
        )
        g.build(verbose=False)
        h = f + g  # algebra-derived; function is None
        path = tmp_path / "h.pcb"
        h.save(path, format="binary")
        loaded = ChebyshevApproximation.load(path)
        for x in [-0.5, 0.0, 0.4]:
            assert abs(h.eval([x], [0]) - loaded.eval([x], [0])) < 1e-12

    def test_extruded_round_trips(self, tmp_path):
        from pychebyshev import ChebyshevApproximation
        f = ChebyshevApproximation(
            function=lambda pt, _: pt[0],
            num_dimensions=1, domain=[(-1.0, 1.0)], n_nodes=[5],
        )
        f.build(verbose=False)
        f2 = f.extrude((1, (-1.0, 1.0), 4))
        path = tmp_path / "extr.pcb"
        f2.save(path, format="binary")
        loaded = ChebyshevApproximation.load(path)
        assert loaded.num_dimensions == 2
        assert abs(f2.eval([0.3, 0.5], [0, 0]) - loaded.eval([0.3, 0.5], [0, 0])) < 1e-12

    def test_sliced_round_trips(self, tmp_path):
        from pychebyshev import ChebyshevApproximation
        f = ChebyshevApproximation(
            function=lambda pt, _: pt[0] + 2 * pt[1],
            num_dimensions=2, domain=[(-1.0, 1.0), (-1.0, 1.0)], n_nodes=[4, 4],
        )
        f.build(verbose=False)
        f1 = f.slice((1, 0.0))
        path = tmp_path / "sl.pcb"
        f1.save(path, format="binary")
        loaded = ChebyshevApproximation.load(path)
        assert loaded.num_dimensions == 1
        assert abs(f1.eval([0.5], [0]) - loaded.eval([0.5], [0])) < 1e-12

    def test_spline_error_estimate_recomputed_after_load(self, tmp_path):
        from pychebyshev import ChebyshevSpline
        s = ChebyshevSpline(
            function=lambda pt, _: abs(pt[0]),
            num_dimensions=1,
            domain=[(-1.0, 1.0)],
            n_nodes=[7],
            knots=[[0.0]],
        )
        s.build(verbose=False)
        original_err = s.error_estimate()
        path = tmp_path / "s.pcb"
        s.save(path, format="binary")
        loaded = ChebyshevSpline.load(path)
        assert abs(loaded.error_estimate() - original_err) < 1e-14

    def test_5d_black_scholes_round_trips(self, cheb_bs_5d):
        """Use existing fixture — large dim, real numerical content."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bs.pcb")
            cheb_bs_5d.save(path, format="binary")
            from pychebyshev import ChebyshevApproximation
            loaded = ChebyshevApproximation.load(path)
            assert np.array_equal(loaded.tensor_values, cheb_bs_5d.tensor_values)
            pt = [100.0, 0.05, 1.0, 0.2, 100.0]
            assert abs(cheb_bs_5d.eval(pt, [0]*5) - loaded.eval(pt, [0]*5)) < 1e-10

    def test_n_nodes_minimum_round_trips(self, tmp_path):
        from pychebyshev import ChebyshevApproximation
        cheb = ChebyshevApproximation(
            function=lambda pt, _: pt[0] * pt[1],
            num_dimensions=2,
            domain=[(0.0, 1.0), (0.0, 1.0)],
            n_nodes=[2, 2],
        )
        cheb.build(verbose=False)
        path = tmp_path / "tiny.pcb"
        cheb.save(path, format="binary")
        loaded = ChebyshevApproximation.load(path)
        assert np.array_equal(loaded.tensor_values, cheb.tensor_values)
