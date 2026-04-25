# Portable Binary Format (`.pcb`)

PyChebyshev v0.14 introduced a portable binary serialization format alongside
the default pickle format. The goal: let consumers in **C, Rust, Julia, or
any other language** read PyChebyshev interpolants without a Python runtime.

The format is intentionally minimal — a fixed header, length-prefixed
sections, raw little-endian `f64` blobs. A C reference reader at
`examples/binary_reader/` weighs ~240 lines.

## When to use which format

| Format | Use when |
|---|---|
| **Pickle** (default) | Python-only round-trips; need full fidelity (build metadata, error caches) |
| **Binary** (`.pcb`) | Cross-language consumers; sharing models with C/Rust/Julia code; long-term archival without Python pickle compatibility risk |

Pickle stays the default because every existing user keeps working with no
change. Opt into binary explicitly:

```python
cheb.save("model.pcb", format='binary')      # portable
cheb.save("model.pkl")                       # pickle (default)

ChebyshevApproximation.load("model.pcb")     # auto-detects
```

`load()` sniffs the first 4 bytes — `b"PCB\x00"` routes to the binary
reader, anything else to the pickle reader.

## Coverage in v0.14

- **`ChebyshevApproximation`** — full support.
- **`ChebyshevSpline`** — full support, with one restriction: the spline
  must use **flat** `n_nodes` (a single `int` per dim, shared across pieces).
  Splines built with nested per-piece `n_nodes` (the `[[n00, n01], …]` form
  introduced for special points) cannot be saved as `.pcb` because the
  underlying `ChebyshevSpline.from_values()` factory does not yet support
  that shape; use pickle for those.
- **`ChebyshevSlider`**, **`ChebyshevTT`** — pickle only in v0.14.

## Format specification (v1)

All multi-byte fields are **little-endian**. Numeric arrays are raw `f64`
blobs in C-order.

### Header (12 bytes)

```
offset  size  field
0       4     magic           = b"PCB\x00"
4       1     major_version   = 1
5       1     minor_version   = 0
6       2     class_tag       1 = ChebyshevApproximation, 2 = ChebyshevSpline
8       4     reserved        = 0x00000000
```

### `ChebyshevApproximation` body (`class_tag = 1`)

```
uint32     num_dimensions                            d
f64[d]     domain_lo                                 [a_0, ..., a_{d-1}]
f64[d]     domain_hi                                 [b_0, ..., b_{d-1}]
uint32[d]  n_nodes                                   [n_0, ..., n_{d-1}]
f64[prod(n_nodes)]  tensor_values                    C-order
```

`barycentric_weights` and `diff_matrices` are **not** stored; they are
recomputed from `(domain, n_nodes)` on load (they are pure functions of
those primitives).

### `ChebyshevSpline` body (`class_tag = 2`)

```
uint32     num_dimensions                            d
f64[d]     domain_lo
f64[d]     domain_hi
uint32[d]  n_nodes                                   shared across pieces
uint32[d]  num_knots_per_dim                         [k_0, ..., k_{d-1}]
f64[k_0 + ... + k_{d-1}]   knots_concatenated        flat, dim-by-dim
uint32     num_pieces                                P = prod(k_i + 1)

# P piece blocks, in C-order over the piece grid:
for p in 0..P-1:
    f64[prod(n_nodes)]     tensor_values_p           C-order
```

### Versioning policy

- New required fields → bump **major**. v1 readers reject `major != 1`.
- New optional trailing fields → bump **minor**. v1 readers ignore unknown
  trailing data.
- Reserved header bytes MUST be zero in v1.

## Worked example: `f(x,y) = x + y`

Python side:

```python
from pychebyshev import ChebyshevApproximation

cheb = ChebyshevApproximation(
    function=lambda pt, _: pt[0] + pt[1],
    num_dimensions=2,
    domain=[(-1.0, 1.0), (-1.0, 1.0)],
    n_nodes=[3, 3],
)
cheb.build()
cheb.save("xy.pcb", format='binary')
```

The resulting file is exactly **128 bytes**:

```
12  header
 4  num_dimensions = 2
16  domain_lo  = [-1.0, -1.0]
16  domain_hi  = [ 1.0,  1.0]
 8  n_nodes    = [3, 3]
72  tensor_values (3 × 3 f64)
```

C reader:

```bash
cd examples/binary_reader
make
./reader ../../xy.pcb 0.3 0.4
# 0.69999999999999996
```

The same IEEE-754 double Python returns (`repr` truncates trailing digits):

```python
cheb.eval([0.3, 0.4], [0, 0])  # 0.7
```

The two strings render the same `float64` value `0x3fe6666666666666`. The
C reader prints with `%.17g`, Python with `repr` — they agree bit-for-bit.

### Spline worked example: `|x|` on `[-1, 1]`

```python
from pychebyshev import ChebyshevSpline

s = ChebyshevSpline(
    function=lambda pt, _: abs(pt[0]),
    num_dimensions=1,
    domain=[(-1.0, 1.0)],
    n_nodes=[3],
    knots=[[0.0]],
)
s.build()
s.save("abs.pcb", format='binary')
```

The resulting file is exactly **100 bytes**:

```
12  header
 4  num_dimensions = 1
 8  domain_lo  = [-1.0]
 8  domain_hi  = [ 1.0]
 4  n_nodes    = [3]
 4  num_knots  = [1]
 8  knots      = [0.0]
 4  num_pieces = 2
48  piece tensor values (2 pieces × 3 × f64)
```

Two pieces because one knot at `0.0` splits the domain `[-1, 1]` into `[-1, 0]`
and `[0, 1]`. Each piece carries its own 3-node Chebyshev grid.

## Writing a reader in another language

The format is small enough to implement in an afternoon:

1. Read 4 bytes; verify equal to `b"PCB\x00"`.
2. Read major/minor version; reject unknown major.
3. Read class tag; dispatch.
4. For class 1: read `uint32 d`, then `d × f64` for `lo`, `d × f64` for `hi`,
   `d × uint32` for `n_nodes`, then `prod(n_nodes) × f64` for tensor values.
5. To evaluate, generate Chebyshev first-kind nodes per dim, compute
   barycentric weights from node positions, evaluate by dim-by-dim collapse.

`examples/binary_reader/reader.c` is the reference. It is intentionally
minimal: ~240 lines, stdlib + `libm` only.

## What the format does not store

These fields are dropped on `format='binary'`:

| Field | Replacement |
|---|---|
| `function` | always dropped (also dropped by pickle) |
| `barycentric_weights`, `diff_matrices` | recomputed on load |
| `_cached_error_estimate` | recomputed lazily |
| `build_time`, `n_evaluations`, `method` | not preserved (use pickle for full fidelity) |
| `max_derivative_order` | resets to default `2` on load — re-set manually after `load()` if you need higher orders |
| `additional_data` | binary save raises `NotImplementedError` if non-`None`; pass `format='pickle'` instead |
| `descriptor` | silently dropped on binary save; restored as `""` on load — pickle preserves |
| Derivative-id registry | not stored; reset to empty on binary load — pickle preserves |

If you need any of those preserved, use pickle.

## Security

The binary reader does no `pickle.loads`-style code execution. It can be
used to load files from untrusted sources — it will reject malformed
files with a `ValueError`.

Pickle remains the default and **does** execute arbitrary code from
loaded files. Treat pickle files like executable code.
