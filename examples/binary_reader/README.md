# PyChebyshev `.pcb` Binary Reader (C Reference)

A ~240-line C program that reads a v1 `.pcb` file (`ChebyshevApproximation`
class) and evaluates it at a query point. It is the reference proof that
the format is implementable from scratch in another language. It assumes
a little-endian host (matches the on-disk format); a port to big-endian
hardware would need explicit byte-swapping.

`ChebyshevSpline` is **not** supported here — this is a minimal proof,
not a full port.

## Build

```bash
cd examples/binary_reader
make
```

Requires a C99 compiler (gcc, clang) and `libm`. No third-party deps.

## Usage

First, save a `.pcb` file from Python:

```python
from pychebyshev import ChebyshevApproximation
cheb = ChebyshevApproximation(
    function=lambda pt, _: pt[0] + pt[1],
    num_dimensions=2,
    domain=[(-1.0, 1.0), (-1.0, 1.0)],
    n_nodes=[3, 3],
)
cheb.build()
cheb.save("test.pcb", format='binary')
print("Python eval:", cheb.eval([0.3, 0.4], [0, 0]))
```

Then read it from C:

```bash
./reader test.pcb 0.3 0.4
```

The two values should agree to at least 1e-12.

## What it covers

- Header parse + validation (magic, major, class tag, reserved bytes)
- ChebyshevApproximation body (num_dim, domain, n_nodes, tensor)
- Chebyshev first-kind node reconstruction
- Barycentric weight computation
- N-D evaluation by dimension-by-dimension collapse

## What it does not cover

- ChebyshevSpline (pieces, knots) — requires an extra outer routing step
- Derivatives — requires the differentiation matrix
- Future format versions

Contributions adding spline support, or porting to Rust/Julia, are welcome.
