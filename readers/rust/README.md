# pcb_reader — Rust reference reader for `.pcb` files

A standalone Rust crate that parses the PyChebyshev portable binary format
(`.pcb`). Not part of the `pychebyshev` PyPI package.

## Format

The `.pcb` format uses fixed little-endian byte order, `f64` for all floats,
and `uint32` for all integers. Full specification:
[`docs/user-guide/binary-format.md`](../../docs/user-guide/binary-format.md).

### Header (12 bytes)

| Bytes | Field       | Type   |
|-------|-------------|--------|
| 0–3   | magic       | `[u8;4]` = `PCB\x00` |
| 4     | format_major | `u8` (currently `1`) |
| 5     | format_minor | `u8` (currently `0`) |
| 6–7   | class_tag   | `uint16 LE` (`1` = Approximation, `2` = Spline) |
| 8–11  | reserved    | `[u8;4]` = `\x00\x00\x00\x00` |

### Approximation body (after header)

```
d             : uint32
domain_lo[d]  : f64 × d   (lower bounds, all dims)
domain_hi[d]  : f64 × d   (upper bounds, all dims)
n_nodes[d]    : uint32 × d
tensor_values : f64 × prod(n_nodes), C-order (row-major)
```

### Spline body (after header)

```
d             : uint32
domain_lo[d]  : f64 × d
domain_hi[d]  : f64 × d
n_nodes[d]    : uint32 × d
num_knots[d]  : uint32 × d
flat_knots    : f64 × sum(num_knots)
num_pieces    : uint32 (== prod(num_knots[i]+1))
pieces        : num_pieces × (f64 × prod(n_nodes)), each C-order
```

## Usage

```rust
use pcb_reader::{read_pcb, PcbInterpolant};

let interp = read_pcb("model.pcb").expect("read failed");
match interp {
    PcbInterpolant::Approximation(a) => {
        println!("dims={} n_nodes={:?}", a.num_dimensions, a.n_nodes);
        println!("first value={}", a.tensor_values[0]);
    }
    PcbInterpolant::Spline(s) => {
        println!("dims={} pieces={}", s.num_dimensions, s.pieces.len());
    }
}
```

## Building

```bash
cargo build
```

## Testing

Tests read fixtures from `../../tests/fixtures/`. Run from this directory:

```bash
cargo test
```
