# Cross-language readers for `.pcb` files

These directories contain reference implementations of `.pcb` file readers
in languages other than Python. They are NOT part of the `pychebyshev`
PyPI package — each is a standalone project.

## Available readers

- [`rust/`](rust/) — Rust crate `pcb_reader`. Built with `cargo`.
- [`julia/`](julia/) — Julia package `PCBReader`. Built with `Pkg`.

## Format spec

The `.pcb` binary format is documented at
[`docs/user-guide/binary-format.md`](../docs/user-guide/binary-format.md).

## Testing

Each reader has its own test suite that reads pre-generated `.pcb`
fixtures from `../tests/fixtures/`. To run all readers' tests:

```bash
cd rust && cargo test
cd ../julia && julia --project=. -e 'using Pkg; Pkg.test()'
```
