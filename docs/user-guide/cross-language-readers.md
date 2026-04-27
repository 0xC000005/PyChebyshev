# Cross-Language `.pcb` Readers (v0.20)

PyChebyshev's `.pcb` portable binary format (v0.14) is now joined by
**reference reader implementations** in Rust and Julia. These let you
consume PyChebyshev models from non-Python production systems.

## Available readers

- **Rust** (`readers/rust/`) — `pcb_reader` crate
- **Julia** (`readers/julia/`) — `PCBReader.jl` package

Both parse the same `.pcb` files; both return a struct with the
deserialized interpolant data (tensor values, weights, diff matrices,
domain bounds, n_nodes per dimension). Neither implements *evaluation*
— that's left to the consumer using the spec.

## Format spec

See [Binary Format](binary-format.md). The format is stable; pre-v0.20
`.pcb` files load unchanged.

## Use cases

- **Latency-critical pricing** — embed a PyChebyshev pricer in a Rust
  trading system without Python startup overhead
- **Embedded inference** — deploy on edge devices where Python isn't
  an option
- **Scientific computing** — consume PyChebyshev models in Julia
  ecosystems (DifferentialEquations.jl, etc.)
- **Model versioning** — `.pcb` decouples model storage from runtime language,
  enabling polyglot toolchains

## Rust example

```rust
use pcb_reader::{read_pcb, ClassTag};

let interp = read_pcb("model.pcb")?;
match interp.tag {
    ClassTag::ChebyshevApproximation => {
        println!("{} dims, {} tensor entries",
                 interp.num_dimensions, interp.tensor_values.len());
    },
    ClassTag::ChebyshevSpline => {
        println!("Spline with {} pieces", interp.num_pieces);
    },
}
```

## Julia example

```julia
using PCBReader

interp = read_pcb("model.pcb")
println("$(interp.num_dimensions) dims")
```

Evaluate using your own BLAS.jl backend or spectral methods in the Julia
ecosystem:

```julia
# Custom barycentric evaluation in Julia
f_eval(x) = barycentric_eval(x, interp.tensor_values, interp.barycentric_weights)
```

## Building & testing readers

### Rust

```bash
cd readers/rust/
cargo build --release
cargo test
```

### Julia

```julia
# In Julia REPL
] dev readers/julia/PCBReader.jl
# Precompile and run tests
] test PCBReader
```

## Shared test fixtures

Both readers are tested against golden `.pcb` files in `tests/fixtures/`:

```
tests/fixtures/
  ├── cheb_1d.pcb           # 1-D ChebyshevApproximation
  ├── cheb_3d.pcb           # 3-D ChebyshevApproximation
  ├── spline_1d.pcb         # 1-D ChebyshevSpline
  └── spline_2d.pcb         # 2-D ChebyshevSpline with multiple pieces
```

Regenerate golden fixtures to test compatibility across reader versions:

```bash
uv run python scripts/generate_test_fixtures.py
```

Then each reader's test suite validates deserialization against the same fixtures.

## Reader specifications

### Rust `pcb_reader`

- **Parsing** — stdlib `std::io::Read` + `byteorder`
- **Validation** — header magic, version, dimension bounds, tensor shape
- **Output** — typed `struct Interpolant { tag, num_dimensions, tensor_values, ... }`
- **Errors** — `anyhow::Result` with context
- **Deps** — `byteorder` only (no numpy, scipy, or heavy deps)

### Julia `PCBReader.jl`

- **Parsing** — stdlib `read()` and `Base.open()`
- **Validation** — magic number, version, shape consistency
- **Output** — named tuple or `struct Interpolant`
- **Errors** — `ErrorException` with clear messages
- **Deps** — stdlib only

## Comparison with Python

The Python reader (`pychebyshev._binary`) does validation + full `.pcb` round-trip
(save/load). Rust and Julia readers are **deserialization-only** and do not require
matching pychebyshev versions, making them forward-compatible with future formats
(as long as the header is stable).

!!! tip "Version alignment"
    Readers track the `.pcb` format version in their `READERS.md` documentation.
    v0.20 uses format version 2 (stable since v0.14).

## Typical integration pattern

1. **In Python:** Train and save model
   ```python
   from pychebyshev import ChebyshevApproximation
   cheb = ChebyshevApproximation(f, 3, domain, n_nodes=10).build()
   cheb.save("model.pcb")
   ```

2. **In Rust:** Load and evaluate
   ```rust
   use pcb_reader::read_pcb;
   let interp = read_pcb("model.pcb")?;
   let value = evaluate(&interp, &[s, k, t])?;
   ```

3. **In Julia:** Load and differentiate
   ```julia
   using PCBReader, ForwardDiff
   interp = read_pcb("model.pcb")
   grad = ForwardDiff.gradient(x -> evaluate(x, interp), [s, k, t])
   ```

## Future directions

- **C header** — C `struct` definition and `read_pcb.h` for C/C++ consumers
- **WebAssembly** — `.pcb` reader compiled to WASM for browser-based inference
- **Format v3** — planned enhancements (lossless compression, metadata versioning)

## See Also

- [Binary Format](binary-format.md) — `.pcb` specification and Python round-trip
- [Saving & Loading](serialization.md) — Python save/load workflow

## References

- PyChebyshev GitHub: [readers/](https://github.com/0xC000005/PyChebyshev/tree/main/readers)
- `.pcb` format spec: `docs/user-guide/binary-format.md`
