# PCBReader.jl

Reference Julia reader for the PyChebyshev `.pcb` portable binary format.

## Usage

```julia
using PCBReader

interp = read_pcb("model.pcb")
if interp isa PCBReader.Approximation
    println("$(interp.num_dimensions) dimensions, $(prod(interp.n_nodes)) grid points")
elseif interp isa PCBReader.Spline
    println("$(interp.num_dimensions) dimensions, $(length(interp.pieces)) pieces")
end
```

## Format spec

See `../../docs/user-guide/binary-format.md`.

## Testing

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'
```

## Reference

Test fixtures live in `../../tests/fixtures/`. Generate via the Python script
in the parent repo: `uv run python scripts/generate_test_fixtures.py`.
