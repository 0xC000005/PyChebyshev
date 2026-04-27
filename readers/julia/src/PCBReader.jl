"""
    PCBReader

Reference Julia reader for the PyChebyshev `.pcb` portable binary format.

# Format overview

Every `.pcb` file starts with a 12-byte header:

    bytes 0–3   magic       b"PCB\\x00"
    byte  4     major       u8   (currently 1)
    byte  5     minor       u8   (currently 0)
    bytes 6–7   class_tag   u16 LE  (1 = Approximation, 2 = Spline)
    bytes 8–11  reserved    [u8;4] = 0x00000000

The body layout depends on `class_tag`. All integers are little-endian uint32;
all floats are little-endian f64. No padding.

## Approximation body

    d               uint32
    domain_lo[d]    f64 × d    (lower bounds, all dims first)
    domain_hi[d]    f64 × d    (upper bounds, all dims)
    n_nodes[d]      uint32 × d
    tensor_values   f64 × prod(n_nodes)   C-order (row-major)

## Spline body

    d               uint32
    domain_lo[d]    f64 × d
    domain_hi[d]    f64 × d
    n_nodes[d]      uint32 × d
    num_knots[d]    uint32 × d
    flat_knots      f64 × sum(num_knots)
    num_pieces      uint32  (must equal prod(num_knots[i]+1))
    pieces          num_pieces × (f64 × prod(n_nodes))   each C-order
"""
module PCBReader

export read_pcb, read_pcb_from_bytes, PcbError, Approximation, Spline, AbstractPcbInterpolant

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

const MAGIC = UInt8[0x50, 0x43, 0x42, 0x00]  # b"PCB\x00"
const HEADER_SIZE = 12
const CLASS_TAG_APPROX = UInt16(1)
const CLASS_TAG_SPLINE = UInt16(2)

# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

"""
    PcbError <: Exception

Thrown when a `.pcb` file cannot be parsed.
"""
struct PcbError <: Exception
    msg::String
end

Base.showerror(io::IO, e::PcbError) = print(io, "PcbError: ", e.msg)

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

"""
Abstract supertype for all parsed `.pcb` interpolants.
"""
abstract type AbstractPcbInterpolant end

"""
    Approximation

A parsed `ChebyshevApproximation` from a `.pcb` file.

Fields mirror the Rust reader (`readers/rust/src/lib.rs`) exactly.
`tensor_values` is kept as a flat `Vector{Float64}` in C-order (row-major)
to preserve byte-exact fidelity with the file. Julia is column-major, so do
not reshape into an N-D array without also permuting the axis order.
"""
struct Approximation <: AbstractPcbInterpolant
    format_major::UInt8
    format_minor::UInt8
    num_dimensions::Int
    "domain[i] = (lo, hi) for dimension i."
    domain::Vector{Tuple{Float64,Float64}}
    "Number of Chebyshev nodes per dimension."
    n_nodes::Vector{Int}
    "Function values at the tensor-product grid, flattened in C order."
    tensor_values::Vector{Float64}
end

"""
    Spline

A parsed `ChebyshevSpline` from a `.pcb` file.

Fields mirror the Rust reader (`readers/rust/src/lib.rs`) exactly.
Each element of `pieces` is a flat `Vector{Float64}` (C-order) of length
`prod(n_nodes)`.
"""
struct Spline <: AbstractPcbInterpolant
    format_major::UInt8
    format_minor::UInt8
    num_dimensions::Int
    "domain[i] = (lo, hi) for dimension i."
    domain::Vector{Tuple{Float64,Float64}}
    "Number of Chebyshev nodes per dimension (shared across all pieces)."
    n_nodes::Vector{Int}
    "Knot coordinates per dimension."
    knots::Vector{Vector{Float64}}
    "Function values for each piece (C-order flat), length == num_pieces."
    pieces::Vector{Vector{Float64}}
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    read_pcb(path) -> AbstractPcbInterpolant

Read a `.pcb` file from the given path and return an [`Approximation`] or
[`Spline`].

Throws [`PcbError`] if the file cannot be read or contains malformed data.
"""
function read_pcb(path::AbstractString)
    bytes = Base.read(path)
    return read_pcb_from_bytes(bytes)
end

"""
    read_pcb_from_bytes(bytes) -> AbstractPcbInterpolant

Parse a `.pcb` file from an in-memory byte vector.

Same semantics as [`read_pcb`] but operates on a `Vector{UInt8}`.
"""
function read_pcb_from_bytes(bytes::Vector{UInt8})
    if length(bytes) < HEADER_SIZE
        throw(PcbError("file is shorter than header ($(length(bytes)) bytes)"))
    end

    # --- Header (12 bytes) --------------------------------------------------
    if bytes[1:4] != MAGIC
        throw(PcbError("invalid .pcb magic: $(bytes[1:4]) (expected $MAGIC)"))
    end

    major = bytes[5]
    minor = bytes[6]
    if major != 0x01
        throw(PcbError("unsupported .pcb version $major.$minor (reader supports major 1)"))
    end

    # Little-endian u16: assemble from bytes to avoid endian ambiguity.
    class_tag = UInt16(bytes[7]) | (UInt16(bytes[8]) << 8)

    reserved = bytes[9:12]
    if any(reserved .!= 0x00)
        throw(PcbError("reserved header bytes are nonzero ($reserved) — file may be corrupt"))
    end

    # --- Body (use IOBuffer for clean sequential reads) ---------------------
    io = IOBuffer(bytes)
    skip(io, HEADER_SIZE)

    if class_tag == CLASS_TAG_APPROX
        return _parse_approx_body(io, major, minor)
    elseif class_tag == CLASS_TAG_SPLINE
        return _parse_spline_body(io, major, minor)
    else
        throw(PcbError("unknown class_tag: $class_tag"))
    end
end

# ---------------------------------------------------------------------------
# Body parsers
# ---------------------------------------------------------------------------

function _parse_approx_body(io::IO, major::UInt8, minor::UInt8)
    d = _read_u32(io)
    d >= 1 || throw(PcbError("num_dimensions must be >= 1, got $d"))

    domain  = _read_domain(io, d)
    n_nodes = [_read_u32(io) for _ in 1:d]

    total = prod(n_nodes)
    tensor_values = [_read_f64(io) for _ in 1:total]

    return Approximation(major, minor, d, domain, n_nodes, tensor_values)
end

function _parse_spline_body(io::IO, major::UInt8, minor::UInt8)
    d = _read_u32(io)
    d >= 1 || throw(PcbError("num_dimensions must be >= 1, got $d"))

    domain    = _read_domain(io, d)
    n_nodes   = [_read_u32(io) for _ in 1:d]
    num_knots = [_read_u32(io) for _ in 1:d]

    total_knots = sum(num_knots)
    flat_knots  = [_read_f64(io) for _ in 1:total_knots]

    # Partition flat_knots back into per-dimension vectors
    knots = Vector{Vector{Float64}}(undef, d)
    offset = 1
    for dim in 1:d
        k = num_knots[dim]
        knots[dim] = flat_knots[offset:offset+k-1]
        offset += k
    end

    num_pieces = _read_u32(io)
    expected_pieces = prod(num_knots .+ 1)
    if num_pieces != expected_pieces
        throw(PcbError(
            "num_pieces=$num_pieces does not match prod(num_knots[i]+1)=$expected_pieces"
        ))
    end

    per_piece = prod(n_nodes)
    pieces = Vector{Vector{Float64}}(undef, num_pieces)
    for p in 1:num_pieces
        pieces[p] = [_read_f64(io) for _ in 1:per_piece]
    end

    return Spline(major, minor, d, domain, n_nodes, knots, pieces)
end

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

"""Read a single little-endian UInt32 from `io`."""
function _read_u32(io::IO)
    try
        return Int(ltoh(read(io, UInt32)))
    catch e
        throw(PcbError("unexpected end of file while reading uint32"))
    end
end

"""Read a single little-endian Float64 from `io`."""
function _read_f64(io::IO)
    try
        return ltoh(read(io, Float64))
    catch e
        throw(PcbError("unexpected end of file while reading f64"))
    end
end

"""
Read domain section: d lo values then d hi values, return as (lo, hi) pairs.

The Python writer emits domain_lo[d] followed by domain_hi[d] as two separate
contiguous arrays — NOT interleaved pairs.
"""
function _read_domain(io::IO, d::Int)
    lo_vals = [_read_f64(io) for _ in 1:d]
    hi_vals = [_read_f64(io) for _ in 1:d]
    domain  = Vector{Tuple{Float64,Float64}}(undef, d)
    for i in 1:d
        lo = lo_vals[i]
        hi = hi_vals[i]
        lo < hi || throw(PcbError("domain[$i]: lo ($lo) must be < hi ($hi)"))
        domain[i] = (lo, hi)
    end
    return domain
end

end # module PCBReader
