using Test
using PCBReader

const FIXTURES_DIR = joinpath(@__DIR__, "..", "..", "..", "tests", "fixtures")

# ---------------------------------------------------------------------------
# Helpers: build minimal valid .pcb bytes in memory
# ---------------------------------------------------------------------------

const MAGIC = UInt8[0x50, 0x43, 0x42, 0x00]  # b"PCB\x00"

function make_approx_bytes(d::Int, los::Vector{Float64}, his::Vector{Float64},
                           n_nodes::Vector{Int}, values::Vector{Float64})
    buf = IOBuffer()
    write(buf, MAGIC)
    write(buf, UInt8(1))   # major
    write(buf, UInt8(0))   # minor
    write(buf, htol(UInt16(1)))   # class_tag = Approximation
    write(buf, zeros(UInt8, 4))   # reserved
    write(buf, htol(UInt32(d)))
    for v in los;    write(buf, htol(v)); end
    for v in his;    write(buf, htol(v)); end
    for n in n_nodes; write(buf, htol(UInt32(n))); end
    for v in values;  write(buf, htol(v)); end
    return take!(buf)
end

function make_spline_bytes(d::Int, los::Vector{Float64}, his::Vector{Float64},
                           n_nodes::Vector{Int},
                           knots::Vector{Vector{Float64}},
                           pieces::Vector{Vector{Float64}})
    buf = IOBuffer()
    write(buf, MAGIC)
    write(buf, UInt8(1))   # major
    write(buf, UInt8(0))   # minor
    write(buf, htol(UInt16(2)))   # class_tag = Spline
    write(buf, zeros(UInt8, 4))   # reserved
    write(buf, htol(UInt32(d)))
    for v in los;    write(buf, htol(v)); end
    for v in his;    write(buf, htol(v)); end
    for n in n_nodes; write(buf, htol(UInt32(n))); end
    num_knots = [length(k) for k in knots]
    for nk in num_knots; write(buf, htol(UInt32(nk))); end
    for k in knots, v in k; write(buf, htol(v)); end
    write(buf, htol(UInt32(length(pieces))))
    for p in pieces, v in p; write(buf, htol(v)); end
    return take!(buf)
end

# ---------------------------------------------------------------------------
# Unit tests: error paths
# ---------------------------------------------------------------------------

@testset "PCBReader.jl" begin

    @testset "Error: truncated file" begin
        bytes = UInt8[0x50, 0x43, 0x42, 0x00]  # too short
        @test_throws PCBReader.PcbError PCBReader.read_pcb_from_bytes(bytes)
    end

    @testset "Error: bad magic" begin
        bytes = make_approx_bytes(1, [0.0], [1.0], [2], [1.0, 2.0])
        bytes[1] = 0x58  # 'X'
        @test_throws PCBReader.PcbError PCBReader.read_pcb_from_bytes(bytes)
    end

    @testset "Error: unsupported major version" begin
        bytes = make_approx_bytes(1, [0.0], [1.0], [2], [1.0, 2.0])
        bytes[5] = 0x63  # major = 99
        @test_throws PCBReader.PcbError PCBReader.read_pcb_from_bytes(bytes)
    end

    @testset "Error: nonzero reserved bytes" begin
        bytes = make_approx_bytes(1, [0.0], [1.0], [2], [1.0, 2.0])
        bytes[9] = 0xFF
        @test_throws PCBReader.PcbError PCBReader.read_pcb_from_bytes(bytes)
    end

    @testset "Error: unknown class_tag" begin
        bytes = make_approx_bytes(1, [0.0], [1.0], [2], [1.0, 2.0])
        bytes[7] = 0x63  # class_tag = 99
        bytes[8] = 0x00
        @test_throws PCBReader.PcbError PCBReader.read_pcb_from_bytes(bytes)
    end

    @testset "Error: domain lo >= hi" begin
        bytes = make_approx_bytes(1, [1.0], [0.0], [2], [1.0, 2.0])  # lo > hi
        @test_throws PCBReader.PcbError PCBReader.read_pcb_from_bytes(bytes)
    end

    # -----------------------------------------------------------------------
    # Approximation round-trips (in-memory)
    # -----------------------------------------------------------------------

    @testset "Approximation: 1D round-trip" begin
        bytes = make_approx_bytes(1, [-1.0], [1.0], [3], [0.0, 1.0, 0.5])
        interp = PCBReader.read_pcb_from_bytes(bytes)
        @test interp isa PCBReader.Approximation
        @test interp.num_dimensions == 1
        @test interp.domain == [(-1.0, 1.0)]
        @test interp.n_nodes == [3]
        @test interp.tensor_values == [0.0, 1.0, 0.5]
        @test interp.format_major == 0x01
        @test interp.format_minor == 0x00
    end

    @testset "Approximation: 2D round-trip" begin
        values = collect(0.0:0.1:0.5)  # 6 values
        bytes = make_approx_bytes(2, [0.0, -2.0], [1.0, 2.0], [2, 3], values)
        interp = PCBReader.read_pcb_from_bytes(bytes)
        @test interp isa PCBReader.Approximation
        @test interp.num_dimensions == 2
        @test interp.domain == [(0.0, 1.0), (-2.0, 2.0)]
        @test interp.n_nodes == [2, 3]
        @test length(interp.tensor_values) == 6
        @test abs(interp.tensor_values[1] - 0.0) < 1e-15
        @test abs(interp.tensor_values[6] - 0.5) < 1e-15
    end

    # -----------------------------------------------------------------------
    # Spline round-trips (in-memory)
    # -----------------------------------------------------------------------

    @testset "Spline: 1D round-trip" begin
        # 1 knot at 0.0 → 2 pieces, each with 3 values
        bytes = make_spline_bytes(
            1, [0.0], [2.0],
            [3],
            [[1.0]],          # knots: dim 1 has 1 knot
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # 2 pieces
        )
        interp = PCBReader.read_pcb_from_bytes(bytes)
        @test interp isa PCBReader.Spline
        @test interp.num_dimensions == 1
        @test interp.domain == [(0.0, 2.0)]
        @test interp.n_nodes == [3]
        @test length(interp.knots) == 1
        @test interp.knots[1] == [1.0]
        @test length(interp.pieces) == 2
        @test interp.pieces[1] == [0.1, 0.2, 0.3]
        @test interp.pieces[2] == [0.4, 0.5, 0.6]
    end

    @testset "Spline: 2D round-trip" begin
        # dim1: 1 knot → 2 intervals; dim2: 0 knots → 1 interval; total pieces = 2
        pieces = [[Float64(i + j) for i in 1:4] for j in 0:1]  # 2 pieces, 4 vals each
        bytes = make_spline_bytes(
            2, [-1.0, -1.0], [1.0, 1.0],
            [2, 2],
            [[0.0], Float64[]],  # dim1: 1 knot, dim2: 0 knots
            pieces
        )
        interp = PCBReader.read_pcb_from_bytes(bytes)
        @test interp isa PCBReader.Spline
        @test interp.num_dimensions == 2
        @test interp.n_nodes == [2, 2]
        @test length(interp.knots) == 2
        @test length(interp.knots[1]) == 1
        @test length(interp.knots[2]) == 0
        @test length(interp.pieces) == 2
        @test length(interp.pieces[1]) == 4
    end

    @testset "Spline: invalid num_pieces rejected" begin
        # Claim 5 pieces but num_knots implies prod([1+1]) = 2
        buf = IOBuffer()
        write(buf, MAGIC)
        write(buf, UInt8(1)); write(buf, UInt8(0))
        write(buf, htol(UInt16(2)))  # Spline
        write(buf, zeros(UInt8, 4))
        write(buf, htol(UInt32(1)))  # d=1
        write(buf, htol(0.0)); write(buf, htol(1.0))  # domain
        write(buf, htol(UInt32(3)))  # n_nodes
        write(buf, htol(UInt32(1)))  # num_knots[1] = 1
        write(buf, htol(0.5))        # flat_knots
        write(buf, htol(UInt32(5)))  # num_pieces = 5 (wrong: should be 2)
        bytes = take!(buf)
        @test_throws PCBReader.PcbError PCBReader.read_pcb_from_bytes(bytes)
    end

    # -----------------------------------------------------------------------
    # Fixture-based tests (real .pcb files from Python)
    # -----------------------------------------------------------------------

    @testset "approx_2d_simple.pcb (fixture)" begin
        path = joinpath(FIXTURES_DIR, "approx_2d_simple.pcb")
        interp = read_pcb(path)
        @test interp isa PCBReader.Approximation
        @test interp.num_dimensions == 2
        @test interp.n_nodes == [4, 4]
        @test length(interp.tensor_values) == 16
        # Domain sanity
        for (lo, hi) in interp.domain
            @test lo < hi
        end
    end

    @testset "approx_5d_bs.pcb (fixture)" begin
        path = joinpath(FIXTURES_DIR, "approx_5d_bs.pcb")
        interp = read_pcb(path)
        @test interp isa PCBReader.Approximation
        @test interp.num_dimensions == 5
        @test interp.n_nodes == [6, 6, 6, 6, 6]
        @test length(interp.tensor_values) == 6^5
    end

    @testset "spline_1d_kink.pcb (fixture)" begin
        path = joinpath(FIXTURES_DIR, "spline_1d_kink.pcb")
        interp = read_pcb(path)
        @test interp isa PCBReader.Spline
        @test interp.num_dimensions == 1
        @test length(interp.knots) == 1
        @test length(interp.knots[1]) >= 1
        @test length(interp.pieces) == length(interp.knots[1]) + 1
        # Each piece has the right number of values
        per_piece = prod(interp.n_nodes)
        for p in interp.pieces
            @test length(p) == per_piece
        end
    end

end
