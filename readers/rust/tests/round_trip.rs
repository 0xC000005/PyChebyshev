//! Integration tests: read pre-generated `.pcb` fixtures written by the
//! Python `pychebyshev` library and assert structural + value correctness.
//!
//! Fixtures live at `../../tests/fixtures/` relative to this crate's root
//! (`readers/rust/`). Values were extracted once from Python and are
//! hardcoded here so endianness / ordering regressions are caught.

use pcb_reader::{read_pcb, PcbInterpolant};
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    // CARGO_MANIFEST_DIR = readers/rust/
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../../tests/fixtures");
    path.push(name);
    path
}

// ---------------------------------------------------------------------------
// approx_2d_simple.pcb  — 2D, n_nodes=[4,4], 184 bytes
// ---------------------------------------------------------------------------

#[test]
fn approx_2d_simple_loads() {
    let interp = read_pcb(fixture_path("approx_2d_simple.pcb")).expect("read failed");
    let PcbInterpolant::Approximation(a) = interp else {
        panic!("expected Approximation, got Spline");
    };
    assert_eq!(a.format_major, 1);
    assert_eq!(a.num_dimensions, 2);
    assert_eq!(a.domain, vec![(-1.0, 1.0), (-1.0, 1.0)]);
    assert_eq!(a.n_nodes, vec![4, 4]);
    assert_eq!(a.tensor_values.len(), 16); // 4 × 4

    // Value extracted from Python: first == last == 0.8535533905932737
    let expected = 0.853_553_390_593_273_7_f64;
    assert!(
        (a.tensor_values[0] - expected).abs() < 1e-12,
        "first value mismatch: got {}, expected {}",
        a.tensor_values[0],
        expected
    );
    assert!(
        (a.tensor_values[15] - expected).abs() < 1e-12,
        "last value mismatch: got {}, expected {}",
        a.tensor_values[15],
        expected
    );
}

// ---------------------------------------------------------------------------
// approx_5d_bs.pcb  — 5D Black-Scholes, n_nodes=[6,6,6,6,6], 62324 bytes
// ---------------------------------------------------------------------------

#[test]
fn approx_5d_bs_loads() {
    let interp = read_pcb(fixture_path("approx_5d_bs.pcb")).expect("read failed");
    let PcbInterpolant::Approximation(a) = interp else {
        panic!("expected Approximation, got Spline");
    };
    assert_eq!(a.format_major, 1);
    assert_eq!(a.num_dimensions, 5);
    assert_eq!(a.domain, vec![
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]);
    assert_eq!(a.n_nodes, vec![6, 6, 6, 6, 6]);
    assert_eq!(a.tensor_values.len(), 6_usize.pow(5)); // 7776

    // Values extracted from Python
    let expected_first = 1.612_105_216_532_309_f64;
    let expected_last  = 3.257_256_706_805_558_f64;
    assert!(
        (a.tensor_values[0] - expected_first).abs() < 1e-9,
        "first value mismatch: got {}, expected {}",
        a.tensor_values[0],
        expected_first
    );
    assert!(
        (a.tensor_values[7775] - expected_last).abs() < 1e-9,
        "last value mismatch: got {}, expected {}",
        a.tensor_values[7775],
        expected_last
    );
}

// ---------------------------------------------------------------------------
// spline_1d_kink.pcb  — 1D Spline, n_nodes=[8], 1 knot at 0.0, 2 pieces, 180 bytes
// ---------------------------------------------------------------------------

#[test]
fn spline_1d_kink_loads() {
    let interp = read_pcb(fixture_path("spline_1d_kink.pcb")).expect("read failed");
    let PcbInterpolant::Spline(s) = interp else {
        panic!("expected Spline, got Approximation");
    };
    assert_eq!(s.format_major, 1);
    assert_eq!(s.num_dimensions, 1);
    assert_eq!(s.domain, vec![(-1.0, 1.0)]);
    assert_eq!(s.n_nodes, vec![8]);
    assert_eq!(s.knots.len(), 1);
    assert_eq!(s.knots[0], vec![0.0_f64]); // single knot at 0.0
    assert_eq!(s.pieces.len(), 2); // 1 knot → 2 pieces
    assert_eq!(s.pieces[0].len(), 8);
    assert_eq!(s.pieces[1].len(), 8);

    // Values extracted from Python
    // piece 0: first=0.9903926402, last=0.0096073598
    // piece 1: first=0.0096073598, last=0.9903926402
    let p0_first = 0.990_392_640_2_f64;
    let p0_last  = 0.009_607_359_8_f64;
    let p1_first = 0.009_607_359_8_f64;
    let p1_last  = 0.990_392_640_2_f64;

    assert!(
        (s.pieces[0][0] - p0_first).abs() < 1e-8,
        "piece0[0] mismatch: got {}, expected {}",
        s.pieces[0][0],
        p0_first
    );
    assert!(
        (s.pieces[0][7] - p0_last).abs() < 1e-8,
        "piece0[7] mismatch: got {}, expected {}",
        s.pieces[0][7],
        p0_last
    );
    assert!(
        (s.pieces[1][0] - p1_first).abs() < 1e-8,
        "piece1[0] mismatch: got {}, expected {}",
        s.pieces[1][0],
        p1_first
    );
    assert!(
        (s.pieces[1][7] - p1_last).abs() < 1e-8,
        "piece1[7] mismatch: got {}, expected {}",
        s.pieces[1][7],
        p1_last
    );
}
