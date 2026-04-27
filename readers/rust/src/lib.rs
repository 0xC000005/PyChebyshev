//! Reference reader for the PyChebyshev `.pcb` portable binary format.
//!
//! # Format overview
//!
//! Every `.pcb` file starts with a 12-byte header:
//!
//! ```text
//! bytes 0–3   magic       b"PCB\x00"
//! byte  4     major       u8   (currently 1)
//! byte  5     minor       u8   (currently 0)
//! bytes 6–7   class_tag   u16 LE  (1 = Approximation, 2 = Spline)
//! bytes 8–11  reserved    [u8;4] = 0x00000000
//! ```
//!
//! The body layout depends on `class_tag`. All integers are little-endian
//! `uint32`; all floats are little-endian `f64`. No padding.
//!
//! ## Approximation body
//! ```text
//! d               uint32
//! domain_lo[d]    f64 × d    (lower bounds, all dims first)
//! domain_hi[d]    f64 × d    (upper bounds, all dims)
//! n_nodes[d]      uint32 × d
//! tensor_values   f64 × prod(n_nodes)   C-order (row-major)
//! ```
//!
//! ## Spline body
//! ```text
//! d               uint32
//! domain_lo[d]    f64 × d
//! domain_hi[d]    f64 × d
//! n_nodes[d]      uint32 × d
//! num_knots[d]    uint32 × d
//! flat_knots      f64 × sum(num_knots)
//! num_pieces      uint32  (must equal prod(num_knots[i]+1))
//! pieces          num_pieces × (f64 × prod(n_nodes))   each C-order
//! ```

use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

// --- Format constants -------------------------------------------------------

const MAGIC: [u8; 4] = [b'P', b'C', b'B', 0];
const HEADER_SIZE: usize = 12;
const CLASS_TAG_APPROX: u16 = 1;
const CLASS_TAG_SPLINE: u16 = 2;

// --- Public types -----------------------------------------------------------

/// A parsed `ChebyshevApproximation` from a `.pcb` file.
#[derive(Debug, Clone)]
pub struct Approximation {
    pub format_major: u8,
    pub format_minor: u8,
    pub num_dimensions: u32,
    /// `domain[i] = (lo, hi)` for dimension `i`.
    pub domain: Vec<(f64, f64)>,
    /// Number of Chebyshev nodes per dimension.
    pub n_nodes: Vec<u32>,
    /// Function values at the tensor-product grid, flattened in C order.
    pub tensor_values: Vec<f64>,
}

/// A parsed `ChebyshevSpline` from a `.pcb` file.
#[derive(Debug, Clone)]
pub struct Spline {
    pub format_major: u8,
    pub format_minor: u8,
    pub num_dimensions: u32,
    /// `domain[i] = (lo, hi)` for dimension `i`.
    pub domain: Vec<(f64, f64)>,
    /// Number of Chebyshev nodes per dimension (shared across all pieces).
    pub n_nodes: Vec<u32>,
    /// Knot coordinates per dimension.
    pub knots: Vec<Vec<f64>>,
    /// Function values for each piece (C-order flat), length == `num_pieces`.
    pub pieces: Vec<Vec<f64>>,
}

/// The top-level result of reading a `.pcb` file.
#[derive(Debug, Clone)]
pub enum PcbInterpolant {
    Approximation(Approximation),
    Spline(Spline),
}

// --- Error type -------------------------------------------------------------

/// Errors that can occur while reading a `.pcb` file.
#[derive(Debug)]
pub enum PcbError {
    /// An I/O error reading or opening the file.
    Io(std::io::Error),
    /// The file's first four bytes do not match `b"PCB\x00"`.
    InvalidMagic([u8; 4]),
    /// The reserved header bytes (8–11) are nonzero — the file may be corrupt.
    NonzeroReserved([u8; 4]),
    /// The format major version is not 1.
    UnsupportedVersion { major: u8, minor: u8 },
    /// The class_tag byte does not correspond to a known interpolant class.
    UnknownClassTag(u16),
    /// The file is shorter than the minimum required length.
    Truncated,
    /// A field value is out of its valid range (e.g. `lo >= hi`).
    InvalidField(String),
}

impl std::fmt::Display for PcbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PcbError::Io(e) => write!(f, "I/O error: {e}"),
            PcbError::InvalidMagic(m) => {
                write!(f, "not a .pcb file (magic {m:?}, expected {:?})", MAGIC)
            }
            PcbError::NonzeroReserved(r) => {
                write!(f, "reserved header bytes nonzero ({r:?}) — file may be corrupt")
            }
            PcbError::UnsupportedVersion { major, minor } => {
                write!(f, "unsupported .pcb version {major}.{minor} (this reader supports major 1)")
            }
            PcbError::UnknownClassTag(tag) => write!(f, "unknown class_tag {tag}"),
            PcbError::Truncated => write!(f, "unexpected end of file"),
            PcbError::InvalidField(msg) => write!(f, "invalid field: {msg}"),
        }
    }
}

impl std::error::Error for PcbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let PcbError::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<std::io::Error> for PcbError {
    fn from(e: std::io::Error) -> Self {
        PcbError::Io(e)
    }
}

// --- Public API -------------------------------------------------------------

/// Read a `.pcb` file from the given path.
///
/// Returns a [`PcbInterpolant`] (either [`PcbInterpolant::Approximation`] or
/// [`PcbInterpolant::Spline`]) on success.
///
/// # Errors
///
/// Returns [`PcbError`] if the file cannot be read, does not have a valid
/// `.pcb` header, or contains malformed body data.
pub fn read_pcb<P: AsRef<Path>>(path: P) -> Result<PcbInterpolant, PcbError> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    read_pcb_from_bytes(&bytes)
}

/// Parse a `.pcb` file from a byte slice.
///
/// Same semantics as [`read_pcb`] but operates on an in-memory buffer.
pub fn read_pcb_from_bytes(bytes: &[u8]) -> Result<PcbInterpolant, PcbError> {
    if bytes.len() < HEADER_SIZE {
        return Err(PcbError::Truncated);
    }

    // --- Header ---
    let mut magic = [0u8; 4];
    magic.copy_from_slice(&bytes[0..4]);
    if magic != MAGIC {
        return Err(PcbError::InvalidMagic(magic));
    }

    let major = bytes[4];
    let minor = bytes[5];
    if major != 1 {
        return Err(PcbError::UnsupportedVersion { major, minor });
    }

    let class_tag = u16::from_le_bytes([bytes[6], bytes[7]]);

    let mut reserved = [0u8; 4];
    reserved.copy_from_slice(&bytes[8..12]);
    if reserved != [0u8; 4] {
        return Err(PcbError::NonzeroReserved(reserved));
    }

    // --- Body ---
    let mut cur = Cursor::new(&bytes[HEADER_SIZE..]);

    match class_tag {
        CLASS_TAG_APPROX => {
            let interp = parse_approx_body(&mut cur, major, minor)?;
            Ok(PcbInterpolant::Approximation(interp))
        }
        CLASS_TAG_SPLINE => {
            let interp = parse_spline_body(&mut cur, major, minor)?;
            Ok(PcbInterpolant::Spline(interp))
        }
        _ => Err(PcbError::UnknownClassTag(class_tag)),
    }
}

// --- Body parsers -----------------------------------------------------------

fn parse_approx_body(
    cur: &mut Cursor<&[u8]>,
    major: u8,
    minor: u8,
) -> Result<Approximation, PcbError> {
    let d = read_u32(cur)? as usize;
    if d < 1 {
        return Err(PcbError::InvalidField(format!(
            "num_dimensions must be >= 1, got {d}"
        )));
    }

    let domain = read_domain(cur, d)?;
    let n_nodes = read_u32_vec(cur, d)?;

    let total: usize = n_nodes.iter().map(|&n| n as usize).product();
    let tensor_values = read_f64_vec(cur, total)?;

    Ok(Approximation {
        format_major: major,
        format_minor: minor,
        num_dimensions: d as u32,
        domain,
        n_nodes,
        tensor_values,
    })
}

fn parse_spline_body(
    cur: &mut Cursor<&[u8]>,
    major: u8,
    minor: u8,
) -> Result<Spline, PcbError> {
    let d = read_u32(cur)? as usize;
    if d < 1 {
        return Err(PcbError::InvalidField(format!(
            "num_dimensions must be >= 1, got {d}"
        )));
    }

    let domain = read_domain(cur, d)?;
    let n_nodes = read_u32_vec(cur, d)?;
    let num_knots = read_u32_vec(cur, d)?;

    let total_knots: usize = num_knots.iter().map(|&k| k as usize).sum();
    let flat_knots = read_f64_vec(cur, total_knots)?;

    // Partition flat_knots back into per-dimension vecs
    let mut knots: Vec<Vec<f64>> = Vec::with_capacity(d);
    let mut offset = 0usize;
    for &k in &num_knots {
        let k = k as usize;
        knots.push(flat_knots[offset..offset + k].to_vec());
        offset += k;
    }

    let num_pieces = read_u32(cur)? as usize;
    // Validate: must equal prod(num_knots[i]+1)
    let expected_pieces: usize = num_knots.iter().map(|&k| k as usize + 1).product();
    if num_pieces != expected_pieces {
        return Err(PcbError::InvalidField(format!(
            "num_pieces={num_pieces} does not match prod(num_knots[i]+1)={expected_pieces}"
        )));
    }

    let per_piece: usize = n_nodes.iter().map(|&n| n as usize).product();
    let mut pieces: Vec<Vec<f64>> = Vec::with_capacity(num_pieces);
    for _ in 0..num_pieces {
        pieces.push(read_f64_vec(cur, per_piece)?);
    }

    Ok(Spline {
        format_major: major,
        format_minor: minor,
        num_dimensions: d as u32,
        domain,
        n_nodes,
        knots,
        pieces,
    })
}

// --- Low-level helpers ------------------------------------------------------

/// Read a single little-endian uint32.
fn read_u32(cur: &mut Cursor<&[u8]>) -> Result<u32, PcbError> {
    cur.read_u32::<LittleEndian>()
        .map_err(|_| PcbError::Truncated)
}

/// Read `count` consecutive little-endian uint32 values.
fn read_u32_vec(cur: &mut Cursor<&[u8]>, count: usize) -> Result<Vec<u32>, PcbError> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(read_u32(cur)?);
    }
    Ok(v)
}

/// Read `count` consecutive little-endian f64 values.
fn read_f64_vec(cur: &mut Cursor<&[u8]>, count: usize) -> Result<Vec<f64>, PcbError> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        let x = cur
            .read_f64::<LittleEndian>()
            .map_err(|_| PcbError::Truncated)?;
        v.push(x);
    }
    Ok(v)
}

/// Read the domain section: `d` lo values then `d` hi values, return as `(lo, hi)` pairs.
///
/// The Python writer emits `domain_lo[d]` followed by `domain_hi[d]` as two
/// separate contiguous arrays — NOT interleaved pairs.
fn read_domain(cur: &mut Cursor<&[u8]>, d: usize) -> Result<Vec<(f64, f64)>, PcbError> {
    let lo_vals = read_f64_vec(cur, d)?;
    let hi_vals = read_f64_vec(cur, d)?;

    let mut domain = Vec::with_capacity(d);
    for i in 0..d {
        let lo = lo_vals[i];
        let hi = hi_vals[i];
        if lo >= hi {
            return Err(PcbError::InvalidField(format!(
                "domain[{i}]: lo ({lo}) must be < hi ({hi})"
            )));
        }
        domain.push((lo, hi));
    }
    Ok(domain)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a minimal valid Approximation header + body in memory
    fn make_approx_bytes(d: u32, los: &[f64], his: &[f64], n_nodes: &[u32], values: &[f64]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.push(1u8); // major
        buf.push(0u8); // minor
        buf.extend_from_slice(&1u16.to_le_bytes()); // class_tag = Approximation
        buf.extend_from_slice(&[0u8; 4]); // reserved
        buf.extend_from_slice(&d.to_le_bytes());
        for &v in los { buf.extend_from_slice(&v.to_le_bytes()); }
        for &v in his { buf.extend_from_slice(&v.to_le_bytes()); }
        for &n in n_nodes { buf.extend_from_slice(&n.to_le_bytes()); }
        for &v in values { buf.extend_from_slice(&v.to_le_bytes()); }
        buf
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = make_approx_bytes(1, &[0.0], &[1.0], &[2], &[1.0, 2.0]);
        bytes[0] = b'X'; // corrupt magic
        let err = read_pcb_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, PcbError::InvalidMagic(_)));
    }

    #[test]
    fn unsupported_major_rejected() {
        let mut bytes = make_approx_bytes(1, &[0.0], &[1.0], &[2], &[1.0, 2.0]);
        bytes[4] = 99; // bad major
        let err = read_pcb_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, PcbError::UnsupportedVersion { .. }));
    }

    #[test]
    fn nonzero_reserved_rejected() {
        let mut bytes = make_approx_bytes(1, &[0.0], &[1.0], &[2], &[1.0, 2.0]);
        bytes[8] = 0xFF; // corrupt reserved
        let err = read_pcb_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, PcbError::NonzeroReserved(_)));
    }

    #[test]
    fn unknown_class_tag_rejected() {
        let mut bytes = make_approx_bytes(1, &[0.0], &[1.0], &[2], &[1.0, 2.0]);
        bytes[6] = 99;
        bytes[7] = 0;
        let err = read_pcb_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, PcbError::UnknownClassTag(99)));
    }

    #[test]
    fn truncated_rejected() {
        let bytes = &[b'P', b'C', b'B', 0u8];
        let err = read_pcb_from_bytes(bytes).unwrap_err();
        assert!(matches!(err, PcbError::Truncated));
    }

    #[test]
    fn approx_1d_roundtrip() {
        let bytes = make_approx_bytes(1, &[-1.0], &[1.0], &[3], &[0.0, 1.0, 0.5]);
        let interp = read_pcb_from_bytes(&bytes).unwrap();
        let PcbInterpolant::Approximation(a) = interp else {
            panic!("expected Approximation");
        };
        assert_eq!(a.num_dimensions, 1);
        assert_eq!(a.domain, vec![(-1.0, 1.0)]);
        assert_eq!(a.n_nodes, vec![3]);
        assert_eq!(a.tensor_values, vec![0.0, 1.0, 0.5]);
    }

    #[test]
    fn approx_2d_roundtrip() {
        let values: Vec<f64> = (0..6).map(|i| i as f64 * 0.1).collect();
        let bytes = make_approx_bytes(2, &[0.0, -2.0], &[1.0, 2.0], &[2, 3], &values);
        let interp = read_pcb_from_bytes(&bytes).unwrap();
        let PcbInterpolant::Approximation(a) = interp else {
            panic!("expected Approximation");
        };
        assert_eq!(a.num_dimensions, 2);
        assert_eq!(a.domain, vec![(0.0, 1.0), (-2.0, 2.0)]);
        assert_eq!(a.n_nodes, vec![2, 3]);
        assert_eq!(a.tensor_values.len(), 6);
        assert!((a.tensor_values[0] - 0.0).abs() < 1e-15);
        assert!((a.tensor_values[5] - 0.5).abs() < 1e-15);
    }
}
