// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Zero-allocation binary scanner for MNode records.
//!
//! Scans raw MNode bytes and evaluates predicate conditions directly against
//! binary data without materializing [`MNode`] or [`MValue`] objects. Combined
//! with schema-compiled predicates, this eliminates all heap allocation in the
//! per-record hot path.
//!
//! ## Design
//!
//! 1. **Memoize** — predicates are flattened into field-indexed conditions
//!    (once, at startup).
//! 2. **Discover schema** — the first record's field layout is captured as a
//!    [`RecordSchema`].
//! 3. **Compile** — memoized conditions are mapped to schema field positions,
//!    producing a [`CompiledScanPredicates`] that uses positional dispatch
//!    instead of per-field name hashing.
//! 4. **Scan** — [`scan_record`] walks raw MNode bytes, skipping non-targeted
//!    fields and comparing targeted fields directly against comparand values
//!    using [`check_condition_raw`].

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

use crate::mnode::DIALECT_MNODE;
use crate::pnode::{Comparand, ConjugateType, FieldRef, OpType, PNode};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from binary MNode scanning.
#[derive(Debug)]
pub enum ScanError {
    /// Data buffer ended before the expected structure was complete.
    UnexpectedEof,
    /// Unrecognized type tag byte.
    InvalidTag(u8),
    /// Expected the MNode dialect leader (`0x01`), got something else.
    InvalidDialect(u8),
}

impl fmt::Display for ScanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScanError::UnexpectedEof => write!(f, "unexpected end of data"),
            ScanError::InvalidTag(t) => write!(f, "invalid type tag: {}", t),
            ScanError::InvalidDialect(d) => write!(f, "invalid dialect: 0x{:02x}", d),
        }
    }
}

// ---------------------------------------------------------------------------
// Skip / advance helpers
// ---------------------------------------------------------------------------

/// Advance past one typed MNode value in `data` starting at `pos`.
///
/// `pos` must point to the first byte of the value payload, immediately after
/// the type tag byte. Returns the position of the first byte after the value.
///
/// Pure slice indexing — no `Cursor`, no `io::Read`.
pub fn skip_value(data: &[u8], pos: usize, tag: u8) -> Result<usize, ScanError> {
    match tag {
        5 => Ok(pos),                          // Null: 0 bytes
        3 => check_bounds(data, pos, 1),       // Bool: 1 byte
        13 | 17 => check_bounds(data, pos, 2), // Short, Half: 2 bytes
        7 | 12 | 16 => check_bounds(data, pos, 4), // EnumOrd, Int32, Float32
        1 | 2 | 18 => check_bounds(data, pos, 8),  // Int, Float, Millis
        19 => check_bounds(data, pos, 12),     // Nanos (i64 + i32)
        23..=25 => check_bounds(data, pos, 16), // UuidV1, UuidV7, Ulid
        // Length-prefixed: Text(0), Bytes(4), EnumStr(6), Map(9),
        //   TextValidated(10), Ascii(11), Date(20), Time(21), DateTime(22)
        0 | 4 | 6 | 9 | 10 | 11 | 20 | 21 | 22 => skip_len32(data, pos),
        // Decimal(14): i32 scale + u32 len + data
        14 => {
            if pos + 8 > data.len() {
                return Err(ScanError::UnexpectedEof);
            }
            let len = read_u32(data, pos + 4) as usize;
            check_bounds(data, pos, 8 + len)
        }
        // Varint(15): u32 len + data
        15 => skip_len32(data, pos),
        // List(8), Set(27): u32 count + tagged elements
        8 | 27 => skip_tagged_list(data, pos),
        // Array(26): u8 elem_tag + u32 count + untagged elements
        26 => {
            if pos + 5 > data.len() {
                return Err(ScanError::UnexpectedEof);
            }
            let elem_tag = data[pos];
            let count = read_u32(data, pos + 1) as usize;
            let mut p = pos + 5;
            for _ in 0..count {
                p = skip_value(data, p, elem_tag)?;
            }
            Ok(p)
        }
        // TypedMap(28): u32 count + pairs of tagged values
        28 => {
            if pos + 4 > data.len() {
                return Err(ScanError::UnexpectedEof);
            }
            let count = read_u32(data, pos) as usize;
            let mut p = pos + 4;
            for _ in 0..count {
                if p >= data.len() {
                    return Err(ScanError::UnexpectedEof);
                }
                let k_tag = data[p];
                p = skip_value(data, p + 1, k_tag)?;
                if p >= data.len() {
                    return Err(ScanError::UnexpectedEof);
                }
                let v_tag = data[p];
                p = skip_value(data, p + 1, v_tag)?;
            }
            Ok(p)
        }
        _ => Err(ScanError::InvalidTag(tag)),
    }
}

#[inline(always)]
fn check_bounds(data: &[u8], pos: usize, size: usize) -> Result<usize, ScanError> {
    let end = pos + size;
    if end > data.len() {
        Err(ScanError::UnexpectedEof)
    } else {
        Ok(end)
    }
}

#[inline]
fn skip_len32(data: &[u8], pos: usize) -> Result<usize, ScanError> {
    if pos + 4 > data.len() {
        return Err(ScanError::UnexpectedEof);
    }
    let len = read_u32(data, pos) as usize;
    check_bounds(data, pos, 4 + len)
}

fn skip_tagged_list(data: &[u8], pos: usize) -> Result<usize, ScanError> {
    if pos + 4 > data.len() {
        return Err(ScanError::UnexpectedEof);
    }
    let count = read_u32(data, pos) as usize;
    let mut p = pos + 4;
    for _ in 0..count {
        if p >= data.len() {
            return Err(ScanError::UnexpectedEof);
        }
        let tag = data[p];
        p = skip_value(data, p + 1, tag)?;
    }
    Ok(p)
}

// ---------------------------------------------------------------------------
// Raw integer reads (no allocation)
// ---------------------------------------------------------------------------

#[inline(always)]
fn read_u16(data: &[u8], pos: usize) -> u16 {
    u16::from_le_bytes([data[pos], data[pos + 1]])
}

#[inline(always)]
fn read_u32(data: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
}

#[inline(always)]
fn read_i16(data: &[u8], pos: usize) -> i16 {
    i16::from_le_bytes([data[pos], data[pos + 1]])
}

#[inline(always)]
fn read_i32(data: &[u8], pos: usize) -> i32 {
    i32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
}

#[inline(always)]
fn read_i64(data: &[u8], pos: usize) -> i64 {
    i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap())
}

#[inline(always)]
fn read_f32(data: &[u8], pos: usize) -> f32 {
    f32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
}

#[inline(always)]
fn read_f64(data: &[u8], pos: usize) -> f64 {
    f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap())
}

/// Widen an IEEE binary16 bit pattern to `f64`.
///
/// `MValue::Half` stores the bits, not a value, so this is the only
/// correct way to read one as a number.
fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0f64 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x03FF) as f64;
    match exp {
        // Subnormal, including zero.
        0 => sign * frac * 2f64.powi(-24),
        // Infinity and NaN.
        31 => {
            if frac == 0.0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
    }
}

// ---------------------------------------------------------------------------
// Raw comparison (zero-alloc)
// ---------------------------------------------------------------------------

/// Compare a raw value at `pos` for equality with a [`Comparand`].
///
/// `pos` is the position of the first value byte (immediately after the type
/// tag). Mirrors the type coercion rules in [`pnode::eval::compare_eq`] but
/// operates directly on raw bytes without constructing [`MValue`].
///
/// Type coercion rules:
/// - Int family (Int, Int32, Short, Millis) coerces with Int and Float comparands
/// - Float family (Float, Float32) coerces with Float and Int comparands
/// - Text family (Text, Ascii, EnumStr, TextValidated) compares with Text comparand
/// - Bool matches Bool, Bytes matches Bytes, Null matches Null
fn raw_eq(data: &[u8], pos: usize, tag: u8, c: &Comparand) -> bool {
    match (tag, c) {
        // Null
        (5, Comparand::Null) => true,
        (5, _) | (_, Comparand::Null) => false,

        // Bool (tag 3)
        (3, Comparand::Bool(b)) => (data[pos] != 0) == *b,

        // Int family vs Int: Int(1), Millis(18)
        (1 | 18, Comparand::Int(ci)) => read_i64(data, pos) == *ci,
        // Int32(12)
        (12, Comparand::Int(ci)) => read_i32(data, pos) as i64 == *ci,
        // Short(13)
        (13, Comparand::Int(ci)) => read_i16(data, pos) as i64 == *ci,

        // Int family vs Float
        (1 | 18, Comparand::Float(cf)) => read_i64(data, pos) as f64 == *cf,
        (12, Comparand::Float(cf)) => read_i32(data, pos) as f64 == *cf,
        (13, Comparand::Float(cf)) => read_i16(data, pos) as f64 == *cf,

        // Float family vs Float: Float(2), Float32(16)
        (2, Comparand::Float(cf)) => read_f64(data, pos) == *cf,
        (16, Comparand::Float(cf)) => read_f32(data, pos) as f64 == *cf,

        // Float family vs Int
        (2, Comparand::Int(ci)) => read_f64(data, pos) == *ci as f64,
        (16, Comparand::Int(ci)) => read_f32(data, pos) as f64 == *ci as f64,

        // Text family vs Text: Text(0), EnumStr(6), TextValidated(10), Ascii(11)
        (0 | 6 | 10 | 11, Comparand::Text(ct)) => {
            let len = read_u32(data, pos) as usize;
            &data[pos + 4..pos + 4 + len] == ct.as_bytes()
        }

        // Bytes(4) vs Bytes
        (4, Comparand::Bytes(cb)) => {
            let len = read_u32(data, pos) as usize;
            &data[pos + 4..pos + 4 + len] == cb.as_slice()
        }

        _ => false,
    }
}

/// Compare a raw value at `pos` for ordering with a [`Comparand`].
///
/// Returns `None` for incompatible types or types without ordering.
fn raw_ord(data: &[u8], pos: usize, tag: u8, c: &Comparand) -> Option<Ordering> {
    match (tag, c) {
        // Int family vs Int
        (1 | 18, Comparand::Int(ci)) => Some(read_i64(data, pos).cmp(ci)),
        (12, Comparand::Int(ci)) => Some((read_i32(data, pos) as i64).cmp(ci)),
        (13, Comparand::Int(ci)) => Some((read_i16(data, pos) as i64).cmp(ci)),

        // Int family vs Float
        (1 | 18, Comparand::Float(cf)) => (read_i64(data, pos) as f64).partial_cmp(cf),
        (12, Comparand::Float(cf)) => (read_i32(data, pos) as f64).partial_cmp(cf),
        (13, Comparand::Float(cf)) => (read_i16(data, pos) as f64).partial_cmp(cf),

        // Float family vs Float
        (2, Comparand::Float(cf)) => read_f64(data, pos).partial_cmp(cf),
        (16, Comparand::Float(cf)) => (read_f32(data, pos) as f64).partial_cmp(cf),

        // Float family vs Int
        (2, Comparand::Int(ci)) => read_f64(data, pos).partial_cmp(&(*ci as f64)),
        (16, Comparand::Int(ci)) => (read_f32(data, pos) as f64).partial_cmp(&(*ci as f64)),

        // Text family vs Text
        (0 | 6 | 10 | 11, Comparand::Text(ct)) => {
            let len = read_u32(data, pos) as usize;
            let text = &data[pos + 4..pos + 4 + len];
            Some(text.cmp(ct.as_bytes()))
        }

        _ => None,
    }
}

/// Evaluate a single condition against a raw value.
///
/// `pos` is the position of the value payload (after the type tag byte).
/// Mirrors the behaviour of `check_condition` in `gen_metadata_indices` but
/// operates on raw bytes.
pub fn check_condition_raw(
    data: &[u8],
    pos: usize,
    tag: u8,
    op: &OpType,
    comparands: &[Comparand],
) -> bool {
    match op {
        OpType::Eq => comparands.iter().any(|c| raw_eq(data, pos, tag, c)),
        OpType::Ne => comparands.iter().all(|c| !raw_eq(data, pos, tag, c)),
        OpType::In => comparands.iter().any(|c| raw_eq(data, pos, tag, c)),
        OpType::Gt => {
            !comparands.is_empty()
                && raw_ord(data, pos, tag, &comparands[0]) == Some(Ordering::Greater)
        }
        OpType::Lt => {
            !comparands.is_empty()
                && raw_ord(data, pos, tag, &comparands[0]) == Some(Ordering::Less)
        }
        OpType::Ge => {
            !comparands.is_empty()
                && matches!(
                    raw_ord(data, pos, tag, &comparands[0]),
                    Some(Ordering::Greater | Ordering::Equal)
                )
        }
        OpType::Le => {
            !comparands.is_empty()
                && matches!(
                    raw_ord(data, pos, tag, &comparands[0]),
                    Some(Ordering::Less | Ordering::Equal)
                )
        }
        OpType::Matches => comparands.iter().any(|c| raw_matches(data, pos, tag, c)),
    }
}

/// `field MATCHES comparand` — substring containment. Mirrors
/// the SQL `LIKE '%pattern%'` semantic used by
/// `verify_predicates::pnode_to_sql`. Operates on length-prefixed
/// text-family values without allocating; pattern is matched as a
/// byte sequence (safe across UTF-8 because byte-substring search
/// is encoding-stable for valid UTF-8).
fn raw_matches(data: &[u8], pos: usize, tag: u8, c: &Comparand) -> bool {
    match (tag, c) {
        // Text family — substring contains.
        (0 | 6 | 10 | 11, Comparand::Text(pat)) => {
            let len = read_u32(data, pos) as usize;
            let haystack = &data[pos + 4..pos + 4 + len];
            byte_contains(haystack, pat.as_bytes())
        }
        // Bytes — byte-substring contains (lets predicates target
        // binary fields without a string round-trip).
        (4, Comparand::Bytes(pat)) => {
            let len = read_u32(data, pos) as usize;
            let haystack = &data[pos + 4..pos + 4 + len];
            byte_contains(haystack, pat.as_slice())
        }
        // MATCHES on non-textual types is meaningless → false.
        _ => false,
    }
}

/// Empty needles match anywhere (mirrors SQL `LIKE '%%'` ≡ true);
/// `windows()` would otherwise yield nothing for an empty pattern.
fn byte_contains(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() { return true; }
    if needle.len() > haystack.len() { return false; }
    haystack.windows(needle.len()).any(|w| w == needle)
}

// ---------------------------------------------------------------------------
// Schema discovery
// ---------------------------------------------------------------------------
// Raw field access
// ---------------------------------------------------------------------------

/// One field of a record, borrowed from the record's bytes.
///
/// Nothing is copied: the name and the value payload are slices into the
/// buffer the record was read from, and the tag is the wire discriminant.
/// A caller that wants a materialized value converts; a caller on a hot
/// path reads through the accessors below.
#[derive(Debug, Clone, Copy)]
pub struct Field<'a> {
    /// Position in the record's field order — the stable identity a
    /// compiled binder or predicate addresses fields by.
    pub index: usize,
    /// Field name, as written.
    pub name: &'a [u8],
    /// Wire type discriminant. See [`crate::mnode::TypeTag`].
    pub tag: u8,
    /// The value payload, after the tag byte.
    pub value: &'a [u8],
}

impl Field<'_> {
    /// The name as UTF-8, when it is.
    pub fn name_str(&self) -> Option<&str> {
        std::str::from_utf8(self.name).ok()
    }

    /// The value as a signed integer.
    ///
    /// Only the tags whose payload **is** an integer: `Int` and
    /// `Millis` (i64), `EnumOrd` and `Int32` (i32), `Short` (i16).
    ///
    /// `Date`, `Time` and `DateTime` are deliberately absent: they are
    /// length-prefixed strings on the wire, and reading one as an
    /// integer returns its length prefix. `Nanos` is absent because it
    /// is two fields — see [`Self::as_epoch_nanos`], which returns
    /// both rather than silently dropping the adjustment.
    pub fn as_i64(&self) -> Option<i64> {
        match self.tag {
            1 | 18 => Some(read_i64(self.value, 0)),
            7 | 12 => Some(read_i32(self.value, 0) as i64),
            13 => Some(read_i16(self.value, 0) as i64),
            _ => None,
        }
    }

    /// The value as a float.
    ///
    /// `Half` is included and widened from its binary16 bits. It is
    /// stored as a `u16`, so a caller reading it as an integer would
    /// get the bit pattern — 15360 for 1.0 — which is why this is the
    /// accessor for it and `as_i64` is not.
    pub fn as_f64(&self) -> Option<f64> {
        match self.tag {
            2 => Some(read_f64(self.value, 0)),
            16 => Some(read_f32(self.value, 0) as f64),
            17 => Some(f16_to_f64(read_u16(self.value, 0))),
            _ => None,
        }
    }

    /// The value as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        (self.tag == 3).then(|| self.value.first().is_some_and(|b| *b != 0))
    }

    /// The value as text.
    ///
    /// The length prefix is stripped and the result borrows the record.
    /// Covers `Text`, `EnumStr`, `TextValidated` and `Ascii` — and the
    /// temporal trio `Date`, `Time` and `DateTime`, which are textual
    /// on the wire whatever their names suggest.
    pub fn as_str(&self) -> Option<&str> {
        match self.tag {
            0 | 6 | 10 | 11 | 20 | 21 | 22 => {
                let len = read_u32(self.value, 0) as usize;
                self.value
                    .get(4..4 + len)
                    .and_then(|b| std::str::from_utf8(b).ok())
            }
            _ => None,
        }
    }

    /// The value as raw bytes.
    ///
    /// `Bytes` and `Varint` are length-prefixed; the identifiers are
    /// sixteen bytes with no prefix.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self.tag {
            4 | 15 => {
                let len = read_u32(self.value, 0) as usize;
                self.value.get(4..4 + len)
            }
            // UuidV1, UuidV7, Ulid: fixed width, no length prefix.
            23..=25 => Some(self.value),
            _ => None,
        }
    }

    /// A `Nanos` instant, as `(epoch_seconds, nano_adjust)`.
    ///
    /// Two fields, returned as two. Reading only the seconds would
    /// look right and lose the sub-second part.
    pub fn as_epoch_nanos(&self) -> Option<(i64, i32)> {
        (self.tag == 19).then(|| (read_i64(self.value, 0), read_i32(self.value, 8)))
    }

    /// A `Decimal`, as `(scale, unscaled digits)`.
    ///
    /// The digits are the big-integer bytes; the value is those digits
    /// scaled by `10^-scale`. Handed over rather than converted, since
    /// this crate carries no arbitrary-precision type and inventing a
    /// lossy one would be worse than making the caller choose.
    pub fn as_decimal(&self) -> Option<(i32, &[u8])> {
        if self.tag != 14 {
            return None;
        }
        let scale = read_i32(self.value, 0);
        let len = read_u32(self.value, 4) as usize;
        self.value.get(8..8 + len).map(|d| (scale, d))
    }

    /// Whether this field is the null value.
    pub fn is_null(&self) -> bool {
        self.tag == 5
    }
}

/// Walk a record's fields without allocating.
///
/// The same traversal [`scan_record`] performs, exposed so a caller
/// that binds values can reuse it rather than re-implement the wire
/// layout. Field names are skipped over rather than copied, and a
/// caller addressing fields by position need never look at them.
pub fn fields(data: &[u8]) -> Result<Fields<'_>, ScanError> {
    if data.is_empty() {
        return Err(ScanError::UnexpectedEof);
    }
    if data[0] != DIALECT_MNODE {
        return Err(ScanError::InvalidDialect(data[0]));
    }
    if data.len() < 3 {
        return Err(ScanError::UnexpectedEof);
    }
    Ok(Fields {
        data,
        pos: 3,
        remaining: read_u16(data, 1) as usize,
        index: 0,
    })
}

/// Iterator over a record's fields. See [`fields`].
pub struct Fields<'a> {
    data: &'a [u8],
    pos: usize,
    remaining: usize,
    index: usize,
}

impl<'a> Iterator for Fields<'a> {
    type Item = Result<Field<'a>, ScanError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let index = self.index;
        self.index += 1;

        let mut step = || -> Result<Field<'a>, ScanError> {
            if self.pos + 2 > self.data.len() {
                return Err(ScanError::UnexpectedEof);
            }
            let name_len = read_u16(self.data, self.pos) as usize;
            self.pos += 2;
            let name = self
                .data
                .get(self.pos..self.pos + name_len)
                .ok_or(ScanError::UnexpectedEof)?;
            self.pos += name_len;
            let tag = *self.data.get(self.pos).ok_or(ScanError::UnexpectedEof)?;
            let value_pos = self.pos + 1;
            let end = skip_value(self.data, value_pos, tag)?;
            let value = self
                .data
                .get(value_pos..end)
                .ok_or(ScanError::UnexpectedEof)?;
            self.pos = end;
            Ok(Field { index, name, tag, value })
        };
        match step() {
            Ok(f) => Some(Ok(f)),
            Err(e) => {
                // A malformed record cannot be walked past; stop rather
                // than report the same fault once per declared field.
                self.remaining = 0;
                Some(Err(e))
            }
        }
    }
}

// ---------------------------------------------------------------------------

/// Field layout discovered from a sample MNode record.
///
/// Captures the ordered field names as raw byte slices. Used to compile
/// predicates into positional form for schema-matched scanning.
pub struct RecordSchema {
    /// Field names in wire-format order (raw UTF-8 bytes, no allocation during scan).
    pub field_names: Vec<Vec<u8>>,
    /// Number of fields.
    pub field_count: usize,
}

/// Discover the field layout from a raw MNode record.
///
/// Parses the dialect byte, field count, and field names. Values are skipped
/// without allocation. Returns the schema that can be used to compile
/// predicates for positional evaluation.
pub fn discover_schema(data: &[u8]) -> Result<RecordSchema, ScanError> {
    if data.is_empty() {
        return Err(ScanError::UnexpectedEof);
    }
    if data[0] != DIALECT_MNODE {
        return Err(ScanError::InvalidDialect(data[0]));
    }
    if data.len() < 3 {
        return Err(ScanError::UnexpectedEof);
    }
    let field_count = read_u16(data, 1) as usize;
    let mut pos = 3;
    let mut field_names = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        if pos + 2 > data.len() {
            return Err(ScanError::UnexpectedEof);
        }
        let name_len = read_u16(data, pos) as usize;
        pos += 2;
        if pos + name_len > data.len() {
            return Err(ScanError::UnexpectedEof);
        }
        field_names.push(data[pos..pos + name_len].to_vec());
        pos += name_len;
        // Skip type tag + value
        if pos >= data.len() {
            return Err(ScanError::UnexpectedEof);
        }
        let tag = data[pos];
        pos = skip_value(data, pos + 1, tag)?;
    }
    Ok(RecordSchema {
        field_names,
        field_count,
    })
}

// ---------------------------------------------------------------------------
// Schema-compiled predicates
// ---------------------------------------------------------------------------

/// Predicates compiled against a known [`RecordSchema`] for zero-allocation
/// evaluation.
///
/// Instead of hashing field names at runtime, conditions are indexed by their
/// position in the record's field layout. During scanning, each field position
/// is checked by index rather than by name.
///
/// Fields referenced by predicates but absent from the schema are resolved at
/// compile time: `Eq Null` / `In Null` conditions always pass (the required
/// count is decremented), while other conditions mark the predicate as
/// unmatchable.
/// A single compiled condition: `(predicate_index, operator, comparand_values)`.
pub type FieldCondition = (usize, OpType, Vec<Comparand>);

pub struct CompiledScanPredicates {
    /// Total number of predicates.
    pub pred_count: usize,
    /// `field_conditions[field_idx]` = conditions to evaluate at that position.
    /// Empty `Vec` means the field is not referenced by any predicate.
    field_conditions: Vec<Vec<FieldCondition>>,
    /// `condition_counts[pred_idx]` = number of conditions that must pass.
    /// Adjusted during compilation for always-missing fields.
    /// `u32::MAX` marks predicates that can never match.
    condition_counts: Vec<u32>,
    /// Fallback predicates requiring full tree evaluation.
    fallback: Vec<(usize, PNode)>,
    /// Number of fields in the schema.
    schema_field_count: usize,
    /// Quick lookup: is this field position targeted by any condition?
    targeted_fields: Vec<bool>,
}

/// Attempt to flatten a PNode into AND-joined `(field_name, op, comparands)`
/// conditions. Returns `None` for OR conjugates or `FieldRef::Index` leaves.
pub fn flatten_and(pnode: &PNode) -> Option<Vec<(String, OpType, Vec<Comparand>)>> {
    match pnode {
        PNode::Predicate(pred) => {
            if let FieldRef::Named(name) = &pred.field {
                Some(vec![(name.clone(), pred.op, pred.comparands.clone())])
            } else {
                None
            }
        }
        PNode::Conjugate(conj) if conj.conjugate_type == ConjugateType::And => {
            let mut result = Vec::new();
            for child in &conj.children {
                result.extend(flatten_and(child)?);
            }
            Some(result)
        }
        _ => None,
    }
}

/// Evaluate whether a missing field satisfies a condition.
///
/// Mirrors the semantics in [`pnode::eval::evaluate`]: missing fields match
/// `Eq Null`, `In Null`, and `Ne <non-Null>`.
pub fn missing_field_passes(op: &OpType, comparands: &[Comparand]) -> bool {
    match op {
        OpType::Eq => comparands.iter().any(|c| matches!(c, Comparand::Null)),
        OpType::In => comparands.iter().any(|c| matches!(c, Comparand::Null)),
        OpType::Ne => comparands.iter().all(|c| !matches!(c, Comparand::Null)),
        _ => false,
    }
}

impl CompiledScanPredicates {
    /// Compile flattened predicates against a discovered record schema.
    ///
    /// `field_conditions_by_name` maps field names to their conditions
    /// (predicate index, operator, comparand values). `condition_counts` gives
    /// the number of conditions per predicate. `fallback` holds predicates
    /// that could not be flattened.
    ///
    /// The compilation step maps field names to schema positions. Fields not
    /// found in the schema are treated as permanently missing and their
    /// conditions are resolved at compile time.
    pub fn compile(
        pred_count: usize,
        field_conditions_by_name: &HashMap<String, Vec<FieldCondition>>,
        condition_counts_orig: &[u32],
        fallback: Vec<(usize, PNode)>,
        schema: &RecordSchema,
    ) -> Self {
        let fc = schema.field_count;
        let mut field_conditions: Vec<Vec<FieldCondition>> =
            vec![Vec::new(); fc];
        let mut targeted_fields = vec![false; fc];
        let mut condition_counts = condition_counts_orig.to_vec();

        // Build name -> schema-index map
        let name_to_idx: HashMap<&[u8], usize> = schema
            .field_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_slice(), i))
            .collect();

        for (field_name, conditions) in field_conditions_by_name {
            if let Some(&field_idx) = name_to_idx.get(field_name.as_bytes()) {
                // Field exists in schema — wire conditions to this position
                targeted_fields[field_idx] = true;
                for (pred_idx, op, comparands) in conditions {
                    field_conditions[field_idx].push((*pred_idx, *op, comparands.clone()));
                }
            } else {
                // Field is permanently missing — resolve conditions now
                for (pred_idx, op, comparands) in conditions {
                    if condition_counts[*pred_idx] == u32::MAX {
                        continue; // already dead
                    }
                    if missing_field_passes(op, comparands) {
                        condition_counts[*pred_idx] =
                            condition_counts[*pred_idx].saturating_sub(1);
                    } else {
                        condition_counts[*pred_idx] = u32::MAX; // can never match
                    }
                }
            }
        }

        CompiledScanPredicates {
            pred_count,
            field_conditions,
            condition_counts,
            fallback,
            schema_field_count: fc,
            targeted_fields,
        }
    }

    /// Number of conditions that must pass for predicate `idx` to match.
    #[inline]
    pub fn required_count(&self, idx: usize) -> u32 {
        self.condition_counts[idx]
    }

    /// Fallback predicates that need full MNode evaluation.
    pub fn fallback(&self) -> &[(usize, PNode)] {
        &self.fallback
    }
}

// ---------------------------------------------------------------------------
// Record scanning
// ---------------------------------------------------------------------------

/// Scan a raw MNode record and evaluate all compiled predicate conditions.
///
/// Walks the record's fields by position. For each targeted field, evaluates
/// the associated conditions and increments the corresponding `pass_counts`
/// entries. Non-targeted fields are skipped with zero allocation.
///
/// Returns `Ok(true)` if the record matched the compiled schema (same field
/// count) and was fully evaluated. Returns `Ok(false)` if the field count
/// does not match — the caller should fall back to `MNode::from_bytes`.
///
/// `pass_counts` must be pre-zeroed by the caller and have length >=
/// `compiled.pred_count`.
pub fn scan_record(
    data: &[u8],
    compiled: &CompiledScanPredicates,
    pass_counts: &mut [u32],
) -> Result<bool, ScanError> {
    if data.is_empty() {
        return Err(ScanError::UnexpectedEof);
    }
    if data[0] != DIALECT_MNODE {
        return Err(ScanError::InvalidDialect(data[0]));
    }
    if data.len() < 3 {
        return Err(ScanError::UnexpectedEof);
    }
    let field_count = read_u16(data, 1) as usize;
    if field_count != compiled.schema_field_count {
        return Ok(false); // schema mismatch -> caller must use fallback
    }

    let mut pos = 3;
    for field_idx in 0..field_count {
        // Skip field name
        if pos + 2 > data.len() {
            return Err(ScanError::UnexpectedEof);
        }
        let name_len = read_u16(data, pos) as usize;
        pos += 2 + name_len;

        // Read type tag
        if pos >= data.len() {
            return Err(ScanError::UnexpectedEof);
        }
        let tag = data[pos];
        let value_pos = pos + 1;

        if compiled.targeted_fields[field_idx] {
            for (pred_idx, op, comparands) in &compiled.field_conditions[field_idx] {
                if check_condition_raw(data, value_pos, tag, op, comparands) {
                    pass_counts[*pred_idx] += 1;
                }
            }
        }

        // Advance past value
        pos = skip_value(data, value_pos, tag)?;
    }

    Ok(true)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnode::{MNode, MValue};
    use crate::pnode::{ConjugateNode, PredicateNode};

    // -- skip_value --

    #[test]
    fn test_skip_null() {
        let data = [0u8; 0];
        assert_eq!(skip_value(&data, 0, 5).unwrap(), 0);
    }

    #[test]
    fn test_skip_bool() {
        let data = [1u8];
        assert_eq!(skip_value(&data, 0, 3).unwrap(), 1);
    }

    #[test]
    fn test_skip_int() {
        let data = [0u8; 8];
        assert_eq!(skip_value(&data, 0, 1).unwrap(), 8);
    }

    #[test]
    fn test_skip_float() {
        let data = [0u8; 8];
        assert_eq!(skip_value(&data, 0, 2).unwrap(), 8);
    }

    #[test]
    fn test_skip_text() {
        // u32 length = 5, then 5 bytes of text
        let mut data = vec![5, 0, 0, 0]; // length 5
        data.extend_from_slice(b"hello");
        assert_eq!(skip_value(&data, 0, 0).unwrap(), 9);
    }

    #[test]
    fn test_skip_nanos() {
        let data = [0u8; 12];
        assert_eq!(skip_value(&data, 0, 19).unwrap(), 12);
    }

    #[test]
    fn test_skip_uuid() {
        let data = [0u8; 16];
        assert_eq!(skip_value(&data, 0, 23).unwrap(), 16);
    }

    #[test]
    fn test_skip_invalid_tag() {
        let data = [0u8; 8];
        assert!(matches!(skip_value(&data, 0, 255), Err(ScanError::InvalidTag(255))));
    }

    #[test]
    fn test_skip_eof() {
        let data = [0u8; 2];
        assert!(matches!(skip_value(&data, 0, 1), Err(ScanError::UnexpectedEof)));
    }

    // -- discover_schema --

    #[test]
    fn test_discover_schema_basic() {
        let mut node = MNode::new();
        node.insert("name".into(), MValue::Text("alice".into()));
        node.insert("age".into(), MValue::Int(30));
        node.insert("active".into(), MValue::Bool(true));
        let bytes = node.to_bytes();

        let schema = discover_schema(&bytes).unwrap();
        assert_eq!(schema.field_count, 3);
        assert_eq!(schema.field_names[0], b"name");
        assert_eq!(schema.field_names[1], b"age");
        assert_eq!(schema.field_names[2], b"active");
    }

    #[test]
    fn test_discover_schema_empty_mnode() {
        let node = MNode::new();
        let bytes = node.to_bytes();
        let schema = discover_schema(&bytes).unwrap();
        assert_eq!(schema.field_count, 0);
        assert!(schema.field_names.is_empty());
    }

    // -- check_condition_raw --

    fn build_record(fields: &[(&str, MValue)]) -> Vec<u8> {
        let mut node = MNode::new();
        for (name, val) in fields {
            node.insert(name.to_string(), val.clone());
        }
        node.to_bytes()
    }

    /// Locate a field's value position and tag in raw MNode bytes.
    fn find_field_value(data: &[u8], field_idx: usize) -> (usize, u8) {
        let field_count = read_u16(data, 1) as usize;
        assert!(field_idx < field_count);
        let mut pos = 3;
        for i in 0..=field_idx {
            let name_len = read_u16(data, pos) as usize;
            pos += 2 + name_len;
            let tag = data[pos];
            if i == field_idx {
                return (pos + 1, tag);
            }
            pos = skip_value(data, pos + 1, tag).unwrap();
        }
        unreachable!()
    }

    #[test]
    fn test_raw_eq_int() {
        let data = build_record(&[("x", MValue::Int(42))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Int(42)]));
        assert!(!check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Int(99)]));
    }

    #[test]
    fn test_raw_eq_float() {
        let data = build_record(&[("x", MValue::Float(3.25))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Float(3.25)]));
        assert!(!check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Float(2.0)]));
    }

    #[test]
    fn test_raw_eq_text() {
        let data = build_record(&[("x", MValue::Text("hello".into()))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(
            &data,
            vpos,
            tag,
            &OpType::Eq,
            &[Comparand::Text("hello".into())]
        ));
        assert!(!check_condition_raw(
            &data,
            vpos,
            tag,
            &OpType::Eq,
            &[Comparand::Text("world".into())]
        ));
    }

    #[test]
    fn test_raw_eq_bool() {
        let data = build_record(&[("x", MValue::Bool(true))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Bool(true)]));
        assert!(!check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Bool(false)]));
    }

    // -- MATCHES (substring containment, SQL LIKE '%pattern%') --

    /// MATCHES on a text field is a substring search. Mirrors
    /// the high-level [`evaluate`](crate::pnode::eval::evaluate)
    /// path so the raw scanner produces identical truth values.
    /// Regression guard for the bug where every trigram-MATCHES
    /// predicate matched zero records because this branch
    /// returned `false` unconditionally.
    #[test]
    fn test_raw_matches_text_substring_hit() {
        let data = build_record(&[("caption", MValue::Text("the cat sat".into()))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(
            &data, vpos, tag,
            &OpType::Matches,
            &[Comparand::Text("the".into())],
        ));
        assert!(check_condition_raw(
            &data, vpos, tag,
            &OpType::Matches,
            &[Comparand::Text("cat".into())],
        ));
        assert!(check_condition_raw(
            &data, vpos, tag,
            &OpType::Matches,
            &[Comparand::Text("at sa".into())],
        ));
    }

    #[test]
    fn test_raw_matches_text_substring_miss() {
        let data = build_record(&[("caption", MValue::Text("the cat".into()))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(!check_condition_raw(
            &data, vpos, tag,
            &OpType::Matches,
            &[Comparand::Text("xyz".into())],
        ));
    }

    #[test]
    fn test_raw_matches_empty_pattern_is_true() {
        let data = build_record(&[("caption", MValue::Text("hello".into()))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(
            &data, vpos, tag,
            &OpType::Matches,
            &[Comparand::Text("".into())],
        ));
    }

    #[test]
    fn test_raw_matches_bytes_substring() {
        let data = build_record(&[("blob",
            MValue::Bytes(vec![0x01, 0xAB, 0xCD, 0xEF]))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(
            &data, vpos, tag,
            &OpType::Matches,
            &[Comparand::Bytes(vec![0xAB, 0xCD])],
        ));
        assert!(!check_condition_raw(
            &data, vpos, tag,
            &OpType::Matches,
            &[Comparand::Bytes(vec![0xFE, 0xED])],
        ));
    }

    #[test]
    fn test_raw_matches_on_int_is_false() {
        let data = build_record(&[("x", MValue::Int(42))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(!check_condition_raw(
            &data, vpos, tag,
            &OpType::Matches,
            &[Comparand::Text("42".into())],
        ));
    }

    #[test]
    fn test_raw_matches_multi_comparand_any() {
        let data = build_record(&[("caption", MValue::Text("the cat".into()))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(
            &data, vpos, tag,
            &OpType::Matches,
            &[
                Comparand::Text("xyz".into()),
                Comparand::Text("cat".into()),
            ],
        ));
    }

    #[test]
    fn test_raw_eq_null() {
        let data = build_record(&[("x", MValue::Null)]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Null]));
        assert!(!check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Int(0)]));
    }

    #[test]
    fn test_raw_gt_int() {
        let data = build_record(&[("x", MValue::Int(15))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Gt, &[Comparand::Int(10)]));
        assert!(!check_condition_raw(&data, vpos, tag, &OpType::Gt, &[Comparand::Int(15)]));
        assert!(!check_condition_raw(&data, vpos, tag, &OpType::Gt, &[Comparand::Int(20)]));
    }

    #[test]
    fn test_raw_le_float() {
        let data = build_record(&[("x", MValue::Float(5.0))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Le, &[Comparand::Float(5.0)]));
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Le, &[Comparand::Float(6.0)]));
        assert!(!check_condition_raw(&data, vpos, tag, &OpType::Le, &[Comparand::Float(4.0)]));
    }

    #[test]
    fn test_raw_in() {
        let data = build_record(&[("x", MValue::Int(3))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(
            &data,
            vpos,
            tag,
            &OpType::In,
            &[Comparand::Int(1), Comparand::Int(3), Comparand::Int(5)]
        ));
        assert!(!check_condition_raw(
            &data,
            vpos,
            tag,
            &OpType::In,
            &[Comparand::Int(2), Comparand::Int(4)]
        ));
    }

    #[test]
    fn test_raw_ne() {
        let data = build_record(&[("x", MValue::Int(42))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(!check_condition_raw(&data, vpos, tag, &OpType::Ne, &[Comparand::Int(42)]));
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Ne, &[Comparand::Int(99)]));
    }

    #[test]
    fn test_raw_cross_type_int_vs_float() {
        let data = build_record(&[("x", MValue::Int(42))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Float(42.0)]));
        assert!(!check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Float(42.5)]));
    }

    #[test]
    fn test_raw_cross_type_float_vs_int() {
        let data = build_record(&[("x", MValue::Float(42.0))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Int(42)]));
        assert!(!check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Int(43)]));
    }

    #[test]
    fn test_raw_int32_vs_int() {
        let data = build_record(&[("x", MValue::Int32(42))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Eq, &[Comparand::Int(42)]));
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Gt, &[Comparand::Int(10)]));
    }

    #[test]
    fn test_raw_short_vs_int() {
        let data = build_record(&[("x", MValue::Short(11))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(&data, vpos, tag, &OpType::Gt, &[Comparand::Int(10)]));
        assert!(!check_condition_raw(&data, vpos, tag, &OpType::Gt, &[Comparand::Int(11)]));
    }

    #[test]
    fn test_raw_ascii_vs_text() {
        let data = build_record(&[("x", MValue::Ascii("hello".into()))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(
            &data,
            vpos,
            tag,
            &OpType::Eq,
            &[Comparand::Text("hello".into())]
        ));
    }

    #[test]
    fn test_raw_enumstr_vs_text() {
        let data = build_record(&[("x", MValue::EnumStr("red".into()))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(
            &data,
            vpos,
            tag,
            &OpType::Eq,
            &[Comparand::Text("red".into())]
        ));
    }

    #[test]
    fn test_raw_bytes() {
        let data = build_record(&[("x", MValue::Bytes(vec![1, 2, 3]))]);
        let (vpos, tag) = find_field_value(&data, 0);
        assert!(check_condition_raw(
            &data,
            vpos,
            tag,
            &OpType::Eq,
            &[Comparand::Bytes(vec![1, 2, 3])]
        ));
        assert!(!check_condition_raw(
            &data,
            vpos,
            tag,
            &OpType::Eq,
            &[Comparand::Bytes(vec![4, 5])]
        ));
    }

    // -- scan_record --

    #[test]
    fn test_scan_record_basic() {
        let data = build_record(&[
            ("user_id", MValue::Int(10)),
            ("name", MValue::Text("alice".into())),
            ("active", MValue::Bool(true)),
        ]);

        let schema = discover_schema(&data).unwrap();

        // Predicate 0: user_id > 5
        // Predicate 1: name = 'alice'
        let mut fc: HashMap<String, Vec<FieldCondition>> = HashMap::new();
        fc.entry("user_id".into())
            .or_default()
            .push((0, OpType::Gt, vec![Comparand::Int(5)]));
        fc.entry("name".into())
            .or_default()
            .push((1, OpType::Eq, vec![Comparand::Text("alice".into())]));

        let condition_counts = vec![1u32, 1];
        let compiled =
            CompiledScanPredicates::compile(2, &fc, &condition_counts, Vec::new(), &schema);

        let mut pass_counts = vec![0u32; 2];
        let matched = scan_record(&data, &compiled, &mut pass_counts).unwrap();
        assert!(matched);
        assert_eq!(pass_counts[0], 1); // user_id > 5 passes
        assert_eq!(pass_counts[1], 1); // name = 'alice' passes
    }

    #[test]
    fn test_scan_record_missing_field_in_schema() {
        let data = build_record(&[
            ("user_id", MValue::Int(10)),
            ("name", MValue::Text("alice".into())),
        ]);

        let schema = discover_schema(&data).unwrap();

        // Predicate: missing_field Eq Null
        let mut fc: HashMap<String, Vec<FieldCondition>> = HashMap::new();
        fc.entry("missing_field".into())
            .or_default()
            .push((0, OpType::Eq, vec![Comparand::Null]));

        let condition_counts = vec![1u32];
        let compiled =
            CompiledScanPredicates::compile(1, &fc, &condition_counts, Vec::new(), &schema);

        // missing_field resolved at compile time -> condition_counts decremented
        assert_eq!(compiled.required_count(0), 0);
    }

    #[test]
    fn test_scan_record_missing_field_non_null() {
        let data = build_record(&[("user_id", MValue::Int(10))]);

        let schema = discover_schema(&data).unwrap();

        // Predicate: missing_field > 5 -> can never match
        let mut fc: HashMap<String, Vec<FieldCondition>> = HashMap::new();
        fc.entry("missing".into())
            .or_default()
            .push((0, OpType::Gt, vec![Comparand::Int(5)]));

        let condition_counts = vec![1u32];
        let compiled =
            CompiledScanPredicates::compile(1, &fc, &condition_counts, Vec::new(), &schema);

        // marked as unmatchable
        assert_eq!(compiled.required_count(0), u32::MAX);
    }

    #[test]
    fn test_scan_record_schema_mismatch() {
        // Record has 2 fields, schema has 3
        let data2 = build_record(&[("a", MValue::Int(1)), ("b", MValue::Int(2))]);

        let data3 = build_record(&[
            ("a", MValue::Int(1)),
            ("b", MValue::Int(2)),
            ("c", MValue::Int(3)),
        ]);
        let schema3 = discover_schema(&data3).unwrap();

        let compiled =
            CompiledScanPredicates::compile(0, &HashMap::new(), &[], Vec::new(), &schema3);

        let mut pass_counts = vec![];
        let matched = scan_record(&data2, &compiled, &mut pass_counts).unwrap();
        assert!(!matched); // field count mismatch
    }

    #[test]
    fn test_scan_record_and_predicate() {
        // AND: user_id >= 10 AND active = true
        let data = build_record(&[
            ("user_id", MValue::Int(15)),
            ("active", MValue::Bool(true)),
        ]);
        let schema = discover_schema(&data).unwrap();

        let mut fc: HashMap<String, Vec<FieldCondition>> = HashMap::new();
        fc.entry("user_id".into())
            .or_default()
            .push((0, OpType::Ge, vec![Comparand::Int(10)]));
        fc.entry("active".into())
            .or_default()
            .push((0, OpType::Eq, vec![Comparand::Bool(true)]));

        let condition_counts = vec![2u32]; // 2 conditions must both pass
        let compiled =
            CompiledScanPredicates::compile(1, &fc, &condition_counts, Vec::new(), &schema);

        let mut pass_counts = vec![0u32; 1];
        scan_record(&data, &compiled, &mut pass_counts).unwrap();
        assert_eq!(pass_counts[0], 2); // both conditions pass
        assert_eq!(compiled.required_count(0), 2);
    }

    #[test]
    fn test_scan_skips_non_targeted_fields() {
        // Record has 3 fields but only 'score' is targeted
        let data = build_record(&[
            ("name", MValue::Text("alice".into())),
            ("score", MValue::Float(99.5)),
            ("extra", MValue::Bytes(vec![1, 2, 3, 4, 5])),
        ]);
        let schema = discover_schema(&data).unwrap();

        let mut fc: HashMap<String, Vec<FieldCondition>> = HashMap::new();
        fc.entry("score".into())
            .or_default()
            .push((0, OpType::Gt, vec![Comparand::Float(50.0)]));

        let condition_counts = vec![1u32];
        let compiled =
            CompiledScanPredicates::compile(1, &fc, &condition_counts, Vec::new(), &schema);

        let mut pass_counts = vec![0u32; 1];
        let matched = scan_record(&data, &compiled, &mut pass_counts).unwrap();
        assert!(matched);
        assert_eq!(pass_counts[0], 1);
    }

    // -- flatten_and --

    #[test]
    fn test_flatten_and_simple_predicate() {
        let p = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("x".into()),
            op: OpType::Eq,
            comparands: vec![Comparand::Int(42)],
        });
        let flat = flatten_and(&p).unwrap();
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].0, "x");
    }

    #[test]
    fn test_flatten_and_conjugate() {
        let p = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("a".into()),
                    op: OpType::Gt,
                    comparands: vec![Comparand::Int(1)],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("b".into()),
                    op: OpType::Lt,
                    comparands: vec![Comparand::Float(5.0)],
                }),
            ],
        });
        let flat = flatten_and(&p).unwrap();
        assert_eq!(flat.len(), 2);
    }

    #[test]
    fn test_flatten_and_or_returns_none() {
        let p = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::Or,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("a".into()),
                    op: OpType::Eq,
                    comparands: vec![Comparand::Int(1)],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("b".into()),
                    op: OpType::Eq,
                    comparands: vec![Comparand::Int(2)],
                }),
            ],
        });
        assert!(flatten_and(&p).is_none());
    }

    #[test]
    fn test_flatten_and_index_returns_none() {
        let p = PNode::Predicate(PredicateNode {
            field: FieldRef::Index(0),
            op: OpType::Eq,
            comparands: vec![Comparand::Int(1)],
        });
        assert!(flatten_and(&p).is_none());
    }

    // -- missing_field_passes --

    #[test]
    fn test_missing_field_eq_null() {
        assert!(missing_field_passes(&OpType::Eq, &[Comparand::Null]));
    }

    #[test]
    fn test_missing_field_eq_int() {
        assert!(!missing_field_passes(&OpType::Eq, &[Comparand::Int(42)]));
    }

    #[test]
    fn test_missing_field_in_null() {
        assert!(missing_field_passes(
            &OpType::In,
            &[Comparand::Int(1), Comparand::Null]
        ));
    }

    #[test]
    fn test_missing_field_ne_null() {
        assert!(!missing_field_passes(&OpType::Ne, &[Comparand::Null]));
    }

    #[test]
    fn test_missing_field_ne_int() {
        assert!(missing_field_passes(&OpType::Ne, &[Comparand::Int(42)]));
    }

    #[test]
    fn test_missing_field_gt() {
        assert!(!missing_field_passes(&OpType::Gt, &[Comparand::Int(0)]));
    }
}

#[cfg(test)]
mod field_access_tests {
    use super::*;
    use crate::mnode::{MNode, MValue};

    fn record() -> Vec<u8> {
        let mut n = MNode::new();
        n.insert("id".into(), MValue::Int(42));
        n.insert("score".into(), MValue::Float(1.5));
        n.insert("tag".into(), MValue::Text("hello".into()));
        n.insert("ok".into(), MValue::Bool(true));
        n.insert("small".into(), MValue::Int32(7));
        n.insert("absent".into(), MValue::Null);
        n.to_bytes()
    }

    /// Fields come back in wire order, which is the order they were
    /// inserted — the stable position a binder addresses them by.
    #[test]
    fn fields_walk_in_declared_order() {
        let bytes = record();
        let names: Vec<String> = fields(&bytes)
            .unwrap()
            .map(|f| f.unwrap().name_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["id", "score", "tag", "ok", "small", "absent"]);
        let idx: Vec<usize> = fields(&bytes).unwrap().map(|f| f.unwrap().index).collect();
        assert_eq!(idx, [0, 1, 2, 3, 4, 5]);
    }

    /// Values read without materializing an `MValue`, and the borrows
    /// point into the record.
    #[test]
    fn values_read_raw_by_shape() {
        let bytes = record();
        let f: Vec<Field<'_>> = fields(&bytes).unwrap().map(|f| f.unwrap()).collect();

        assert_eq!(f[0].as_i64(), Some(42));
        assert_eq!(f[1].as_f64(), Some(1.5));
        assert_eq!(f[2].as_str(), Some("hello"));
        assert_eq!(f[3].as_bool(), Some(true));
        assert_eq!(f[4].as_i64(), Some(7), "Int32 is integer-shaped");
        assert!(f[5].is_null());

        // The borrow is into the record's own buffer, not a copy.
        let s = f[2].as_str().unwrap();
        assert!(bytes.as_ptr_range().contains(&s.as_ptr()));
    }

    /// Asking the wrong question of a field answers `None` rather than
    /// coercing — a text field is not a number, and saying so is not
    /// the same as the field being absent.
    #[test]
    fn a_mismatched_accessor_declines_rather_than_coercing() {
        let bytes = record();
        let f: Vec<Field<'_>> = fields(&bytes).unwrap().map(|f| f.unwrap()).collect();
        assert_eq!(f[2].as_i64(), None, "text is not an integer");
        assert_eq!(f[0].as_str(), None, "an integer is not text");
        assert_eq!(f[0].as_f64(), None, "integer-shaped is not float-shaped");
        assert!(!f[0].is_null());
    }

    /// The walk agrees with `discover_schema`, which is what lets a
    /// binder compile positions from one and read values with the
    /// other.
    #[test]
    fn the_walk_agrees_with_the_discovered_schema() {
        let bytes = record();
        let schema = discover_schema(&bytes).unwrap();
        let walked: Vec<&[u8]> = fields(&bytes).unwrap().map(|f| f.unwrap().name).collect();
        assert_eq!(schema.field_count, walked.len());
        for (i, name) in walked.iter().enumerate() {
            assert_eq!(&schema.field_names[i], name, "position {i}");
        }
    }

    /// A record that is not an MNode is refused by its leader byte,
    /// not walked into.
    #[test]
    fn a_foreign_dialect_is_refused() {
        let mut bytes = record();
        bytes[0] = crate::pnode::DIALECT_PNODE;
        assert!(matches!(fields(&bytes), Err(ScanError::InvalidDialect(_))));
    }

    /// A truncated record stops at the fault rather than reporting one
    /// per declared field.
    #[test]
    fn a_truncated_record_stops_at_the_fault() {
        let bytes = record();
        let short = &bytes[..bytes.len() - 4];
        let results: Vec<_> = fields(short).unwrap().collect();
        assert!(results.iter().any(|r| r.is_err()));
        assert_eq!(results.iter().filter(|r| r.is_err()).count(), 1);
    }
}

#[cfg(test)]
mod wire_shape_tests {
    use super::*;
    use crate::mnode::{MNode, MValue, TypeTag};

    fn one(v: MValue) -> Vec<u8> {
        let mut n = MNode::new();
        n.insert("f".into(), v);
        n.to_bytes()
    }

    fn field(bytes: &[u8]) -> Field<'_> {
        fields(bytes).unwrap().next().unwrap().unwrap()
    }

    /// **The temporal trio is textual.** `Date`, `Time` and `DateTime`
    /// hold strings on the wire whatever their names suggest, and an
    /// accessor that read them as integers would return their length
    /// prefix — a plausible number, silently wrong.
    #[test]
    fn date_time_and_datetime_are_strings_not_integers() {
        for (v, expect) in [
            (MValue::Date("2026-09-01".into()), "2026-09-01"),
            (MValue::Time("12:34:56".into()), "12:34:56"),
            (MValue::DateTime("2026-09-01T12:34:56Z".into()), "2026-09-01T12:34:56Z"),
        ] {
            let bytes = one(v);
            let f = field(&bytes);
            assert_eq!(f.as_str(), Some(expect));
            assert_eq!(f.as_i64(), None, "reading {expect} as an integer must decline");
        }
    }

    /// **`Nanos` is two fields**, and both come back. Returning only
    /// the seconds would look right and lose the adjustment.
    #[test]
    fn nanos_yields_both_of_its_fields() {
        let bytes = one(MValue::Nanos {
            epoch_seconds: 1_700_000_000,
            nano_adjust: 123_456_789,
        });
        let f = field(&bytes);
        assert_eq!(f.as_epoch_nanos(), Some((1_700_000_000, 123_456_789)));
        assert_eq!(f.as_i64(), None, "not a plain integer");
    }

    /// **`Half` reads as the float it is**, not as its bit pattern.
    #[test]
    fn half_widens_from_its_bits() {
        for (bits, expect) in [(0x3C00u16, 1.0f64), (0x4000, 2.0), (0xC000, -2.0), (0x0000, 0.0)] {
            let bytes = one(MValue::Half(bits));
            let f = field(&bytes);
            assert_eq!(f.as_f64(), Some(expect), "bits {bits:#06x}");
            assert_eq!(f.as_i64(), None, "the bit pattern is not the value");
        }
    }

    /// The integer family reads by its own width, and nothing else
    /// answers as an integer.
    #[test]
    fn the_integer_family_reads_by_width() {
        assert_eq!(field(&one(MValue::Int(-5))).as_i64(), Some(-5));
        assert_eq!(field(&one(MValue::Millis(1_700_000_000_000))).as_i64(), Some(1_700_000_000_000));
        assert_eq!(field(&one(MValue::Int32(-7))).as_i64(), Some(-7));
        assert_eq!(field(&one(MValue::Short(-9))).as_i64(), Some(-9));
        assert_eq!(field(&one(MValue::EnumOrd(3))).as_i64(), Some(3));
    }

    /// Identifiers are sixteen bytes with no length prefix; `Bytes` has
    /// one. Both come back as their content.
    #[test]
    fn byte_shaped_values_strip_their_prefix_or_do_not_have_one() {
        let id = [7u8; 16];
        assert_eq!(field(&one(MValue::UuidV7(id))).as_bytes(), Some(&id[..]));
        assert_eq!(field(&one(MValue::Ulid(id))).as_bytes(), Some(&id[..]));
        assert_eq!(
            field(&one(MValue::Bytes(vec![1, 2, 3]))).as_bytes(),
            Some(&[1u8, 2, 3][..])
        );
    }

    /// **Every tag the scanner can skip is a tag `fields` can walk.**
    ///
    /// The tag set is a cross-language contract wider than this crate's
    /// value enum — `TextValidated`, `Decimal` and `Varint` have no
    /// `MValue` — so a record written by another implementation can
    /// carry them. Walking must not stop at one.
    #[test]
    fn every_tag_can_be_walked_even_without_a_local_value_type() {
        // Hand-built: a Decimal field, which has no MValue to build from.
        let mut rec = vec![DIALECT_MNODE];
        rec.extend_from_slice(&1u16.to_le_bytes()); // one field
        rec.extend_from_slice(&1u16.to_le_bytes()); // name length
        rec.push(b'd');
        rec.push(TypeTag::Decimal as u8);
        rec.extend_from_slice(&2i32.to_le_bytes()); // scale
        rec.extend_from_slice(&3u32.to_le_bytes()); // digit length
        rec.extend_from_slice(&[9, 8, 7]);

        let f = field(&rec);
        assert_eq!(f.name_str(), Some("d"));
        assert_eq!(f.as_decimal(), Some((2, &[9u8, 8, 7][..])));
        // And the walk completes rather than stopping at an unknown shape.
        assert_eq!(fields(&rec).unwrap().count(), 1);
    }
}
