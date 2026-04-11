// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Checked integer type conversions between scalar formats.
//!
//! Widening conversions always succeed. Narrowing and signed↔unsigned
//! conversions check every value and return an error with the offending
//! ordinal and value on overflow.

use std::fmt;

/// Element type for scalar conversion dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    U8, I8, U16, I16, U32, I32, U64, I64,
}

impl ScalarType {
    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 => 4,
            Self::U64 | Self::I64 => 8,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8", Self::I8 => "i8",
            Self::U16 => "u16", Self::I16 => "i16",
            Self::U32 => "u32", Self::I32 => "i32",
            Self::U64 => "u64", Self::I64 => "i64",
        }
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Error from a narrowing or sign-mismatch conversion.
#[derive(Debug)]
pub struct ConvertError {
    pub ordinal: u64,
    pub value: i128,
    pub from_type: ScalarType,
    pub to_type: ScalarType,
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "narrowing overflow at ordinal {}: value {} ({}) does not fit in {}",
            self.ordinal, self.value, self.from_type, self.to_type)
    }
}

impl std::error::Error for ConvertError {}

/// Read a single value from `data` at byte offset 0 as i128 for comparison.
fn read_value(data: &[u8], ty: ScalarType) -> i128 {
    match ty {
        ScalarType::U8 => data[0] as i128,
        ScalarType::I8 => data[0] as i8 as i128,
        ScalarType::U16 => u16::from_le_bytes([data[0], data[1]]) as i128,
        ScalarType::I16 => i16::from_le_bytes([data[0], data[1]]) as i128,
        ScalarType::U32 => u32::from_le_bytes(data[..4].try_into().unwrap()) as i128,
        ScalarType::I32 => i32::from_le_bytes(data[..4].try_into().unwrap()) as i128,
        ScalarType::U64 => u64::from_le_bytes(data[..8].try_into().unwrap()) as i128,
        ScalarType::I64 => i64::from_le_bytes(data[..8].try_into().unwrap()) as i128,
    }
}

/// Write a value as the target type's little-endian bytes.
fn write_value(val: i128, ty: ScalarType) -> Vec<u8> {
    match ty {
        ScalarType::U8 => vec![val as u8],
        ScalarType::I8 => vec![val as i8 as u8],
        ScalarType::U16 => (val as u16).to_le_bytes().to_vec(),
        ScalarType::I16 => (val as i16).to_le_bytes().to_vec(),
        ScalarType::U32 => (val as u32).to_le_bytes().to_vec(),
        ScalarType::I32 => (val as i32).to_le_bytes().to_vec(),
        ScalarType::U64 => (val as u64).to_le_bytes().to_vec(),
        ScalarType::I64 => (val as i64).to_le_bytes().to_vec(),
    }
}

/// Check if a value fits in the target type's range.
fn fits(val: i128, ty: ScalarType) -> bool {
    match ty {
        ScalarType::U8 => val >= 0 && val <= u8::MAX as i128,
        ScalarType::I8 => val >= i8::MIN as i128 && val <= i8::MAX as i128,
        ScalarType::U16 => val >= 0 && val <= u16::MAX as i128,
        ScalarType::I16 => val >= i16::MIN as i128 && val <= i16::MAX as i128,
        ScalarType::U32 => val >= 0 && val <= u32::MAX as i128,
        ScalarType::I32 => val >= i32::MIN as i128 && val <= i32::MAX as i128,
        ScalarType::U64 => val >= 0 && val <= u64::MAX as i128,
        ScalarType::I64 => val >= i64::MIN as i128 && val <= i64::MAX as i128,
    }
}

/// Convert a flat packed scalar array from one integer type to another.
///
/// Returns the converted bytes, or an error if any value overflows the
/// target type (narrowing or signed↔unsigned mismatch).
///
/// Identity conversions (same type) are a no-op copy.
pub fn convert_scalars(
    data: &[u8],
    from: ScalarType,
    to: ScalarType,
) -> Result<Vec<u8>, ConvertError> {
    let from_size = from.size();
    let to_size = to.size();

    if data.len() % from_size != 0 {
        return Err(ConvertError {
            ordinal: 0, value: 0, from_type: from, to_type: to,
        });
    }

    let count = data.len() / from_size;
    let mut out = Vec::with_capacity(count * to_size);

    if from == to {
        out.extend_from_slice(data);
        return Ok(out);
    }

    for i in 0..count {
        let offset = i * from_size;
        let val = read_value(&data[offset..offset + from_size], from);
        if !fits(val, to) {
            return Err(ConvertError {
                ordinal: i as u64,
                value: val,
                from_type: from,
                to_type: to,
            });
        }
        out.extend_from_slice(&write_value(val, to));
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data<T: Copy>(vals: &[T]) -> Vec<u8> {
        let mut out = Vec::new();
        for v in vals {
            let bytes = unsafe {
                std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>())
            };
            out.extend_from_slice(bytes);
        }
        out
    }

    // ── Identity conversions ───────────────────────────────────────

    #[test]
    fn identity_u8() {
        let data = make_data(&[0u8, 127, 255]);
        let out = convert_scalars(&data, ScalarType::U8, ScalarType::U8).unwrap();
        assert_eq!(data, out);
    }

    #[test]
    fn identity_i64() {
        let data = make_data(&[i64::MIN, 0i64, i64::MAX]);
        let out = convert_scalars(&data, ScalarType::I64, ScalarType::I64).unwrap();
        assert_eq!(data, out);
    }

    // ── Widening conversions (always succeed) ──────────────────────

    #[test] fn u8_to_u16() {
        let data = make_data(&[0u8, 255]);
        let out = convert_scalars(&data, ScalarType::U8, ScalarType::U16).unwrap();
        assert_eq!(out, make_data(&[0u16, 255]));
    }

    #[test] fn u8_to_u32() {
        let data = make_data(&[0u8, 255]);
        let out = convert_scalars(&data, ScalarType::U8, ScalarType::U32).unwrap();
        assert_eq!(out, make_data(&[0u32, 255]));
    }

    #[test] fn u8_to_u64() {
        let data = make_data(&[255u8]);
        let out = convert_scalars(&data, ScalarType::U8, ScalarType::U64).unwrap();
        assert_eq!(out, make_data(&[255u64]));
    }

    #[test] fn u8_to_i16() {
        let data = make_data(&[0u8, 255]);
        let out = convert_scalars(&data, ScalarType::U8, ScalarType::I16).unwrap();
        assert_eq!(out, make_data(&[0i16, 255]));
    }

    #[test] fn u8_to_i32() {
        let data = make_data(&[255u8]);
        let out = convert_scalars(&data, ScalarType::U8, ScalarType::I32).unwrap();
        assert_eq!(out, make_data(&[255i32]));
    }

    #[test] fn u8_to_i64() {
        let data = make_data(&[255u8]);
        let out = convert_scalars(&data, ScalarType::U8, ScalarType::I64).unwrap();
        assert_eq!(out, make_data(&[255i64]));
    }

    #[test] fn i8_to_i16() {
        let data = make_data(&[-128i8, 0, 127]);
        let out = convert_scalars(&data, ScalarType::I8, ScalarType::I16).unwrap();
        assert_eq!(out, make_data(&[-128i16, 0, 127]));
    }

    #[test] fn i8_to_i32() {
        let data = make_data(&[-1i8]);
        let out = convert_scalars(&data, ScalarType::I8, ScalarType::I32).unwrap();
        assert_eq!(out, make_data(&[-1i32]));
    }

    #[test] fn i8_to_i64() {
        let data = make_data(&[i8::MIN]);
        let out = convert_scalars(&data, ScalarType::I8, ScalarType::I64).unwrap();
        assert_eq!(out, make_data(&[i8::MIN as i64]));
    }

    #[test] fn u16_to_u32() {
        let data = make_data(&[0u16, 65535]);
        let out = convert_scalars(&data, ScalarType::U16, ScalarType::U32).unwrap();
        assert_eq!(out, make_data(&[0u32, 65535]));
    }

    #[test] fn u16_to_u64() {
        let data = make_data(&[65535u16]);
        let out = convert_scalars(&data, ScalarType::U16, ScalarType::U64).unwrap();
        assert_eq!(out, make_data(&[65535u64]));
    }

    #[test] fn i16_to_i32() {
        let data = make_data(&[i16::MIN, i16::MAX]);
        let out = convert_scalars(&data, ScalarType::I16, ScalarType::I32).unwrap();
        assert_eq!(out, make_data(&[i16::MIN as i32, i16::MAX as i32]));
    }

    #[test] fn u32_to_u64() {
        let data = make_data(&[u32::MAX]);
        let out = convert_scalars(&data, ScalarType::U32, ScalarType::U64).unwrap();
        assert_eq!(out, make_data(&[u32::MAX as u64]));
    }

    #[test] fn i32_to_i64() {
        let data = make_data(&[i32::MIN, i32::MAX]);
        let out = convert_scalars(&data, ScalarType::I32, ScalarType::I64).unwrap();
        assert_eq!(out, make_data(&[i32::MIN as i64, i32::MAX as i64]));
    }

    // ── Narrowing conversions that succeed ──────────────────────────

    #[test] fn u16_to_u8_ok() {
        let data = make_data(&[0u16, 255]);
        let out = convert_scalars(&data, ScalarType::U16, ScalarType::U8).unwrap();
        assert_eq!(out, make_data(&[0u8, 255]));
    }

    #[test] fn i16_to_i8_ok() {
        let data = make_data(&[-128i16, 127]);
        let out = convert_scalars(&data, ScalarType::I16, ScalarType::I8).unwrap();
        assert_eq!(out, make_data(&[-128i8, 127]));
    }

    #[test] fn u32_to_u16_ok() {
        let data = make_data(&[0u32, 65535]);
        let out = convert_scalars(&data, ScalarType::U32, ScalarType::U16).unwrap();
        assert_eq!(out, make_data(&[0u16, 65535]));
    }

    #[test] fn i32_to_i16_ok() {
        let data = make_data(&[-32768i32, 32767]);
        let out = convert_scalars(&data, ScalarType::I32, ScalarType::I16).unwrap();
        assert_eq!(out, make_data(&[-32768i16, 32767]));
    }

    #[test] fn u64_to_u32_ok() {
        let data = make_data(&[0u64, u32::MAX as u64]);
        let out = convert_scalars(&data, ScalarType::U64, ScalarType::U32).unwrap();
        assert_eq!(out, make_data(&[0u32, u32::MAX]));
    }

    #[test] fn i64_to_i32_ok() {
        let data = make_data(&[i32::MIN as i64, i32::MAX as i64]);
        let out = convert_scalars(&data, ScalarType::I64, ScalarType::I32).unwrap();
        assert_eq!(out, make_data(&[i32::MIN, i32::MAX]));
    }

    // ── Narrowing conversions that fail ─────────────────────────────

    #[test] fn u16_to_u8_overflow() {
        let data = make_data(&[256u16]);
        let err = convert_scalars(&data, ScalarType::U16, ScalarType::U8).unwrap_err();
        assert_eq!(err.ordinal, 0);
        assert_eq!(err.value, 256);
    }

    #[test] fn i16_to_i8_overflow_positive() {
        let data = make_data(&[128i16]);
        let err = convert_scalars(&data, ScalarType::I16, ScalarType::I8).unwrap_err();
        assert_eq!(err.value, 128);
    }

    #[test] fn i16_to_i8_overflow_negative() {
        let data = make_data(&[-129i16]);
        let err = convert_scalars(&data, ScalarType::I16, ScalarType::I8).unwrap_err();
        assert_eq!(err.value, -129);
    }

    #[test] fn u32_to_u8_overflow() {
        let data = make_data(&[0u32, 256]);
        let err = convert_scalars(&data, ScalarType::U32, ScalarType::U8).unwrap_err();
        assert_eq!(err.ordinal, 1);
        assert_eq!(err.value, 256);
    }

    #[test] fn i32_to_i8_overflow() {
        let data = make_data(&[129i32]);
        assert!(convert_scalars(&data, ScalarType::I32, ScalarType::I8).is_err());
    }

    #[test] fn u64_to_u8_overflow() {
        let data = make_data(&[u64::MAX]);
        assert!(convert_scalars(&data, ScalarType::U64, ScalarType::U8).is_err());
    }

    #[test] fn i64_to_i16_overflow() {
        let data = make_data(&[i64::MAX]);
        assert!(convert_scalars(&data, ScalarType::I64, ScalarType::I16).is_err());
    }

    // ── Signed ↔ unsigned crossings ────────────────────────────────

    #[test] fn u8_to_i8_ok() {
        let data = make_data(&[0u8, 127]);
        let out = convert_scalars(&data, ScalarType::U8, ScalarType::I8).unwrap();
        assert_eq!(out, make_data(&[0i8, 127]));
    }

    #[test] fn u8_to_i8_overflow() {
        let data = make_data(&[128u8]);
        assert!(convert_scalars(&data, ScalarType::U8, ScalarType::I8).is_err());
    }

    #[test] fn i8_to_u8_ok() {
        let data = make_data(&[0i8, 127]);
        let out = convert_scalars(&data, ScalarType::I8, ScalarType::U8).unwrap();
        assert_eq!(out, make_data(&[0u8, 127]));
    }

    #[test] fn i8_to_u8_negative_fails() {
        let data = make_data(&[-1i8]);
        assert!(convert_scalars(&data, ScalarType::I8, ScalarType::U8).is_err());
    }

    #[test] fn u16_to_i16_ok() {
        let data = make_data(&[0u16, 32767]);
        convert_scalars(&data, ScalarType::U16, ScalarType::I16).unwrap();
    }

    #[test] fn u16_to_i16_overflow() {
        let data = make_data(&[32768u16]);
        assert!(convert_scalars(&data, ScalarType::U16, ScalarType::I16).is_err());
    }

    #[test] fn i16_to_u16_negative_fails() {
        let data = make_data(&[-1i16]);
        assert!(convert_scalars(&data, ScalarType::I16, ScalarType::U16).is_err());
    }

    #[test] fn u32_to_i32_ok() {
        let data = make_data(&[0u32, i32::MAX as u32]);
        convert_scalars(&data, ScalarType::U32, ScalarType::I32).unwrap();
    }

    #[test] fn u32_to_i32_overflow() {
        let data = make_data(&[u32::MAX]);
        assert!(convert_scalars(&data, ScalarType::U32, ScalarType::I32).is_err());
    }

    #[test] fn i32_to_u32_negative_fails() {
        let data = make_data(&[-1i32]);
        assert!(convert_scalars(&data, ScalarType::I32, ScalarType::U32).is_err());
    }

    #[test] fn u64_to_i64_ok() {
        let data = make_data(&[0u64, i64::MAX as u64]);
        convert_scalars(&data, ScalarType::U64, ScalarType::I64).unwrap();
    }

    #[test] fn u64_to_i64_overflow() {
        let data = make_data(&[u64::MAX]);
        assert!(convert_scalars(&data, ScalarType::U64, ScalarType::I64).is_err());
    }

    #[test] fn i64_to_u64_negative_fails() {
        let data = make_data(&[-1i64]);
        assert!(convert_scalars(&data, ScalarType::I64, ScalarType::U64).is_err());
    }

    // ── Full 8×8 matrix: boundary values ───────────────────────────

    /// For every (from, to) pair, verify that the conversion of a zero
    /// value always succeeds (zero is representable in all types).
    #[test]
    fn full_matrix_zero_succeeds() {
        let types = [
            ScalarType::U8, ScalarType::I8, ScalarType::U16, ScalarType::I16,
            ScalarType::U32, ScalarType::I32, ScalarType::U64, ScalarType::I64,
        ];
        for &from in &types {
            // Create a zero value of the source type
            let data = vec![0u8; from.size()];
            for &to in &types {
                let result = convert_scalars(&data, from, to);
                assert!(result.is_ok(), "zero {} -> {} should succeed", from, to);
                // The output should also be all zeros
                let out = result.unwrap();
                assert_eq!(out.len(), to.size());
                assert!(out.iter().all(|&b| b == 0), "zero {} -> {} not zero", from, to);
            }
        }
    }

    /// For every (from, to) pair where from.size() > to.size(),
    /// verify that the max value of `from` fails conversion to `to`.
    #[test]
    fn full_matrix_max_narrowing_fails() {
        let type_max: &[(ScalarType, i128)] = &[
            (ScalarType::U8, u8::MAX as i128),
            (ScalarType::I8, i8::MAX as i128),
            (ScalarType::U16, u16::MAX as i128),
            (ScalarType::I16, i16::MAX as i128),
            (ScalarType::U32, u32::MAX as i128),
            (ScalarType::I32, i32::MAX as i128),
            (ScalarType::U64, u64::MAX as i128),
            (ScalarType::I64, i64::MAX as i128),
        ];

        for &(from, max_val) in type_max {
            for &(to, _) in type_max {
                if from == to { continue; }
                let data = write_value(max_val, from);
                let result = convert_scalars(&data, from, to);
                let should_fit = fits(max_val, to);
                if should_fit {
                    assert!(result.is_ok(),
                        "max({})={} -> {} should fit", from, max_val, to);
                } else {
                    assert!(result.is_err(),
                        "max({})={} -> {} should NOT fit", from, max_val, to);
                }
            }
        }
    }

    /// Signed types: verify that min negative values fail unsigned targets.
    #[test]
    fn full_matrix_min_signed_fails_unsigned() {
        let signed_mins: &[(ScalarType, i128)] = &[
            (ScalarType::I8, i8::MIN as i128),
            (ScalarType::I16, i16::MIN as i128),
            (ScalarType::I32, i32::MIN as i128),
            (ScalarType::I64, i64::MIN as i128),
        ];
        let unsigned = [ScalarType::U8, ScalarType::U16, ScalarType::U32, ScalarType::U64];

        for &(from, min_val) in signed_mins {
            let data = write_value(min_val, from);
            for &to in &unsigned {
                let result = convert_scalars(&data, from, to);
                assert!(result.is_err(),
                    "min({})={} -> {} should fail (negative to unsigned)", from, min_val, to);
            }
        }
    }

    // ── Empty input ────────────────────────────────────────────────

    #[test]
    fn empty_input() {
        let out = convert_scalars(&[], ScalarType::U8, ScalarType::U16).unwrap();
        assert!(out.is_empty());
    }

    // ── Error message formatting ───────────────────────────────────

    #[test]
    fn error_display() {
        let err = ConvertError {
            ordinal: 42, value: 256, from_type: ScalarType::U16, to_type: ScalarType::U8,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("42"));
        assert!(msg.contains("256"));
        assert!(msg.contains("u16"));
        assert!(msg.contains("u8"));
    }
}
