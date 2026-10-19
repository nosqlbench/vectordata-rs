// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Checked conversion of a packed column between element types.
//!
//! The conversion table ([`ElementType::conversion_to`]) says what a
//! *reader* may do, and a reader never narrows. A converter is asked
//! to: it writes a new column, so an integer column may be narrowed
//! when every value fits, with the first value that does not named by
//! ordinal. Everything the table refuses for a reason other than
//! integer width — a float to an integer, a float to a narrower float,
//! an integer to a float that would round it — is refused here too.

use std::fmt;

use veks_datatypes::{Element, ElementConversion, ElementType, Value, convert, dispatch_element};

/// The element types a column may hold. The name predates the shared
/// model; the type is [`ElementType`].
#[deprecated(note = "use `ElementType`")]
pub type ScalarType = ElementType;

/// Why a conversion did not produce a column.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ConvertError {
    /// A value of the source type does not fit the target type.
    Overflow {
        /// The ordinal of the value, counting from 0.
        ordinal: u64,
        /// The value as the source holds it.
        value: Value,
        /// The source element type.
        from_type: ElementType,
        /// The target element type.
        to_type: ElementType,
    },
    /// No conversion between the two types is defined: a float to an
    /// integer, a float to a narrower float, or an integer to a float
    /// that cannot hold every value of it.
    Refused {
        /// The source element type.
        from_type: ElementType,
        /// The target element type.
        to_type: ElementType,
    },
    /// The input is not a whole number of source elements.
    Length {
        /// The input's length in bytes.
        bytes: usize,
        /// The source element type.
        from_type: ElementType,
    },
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow { ordinal, value, from_type, to_type } => write!(
                f,
                "narrowing overflow at ordinal {ordinal}: value {value} ({from_type}) does not fit in {to_type}"
            ),
            Self::Refused { from_type, to_type } => {
                write!(f, "no conversion from {from_type} to {to_type}")
            }
            Self::Length { bytes, from_type } => write!(
                f,
                "{bytes} bytes is not a whole number of {}-byte {from_type} elements",
                from_type.byte_width()
            ),
        }
    }
}

impl std::error::Error for ConvertError {}

/// Whether the converter takes the pair at all: anything the reader
/// table allows, plus integer narrowing, which is checked per value.
fn allowed(from: ElementType, to: ElementType) -> bool {
    from.conversion_to(to) != ElementConversion::Refused || (from.is_integer() && to.is_integer())
}

/// Convert a packed column of `from` elements to a packed column of
/// `to` elements, checking every value.
pub fn convert_scalars(data: &[u8], from: ElementType, to: ElementType) -> Result<Vec<u8>, ConvertError> {
    let from_size = from.byte_width();
    let to_size = to.byte_width();

    if !data.len().is_multiple_of(from_size) {
        return Err(ConvertError::Length { bytes: data.len(), from_type: from });
    }
    if from == to {
        return Ok(data.to_vec());
    }
    if !allowed(from, to) {
        return Err(ConvertError::Refused { from_type: from, to_type: to });
    }

    let count = data.len() / from_size;
    let mut out = vec![0u8; count * to_size];
    dispatch_element!(from, F => dispatch_element!(to, T => {
        for (i, (src, dst)) in data.chunks_exact(from_size).zip(out.chunks_exact_mut(to_size)).enumerate() {
            // Qualified: inside a dispatch arm `F` is a type alias, and
            // the primitives have an inherent `from_le_bytes` of their own.
            let value = <F as Element>::from_le_bytes(src);
            let converted: T = convert::<F, T>(value).map_err(|e| ConvertError::Overflow {
                ordinal: i as u64,
                value: e.value,
                from_type: from,
                to_type: to,
            })?;
            converted.write_le_bytes(dst);
        }
    }));
    Ok(out)
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    fn make_data<T: Element>(vals: &[T]) -> Vec<u8> {
        let mut out = vec![0u8; vals.len() * T::SIZE];
        for (v, slot) in vals.iter().zip(out.chunks_exact_mut(T::SIZE)) {
            v.write_le_bytes(slot);
        }
        out
    }

    /// One value of an integer type as its bytes.
    fn write_value(val: i128, ty: ElementType) -> Vec<u8> {
        dispatch_element!(ty, T => make_data(&[T::from_value(Value::Int(val)).expect("fits")]))
    }

    const INTEGERS: [ElementType; 8] = [
        ElementType::U8, ElementType::I8, ElementType::U16, ElementType::I16,
        ElementType::U32, ElementType::I32, ElementType::U64, ElementType::I64,
    ];

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

    // ── Widening ───────────────────────────────────────────────────

    fn widens<F: Element, T: Element + PartialEq + std::fmt::Debug>(input: &[F], want: &[T]) {
        let out = convert_scalars(&make_data(input), F::TYPE, T::TYPE).unwrap();
        assert_eq!(out, make_data(want), "{} -> {}", F::TYPE, T::TYPE);
    }

    #[test]
    fn u8_to_u16() { widens(&[0u8, 127, 255], &[0u16, 127, 255]); }
    #[test]
    fn u8_to_u32() { widens(&[0u8, 255], &[0u32, 255]); }
    #[test]
    fn u8_to_u64() { widens(&[255u8], &[255u64]); }
    #[test]
    fn u8_to_i16() { widens(&[0u8, 255], &[0i16, 255]); }
    #[test]
    fn u8_to_i32() { widens(&[255u8], &[255i32]); }
    #[test]
    fn u8_to_i64() { widens(&[255u8], &[255i64]); }
    #[test]
    fn i8_to_i16() { widens(&[-128i8, 127], &[-128i16, 127]); }
    #[test]
    fn i8_to_i32() { widens(&[-128i8], &[-128i32]); }
    #[test]
    fn i8_to_i64() { widens(&[-1i8], &[-1i64]); }
    #[test]
    fn u16_to_u32() { widens(&[u16::MAX], &[65535u32]); }
    #[test]
    fn u16_to_u64() { widens(&[u16::MAX], &[65535u64]); }
    #[test]
    fn i16_to_i32() { widens(&[i16::MIN, i16::MAX], &[-32768i32, 32767]); }
    #[test]
    fn u32_to_u64() { widens(&[u32::MAX], &[u32::MAX as u64]); }
    #[test]
    fn i32_to_i64() { widens(&[i32::MIN], &[i32::MIN as i64]); }
    #[test]
    fn u8_to_f32_and_bf16() {
        widens(&[0u8, 1, 255], &[0.0f32, 1.0, 255.0]);
        widens(&[0u8, 255], &[half::bf16::from_f32(0.0), half::bf16::from_f32(255.0)]);
    }
    #[test]
    fn f16_and_bf16_to_f32() {
        widens(&[half::f16::from_f32(1.5), half::f16::MAX], &[1.5f32, half::f16::MAX.to_f32()]);
        widens(&[half::bf16::from_f32(-3.0e30)], &[half::bf16::from_f32(-3.0e30).to_f32()]);
    }

    // ── Narrowing that fits ────────────────────────────────────────

    #[test]
    fn u16_to_u8_ok() { widens(&[0u16, 255], &[0u8, 255]); }
    #[test]
    fn i16_to_i8_ok() { widens(&[-128i16, 127], &[-128i8, 127]); }
    #[test]
    fn u32_to_u16_ok() { widens(&[65535u32], &[65535u16]); }
    #[test]
    fn i32_to_i16_ok() { widens(&[-32768i32], &[-32768i16]); }
    #[test]
    fn u64_to_u32_ok() { widens(&[u32::MAX as u64], &[u32::MAX]); }
    #[test]
    fn i64_to_i32_ok() { widens(&[i32::MIN as i64], &[i32::MIN]); }

    // ── Narrowing that overflows ───────────────────────────────────

    fn overflows<F: Element>(input: &[F], to: ElementType, ordinal: u64) {
        match convert_scalars(&make_data(input), F::TYPE, to) {
            Err(ConvertError::Overflow { ordinal: o, from_type, to_type, .. }) => {
                assert_eq!((o, from_type, to_type), (ordinal, F::TYPE, to));
            }
            other => panic!("{} -> {to}: expected an overflow, got {other:?}", F::TYPE),
        }
    }

    #[test]
    fn u16_to_u8_overflow() { overflows(&[1u16, 256], ElementType::U8, 1); }
    #[test]
    fn i16_to_i8_overflow_positive() { overflows(&[128i16], ElementType::I8, 0); }
    #[test]
    fn i16_to_i8_overflow_negative() { overflows(&[0i16, 0, -129], ElementType::I8, 2); }
    #[test]
    fn u32_to_u8_overflow() { overflows(&[256u32], ElementType::U8, 0); }
    #[test]
    fn i32_to_i8_overflow() { overflows(&[i32::MAX], ElementType::I8, 0); }
    #[test]
    fn u64_to_u8_overflow() { overflows(&[u64::MAX], ElementType::U8, 0); }
    #[test]
    fn i64_to_i16_overflow() { overflows(&[40_000i64], ElementType::I16, 0); }

    // ── Sign changes ───────────────────────────────────────────────

    #[test]
    fn u8_to_i8_ok() { widens(&[0u8, 127], &[0i8, 127]); }
    #[test]
    fn u8_to_i8_overflow() { overflows(&[128u8], ElementType::I8, 0); }
    #[test]
    fn i8_to_u8_ok() { widens(&[0i8, 127], &[0u8, 127]); }
    #[test]
    fn i8_to_u8_negative_fails() { overflows(&[-1i8], ElementType::U8, 0); }
    #[test]
    fn u16_to_i16_ok() { widens(&[32767u16], &[32767i16]); }
    #[test]
    fn u16_to_i16_overflow() { overflows(&[32768u16], ElementType::I16, 0); }
    #[test]
    fn i16_to_u16_negative_fails() { overflows(&[-1i16], ElementType::U16, 0); }
    #[test]
    fn u32_to_i32_ok() { widens(&[i32::MAX as u32], &[i32::MAX]); }
    #[test]
    fn u32_to_i32_overflow() { overflows(&[u32::MAX], ElementType::I32, 0); }
    #[test]
    fn i32_to_u32_negative_fails() { overflows(&[-1i32], ElementType::U32, 0); }
    #[test]
    fn u64_to_i64_ok() { widens(&[i64::MAX as u64], &[i64::MAX]); }
    #[test]
    fn u64_to_i64_overflow() { overflows(&[u64::MAX], ElementType::I64, 0); }
    #[test]
    fn i64_to_u64_negative_fails() { overflows(&[-1i64], ElementType::U64, 0); }

    // ── Refused pairs ──────────────────────────────────────────────

    #[test]
    fn floats_never_become_integers_and_never_narrow() {
        for (from, to) in [
            (ElementType::F32, ElementType::I32),
            (ElementType::F64, ElementType::F32),
            (ElementType::F16, ElementType::BF16),
            (ElementType::BF16, ElementType::F16),
            (ElementType::I32, ElementType::F32),
            (ElementType::U16, ElementType::BF16),
        ] {
            let data = vec![0u8; from.byte_width()];
            assert_eq!(
                convert_scalars(&data, from, to),
                Err(ConvertError::Refused { from_type: from, to_type: to }),
                "{from} -> {to}"
            );
        }
    }

    // ── The whole integer matrix ───────────────────────────────────

    #[test]
    fn full_matrix_zero_succeeds() {
        for &from in &INTEGERS {
            let data = vec![0u8; from.byte_width()];
            for &to in &INTEGERS {
                let out = convert_scalars(&data, from, to).unwrap_or_else(|e| panic!("zero {from} -> {to}: {e}"));
                assert_eq!(out.len(), to.byte_width());
                assert!(out.iter().all(|&b| b == 0), "zero {from} -> {to} not zero");
            }
        }
    }

    #[test]
    fn full_matrix_max_narrowing_fails() {
        let type_max: &[(ElementType, i128)] = &[
            (ElementType::U8, u8::MAX as i128),
            (ElementType::I8, i8::MAX as i128),
            (ElementType::U16, u16::MAX as i128),
            (ElementType::I16, i16::MAX as i128),
            (ElementType::U32, u32::MAX as i128),
            (ElementType::I32, i32::MAX as i128),
            (ElementType::U64, u64::MAX as i128),
            (ElementType::I64, i64::MAX as i128),
        ];
        for &(from, max_val) in type_max {
            for &(to, to_max) in type_max {
                if from == to {
                    continue;
                }
                let data = write_value(max_val, from);
                let result = convert_scalars(&data, from, to);
                if max_val > to_max {
                    assert!(matches!(result, Err(ConvertError::Overflow { ordinal: 0, .. })), "max({from}) -> {to}");
                } else {
                    assert!(result.is_ok(), "max({from})={max_val} -> {to} fits");
                }
            }
        }
    }

    #[test]
    fn full_matrix_min_signed_fails_unsigned() {
        let signed_mins: &[(ElementType, i128)] = &[
            (ElementType::I8, i8::MIN as i128),
            (ElementType::I16, i16::MIN as i128),
            (ElementType::I32, i32::MIN as i128),
            (ElementType::I64, i64::MIN as i128),
        ];
        let unsigned = [ElementType::U8, ElementType::U16, ElementType::U32, ElementType::U64];
        for &(from, min_val) in signed_mins {
            let data = write_value(min_val, from);
            for &to in &unsigned {
                let result = convert_scalars(&data, from, to);
                assert!(
                    matches!(result, Err(ConvertError::Overflow { ordinal: 0, value: Value::Int(v), .. }) if v == min_val),
                    "min({from})={min_val} -> {to} should fail (negative to unsigned)"
                );
            }
        }
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn empty_input() {
        let out = convert_scalars(&[], ElementType::U8, ElementType::I64).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn a_partial_element_is_a_length_error() {
        assert_eq!(
            convert_scalars(&[0u8; 5], ElementType::U32, ElementType::U64),
            Err(ConvertError::Length { bytes: 5, from_type: ElementType::U32 })
        );
    }

    #[test]
    fn error_display() {
        let err = ConvertError::Overflow {
            ordinal: 42,
            value: Value::Int(256),
            from_type: ElementType::U16,
            to_type: ElementType::U8,
        };
        let msg = err.to_string();
        assert!(msg.contains("42"));
        assert!(msg.contains("256"));
        assert!(msg.contains("u16"));
        assert!(msg.contains("u8"));
        assert_eq!(
            ConvertError::Refused { from_type: ElementType::F32, to_type: ElementType::I32 }.to_string(),
            "no conversion from f32 to i32"
        );
        let length = ConvertError::Length { bytes: 5, from_type: ElementType::U32 };
        assert_eq!(length.to_string(), "5 bytes is not a whole number of 4-byte u32 elements");
        let boxed: Box<dyn std::error::Error> = Box::new(length);
        assert!(boxed.to_string().contains("u32"));
    }
}
