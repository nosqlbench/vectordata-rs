// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. The [`Element`] trait: what a reader needs of an
//! element type, implemented for each of the twelve.

use crate::ElementType;

/// A value of some element type, as the conversion path carries it.
///
/// Integers travel as `i128`, which holds every integer element
/// exactly; floats travel as `f64`, which holds every `f16`, `bf16`
/// and `f32` exactly. This is how [`convert`](crate::convert) moves a
/// value between types; the native path never builds one.
///
/// ```
/// use veks_datatypes::{Element, Value, f16};
/// assert_eq!(300u32.to_value(), Value::Int(300));
/// assert_eq!(f16::from_f32(0.5).to_value(), Value::Float(0.5));
/// assert_eq!(Value::Int(-7).to_string(), "-7");
/// assert_eq!(u8::from_value(Value::Int(255)), Some(255));
/// assert_eq!(i32::from_value(Value::Int(1 << 40)), None, "does not fit");
/// assert_eq!(i64::from_value(Value::Float(1.0)), None, "a float never becomes an integer");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    /// An integer element.
    Int(i128),
    /// A floating point element.
    Float(f64),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(x) => write!(f, "{x}"),
        }
    }
}

/// An element type as a reader sees it.
///
/// This is the one bound a reader generic over its element puts on
/// the type parameter. The constants say what the type is and how wide
/// it is on disk; the byte methods are the little-endian encoding
/// every file uses; the value methods are the conversion path, used
/// only when a caller asks for a type the file does not hold and the
/// [conversion table](ElementType::conversion_to) allows it.
///
/// ```
/// use veks_datatypes::{bf16, f16, Element, ElementType, Value};
/// assert_eq!((<i16 as Element>::TYPE, <i16 as Element>::SIZE), (ElementType::I16, 2));
/// let mut buf = [0u8; 8];
/// (-2i16).write_le_bytes(&mut buf);                       // the on-disk encoding: little-endian
/// assert_eq!((&buf[..2], <i16 as Element>::from_le_bytes(&buf)), (&[0xfe, 0xff][..], -2));
/// bf16::from_f32(1.0).write_le_bytes(&mut buf);
/// assert_eq!(&buf[..2], &[0x80, 0x3f], "a bf16 is the top half of its f32");
/// f16::from_f32(1.0).write_le_bytes(&mut buf);
/// assert_eq!(&buf[..2], &[0x00, 0x3c], "an f16 is not");
/// assert_eq!(u64::MAX.to_value(), Value::Int(u64::MAX as i128));   // the conversion path, never the native one
/// assert_eq!(f16::from_value(Value::Int(2048)), Some(f16::from_f32(2048.0)));
/// assert_eq!(i8::from_value(Value::Int(200)), None, "does not fit");
/// assert_eq!(bf16::from_f32(-0.5).as_f64(), -0.5);
/// ```
pub trait Element: Copy + Send + Sync + 'static {
    /// Which element type this is.
    const TYPE: ElementType;
    /// The width of one element in bytes, equal to `TYPE.byte_width()`.
    const SIZE: usize;

    /// Decode one element from its little-endian bytes. `bytes` holds
    /// at least [`SIZE`](Self::SIZE) bytes; more are ignored.
    fn from_le_bytes(bytes: &[u8]) -> Self;

    /// Encode one element as little-endian bytes into `out`, which
    /// holds at least [`SIZE`](Self::SIZE) bytes.
    fn write_le_bytes(self, out: &mut [u8]);

    /// This element as a [`Value`], for conversion to another type.
    fn to_value(self) -> Value;

    /// The element this [`Value`] converts to, or `None` when it does
    /// not fit: an integer out of range, or a float asked of an
    /// integer type. An integer asked of a float type is converted as
    /// `as` does, which is exact where the
    /// [conversion table](ElementType::conversion_to) allows it.
    fn from_value(value: Value) -> Option<Self>;

    /// This element as an `f64`, as `as` converts it: exact for every
    /// float and for integers up to 53 bits, rounded above that.
    fn as_f64(self) -> f64;
}

macro_rules! int_element {
    ($($t:ty => $ty:ident),* $(,)?) => {$(
        impl Element for $t {
            const TYPE: ElementType = ElementType::$ty;
            const SIZE: usize = std::mem::size_of::<$t>();

            #[inline]
            fn from_le_bytes(bytes: &[u8]) -> Self {
                <$t>::from_le_bytes(bytes[..Self::SIZE].try_into().expect("checked width"))
            }

            #[inline]
            fn write_le_bytes(self, out: &mut [u8]) {
                out[..Self::SIZE].copy_from_slice(&self.to_le_bytes());
            }

            #[inline]
            fn to_value(self) -> Value {
                Value::Int(self as i128)
            }

            #[inline]
            fn from_value(value: Value) -> Option<Self> {
                match value {
                    Value::Int(i) => <$t>::try_from(i).ok(),
                    Value::Float(_) => None,
                }
            }

            #[inline]
            fn as_f64(self) -> f64 {
                self as f64
            }
        }
    )*};
}

int_element! {
    u8 => U8, i8 => I8, u16 => U16, i16 => I16,
    u32 => U32, i32 => I32, u64 => U64, i64 => I64,
}

macro_rules! float_element {
    ($($t:ty => $ty:ident, $from_f64:expr, $to_f64:expr),* $(,)?) => {$(
        impl Element for $t {
            const TYPE: ElementType = ElementType::$ty;
            const SIZE: usize = std::mem::size_of::<$t>();

            #[inline]
            fn from_le_bytes(bytes: &[u8]) -> Self {
                <$t>::from_le_bytes(bytes[..Self::SIZE].try_into().expect("checked width"))
            }

            #[inline]
            fn write_le_bytes(self, out: &mut [u8]) {
                out[..Self::SIZE].copy_from_slice(&self.to_le_bytes());
            }

            #[inline]
            fn to_value(self) -> Value {
                Value::Float(($to_f64)(self))
            }

            #[inline]
            fn from_value(value: Value) -> Option<Self> {
                let x = match value {
                    Value::Int(i) => i as f64,
                    Value::Float(x) => x,
                };
                Some(($from_f64)(x))
            }

            #[inline]
            fn as_f64(self) -> f64 {
                ($to_f64)(self)
            }
        }
    )*};
}

float_element! {
    half::f16 => F16, half::f16::from_f64, half::f16::to_f64,
    half::bf16 => BF16, half::bf16::from_f64, half::bf16::to_f64,
    f32 => F32, |x: f64| x as f32, |x: f32| x as f64,
    f64 => F64, |x: f64| x, |x: f64| x,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ElementKind, dispatch_element};

    #[test]
    fn every_type_states_its_own_width() {
        for &ty in ElementType::ALL {
            let size = dispatch_element!(ty, T => <T as Element>::SIZE);
            assert_eq!(size, ty.byte_width(), "{ty}");
        }
    }

    fn bytes_round_trip<T: Element + PartialEq + std::fmt::Debug>(values: &[T]) {
        let mut buf = vec![0u8; T::SIZE + 3];
        for &v in values {
            v.write_le_bytes(&mut buf);
            assert_eq!(T::from_le_bytes(&buf), v);
        }
    }

    #[test]
    fn bytes_round_trip_for_every_type() {
        bytes_round_trip(&[0u8, 1, 200, u8::MAX]);
        bytes_round_trip(&[0i8, -1, 100, i8::MIN, i8::MAX]);
        bytes_round_trip(&[0u16, 0x1234, u16::MAX]);
        bytes_round_trip(&[0i16, -2, i16::MIN, i16::MAX]);
        bytes_round_trip(&[0u32, 0x1234_5678, u32::MAX]);
        bytes_round_trip(&[0i32, -3, i32::MIN, i32::MAX]);
        bytes_round_trip(&[0u64, 0x1234_5678_9abc_def0, u64::MAX]);
        bytes_round_trip(&[0i64, -4, i64::MIN, i64::MAX]);
        bytes_round_trip(&[half::f16::from_f32(1.5), half::f16::from_f32(-0.25), half::f16::MAX]);
        bytes_round_trip(&[half::bf16::from_f32(1.5), half::bf16::from_f32(-3.0e30), half::bf16::MAX]);
        bytes_round_trip(&[0.0f32, -1.5, f32::MAX, f32::MIN_POSITIVE]);
        bytes_round_trip(&[0.0f64, -1.5, f64::MAX, f64::MIN_POSITIVE]);
    }

    #[test]
    fn little_endian_is_the_encoding() {
        let mut buf = [0u8; 4];
        0x0403_0201u32.write_le_bytes(&mut buf);
        assert_eq!(buf, [1, 2, 3, 4]);
        1.0f32.write_le_bytes(&mut buf);
        assert_eq!(buf, [0, 0, 0x80, 0x3f]);
        half::bf16::from_f32(1.0).write_le_bytes(&mut buf[..2]);
        assert_eq!(&buf[..2], &[0x80, 0x3f], "bf16 is the top half of the f32");
        half::f16::from_f32(1.0).write_le_bytes(&mut buf[..2]);
        assert_eq!(&buf[..2], &[0x00, 0x3c]);
    }

    #[test]
    fn integers_refuse_what_does_not_fit() {
        assert_eq!(u8::from_value(Value::Int(255)), Some(255));
        assert_eq!(u8::from_value(Value::Int(256)), None);
        assert_eq!(u8::from_value(Value::Int(-1)), None);
        assert_eq!(i8::from_value(Value::Int(-128)), Some(-128));
        assert_eq!(i8::from_value(Value::Int(128)), None);
        assert_eq!(u64::from_value(Value::Int(i128::from(u64::MAX))), Some(u64::MAX));
        assert_eq!(u64::from_value(Value::Int(i128::from(u64::MAX) + 1)), None);
        assert_eq!(i64::from_value(Value::Int(i128::from(i64::MIN))), Some(i64::MIN));
        assert_eq!(i32::from_value(Value::Float(1.0)), None, "a float never becomes an integer");
    }

    /// The bounds of each integer type, by its width and sign.
    fn bounds(ty: ElementType) -> (i128, i128) {
        let bits = 8 * ty.byte_width() as u32;
        match ty.kind() {
            ElementKind::Unsigned => (0, (1i128 << bits) - 1),
            ElementKind::Signed => (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1),
            ElementKind::Float(_) => unreachable!("not an integer"),
        }
    }

    /// **Every integer type takes exactly its own range**: its minimum
    /// and maximum fit, one past either does not, and no float ever
    /// becomes an integer.
    #[test]
    fn every_integer_type_takes_exactly_its_range() {
        for &ty in ElementType::ALL.iter().filter(|t| t.is_integer()) {
            let (lo, hi) = bounds(ty);
            dispatch_element!(ty, T => {
                assert_eq!(T::from_value(Value::Int(lo)).map(|v| v.to_value()), Some(Value::Int(lo)), "{ty} min");
                assert_eq!(T::from_value(Value::Int(hi)).map(|v| v.to_value()), Some(Value::Int(hi)), "{ty} max");
                assert!(T::from_value(Value::Int(lo - 1)).is_none(), "{ty}: one below the minimum");
                assert!(T::from_value(Value::Int(hi + 1)).is_none(), "{ty}: one above the maximum");
                assert!(T::from_value(Value::Float(1.0)).is_none(), "{ty}: a float never becomes an integer");
                assert_eq!(T::from_value(Value::Int(0)).unwrap().as_f64(), 0.0, "{ty}");
            });
        }
    }

    /// **A float takes a value as `as` does**: exactly where the
    /// conversion table admits the source (that is the table's promise,
    /// not this method's), rounded otherwise — so an inexact integer
    /// into a float is the nearest float, never `None`, and the table
    /// is what keeps such a pair from being read.
    #[test]
    fn a_float_takes_a_value_as_as_does() {
        assert_eq!(half::f16::from_value(Value::Int(2049)), Some(half::f16::from_f64(2049.0)), "rounded, as `as`");
        assert_ne!(half::f16::from_value(Value::Int(2049)).unwrap().as_f64(), 2049.0, "and 2049 is not exact in f16");
        assert_eq!(half::bf16::from_value(Value::Int(257)), Some(half::bf16::from_f64(257.0)));
        assert_ne!(half::bf16::from_value(Value::Int(257)).unwrap().as_f64(), 257.0);
        assert_eq!(f32::from_value(Value::Int(16_777_217)), Some(16_777_217i32 as f32));
        assert_ne!(f32::from_value(Value::Int(16_777_217)).unwrap().as_f64(), 16_777_217.0);
        assert_eq!(f64::from_value(Value::Int(1 << 60)), Some((1i64 << 60) as f64), "exact: a power of two");
        for &ty in ElementType::ALL.iter().filter(|t| t.is_float()) {
            dispatch_element!(ty, T => {
                assert!(T::from_value(Value::Float(f64::NAN)).unwrap().as_f64().is_nan(), "{ty} takes a NaN");
                assert_eq!(T::from_value(Value::Float(f64::INFINITY)).unwrap().as_f64(), f64::INFINITY, "{ty} takes an infinity");
                assert_eq!(T::from_value(Value::Float(-0.0)).unwrap().as_f64().to_bits(), (-0.0f64).to_bits(), "{ty} keeps the sign of zero");
                assert_eq!(T::from_value(Value::Float(1e300)).unwrap().as_f64().is_infinite(), ty != ElementType::F64, "{ty}: out of range saturates to an infinity");
            });
        }
    }

    #[test]
    fn floats_take_integers_exactly_where_they_fit() {
        assert_eq!(f32::from_value(Value::Int(16_777_216)), Some(16_777_216.0));
        assert_eq!(half::bf16::from_value(Value::Int(255)), Some(half::bf16::from_f32(255.0)));
        assert_eq!(half::bf16::from_value(Value::Int(-128)), Some(half::bf16::from_f32(-128.0)));
        assert_eq!(half::f16::from_value(Value::Int(2048)), Some(half::f16::from_f32(2048.0)));
        assert_eq!(f64::from_value(Value::Int(i128::from(u32::MAX))), Some(4_294_967_295.0));
        assert_eq!(f64::from_value(Value::Float(2.5)), Some(2.5));
    }

    #[test]
    fn values_and_f64_agree() {
        assert_eq!(7u8.to_value(), Value::Int(7));
        assert_eq!((-7i64).to_value(), Value::Int(-7));
        assert_eq!(1.5f32.to_value(), Value::Float(1.5));
        assert_eq!(half::bf16::from_f32(-2.0).to_value(), Value::Float(-2.0));
        assert_eq!(half::f16::from_f32(0.5).as_f64(), 0.5);
        assert_eq!(u64::MAX.as_f64(), u64::MAX as f64);
        assert_eq!(Value::Int(3).to_string(), "3");
        assert_eq!(Value::Float(0.5).to_string(), "0.5");
    }
}
