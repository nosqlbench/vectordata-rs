// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. The conversion table: what a reader may do when the
//! caller asks for an element type the file does not hold.

use crate::{Element, ElementKind, ElementType, Value};

/// What reading a file of one element type as another means.
///
/// The table is in [`ElementType::conversion_to`]. The rule behind it:
/// the file's own type is read in place; a conversion that can lose
/// nothing widens; a same-width change of sign is checked value by
/// value; everything else is refused, because a reader never
/// reinterprets bytes and never drops precision on the caller's
/// behalf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementConversion {
    /// The requested type is the file's type. Elements are read in
    /// place: a slice of the mapping is a slice of the requested type.
    Native,
    /// Every value of the file's type is a value of the requested
    /// type. Elements are converted one by one and the conversion
    /// never fails.
    Widening,
    /// The same width, the other sign. Elements are converted one by
    /// one and a value outside the requested type's range is an
    /// [`Overflow`].
    CrossSign,
    /// No conversion. The reader refuses to open. Read the file at its
    /// own type and convert what you need yourself.
    Refused,
}

impl ElementConversion {
    /// Whether the reader opens at all.
    pub const fn is_allowed(self) -> bool {
        !matches!(self, Self::Refused)
    }

    /// Whether elements are converted one by one, rather than read in
    /// place.
    pub const fn converts(self) -> bool {
        matches!(self, Self::Widening | Self::CrossSign)
    }
}

impl ElementType {
    /// The bits of magnitude an integer type holds: all of them for an
    /// unsigned type, all but the sign for a signed one.
    const fn magnitude_bits(self) -> u32 {
        let bits = self.byte_width() as u32 * 8;
        match self.kind() {
            ElementKind::Signed => bits - 1,
            _ => bits,
        }
    }

    /// The bits of significand a float type holds, counting the
    /// implicit leading one: the largest integer it represents exactly
    /// is `2^significand`.
    const fn significand_bits(self) -> u32 {
        match self {
            Self::F16 => 11,
            Self::BF16 => 8,
            Self::F32 => 24,
            Self::F64 => 53,
            _ => 0,
        }
    }

    /// What reading a file of this type as `target` means. The table:
    ///
    /// | file → requested | result |
    /// |---|---|
    /// | the same type | `Native` |
    /// | integer → wider integer, source unsigned or target signed | `Widening` |
    /// | integer → integer of the same width, other sign | `CrossSign` |
    /// | signed integer → wider unsigned integer | `CrossSign` |
    /// | integer → narrower integer | `Refused` |
    /// | integer → float that holds every value (`u8`/`i8`/`u16`/`i16` → `f32`; up to 32 bits → `f64`; `u8`/`i8` → `f16` or `bf16`) | `Widening` |
    /// | integer → float that rounds some value (`i32` → `f32`, `i64` → `f64`) | `Refused` |
    /// | `f16` or `bf16` → `f32` or `f64`; `f32` → `f64` | `Widening` |
    /// | `f16` ↔ `bf16` | `Refused` — neither direction is lossless |
    /// | float → narrower float, float → integer | `Refused` |
    pub const fn conversion_to(self, target: ElementType) -> ElementConversion {
        use ElementConversion::*;
        if self as u8 == target as u8 {
            return Native;
        }
        match (self.kind(), target.kind()) {
            (ElementKind::Float(_), ElementKind::Float(_)) => match (self, target) {
                (Self::F16 | Self::BF16, Self::F32 | Self::F64) | (Self::F32, Self::F64) => Widening,
                _ => Refused,
            },
            (ElementKind::Float(_), _) => Refused,
            (_, ElementKind::Float(_)) => {
                if self.magnitude_bits() <= target.significand_bits() {
                    Widening
                } else {
                    Refused
                }
            }
            (source_kind, target_kind) => {
                let (sw, tw) = (self.byte_width(), target.byte_width());
                let same_sign = matches!(
                    (source_kind, target_kind),
                    (ElementKind::Signed, ElementKind::Signed) | (ElementKind::Unsigned, ElementKind::Unsigned)
                );
                if tw < sw {
                    Refused
                } else if tw == sw {
                    // Same width and not the same type: the other sign.
                    CrossSign
                } else if same_sign || matches!(source_kind, ElementKind::Unsigned) {
                    Widening
                } else {
                    // Signed into wider unsigned: negatives do not fit.
                    CrossSign
                }
            }
        }
    }
}

/// A value that does not fit the requested type under a
/// [`CrossSign`](ElementConversion::CrossSign) conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Overflow {
    /// The value as the file holds it.
    pub value: Value,
    /// The type it was asked to become.
    pub target: ElementType,
}

impl std::fmt::Display for Overflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "value {} does not fit in {}", self.value, self.target)
    }
}

impl std::error::Error for Overflow {}

/// Convert one element between types, as a
/// [`Widening`](ElementConversion::Widening) or
/// [`CrossSign`](ElementConversion::CrossSign) reader does. Under
/// `Widening` this never fails; under `CrossSign` a value outside the
/// target's range is an [`Overflow`]. It is not consulted for
/// [`Native`](ElementConversion::Native) reads, and a
/// [`Refused`](ElementConversion::Refused) pair never reaches it
/// because no reader opens.
#[inline]
pub fn convert<N: Element, T: Element>(n: N) -> Result<T, Overflow> {
    let value = n.to_value();
    T::from_value(value).ok_or(Overflow { value, target: T::TYPE })
}

#[cfg(test)]
mod tests {
    use super::ElementConversion::*;
    use super::*;
    use crate::dispatch_element;

    /// The whole table, file type down the side and requested type
    /// across, in `ElementType::ALL` order: a change to the rule shows
    /// here as a diff.
    #[test]
    fn the_conversion_table() {
        use ElementType::*;
        #[rustfmt::skip]
        let table: [(ElementType, [ElementConversion; 12]); 12] = [
            //      U8        I8        U16       I16       U32       I32       U64       I64       F16       BF16      F32       F64
            (U8,  [Native,   CrossSign, Widening, Widening, Widening, Widening, Widening, Widening, Widening, Widening, Widening, Widening]),
            (I8,  [CrossSign, Native,   CrossSign, Widening, CrossSign, Widening, CrossSign, Widening, Widening, Widening, Widening, Widening]),
            (U16, [Refused,  Refused,  Native,   CrossSign, Widening, Widening, Widening, Widening, Refused,  Refused,  Widening, Widening]),
            (I16, [Refused,  Refused,  CrossSign, Native,   CrossSign, Widening, CrossSign, Widening, Refused,  Refused,  Widening, Widening]),
            (U32, [Refused,  Refused,  Refused,  Refused,  Native,   CrossSign, Widening, Widening, Refused,  Refused,  Refused,  Widening]),
            (I32, [Refused,  Refused,  Refused,  Refused,  CrossSign, Native,   CrossSign, Widening, Refused,  Refused,  Refused,  Widening]),
            (U64, [Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Native,   CrossSign, Refused,  Refused,  Refused,  Refused ]),
            (I64, [Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  CrossSign, Native,   Refused,  Refused,  Refused,  Refused ]),
            (F16, [Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Native,   Refused,  Widening, Widening]),
            (BF16,[Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Native,   Widening, Widening]),
            (F32, [Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Native,   Widening]),
            (F64, [Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Refused,  Native  ]),
        ];
        for (file, row) in table {
            for (requested, want) in ElementType::ALL.iter().zip(row) {
                assert_eq!(file.conversion_to(*requested), want, "{file} -> {requested}");
            }
        }
    }

    #[test]
    fn native_is_the_diagonal_and_cross_sign_is_symmetric() {
        for &a in ElementType::ALL {
            for &b in ElementType::ALL {
                let ab = a.conversion_to(b);
                assert_eq!(ab == Native, a == b, "{a} -> {b}");
                if ab == CrossSign && a.byte_width() == b.byte_width() {
                    assert_eq!(b.conversion_to(a), CrossSign, "{b} -> {a}");
                }
                if ab == Widening {
                    assert!(b.conversion_to(a) == Refused || b.conversion_to(a) == CrossSign, "{b} -> {a} undoes a widening");
                }
            }
        }
    }

    /// Every value of a widening source converts exactly and back; the
    /// sources are small enough to walk whole where they are, and are
    /// walked at their edges otherwise.
    #[test]
    fn widening_is_exact_at_the_edges() {
        fn check<N: Element + PartialEq + std::fmt::Debug>(samples: &[N]) {
            for &requested in ElementType::ALL {
                if N::TYPE.conversion_to(requested) != Widening {
                    continue;
                }
                for &n in samples {
                    let back = dispatch_element!(requested, T => {
                        let t: T = convert::<N, T>(n).expect("widening never fails");
                        t.as_f64()
                    });
                    assert_eq!(back.to_bits(), n.as_f64().to_bits(), "{} -> {requested}: {n:?}", N::TYPE);
                }
            }
        }
        check(&(0..=u8::MAX).collect::<Vec<_>>());
        check(&(i8::MIN..=i8::MAX).collect::<Vec<_>>());
        check(&[0u16, 1, 0x7fff, 0x8000, u16::MAX]);
        check(&[i16::MIN, -1, 0, 1, i16::MAX]);
        check(&[0u32, 1, 1 << 24, (1 << 24) + 1, u32::MAX]);
        check(&[i32::MIN, -1, 0, 1, i32::MAX]);
        check(&[0u64, 1, u64::MAX]);
        check(&[i64::MIN, 0, i64::MAX]);
        check(&[half::f16::MIN, half::f16::from_f32(-0.0), half::f16::MIN_POSITIVE, half::f16::MAX]);
        check(&[half::bf16::MIN, half::bf16::from_f32(0.1), half::bf16::MIN_POSITIVE, half::bf16::MAX]);
        check(&[f32::MIN, -0.0, f32::MIN_POSITIVE, 0.1, f32::MAX]);
    }

    #[test]
    fn cross_sign_checks_each_value() {
        assert_eq!(convert::<u8, i8>(127), Ok(127));
        assert_eq!(convert::<u8, i8>(200), Err(Overflow { value: Value::Int(200), target: ElementType::I8 }));
        assert_eq!(convert::<i8, u8>(-1), Err(Overflow { value: Value::Int(-1), target: ElementType::U8 }));
        assert_eq!(convert::<i8, u16>(-1), Err(Overflow { value: Value::Int(-1), target: ElementType::U16 }));
        assert_eq!(convert::<i8, u16>(100), Ok(100));
        assert_eq!(convert::<i32, u32>(-5), Err(Overflow { value: Value::Int(-5), target: ElementType::U32 }));
        assert_eq!(convert::<u64, i64>(u64::MAX), Err(Overflow { value: Value::Int(u64::MAX as i128), target: ElementType::I64 }));
        assert_eq!(convert::<u64, i64>(5), Ok(5));
        assert_eq!(
            convert::<u8, i8>(200).unwrap_err().to_string(),
            "value 200 does not fit in i8"
        );
    }

    #[test]
    fn the_integer_to_float_boundary_is_the_significand() {
        use ElementType::*;
        // 2^24 is exact in f32; 2^24 + 1 is not, so u32 is refused.
        assert_eq!(U16.conversion_to(F32), Widening);
        assert_eq!(U32.conversion_to(F32), Refused);
        assert_eq!(I32.conversion_to(F32), Refused);
        assert_eq!(U32.conversion_to(F64), Widening);
        assert_eq!(I64.conversion_to(F64), Refused);
        // bf16 holds 8 bits of significand: exactly a u8 or an i8.
        assert_eq!(U8.conversion_to(BF16), Widening);
        assert_eq!(I8.conversion_to(BF16), Widening);
        assert_eq!(U16.conversion_to(BF16), Refused);
        // f16 holds 11 bits: a u8 or i8, not a u16.
        assert_eq!(U8.conversion_to(F16), Widening);
        assert_eq!(U16.conversion_to(F16), Refused);
        assert_eq!(convert::<u8, half::bf16>(255), Ok(half::bf16::from_f32(255.0)));
    }

    #[test]
    fn f16_and_bf16_never_convert_to_each_other() {
        assert_eq!(ElementType::F16.conversion_to(ElementType::BF16), Refused);
        assert_eq!(ElementType::BF16.conversion_to(ElementType::F16), Refused);
        assert_eq!(ElementType::U16.conversion_to(ElementType::BF16), Refused);
        assert_eq!(ElementType::BF16.conversion_to(ElementType::U16), Refused);
        assert!(!Refused.is_allowed());
        assert!(Native.is_allowed() && !Native.converts());
        assert!(Widening.converts() && CrossSign.converts());
    }
}
