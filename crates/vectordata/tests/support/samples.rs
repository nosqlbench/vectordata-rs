// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

//! Per-type sample values for the tests that iterate `ElementType::ALL`
//! (SRD element access, §13.10): one list per type, with the values a
//! reader or a conversion must get right — zero, one, the extremes, a
//! value in between, and for a float a negative, a subnormal, both
//! infinities and a NaN.

use vectordata::typed_access::{Element, bf16, f16};

/// The sample values of an element type.
pub trait Samples: Element {
    /// Zero, one, the extremes, a mid value; for floats also a
    /// negative, a subnormal, both infinities and a NaN.
    fn samples() -> Vec<Self>;
}

macro_rules! int_samples {
    ($($t:ty),*) => {$(
        impl Samples for $t {
            fn samples() -> Vec<Self> {
                vec![0, 1, <$t>::MIN, <$t>::MAX, <$t>::MAX / 3]
            }
        }
    )*};
}
int_samples!(u8, i8, u16, i16, u32, i32, u64, i64);

macro_rules! float_samples {
    ($($t:ty),*) => {$(
        impl Samples for $t {
            fn samples() -> Vec<Self> {
                vec![0.0, 1.0, -1.5, <$t>::MIN, <$t>::MAX, <$t>::MIN_POSITIVE / 2.0, <$t>::INFINITY, <$t>::NEG_INFINITY, <$t>::NAN]
            }
        }
    )*};
}
float_samples!(f32, f64);

impl Samples for f16 {
    fn samples() -> Vec<Self> {
        vec![
            f16::ZERO,
            f16::ONE,
            f16::from_f32(-1.5),
            f16::MIN,
            f16::MAX,
            f16::from_bits(1),
            f16::INFINITY,
            f16::NEG_INFINITY,
            f16::NAN,
        ]
    }
}

impl Samples for bf16 {
    fn samples() -> Vec<Self> {
        vec![
            bf16::ZERO,
            bf16::ONE,
            bf16::from_f32(-1.5),
            bf16::MIN,
            bf16::MAX,
            bf16::from_bits(1),
            bf16::INFINITY,
            bf16::NEG_INFINITY,
            bf16::NAN,
        ]
    }
}

/// The samples a computation can consume: no infinity, no NaN.
pub fn finite<T: Samples>() -> Vec<T> {
    T::samples().into_iter().filter(|v| v.as_f64().is_finite()).collect()
}

/// Whether two elements are the same value: the same bytes, or both
/// NaN (a NaN's payload may change across a conversion).
pub fn same<T: Element>(a: T, b: T) -> bool {
    let (mut x, mut y) = (vec![0u8; T::SIZE], vec![0u8; T::SIZE]);
    a.write_le_bytes(&mut x);
    b.write_le_bytes(&mut y);
    x == y || (a.as_f64().is_nan() && b.as_f64().is_nan())
}

/// Whether `wide` is `narrow` widened: equal as f64, or both NaN.
pub fn widened<N: Element, T: Element>(narrow: N, wide: T) -> bool {
    let (n, w) = (narrow.as_f64(), wide.as_f64());
    n.to_bits() == w.to_bits() || (n.is_nan() && w.is_nan())
}

/// `values` as one record each, written as `T` under a dimension
/// header of one, so record `i` holds `values[i]`.
pub fn rows_of<T: Element>(values: &[T]) -> Vec<Vec<T>> {
    values.iter().map(|v| vec![*v]).collect()
}
