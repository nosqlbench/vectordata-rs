// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Per-type sample values for the element type matrix (SRD §13.10).
//!
//! Shared by the integration tests (`mod support;`) and by the unit
//! tests of `src/pipeline/element_type.rs`, which include this file by
//! path so that both sides draw the same values.

use veks_datatypes::{Element, bf16, f16};

/// Per-type samples: zero, one, the extremes, a mid value; for floats also a negative, a subnormal, both infinities and a NaN.
pub trait Samples: Element {
    /// The samples of this type, in a fixed order.
    fn samples() -> Vec<Self>;
}

macro_rules! int_samples {
    ($($t:ty),*) => { $(impl Samples for $t { fn samples() -> Vec<Self> { vec![0, 1, <$t>::MIN, <$t>::MAX, <$t>::MAX / 3] } })* }
}
int_samples!(u8, i8, u16, i16, u32, i32, u64, i64);

macro_rules! float_samples {
    ($($t:ty),*) => { $(impl Samples for $t { fn samples() -> Vec<Self> { vec![0.0, 1.0, -1.5, <$t>::MIN, <$t>::MAX, <$t>::MIN_POSITIVE / 2.0, <$t>::INFINITY, <$t>::NEG_INFINITY, <$t>::NAN] } })* }
}
float_samples!(f32, f64);

impl Samples for f16 {
    fn samples() -> Vec<Self> {
        vec![f16::ZERO, f16::ONE, f16::from_f32(-1.5), f16::MIN, f16::MAX, f16::from_bits(1), f16::INFINITY, f16::NEG_INFINITY, f16::NAN]
    }
}

impl Samples for bf16 {
    fn samples() -> Vec<Self> {
        vec![bf16::ZERO, bf16::ONE, bf16::from_f32(-1.5), bf16::MIN, bf16::MAX, bf16::from_bits(1), bf16::INFINITY, bf16::NEG_INFINITY, bf16::NAN]
    }
}

/// Whether two `f64` readings of an element agree: bit for bit, except
/// that a NaN matches any NaN.
pub fn same_f64(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}
