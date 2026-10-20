// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Per-type sample values for the tests that iterate `ElementType::ALL`
//! (SRD element-access §13.10): zero, one, the type's extremes, and
//! for a float a negative, a subnormal, both infinities and a NaN.

use vectordata::typed_access::{Element, bf16, f16};

/// The values a test feeds through a reader, writer or command at `Self`.
pub trait Samples: Element {
    /// The sample values, in a fixed order.
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

/// `values` in the little-endian encoding every file uses.
pub fn encode<T: Element>(values: &[T]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * T::SIZE];
    for (v, chunk) in values.iter().zip(out.chunks_mut(T::SIZE)) {
        v.write_le_bytes(chunk);
    }
    out
}

/// Whether two runs hold the same bits: equality for every finite
/// value, and identity for a NaN, which `==` never grants.
pub fn same_bits<T: Element>(a: &[T], b: &[T]) -> bool {
    encode(a) == encode(b)
}

/// `records` records, each the samples rotated by its ordinal, so no
/// two records are alike and every sample appears at every position.
pub fn rotated_records<T: Samples>(records: usize) -> Vec<Vec<T>> {
    let s = T::samples();
    (0..records).map(|r| (0..s.len()).map(|i| s[(i + r) % s.len()]).collect()).collect()
}
