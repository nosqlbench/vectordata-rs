// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Per-type sample values and uniform-vector fixtures for the tests
//! that iterate `ElementType::ALL` (SRD element-access §13.10).

#![allow(dead_code)]

use std::path::Path;

use vectordata::typed_access::{Container, Element, bf16, f16};

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

/// The finite samples of `T`, for a pipeline step that computes over
/// them: an infinity or a NaN would be removed or refused as a zero or
/// a non-vector rather than carried through.
pub fn finite_samples<T: Samples>() -> Vec<T> {
    T::samples().into_iter().filter(|v| v.as_f64().is_finite()).collect()
}

/// Write `records` uniform vectors of `dim` elements to `path`, record
/// `r` element `c` holding `value(r, c)`.
pub fn write_vectors<T: Element>(path: &Path, records: usize, dim: usize, value: impl Fn(usize, usize) -> T) {
    let mut bytes = Vec::with_capacity(records * (4 + dim * T::SIZE));
    for r in 0..records {
        bytes.extend_from_slice(&(dim as i32).to_le_bytes());
        for c in 0..dim {
            let at = bytes.len();
            bytes.resize(at + T::SIZE, 0);
            value(r, c).write_le_bytes(&mut bytes[at..]);
        }
    }
    std::fs::write(path, bytes).unwrap();
}

/// The canonical uniform-vector extension of `T`: `f32vecs`, `bf16vecs`.
pub fn vec_extension<T: Element>() -> &'static str {
    T::TYPE.extension(Container::Vec)
}
