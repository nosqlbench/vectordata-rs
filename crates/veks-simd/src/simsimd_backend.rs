// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! simsimd's pairwise kernels behind this crate's function-pointer
//! types, for parity runs against an independent implementation.
//!
//! simsimd has no L1 kernel and no query-batch kernels, so it offers
//! L2, cosine and dot product for pairwise scans only. Distances follow
//! this crate's conventions: L2 squared, cosine as `1 − cos θ`, dot
//! negated.

use simsimd::SpatialSimilarity;

use crate::metric::Metric;
use crate::table::{DistFnF16, DistFnF32, DistFnF64};

/// simsimd's f32 kernel for `metric`, or `None` for [`Metric::L1`].
pub fn distance_f32(metric: Metric) -> Option<DistFnF32> {
    match metric {
        Metric::L2 => Some(|a, b| <f32 as SpatialSimilarity>::l2sq(a, b).unwrap_or(0.0) as f32),
        Metric::Cosine => Some(|a, b| <f32 as SpatialSimilarity>::cos(a, b).unwrap_or(1.0) as f32),
        Metric::DotProduct => Some(|a, b| -(<f32 as SpatialSimilarity>::dot(a, b).unwrap_or(0.0) as f32)),
        Metric::L1 => None,
    }
}

/// simsimd's f16 kernel for `metric`, or `None` for [`Metric::L1`].
pub fn distance_f16(metric: Metric) -> Option<DistFnF16> {
    match metric {
        Metric::L2 => Some(|a, b| <simsimd::f16 as SpatialSimilarity>::l2sq(cast(a), cast(b)).unwrap_or(0.0) as f32),
        Metric::Cosine => Some(|a, b| <simsimd::f16 as SpatialSimilarity>::cos(cast(a), cast(b)).unwrap_or(1.0) as f32),
        Metric::DotProduct => Some(|a, b| -(<simsimd::f16 as SpatialSimilarity>::dot(cast(a), cast(b)).unwrap_or(0.0) as f32)),
        Metric::L1 => None,
    }
}

/// simsimd's f64 kernel for `metric`, or `None` for [`Metric::L1`].
pub fn distance_f64(metric: Metric) -> Option<DistFnF64> {
    match metric {
        Metric::L2 => Some(|a, b| <f64 as SpatialSimilarity>::l2sq(a, b).unwrap_or(0.0) as f32),
        Metric::Cosine => Some(|a, b| <f64 as SpatialSimilarity>::cos(a, b).unwrap_or(1.0) as f32),
        Metric::DotProduct => Some(|a, b| -(<f64 as SpatialSimilarity>::dot(a, b).unwrap_or(0.0) as f32)),
        Metric::L1 => None,
    }
}

/// Reinterpret `half::f16` as `simsimd::f16`.
fn cast(s: &[half::f16]) -> &[simsimd::f16] {
    // SAFETY: both types are `#[repr(transparent)]` over `u16` holding
    // IEEE 754 binary16 bits, so the layouts are identical.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<simsimd::f16>(), s.len()) }
}
