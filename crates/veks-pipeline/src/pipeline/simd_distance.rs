// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The pipeline's view of the vector kernels.
//!
//! Every distance, query-batch and f16 conversion kernel the commands use
//! comes from [`veks_simd`], which dispatches to the best instruction set
//! once per process. This module re-exports its types under the names the
//! commands use, hands out function pointers through `select_*`
//! functions (resolve once, outside the hot loop), and owns the
//! `backend` option that lets a pairwise scan run on simsimd's kernels
//! instead, when the binary is built with the `simsimd` feature.

use veks_simd::Kernels;

use crate::pipeline::command::{OptionDesc, OptionRole, Options};

pub use veks_simd::{BATCH_WIDTH as SIMD_BATCH_WIDTH, Metric, PackedBatches, TransposedBatch};

/// One base vector against one [`TransposedBatch`] (16 distances).
pub type BatchedDistFnF32 = veks_simd::BatchFnF32;

/// One base vector against two [`TransposedBatch`]es (32 distances).
pub type DualBatchedDistFnF32 = veks_simd::DualBatchFnF32;

/// One base vector against every sub-batch of a [`PackedBatches`].
pub type PackedNegDotFn = veks_simd::PackedNegDotFn;

/// Bulk f16 → f32.
pub type F16ToF32Fn = veks_simd::ConvertF16ToF32;

fn kernels() -> &'static Kernels {
    Kernels::detected()
}

/// The native f32 kernel for `metric`.
pub fn select_distance_fn(metric: Metric) -> fn(&[f32], &[f32]) -> f32 {
    kernels().distance_f32(metric)
}

/// The native f16 kernel for `metric` (accumulates in f32).
pub fn select_distance_fn_f16(metric: Metric) -> fn(&[half::f16], &[half::f16]) -> f32 {
    kernels().distance_f16(metric)
}

/// The native f64 kernel for `metric` (accumulates in f64).
pub fn select_distance_fn_f64(metric: Metric) -> fn(&[f64], &[f64]) -> f32 {
    kernels().distance_f64(metric)
}

/// The plain (not negated) f32 inner product.
pub fn select_dot_fn() -> fn(&[f32], &[f32]) -> f32 {
    kernels().dot_f32()
}

/// The f32 L2 norm.
pub fn select_norm_fn() -> fn(&[f32]) -> f32 {
    kernels().norm_f32()
}

/// The single-batch kernel for `metric`. Every metric has one, at every
/// dispatch level.
pub fn select_batched_fn_f32(metric: Metric) -> BatchedDistFnF32 {
    kernels().batch_f32(metric)
}

/// The dual-batch kernel for `metric`.
pub fn select_dual_batched_fn_f32(metric: Metric) -> DualBatchedDistFnF32 {
    kernels().dual_batch_f32(metric)
}

/// The packed negated-dot kernel.
pub fn select_packed_neg_dot_f32() -> PackedNegDotFn {
    kernels().packed_neg_dot_f32()
}

/// The bulk f16 → f32 converter.
pub fn select_f16_to_f32() -> F16ToF32Fn {
    kernels().f16_to_f32()
}

/// Whether `metric` should use the packed neg-dot kernel path.
///
/// Only DotProduct is eligible: the packed kernel computes raw `-dot(a, b)`
/// which is only a valid distance for unit-length vectors where cosine
/// reduces to negative dot product. All other metrics (Cosine, L2, L1) must
/// use the TransposedBatch path which applies the correct per-metric formula
/// (norm division, squared differences, etc.).
///
/// Every command that chooses between packed and TransposedBatch paths must
/// call this function instead of making the decision locally — duplicating
/// this logic is how compute-knn and verify-knn diverged in the past.
pub fn metric_uses_packed_path(metric: Metric) -> bool {
    metric == Metric::DotProduct
}

/// The instruction-set level the native kernels dispatched to, as written
/// in logs and provenance (`avx512`, `avx2`, `neon`, …).
pub fn simd_level() -> &'static str {
    kernels().level().name()
}

/// Which implementation serves a command's pairwise distance kernels.
///
/// [`Native`](PairwiseBackend::Native) is [`veks_simd`] and always
/// available. [`Simsimd`](PairwiseBackend::Simsimd) is simsimd's C kernels,
/// compiled in only with the `simsimd` feature and kept as an independent
/// implementation for parity runs. simsimd has no batch kernels and no
/// L1, so a simsimd run scans pairwise and refuses L1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairwiseBackend {
    /// The pure-Rust kernels in `veks-simd`.
    Native,
    /// simsimd (C), with the `simsimd` feature.
    Simsimd,
}

impl PairwiseBackend {
    /// The option name, shared by every command that takes it.
    pub const OPTION: &'static str = "backend";

    /// The backend named by the `backend` option, [`Native`] when unset.
    /// Errors on an unknown name, or on `simsimd` in a build without it.
    ///
    /// [`Native`]: PairwiseBackend::Native
    pub fn from_options(options: &Options) -> Result<Self, String> {
        Self::parse(options.get(Self::OPTION).unwrap_or("native"))
    }

    /// Parse a backend name.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "native" => Ok(PairwiseBackend::Native),
            "simsimd" if cfg!(feature = "simsimd") => Ok(PairwiseBackend::Simsimd),
            "simsimd" => Err("backend 'simsimd' is not available: this binary was built without the `simsimd` feature".into()),
            other => Err(format!("unknown backend '{other}': expected native or simsimd")),
        }
    }

    /// The backend's name as written in logs and engine identifiers.
    pub fn name(self) -> &'static str {
        match self {
            PairwiseBackend::Native => "native",
            PairwiseBackend::Simsimd => "simsimd",
        }
    }

    /// Whether this backend has query-batch kernels.
    pub fn has_batch_kernels(self) -> bool {
        self == PairwiseBackend::Native
    }

    /// The f32 kernel for `metric` on this backend.
    pub fn distance_f32(self, metric: Metric) -> Result<fn(&[f32], &[f32]) -> f32, String> {
        match self {
            PairwiseBackend::Native => Ok(select_distance_fn(metric)),
            PairwiseBackend::Simsimd => simsimd_kernel(metric, |m| simsimd_f32(m)),
        }
    }

    /// The f16 kernel for `metric` on this backend.
    pub fn distance_f16(self, metric: Metric) -> Result<fn(&[half::f16], &[half::f16]) -> f32, String> {
        match self {
            PairwiseBackend::Native => Ok(select_distance_fn_f16(metric)),
            PairwiseBackend::Simsimd => simsimd_kernel(metric, |m| simsimd_f16(m)),
        }
    }

    /// The f64 kernel for `metric` on this backend.
    pub fn distance_f64(self, metric: Metric) -> Result<fn(&[f64], &[f64]) -> f32, String> {
        match self {
            PairwiseBackend::Native => Ok(select_distance_fn_f64(metric)),
            PairwiseBackend::Simsimd => simsimd_kernel(metric, |m| simsimd_f64(m)),
        }
    }

    /// The `backend` option, as each command that takes it declares it.
    pub fn option_desc() -> OptionDesc {
        OptionDesc {
            name: Self::OPTION.to_string(),
            type_name: "enum".to_string(),
            required: false,
            default: Some("native".to_string()),
            description: "Pairwise distance kernels: native (pure Rust, default) or simsimd (needs the `simsimd` build feature)".to_string(),
            extended_description: Some(
                "native: veks-simd's kernels, dispatched to the best instruction set the CPU supports, \
                 with 16-query batch kernels.\n\
                 simsimd: simsimd's C kernels, as an independent implementation for parity runs. \
                 It has no batch kernels, so the scan is pairwise, and no L1 kernel. Its results are \
                 cached separately from the native backend's."
                    .to_string(),
            ),
            role: OptionRole::Config,
        }
    }
}

fn simsimd_kernel<F>(metric: Metric, pick: impl Fn(Metric) -> Option<F>) -> Result<F, String> {
    pick(metric).ok_or_else(|| format!("backend 'simsimd' has no {metric:?} kernel"))
}

#[cfg(feature = "simsimd")]
fn simsimd_f32(m: Metric) -> Option<fn(&[f32], &[f32]) -> f32> {
    veks_simd::simsimd_backend::distance_f32(m)
}
#[cfg(feature = "simsimd")]
fn simsimd_f16(m: Metric) -> Option<fn(&[half::f16], &[half::f16]) -> f32> {
    veks_simd::simsimd_backend::distance_f16(m)
}
#[cfg(feature = "simsimd")]
fn simsimd_f64(m: Metric) -> Option<fn(&[f64], &[f64]) -> f32> {
    veks_simd::simsimd_backend::distance_f64(m)
}

// Without the feature, `parse` never yields `Simsimd`, so these are
// unreachable; they exist so the match above needs no cfg.
#[cfg(not(feature = "simsimd"))]
fn simsimd_f32(_: Metric) -> Option<fn(&[f32], &[f32]) -> f32> {
    None
}
#[cfg(not(feature = "simsimd"))]
fn simsimd_f16(_: Metric) -> Option<fn(&[half::f16], &[half::f16]) -> f32> {
    None
}
#[cfg(not(feature = "simsimd"))]
fn simsimd_f64(_: Metric) -> Option<fn(&[f64], &[f64]) -> f32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_names_parse() {
        assert_eq!(Metric::parse("l2"), Some(Metric::L2));
        assert_eq!(Metric::parse("EUCLIDEAN"), Some(Metric::L2));
        assert_eq!(Metric::parse("cosine"), Some(Metric::Cosine));
        assert_eq!(Metric::parse("dot"), Some(Metric::DotProduct));
        assert_eq!(Metric::parse("DOT_PRODUCT"), Some(Metric::DotProduct));
        assert_eq!(Metric::parse("manhattan"), Some(Metric::L1));
        assert_eq!(Metric::parse("hamming"), None);
    }

    #[test]
    fn native_backend_is_the_default() {
        let options = Options::new();
        assert_eq!(PairwiseBackend::from_options(&options), Ok(PairwiseBackend::Native));
    }

    #[test]
    fn unknown_backend_is_an_error() {
        let err = PairwiseBackend::parse("faiss").unwrap_err();
        assert!(err.contains("unknown backend 'faiss'"), "{err}");
    }

    #[test]
    fn simsimd_backend_follows_the_feature() {
        let parsed = PairwiseBackend::parse("simsimd");
        if cfg!(feature = "simsimd") {
            assert_eq!(parsed, Ok(PairwiseBackend::Simsimd));
            assert!(PairwiseBackend::Simsimd.distance_f32(Metric::L1).is_err(), "simsimd has no L1");
            let l2 = PairwiseBackend::Simsimd.distance_f32(Metric::L2).unwrap();
            assert_eq!(l2(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
        } else {
            let err = parsed.unwrap_err();
            assert!(err.contains("without the `simsimd` feature"), "{err}");
        }
    }

    #[test]
    fn native_backend_serves_every_metric() {
        for metric in Metric::ALL {
            let f = PairwiseBackend::Native.distance_f32(metric).unwrap();
            assert!(f(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).is_finite());
        }
    }

    #[test]
    fn reported_level_is_the_dispatched_level() {
        assert_eq!(simd_level(), veks_simd::SimdLevel::detected().name());
    }
}
