// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SIMD dispatch levels and their one-time detection.

use std::sync::OnceLock;

use fearless_simd::Level;

/// An instruction-set level a kernel table is compiled for.
///
/// Only levels that exist on the compilation target are constructible:
/// x86 builds have [`Avx512`](SimdLevel::Avx512), [`Avx2`](SimdLevel::Avx2),
/// [`Sse4_2`](SimdLevel::Sse4_2) and [`Sse2`](SimdLevel::Sse2); aarch64
/// builds have [`Neon`](SimdLevel::Neon), which the architecture
/// guarantees; wasm32 builds with `simd128` have
/// [`Wasm128`](SimdLevel::Wasm128); every other target has
/// [`Scalar`](SimdLevel::Scalar).
///
/// `Avx512` means the Ice Lake feature set, not bare `avx512f`: on
/// earlier AVX-512 parts (Skylake-SP, Cascade Lake), whose 512-bit units
/// throttle the clock, detection selects `Avx2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimdLevel {
    /// Ice Lake-class AVX-512 (x86).
    Avx512,
    /// x86-64-v3: AVX2, FMA and F16C.
    Avx2,
    /// x86-64-v2: SSE4.2.
    Sse4_2,
    /// The x86-64 baseline: SSE2.
    Sse2,
    /// NEON (aarch64).
    Neon,
    /// WebAssembly 128-bit SIMD.
    Wasm128,
    /// No SIMD instruction set.
    Scalar,
}

impl SimdLevel {
    /// A short, stable name for logs and provenance: `avx512`, `avx2`,
    /// `sse4.2`, `sse2`, `neon`, `wasm128` or `scalar`.
    pub const fn name(self) -> &'static str {
        match self {
            SimdLevel::Avx512 => "avx512",
            SimdLevel::Avx2 => "avx2",
            SimdLevel::Sse4_2 => "sse4.2",
            SimdLevel::Sse2 => "sse2",
            SimdLevel::Neon => "neon",
            SimdLevel::Wasm128 => "wasm128",
            SimdLevel::Scalar => "scalar",
        }
    }

    /// Every level a kernel table exists for on this compilation target,
    /// strongest first.
    pub const fn compiled() -> &'static [SimdLevel] {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            &[SimdLevel::Avx512, SimdLevel::Avx2, SimdLevel::Sse4_2, SimdLevel::Sse2]
        }
        #[cfg(target_arch = "aarch64")]
        {
            &[SimdLevel::Neon]
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            &[SimdLevel::Wasm128]
        }
        #[cfg(not(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
        {
            &[SimdLevel::Scalar]
        }
    }

    /// Every compiled level this CPU can run, strongest first. The first
    /// entry is [`SimdLevel::detected`].
    pub fn supported() -> Vec<SimdLevel> {
        let best = SimdLevel::detected();
        let compiled = SimdLevel::compiled();
        let from = compiled.iter().position(|&l| l == best).unwrap_or(0);
        compiled[from..].to_vec()
    }

    /// The strongest level this CPU supports. Detected on the first call
    /// and cached for the life of the process.
    pub fn detected() -> SimdLevel {
        from_level(detected_level())
    }
}

/// The `fearless_simd` level for this CPU, detected once.
pub(crate) fn detected_level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(Level::new)
}

/// Map a `fearless_simd` level onto ours.
fn from_level(level: Level) -> SimdLevel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if level.as_avx512().is_some() {
            return SimdLevel::Avx512;
        }
        if level.as_avx2().is_some() {
            return SimdLevel::Avx2;
        }
        if level.as_sse4_2().is_some() {
            return SimdLevel::Sse4_2;
        }
        SimdLevel::Sse2
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = level;
        SimdLevel::Neon
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        let _ = level;
        SimdLevel::Wasm128
    }
    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        let _ = level;
        SimdLevel::Scalar
    }
}
