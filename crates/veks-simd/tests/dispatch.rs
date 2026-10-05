// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dispatch guards (SRD SK-16, SK-22, acceptance case 6).
//!
//! The failure this design is most exposed to is a dispatch mistake that
//! silently selects a weaker level: every result stays correct, and the
//! only symptom is a slow machine. These tests pin the selected level's
//! identity against an independent reading of the CPU's features rather
//! than timing anything.

use veks_simd::{Kernels, SimdLevel};

/// The level this CPU should get, derived from its feature flags without
/// going through the crate's own detection.
#[cfg(target_arch = "x86_64")]
fn expected_level() -> SimdLevel {
    macro_rules! all {
        ($($f:tt),+) => { true $(&& std::arch::is_x86_feature_detected!($f))+ };
    }
    // `fearless_simd`'s Ice Lake set: AVX-512 is used only where it does
    // not throttle the clock.
    let icelake = all!(
        "adx", "aes", "avx512bitalg", "avx512bw", "avx512cd", "avx512dq", "avx512f",
        "avx512ifma", "avx512vbmi", "avx512vbmi2", "avx512vl", "avx512vnni", "avx512vpopcntdq",
        "bmi1", "bmi2", "cmpxchg16b", "fma", "fxsr", "gfni", "lzcnt", "movbe", "pclmulqdq",
        "popcnt", "rdrand", "rdseed", "sha", "vaes", "vpclmulqdq", "xsave", "xsavec",
        "xsaveopt", "xsaves"
    );
    let v3 = all!(
        "avx2", "bmi1", "bmi2", "cmpxchg16b", "f16c", "fma", "fxsr", "lzcnt", "movbe",
        "popcnt", "xsave"
    );
    let v2 = all!("sse4.2", "cmpxchg16b", "popcnt", "fxsr");
    if icelake {
        SimdLevel::Avx512
    } else if v3 {
        SimdLevel::Avx2
    } else if v2 {
        SimdLevel::Sse4_2
    } else {
        SimdLevel::Sse2
    }
}

#[cfg(target_arch = "aarch64")]
fn expected_level() -> SimdLevel {
    SimdLevel::Neon
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn expected_level() -> SimdLevel {
    SimdLevel::detected()
}

#[test]
fn detection_selects_the_strongest_level_the_cpu_supports() {
    assert_eq!(SimdLevel::detected(), expected_level());
}

#[test]
fn detected_table_is_the_detected_level() {
    assert_eq!(Kernels::detected().level(), SimdLevel::detected());
    assert_eq!(SimdLevel::supported()[0], SimdLevel::detected());
}

#[test]
fn every_supported_level_has_its_own_table() {
    for level in SimdLevel::supported() {
        assert_eq!(Kernels::for_level(level).map(Kernels::level), Some(level));
    }
}

#[test]
fn unsupported_levels_have_no_table() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    assert!(Kernels::for_level(SimdLevel::Neon).is_none());
    #[cfg(target_arch = "aarch64")]
    assert!(Kernels::for_level(SimdLevel::Avx2).is_none());
    assert!(Kernels::for_level(SimdLevel::Scalar).is_none() || SimdLevel::compiled() == [SimdLevel::Scalar]);
    // A level stronger than the detected one is refused even when it is
    // compiled for this target.
    let compiled = SimdLevel::compiled();
    let best = compiled.iter().position(|&l| l == SimdLevel::detected()).unwrap();
    for &stronger in &compiled[..best] {
        assert!(Kernels::for_level(stronger).is_none(), "{stronger:?} is not supported here");
    }
}

#[test]
fn level_names_are_stable() {
    // These strings are written into logs and provenance; renaming one
    // is a format change.
    let names: Vec<&str> = [
        SimdLevel::Avx512, SimdLevel::Avx2, SimdLevel::Sse4_2, SimdLevel::Sse2,
        SimdLevel::Neon, SimdLevel::Wasm128, SimdLevel::Scalar,
    ]
    .iter()
    .map(|l| l.name())
    .collect();
    assert_eq!(names, ["avx512", "avx2", "sse4.2", "sse2", "neon", "wasm128", "scalar"]);
}
