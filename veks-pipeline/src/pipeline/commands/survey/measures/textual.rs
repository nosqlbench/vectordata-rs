// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Textual measures for `Text` / `Ascii` / `EnumStr` / `Date` /
//! `Time` / `DateTime` MValue variants. Also run against any field
//! whose `SemanticType` is `Identifier(_)` or `Structured(_)` so
//! the report carries length and char-class shape data alongside
//! the semantic verdict.

use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use crate::pipeline::commands::survey::measure::{
    Measure, MeasureCtx, MeasureKind, MeasureReport,
};
use crate::pipeline::commands::survey::sketches::KllSketch;

// ---------------------------------------------------------------------------
// ExactLengthMoments
// ---------------------------------------------------------------------------

/// Moments over text length (in bytes and chars). Both are reported
/// because a UTF-8 byte length can exceed the visible character
/// count for non-ASCII content.
pub struct ExactLengthMoments {
    bytes: LenAccumulator,
    chars: LenAccumulator,
}

impl Default for ExactLengthMoments {
    fn default() -> Self { Self::new() }
}

impl ExactLengthMoments {
    pub fn new() -> Self {
        ExactLengthMoments {
            bytes: LenAccumulator::new(),
            chars: LenAccumulator::new(),
        }
    }
}

impl Measure for ExactLengthMoments {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        if let Some(s) = string_of(value) {
            self.bytes.update(s.len());
            self.chars.update(s.chars().count());
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        MeasureReport::ExactLengthMoments(ExactLengthMomentsReport {
            bytes: self.bytes.finish(),
            chars: self.chars.finish(),
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::ExactLengthMoments
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactLengthMomentsReport {
    pub bytes: LenSummary,
    pub chars: LenSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LenSummary {
    pub count: u64,
    pub min: u64,
    pub max: u64,
    pub mean: Option<f64>,
    pub stddev: Option<f64>,
}

struct LenAccumulator {
    n: u64,
    min: u64,
    max: u64,
    /// Welford running moments over the integer length values
    /// promoted to f64.
    m1: f64,
    m2: f64,
}

impl LenAccumulator {
    fn new() -> Self {
        LenAccumulator { n: 0, min: u64::MAX, max: 0, m1: 0.0, m2: 0.0 }
    }
    fn update(&mut self, len: usize) {
        let n0 = self.n as f64;
        self.n += 1;
        let n = self.n as f64;
        let x = len as f64;
        let delta = x - self.m1;
        self.m1 += delta / n;
        self.m2 += delta * (x - self.m1);
        let l = len as u64;
        if l < self.min { self.min = l; }
        if l > self.max { self.max = l; }
        // `n0` retained to silence unused-warnings in tight builds.
        let _ = n0;
    }
    fn finish(self) -> LenSummary {
        let (mean, stddev) = if self.n == 0 {
            (None, None)
        } else if self.n < 2 {
            (Some(self.m1), None)
        } else {
            (Some(self.m1), Some((self.m2 / (self.n as f64 - 1.0)).sqrt()))
        };
        LenSummary {
            count: self.n,
            min: if self.n == 0 { 0 } else { self.min },
            max: self.max,
            mean,
            stddev,
        }
    }
}

// ---------------------------------------------------------------------------
// LengthQuantiles
// ---------------------------------------------------------------------------

/// Quantile sketch over text length (bytes). Driven by the same KLL
/// sketch used elsewhere.
pub struct LengthQuantiles {
    sketch: KllSketch,
}

impl LengthQuantiles {
    pub fn new(kll_k: usize, seed: u64) -> Self {
        LengthQuantiles {
            sketch: KllSketch::with_seed(kll_k, seed),
        }
    }
}

impl Measure for LengthQuantiles {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        if let Some(s) = string_of(value) {
            self.sketch.add(s.len() as f64);
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];
        let values = self.sketch.quantiles(&qs).unwrap_or_default();
        let pairs: Vec<(String, f64)> = qs
            .iter()
            .zip(values.iter())
            .map(|(q, v)| (format!("p{}", (q * 100.0).round() as u32), *v))
            .collect();
        MeasureReport::LengthQuantiles(LengthQuantilesReport {
            count: self.sketch.count(),
            min: self.sketch.min(),
            max: self.sketch.max(),
            quantiles: pairs,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::LengthQuantiles
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LengthQuantilesReport {
    pub count: u64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// Quantile/value pairs, sorted by quantile ascending.
    pub quantiles: Vec<(String, f64)>,
}

// ---------------------------------------------------------------------------
// CharClassMix
// ---------------------------------------------------------------------------

/// Per-character-class fractions averaged across observations.
///
/// Each string contributes one observation; within a string we
/// count `(alpha, digit, punct, whitespace, other)` chars,
/// normalize to the string's length, and accumulate the per-class
/// running mean.
pub struct CharClassMix {
    n: u64,
    sum_alpha: f64,
    sum_digit: f64,
    sum_punct: f64,
    sum_whitespace: f64,
    sum_other: f64,
}

impl Default for CharClassMix {
    fn default() -> Self { Self::new() }
}

impl CharClassMix {
    pub fn new() -> Self {
        CharClassMix {
            n: 0,
            sum_alpha: 0.0,
            sum_digit: 0.0,
            sum_punct: 0.0,
            sum_whitespace: 0.0,
            sum_other: 0.0,
        }
    }
}

impl Measure for CharClassMix {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        let s = match string_of(value) {
            Some(s) => s,
            None => return,
        };
        if s.is_empty() {
            return;
        }
        let mut a = 0u32;
        let mut d = 0u32;
        let mut p = 0u32;
        let mut w = 0u32;
        let mut o = 0u32;
        for c in s.chars() {
            if c.is_alphabetic() { a += 1; }
            else if c.is_ascii_digit() { d += 1; }
            else if c.is_whitespace() { w += 1; }
            else if c.is_ascii_punctuation() { p += 1; }
            else { o += 1; }
        }
        let total = (a + d + p + w + o).max(1) as f64;
        self.n += 1;
        self.sum_alpha += a as f64 / total;
        self.sum_digit += d as f64 / total;
        self.sum_punct += p as f64 / total;
        self.sum_whitespace += w as f64 / total;
        self.sum_other += o as f64 / total;
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let n = self.n.max(1) as f64;
        MeasureReport::CharClassMix(CharClassMixReport {
            count: self.n,
            alpha: self.sum_alpha / n,
            digit: self.sum_digit / n,
            punct: self.sum_punct / n,
            whitespace: self.sum_whitespace / n,
            other: self.sum_other / n,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::CharClassMix
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CharClassMixReport {
    pub count: u64,
    pub alpha: f64,
    pub digit: f64,
    pub punct: f64,
    pub whitespace: f64,
    pub other: f64,
}

// ---------------------------------------------------------------------------
// PatternSkeletonMeasure
// ---------------------------------------------------------------------------

/// Collapses each observed string into a regex-like skeleton —
/// `A` for letters, `9` for digits, `.` for punctuation, `_` for
/// whitespace, `?` for other — and reports the top-K skeleton
/// frequencies. Useful for detecting fields where every value
/// follows the same shape (`AAA-9999` license plates, `9+@A+.A+`
/// emails, etc.) versus genuine free text.
///
/// Runs `Misra-Gries` heavy-hitters under the hood to keep memory
/// bounded.
pub struct PatternSkeletonMeasure {
    inner: crate::pipeline::commands::survey::sketches::MisraGries<String>,
    top_k: usize,
}

impl PatternSkeletonMeasure {
    pub fn new(top_k: usize) -> Self {
        PatternSkeletonMeasure {
            inner: crate::pipeline::commands::survey::sketches::MisraGries::new(top_k),
            top_k,
        }
    }

    /// Compact a string into its character-class skeleton. Runs of
    /// the same class collapse to a single representative + `+`
    /// (e.g. `"foo123"` → `"A+9+"`, `"a@b.com"` → `"A@A.A+"`).
    pub fn skeleton_for(s: &str) -> String {
        if s.is_empty() { return String::new(); }
        let mut out = String::with_capacity(s.len());
        let mut last_class: Option<char> = None;
        let mut last_kept: Option<char> = None;
        for c in s.chars() {
            let cls = classify_char(c);
            // Preserve specific structural punctuation verbatim so
            // patterns like `99-99-9999` and `A+@A+.A+` remain
            // distinguishable from generic `.` punctuation.
            if matches!(c, '@' | '.' | '-' | '_' | ':' | '/' | ',' | '+') {
                out.push(c);
                last_class = None;
                last_kept = Some(c);
                continue;
            }
            if Some(cls) == last_class {
                // Already in a run — only append `+` on the first
                // repeat to keep the skeleton stable.
                if last_kept != Some('+') {
                    out.push('+');
                    last_kept = Some('+');
                }
            } else {
                out.push(cls);
                last_class = Some(cls);
                last_kept = Some(cls);
            }
        }
        out
    }
}

fn classify_char(c: char) -> char {
    if c.is_ascii_alphabetic() { 'A' }
    else if c.is_ascii_digit() { '9' }
    else if c.is_whitespace() { '_' }
    else { '?' }
}

impl Measure for PatternSkeletonMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        if let Some(s) = string_of(value) {
            self.inner.add(Self::skeleton_for(s));
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let observed = self.inner.seen();
        let entries = self.inner.top_k();
        let pairs: Vec<(String, f64)> = entries
            .into_iter()
            .map(|(k, v)| {
                let rate = if observed == 0 { 0.0 } else { v as f64 / observed as f64 };
                (k, rate)
            })
            .collect();
        MeasureReport::PatternSkeleton(PatternSkeletonReport {
            observed,
            top_k: self.top_k as u32,
            patterns: pairs,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::PatternSkeleton
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternSkeletonReport {
    pub observed: u64,
    pub top_k: u32,
    /// `(skeleton, fraction)` pairs in descending fraction order.
    pub patterns: Vec<(String, f64)>,
}

// ---------------------------------------------------------------------------
// Shared helper
// ---------------------------------------------------------------------------

/// Extract the string content from any string-shaped `MValue` variant.
fn string_of(value: &MValue) -> Option<&str> {
    match value {
        MValue::Text(s) | MValue::Ascii(s) | MValue::EnumStr(s)
        | MValue::Date(s) | MValue::Time(s) | MValue::DateTime(s) => Some(s.as_str()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    #[test]
    fn length_moments_basic() {
        let mut m = ExactLengthMoments::new();
        for s in ["a", "bb", "ccc", "dddd", "eeeee"] {
            m.observe(&MValue::Text(s.into()), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::ExactLengthMoments(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.bytes.count, 5);
        assert_eq!(r.bytes.min, 1);
        assert_eq!(r.bytes.max, 5);
        assert!((r.bytes.mean.unwrap() - 3.0).abs() < 1e-9);
        assert!((r.bytes.stddev.unwrap() - (2.5_f64).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn length_moments_byte_vs_char_for_multibyte() {
        let mut m = ExactLengthMoments::new();
        // "ä" is 2 bytes / 1 char in UTF-8.
        m.observe(&MValue::Text("ä".into()), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::ExactLengthMoments(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.bytes.min, 2);
        assert_eq!(r.chars.min, 1);
    }

    #[test]
    fn length_quantiles_recover_percentiles() {
        let mut m = LengthQuantiles::new(200, 7);
        for i in 1..=1_000 {
            let s = "x".repeat(i);
            m.observe(&MValue::Text(s), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::LengthQuantiles(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.count, 1_000);
        // Look up p50 from the pairs.
        let p50 = r.quantiles.iter().find(|(q, _)| q == "p50").unwrap().1;
        // True p50 of 1..=1000 is 500.
        assert!((p50 - 500.0).abs() < 50.0, "p50 = {}", p50);
    }

    #[test]
    fn char_class_mix_email_like() {
        let mut m = CharClassMix::new();
        for addr in &[
            "alice@example.com",
            "bob.smith@company.io",
            "carol99@x.org",
        ] {
            m.observe(&MValue::Text((*addr).into()), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::CharClassMix(r) => r,
            _ => panic!("wrong report kind"),
        };
        // Email is mostly alpha; punctuation comes from "@" and ".".
        assert!(r.alpha > 0.7);
        assert!(r.punct > 0.05);
        assert!(r.digit < 0.2);
        // Fractions sum to ~1.0.
        let sum = r.alpha + r.digit + r.punct + r.whitespace + r.other;
        assert!((sum - 1.0).abs() < 1e-9, "sum = {}", sum);
    }

    #[test]
    fn char_class_mix_pure_digit_strings() {
        let mut m = CharClassMix::new();
        for s in &["12345", "67890", "00001"] {
            m.observe(&MValue::Text((*s).into()), &ctx());
        }
        let r = match Box::new(m).finalize() {
            MeasureReport::CharClassMix(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.digit, 1.0);
        assert_eq!(r.alpha, 0.0);
    }

    #[test]
    fn ignores_non_string_values() {
        let mut m = CharClassMix::new();
        m.observe(&MValue::Int(42), &ctx());
        m.observe(&MValue::Bool(true), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::CharClassMix(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.count, 0);
    }
}
