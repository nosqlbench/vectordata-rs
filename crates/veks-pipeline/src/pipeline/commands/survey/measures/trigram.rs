// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Character-trigram heavy-hitters measure.
//!
//! Tracks the top-K most frequent overlapping 3-character windows
//! (trigrams) observed across a text field's sample, using a
//! [`MisraGries`] sketch over `String` keys. Designed to power
//! calibrated `MATCHES` predicate synthesis: for a target
//! selectivity `s`, picking a trigram whose `count_lower_bound /
//! sampled_chars ≈ s` produces a substring pattern whose match
//! rate against the source field approximates `s`.
//!
//! ## UTF-8 correctness
//!
//! Trigrams are windows over `char` boundaries, not bytes. A
//! multi-byte character is one window position, not three. This
//! matches what a downstream `MATCHES` evaluator expects when
//! comparing on character semantics rather than raw bytes.
//!
//! ## Gating
//!
//! The measure is intended for text-shaped fields:
//! `SemanticType::FreeText`, `SemanticType::Categorical(_)`,
//! `SemanticType::Structured(_)`. The orchestrator decides which
//! fields get the measure attached based on Pass 1's verdict.

use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use crate::pipeline::commands::survey::measure::{
    Measure, MeasureCtx, MeasureKind, MeasureReport,
};
use crate::pipeline::commands::survey::sketches::MisraGries;

/// Default top-K for the trigram sketch. Matches the design note in
/// the predicate generator: 1024 keys × ~32 B per `String` ≈ 32 KB
/// per text field. Large enough to recover the heavy hitters on
/// realistic free-text corpora; small enough to stay incidental
/// next to numeric measures.
pub const DEFAULT_TRIGRAM_TOP_K: usize = 1024;

/// Misra-Gries top-K heavy hitters over character 3-grams.
pub struct TrigramHeavyHittersMeasure {
    inner: MisraGries<String>,
    top_k: usize,
    /// Accumulated total of trigram observations (≈ total chars,
    /// minus 2 per value because of the windowing).
    sampled_trigrams: u64,
    /// Reusable buffer for trigram construction; avoids one
    /// allocation per window across the corpus.
    scratch: String,
    /// Reusable character window so we don't materialize a `Vec<char>`
    /// per value.
    window: [char; 3],
    /// Reusable char-iterator buffer to detect under-3-char strings
    /// without walking the whole string twice.
    chars_in_value: Vec<char>,
}

impl TrigramHeavyHittersMeasure {
    pub fn new(top_k: usize) -> Self {
        let top_k = top_k.max(1);
        TrigramHeavyHittersMeasure {
            inner: MisraGries::new(top_k),
            top_k,
            sampled_trigrams: 0,
            scratch: String::with_capacity(12), // 3 chars * up to 4 bytes
            window: ['\0'; 3],
            chars_in_value: Vec::with_capacity(64),
        }
    }
}

impl Measure for TrigramHeavyHittersMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        let text: &str = match value {
            MValue::Text(s) | MValue::Ascii(s) | MValue::EnumStr(s) => s.as_str(),
            _ => return,
        };

        // UTF-8 char-window iteration. `text.chars()` yields code
        // points; we slide a 3-position buffer across them without
        // ever indexing by byte. A trigram on `"日本語"` is the
        // whole string (3 code points, 9 bytes); a trigram on
        // `"abc"` is also one window.
        self.chars_in_value.clear();
        self.chars_in_value.extend(text.chars());
        if self.chars_in_value.len() < 3 { return; }

        // Prime the first window.
        for i in 0..3 { self.window[i] = self.chars_in_value[i]; }
        self.scratch.clear();
        for c in &self.window { self.scratch.push(*c); }
        self.inner.add(self.scratch.clone());
        self.sampled_trigrams += 1;

        // Slide.
        for idx in 3..self.chars_in_value.len() {
            self.window[0] = self.window[1];
            self.window[1] = self.window[2];
            self.window[2] = self.chars_in_value[idx];
            self.scratch.clear();
            for c in &self.window { self.scratch.push(*c); }
            self.inner.add(self.scratch.clone());
            self.sampled_trigrams += 1;
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let entries = self.inner.top_k();
        let observations = self.inner.seen();
        let error_bound = self.inner.error_bound();
        MeasureReport::TrigramHeavyHitters(TrigramHeavyHittersReport {
            top_k: self.top_k,
            items: entries
                .into_iter()
                .map(|(k, v)| TrigramEntry { trigram: k, count_lower_bound: v })
                .collect(),
            observations,
            error_bound,
            sampled_trigrams: self.sampled_trigrams,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::TrigramHeavyHitters
    }
}

/// Report shape for [`TrigramHeavyHittersMeasure::finalize`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrigramHeavyHittersReport {
    /// Configured top-K (capacity).
    pub top_k: usize,
    /// Top trigrams in descending order of estimated frequency.
    pub items: Vec<TrigramEntry>,
    /// Total trigram windows observed across the sample.
    pub observations: u64,
    /// Misra-Gries upper-bound undercount: every `count_lower_bound`
    /// is at most `error_bound` below the true count.
    pub error_bound: u64,
    /// Total trigram observations, mirroring `observations` (kept
    /// as a named field so downstream selectivity math is
    /// self-documenting: `sel ≈ count_lower_bound / sampled_trigrams`).
    pub sampled_trigrams: u64,
}

/// Single trigram entry — the 3-character substring and its
/// Misra-Gries lower-bound count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrigramEntry {
    pub trigram: String,
    pub count_lower_bound: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    #[test]
    fn trigrams_of_short_string_are_skipped() {
        let mut m = TrigramHeavyHittersMeasure::new(64);
        m.observe(&MValue::Text("ab".into()), &ctx()); // < 3 chars
        m.observe(&MValue::Text("".into()), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::TrigramHeavyHitters(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.observations, 0);
        assert!(r.items.is_empty());
    }

    #[test]
    fn ascii_trigrams_count_overlapping_windows() {
        let mut m = TrigramHeavyHittersMeasure::new(64);
        // "abcd" → ["abc", "bcd"] = 2 windows.
        m.observe(&MValue::Text("abcd".into()), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::TrigramHeavyHitters(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.observations, 2);
        let names: Vec<&str> = r.items.iter().map(|e| e.trigram.as_str()).collect();
        assert!(names.contains(&"abc"));
        assert!(names.contains(&"bcd"));
    }

    /// Multi-byte chars: each one counts as a single window position,
    /// not as N bytes. `"日本語"` is exactly one trigram window
    /// covering all three characters (9 bytes).
    #[test]
    fn multibyte_chars_are_single_window_positions() {
        let mut m = TrigramHeavyHittersMeasure::new(64);
        m.observe(&MValue::Text("日本語".into()), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::TrigramHeavyHitters(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.observations, 1);
        assert_eq!(r.items[0].trigram, "日本語");
    }

    /// A repeating substring should rise to the top. With `"the "`
    /// repeating, `"the"` and the shifted variants `"he "`, `" th"`
    /// each occur once per cycle (tied in frequency); the assertion
    /// is that `"the"` is *among* the top hitters with a high count,
    /// not that it's strictly #1 (the Misra-Gries top-K order
    /// between tied entries is unspecified).
    #[test]
    fn dominant_trigram_recovered() {
        let mut m = TrigramHeavyHittersMeasure::new(128);
        let text = "the ".repeat(20);
        m.observe(&MValue::Text(text), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::TrigramHeavyHitters(r) => r,
            _ => panic!("wrong report kind"),
        };
        let the_entry = r
            .items
            .iter()
            .find(|e| e.trigram == "the")
            .expect("`the` trigram should be in the top hitters");
        assert!(
            the_entry.count_lower_bound >= 15,
            "expected `the` count >= 15, got {}",
            the_entry.count_lower_bound,
        );
    }

    /// Non-text MValue variants are ignored.
    #[test]
    fn non_text_values_are_ignored() {
        let mut m = TrigramHeavyHittersMeasure::new(64);
        m.observe(&MValue::Int(42), &ctx());
        m.observe(&MValue::Bool(true), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::TrigramHeavyHitters(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.observations, 0);
    }
}
