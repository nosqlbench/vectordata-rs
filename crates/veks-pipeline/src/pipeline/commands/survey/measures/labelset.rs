// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Per-label heavy-hitters measure for labelset-shaped text fields.
//!
//! When [`LabelsetProbe`](super::super::probes::LabelsetProbe)
//! commits a field as `Categorical(Labelset)`, the orchestrator
//! attaches this measure. Each observed value is split on `,`,
//! each chunk is normalized (trim outer whitespace, preserve
//! interior whitespace), and the resulting labels are fed into a
//! Misra-Gries sketch over `String` keys. The report holds the
//! top-K labels, their lower-bound counts, and the total observed
//! label count so the predicate generator can compute
//! `sel ≈ count_lower_bound / observed_labels`.
//!
//! ## UTF-8 correctness
//!
//! Splitting is done with `str::split(',')` on the literal byte
//! `,` (ASCII, safe by definition) and per-chunk normalization
//! uses `str::trim` (UTF-8-aware whitespace handling). No raw
//! byte indexing is performed anywhere.

use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use crate::pipeline::commands::survey::measure::{
    Measure, MeasureCtx, MeasureKind, MeasureReport,
};
use crate::pipeline::commands::survey::probes::is_labelset_shape;
use crate::pipeline::commands::survey::sketches::MisraGries;

/// Default top-K for the labelset sketch. Labels are typically
/// shorter than free-text trigrams (think `"sports"`, `"indie
/// rock"`) and the long tail is interesting, so a slightly larger
/// budget than the trigram path is reasonable.
pub const DEFAULT_LABELSET_TOP_K: usize = 2048;

/// Misra-Gries top-K heavy hitters over the *labels* extracted
/// from a labelset-shaped text field. Whole-string distinct
/// frequencies are tracked separately by `HeavyHitters`; this
/// measure dives one layer deeper and asks "of the individual
/// labels in this corpus, which are the most frequent?"
pub struct LabelsetHeavyHittersMeasure {
    inner: MisraGries<String>,
    top_k: usize,
    observed_labels: u64,
    observed_records: u64,
    /// Records that didn't fit the labelset shape (off-pattern
    /// values within an otherwise labelset-shaped field).
    skipped_records: u64,
}

impl LabelsetHeavyHittersMeasure {
    pub fn new(top_k: usize) -> Self {
        let top_k = top_k.max(1);
        LabelsetHeavyHittersMeasure {
            inner: MisraGries::new(top_k),
            top_k,
            observed_labels: 0,
            observed_records: 0,
            skipped_records: 0,
        }
    }
}

impl Measure for LabelsetHeavyHittersMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        let text: &str = match value {
            MValue::Text(s) | MValue::Ascii(s) | MValue::EnumStr(s) => s.as_str(),
            _ => return,
        };
        // Defensive re-check: in steady state the orchestrator only
        // attaches this measure to labelset-shaped fields, but
        // individual records can still drift from the shape. Skip
        // those so they don't contaminate the label frequencies.
        if !is_labelset_shape(text) {
            self.skipped_records += 1;
            return;
        }
        self.observed_records += 1;
        for chunk in text.split(',') {
            let label = chunk.trim();
            if label.is_empty() { continue; }
            self.inner.add(label.to_string());
            self.observed_labels += 1;
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let entries = self.inner.top_k();
        let observations = self.inner.seen();
        let error_bound = self.inner.error_bound();
        MeasureReport::LabelsetHeavyHitters(LabelsetHeavyHittersReport {
            top_k: self.top_k,
            items: entries
                .into_iter()
                .map(|(k, v)| LabelsetEntry { label: k, count_lower_bound: v })
                .collect(),
            observations,
            error_bound,
            observed_labels: self.observed_labels,
            observed_records: self.observed_records,
            skipped_records: self.skipped_records,
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::LabelsetHeavyHitters
    }
}

/// Report shape for [`LabelsetHeavyHittersMeasure::finalize`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelsetHeavyHittersReport {
    /// Configured top-K (capacity).
    pub top_k: usize,
    /// Top labels in descending order of estimated frequency.
    pub items: Vec<LabelsetEntry>,
    /// Total label observations the sketch saw (sum across all
    /// records' chunks).
    pub observations: u64,
    /// Misra-Gries upper-bound undercount: every `count_lower_bound`
    /// is at most `error_bound` below the true count.
    pub error_bound: u64,
    /// Total individual labels extracted across the sample —
    /// drives selectivity math: `sel ≈ count_lower_bound /
    /// observed_labels`.
    pub observed_labels: u64,
    /// Number of records that successfully decomposed into labels.
    pub observed_records: u64,
    /// Records that the measure saw but that did not match the
    /// labelset shape (drift from the Pass 1 verdict). High
    /// values suggest the field is actually mixed-shape.
    pub skipped_records: u64,
}

/// One label entry — the label string and its Misra-Gries lower-
/// bound count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelsetEntry {
    pub label: String,
    pub count_lower_bound: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    #[test]
    fn flattens_labels_across_records() {
        let mut m = LabelsetHeavyHittersMeasure::new(128);
        m.observe(&MValue::Text("music, indie rock, hip hop".into()), &ctx());
        m.observe(&MValue::Text("music, jazz".into()), &ctx());
        m.observe(&MValue::Text("indie rock, ambient".into()), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::LabelsetHeavyHitters(r) => r,
            _ => panic!("wrong report kind"),
        };
        let labels: Vec<&str> = r.items.iter().map(|e| e.label.as_str()).collect();
        assert!(labels.contains(&"music"));
        assert!(labels.contains(&"indie rock"));
        assert!(labels.contains(&"jazz"));
        assert_eq!(r.observed_records, 3);
        assert!(r.observed_labels >= 7);
    }

    /// Off-shape values are skipped so they don't pollute the
    /// per-label frequencies.
    #[test]
    fn off_shape_records_are_skipped() {
        let mut m = LabelsetHeavyHittersMeasure::new(64);
        m.observe(&MValue::Text("not a labelset value".into()), &ctx());
        m.observe(&MValue::Text("a, b".into()), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::LabelsetHeavyHitters(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.observed_records, 1);
        assert_eq!(r.skipped_records, 1);
        assert_eq!(r.observed_labels, 2);
    }

    /// Multi-byte chars round-trip through label keys unchanged.
    #[test]
    fn multibyte_labels_preserved() {
        let mut m = LabelsetHeavyHittersMeasure::new(64);
        m.observe(&MValue::Text("日本語, español".into()), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::LabelsetHeavyHitters(r) => r,
            _ => panic!("wrong report kind"),
        };
        let labels: Vec<&str> = r.items.iter().map(|e| e.label.as_str()).collect();
        assert!(labels.contains(&"日本語"));
        assert!(labels.contains(&"español"));
    }
}
