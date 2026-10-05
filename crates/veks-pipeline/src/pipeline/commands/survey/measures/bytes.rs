// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Bytes-only measures.
//!
//! `ByteEntropyMeasure` computes the Shannon entropy of the byte
//! value distribution across observations, useful for separating
//! "high entropy → compressed or random" from "low entropy → text
//! or sparse-structured".

use serde::{Deserialize, Serialize};

use veks_core::formats::mnode::MValue;

use crate::pipeline::commands::survey::measure::{
    Measure, MeasureCtx, MeasureKind, MeasureReport,
};

pub struct ByteEntropyMeasure {
    /// Per-byte-value histogram across every observation's bytes.
    histogram: [u64; 256],
    total_bytes: u64,
    observations: u64,
}

impl Default for ByteEntropyMeasure {
    fn default() -> Self { Self::new() }
}

impl ByteEntropyMeasure {
    pub fn new() -> Self {
        ByteEntropyMeasure {
            histogram: [0u64; 256],
            total_bytes: 0,
            observations: 0,
        }
    }
}

impl Measure for ByteEntropyMeasure {
    fn observe(&mut self, value: &MValue, _ctx: &MeasureCtx) {
        if let MValue::Bytes(b) = value {
            self.observations += 1;
            self.total_bytes += b.len() as u64;
            for &byte in b {
                self.histogram[byte as usize] += 1;
            }
        }
    }

    fn finalize(self: Box<Self>) -> MeasureReport {
        let total = self.total_bytes;
        let (entropy_bits, classification) = if total == 0 {
            (0.0, "Empty")
        } else {
            let total_f = total as f64;
            let mut h = 0f64;
            for &c in &self.histogram {
                if c == 0 { continue }
                let p = c as f64 / total_f;
                h -= p * p.log2();
            }
            // Heuristic banding: max possible entropy is log2(256) = 8.
            let band = match h {
                x if x >= 7.5 => "Compressed",
                x if x >= 5.0 => "MixedHighEntropy",
                x if x >= 2.0 => "Structured",
                _ => "LowEntropy",
            };
            (h, band)
        };
        MeasureReport::ByteEntropy(ByteEntropyReport {
            observations: self.observations,
            total_bytes: total,
            entropy_bits,
            classification: classification.into(),
        })
    }

    fn kind(&self) -> MeasureKind {
        MeasureKind::ByteEntropy
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ByteEntropyReport {
    pub observations: u64,
    pub total_bytes: u64,
    /// Shannon entropy of the byte-value distribution, in bits.
    /// 0 .. 8 (uniform random reaches the upper bound).
    pub entropy_bits: f64,
    /// Heuristic classification: Compressed / MixedHighEntropy /
    /// Structured / LowEntropy / Empty.
    pub classification: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> MeasureCtx<'static> {
        MeasureCtx { record_index: 0, semantic_type: None }
    }

    #[test]
    fn empty_reports_zero_entropy() {
        let r = match Box::new(ByteEntropyMeasure::new()).finalize() {
            MeasureReport::ByteEntropy(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.entropy_bits, 0.0);
        assert_eq!(r.classification, "Empty");
    }

    #[test]
    fn constant_bytes_zero_entropy() {
        let mut m = ByteEntropyMeasure::new();
        m.observe(&MValue::Bytes(vec![0xff; 1024]), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::ByteEntropy(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.entropy_bits, 0.0);
        assert_eq!(r.classification, "LowEntropy");
    }

    #[test]
    fn uniform_random_high_entropy() {
        let mut m = ByteEntropyMeasure::new();
        // Construct a uniform byte stream by emitting every byte
        // value the same number of times.
        let payload: Vec<u8> = (0..256u16).flat_map(|b| std::iter::repeat_n(b as u8, 16)).collect();
        m.observe(&MValue::Bytes(payload), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::ByteEntropy(r) => r,
            _ => panic!("wrong report kind"),
        };
        // Exactly uniform → 8 bits.
        assert!((r.entropy_bits - 8.0).abs() < 1e-9);
        assert_eq!(r.classification, "Compressed");
    }

    #[test]
    fn ascii_text_structured_band() {
        let mut m = ByteEntropyMeasure::new();
        // ASCII English text uses roughly 4-5 bits per byte.
        m.observe(
            &MValue::Bytes(
                "the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog".as_bytes().to_vec()
            ),
            &ctx(),
        );
        let r = match Box::new(m).finalize() {
            MeasureReport::ByteEntropy(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert!(r.entropy_bits >= 2.0 && r.entropy_bits < 7.5);
    }

    #[test]
    fn ignores_non_bytes() {
        let mut m = ByteEntropyMeasure::new();
        m.observe(&MValue::Int(42), &ctx());
        m.observe(&MValue::Text("hi".into()), &ctx());
        let r = match Box::new(m).finalize() {
            MeasureReport::ByteEntropy(r) => r,
            _ => panic!("wrong report kind"),
        };
        assert_eq!(r.observations, 0);
    }
}
