// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Canonical KNN result comparison logic.
//!
//! Single source of truth for comparing computed KNN results against ground
//! truth. Used by all verification commands. Implements the same logic as
//! the knnutils personality's `verify dataset-knnutils`.
//!
//! ## Comparison model
//!
//! KNN results are compared as **neighbor sets**, not ordered lists.
//! Multi-threaded BLAS (MKL/OpenBLAS) is non-deterministic across calls
//! with different batch sizes — the thread block decomposition in sgemm
//! produces different floating-point rounding. Queries where the only
//! difference is a small number of boundary neighbors (swapped at ULP-level
//! distance ties) are expected and acceptable.
//!
//! A query is a "boundary mismatch" when ≤ 5 neighbors differ; larger
//! mismatches indicate a real problem.

use std::collections::HashSet;

/// Maximum number of neighbor swaps before a query is considered a real
/// mismatch. BLAS non-determinism in multi-threaded environments can
/// cause a small number of swaps at the k-th boundary.
pub const BOUNDARY_THRESHOLD: usize = 5;

/// Result of comparing one query's KNN results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryResult {
    /// All k neighbors match in exact order.
    ExactMatch,
    /// Same neighbor set, different order (tie-breaking).
    SetMatch,
    /// Sets differ by ≤ BOUNDARY_THRESHOLD neighbors (BLAS rounding).
    BoundaryMismatch(usize),
    /// Sets differ by > BOUNDARY_THRESHOLD neighbors (real error).
    RealMismatch(usize),
}

impl QueryResult {
    /// Whether this result is acceptable (not a real mismatch).
    pub fn is_acceptable(&self) -> bool {
        !matches!(self, QueryResult::RealMismatch(_))
    }
}

/// Compare one query's KNN result against ground truth.
///
/// `computed` and `expected` are ordinal slices of length k.
/// Negative values are filtered out (FAISS uses -1 for invalid).
pub fn compare_query_ordinals(computed: &[i32], expected: &[i32]) -> QueryResult {
    let computed_set: HashSet<i32> = computed.iter().copied().filter(|&v| v >= 0).collect();
    let expected_set: HashSet<i32> = expected.iter().copied().filter(|&v| v >= 0).collect();

    if computed_set == expected_set {
        if computed == expected {
            QueryResult::ExactMatch
        } else {
            QueryResult::SetMatch
        }
    } else {
        let diff_count = computed_set.symmetric_difference(&expected_set).count() / 2;
        if diff_count <= BOUNDARY_THRESHOLD {
            QueryResult::BoundaryMismatch(diff_count)
        } else {
            QueryResult::RealMismatch(diff_count)
        }
    }
}

/// Summary of a batch of query comparisons.
#[derive(Debug, Default)]
pub struct VerifySummary {
    pub total: usize,
    pub exact_match: usize,
    pub set_match: usize,
    pub boundary_mismatch: usize,
    pub real_mismatch: usize,
}

impl VerifySummary {
    pub fn record(&mut self, result: &QueryResult) {
        self.total += 1;
        match result {
            QueryResult::ExactMatch => self.exact_match += 1,
            QueryResult::SetMatch => self.set_match += 1,
            QueryResult::BoundaryMismatch(_) => self.boundary_mismatch += 1,
            QueryResult::RealMismatch(_) => self.real_mismatch += 1,
        }
    }

    pub fn all_acceptable(&self) -> bool {
        self.real_mismatch == 0
    }

    pub fn pass_count(&self) -> usize {
        self.exact_match + self.set_match + self.boundary_mismatch
    }

    /// Format a human-readable summary.
    pub fn summary_line(&self) -> String {
        format!(
            "{}/{} pass (exact={}, set={}, boundary≤{}={}, real>{}={})",
            self.pass_count(), self.total,
            self.exact_match, self.set_match,
            BOUNDARY_THRESHOLD, self.boundary_mismatch,
            BOUNDARY_THRESHOLD, self.real_mismatch,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        assert_eq!(
            compare_query_ordinals(&[0, 1, 2], &[0, 1, 2]),
            QueryResult::ExactMatch,
        );
    }

    #[test]
    fn test_set_match() {
        assert_eq!(
            compare_query_ordinals(&[1, 0, 2], &[0, 1, 2]),
            QueryResult::SetMatch,
        );
    }

    #[test]
    fn test_boundary_mismatch() {
        // 1 swap: ordinal 3 vs ordinal 2
        assert_eq!(
            compare_query_ordinals(&[0, 1, 3], &[0, 1, 2]),
            QueryResult::BoundaryMismatch(1),
        );
    }

    #[test]
    fn test_boundary_at_threshold() {
        // 5 swaps (at threshold)
        assert_eq!(
            compare_query_ordinals(
                &[10, 11, 12, 13, 14, 5, 6, 7, 8, 9],
                &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            QueryResult::BoundaryMismatch(5),
        );
        assert!(QueryResult::BoundaryMismatch(5).is_acceptable());
    }

    #[test]
    fn test_real_mismatch() {
        // 6 swaps (over threshold)
        assert_eq!(
            compare_query_ordinals(
                &[10, 11, 12, 13, 14, 15, 6, 7, 8, 9],
                &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),
            QueryResult::RealMismatch(6),
        );
        assert!(!QueryResult::RealMismatch(6).is_acceptable());
    }

    #[test]
    fn test_negative_indices_filtered() {
        assert_eq!(
            compare_query_ordinals(&[0, 1, -1], &[0, 1, -1]),
            QueryResult::ExactMatch,
        );
    }

    #[test]
    fn test_summary() {
        let mut s = VerifySummary::default();
        s.record(&QueryResult::ExactMatch);
        s.record(&QueryResult::SetMatch);
        s.record(&QueryResult::BoundaryMismatch(2));
        s.record(&QueryResult::RealMismatch(10));
        assert_eq!(s.total, 4);
        assert_eq!(s.pass_count(), 3);
        assert!(!s.all_acceptable());
    }
}
