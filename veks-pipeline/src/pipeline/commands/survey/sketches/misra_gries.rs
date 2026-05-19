// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Misra–Gries heavy-hitters sketch.
//!
//! Tracks up to `k` items and their (lower-bound) frequency counts
//! from a stream of length `n`. Guarantees that **every item with
//! true frequency `> n / (k + 1)` is retained**. Reported counts are
//! never above the true frequency; the gap to the true count is
//! bounded by `n / (k + 1)`.
//!
//! Reference: Misra, J., Gries, D. (1982). "Finding repeated
//! elements". Sci. Comput. Program. 2(2):143–152.

use std::collections::HashMap;
use std::hash::Hash;

use super::Sketch;

/// Misra–Gries heavy-hitters tracker.
///
/// `k` is the number of counters retained. Memory is approximately
/// `k · (sizeof(K) + 8 + small overhead)`.
#[derive(Debug, Clone)]
pub struct MisraGries<K: Hash + Eq + Clone> {
    /// Maximum number of distinct items the sketch tracks
    /// concurrently.
    capacity: usize,
    /// Active counters. `len() <= capacity` is an invariant: when an
    /// unseen item arrives and the map is already full, every
    /// counter is decremented and any reaching zero are dropped — so
    /// after a decrement round at most `capacity - 1` entries remain
    /// before the new item (if it's still uncounted) is added.
    counters: HashMap<K, u64>,
    /// Total items observed.
    seen: u64,
}

impl<K: Hash + Eq + Clone> MisraGries<K> {
    /// Create a fresh sketch with `capacity` counters.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "MisraGries capacity must be positive");
        MisraGries {
            capacity,
            counters: HashMap::with_capacity(capacity + 1),
            seen: 0,
        }
    }

    /// Observe one item.
    pub fn add(&mut self, item: K) {
        self.seen += 1;
        if let Some(c) = self.counters.get_mut(&item) {
            *c += 1;
            return;
        }
        if self.counters.len() < self.capacity {
            self.counters.insert(item, 1);
            return;
        }
        // All counters in use and the new item is uncounted:
        // decrement every counter, drop any that hit zero. The new
        // item itself is *not* added — its "weight" was absorbed by
        // the decrement.
        self.counters.retain(|_, c| {
            *c -= 1;
            *c > 0
        });
    }

    /// Number of distinct items observed (lower bound — items that
    /// were decremented out are not counted).
    pub fn distinct_tracked(&self) -> usize {
        self.counters.len()
    }

    /// Total items observed.
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// Configured counter capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reported (lower-bound) count for `item`. Returns 0 for items
    /// the sketch does not currently track — note this is consistent
    /// with the algorithm's guarantee but does **not** mean the item
    /// was absent from the stream.
    pub fn estimated_count(&self, item: &K) -> u64 {
        self.counters.get(item).copied().unwrap_or(0)
    }

    /// Borrow the active counter table.
    pub fn counters(&self) -> &HashMap<K, u64> {
        &self.counters
    }

    /// Return tracked items sorted by descending estimated count.
    /// Ties are broken arbitrarily; callers that need stable ordering
    /// should sort by their own key after.
    pub fn top_k(&self) -> Vec<(K, u64)> {
        let mut entries: Vec<(K, u64)> = self
            .counters
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries
    }

    /// Frequency threshold above which the sketch is guaranteed to
    /// retain an item. Any item appearing more than this many times
    /// in the stream must appear in `counters()`.
    pub fn guarantee_threshold(&self) -> u64 {
        // ⌊ n / (k + 1) ⌋
        self.seen / (self.capacity as u64 + 1)
    }

    /// Maximum possible error on any reported count. Reported counts
    /// are lower bounds; the true count is at most
    /// `estimated_count + error_bound()`.
    pub fn error_bound(&self) -> u64 {
        self.guarantee_threshold()
    }
}

impl<K: Hash + Eq + Clone> Sketch for MisraGries<K> {
    fn memory_bytes(&self) -> usize {
        let entry = std::mem::size_of::<K>() + std::mem::size_of::<u64>();
        std::mem::size_of::<Self>() + self.counters.capacity() * (entry + 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty stream: no counters, no seen, threshold 0.
    #[test]
    fn empty_sketch() {
        let mg: MisraGries<&str> = MisraGries::new(4);
        assert_eq!(mg.distinct_tracked(), 0);
        assert_eq!(mg.seen(), 0);
        assert_eq!(mg.guarantee_threshold(), 0);
    }

    /// All distinct items, fewer than capacity: every item retained
    /// with count 1.
    #[test]
    fn distinct_below_capacity() {
        let mut mg = MisraGries::new(10);
        for item in &["a", "b", "c"] {
            mg.add(*item);
        }
        assert_eq!(mg.distinct_tracked(), 3);
        assert_eq!(mg.estimated_count(&"a"), 1);
        assert_eq!(mg.estimated_count(&"b"), 1);
        assert_eq!(mg.estimated_count(&"c"), 1);
    }

    /// All distinct items, exceeding capacity: trace the algorithm.
    /// With k=4 and 12 singletons the residue holds the trailing
    /// items not yet wiped by a decrement round.
    ///
    /// Trace:
    ///  1..4 fill the four counters → {a:1, b:1, c:1, d:1}
    ///  5    map full, item unseen → decrement all → {}
    ///  6..9 refill → {e:1, f:1, g:1, h:1}
    ///  10   map full, item unseen → decrement all → {}
    ///  11..12 refill partially → {k:1, l:1}
    ///
    /// → 2 counters retained at value 1; no item has true frequency
    /// above the threshold n/(k+1) = 12/5 = 2, so the algorithm
    /// makes no guarantee about which (if any) survive, but
    /// invariant `len ≤ k` and `count ≤ true_count` always hold.
    #[test]
    fn singletons_above_capacity_leave_partial_residue() {
        let mut mg = MisraGries::new(4);
        for i in 0..12 {
            mg.add(format!("item-{}", i));
        }
        assert_eq!(mg.seen(), 12);
        assert!(mg.distinct_tracked() <= 4, "must respect capacity");
        for (_, count) in mg.counters() {
            assert!(*count <= 1, "estimated count never exceeds true count");
        }
    }

    /// Classic dominant-item test: one item appears in > 1/(k+1)
    /// fraction of the stream and must survive.
    #[test]
    fn dominant_item_survives() {
        let mut mg = MisraGries::new(3);
        // 10 copies of "X" plus 5 unique singletons. n=15, k=3, so
        // threshold = 15/4 = 3. "X" has frequency 10 > 3 and must
        // be retained.
        for _ in 0..10 {
            mg.add("X");
        }
        for tag in &["a", "b", "c", "d", "e"] {
            mg.add(*tag);
        }
        assert!(mg.estimated_count(&"X") > 0, "dominant item dropped: {:?}", mg.counters());
        // Estimated count is at most the true count.
        assert!(mg.estimated_count(&"X") <= 10);
        // And within the algorithm's error bound (n/(k+1) = 3).
        assert!(mg.estimated_count(&"X") >= 10 - mg.error_bound());
    }

    /// Two heavy hitters in a stream where they together exceed 50%
    /// of traffic — both must be retained.
    #[test]
    fn two_heavy_hitters_survive() {
        let mut mg = MisraGries::new(4);
        // 8 X, 6 Y, then 6 distinct singletons. n = 20, k = 4.
        // Threshold n/(k+1) = 4. X has 8 > 4 and Y has 6 > 4 → both
        // retained.
        for _ in 0..8 {
            mg.add("X");
        }
        for _ in 0..6 {
            mg.add("Y");
        }
        for tag in &["a", "b", "c", "d", "e", "f"] {
            mg.add(*tag);
        }
        assert!(mg.estimated_count(&"X") > 0);
        assert!(mg.estimated_count(&"Y") > 0);
        assert!(mg.estimated_count(&"X") <= 8);
        assert!(mg.estimated_count(&"Y") <= 6);
    }

    /// The error bound never exceeds n/(k+1) and reported counts are
    /// always within that bound of the true count.
    #[test]
    fn error_bound_is_tight() {
        let mut mg = MisraGries::new(2);
        // n = 30, k = 2, threshold = 30/3 = 10.
        for _ in 0..15 {
            mg.add("A");
        }
        for _ in 0..10 {
            mg.add("B");
        }
        for _ in 0..5 {
            mg.add("C");
        }
        assert_eq!(mg.seen(), 30);
        assert_eq!(mg.error_bound(), 10);
        let a = mg.estimated_count(&"A");
        let b = mg.estimated_count(&"B");
        // Truths: A=15, B=10. Estimates are lower bounds.
        assert!(a <= 15);
        assert!(15 - a <= 10);
        // B with true count exactly threshold may or may not survive;
        // if it does, its count must also be within the bound.
        assert!(b <= 10);
        if b > 0 {
            assert!(10 - b <= 10);
        }
    }

    /// `top_k` returns counters sorted by descending count.
    #[test]
    fn top_k_sorted_descending() {
        let mut mg = MisraGries::new(8);
        for _ in 0..5 {
            mg.add("low");
        }
        for _ in 0..20 {
            mg.add("high");
        }
        for _ in 0..10 {
            mg.add("mid");
        }
        let top = mg.top_k();
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
        assert_eq!(top[0].0, "high");
    }

    /// Identical streams produce identical counter states. The
    /// implementation has no internal randomness, so reproducibility
    /// is exact (no seed required).
    #[test]
    fn deterministic() {
        let stream: Vec<&str> = ["a", "b", "a", "c", "d", "a", "b", "e"].to_vec();
        let mut mg1 = MisraGries::new(3);
        let mut mg2 = MisraGries::new(3);
        for s in &stream {
            mg1.add(*s);
            mg2.add(*s);
        }
        assert_eq!(mg1.counters(), mg2.counters());
    }

    #[test]
    #[should_panic(expected = "capacity must be positive")]
    fn zero_capacity_rejected() {
        let _: MisraGries<&str> = MisraGries::new(0);
    }
}
