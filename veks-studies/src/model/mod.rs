// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The virtualized world: store geometry, reorder maps, and the trace of
//! operations an algorithm performs against them.

pub mod trace;

pub use trace::{Metrics, Op, Trace};

use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

/// A store described the way the cost model describes one.
///
/// No bytes exist. A record is its ordinal; a container is a fixed run of
/// records; addresses are derived. This is deliberately the same handful
/// of parameters the documents use, so a study's inputs and a document's
/// symbols are the same things.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// `N_src` — records in the source.
    pub records: u64,
    /// `R` — bytes per record.
    pub record_bytes: u64,
    /// `W` — bytes per container, the unit the tier fetches whole.
    pub container_bytes: u64,
}

impl Geometry {
    pub fn new(records: u64, record_bytes: u64, container_bytes: u64) -> Self {
        assert!(records > 0, "a store needs records");
        assert!(record_bytes > 0, "records need a size");
        assert!(
            container_bytes >= record_bytes,
            "a container must hold at least one record"
        );
        Self {
            records,
            record_bytes,
            container_bytes,
        }
    }

    /// `w` — records per container.
    pub fn records_per_container(&self) -> u64 {
        (self.container_bytes / self.record_bytes).max(1)
    }

    /// Total containers spanned by the store.
    pub fn container_count(&self) -> u64 {
        self.records.div_ceil(self.records_per_container())
    }

    /// Which container holds a record. Ordinal order is address order —
    /// the family's monotonicity premise — so this is division.
    pub fn container_of(&self, ordinal: u64) -> u64 {
        ordinal / self.records_per_container()
    }

    /// Live bytes in the store.
    pub fn payload_bytes(&self) -> u64 {
        self.records * self.record_bytes
    }

    /// `S` — records a segment of `budget` bytes can hold.
    pub fn records_per_segment(&self, budget_bytes: u64) -> u64 {
        (budget_bytes / self.record_bytes).max(1)
    }

    /// `P` — pass count for a budget, with the floor of two segments the
    /// Segment step imposes.
    pub fn passes(&self, output_count: u64, budget_bytes: u64) -> u64 {
        let per_segment = self.records_per_segment(budget_bytes);
        output_count.div_ceil(per_segment).max(2)
    }

    /// The published amplification prediction,
    /// `A(P) = P · (1 − exp(−w / P))`, for a full permutation.
    ///
    /// This is the formula under test. [`crate::study`] compares it
    /// against what the simulator actually counts.
    pub fn predicted_amplification(&self, passes: u64) -> f64 {
        let w = self.records_per_container() as f64;
        let p = passes as f64;
        p * (1.0 - (-w / p).exp())
    }
}

/// A destination-ordered reorder map: `map[i]` is the source ordinal
/// belonging at output position `i`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Map(pub Vec<u64>);

impl Map {
    /// The identity map over `n` records — a rewrite that moves nothing.
    pub fn identity(n: u64) -> Self {
        Map((0..n).collect())
    }

    /// A seeded uniform random permutation, the case the cost model's
    /// uniformity assumption describes.
    pub fn shuffled(n: u64, seed: u64) -> Self {
        let mut v: Vec<u64> = (0..n).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        v.shuffle(&mut rng);
        Map(v)
    }

    /// A rotation by `k`, a permutation with perfect locality — every
    /// output run is a source run.
    pub fn rotated(n: u64, k: u64) -> Self {
        Map((0..n).map(|i| (i + k) % n).collect())
    }

    /// Reverse order: monotone, but descending, so a linearizing pass has
    /// the most work to undo.
    pub fn reversed(n: u64) -> Self {
        Map((0..n).rev().collect())
    }

    pub fn len(&self) -> u64 {
        self.0.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether every source ordinal below `records` appears exactly once.
    pub fn is_permutation_of(&self, records: u64) -> bool {
        if self.len() != records {
            return false;
        }
        let mut seen = vec![false; records as usize];
        for &s in &self.0 {
            if s >= records || seen[s as usize] {
                return false;
            }
            seen[s as usize] = true;
        }
        true
    }
}

/// The output side: slot `i` holds the source ordinal written there, or
/// `None` if nothing has been written yet.
#[derive(Debug, Clone)]
pub struct Sink {
    pub slots: Vec<Option<u64>>,
}

impl Sink {
    pub fn new(len: u64) -> Self {
        Sink {
            slots: vec![None; len as usize],
        }
    }

    /// Every slot filled exactly once, holding what the map called for.
    pub fn matches(&self, map: &Map) -> bool {
        self.slots.len() == map.0.len()
            && self
                .slots
                .iter()
                .zip(&map.0)
                .all(|(got, want)| *got == Some(*want))
    }

    /// Slots still unwritten, if any.
    pub fn unfilled(&self) -> usize {
        self.slots.iter().filter(|s| s.is_none()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_derives_the_documents_symbols() {
        // 4100-byte records, 128 KiB containers — the cost model's own
        // worked example, where w is documented as 31.
        let g = Geometry::new(1_000, 4100, 128 * 1024);
        assert_eq!(g.records_per_container(), 31);
        assert_eq!(g.container_count(), 1_000u64.div_ceil(31));

        // The floor of two segments applies even when everything fits.
        assert_eq!(g.passes(1_000, u64::MAX / 2), 2);
        // And the pass count is ceil(N / S) otherwise.
        let g2 = Geometry::new(1_000, 100, 1_000);
        assert_eq!(g2.records_per_segment(10_000), 100);
        assert_eq!(g2.passes(1_000, 10_000), 10);
    }

    #[test]
    fn predicted_amplification_matches_the_published_table() {
        // docs/gsplat/cost-model.md tabulates A for w = 32.
        let g = Geometry::new(1_000_000, 4096, 128 * 1024);
        assert_eq!(g.records_per_container(), 32);
        // These are the values the formula actually produces, checked to
        // two decimals. The published table originally carried 14.6 at
        // P = 16; this test is what caught it.
        let expect = [
            (2u64, 2.00),
            (4, 4.00),
            (8, 7.85),
            (16, 13.84),
            (32, 20.23),
            (54, 24.14),
            (100, 27.39),
        ];
        for (p, want) in expect {
            let got = g.predicted_amplification(p);
            assert!(
                (got - want).abs() < 0.01,
                "A({p}) = {got:.4}, table says {want}"
            );
        }
    }

    #[test]
    fn maps_are_permutations() {
        assert!(Map::identity(64).is_permutation_of(64));
        assert!(Map::shuffled(64, 7).is_permutation_of(64));
        assert!(Map::rotated(64, 13).is_permutation_of(64));
        assert!(Map::reversed(64).is_permutation_of(64));
        assert!(!Map(vec![0, 0, 2]).is_permutation_of(3));
        assert!(!Map(vec![0, 1, 9]).is_permutation_of(3));
    }

    #[test]
    fn shuffles_are_reproducible_and_seed_dependent() {
        assert_eq!(Map::shuffled(256, 42), Map::shuffled(256, 42));
        assert_ne!(Map::shuffled(256, 42), Map::shuffled(256, 43));
    }
}
