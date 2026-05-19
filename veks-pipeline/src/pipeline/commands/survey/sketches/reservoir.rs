// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Algorithm-R reservoir sampling.
//!
//! Maintains a uniform random sample of fixed size `k` drawn from a
//! stream of unknown length `n`. Every item in the stream ends up in
//! the final sample with probability `k/n`.
//!
//! Reference: Vitter, J. S. (1985). "Random sampling with a reservoir".
//! ACM Trans. Math. Softw. 11(1):37–57.

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::Sketch;

/// Reservoir of fixed capacity `k` over an arbitrary stream item type `T`.
///
/// Memory: `O(k · sizeof(T))` plus 32 bytes of RNG state and a small
/// constant for bookkeeping.
#[derive(Debug, Clone)]
pub struct Reservoir<T> {
    capacity: usize,
    /// Items currently in the reservoir. Length grows to `capacity`
    /// then stays fixed; later items either replace an existing slot
    /// or are discarded.
    items: Vec<T>,
    /// Total items observed (including those discarded). Required for
    /// the replacement probability calculation.
    seen: u64,
    rng: Xoshiro256PlusPlus,
}

impl<T: Clone> Reservoir<T> {
    /// Create a reservoir of capacity `k` seeded from `seed`. Using
    /// an explicit seed keeps surveys reproducible — the same stream
    /// and the same seed produce the same sample.
    pub fn new(capacity: usize, seed: u64) -> Self {
        assert!(capacity > 0, "reservoir capacity must be positive");
        Reservoir {
            capacity,
            items: Vec::with_capacity(capacity),
            seen: 0,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    /// Observe one item from the stream.
    pub fn add(&mut self, item: T) {
        self.seen += 1;
        if self.items.len() < self.capacity {
            self.items.push(item);
        } else {
            // Replace a uniformly-chosen slot with probability k/i,
            // where i is the 1-indexed position of this item.
            let j = self.rng.random_range(0..self.seen);
            if (j as usize) < self.capacity {
                self.items[j as usize] = item;
            }
        }
    }

    /// Borrow the items currently held by the reservoir.
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// Consume the reservoir and return its items.
    pub fn into_items(self) -> Vec<T> {
        self.items
    }

    /// Number of items currently retained. Equals `min(seen, capacity)`.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True iff no items have been observed yet.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Total number of items observed across the full stream (not just
    /// the ones retained).
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// Configured capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Shrink the reservoir to a smaller capacity. Used by the
    /// orchestrator's downscale policy when the memory budget is
    /// pinched. The retained items are a uniform sub-sample of the
    /// current contents.
    pub fn shrink_to(&mut self, new_capacity: usize) {
        assert!(new_capacity > 0, "reservoir capacity must be positive");
        if new_capacity >= self.capacity {
            return;
        }
        // Fisher–Yates partial shuffle: pick new_capacity items
        // uniformly at random without replacement from the current set.
        for i in 0..new_capacity.min(self.items.len()) {
            let j = self.rng.random_range(i..self.items.len());
            self.items.swap(i, j);
        }
        self.items.truncate(new_capacity);
        self.capacity = new_capacity;
    }
}

impl<T> Sketch for Reservoir<T> {
    fn memory_bytes(&self) -> usize {
        let item_size = std::mem::size_of::<T>();
        std::mem::size_of::<Self>() + self.items.capacity() * item_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fills the reservoir until it reaches capacity, then keeps adding.
    /// The retained sample should be exactly the first `k` items if the
    /// stream is shorter than `k`.
    #[test]
    fn short_stream_keeps_everything() {
        let mut r = Reservoir::new(10, 42);
        for i in 0..7 {
            r.add(i);
        }
        assert_eq!(r.len(), 7);
        assert_eq!(r.seen(), 7);
        let mut got = r.into_items();
        got.sort();
        assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    /// At capacity exactly: every item is retained.
    #[test]
    fn exact_capacity_keeps_everything() {
        let mut r = Reservoir::new(5, 7);
        for i in 0..5 {
            r.add(i);
        }
        let mut got = r.into_items();
        got.sort();
        assert_eq!(got, vec![0, 1, 2, 3, 4]);
    }

    /// Uniform-sampling check: with k=100 and n=10_000, every item
    /// should appear in the reservoir with probability k/n = 0.01.
    /// Averaging over many independent runs the empirical inclusion
    /// frequency of item 0 must concentrate near 0.01.
    #[test]
    fn uniform_inclusion_probability() {
        let trials = 5_000;
        let k = 100;
        let n = 10_000;
        let mut count_item_zero = 0;
        for seed in 0..trials {
            let mut r = Reservoir::<u32>::new(k, seed as u64);
            for i in 0..n {
                r.add(i);
            }
            if r.items().contains(&0) {
                count_item_zero += 1;
            }
        }
        let p = count_item_zero as f64 / trials as f64;
        // Expected 0.01 with σ ≈ √(0.01·0.99/5000) ≈ 0.0014.
        // Allow a generous 4σ window so this test never flakes.
        assert!(
            (p - 0.01).abs() < 0.006,
            "empirical inclusion p = {} (expected ≈ 0.01)", p
        );
    }

    /// All `n` items should be equally likely to land in the final
    /// reservoir. The expected count of "item X in final" summed over
    /// X = 0..n is exactly k for every trial; averaged over many
    /// trials the distribution over X should be flat.
    #[test]
    fn distribution_is_flat() {
        let trials = 2_000;
        let k = 50;
        let n = 1_000;
        let mut hist = vec![0usize; n];
        for seed in 0..trials {
            let mut r = Reservoir::<usize>::new(k, seed as u64);
            for i in 0..n {
                r.add(i);
            }
            for &x in r.items() {
                hist[x] += 1;
            }
        }
        // Each bucket: expected (k/n) · trials = 0.05 · 2000 = 100.
        // Check no bucket strays more than 4σ from the expectation,
        // where σ = √(trials · p · (1-p)) = √(2000 · 0.05 · 0.95) ≈ 9.75.
        for (i, &c) in hist.iter().enumerate() {
            assert!(
                (c as f64 - 100.0).abs() < 50.0,
                "bucket {} count {} drifted too far from 100", i, c
            );
        }
    }

    /// Reproducibility: same seed + same stream produces the same
    /// sample, including across separate Reservoir instances.
    #[test]
    fn deterministic_by_seed() {
        let mut a = Reservoir::new(20, 12345);
        let mut b = Reservoir::new(20, 12345);
        for i in 0..1_000 {
            a.add(i);
            b.add(i);
        }
        assert_eq!(a.items(), b.items());
    }

    /// Different seeds: same stream produces different samples
    /// (probabilistically; with k=20 / n=1000 the collision rate is
    /// astronomically low).
    #[test]
    fn distinct_seeds_diverge() {
        let mut a = Reservoir::new(20, 1);
        let mut b = Reservoir::new(20, 2);
        for i in 0..1_000 {
            a.add(i);
            b.add(i);
        }
        assert_ne!(a.items(), b.items());
    }

    /// `shrink_to` reduces the retained sample size and updates the
    /// configured capacity. The remaining items are still drawn from
    /// the original sample.
    #[test]
    fn shrink_to_subsamples() {
        let mut r = Reservoir::new(100, 99);
        for i in 0..1_000 {
            r.add(i);
        }
        let before: std::collections::HashSet<_> = r.items().iter().copied().collect();
        r.shrink_to(25);
        assert_eq!(r.capacity(), 25);
        assert_eq!(r.len(), 25);
        for x in r.items() {
            assert!(before.contains(x), "shrunk item {} was not in original sample", x);
        }
    }

    /// `memory_bytes` scales with the reservoir's Vec capacity. A
    /// larger reservoir reports more bytes than a smaller one over
    /// identically-sized items.
    #[test]
    fn memory_reports_scale_with_capacity() {
        let small = Reservoir::<u64>::new(8, 0);
        let large = Reservoir::<u64>::new(128, 0);
        assert!(large.memory_bytes() > small.memory_bytes());
    }

    #[test]
    #[should_panic(expected = "capacity must be positive")]
    fn zero_capacity_rejected() {
        let _ = Reservoir::<u32>::new(0, 0);
    }
}
