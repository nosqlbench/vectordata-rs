// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! PRNG infrastructure for reproducible seeded randomness.
//!
//! Uses XoShiRo256++ (via `rand_xoshiro`) — the same algorithm family as the
//! Java Apache Commons RNG implementation used by the Java nbvectors commands.
//! This ensures cross-language reproducibility when the same seed is used.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Create a seeded PRNG from an integer seed.
///
/// Expands the integer seed into a 256-bit state by hashing. This matches
/// the Java `RandomSource.XO_SHI_RO_256_PP.create(seed)` behavior of
/// expanding a long seed into the full internal state.
pub fn seeded_rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// Parse a seed from an options string, defaulting to 0 if absent or invalid.
pub fn parse_seed(seed_str: Option<&str>) -> u64 {
    seed_str
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

/// Perform a Fisher-Yates shuffle on a mutable slice using the given PRNG.
///
/// This is the standard inside-out Fisher-Yates algorithm, producing a
/// uniformly random permutation deterministically for a given seed.
pub fn fisher_yates_shuffle<T>(slice: &mut [T], rng: &mut Xoshiro256PlusPlus) {
    use rand::Rng;
    let len = slice.len();
    for i in (1..len).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seeded_rng_deterministic() {
        use rand::Rng;
        let mut rng1 = seeded_rng(42);
        let mut rng2 = seeded_rng(42);
        let vals1: Vec<u32> = (0..10).map(|_| rng1.random()).collect();
        let vals2: Vec<u32> = (0..10).map(|_| rng2.random()).collect();
        assert_eq!(vals1, vals2);
    }

    #[test]
    fn test_different_seeds_differ() {
        use rand::Rng;
        let mut rng1 = seeded_rng(1);
        let mut rng2 = seeded_rng(2);
        let vals1: Vec<u32> = (0..10).map(|_| rng1.random()).collect();
        let vals2: Vec<u32> = (0..10).map(|_| rng2.random()).collect();
        assert_ne!(vals1, vals2);
    }

    #[test]
    fn test_fisher_yates_deterministic() {
        let mut a = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut b = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut rng1 = seeded_rng(42);
        let mut rng2 = seeded_rng(42);
        fisher_yates_shuffle(&mut a, &mut rng1);
        fisher_yates_shuffle(&mut b, &mut rng2);
        assert_eq!(a, b);
    }

    #[test]
    fn test_fisher_yates_permutation() {
        let mut vals: Vec<i32> = (0..100).collect();
        let mut rng = seeded_rng(99);
        fisher_yates_shuffle(&mut vals, &mut rng);
        // Should be a permutation — same elements, different order
        let mut sorted = vals.clone();
        sorted.sort();
        let expected: Vec<i32> = (0..100).collect();
        assert_eq!(sorted, expected);
        // Should not be identity (astronomically unlikely for 100 elements)
        let identity: Vec<i32> = (0..100).collect();
        assert_ne!(vals, identity);
    }

    #[test]
    fn test_parse_seed() {
        assert_eq!(parse_seed(Some("42")), 42);
        assert_eq!(parse_seed(Some("0")), 0);
        assert_eq!(parse_seed(None), 0);
        assert_eq!(parse_seed(Some("invalid")), 0);
    }
}
