// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Survey-side adapter around [`ResourceGovernor`].
//!
//! The orchestrator consults the governor at every batch boundary
//! and obeys throttle / emergency flags between records. This
//! module bundles those operations into a small surface so the
//! orchestrator's two-pass driver doesn't have to repeat the same
//! plumbing in two places, and so a future change to the governor
//! contract has exactly one site to update.
//!
//! It also owns the **downscale policy** (§13.11.2 of the sysref):
//! the priority order in which sketch precision is sacrificed when
//! the memory ceiling is reached. The orchestrator calls
//! [`Downscaler::next_action`] to ask "if I need to free memory
//! right now, what should I shrink next?", applies the returned
//! action to its own state, then re-measures and asks again until
//! the budget is satisfied or every option is exhausted.

use std::time::{Duration, Instant};

use crate::pipeline::resource::ResourceGovernor;

/// Batched-governor adapter.
///
/// A thin wrapper over `&ResourceGovernor` that exposes only the
/// operations the survey orchestrator needs. Holds no state of its
/// own beyond the governor reference and the last-checkpoint timer
/// (which prevents over-frequent governor checkpoints in tight
/// inner loops).
pub struct GovernorAdapter<'g> {
    governor: &'g ResourceGovernor,
    last_checkpoint: Instant,
    min_interval: Duration,
}

impl<'g> GovernorAdapter<'g> {
    pub fn new(governor: &'g ResourceGovernor) -> Self {
        GovernorAdapter {
            governor,
            last_checkpoint: Instant::now()
                .checked_sub(Duration::from_secs(60))
                .unwrap_or_else(Instant::now),
            min_interval: Duration::from_millis(200),
        }
    }

    /// Current effective `mem` budget in bytes, or a fallback when
    /// the governor has no mem ceiling configured (rare; the
    /// `default_governor` always provides one). The fallback is
    /// 256 MB — the per-survey CLI default.
    pub fn mem_bytes(&self) -> u64 {
        self.governor.current("mem").unwrap_or(256 * 1024 * 1024)
    }

    /// Effective thread budget.
    pub fn threads(&self) -> u64 {
        self.governor.current("threads").unwrap_or(1)
    }

    /// Effective segmentsize. The orchestrator uses this as the
    /// records-per-batch default; the CLI's `--batch-size` overrides.
    pub fn segment_size(&self, fallback: u64) -> u64 {
        self.governor.current("segmentsize").unwrap_or(fallback)
    }

    /// True if the governor has asked commands to pause / slow
    /// down. The orchestrator checks this between records and
    /// stalls until the flag clears.
    pub fn should_throttle(&self) -> bool {
        self.governor.should_throttle()
    }

    /// Surface a memory demand to the governor. The orchestrator
    /// calls this when it has a productive use for more sketch
    /// memory (e.g. on large schemas). Returns the granted value.
    pub fn offer_mem_demand(&self, current: u64, desired: u64) -> u64 {
        self.governor.offer_demand("mem", current, desired)
    }

    /// Force a governor checkpoint, rate-limited by `min_interval`.
    /// Returns `true` if a checkpoint actually ran.
    pub fn maybe_checkpoint(&mut self) -> bool {
        if self.last_checkpoint.elapsed() < self.min_interval {
            return false;
        }
        self.last_checkpoint = Instant::now();
        self.governor.checkpoint()
    }

    /// Tell the governor which step the survey is currently
    /// executing. Surfaces in the governor's log so an operator
    /// reviewing budget adjustments can attribute them to the
    /// survey rather than to a sibling step.
    pub fn set_step_id(&self, id: &str) {
        self.governor.set_step_id(id);
    }
}

// ---------------------------------------------------------------------------
// Downscale policy
// ---------------------------------------------------------------------------

/// One step the orchestrator should take to reduce memory pressure.
///
/// The orchestrator applies these in the order returned by
/// [`Downscaler::next_action`]; after each application it
/// re-measures aggregate sketch memory and asks again. The sequence
/// terminates with `None` when nothing more can be shed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownscaleAction {
    /// Shrink every reservoir to the next-smaller tier.
    ShrinkReservoirs { from: usize, to: usize },
    /// Drop the lowest-priority pair analyzers. The orchestrator
    /// supplies the priority ranking; this action just signals
    /// "drop one tier".
    DropLowPriorityPairAnalyzers,
    /// Reduce KLL precision parameter `k`.
    ReduceKllK { from: usize, to: usize },
    /// Reduce HLL precision exponent.
    ReduceHllPrecision { from: u8, to: u8 },
    /// Drop measures for the lowest-priority fields (those with the
    /// largest distinct-tracker / uniqueness scores).
    DropLowPriorityFields,
}

/// Tracker for the downscale sequence. Holds the current "tier" at
/// each level so the orchestrator only has to ask
/// [`Downscaler::next_action`] without remembering what's been done.
#[derive(Debug, Clone)]
pub struct Downscaler {
    reservoir_tier: usize,
    kll_k_tier: usize,
    hll_precision_tier: usize,
    pair_analyzer_tiers_dropped: usize,
    field_tiers_dropped: usize,
}

impl Default for Downscaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Downscaler {
    pub fn new() -> Self {
        Downscaler {
            reservoir_tier: 0,
            kll_k_tier: 0,
            hll_precision_tier: 0,
            pair_analyzer_tiers_dropped: 0,
            field_tiers_dropped: 0,
        }
    }

    /// Sequence: shrink reservoirs → drop low-priority pair
    /// analyzers → reduce KLL k → reduce HLL precision → drop
    /// low-priority fields. Each tier is one step; the orchestrator
    /// keeps calling until `None`.
    pub fn next_action(&mut self) -> Option<DownscaleAction> {
        const RESERVOIR_TIERS: &[usize] = &[1024, 512, 256, 128, 64];
        const KLL_K_TIERS: &[usize] = &[200, 100, 64, 32];
        const HLL_P_TIERS: &[u8] = &[14, 12, 10, 8];

        // 1) Reservoir shrink
        if self.reservoir_tier + 1 < RESERVOIR_TIERS.len() {
            let from = RESERVOIR_TIERS[self.reservoir_tier];
            let to = RESERVOIR_TIERS[self.reservoir_tier + 1];
            self.reservoir_tier += 1;
            return Some(DownscaleAction::ShrinkReservoirs { from, to });
        }
        // 2) Drop low-priority pair analyzers (up to 3 tiers).
        if self.pair_analyzer_tiers_dropped < 3 {
            self.pair_analyzer_tiers_dropped += 1;
            return Some(DownscaleAction::DropLowPriorityPairAnalyzers);
        }
        // 3) KLL k
        if self.kll_k_tier + 1 < KLL_K_TIERS.len() {
            let from = KLL_K_TIERS[self.kll_k_tier];
            let to = KLL_K_TIERS[self.kll_k_tier + 1];
            self.kll_k_tier += 1;
            return Some(DownscaleAction::ReduceKllK { from, to });
        }
        // 4) HLL precision
        if self.hll_precision_tier + 1 < HLL_P_TIERS.len() {
            let from = HLL_P_TIERS[self.hll_precision_tier];
            let to = HLL_P_TIERS[self.hll_precision_tier + 1];
            self.hll_precision_tier += 1;
            return Some(DownscaleAction::ReduceHllPrecision { from, to });
        }
        // 5) Drop low-priority fields (up to 3 tiers).
        if self.field_tiers_dropped < 3 {
            self.field_tiers_dropped += 1;
            return Some(DownscaleAction::DropLowPriorityFields);
        }
        None
    }

    /// True iff every downscale option has been exhausted. The
    /// orchestrator should refuse further work and surface a fatal
    /// error rather than continue past this point.
    pub fn exhausted(&self) -> bool {
        const RESERVOIR_TIERS_LAST: usize = 4;
        const KLL_LAST: usize = 3;
        const HLL_LAST: usize = 3;
        self.reservoir_tier == RESERVOIR_TIERS_LAST
            && self.pair_analyzer_tiers_dropped == 3
            && self.kll_k_tier == KLL_LAST
            && self.hll_precision_tier == HLL_LAST
            && self.field_tiers_dropped == 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downscale_sequence_in_order() {
        let mut d = Downscaler::new();
        // 4 reservoir shrinks
        for from_to in [(1024, 512), (512, 256), (256, 128), (128, 64)] {
            let a = d.next_action().unwrap();
            assert_eq!(
                a,
                DownscaleAction::ShrinkReservoirs { from: from_to.0, to: from_to.1 }
            );
        }
        // 3 pair-analyzer drops
        for _ in 0..3 {
            assert_eq!(d.next_action().unwrap(), DownscaleAction::DropLowPriorityPairAnalyzers);
        }
        // 3 KLL k reductions
        for from_to in [(200, 100), (100, 64), (64, 32)] {
            let a = d.next_action().unwrap();
            assert_eq!(a, DownscaleAction::ReduceKllK { from: from_to.0, to: from_to.1 });
        }
        // 3 HLL precision reductions
        for from_to in [(14u8, 12), (12, 10), (10, 8)] {
            let a = d.next_action().unwrap();
            assert_eq!(a, DownscaleAction::ReduceHllPrecision { from: from_to.0, to: from_to.1 });
        }
        // 3 field drops
        for _ in 0..3 {
            assert_eq!(d.next_action().unwrap(), DownscaleAction::DropLowPriorityFields);
        }
        // Exhausted.
        assert!(d.next_action().is_none());
        assert!(d.exhausted());
    }

    #[test]
    fn downscale_yields_none_when_exhausted() {
        let mut d = Downscaler::new();
        while d.next_action().is_some() {}
        assert!(d.exhausted());
        assert!(d.next_action().is_none());
    }
}
