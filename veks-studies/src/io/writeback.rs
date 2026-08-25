// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dirty-page accounting and the throttle that paces a writer to it.
//!
//! A buffered write does not reach the device. It dirties a page and
//! returns, and the illusion holds right up until the dirty total gets
//! large enough that the kernel starts pushing back. Everything about
//! how a buffered rewrite actually behaves — why it is fast for the
//! first second and then is not, why its write throughput settles at the
//! device's drain rate rather than memory's, why a big page cache moves
//! the cliff instead of removing it — lives in that pushback.
//!
//! # The two thresholds and the timer
//!
//! Linux keeps three knobs and they do three different things
//! ([`Documentation/admin-guide/sysctl/vm.rst`](https://docs.kernel.org/admin-guide/sysctl/vm.html)):
//!
//! | Knob | Default | What crossing it does |
//! |---|---|---|
//! | `dirty_background_ratio` | 10% of available memory | wakes the flusher; the writer is not slowed |
//! | `dirty_ratio` | 20% | the writer is **paused** in `balance_dirty_pages` until writeback catches up |
//! | `dirty_expire_centisecs` | 3000 (30 s) | a page older than this is due, whatever the totals say |
//! | `dirty_writeback_centisecs` | 500 (5 s) | how often the periodic flusher wakes to look |
//!
//! The gap between the two ratios is the whole design: between 10% and
//! 20% the flusher is working and the writer is not blocked, so a burst
//! shorter than that gap costs nothing. Past 20% the writer is a
//! participant in writeback whether it wants to be or not.
//!
//! # Why the pause is a sleep and not an I/O
//!
//! Before 3.2, `balance_dirty_pages` made the dirtying task submit
//! writeback itself, which produced the interleaved, seek-heavy pattern
//! that made buffered writes on rotating media so erratic. Wu
//! Fengguang's *IO-less dirty throttling*
//! ([LWN, 2011](https://lwn.net/Articles/456904/)) replaced it: the task
//! is simply **put to sleep** for a computed interval, leaving writeback
//! to the flusher where it can be issued in a sensible order. The
//! control loop targets a setpoint between the two thresholds, computes
//! a `pos_ratio` from a cubic in the distance from it, scales a per-BDI
//! `dirty_ratelimit` by that ratio, and sleeps the task long enough that
//! its dirtying rate matches. **The sleep is capped at 200 ms**, so a
//! writer that is far over the limit pauses repeatedly rather than once
//! for a long time.
//!
//! What that produces, and what is modelled here, is a writer whose
//! sustained rate equals the device's writeback bandwidth divided by the
//! number of writers, with a burst allowance of `dirty_ratio × RAM`
//! before the pacing engages at all.
//!
//! # What this costs a staged rewrite
//!
//! A staged rewrite writes the payload through this path twice — once
//! into spill and once into the output. Both are sequential, so the
//! flusher can issue them well and the pacing settles at the device's
//! sequential write bandwidth. A scatter writes the same bytes randomly,
//! so the flusher's batches are randomly placed, the drain rate is the
//! device's *random* write bandwidth, and the pacing settles far lower.
//! The throttle does not create that difference — it **transmits** it,
//! from the device back to the application, as a rate the writer cannot
//! exceed no matter how much memory the machine has.

/// Dirty-page thresholds and flusher timing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Writeback {
    pub enabled: bool,
    /// `dirty_background_ratio` — fraction of the cache at which the
    /// flusher starts working in the background.
    pub background_ratio: f64,
    /// `dirty_ratio` — fraction at which a dirtying task is throttled.
    pub ratio: f64,
    /// `dirty_expire_centisecs`, in seconds: how old a dirty page may get
    /// before the periodic flusher writes it regardless of totals.
    pub expire_s: f64,
    /// `dirty_writeback_centisecs`, in seconds: the flusher's wake-up
    /// interval.
    pub interval_s: f64,
    /// The cap on one `balance_dirty_pages` sleep.
    pub max_pause_s: f64,
    /// Pages the flusher submits in one batch. Larger batches are more
    /// sequential on the device and less responsive to the threshold.
    pub batch_pages: usize,
}

impl Writeback {
    /// Linux defaults: 10% background, 20% hard, 30 s expiry, 5 s timer,
    /// 200 ms maximum pause.
    pub const DEFAULT: Self = Writeback {
        enabled: true,
        background_ratio: 0.10,
        ratio: 0.20,
        expire_s: 30.0,
        interval_s: 5.0,
        max_pause_s: 0.200,
        batch_pages: 1024,
    };

    /// `O_DIRECT`, or `O_SYNC`: nothing is buffered, so nothing is paced.
    /// Every write is the application's own I/O and its rate is whatever
    /// the device gives it.
    pub const OFF: Self = Writeback {
        enabled: false,
        background_ratio: 1.0,
        ratio: 1.0,
        expire_s: f64::INFINITY,
        interval_s: f64::INFINITY,
        max_pause_s: 0.0,
        batch_pages: 0,
    };

    /// The configuration a database or a bulk loader typically sets:
    /// small thresholds so that writeback is continuous and the dirty
    /// backlog never becomes a stall long enough to notice.
    ///
    /// Trading burst absorption for predictability is the point — the
    /// sustained rate is unchanged, because it was always the device's.
    pub const SHALLOW: Self = Writeback {
        background_ratio: 0.02,
        ratio: 0.05,
        expire_s: 1.0,
        interval_s: 0.5,
        ..Writeback::DEFAULT
    };

    /// Bytes of dirty data allowed before the flusher is woken.
    pub fn background_bytes(&self, ram_bytes: u64) -> u64 {
        (ram_bytes as f64 * self.background_ratio) as u64
    }

    /// Bytes of dirty data allowed before the writer is throttled.
    pub fn limit_bytes(&self, ram_bytes: u64) -> u64 {
        (ram_bytes as f64 * self.ratio) as u64
    }

    /// The control loop's target, midway between the two thresholds —
    /// `(background + limit) / 2`, which is where `pos_ratio` is 1.
    pub fn setpoint_bytes(&self, ram_bytes: u64) -> u64 {
        (self.background_bytes(ram_bytes) + self.limit_bytes(ram_bytes)) / 2
    }

    /// The position ratio: how hard the loop is leaning on the writer.
    ///
    /// One at the setpoint, rising toward the free-run threshold below
    /// it and falling to zero at the hard limit. Linux uses a cubic in
    /// the normalized distance,
    ///
    /// ```text
    ///   pos_ratio = 1 - ( (dirty - setpoint) / (limit - setpoint) )^3
    /// ```
    ///
    /// which is flat near the setpoint — so small excursions are not
    /// punished — and steep near the limit, where they must be. The
    /// throttled rate is `dirty_ratelimit × pos_ratio`; at the limit that
    /// is zero, which is what makes the limit a limit.
    pub fn position_ratio(&self, dirty_bytes: u64, ram_bytes: u64) -> f64 {
        let setpoint = self.setpoint_bytes(ram_bytes) as f64;
        let limit = self.limit_bytes(ram_bytes) as f64;
        if limit <= setpoint {
            return 1.0;
        }
        let x = ((dirty_bytes as f64 - setpoint) / (limit - setpoint)).clamp(-1.0, 1.0);
        (1.0 - x * x * x).clamp(0.0, 2.0)
    }

    /// How long a task that has just dirtied `bytes` must sleep.
    ///
    /// Below the setpoint this is zero — the free-run region, where a
    /// buffered write really is as fast as memory. Above it the task is
    /// paced to `drain_rate × pos_ratio`, and the sleep is however long
    /// the bytes it just dirtied would take at that rate, less the time
    /// it already spent producing them. The result is capped at
    /// `max_pause_s`; a task further over than one cap can absorb simply
    /// comes back and pauses again.
    pub fn pause_seconds(
        &self,
        dirty_bytes: u64,
        ram_bytes: u64,
        bytes_just_dirtied: u64,
        drain_rate: f64,
    ) -> f64 {
        if !self.enabled || drain_rate <= 0.0 {
            return 0.0;
        }
        if dirty_bytes <= self.setpoint_bytes(ram_bytes) {
            return 0.0;
        }
        let ratio = self.position_ratio(dirty_bytes, ram_bytes);
        if ratio <= 0.0 {
            return self.max_pause_s;
        }
        let allowed_rate = drain_rate * ratio;
        (bytes_just_dirtied as f64 / allowed_rate).min(self.max_pause_s)
    }

    /// Whether the flusher should be running.
    pub fn flusher_wanted(&self, dirty_bytes: u64, ram_bytes: u64) -> bool {
        self.enabled && dirty_bytes >= self.background_bytes(ram_bytes)
    }

    /// Whether the writer must be blocked outright — the dirty total has
    /// reached the hard limit and no amount of pausing has kept up.
    pub fn blocking(&self, dirty_bytes: u64, ram_bytes: u64) -> bool {
        self.enabled && dirty_bytes >= self.limit_bytes(ram_bytes)
    }
}

/// Running state of the flusher for one run.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlusherState {
    /// When the periodic flusher last woke.
    pub last_wake_s: f64,
    /// Total pages the flusher has submitted.
    pub flushed_pages: u64,
    /// Seconds writers spent asleep in `balance_dirty_pages`.
    pub throttled_s: f64,
    /// Times a writer was paused.
    pub pauses: u64,
}

impl FlusherState {
    /// Whether the periodic timer has fired since `last_wake_s`.
    pub fn timer_due(&self, now: f64, policy: &Writeback) -> bool {
        policy.enabled && now - self.last_wake_s >= policy.interval_s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAM: u64 = 1 << 30; // 1 GiB

    #[test]
    fn the_defaults_are_the_kernel_defaults() {
        let w = Writeback::DEFAULT;
        assert_eq!(w.background_bytes(RAM), 107_374_182); // 10%
        assert_eq!(w.limit_bytes(RAM), 214_748_364); // 20%
        assert_eq!(w.expire_s, 30.0);
        assert_eq!(w.interval_s, 5.0);
        assert_eq!(w.max_pause_s, 0.2);
    }

    /// **Free run.** Below the setpoint a buffered write is not paced at
    /// all — which is why a short burst into a large page cache looks
    /// like it ran at memory speed, and why measuring one proves nothing
    /// about sustained throughput.
    #[test]
    fn a_writer_below_the_setpoint_is_not_paced() {
        let w = Writeback::DEFAULT;
        let drain = 500e6;
        for dirty in [0u64, 1 << 20, 50 << 20, 100 << 20] {
            assert_eq!(w.pause_seconds(dirty, RAM, 4_096, drain), 0.0);
        }
        assert!(w.pause_seconds(w.setpoint_bytes(RAM) + 1, RAM, 4_096, drain) > 0.0);
    }

    /// The flusher wakes at the background threshold, well before the
    /// writer notices anything — that gap is what absorbs a burst.
    #[test]
    fn the_flusher_starts_before_the_writer_is_throttled() {
        let w = Writeback::DEFAULT;
        let background = w.background_bytes(RAM);
        assert!(w.flusher_wanted(background, RAM));
        assert!(!w.blocking(background, RAM));
        assert_eq!(w.pause_seconds(background, RAM, 4_096, 500e6), 0.0);
        assert!(w.blocking(w.limit_bytes(RAM), RAM));
    }

    /// The position ratio is one at the setpoint, zero at the limit, and
    /// flat in between near the setpoint — the cubic's whole purpose.
    #[test]
    fn the_position_ratio_is_a_cubic_between_setpoint_and_limit() {
        let w = Writeback::DEFAULT;
        let setpoint = w.setpoint_bytes(RAM);
        let limit = w.limit_bytes(RAM);
        assert!((w.position_ratio(setpoint, RAM) - 1.0).abs() < 1e-9);
        assert!(w.position_ratio(limit, RAM).abs() < 1e-9);

        // A tenth of the way to the limit costs a thousandth of the rate.
        let tenth = setpoint + (limit - setpoint) / 10;
        assert!(
            (w.position_ratio(tenth, RAM) - 0.999).abs() < 0.001,
            "flat near the setpoint: {}",
            w.position_ratio(tenth, RAM)
        );
        // Nine tenths of the way costs most of it.
        let ninth = setpoint + 9 * (limit - setpoint) / 10;
        assert!(
            w.position_ratio(ninth, RAM) < 0.3,
            "steep near the limit: {}",
            w.position_ratio(ninth, RAM)
        );
    }

    /// **The pacing result.** A writer well past the setpoint is held to
    /// the drain rate, so its sustained throughput is the device's, not
    /// memory's — with memory deciding only how long the illusion lasts.
    #[test]
    fn sustained_throughput_converges_on_the_drain_rate() {
        let w = Writeback::DEFAULT;
        let drain = 500e6;
        let chunk = 1 << 20;
        let near_setpoint =
            w.setpoint_bytes(RAM) + (w.limit_bytes(RAM) - w.setpoint_bytes(RAM)) / 20;
        let pause = w.pause_seconds(near_setpoint, RAM, chunk, drain);
        let achieved = chunk as f64 / pause;
        assert!(
            (achieved / drain - 1.0).abs() < 0.02,
            "paced rate {achieved:.0} B/s should track the drain rate {drain:.0} B/s"
        );
    }

    /// And no single pause exceeds the cap, however far over the limit
    /// the writer is — it comes back and pauses again instead.
    #[test]
    fn no_single_pause_exceeds_two_hundred_milliseconds() {
        let w = Writeback::DEFAULT;
        for dirty in [w.limit_bytes(RAM), w.limit_bytes(RAM) * 2, RAM] {
            for chunk in [4_096u64, 1 << 20, 1 << 28] {
                let pause = w.pause_seconds(dirty, RAM, chunk, 10e6);
                assert!(pause <= w.max_pause_s + 1e-12, "{pause} at dirty {dirty}");
            }
        }
    }

    /// A slower drain means a longer pause for the same bytes, which is
    /// the mechanism by which a device's random-write weakness reaches
    /// the application as a rate rather than as a latency.
    #[test]
    fn a_slower_device_paces_the_writer_harder() {
        let w = Writeback::DEFAULT;
        let dirty = w.setpoint_bytes(RAM) + (w.limit_bytes(RAM) - w.setpoint_bytes(RAM)) / 10;
        let fast = w.pause_seconds(dirty, RAM, 1 << 20, 3_000e6);
        let slow = w.pause_seconds(dirty, RAM, 1 << 20, 100e6);
        assert!(
            slow > fast * 20.0,
            "fast {fast:.6}s against slow {slow:.6}s"
        );
    }

    /// With writeback off there is no pacing at all — the `O_DIRECT`
    /// case, where every write is the application's own I/O.
    #[test]
    fn direct_io_is_not_paced() {
        let w = Writeback::OFF;
        assert_eq!(w.pause_seconds(RAM, RAM, 1 << 20, 100e6), 0.0);
        assert!(!w.flusher_wanted(RAM, RAM));
        assert!(!w.blocking(RAM, RAM));
    }

    /// Shallow thresholds move the cliff, not the sustained rate. This is
    /// the tuning most bulk loaders actually want, and it is worth being
    /// explicit that it buys smoothness rather than throughput.
    #[test]
    fn shallow_thresholds_trade_burst_for_predictability() {
        let deep = Writeback::DEFAULT;
        let shallow = Writeback::SHALLOW;
        assert!(shallow.limit_bytes(RAM) < deep.limit_bytes(RAM) / 3);

        // Same distance *proportionally* into the throttled band gives
        // the same pace: the sustained rate is the device's either way.
        let at = |w: &Writeback| {
            let s = w.setpoint_bytes(RAM);
            let l = w.limit_bytes(RAM);
            w.pause_seconds(s + (l - s) / 10, RAM, 1 << 20, 500e6)
        };
        assert!((at(&deep) - at(&shallow)).abs() / at(&deep) < 0.01);
    }
}
