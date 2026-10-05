// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! What a storage device physically is, in enough detail to derive its
//! behaviour rather than tabulate it.
//!
//! The closed-form model in [`crate::device`] lumps everything into four
//! fitted numbers. That is fine for pricing a trace, but it cannot answer
//! why a deeper queue helps a disk, why it stops helping, or what happens
//! when a writer and a reader contend — because those are consequences of
//! mechanism, and the mechanism was fitted away.
//!
//! This module keeps the mechanism. A device here has a media transfer
//! rate, a bus ceiling, a number of requests it will accept, a number it
//! can work on at once, and — if it rotates — a head that has to travel
//! and a platter that has to come around. Everything the sweeps measure
//! is supposed to fall out of that.
//!
//! Two separations do most of the work:
//!
//! - **Accepting a request is not working on it.** `queue_slots` is how
//!   many commands the device will hold; `service_parallelism` is how many
//!   it can actually progress. A disk accepts 32 and works on one, because
//!   it has one head. Conflating the two is what forces a model to
//!   pretend a disk has an effective queue depth of 1.
//! - **Positioning is not transfer.** A request waits for the head and the
//!   platter without consuming any bandwidth, then transfers while
//!   consuming it. Only the second phase contends.
//!
//! Together those give the device a reason to reorder: with several
//! commands in hand it can serve whichever is cheapest to reach next.
//! That is NCQ, and here it is derived rather than assumed.

use rand::Rng;

/// Angular tolerance for "the platter is already where we need it".
/// Far above floating-point noise and far below one sector's worth of
/// arc, so it can only ever absorb rounding.
const ANGLE_EPSILON: f64 = 1e-9;

/// How a device pays to reach a location.
#[derive(Debug, Clone, Copy)]
pub enum Positioning {
    /// Flash. Reaching any address costs the same fixed command latency.
    Flat { access_latency_s: f64 },

    /// Rotating media. Reaching an address costs head travel plus waiting
    /// for the platter to bring the sector under the head.
    ///
    /// The track capacity is not a parameter: a track passes under the
    /// head once per rotation, so it holds exactly
    /// `media_rate × rotation_s` bytes. Deriving it keeps the geometry
    /// consistent with the transfer rate instead of letting the two drift.
    Rotational {
        /// Adjacent-track seek.
        track_to_track_s: f64,
        /// End-to-end seek.
        full_stroke_s: f64,
        /// One revolution.
        rotation_s: f64,
        /// Addressable span, for scaling seek distance.
        capacity_bytes: u64,
    },
}

impl Positioning {
    /// Time to bring `offset` under the head, given where the head is and
    /// what the clock reads.
    ///
    /// The clock matters for rotating media and only for rotating media:
    /// the platter's angular position is a function of time, so *when* a
    /// request is issued determines how long it waits. That is the whole
    /// basis of rotational position ordering, and a model without a clock
    /// term cannot express it.
    pub fn access_time_s(&self, head: u64, offset: u64, now: f64, media_rate: f64) -> f64 {
        match *self {
            Positioning::Flat { access_latency_s } => access_latency_s,

            Positioning::Rotational {
                track_to_track_s,
                full_stroke_s,
                rotation_s,
                capacity_bytes,
            } => {
                let distance = head.abs_diff(offset) as f64;
                let bytes_per_track = media_rate * rotation_s;

                // Staying on the current track costs nothing to reach:
                // the arm does not move. This is what makes a sequential
                // read run at the media rate instead of paying a seek per
                // block, and leaving it out costs a disk half its
                // streaming throughput.
                let seek = if distance < bytes_per_track {
                    0.0
                } else {
                    // Beyond a track, seek time grows with the square root
                    // of distance — the arm accelerates and decelerates
                    // rather than travelling at constant speed.
                    let fraction = (distance / capacity_bytes.max(1) as f64).clamp(0.0, 1.0);
                    track_to_track_s + (full_stroke_s - track_to_track_s) * fraction.sqrt()
                };

                let target_angle = (offset as f64 / bytes_per_track).fract();
                // Where the platter has turned to by the time the head
                // arrives.
                let arrival_angle = ((now + seek) / rotation_s).fract();
                let mut wait = target_angle - arrival_angle;
                if wait < 0.0 {
                    wait += 1.0;
                }
                // A sector the head is already sitting on requires no
                // wait. In exact arithmetic that is `wait == 0`, but the
                // two angles are computed by different routes and the
                // difference lands a hair either side of zero — and on
                // the wrong side it becomes a *full rotation*. Reading
                // contiguously would then pay 8.3 ms per block for a
                // platter that was already in position.
                if wait > 1.0 - ANGLE_EPSILON {
                    wait = 0.0;
                }
                seek + wait * rotation_s
            }
        }
    }

    /// Whether reordering can save anything. Flash has no geometry to
    /// exploit, so a device-side scheduler is pointless there.
    pub fn is_positional(&self) -> bool {
        matches!(self, Positioning::Rotational { .. })
    }
}

/// How the device chooses which accepted command to work on next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServicePolicy {
    /// Serve in arrival order. Correct for flash, where nothing is
    /// cheaper to reach than anything else.
    Fifo,
    /// Serve whichever accepted command is cheapest to reach right now.
    /// This is native command queueing: it is why a disk's throughput
    /// rises with queue depth even though it still has one head.
    NearestFirst,
}

/// How per-request cost changes with how many requests are in flight.
///
/// The [MQSSD model](https://arxiv.org/abs/2507.06349) finds per-request
/// setup cost falling steeply with concurrency — on a Samsung 990 PRO,
/// its fitted read setup term drops by roughly 700× between one
/// outstanding request and 128 — because the flash translation layer
/// pipelines address translation across whatever work it has in hand.
///
/// Most of that effect is already explicit here as parallel dies, so
/// this captures only the residual: the per-request cost a device pays
/// when it has nothing to overlap against. It falls from `solo_penalty`
/// at one request in flight toward 1 as the device fills up, on a
/// rational curve of the same family MQSSD fits.
///
/// **It is [`NONE`](ConcurrencyScaling::NONE) for every historical
/// device here**, because the perfscripts corpus sweeps block size at a
/// fixed `iodepth=10` and contains no queue-depth sweep to fit against.
/// Asserting a curve through a single measured point would be invention.
#[derive(Debug, Clone, Copy)]
pub struct ConcurrencyScaling {
    /// Per-request cost multiplier with a single request in flight.
    pub solo_penalty: f64,
    /// In-flight count at which half the benefit has been realised.
    pub half_depth: f64,
}

impl ConcurrencyScaling {
    /// Concurrency does not change per-request cost — the honest setting
    /// when no queue-depth sweep was measured.
    pub const NONE: Self = ConcurrencyScaling {
        solo_penalty: 1.0,
        half_depth: 1.0,
    };

    pub fn factor(&self, in_flight: usize) -> f64 {
        let k = in_flight.max(1) as f64;
        1.0 + (self.solo_penalty - 1.0) * self.half_depth / (self.half_depth + (k - 1.0))
    }
}

/// How a single read's service time varies.
///
/// A flash read is not one fixed cost. Within a wordline the pages
/// programmed at different bit significances read at different speeds —
/// on MLC the upper page takes markedly longer than the lower — and a
/// small share of reads need a retry at a shifted reference voltage.
/// Neither is unusual or a defect; both are how the medium works.
///
/// SSD simulators represent this the same way: MQSim exposes
/// `Page_Read_Latency_LSB/CSB/MSB` as separate parameters, and
/// SimpleSSD carries per-page-type timings too.
///
/// The values here are **read off published NAND characterisation**
/// rather than fitted. Both measured drives use Samsung MLC V-NAND, and
/// for MLC there are two page types: an LSB page needs one sensing pass
/// at roughly **40 µs**, an MSB page needs two at roughly **70 µs**
/// ([Device-Level Optimization Techniques for
/// SSDs](https://arxiv.org/abs/2507.10573), which puts MLC read at
/// 40–110 µs overall). An even split gives a mean of 55 µs — against
/// the 57 µs this model had already fitted for the 950 PRO from
/// throughput alone, which is a useful independent check on both.
///
/// The retry share is the one term still fitted. Read-retry at shifted
/// reference voltages is real and documented
/// ([Park et al.](https://arxiv.org/pdf/2104.09611)), but its *rate* on
/// a specific drive at a specific wear level is not something the
/// published characterisation pins down.
#[derive(Debug, Clone, Copy)]
pub struct ReadVariation {
    /// Share of reads and their multiplier on the base access latency,
    /// as page types within a wordline.
    pub page_types: &'static [(f64, f64)],
    /// Share of reads needing a retry.
    pub retry_share: f64,
    /// What a retry costs, as a multiplier.
    pub retry_multiplier: f64,
}

impl ReadVariation {
    /// Every read costs the same — right for rotating media, whose
    /// variation is positional and modelled elsewhere.
    pub const NONE: Self = ReadVariation {
        page_types: &[(1.0, 1.0)],
        retry_share: 0.0,
        retry_multiplier: 1.0,
    };

    /// Draw a multiplier for one read.
    ///
    /// **Mean-preserving.** The draw is normalised so its expectation is
    /// exactly 1, which means adding variation changes the shape of the
    /// latency distribution without moving the mean — and therefore
    /// without disturbing the throughput fit, which was calibrated
    /// before any of this existed and should not be quietly re-tuned by
    /// a distribution parameter.
    pub fn draw(&self, u: f64, v: f64) -> f64 {
        self.draw_raw(u, v) / self.mean_multiplier().max(1e-12)
    }

    fn draw_raw(&self, u: f64, v: f64) -> f64 {
        let mut acc = 0.0;
        let mut base = 1.0;
        for &(share, multiplier) in self.page_types {
            acc += share;
            if u <= acc {
                base = multiplier;
                break;
            }
        }
        if v < self.retry_share {
            base *= self.retry_multiplier;
        }
        base
    }

    /// Mean multiplier, for checking the fit does not shift the mean.
    pub fn mean_multiplier(&self) -> f64 {
        let base: f64 = self.page_types.iter().map(|(s, m)| s * m).sum();
        base * (1.0 - self.retry_share) + base * self.retry_multiplier * self.retry_share
    }
}

/// A device, specified physically.
#[derive(Debug, Clone, Copy)]
pub struct Hardware {
    pub name: &'static str,
    /// What the medium itself sustains, bytes per second.
    pub media_rate: f64,
    /// What the interconnect sustains, bytes per second. The lower of
    /// this and `media_rate × dies` is the real ceiling.
    pub bus_rate: f64,
    /// Aggregate write bandwidth, which on flash is well below the read
    /// ceiling.
    ///
    /// Getting sequential write throughput right by lowering the per-die
    /// program rate instead is a trap: it reproduces the number, but only
    /// by having an implausible number of programs in flight at once,
    /// and those hold dies that a concurrent reader needs. The model then
    /// starves the reader an order of magnitude harder than the mixed
    /// measurements show. Separating the two lets each be set from what
    /// actually determines it — a write-path ceiling from measured
    /// sequential write, and a per-die program rate from NAND page
    /// timings.
    pub write_bandwidth: f64,
    /// Commands the device will accept and hold.
    pub queue_slots: usize,
    /// Independent units that can each work on one command at a time.
    ///
    /// A spinning disk has one, because it has one head. Flash has many,
    /// and **which one serves a request is decided by its address, not by
    /// which happens to be free** — see [`Hardware::die_of`]. That
    /// distinction is the whole of read/write interference: a read whose
    /// die is mid-program waits for the program to finish, however idle
    /// the rest of the device is.
    pub dies: usize,
    /// Address interleave across dies.
    pub die_stripe_bytes: u64,
    /// Bytes per second one die can program.
    ///
    /// Programming is the write path's real cost and it is *rate*, not a
    /// fixed delay: a die programming 128 KiB is busy eight times as long
    /// as one programming 16 KiB. Modelling it as a flat per-request
    /// constant makes large writes far too cheap — it overstated
    /// sequential write throughput by 57% on NVMe until this replaced it.
    ///
    /// Per-die program bandwidth is roughly an order of magnitude below
    /// read bandwidth, which is why a writer can lock a reader out of a
    /// die for so long. The values here are derived from each device's
    /// measured sequential-write throughput divided by the dies a
    /// streaming writer keeps engaged.
    pub program_rate_per_die: f64,
    /// How many dies one write programs at once.
    ///
    /// Programming is pipelined: the controller feeds pages to dies in
    /// sequence rather than firing every die a request spans
    /// simultaneously, and NAND program-suspend lets reads slip in
    /// between pages. This bounds how many dies a single write can hold
    /// against a concurrent reader — the parameter that decides how hard
    /// a writer starves one.
    pub program_die_concurrency: usize,
    /// Residual per-request cost dependence on concurrency.
    pub concurrency: ConcurrencyScaling,
    /// Per-read service-time variation.
    pub read_variation: ReadVariation,
    /// How unevenly concurrent transfers share the device's bandwidth,
    /// as the half-width of a uniform weight around 1.
    ///
    /// Perfect processor-sharing — every in-flight request progressing at
    /// exactly the same rate — is a convenient assumption and a wrong
    /// one. A real controller interleaves channels unevenly, so some
    /// requests finish ahead of others that started with them. The
    /// aggregate is unaffected, because the weights are normalised; what
    /// changes is the spread, and at block sizes where transfer dominates
    /// the access latency this is the *only* thing that can produce any
    /// spread at all. Without it a model reports a suspiciously tight
    /// distribution for large reads: median too high, tails too light.
    /// Set modestly rather than optimally: raising it monotonically
    /// trades scatter for bias — at 0 the 99th percentile carries a
    /// systematic −7.7% understatement, at 0.45 that falls to −1.3% but
    /// the mean absolute error rises from 9.5% to 14.2%. There is no
    /// optimum, only a judgement, and the value here is not strongly
    /// determined by anything.
    pub transfer_share_spread: f64,
    /// How well the device can predict rotational position when choosing
    /// what to serve next, from 0 (seek distance only) to 1 (perfect
    /// rotational position ordering).
    ///
    /// This is the parameter that separates *how much* a device reorders
    /// from *how well*. Fitting a disk with only a window size forces a
    /// choice between the two: a narrow window matches throughput and
    /// produces far too tight a latency distribution, while a wide one
    /// matches the measured tail and overshoots throughput by a fifth.
    /// Real firmware commits to requests some way ahead and knows the
    /// platter's phase only approximately, so it reorders widely and
    /// chooses imperfectly — which widens the distribution without
    /// delivering the throughput perfect selection would.
    pub rotational_awareness: f64,
    /// How long the device will defer a command before serving it
    /// regardless of cost. Bounds the starvation aggressive reordering
    /// otherwise causes.
    pub command_expiry_s: f64,
    /// Volatile write buffer. A write lands here and is acknowledged
    /// immediately; the medium catches up afterwards.
    ///
    /// **Not validated.** The perfscripts corpus has no random-write
    /// workload, so everything about the write path here is asserted from
    /// device datasheets and general knowledge rather than checked
    /// against measurement. Treat write predictions as structurally
    /// reasonable and numerically unverified.
    pub write_buffer_bytes: u64,
    /// Extra medium writes per logical write once garbage collection is
    /// running — write amplification.
    ///
    /// **Not validated**, for the same reason. A drive at steady state
    /// under sustained writes rewrites live data to reclaim blocks, and
    /// that work competes for the same dies. A value of 1.0 models a
    /// fresh or lightly-used drive, which is what a benchmark measures
    /// and not what a long-running import experiences.
    pub write_amplification: f64,
    /// Commands per second the controller can *start*, whatever their
    /// size and however many channels are idle. This is a serial
    /// resource, so it is what flattens the small-block end of every
    /// flash curve: below a few KiB, throughput stops depending on size
    /// entirely because the limit is command processing, not bytes.
    pub max_command_rate: f64,
    /// How many accepted commands the device actually considers when
    /// choosing what to serve next.
    ///
    /// Only meaningful with [`ServicePolicy::NearestFirst`]. Firmware
    /// does not evaluate the whole queue against exact rotational
    /// positions — it commits some way ahead — so perfect best-of-queue
    /// selection overstates what reordering achieves. This bounds it.
    pub reorder_window: usize,
    pub positioning: Positioning,
    pub policy: ServicePolicy,
}

impl Hardware {
    /// The bandwidth ceiling actually in force.
    pub fn peak_bandwidth(&self) -> f64 {
        self.bus_rate.min(self.media_rate * self.dies as f64)
    }

    /// Which unit holds the stripe containing this address.
    /// Address-determined, not load-balanced: that is what makes a busy
    /// die block a request that lands on it.
    pub fn die_of(&self, offset: u64) -> usize {
        if self.dies <= 1 {
            return 0;
        }
        ((offset / self.die_stripe_bytes.max(1)) % self.dies as u64) as usize
    }

    /// How many dies a request of this size spans.
    ///
    /// A request larger than the stripe is served by several dies at
    /// once — that striping is *how* a device turns request size into
    /// bandwidth, and a model that pins a whole request to one die caps
    /// large reads at a single die's rate. It also aliases badly:
    /// 1 MiB-aligned offsets over a 4 KiB stripe and 32 dies all land on
    /// die zero.
    pub fn dies_spanned(&self, len: u64) -> usize {
        if self.dies <= 1 {
            return 1;
        }
        let stripes = len.div_ceil(self.die_stripe_bytes.max(1)) as usize;
        stripes.clamp(1, self.dies)
    }

    /// Whether every die a request needs is free.
    pub fn dies_free(&self, offset: u64, len: u64, busy: &[bool]) -> bool {
        let first = self.die_of(offset);
        (0..self.dies_spanned(len)).all(|i| !busy[(first + i) % self.dies.max(1)])
    }

    /// The rate a single request can absorb, given how many dies it
    /// spans. Parallelism across dies is what lets a large request beat
    /// one die's transfer rate.
    pub fn request_rate(&self, len: u64) -> f64 {
        self.media_rate * self.dies_spanned(len) as f64
    }

    /// Extra die occupancy a write incurs beyond its transfer.
    ///
    /// Proportional to the bytes each die has to program, so a large
    /// write occupies its dies for correspondingly longer.
    ///
    /// Garbage collection is folded in as a multiplier: at steady state a
    /// logical write costs several medium writes, and they occupy the
    /// same dies. Setting [`Self::write_amplification`] above 1 is how a
    /// drive that has been written to for a long time differs from the
    /// fresh one a benchmark measures.
    pub fn write_occupancy_s(&self, write: bool, len: u64) -> f64 {
        if !write || self.program_rate_per_die <= 0.0 {
            return 0.0;
        }
        let per_die = len as f64 / self.dies_spanned(len) as f64;
        per_die / self.program_rate_per_die * self.write_amplification.max(1.0)
    }

    /// Sequential throughput: no seeking, no rotational waiting, so the
    /// only limits are the medium and the bus.
    pub fn sequential_bandwidth(&self) -> f64 {
        self.peak_bandwidth()
    }

    pub fn access_time_s(&self, head: u64, offset: u64, now: f64) -> f64 {
        self.access_time_at_depth(head, offset, now, usize::MAX)
    }

    /// The same, with the residual concurrency effect applied.
    pub fn access_time_at_depth(&self, head: u64, offset: u64, now: f64, in_flight: usize) -> f64 {
        self.access_time_full(head, offset, now, in_flight, 1.0)
    }

    /// What the device *believes* a request will cost when choosing what
    /// to serve next, as distinct from what it will actually cost.
    ///
    /// Selection uses this; service uses [`Self::access_time_full`]. The
    /// gap between them is the firmware's imperfect knowledge, and it is
    /// why aggressive reordering does not deliver proportional
    /// throughput.
    pub fn selection_cost_s(&self, head: u64, offset: u64, now: f64) -> f64 {
        match self.positioning {
            Positioning::Flat { .. } => 0.0,
            Positioning::Rotational { rotation_s, .. } => {
                let truth = self
                    .positioning
                    .access_time_s(head, offset, now, self.media_rate);
                if self.rotational_awareness >= 1.0 {
                    return truth;
                }
                // The phase-independent estimate: seek plus the average
                // wait, which is what firmware knows when it cannot
                // resolve where the platter actually is.
                let blind = self.mean_access_over_a_revolution(head, offset, now);
                let _ = rotation_s;
                truth * self.rotational_awareness + blind * (1.0 - self.rotational_awareness)
            }
        }
    }

    fn mean_access_over_a_revolution(&self, head: u64, offset: u64, now: f64) -> f64 {
        match self.positioning {
            Positioning::Flat { .. } => 0.0,
            Positioning::Rotational { rotation_s, .. } => {
                let samples = 8;
                let total: f64 = (0..samples)
                    .map(|i| {
                        self.positioning.access_time_s(
                            head,
                            offset,
                            now + i as f64 * rotation_s / samples as f64,
                            self.media_rate,
                        )
                    })
                    .sum();
                total / samples as f64
            }
        }
    }

    /// The same, with a drawn per-read variation multiplier applied.
    pub fn access_time_full(
        &self,
        head: u64,
        offset: u64,
        now: f64,
        in_flight: usize,
        variation: f64,
    ) -> f64 {
        self.positioning
            .access_time_s(head, offset, now, self.media_rate)
            * self.concurrency.factor(in_flight)
            * variation
    }

    /// Seconds of the controller's serial attention each command needs.
    pub fn command_time_s(&self) -> f64 {
        if self.max_command_rate <= 0.0 {
            0.0
        } else {
            1.0 / self.max_command_rate
        }
    }
}

/// 7200 RPM SATA disk.
///
/// Seek and rotation figures are the class-typical values for a 3.5"
/// 7200 RPM drive; the transfer rate is the measured sequential read.
/// One head means `service_parallelism = 1`, and the 32 queue slots are
/// SATA's NCQ depth — the gap between those two numbers is where all of
/// this device's queue-depth behaviour comes from.
pub const SPINNING_SATA_HW: Hardware = Hardware {
    name: "spinning-sata",
    media_rate: 201.0e6,
    bus_rate: 600.0e6,
    // None: a disk has no separate write path, and giving it one is
    // actively harmful. A write ceiling even slightly below the media
    // rate desynchronises sequential writes from the platter — each
    // block finishes a hair after its successor's sector has passed, so
    // the head waits almost a full revolution — and halves the modelled
    // sequential write throughput.
    write_bandwidth: f64::INFINITY,
    queue_slots: 32,
    dies: 1,
    die_stripe_bytes: 1 << 20,
    // Zero, and deliberately: a disk has no program phase. The head
    // writes the data as it passes under it, so the transfer *is* the
    // write. Charging a separate programming cost on top halves the
    // modelled sequential write throughput.
    program_rate_per_die: 0.0,
    program_die_concurrency: 1,
    concurrency: ConcurrencyScaling::NONE,
    // A disk's read variation is positional — seek distance and
    // rotational phase — and is generated by the geometry model rather
    // than drawn.
    read_variation: ReadVariation::NONE,
    // A disk transfers one request at a time, so there is no sharing to
    // be uneven about.
    transfer_share_spread: 0.0,
    // Fitted against the measured latency *distribution*, not just its
    // mean: 0.12 reproduces throughput to 2%, the median to 6% and the
    // 99th percentile to 5%. Perfect selection at the same window
    // overshoots throughput by a fifth and understates the tail by half,
    // which is what a single-knob model is forced into.
    rotational_awareness: 0.12,
    // Bounded deferral. Without it, sweeping with a 32-deep window drives
    // a competing random reader to zero against a sequential stream —
    // which the mixed-workload measurements plainly do not show.
    //
    // 600 ms was chosen by fitting the contended sweep, and then found to
    // agree with the longest completion latency fio recorded on this
    // drive (607.7 ms). Two independent routes to the same number is
    // better evidence than either alone.
    command_expiry_s: 600.0e-3,
    // A disk's track buffer is small and does not change its economics.
    write_buffer_bytes: 64 << 20,
    write_amplification: 1.0,
    max_command_rate: 100_000.0,
    // Wide, because a drive with 32 NCQ slots really does consider all of
    // them; the throughput this would otherwise imply is held back by
    // `rotational_awareness` rather than by pretending the queue is short.
    reorder_window: 32,
    positioning: Positioning::Rotational {
        track_to_track_s: 1.5e-3,
        full_stroke_s: 17.0e-3,
        rotation_s: 60.0 / 7200.0,
        capacity_bytes: 1_000_204_886_016,
    },
    policy: ServicePolicy::NearestFirst,
};

/// SATA SSD. No geometry, so no reordering benefit; the SATA link is the
/// ceiling and the controller's per-command cost is the small-block
/// limit.
pub const SATA_SSD_HW: Hardware = Hardware {
    name: "sata-ssd",
    media_rate: 75.0e6,
    bus_rate: 568.0e6,
    // Calibrated to reproduce the measured 538 MB/s sequential write.
    // Slightly above that figure because the model keeps marginally
    // fewer requests in the transfer phase than the offered depth.
    write_bandwidth: 600.0e6,
    queue_slots: 32,
    // 8 channels of 4 dies is typical for a drive of this generation.
    // 8 channels of 8 dies. The count matters even when requests are far
    // smaller than the device: at ten outstanding random requests over 32
    // dies, birthday collisions cost about 13% of the available
    // concurrency, and the measured curve does not show that loss.
    dies: 64,
    // Coarse: the measured curve shows a single request served at a
    // roughly constant ~75 MB/s from 4 KiB to 64 KiB, so this controller
    // does not split one command across dies until it is large.
    die_stripe_bytes: 65_536,
    // A 16 KiB page in roughly 700 µs, which is ordinary MLC.
    program_rate_per_die: 23.0e6,
    program_die_concurrency: 2,
    concurrency: ConcurrencyScaling::NONE,
    // MLC across its published 40–110 µs read band: LSB pages sense once
    // (~40 µs), MSB pages twice (~70 µs), and the upper end of the band
    // covers pages needing more. A strict two-point split reproduces the
    // tail but leaves the median unstable — a bimodal distribution has no
    // well-defined middle — so the band is sampled rather than its
    // endpoints.
    read_variation: ReadVariation {
        page_types: &[(0.42, 40.0), (0.42, 70.0), (0.16, 110.0)],
        retry_share: 0.015,
        retry_multiplier: 2.4,
    },
    transfer_share_spread: 0.20,
    rotational_awareness: 1.0,
    command_expiry_s: f64::INFINITY,
    write_buffer_bytes: 512 << 20,
    write_amplification: 1.0,
    max_command_rate: 80_000.0,
    reorder_window: 1,
    positioning: Positioning::Flat {
        access_latency_s: 68.0e-6,
    },
    policy: ServicePolicy::Fifo,
};

/// Consumer NVMe. Deeper queues and more parallel channels; the command
/// overhead is small enough that bandwidth binds first at almost any
/// useful block size.
pub const NVME_CONSUMER_HW: Hardware = Hardware {
    name: "nvme-consumer",
    media_rate: 190.0e6,
    bus_rate: 1_500.0e6,
    // Measured sequential write: 933819 KB/s, well under the read path.
    write_bandwidth: 956.0e6,
    queue_slots: 256,
    // The 950 PRO's UBX controller addresses eight channels with eight-way
    // interleaving — 64 units of addressable parallelism, not the 128
    // this model previously assumed. That is the controller's reach
    // rather than the physical die count of any one capacity point; what
    // the model needs is how many independent things can be in flight.
    dies: 64,
    // Same reasoning as the SATA drive: the measured per-request rate is
    // flat at ~185 MB/s out to 128 KiB.
    die_stripe_bytes: 131_072,
    // A 16 KiB page in roughly 700 µs, which is ordinary MLC.
    program_rate_per_die: 23.0e6,
    program_die_concurrency: 2,
    concurrency: ConcurrencyScaling::NONE,
    // MLC across its published 40–110 µs read band: LSB pages sense once
    // (~40 µs), MSB pages twice (~70 µs), and the upper end of the band
    // covers pages needing more. A strict two-point split reproduces the
    // tail but leaves the median unstable — a bimodal distribution has no
    // well-defined middle — so the band is sampled rather than its
    // endpoints.
    read_variation: ReadVariation {
        page_types: &[(0.42, 40.0), (0.42, 70.0), (0.16, 110.0)],
        retry_share: 0.015,
        retry_multiplier: 2.4,
    },
    transfer_share_spread: 0.20,
    rotational_awareness: 1.0,
    command_expiry_s: f64::INFINITY,
    write_buffer_bytes: 512 << 20,
    write_amplification: 1.0,
    max_command_rate: 124_000.0,
    reorder_window: 1,
    positioning: Positioning::Flat {
        access_latency_s: 57.0e-6,
    },
    policy: ServicePolicy::Fifo,
};

/// A current-generation NVMe drive, calibrated to published figures
/// rather than to a sweep run here.
///
/// The historical devices in this module come from the perfscripts fio
/// corpus, which was captured in 2016. Keeping them is right — they are
/// real regimes — but treating a 2016 consumer drive as "NVMe" understates
/// current hardware by roughly an order of magnitude in operation rate,
/// and conclusions drawn about ordering are sensitive to exactly that.
///
/// Calibration targets, all published:
///
/// - **~1M random-read IOPS at 4 KiB and 7.0 GB/s sequential**, from the
///   Samsung 980 PRO in Table 1 of
///   [Ren et al., ICPE '24](https://dl.acm.org/doi/10.1145/3629526.3645053),
///   whose eight-device testbed reached 5.9M 4 KiB IOPS in aggregate.
/// - **68 µs mean read latency**, from the same table.
/// - **A random-to-sequential read ratio of 1.3–1.5× at high
///   concurrency**, from the [MQSSD model's](https://arxiv.org/abs/2507.06349)
///   Samsung 990 PRO measurements. The command rate here is set to land
///   inside that band, which is the one parameter fitted to an outcome
///   rather than read off a specification.
///
/// The IOPS peak is reached at high *total* concurrency, not at 32
/// outstanding requests: vendor "QD32" figures are quoted per thread
/// across many threads, and the ICPE '24 aggregate needed many cores.
pub const NVME_MODERN_HW: Hardware = Hardware {
    name: "nvme-modern",
    media_rate: 400.0e6,
    // PCIe 4.0 x4, practical.
    bus_rate: 7_000.0e6,
    // **Not measured** — a current drive's specified sequential write.
    write_bandwidth: 5_000.0e6,
    queue_slots: 1_024,
    dies: 128,
    // An effective interleave, not a physical page stripe: chosen so
    // that both published anchors are reproduced at once. Too fine and a
    // 1 MiB request holds so many dies that only two fit and the access
    // latency stops being hidden, costing 20% of sequential throughput;
    // too coarse and large requests cannot reach the bus ceiling.
    die_stripe_bytes: 65_536,
    // **Not calibrated** — no write measurement exists for a device of
    // this class here. Scaled from the 2016 drive in proportion to its
    // read advantage, which is an assumption, not a finding.
    program_rate_per_die: 30.0e6,
    program_die_concurrency: 2,
    // TLC: three page types needing 2, 3 and 2 sensing passes, over a
    // published 66–170 µs read band.
    read_variation: ReadVariation {
        page_types: &[(0.34, 70.0), (0.33, 110.0), (0.33, 90.0)],
        retry_share: 0.02,
        retry_multiplier: 2.6,
    },
    transfer_share_spread: 0.20,
    rotational_awareness: 1.0,
    command_expiry_s: f64::INFINITY,
    write_buffer_bytes: 512 << 20,
    write_amplification: 1.0,
    // The only device here with a fitted concurrency curve, because it is
    // the only one for which published queue-depth behaviour exists.
    concurrency: ConcurrencyScaling {
        solo_penalty: 1.6,
        half_depth: 8.0,
    },
    max_command_rate: 1_200_000.0,
    reorder_window: 1,
    positioning: Positioning::Flat {
        access_latency_s: 68.0e-6,
    },
    policy: ServicePolicy::Fifo,
};

/// The three devices measured in the perfscripts corpus. These have
/// matching [`crate::regime::Regime`] entries and are what the
/// forward-simulation fit is checked against.
pub const HISTORICAL_HARDWARE: &[Hardware] = &[SPINNING_SATA_HW, SATA_SSD_HW, NVME_CONSUMER_HW];

pub const ALL_HARDWARE: &[Hardware] = HISTORICAL_HARDWARE;

/// Every device including the modern one, for studies that are not being
/// checked against the 2016 sweeps.
pub const ALL_HARDWARE_WITH_MODERN: &[Hardware] = &[
    SPINNING_SATA_HW,
    SATA_SSD_HW,
    NVME_CONSUMER_HW,
    NVME_MODERN_HW,
];

/// The cost of issuing an I/O on the host, which is not free and at
/// modern device rates is frequently the binding constraint.
///
/// [Ren et al., ICPE '24](https://dl.acm.org/doi/10.1145/3629526.3645053)
/// find that with high-performance NVMe SSDs "the CPU is the primary
/// bottleneck ... yet the SSD device itself is not saturated" at 4 KiB,
/// and that Linux I/O schedulers can add up to 63.4% overhead on top.
/// A storage model with no host-side cost is modelling the wrong
/// bottleneck above roughly half a million operations per second.
#[derive(Debug, Clone, Copy)]
pub struct HostModel {
    /// Serial CPU time to submit and complete one request, per core.
    pub per_request_s: f64,
    /// Cores issuing I/O.
    pub cores: usize,
    /// Memory bandwidth available to this host, bytes per second.
    ///
    /// A storage model without this term is assuming memory is free, and
    /// it is not: the cost model in `docs/sysref/splat/cost-model.md`
    /// already carries a `N·R / BW_mem` term for the assemble memcpy that
    /// the simulator did not implement.
    pub memory_bandwidth: f64,
    /// How many times each transferred byte crosses the memory bus.
    ///
    /// A byte arriving from storage is DMA-written to memory, read back
    /// to be copied into a user buffer, written again, and read once more
    /// when it is scattered into an output segment. Three to four touches
    /// per byte is ordinary, and it means a 7 GB/s drive can generate
    /// 25 GB/s of memory traffic — the whole budget of a DDR4 channel.
    pub memory_touches_per_byte: f64,
}

impl HostModel {
    /// What one core can push, measured, by storage API.
    ///
    /// [Didona et al., SYSTOR '22](https://atlarge-research.com/pdfs/2022-systor-apis.pdf)
    /// measure single-core peak throughput for each of the Linux storage
    /// APIs on Intel DC P3600 NVMe drives under kernel 5.13:
    ///
    /// | API | peak KIOPS on one core | implied cost per request |
    /// |---|---|---|
    /// | libaio | 144.9 | 6.90 µs |
    /// | io_uring | 171.5 | 5.83 µs |
    /// | io_uring, polled | 173.0 | 5.78 µs |
    /// | SPDK | 305.9 | 3.27 µs |
    ///
    /// [Ren et al., ICPE '24](https://dl.acm.org/doi/10.1145/3629526.3645053)
    /// reach 5.9M IOPS across ten cores with io_uring on kernel 6.3.8 —
    /// about 590 KIOPS per core, three times the SYSTOR figure two kernel
    /// generations earlier. Both are real; the API and the kernel matter
    /// more than the hardware here, which is why this is a set of named
    /// presets rather than one number.
    ///
    /// One measured configuration is worth avoiding rather than
    /// modelling: `io_uring` with SQPOLL sharing a single core with the
    /// issuer manages **13.7 KIOPS**, because the poller thread takes
    /// half the cycles.
    pub const LIBAIO: Self = HostModel {
        per_request_s: 6.90e-6,
        cores: 1,
        memory_bandwidth: 40.0e9,
        memory_touches_per_byte: 3.0,
    };

    /// io_uring on a 5.13-era kernel.
    pub const IO_URING: Self = HostModel {
        per_request_s: 5.83e-6,
        ..Self::LIBAIO
    };

    /// Kernel bypass.
    pub const SPDK: Self = HostModel {
        per_request_s: 3.27e-6,
        ..Self::LIBAIO
    };

    /// io_uring on a 6.3-era kernel, implied by the ICPE '24 aggregate.
    /// This is the default because it is the most recent measurement,
    /// and it is the optimistic end of the range.
    pub const DEFAULT: Self = HostModel {
        per_request_s: 1.7e-6,
        cores: 1,
        // One DDR4-3200 channel pair, which is what a modest server gives
        // a single socket.
        memory_bandwidth: 40.0e9,
        memory_touches_per_byte: 3.0,
    };

    /// No host cost at all, for isolating device behaviour.
    pub const FREE: Self = HostModel {
        per_request_s: 0.0,
        cores: 1,
        memory_bandwidth: f64::INFINITY,
        memory_touches_per_byte: 0.0,
    };

    pub fn cores(n: usize) -> Self {
        HostModel {
            cores: n.max(1),
            ..Self::DEFAULT
        }
    }

    /// Bytes per second of storage traffic this host's memory can carry,
    /// given every byte is touched several times.
    pub fn io_bandwidth_ceiling(&self) -> f64 {
        if self.memory_touches_per_byte <= 0.0 {
            f64::INFINITY
        } else {
            self.memory_bandwidth / self.memory_touches_per_byte
        }
    }

    /// Seconds of memory time `bytes` of I/O traffic occupies.
    pub fn memory_time_s(&self, bytes: u64) -> f64 {
        if !self.memory_bandwidth.is_finite() || self.memory_bandwidth <= 0.0 {
            0.0
        } else {
            bytes as f64 * self.memory_touches_per_byte / self.memory_bandwidth
        }
    }

    /// Operations per second this host can sustain, ignoring the device.
    pub fn ceiling_iops(&self) -> f64 {
        if self.per_request_s <= 0.0 {
            f64::INFINITY
        } else {
            self.cores as f64 / self.per_request_s
        }
    }
}

/// The link a device reaches the host through, and what else is on it.
///
/// A device's own `bus_rate` is its link. It is not the whole story:
/// several devices commonly share one upstream port, and the aggregate
/// is what binds. Eight drives that each claim 7 GB/s behind a single
/// PCIe 4.0 x16 root port get about 3.5 GB/s apiece, and a per-device
/// ceiling cannot express that.
#[derive(Debug, Clone, Copy)]
pub struct Fabric {
    /// Usable bandwidth of the shared upstream link, bytes per second.
    pub link_bytes_per_s: f64,
    /// Devices contending for it.
    pub devices: usize,
}

impl Fabric {
    /// A device with the link to itself.
    pub const DEDICATED: Self = Fabric {
        link_bytes_per_s: f64::INFINITY,
        devices: 1,
    };

    /// PCIe 4.0 x16, practical, shared by `devices`.
    pub fn pcie4_x16(devices: usize) -> Self {
        Fabric {
            link_bytes_per_s: 28.0e9,
            devices: devices.max(1),
        }
    }

    /// PCIe 4.0 x4, practical, shared by `devices`.
    pub fn pcie4_x4(devices: usize) -> Self {
        Fabric {
            link_bytes_per_s: 7.0e9,
            devices: devices.max(1),
        }
    }

    /// This device's share of the link.
    pub fn share(&self) -> f64 {
        self.link_bytes_per_s / self.devices.max(1) as f64
    }
}

/// Where the issuing thread sits relative to the device.
///
/// A drive hangs off one socket's root complex. A thread on another
/// socket reaches it across the interconnect: completions are routed
/// further, and DMA lands in memory that is remote to whichever end
/// wants it next. Neither effect is large per request; the bandwidth one
/// is large in aggregate, because an inter-socket link carries a small
/// fraction of what local memory does.
///
/// The measurements this crate calibrates against were taken on a single
/// socket, so `LOCAL` is what they describe and anything else is
/// extrapolation.
#[derive(Debug, Clone, Copy)]
pub struct Numa {
    /// Whether the issuer shares a node with the device.
    pub local: bool,
    /// Extra per-request cost when it does not — completion routing and
    /// remote DMA setup.
    pub remote_latency_s: f64,
    /// Share of local memory bandwidth available across the interconnect.
    pub remote_bandwidth_fraction: f64,
}

impl Numa {
    /// Issuer and device on the same node.
    pub const LOCAL: Self = Numa {
        local: true,
        remote_latency_s: 0.0,
        remote_bandwidth_fraction: 1.0,
    };

    /// Issuer on a different socket from the device.
    pub const REMOTE: Self = Numa {
        local: false,
        // Completion interrupt and its cache traffic crossing the link.
        remote_latency_s: 2.0e-6,
        // An inter-socket link carries well under what local memory does.
        remote_bandwidth_fraction: 0.55,
    };

    pub fn latency_penalty_s(&self) -> f64 {
        if self.local {
            0.0
        } else {
            self.remote_latency_s
        }
    }

    pub fn bandwidth_factor(&self) -> f64 {
        if self.local {
            1.0
        } else {
            self.remote_bandwidth_fraction
        }
    }
}

/// Draw a rotational offset so that a simulation does not start every
/// run with the platter in the same place.
pub fn random_start_angle(rng: &mut impl Rng, rotation_s: f64) -> f64 {
    rng.random::<f64>() * rotation_s
}

#[cfg(all(test, feature = "heavy-tests"))]
mod tests {
    use super::*;

    #[test]
    fn a_disk_charges_more_for_a_longer_seek() {
        let hw = SPINNING_SATA_HW;
        // Averaged over rotational position, so the seek term is what
        // differs.
        let near: f64 = (0..64)
            .map(|i| hw.access_time_s(0, 1 << 20, i as f64 * 1e-4))
            .sum::<f64>()
            / 64.0;
        let far: f64 = (0..64)
            .map(|i| hw.access_time_s(0, 900_000_000_000, i as f64 * 1e-4))
            .sum::<f64>()
            / 64.0;
        assert!(far > near * 3.0, "near {near:.6}s vs far {far:.6}s");
    }

    #[test]
    fn flash_charges_the_same_wherever_it_reaches() {
        let hw = NVME_CONSUMER_HW;
        let a = hw.access_time_s(0, 4_096, 0.0);
        let b = hw.access_time_s(0, 900_000_000_000, 0.31);
        assert_eq!(a, b);
    }

    /// Rotational latency has to actually vary with the clock, or
    /// rotational position ordering has nothing to order.
    #[test]
    fn rotational_wait_depends_on_when_you_ask() {
        let hw = SPINNING_SATA_HW;
        let rotation = 60.0 / 7200.0;
        let samples: Vec<f64> = (0..16)
            .map(|i| hw.access_time_s(1 << 30, 1 << 30, i as f64 * rotation / 16.0))
            .collect();
        let lo = samples.iter().cloned().fold(f64::MAX, f64::min);
        let hi = samples.iter().cloned().fold(0.0, f64::max);
        assert!(
            hi - lo > rotation * 0.8,
            "wait should span most of a rotation"
        );
    }

    #[test]
    fn track_capacity_follows_from_rate_and_rotation() {
        // 201 MB/s at 120 revolutions per second is about 1.7 MB a track.
        if let Positioning::Rotational { rotation_s, .. } = SPINNING_SATA_HW.positioning {
            let track = SPINNING_SATA_HW.media_rate * rotation_s;
            assert!(
                (1.5e6..2.0e6).contains(&track),
                "track holds {track:.0} bytes"
            );
        } else {
            panic!("expected rotational geometry");
        }
    }
}
