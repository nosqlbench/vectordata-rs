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

/// A device, specified physically.
#[derive(Debug, Clone, Copy)]
pub struct Hardware {
    pub name: &'static str,
    /// What the medium itself sustains, bytes per second.
    pub media_rate: f64,
    /// What the interconnect sustains, bytes per second. The lower of
    /// this and `media_rate × dies` is the real ceiling.
    pub bus_rate: f64,
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
    /// How long a die is occupied programming a written page, on top of
    /// the transfer. Flash programming is roughly an order of magnitude
    /// slower than reading and cannot be interrupted, which is why a
    /// writer can lock a reader out of a die.
    pub program_time_s: f64,
    /// Residual per-request cost dependence on concurrency.
    pub concurrency: ConcurrencyScaling,
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
    pub fn write_occupancy_s(&self, write: bool) -> f64 {
        if write { self.program_time_s } else { 0.0 }
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
        self.positioning
            .access_time_s(head, offset, now, self.media_rate)
            * self.concurrency.factor(in_flight)
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
    queue_slots: 32,
    dies: 1,
    die_stripe_bytes: 1 << 20,
    program_time_s: 0.0,
    concurrency: ConcurrencyScaling::NONE,
    max_command_rate: 100_000.0,
    reorder_window: 4,
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
    // ~1.3 ms TLC program, an order of magnitude past the read latency.
    program_time_s: 1.3e-3,
    concurrency: ConcurrencyScaling::NONE,
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
    queue_slots: 256,
    dies: 128,
    // Same reasoning as the SATA drive: the measured per-request rate is
    // flat at ~185 MB/s out to 128 KiB.
    die_stripe_bytes: 131_072,
    program_time_s: 700.0e-6,
    concurrency: ConcurrencyScaling::NONE,
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
    queue_slots: 1_024,
    dies: 128,
    // An effective interleave, not a physical page stripe: chosen so
    // that both published anchors are reproduced at once. Too fine and a
    // 1 MiB request holds so many dies that only two fit and the access
    // latency stops being hidden, costing 20% of sequential throughput;
    // too coarse and large requests cannot reach the bus ceiling.
    die_stripe_bytes: 65_536,
    program_time_s: 350.0e-6,
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
}

impl HostModel {
    /// A single core at ~1.7 µs per request — about 590k IOPS, which is
    /// the order at which the ICPE '24 testbed found the CPU saturating
    /// before the device did.
    pub const DEFAULT: Self = HostModel {
        per_request_s: 1.7e-6,
        cores: 1,
    };

    /// No host cost, for isolating device behaviour.
    pub const FREE: Self = HostModel {
        per_request_s: 0.0,
        cores: 1,
    };

    pub fn cores(n: usize) -> Self {
        HostModel {
            per_request_s: Self::DEFAULT.per_request_s,
            cores: n.max(1),
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

/// Draw a rotational offset so that a simulation does not start every
/// run with the platter in the same place.
pub fn random_start_angle(rng: &mut impl Rng, rotation_s: f64) -> f64 {
    rng.random::<f64>() * rotation_s
}

#[cfg(test)]
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
