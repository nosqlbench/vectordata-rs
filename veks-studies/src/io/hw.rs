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

/// A device, specified physically.
#[derive(Debug, Clone, Copy)]
pub struct Hardware {
    pub name: &'static str,
    /// What the medium itself sustains, bytes per second.
    pub media_rate: f64,
    /// What the interconnect sustains, bytes per second. The lower of
    /// this and `media_rate × service_parallelism` is the real ceiling.
    pub bus_rate: f64,
    /// Commands the device will accept and hold.
    pub queue_slots: usize,
    /// Commands it can make progress on simultaneously.
    pub service_parallelism: usize,
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
        self.bus_rate
            .min(self.media_rate * self.service_parallelism as f64)
    }

    /// Sequential throughput: no seeking, no rotational waiting, so the
    /// only limits are the medium and the bus.
    pub fn sequential_bandwidth(&self) -> f64 {
        self.peak_bandwidth()
    }

    pub fn access_time_s(&self, head: u64, offset: u64, now: f64) -> f64 {
        self.positioning
            .access_time_s(head, offset, now, self.media_rate)
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
    service_parallelism: 1,
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
    service_parallelism: 8,
    max_command_rate: 80_000.0,
    reorder_window: 1,
    positioning: Positioning::Flat {
        access_latency_s: 51.0e-6,
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
    service_parallelism: 16,
    max_command_rate: 124_000.0,
    reorder_window: 1,
    positioning: Positioning::Flat {
        access_latency_s: 60.0e-6,
    },
    policy: ServicePolicy::Fifo,
};

pub const ALL_HARDWARE: &[Hardware] = &[SPINNING_SATA_HW, SATA_SSD_HW, NVME_CONSUMER_HW];

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
