// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Named cost regimes, measured rather than assumed.
//!
//! The cost model's device constants used to be asserted from memory.
//! These are not: every number below is transcribed from an fio run in
//! the [perfscripts](https://github.com/jshook/perfscripts) historic
//! result set, which sweeps random reads across block sizes from 512 B to
//! 16 MiB, measures sequential read and write separately, and runs a
//! mixed reader/writer contention sweep. Run conditions are
//! `direct=1, ioengine=libaio, iodepth=10, size=5G, time_based,
//! runtime=1m`; every conclusion drawn from these numbers inherits those
//! conditions, and in particular the fixed queue depth. See
//! [the crate bibliography](crate#sources) for everything else this
//! simulator is grounded on.
//!
//! Values are stored **in the units fio reported them**, so each literal
//! can be checked against its source file by eye; conversion to bytes
//! per second happens once, in [`Bandwidth::bytes_per_s`]. That is
//! deliberate — hand-converted constants are how wrong numbers get into
//! documents.
//!
//! **These runs are unbuffered.** The fio configs set `direct=1`, so
//! nothing passes through the page cache and no readahead occurs. That
//! matters for how the cost model should be read: there is no imposed
//! fetch granularity in this data. A read returns exactly the block that
//! was asked for, and the curve prices that choice.
//!
//! So `W` is better understood as **the read size the algorithm chooses**
//! than as an overhead the tier inflicts. The sweep then says what that
//! choice is worth: small blocks sit at the device's operation-rate
//! ceiling and waste its bandwidth; large blocks reach peak bandwidth at
//! a fraction of the operation rate. Ordering matters twice over — it
//! lets adjacent plan entries coalesce into larger reads, which moves the
//! workload rightwards along the curve, and it turns scattered seeks into
//! forward ones.
//!
//! What the sweep gives that an assumed constant cannot:
//!
//! - **`W` is measurable.** The block size at which random reads reach
//!   sequential throughput is a property of the device, and it differs by
//!   an order of magnitude across these three. [`Regime::efficient_block`]
//!   reads it off the curve instead of assuming 128 KiB.
//! - **The random penalty is measurable**, and it is nothing like
//!   uniform: at 4 KiB it is 183× on the spinning disk and under 3× on
//!   the NVMe drive. Conclusions that hold in one regime do not survive
//!   into the other, which is the point of naming them.

/// A bandwidth figure as fio printed it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bandwidth {
    /// fio's `bw=NNNB/s`.
    BytesPerS(u64),
    /// fio's `bw=NNNKB/s` — kibibytes per second.
    KibPerS(u64),
    /// fio's `bw=NNN.NKB/s`, in tenths of a kibibyte per second.
    KibTenthsPerS(u64),
    /// fio's `bw=NNN.NMB/s`, in tenths of a mebibyte per second.
    MibTenthsPerS(u64),
}

impl Bandwidth {
    pub fn bytes_per_s(self) -> u64 {
        match self {
            Bandwidth::BytesPerS(v) => v,
            Bandwidth::KibPerS(v) => v * 1024,
            Bandwidth::KibTenthsPerS(v) => v * 1024 / 10,
            Bandwidth::MibTenthsPerS(v) => v * 1024 * 1024 / 10,
        }
    }

    pub fn mib_per_s(self) -> f64 {
        self.bytes_per_s() as f64 / (1024.0 * 1024.0)
    }
}

/// One measured point on a device's random-read curve.
#[derive(Debug, Clone, Copy)]
pub struct RandomPoint {
    pub block_bytes: u64,
    pub iops: u64,
    pub bandwidth: Bandwidth,
}

/// One point from the *mixed* sweep: a random reader running concurrently
/// with a sequential reader and a sequential writer, each sequential
/// stream rate-capped at `seq_cap` (uncapped when `None`).
///
/// This is the Transfer stage's actual shape — scattered input reads
/// competing with a streaming output write — so it is the axis on which
/// the additive read-cost-plus-write-cost model either holds or fails.
#[derive(Debug, Clone, Copy)]
pub struct ContentionPoint {
    /// The `rate=` cap fio applied to *each* sequential job.
    pub seq_cap: Option<Bandwidth>,
    /// What the random reader actually achieved.
    pub random_iops: u64,
    pub random_bw: Bandwidth,
    /// What the sequential jobs actually achieved, which is below the cap
    /// whenever the device could not sustain it.
    pub seq_read: Bandwidth,
    pub seq_write: Bandwidth,
}

impl ContentionPoint {
    /// Total device throughput across all three jobs.
    pub fn total_bytes_per_s(&self) -> u64 {
        self.random_bw.bytes_per_s() + self.seq_read.bytes_per_s() + self.seq_write.bytes_per_s()
    }
}

/// A device's measured cost behavior.
#[derive(Debug, Clone, Copy)]
pub struct Regime {
    /// Short handle used in study output.
    pub name: &'static str,
    /// The device the numbers came from.
    pub device: &'static str,
    /// Where to find the run these numbers were read from.
    pub source: &'static str,
    /// Queue depth the random sweep was run at.
    pub queue_depth: u32,
    pub seq_read: Bandwidth,
    pub seq_write: Bandwidth,
    /// Random reads by block size, ascending.
    pub random_read: &'static [RandomPoint],
    /// Random reads contending with sequential streams, by ascending
    /// sequential rate cap, uncapped last. The random job uses an
    /// 8 KiB–16 KiB block range throughout.
    pub contention: &'static [ContentionPoint],
}

impl Regime {
    /// The measured random-read point at or above `block_bytes`, falling
    /// back to the largest measured block.
    pub fn random_at(&self, block_bytes: u64) -> RandomPoint {
        self.random_read
            .iter()
            .find(|p| p.block_bytes >= block_bytes)
            .copied()
            .unwrap_or_else(|| *self.random_read.last().expect("curve is non-empty"))
    }

    /// **`W`, measured.** The smallest block size at which random reads
    /// reach `fraction` of sequential read throughput — the point past
    /// which ordering stops paying.
    pub fn efficient_block(&self, fraction: f64) -> Option<u64> {
        let target = self.seq_read.bytes_per_s() as f64 * fraction;
        self.random_read
            .iter()
            .find(|p| p.bandwidth.bytes_per_s() as f64 >= target)
            .map(|p| p.block_bytes)
    }

    /// How much worse random access is than sequential, at a block size.
    /// This is the ratio the whole algorithm trades against.
    pub fn random_penalty(&self, block_bytes: u64) -> f64 {
        let seq = self.seq_read.bytes_per_s() as f64;
        let rnd = self.random_at(block_bytes).bandwidth.bytes_per_s() as f64;
        if rnd == 0.0 { f64::INFINITY } else { seq / rnd }
    }

    /// Seconds to read `count` blocks of `block_bytes` at random.
    pub fn random_read_seconds(&self, count: u64, block_bytes: u64) -> f64 {
        let p = self.random_at(block_bytes);
        // Bounded by both the operation rate and the byte rate at that
        // block size; fio measures them together, so either is a valid
        // read of the same point.
        let by_ops = count as f64 / p.iops as f64;
        let by_bytes = (count * block_bytes) as f64 / p.bandwidth.bytes_per_s() as f64;
        by_ops.max(by_bytes)
    }

    /// Seconds to read `bytes` sequentially.
    pub fn sequential_read_seconds(&self, bytes: u64) -> f64 {
        bytes as f64 / self.seq_read.bytes_per_s() as f64
    }

    /// Seconds to write `bytes` sequentially.
    pub fn sequential_write_seconds(&self, bytes: u64) -> f64 {
        bytes as f64 / self.seq_write.bytes_per_s() as f64
    }

    /// The contention points where the sequential streams were held to a
    /// rate.
    pub fn capped_contention(&self) -> impl Iterator<Item = &ContentionPoint> {
        self.contention.iter().filter(|p| p.seq_cap.is_some())
    }

    /// The contention point where the sequential streams ran free.
    pub fn uncapped_contention(&self) -> Option<&ContentionPoint> {
        self.contention.iter().find(|p| p.seq_cap.is_none())
    }

    /// **How much a concurrent random reader loses by letting the
    /// sequential streams run uncapped**, as a ratio of operation rates
    /// against the gentlest capped point.
    ///
    /// This is the number that makes rate governance a correctness
    /// concern for Transfer rather than a tuning preference.
    pub fn starvation_ratio(&self) -> Option<f64> {
        let gentlest = self.capped_contention().next()?;
        let free = self.uncapped_contention()?;
        Some(gentlest.random_iops as f64 / free.random_iops.max(1) as f64)
    }
}

/// 7200 RPM SATA disk. Seek-bound: the random-read rate is flat at about
/// 260 IOPS from 512 B all the way to 16 KiB, because the seek costs the
/// same regardless of how much is transferred once the head arrives.
pub const SPINNING_SATA: Regime = Regime {
    name: "spinning-sata",
    device: "TOSHIBA HDWD110 1TB, xfs",
    source: "perfscripts historic/TOSHIBA_HDWD110_xfs_1TB",
    queue_depth: 10,
    seq_read: Bandwidth::KibPerS(196_485),
    seq_write: Bandwidth::KibPerS(195_143),
    random_read: &[
        RandomPoint {
            block_bytes: 512,
            iops: 255,
            bandwidth: Bandwidth::BytesPerS(130_830),
        },
        RandomPoint {
            block_bytes: 1_024,
            iops: 267,
            bandwidth: Bandwidth::BytesPerS(274_150),
        },
        RandomPoint {
            block_bytes: 2_048,
            iops: 267,
            bandwidth: Bandwidth::BytesPerS(547_654),
        },
        RandomPoint {
            block_bytes: 4_096,
            iops: 266,
            bandwidth: Bandwidth::MibTenthsPerS(10),
        },
        RandomPoint {
            block_bytes: 8_192,
            iops: 262,
            bandwidth: Bandwidth::MibTenthsPerS(20),
        },
        RandomPoint {
            block_bytes: 16_384,
            iops: 264,
            bandwidth: Bandwidth::KibPerS(4_225),
        },
        RandomPoint {
            block_bytes: 32_768,
            iops: 256,
            bandwidth: Bandwidth::KibPerS(8_210),
        },
        RandomPoint {
            block_bytes: 65_536,
            iops: 246,
            bandwidth: Bandwidth::KibPerS(15_779),
        },
        RandomPoint {
            block_bytes: 131_072,
            iops: 225,
            bandwidth: Bandwidth::KibPerS(28_902),
        },
        RandomPoint {
            block_bytes: 262_144,
            iops: 190,
            bandwidth: Bandwidth::KibPerS(48_750),
        },
        RandomPoint {
            block_bytes: 524_288,
            iops: 146,
            bandwidth: Bandwidth::KibPerS(75_051),
        },
        RandomPoint {
            block_bytes: 1_048_576,
            iops: 111,
            bandwidth: Bandwidth::KibPerS(114_170),
        },
        RandomPoint {
            block_bytes: 2_097_152,
            iops: 58,
            bandwidth: Bandwidth::KibPerS(119_183),
        },
        RandomPoint {
            block_bytes: 4_194_304,
            iops: 36,
            bandwidth: Bandwidth::KibPerS(147_847),
        },
        RandomPoint {
            block_bytes: 8_388_608,
            iops: 20,
            bandwidth: Bandwidth::KibPerS(170_828),
        },
        RandomPoint {
            block_bytes: 16_777_216,
            iops: 11,
            bandwidth: Bandwidth::KibPerS(181_440),
        },
    ],
    // The disk cannot sustain the higher caps at all: past 40 MiB/s the
    // sequential jobs themselves fall short of their rate, and the random
    // reader is down to a few dozen operations per second.
    contention: &[
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(100)),
            random_iops: 208,
            random_bw: Bandwidth::KibTenthsPerS(16_654),
            seq_read: Bandwidth::KibPerS(10_261),
            seq_write: Bandwidth::KibPerS(10_261),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(200)),
            random_iops: 177,
            random_bw: Bandwidth::KibTenthsPerS(14_204),
            seq_read: Bandwidth::KibPerS(20_453),
            seq_write: Bandwidth::KibPerS(20_602),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(400)),
            random_iops: 94,
            random_bw: Bandwidth::BytesPerS(772_347),
            seq_read: Bandwidth::KibPerS(40_972),
            seq_write: Bandwidth::KibPerS(40_889),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(800)),
            random_iops: 35,
            random_bw: Bandwidth::BytesPerS(294_341),
            seq_read: Bandwidth::KibPerS(81_562),
            seq_write: Bandwidth::KibPerS(44_586),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(1_200)),
            random_iops: 21,
            random_bw: Bandwidth::BytesPerS(174_347),
            seq_read: Bandwidth::KibPerS(121_127),
            seq_write: Bandwidth::KibPerS(34_803),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(1_600)),
            random_iops: 24,
            random_bw: Bandwidth::BytesPerS(198_284),
            seq_read: Bandwidth::KibPerS(117_298),
            seq_write: Bandwidth::KibPerS(30_679),
        },
        ContentionPoint {
            seq_cap: None,
            random_iops: 25,
            random_bw: Bandwidth::BytesPerS(212_780),
            seq_read: Bandwidth::KibPerS(116_739),
            seq_write: Bandwidth::KibPerS(30_128),
        },
    ],
};

/// SATA SSD. The link saturates near 555 MB/s, and random reads reach it
/// by 64 KiB — so above that block size, ordering buys almost nothing.
pub const SATA_SSD: Regime = Regime {
    name: "sata-ssd",
    device: "Samsung 850 Pro 256GB",
    source: "perfscripts historic/Samsung_Evo850Pro_xvs_256GB",
    queue_depth: 10,
    seq_read: Bandwidth::KibPerS(554_876),
    seq_write: Bandwidth::KibPerS(524_907),
    random_read: &[
        RandomPoint {
            block_bytes: 512,
            iops: 77_400,
            bandwidth: Bandwidth::KibPerS(38_700),
        },
        RandomPoint {
            block_bytes: 1_024,
            iops: 80_051,
            bandwidth: Bandwidth::KibPerS(80_051),
        },
        RandomPoint {
            block_bytes: 2_048,
            iops: 80_049,
            bandwidth: Bandwidth::KibPerS(160_099),
        },
        RandomPoint {
            block_bytes: 4_096,
            iops: 75_515,
            bandwidth: Bandwidth::KibPerS(302_063),
        },
        RandomPoint {
            block_bytes: 8_192,
            iops: 53_107,
            bandwidth: Bandwidth::KibPerS(424_859),
        },
        RandomPoint {
            block_bytes: 16_384,
            iops: 31_672,
            bandwidth: Bandwidth::KibPerS(506_766),
        },
        RandomPoint {
            block_bytes: 32_768,
            iops: 16_582,
            bandwidth: Bandwidth::KibPerS(530_640),
        },
        RandomPoint {
            block_bytes: 65_536,
            iops: 8_481,
            bandwidth: Bandwidth::KibPerS(542_838),
        },
        RandomPoint {
            block_bytes: 131_072,
            iops: 4_290,
            bandwidth: Bandwidth::KibPerS(549_133),
        },
        RandomPoint {
            block_bytes: 262_144,
            iops: 2_158,
            bandwidth: Bandwidth::KibPerS(552_517),
        },
        RandomPoint {
            block_bytes: 524_288,
            iops: 1_082,
            bandwidth: Bandwidth::KibPerS(554_114),
        },
        RandomPoint {
            block_bytes: 1_048_576,
            iops: 541,
            bandwidth: Bandwidth::KibPerS(554_859),
        },
        RandomPoint {
            block_bytes: 2_097_152,
            iops: 270,
            bandwidth: Bandwidth::KibPerS(554_846),
        },
        RandomPoint {
            block_bytes: 4_194_304,
            iops: 135,
            bandwidth: Bandwidth::KibPerS(554_938),
        },
        RandomPoint {
            block_bytes: 8_388_608,
            iops: 67,
            bandwidth: Bandwidth::KibPerS(554_905),
        },
        RandomPoint {
            block_bytes: 16_777_216,
            iops: 33,
            bandwidth: Bandwidth::KibPerS(554_920),
        },
    ],
    // Capped, the trade is orderly. Uncapped, the random reader falls
    // from 48k to 254 operations per second.
    contention: &[
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(100)),
            random_iops: 48_474,
            random_bw: Bandwidth::KibPerS(387_818),
            seq_read: Bandwidth::KibPerS(10_273),
            seq_write: Bandwidth::KibPerS(10_273),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(200)),
            random_iops: 45_809,
            random_bw: Bandwidth::KibPerS(366_490),
            seq_read: Bandwidth::KibPerS(20_512),
            seq_write: Bandwidth::KibPerS(20_512),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(400)),
            random_iops: 40_761,
            random_bw: Bandwidth::KibPerS(326_104),
            seq_read: Bandwidth::KibPerS(40_991),
            seq_write: Bandwidth::KibPerS(40_991),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(800)),
            random_iops: 30_777,
            random_bw: Bandwidth::KibPerS(246_232),
            seq_read: Bandwidth::KibPerS(81_947),
            seq_write: Bandwidth::KibPerS(81_947),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(1_200)),
            random_iops: 21_797,
            random_bw: Bandwidth::KibPerS(174_387),
            seq_read: Bandwidth::KibPerS(122_930),
            seq_write: Bandwidth::KibPerS(122_929),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(1_600)),
            random_iops: 15_126,
            random_bw: Bandwidth::KibPerS(121_019),
            seq_read: Bandwidth::KibPerS(163_883),
            seq_write: Bandwidth::KibPerS(163_883),
        },
        ContentionPoint {
            seq_cap: None,
            random_iops: 254,
            random_bw: Bandwidth::KibTenthsPerS(20_399),
            seq_read: Bandwidth::KibPerS(261_182),
            seq_write: Bandwidth::KibPerS(260_917),
        },
    ],
};

/// Consumer NVMe. Random reads *exceed* the single-stream sequential
/// figure once blocks reach 32 KiB, because parallelism across the queue
/// beats one sequential reader.
pub const NVME_CONSUMER: Regime = Regime {
    name: "nvme-consumer",
    device: "Samsung 950 Pro 256GB NVMe",
    source: "perfscripts historic/Samsung_NVMe_950Pro_256GB",
    queue_depth: 10,
    seq_read: Bandwidth::MibTenthsPerS(14_274),
    seq_write: Bandwidth::KibPerS(933_819),
    random_read: &[
        RandomPoint {
            block_bytes: 512,
            iops: 124_013,
            bandwidth: Bandwidth::KibPerS(62_007),
        },
        RandomPoint {
            block_bytes: 1_024,
            iops: 118_830,
            bandwidth: Bandwidth::KibPerS(118_831),
        },
        RandomPoint {
            block_bytes: 2_048,
            iops: 119_639,
            bandwidth: Bandwidth::KibPerS(239_280),
        },
        RandomPoint {
            block_bytes: 4_096,
            iops: 122_099,
            bandwidth: Bandwidth::KibPerS(488_400),
        },
        RandomPoint {
            block_bytes: 8_192,
            iops: 104_121,
            bandwidth: Bandwidth::KibPerS(832_975),
        },
        RandomPoint {
            block_bytes: 16_384,
            iops: 72_908,
            bandwidth: Bandwidth::MibTenthsPerS(11_392),
        },
        RandomPoint {
            block_bytes: 32_768,
            iops: 47_619,
            bandwidth: Bandwidth::MibTenthsPerS(14_882),
        },
        RandomPoint {
            block_bytes: 65_536,
            iops: 25_196,
            bandwidth: Bandwidth::MibTenthsPerS(15_749),
        },
        RandomPoint {
            block_bytes: 131_072,
            iops: 13_339,
            bandwidth: Bandwidth::MibTenthsPerS(16_675),
        },
        RandomPoint {
            block_bytes: 262_144,
            iops: 5_361,
            bandwidth: Bandwidth::MibTenthsPerS(13_405),
        },
        RandomPoint {
            block_bytes: 524_288,
            iops: 2_797,
            bandwidth: Bandwidth::MibTenthsPerS(13_986),
        },
        RandomPoint {
            block_bytes: 1_048_576,
            iops: 1_324,
            bandwidth: Bandwidth::MibTenthsPerS(13_245),
        },
        RandomPoint {
            block_bytes: 2_097_152,
            iops: 671,
            bandwidth: Bandwidth::MibTenthsPerS(13_432),
        },
        RandomPoint {
            block_bytes: 4_194_304,
            iops: 329,
            bandwidth: Bandwidth::MibTenthsPerS(13_168),
        },
        RandomPoint {
            block_bytes: 8_388_608,
            iops: 165,
            bandwidth: Bandwidth::MibTenthsPerS(13_205),
        },
        RandomPoint {
            block_bytes: 16_777_216,
            iops: 81,
            bandwidth: Bandwidth::MibTenthsPerS(13_082),
        },
    ],
    // Total throughput stays near 830 MB/s across every capped point —
    // the jobs are splitting one bandwidth pool. Uncapped, the split
    // stops being a split: the random reader gets 567 operations per
    // second, down from 100k.
    contention: &[
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(100)),
            random_iops: 100_937,
            random_bw: Bandwidth::KibPerS(807_534),
            seq_read: Bandwidth::KibPerS(10_274),
            seq_write: Bandwidth::KibPerS(10_274),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(200)),
            random_iops: 98_634,
            random_bw: Bandwidth::KibPerS(789_114),
            seq_read: Bandwidth::KibPerS(20_513),
            seq_write: Bandwidth::KibPerS(20_514),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(400)),
            random_iops: 93_705,
            random_bw: Bandwidth::KibPerS(749_679),
            seq_read: Bandwidth::KibPerS(40_993),
            seq_write: Bandwidth::KibPerS(40_993),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(800)),
            random_iops: 84_033,
            random_bw: Bandwidth::KibPerS(672_301),
            seq_read: Bandwidth::KibPerS(81_951),
            seq_write: Bandwidth::KibPerS(81_953),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(1_200)),
            random_iops: 74_395,
            random_bw: Bandwidth::KibPerS(595_188),
            seq_read: Bandwidth::KibPerS(122_940),
            seq_write: Bandwidth::KibPerS(122_944),
        },
        ContentionPoint {
            seq_cap: Some(Bandwidth::MibTenthsPerS(1_600)),
            random_iops: 64_491,
            random_bw: Bandwidth::KibPerS(515_956),
            seq_read: Bandwidth::KibPerS(163_897),
            seq_write: Bandwidth::KibPerS(163_903),
        },
        ContentionPoint {
            seq_cap: None,
            random_iops: 567,
            random_bw: Bandwidth::KibTenthsPerS(45_403),
            seq_read: Bandwidth::KibPerS(518_466),
            seq_write: Bandwidth::KibPerS(555_975),
        },
    ],
};

/// Every regime, for sweeps.
pub const ALL: &[Regime] = &[SPINNING_SATA, SATA_SSD, NVME_CONSUMER];

/// One measured latency distribution for a device at a block size, in
/// microseconds, on a **submission-to-completion** basis.
///
/// fio separates `slat` (time spent submitting) from `clat` (time from
/// submitted to completed) and reports `lat` as their sum. A simulator
/// that timestamps a request when it is created and again when it
/// finishes is measuring `lat`, so comparing it against `clat` — as an
/// earlier version of this did — understates the target by a couple of
/// percent and shows up as a systematic positive bias in every latency
/// metric.
///
/// The mean here is fio's `lat` mean directly. The percentiles are its
/// `clat` percentiles plus the mean `slat`, because fio reports
/// percentiles only for `clat`.
///
/// Throughput alone is a weak check — two models can agree on operations
/// per second while disagreeing entirely about what any single request
/// experienced. The storage-simulation literature validates against
/// latency for that reason, and these are the numbers to validate
/// against.
#[derive(Debug, Clone, Copy)]
pub struct LatencyPoint {
    pub block_bytes: u64,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
}

impl LatencyPoint {
    /// How heavy the measured tail is.
    pub fn tail_ratio(&self) -> f64 {
        if self.p50_us <= 0.0 {
            0.0
        } else {
            self.p99_us / self.p50_us
        }
    }
}

pub const MEASURED_LATENCY: &[(&str, &[LatencyPoint])] = &[
    (
        "spinning-sata",
        &[
            LatencyPoint {
                block_bytes: 512,
                mean_us: 39128.8,
                p50_us: 25003.0,
                p95_us: 120003.0,
                p99_us: 198003.0,
            },
            LatencyPoint {
                block_bytes: 1024,
                mean_us: 37344.9,
                p50_us: 24002.9,
                p95_us: 116002.9,
                p99_us: 182002.9,
            },
            LatencyPoint {
                block_bytes: 2048,
                mean_us: 37390.1,
                p50_us: 24003.0,
                p95_us: 115003.0,
                p99_us: 186003.0,
            },
            LatencyPoint {
                block_bytes: 4096,
                mean_us: 37488.8,
                p50_us: 24002.9,
                p95_us: 115002.9,
                p99_us: 184002.9,
            },
            LatencyPoint {
                block_bytes: 8192,
                mean_us: 38157.2,
                p50_us: 25003.1,
                p95_us: 117003.1,
                p99_us: 186003.1,
            },
            LatencyPoint {
                block_bytes: 16384,
                mean_us: 37859.7,
                p50_us: 24003.3,
                p95_us: 117003.3,
                p99_us: 190003.3,
            },
            LatencyPoint {
                block_bytes: 32768,
                mean_us: 38967.7,
                p50_us: 25004.0,
                p95_us: 120004.0,
                p99_us: 188004.0,
            },
            LatencyPoint {
                block_bytes: 65536,
                mean_us: 40553.7,
                p50_us: 26004.7,
                p95_us: 128004.7,
                p99_us: 202004.7,
            },
            LatencyPoint {
                block_bytes: 131072,
                mean_us: 44283.3,
                p50_us: 28006.7,
                p95_us: 137006.7,
                p99_us: 221006.7,
            },
        ],
    ),
    (
        "sata-ssd",
        &[
            LatencyPoint {
                block_bytes: 512,
                mean_us: 128.8,
                p50_us: 110.7,
                p95_us: 227.7,
                p99_us: 308.7,
            },
            LatencyPoint {
                block_bytes: 1024,
                mean_us: 124.5,
                p50_us: 110.8,
                p95_us: 195.8,
                p99_us: 249.8,
            },
            LatencyPoint {
                block_bytes: 2048,
                mean_us: 124.5,
                p50_us: 113.9,
                p95_us: 185.9,
                p99_us: 227.9,
            },
            LatencyPoint {
                block_bytes: 4096,
                mean_us: 132.0,
                p50_us: 122.7,
                p95_us: 189.7,
                p99_us: 225.7,
            },
            LatencyPoint {
                block_bytes: 8192,
                mean_us: 187.9,
                p50_us: 179.8,
                p95_us: 260.8,
                p99_us: 312.8,
            },
            LatencyPoint {
                block_bytes: 16384,
                mean_us: 315.2,
                p50_us: 316.8,
                p95_us: 348.8,
                p99_us: 380.8,
            },
            LatencyPoint {
                block_bytes: 32768,
                mean_us: 602.5,
                p50_us: 599.2,
                p95_us: 607.2,
                p99_us: 607.2,
            },
            LatencyPoint {
                block_bytes: 65536,
                mean_us: 1178.4,
                p50_us: 1180.1,
                p95_us: 1180.1,
                p99_us: 1180.1,
            },
            LatencyPoint {
                block_bytes: 131072,
                mean_us: 2330.3,
                p50_us: 2325.9,
                p95_us: 2325.9,
                p99_us: 2325.9,
            },
        ],
    ),
    (
        "nvme-consumer",
        &[
            LatencyPoint {
                block_bytes: 512,
                mean_us: 80.2,
                p50_us: 72.8,
                p95_us: 132.8,
                p99_us: 178.8,
            },
            LatencyPoint {
                block_bytes: 1024,
                mean_us: 83.7,
                p50_us: 72.8,
                p95_us: 146.8,
                p99_us: 214.8,
            },
            LatencyPoint {
                block_bytes: 2048,
                mean_us: 83.2,
                p50_us: 73.8,
                p95_us: 142.8,
                p99_us: 196.8,
            },
            LatencyPoint {
                block_bytes: 4096,
                mean_us: 81.4,
                p50_us: 73.9,
                p95_us: 130.9,
                p99_us: 172.9,
            },
            LatencyPoint {
                block_bytes: 8192,
                mean_us: 95.6,
                p50_us: 85.1,
                p95_us: 157.1,
                p99_us: 207.1,
            },
            LatencyPoint {
                block_bytes: 16384,
                mean_us: 136.7,
                p50_us: 122.2,
                p95_us: 233.2,
                p99_us: 308.2,
            },
            LatencyPoint {
                block_bytes: 32768,
                mean_us: 209.5,
                p50_us: 185.7,
                p95_us: 376.7,
                p99_us: 488.7,
            },
            LatencyPoint {
                block_bytes: 65536,
                mean_us: 396.4,
                p50_us: 413.6,
                p95_us: 599.6,
                p99_us: 743.6,
            },
            LatencyPoint {
                block_bytes: 131072,
                mean_us: 749.1,
                p50_us: 721.4,
                p95_us: 1149.3,
                p99_us: 1165.3,
            },
        ],
    ),
];

/// Measured latency for one device, if it was captured.
pub fn measured_latency(device: &str) -> &'static [LatencyPoint] {
    MEASURED_LATENCY
        .iter()
        .find(|(name, _)| *name == device)
        .map(|(_, points)| *points)
        .unwrap_or(&[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandwidth_units_convert_as_fio_reports_them() {
        assert_eq!(Bandwidth::KibPerS(1).bytes_per_s(), 1024);
        assert_eq!(Bandwidth::MibTenthsPerS(10).bytes_per_s(), 1024 * 1024);
        assert_eq!(Bandwidth::BytesPerS(7).bytes_per_s(), 7);
        // 1427.4 MB/s, the NVMe sequential figure.
        assert!((Bandwidth::MibTenthsPerS(14_274).mib_per_s() - 1427.4).abs() < 0.1);
    }

    #[test]
    fn curves_are_ascending_and_complete() {
        for r in ALL {
            assert_eq!(r.random_read.len(), 16, "{}: full 512B..16M sweep", r.name);
            for pair in r.random_read.windows(2) {
                assert!(
                    pair[1].block_bytes > pair[0].block_bytes,
                    "{}: blocks ascend",
                    r.name
                );
            }
        }
    }

    /// The measured spread that makes "regime" a meaningful word: the
    /// penalty for reading 4 KiB at random spans two orders of magnitude
    /// across these devices.
    #[test]
    fn the_random_penalty_differs_by_regime() {
        let hdd = SPINNING_SATA.random_penalty(4_096);
        let ssd = SATA_SSD.random_penalty(4_096);
        let nvme = NVME_CONSUMER.random_penalty(4_096);

        assert!(
            hdd > 100.0,
            "spinning disk penalty is {hdd:.0}×, expected >100"
        );
        assert!(ssd < 3.0, "SATA SSD penalty is {ssd:.1}×, expected <3");
        assert!(nvme < 4.0, "NVMe penalty is {nvme:.1}×, expected <4");
        assert!(hdd / nvme > 25.0, "the regimes must be far apart to matter");
    }

    /// `W` is a device property, and reading it off the curve gives very
    /// different answers than the 128 KiB the documents assume.
    #[test]
    fn the_efficient_block_is_measured_not_assumed() {
        // Both flash devices reach 95% of sequential at 32 KiB — well
        // under the 128 KiB the documents assume, and the same answer
        // despite a 2.6× difference in peak bandwidth.
        assert_eq!(SATA_SSD.efficient_block(0.95), Some(32_768));
        assert_eq!(NVME_CONSUMER.efficient_block(0.95), Some(32_768));
        // The spinning disk never gets there inside the sweep: even 16 MiB
        // random reads run at 92% of sequential.
        assert_eq!(SPINNING_SATA.efficient_block(0.95), None);
        assert_eq!(SPINNING_SATA.efficient_block(0.90), Some(16_777_216));
    }

    /// An unthrottled sequential writer does not merely slow a concurrent
    /// random reader down — it removes it from the schedule. Every device
    /// measured shows it, and the two flash devices show it worst,
    /// because they had the most to lose.
    #[test]
    fn an_uncapped_sequential_stream_starves_the_random_reader() {
        for r in ALL {
            let ratio = r.starvation_ratio().expect("every regime has both points");
            assert!(
                ratio > 8.0,
                "{}: uncapped sequential costs the random reader {ratio:.0}×, expected >8×",
                r.name
            );
        }
        assert!(SATA_SSD.starvation_ratio().unwrap() > 150.0);
        assert!(NVME_CONSUMER.starvation_ratio().unwrap() > 150.0);
    }

    /// Under a cap, the three jobs split one roughly fixed bandwidth
    /// pool — which is the condition under which pricing reads and writes
    /// as separate additive terms is defensible at all.
    #[test]
    fn capped_contention_splits_a_stable_bandwidth_pool() {
        let totals: Vec<u64> = NVME_CONSUMER
            .capped_contention()
            .map(|p| p.total_bytes_per_s())
            .collect();
        let lo = *totals.iter().min().unwrap() as f64;
        let hi = *totals.iter().max().unwrap() as f64;
        assert!(
            hi / lo < 1.05,
            "capped totals span {:.0}..{:.0} MB/s, expected within 5%",
            lo / 1e6,
            hi / 1e6
        );
    }

    /// The pool is only stable while the device can meet the cap. The
    /// spinning disk cannot, and its sequential *write* actually goes
    /// backwards as the demanded rate rises.
    #[test]
    fn a_saturated_device_stops_honouring_the_cap() {
        let points: Vec<&ContentionPoint> = SPINNING_SATA.capped_contention().collect();
        let gentle = points.first().unwrap();
        let harsh = points.last().unwrap();

        assert!(gentle.seq_write.bytes_per_s() >= gentle.seq_cap.unwrap().bytes_per_s() * 99 / 100);
        assert!(
            harsh.seq_write.bytes_per_s() < harsh.seq_cap.unwrap().bytes_per_s() / 4,
            "the disk should fall far short of a 160 MiB/s write cap"
        );
    }

    #[test]
    fn random_at_snaps_to_the_next_measured_block() {
        // 4100-byte records — the vector-dataset case — price at the 8 KiB
        // point, since that is the next block fio measured.
        assert_eq!(NVME_CONSUMER.random_at(4_100).block_bytes, 8_192);
        // Anything past the sweep prices at its top end.
        assert_eq!(NVME_CONSUMER.random_at(1 << 30).block_bytes, 16_777_216);
    }
}
