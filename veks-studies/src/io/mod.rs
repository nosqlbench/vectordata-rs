// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! An event-driven model of the storage path.
//!
//! [`crate::device`] prices a request stream with a closed form. It is
//! fast and it fits the measurements, but it assumes its answer: the
//! shape of the curve is baked into `min(concurrency/service, iops,
//! bw/b)`, so agreeing with a sweep tells you the parameters were fitted
//! well, not that the mechanism is right.
//!
//! This module asserts no throughput expression at all. It advances a
//! clock, and throughput is whatever comes out. What it does model:
//!
//! - **Bandwidth as a resource that is consumed.** Requests in their
//!   transfer phase share the device's ceiling; a request in its
//!   positioning phase consumes none of it. Utilization is measured, not
//!   assumed, and a stream that saturates the bus slows every other
//!   stream by exactly the share it takes.
//! - **Request size**, because transfer time is bytes divided by the
//!   share a request is currently getting — which changes as other
//!   requests start and finish.
//! - **Command queue saturation.** The device accepts `queue_slots`
//!   commands. An issuer that wants more outstanding than that blocks,
//!   and the time it spends blocked is recorded.
//! - **Scheduling order**, in two places: the OS-level [`sched`]
//!   scheduler choosing what to submit, and the device choosing which
//!   accepted command to serve next. On rotating media the second one is
//!   native command queueing, and it is where the disk's queue-depth
//!   scaling comes from.
//! - **The page cache**, in the request path. Logical accesses become
//!   page-aligned device requests, and hits never reach the device at
//!   all.
//!
//! **Fit, and one known divergence.** Against the perfscripts random-read
//! sweeps the simulator lands within 5% on the spinning disk, 9% on the
//! SATA SSD and 17% on the NVMe drive for every block size up to 1 MiB,
//! and reproduces all three sequential figures to within a percent. The
//! NVMe residual is the same non-monotonic curve that defeats the closed
//! form: that drive's random 128 KiB reads outrun its own single-stream
//! sequential read, and no single bus rate reproduces both.
//!
//! The divergence worth naming is contention. [`contended`] reproduces
//! the *direction* of the starvation the mixed fio jobs measured — an
//! uncapped writer always costs the reader — and roughly its magnitude on
//! the spinning disk, but it understates it badly on flash: the measured
//! collapse on the NVMe drive is 178x and this model produces about 2x.
//! Fluid processor-sharing gives a small read a fair slice of bandwidth
//! and it therefore completes quickly, where the real device evidently
//! does not schedule anything like that fairly once a bulk stream is
//! saturating it. The engineering conclusion is unaffected — govern the
//! writer — but the number here is a floor on the harm, not an estimate
//! of it. Closing that gap needs a fairness model this does not have.
//!
//! The test of all this is [`fio_like`]: run the same workload fio ran —
//! uniform random reads at a given block size, ten outstanding, no page
//! cache — and see whether the simulator lands on the measured IOPS. It
//! does, across three devices and five orders of magnitude of block size,
//! without any throughput formula in it.

pub mod hw;
pub mod sched;

use hw::{Hardware, ServicePolicy};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use sched::Scheduler;

/// One I/O request.
#[derive(Debug, Clone, Copy)]
pub struct Request {
    pub id: u64,
    /// Which stream issued this, so contention can be attributed.
    pub stream: usize,
    pub offset: u64,
    pub len: u64,
    pub write: bool,
    pub submitted_at: f64,
}

/// A request the device has accepted and is working on.
#[derive(Debug, Clone, Copy)]
struct InService {
    req: Request,
    /// Seconds of head travel and rotational wait still to go. While this
    /// is positive the request consumes no bandwidth.
    positioning_remaining: f64,
    bytes_remaining: f64,
    /// Extra die occupancy after the transfer — flash programming.
    program_remaining: f64,
    first_die: usize,
    die_count: usize,
}

/// Where a request stream comes from.
pub trait Issuer {
    /// The next access, or `None` when the workload is done.
    fn next(&mut self) -> Option<(u64, u64, bool)>;
}

/// Uniform random accesses of a fixed size within a span — the workload
/// the fio sweeps ran.
pub struct RandomAccess {
    rng: Xoshiro256PlusPlus,
    span_bytes: u64,
    block_bytes: u64,
    /// Upper bound when the workload uses a size range, as fio's
    /// `bsrange` does.
    block_bytes_max: u64,
    remaining: u64,
    write: bool,
}

impl RandomAccess {
    pub fn new(span_bytes: u64, block_bytes: u64, count: u64, seed: u64) -> Self {
        RandomAccess {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            span_bytes,
            block_bytes,
            block_bytes_max: block_bytes,
            remaining: count,
            write: false,
        }
    }

    /// Sizes drawn uniformly from `[low, high]`, in multiples of `low`.
    pub fn ranged(span_bytes: u64, low: u64, high: u64, count: u64, seed: u64) -> Self {
        RandomAccess {
            block_bytes_max: high,
            ..RandomAccess::new(span_bytes, low, count, seed)
        }
    }
}

impl Issuer for RandomAccess {
    fn next(&mut self) -> Option<(u64, u64, bool)> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let len = if self.block_bytes_max > self.block_bytes {
            let steps = self.block_bytes_max / self.block_bytes;
            self.block_bytes * self.rng.random_range(1..=steps)
        } else {
            self.block_bytes
        };
        let blocks = (self.span_bytes / self.block_bytes).max(1);
        let block = self.rng.random_range(0..blocks);
        Some((block * self.block_bytes, len, self.write))
    }
}

/// Accesses that walk forward through a span.
pub struct SequentialAccess {
    offset: u64,
    span_bytes: u64,
    block_bytes: u64,
    remaining: u64,
    write: bool,
}

impl SequentialAccess {
    pub fn new(span_bytes: u64, block_bytes: u64, count: u64, write: bool) -> Self {
        SequentialAccess {
            offset: 0,
            span_bytes,
            block_bytes,
            remaining: count,
            write,
        }
    }
}

impl Issuer for SequentialAccess {
    fn next(&mut self) -> Option<(u64, u64, bool)> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let at = self.offset;
        self.offset = (self.offset + self.block_bytes) % self.span_bytes.max(1);
        Some((at, self.block_bytes, self.write))
    }
}

/// A fixed list of accesses, for replaying an algorithm's trace.
pub struct Recorded {
    accesses: Vec<(u64, u64, bool)>,
    at: usize,
}

impl Recorded {
    pub fn new(accesses: Vec<(u64, u64, bool)>) -> Self {
        Recorded { accesses, at: 0 }
    }
}

impl Issuer for Recorded {
    fn next(&mut self) -> Option<(u64, u64, bool)> {
        let out = self.accesses.get(self.at).copied();
        if out.is_some() {
            self.at += 1;
        }
        out
    }
}

/// What a run cost, all of it measured from the clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct IoStats {
    pub elapsed_s: f64,
    pub requests_completed: u64,
    pub bytes_transferred: u64,
    /// Requests the page cache served without reaching the device.
    pub cache_hits: u64,
    /// Device requests issued speculatively by readahead.
    pub readahead_requests: u64,
    /// Logical accesses the issuers made.
    pub accesses: u64,
    /// Time with at least one command in service.
    pub busy_s: f64,
    /// Time with at least one command actually moving bytes. The gap
    /// between this and `busy_s` is positioning — pure loss on a disk.
    pub transferring_s: f64,
    /// Time the issuer was blocked because the device would accept no
    /// more commands.
    pub queue_blocked_s: f64,
    /// Time the issuer was blocked waiting for a CPU core to submit on.
    /// A large value here means the model is host-bound, not
    /// device-bound — the state Ren et al. report for modern NVMe.
    pub host_blocked_s: f64,
    /// Time the issuer was blocked on memory bandwidth.
    pub memory_blocked_s: f64,
    /// The device's own ceiling, before the link, memory and interconnect
    /// are taken into account. The gap between this and
    /// `peak_bandwidth` is what the platform costs.
    pub device_peak_bandwidth: f64,
    /// Integral of in-service command count over time, so mean queue
    /// occupancy is this divided by elapsed.
    pub service_occupancy_integral: f64,
    pub total_latency_s: f64,
    pub max_latency_s: f64,
    pub peak_bandwidth: f64,
}

impl IoStats {
    pub fn iops(&self) -> f64 {
        if self.elapsed_s <= 0.0 {
            0.0
        } else {
            self.requests_completed as f64 / self.elapsed_s
        }
    }

    pub fn throughput(&self) -> f64 {
        if self.elapsed_s <= 0.0 {
            0.0
        } else {
            self.bytes_transferred as f64 / self.elapsed_s
        }
    }

    /// Fraction of the device's bandwidth ceiling actually used. This is
    /// the number that says whether a workload is bandwidth-bound.
    pub fn bandwidth_utilization(&self) -> f64 {
        if self.elapsed_s <= 0.0 || self.peak_bandwidth <= 0.0 {
            return 0.0;
        }
        self.throughput() / self.peak_bandwidth
    }

    /// Fraction of busy time spent positioning rather than transferring.
    /// On flash this is small; on a disk under random access it is nearly
    /// all of it, which is the entire reason ordering matters.
    pub fn positioning_fraction(&self) -> f64 {
        if self.busy_s <= 0.0 {
            0.0
        } else {
            1.0 - self.transferring_s / self.busy_s
        }
    }

    pub fn mean_service_occupancy(&self) -> f64 {
        if self.elapsed_s <= 0.0 {
            0.0
        } else {
            self.service_occupancy_integral / self.elapsed_s
        }
    }

    pub fn mean_latency_s(&self) -> f64 {
        if self.requests_completed == 0 {
            0.0
        } else {
            self.total_latency_s / self.requests_completed as f64
        }
    }

    /// How much of the device's own bandwidth the platform leaves
    /// reachable. Below 1.0 the device is not the constraint.
    pub fn platform_headroom(&self) -> f64 {
        if self.device_peak_bandwidth <= 0.0 {
            1.0
        } else {
            self.peak_bandwidth / self.device_peak_bandwidth
        }
    }

    /// Logical accesses per second across all streams.
    pub fn stream_access_rate(&self) -> f64 {
        if self.elapsed_s <= 0.0 {
            0.0
        } else {
            self.accesses as f64 / self.elapsed_s
        }
    }

    /// Fraction of the run spent waiting on memory bandwidth.
    pub fn memory_saturation(&self) -> f64 {
        if self.elapsed_s <= 0.0 {
            0.0
        } else {
            self.memory_blocked_s / self.elapsed_s
        }
    }

    /// Fraction of the run spent waiting on the host rather than the
    /// device.
    pub fn host_saturation(&self) -> f64 {
        if self.elapsed_s <= 0.0 {
            0.0
        } else {
            self.host_blocked_s / self.elapsed_s
        }
    }

    /// Fraction of the run in which the issuer could not submit.
    pub fn queue_saturation(&self) -> f64 {
        if self.elapsed_s <= 0.0 {
            0.0
        } else {
            self.queue_blocked_s / self.elapsed_s
        }
    }
}

/// Operating-system readahead.
///
/// The kernel does not fetch only what was asked for. On a stream it
/// judges sequential it fetches ahead, doubling the window up to
/// `max_bytes` (Linux's `ra_pages`, 128 KiB by default and 256 KiB after
/// `POSIX_FADV_SEQUENTIAL`); on access it judges random it fetches the
/// requested pages and nothing more.
///
/// **That asymmetry is the mechanism this whole family of algorithms
/// trades on**, and a model without it cannot represent the advantage
/// correctly. A fixed page size — which is what this simulator used
/// before — gives an ordered reader and a scattered reader the same
/// fetch granularity, which is exactly the distinction that matters.
///
/// Readahead also does two things a page size cannot. It **coalesces**:
/// one window-sized request replaces many page faults, which is most of
/// why sequential reading is fast. And it **wastes**: pages fetched on
/// spec that are never asked for are bandwidth spent on a guess, which
/// [`crate::cache::CacheStats::wasted_readahead_pages`] counts.
#[derive(Debug, Clone, Copy)]
pub struct Readahead {
    pub enabled: bool,
    /// Window a fresh sequential run starts at.
    pub initial_bytes: u64,
    /// Ceiling the window doubles toward.
    pub max_bytes: u64,
}

impl Readahead {
    /// Linux defaults: 128 KiB ceiling, ramping from 16 KiB.
    pub const DEFAULT: Self = Readahead {
        enabled: true,
        initial_bytes: 16 * 1024,
        max_bytes: 128 * 1024,
    };

    /// After `POSIX_FADV_SEQUENTIAL`, which doubles the ceiling.
    pub const SEQUENTIAL_ADVICE: Self = Readahead {
        enabled: true,
        initial_bytes: 32 * 1024,
        max_bytes: 256 * 1024,
    };

    /// `POSIX_FADV_RANDOM`, or `direct=1`: fetch exactly what was asked.
    pub const OFF: Self = Readahead {
        enabled: false,
        initial_bytes: 0,
        max_bytes: 0,
    };
}

/// Per-stream readahead position.
///
/// Detection tolerates gaps. Linux does not require each read to begin
/// exactly where the last ended: it maintains a readahead region and
/// triggers the next fetch when a read crosses a marker inside it, so an
/// ascending reader that *skips* — which is exactly what a pass of an
/// ordered rewrite does — keeps its readahead. Requiring strict
/// contiguity would switch readahead off for the very access pattern the
/// algorithm produces, and make page size look far more important than
/// it is.
#[derive(Debug, Clone, Copy, Default)]
struct ReadaheadState {
    /// Start of the region readahead is currently tracking.
    region_start: u64,
    /// How far ahead pages have already been brought in.
    prefetched_to: u64,
    /// Current window, doubling toward the ceiling while the pattern
    /// holds.
    window: u64,
    active: bool,
}

/// How a run is configured.
#[derive(Debug, Clone, Copy)]
pub struct RunConfig {
    /// Requests the issuer tries to keep outstanding — fio's `iodepth`.
    pub offered_depth: usize,
    /// Cache configuration, or `None` for `direct=1`.
    pub cache: Option<crate::cache::CacheConfig>,
    /// Bytes the source occupies, for sizing the cache's page space.
    pub span_bytes: u64,
    /// What it costs the host to issue an I/O.
    pub host: hw::HostModel,
    /// The link this device reaches the host through, and what shares it.
    pub fabric: hw::Fabric,
    /// Where the issuing thread sits relative to the device.
    pub numa: hw::Numa,
    /// Operating-system readahead policy.
    pub readahead: Readahead,
    pub seed: u64,
}

impl RunConfig {
    /// Unbuffered, which is how every perfscripts number was measured.
    pub fn direct(offered_depth: usize, span_bytes: u64) -> Self {
        RunConfig {
            offered_depth,
            cache: None,
            span_bytes,
            host: hw::HostModel::DEFAULT,
            fabric: hw::Fabric::DEDICATED,
            numa: hw::Numa::LOCAL,
            // Unbuffered by default, matching how the fio corpus was
            // captured: `direct=1` bypasses the page cache and with it
            // readahead.
            readahead: Readahead::OFF,
            seed: 0x5A17,
        }
    }

    /// A buffered run: page cache with kernel-default readahead.
    pub fn buffered(
        offered_depth: usize,
        span_bytes: u64,
        cache: crate::cache::CacheConfig,
    ) -> Self {
        RunConfig {
            cache: Some(cache),
            readahead: Readahead::DEFAULT,
            ..Self::direct(offered_depth, span_bytes)
        }
    }

    /// Unbuffered, with the host taken out of the picture, for isolating
    /// what the device alone does.
    pub fn device_only(offered_depth: usize, span_bytes: u64) -> Self {
        RunConfig {
            host: hw::HostModel::FREE,
            ..Self::direct(offered_depth, span_bytes)
        }
    }
}

/// One independent source of requests, with its own concurrency and its
/// own optional rate limit.
///
/// Streams are the point at which contention becomes expressible. A
/// single stream can only tell you how fast a device is; two of them
/// tell you what one costs the other, which is the question that matters
/// for a rewrite that reads scattered input while writing streaming
/// output.
pub struct Stream<'a> {
    pub label: &'static str,
    pub issuer: &'a mut dyn Issuer,
    /// Requests this stream tries to keep outstanding.
    pub offered_depth: usize,
    /// Bytes per second this stream may submit, or `None` for unlimited.
    /// This is fio's `rate=`, and it is the governor a Transfer stage
    /// would apply to its own output.
    pub rate_cap: Option<f64>,
    /// A background stream that runs for as long as the measurement
    /// lasts rather than for a fixed number of requests.
    ///
    /// This matters more than it sounds. fio's mixed jobs are
    /// `time_based`: every job runs the whole 60 seconds, so each one's
    /// rate is measured over the same window. Giving a background stream
    /// a fixed request count instead makes it finish early or late and
    /// silently changes what the other streams' averages mean — a
    /// rate-capped writer stretches the run and dilutes the reader's
    /// figure into the idle tail.
    pub looping: bool,
}

impl<'a> Stream<'a> {
    pub fn new(label: &'static str, issuer: &'a mut dyn Issuer, offered_depth: usize) -> Self {
        Stream {
            label,
            issuer,
            offered_depth,
            rate_cap: None,
            looping: false,
        }
    }

    pub fn capped(mut self, bytes_per_s: f64) -> Self {
        self.rate_cap = Some(bytes_per_s);
        self
    }

    /// Run for the duration of the measurement rather than for a fixed
    /// number of requests.
    pub fn background(mut self) -> Self {
        self.looping = true;
        self
    }
}

/// Per-stream results.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamStats {
    pub label: &'static str,
    /// Logical accesses the issuer made. Distinct from `completed`
    /// because one access may be served from cache (no device request),
    /// split into several (a scattered range), or cause extra device
    /// work that nobody asked for (readahead).
    pub accesses: u64,
    /// Device requests completed on this stream's behalf.
    pub completed: u64,
    pub bytes: u64,
    pub cache_hits: u64,
    pub total_latency_s: f64,
    pub elapsed_s: f64,
}

impl StreamStats {
    pub fn iops(&self) -> f64 {
        if self.elapsed_s <= 0.0 {
            0.0
        } else {
            self.completed as f64 / self.elapsed_s
        }
    }

    pub fn throughput(&self) -> f64 {
        if self.elapsed_s <= 0.0 {
            0.0
        } else {
            self.bytes as f64 / self.elapsed_s
        }
    }

    pub fn mean_latency_s(&self) -> f64 {
        if self.completed == 0 {
            0.0
        } else {
            self.total_latency_s / self.completed as f64
        }
    }

    /// Logical accesses per second — the application's view, which
    /// diverges from device IOPS as soon as a cache is involved.
    pub fn access_rate(&self) -> f64 {
        if self.elapsed_s <= 0.0 {
            0.0
        } else {
            self.accesses as f64 / self.elapsed_s
        }
    }
}

/// Everything a run produced.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub total: IoStats,
    pub streams: Vec<StreamStats>,
}

impl RunResult {
    pub fn stream(&self, label: &str) -> &StreamStats {
        self.streams
            .iter()
            .find(|s| s.label == label)
            .expect("no such stream")
    }
}

/// Run a workload against a device and report what the clock said.
pub fn run(
    hardware: &Hardware,
    scheduler: &mut dyn Scheduler,
    issuer: &mut dyn Issuer,
    config: RunConfig,
) -> IoStats {
    let depth = config.offered_depth;
    let mut streams = [Stream::new("main", issuer, depth)];
    run_streams(hardware, scheduler, &mut streams, config).total
}

/// The general case: several independent streams contending for one
/// device.
pub fn run_streams(
    hardware: &Hardware,
    scheduler: &mut dyn Scheduler,
    streams: &mut [Stream<'_>],
    config: RunConfig,
) -> RunResult {
    let mut cache = config
        .cache
        .map(|c| crate::cache::PageCache::new(c, config.span_bytes, config.span_bytes));

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
    let mut clock = 0.0f64;
    let mut head: u64 = rng.random_range(0..config.span_bytes.max(1));

    let n = streams.len();
    let mut outstanding = vec![0usize; n];
    let mut next_submit_at = vec![0.0f64; n];
    let mut exhausted = vec![false; n];
    let mut per_stream: Vec<StreamStats> = streams
        .iter()
        .map(|s| StreamStats {
            label: s.label,
            ..StreamStats::default()
        })
        .collect();

    let mut serving: Vec<InService> = Vec::with_capacity(hardware.dies);
    let mut die_busy = vec![false; hardware.dies.max(1)];
    let mut host_free_at = vec![0.0f64; config.host.cores.max(1)];
    let mut controller_free_at = 0.0f64;
    let mut next_id: u64 = 0;

    // Three ceilings above the device's own: its share of the upstream
    // link, what the host's memory can carry given every byte is touched
    // several times, and whatever the interconnect leaves of that when
    // the issuer is on another node.
    let memory_ceiling = config.host.io_bandwidth_ceiling() * config.numa.bandwidth_factor();
    let peak = hardware
        .peak_bandwidth()
        .min(config.fabric.share())
        .min(memory_ceiling);
    let numa_latency = config.numa.latency_penalty_s();
    let mut ra_state = vec![ReadaheadState::default(); n];
    let mut memory_free_at = 0.0f64;
    let mut stats = IoStats {
        peak_bandwidth: peak,
        device_peak_bandwidth: hardware.peak_bandwidth(),
        ..IoStats::default()
    };

    loop {
        // ---- Submission -------------------------------------------------
        let mut blocked = false;
        // Once every foreground stream is finished the measurement is
        // over, whatever the background streams still have queued.
        let foreground_done = (0..n)
            .filter(|&i| !streams[i].looping)
            .all(|i| exhausted[i] && outstanding[i] == 0);
        if foreground_done && (0..n).any(|i| !streams[i].looping) {
            break;
        }

        let mut host_blocked = false;
        let mut memory_blocked = false;
        for (i, stream) in streams.iter_mut().enumerate() {
            while !exhausted[i] && outstanding[i] < stream.offered_depth {
                // A rate-capped stream may simply not be allowed to
                // submit yet, however idle the device is.
                if next_submit_at[i] > clock {
                    break;
                }
                // The device holds a finite number of commands. Wanting
                // to submit when none are free is queue saturation, and
                // it is the issuer that waits.
                if serving.len() + scheduler.len() >= hardware.queue_slots {
                    blocked = true;
                    break;
                }
                // Issuing an I/O costs the host CPU. Above roughly half a
                // million operations per second this, and not the device,
                // is what runs out.
                //
                // Every gate has to be checked *before* the issuer is
                // asked for an access, because breaking out afterwards
                // throws that access away. Doing it the other way round
                // silently dropped every second access.
                let core = host_free_at
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .expect("at least one core");
                if host_free_at[core] > clock {
                    host_blocked = true;
                    break;
                }
                // A cache hit is a memcpy and copies cost memory time, so
                // the memory subsystem gates submission too.
                if cache.is_some() && memory_free_at > clock {
                    memory_blocked = true;
                    break;
                }
                let Some((offset, len, write)) = stream.issuer.next() else {
                    exhausted[i] = true;
                    break;
                };
                host_free_at[core] = clock + config.host.per_request_s;
                per_stream[i].accesses += 1;
                stats.accesses += 1;

                if let Some(c) = cache.as_mut() {
                    let page = c.stats().page_bytes;
                    let issued_before = outstanding[i];
                    // What the device has to fetch for this access, as
                    // contiguous runs rather than page at a time.
                    let runs = c.missing_runs(crate::cache::Region::Input, offset, len);

                    let before = c.stats();
                    c.access(crate::cache::Region::Input, offset, len, write);
                    let after = c.stats();
                    let hits = (after.read_hits + after.write_hits)
                        - (before.read_hits + before.write_hits);
                    stats.cache_hits += hits;
                    per_stream[i].cache_hits += hits;

                    memory_free_at = clock.max(memory_free_at) + config.host.memory_time_s(len);

                    if let Some(cap) = stream.rate_cap {
                        next_submit_at[i] = clock.max(next_submit_at[i]) + len as f64 / cap;
                    }

                    // Readahead runs on reads, and only where the access
                    // pattern looks sequential.
                    let mut prefetch: Option<(u64, u64)> = None;
                    if config.readahead.enabled && !write {
                        let ra = &mut ra_state[i];
                        // Ascending and within reach of what readahead is
                        // already tracking counts as sequential, gaps
                        // included. Going backwards, or forwards past the
                        // window, does not.
                        let reach = ra.window.max(config.readahead.initial_bytes);
                        let sequential = ra.active
                            && offset >= ra.region_start
                            && offset <= ra.prefetched_to + reach;

                        if sequential {
                            // Fire once per window, when the reader gets
                            // within half a window of the end of what has
                            // been fetched — Linux's async marker. Firing
                            // on every read instead would issue one tiny
                            // prefetch per read and coalesce nothing,
                            // which is the opposite of the point.
                            let next_window = if ra.window == 0 {
                                config.readahead.initial_bytes
                            } else {
                                (ra.window * 2).min(config.readahead.max_bytes)
                            };
                            let marker = ra.prefetched_to.saturating_sub(ra.window / 2);
                            if offset + len >= marker {
                                ra.window = next_window;
                                let from = ra.prefetched_to.max(offset + len);
                                prefetch = Some((from, ra.window));
                                ra.prefetched_to = from + ra.window;
                            }
                        } else {
                            // Pattern broken: the kernel stops guessing
                            // and reads only what was asked for.
                            ra.active = true;
                            ra.window = 0;
                            ra.region_start = offset;
                            ra.prefetched_to = offset + len;
                        }
                    }

                    for (start_page, count) in runs {
                        scheduler.push(Request {
                            id: next_id,
                            stream: i,
                            offset: start_page * page,
                            len: count * page,
                            write,
                            submitted_at: clock,
                        });
                        next_id += 1;
                        outstanding[i] += 1;
                    }

                    // The readahead itself: one request, however many
                    // pages it covers. That coalescing is most of why a
                    // sequential read is fast.
                    if let Some((from, bytes)) = prefetch {
                        let fetched = c.prefetch(crate::cache::Region::Input, from, bytes);
                        if fetched > 0 {
                            scheduler.push(Request {
                                id: next_id,
                                stream: i,
                                offset: from - (from % page),
                                len: fetched * page,
                                write: false,
                                submitted_at: clock,
                            });
                            next_id += 1;
                            outstanding[i] += 1;
                            stats.readahead_requests += 1;
                        }
                    }

                    // A pure hit reaches no device, so nothing completes
                    // for it later; it is counted as an access above and
                    // deliberately not as a device request.
                    let _ = issued_before;
                    continue;
                }

                scheduler.push(Request {
                    id: next_id,
                    stream: i,
                    offset,
                    len,
                    write,
                    submitted_at: clock,
                });
                next_id += 1;
                outstanding[i] += 1;
                if let Some(cap) = stream.rate_cap {
                    next_submit_at[i] = clock.max(next_submit_at[i]) + len as f64 / cap;
                }
            }
        }

        // ---- Dispatch ---------------------------------------------------
        while serving.len() < hardware.dies {
            // A request can only start if *its own* die is free. This is
            // the whole of read/write interference: nothing else in the
            // device being idle helps a request whose die is mid-program.
            let free_die = |r: &Request| hardware.dies_free(r.offset, r.len, &die_busy);
            let in_flight = serving.len() + scheduler.len();
            let picked = match hardware.policy {
                ServicePolicy::Fifo => scheduler.pop_first_where(&free_die),
                ServicePolicy::NearestFirst => scheduler.pop_best_within_where(
                    hardware.reorder_window,
                    &|r: &Request| hardware.access_time_at_depth(head, r.offset, clock, in_flight),
                    &free_die,
                ),
            };
            let Some(req) = picked else { break };
            let controller_wait = (controller_free_at - clock).max(0.0);
            controller_free_at = clock + controller_wait + hardware.command_time_s();
            let positioning = controller_wait
                + numa_latency
                + hardware.access_time_at_depth(
                    head,
                    req.offset,
                    clock + controller_wait,
                    in_flight,
                );
            head = req.offset + req.len;
            let first_die = hardware.die_of(req.offset);
            let die_count = hardware.dies_spanned(req.len);
            for i in 0..die_count {
                die_busy[(first_die + i) % hardware.dies.max(1)] = true;
            }
            serving.push(InService {
                req,
                positioning_remaining: positioning,
                bytes_remaining: req.len as f64,
                program_remaining: hardware.write_occupancy_s(req.write),
                first_die,
                die_count,
            });
        }

        // ---- Advance ----------------------------------------------------
        if serving.is_empty() {
            // Nothing to do. Either everything is finished, or every
            // stream is waiting on its rate limit and the clock has to
            // move forward to the next moment one may submit.
            if !scheduler.is_empty() {
                continue;
            }
            let all_done = (0..n).all(|i| streams[i].looping || exhausted[i]);
            let host_ready = host_free_at.iter().cloned().fold(f64::INFINITY, f64::min);
            let waiting = (0..n)
                .filter(|&i| !exhausted[i] && next_submit_at[i] > clock)
                .map(|i| next_submit_at[i])
                .fold(f64::INFINITY, f64::min);
            if all_done {
                break;
            }
            if waiting.is_finite() {
                clock = waiting.min(host_ready.max(clock));
                continue;
            }
            if host_ready > clock && host_ready.is_finite() {
                stats.host_blocked_s += host_ready - clock;
                clock = host_ready;
                continue;
            }
            if memory_free_at > clock && memory_free_at.is_finite() {
                stats.memory_blocked_s += memory_free_at - clock;
                clock = memory_free_at;
                continue;
            }
            break;
        }

        let transferring = serving
            .iter()
            .filter(|s| s.positioning_remaining <= 0.0 && s.bytes_remaining > 1e-6)
            .count();
        // Each transferring request takes an equal share of the device's
        // bandwidth, capped by what its own dies can supply.
        let share = if transferring == 0 {
            0.0
        } else {
            peak / transferring as f64
        };
        let rate_of = |s: &InService| -> f64 {
            if s.positioning_remaining > 0.0 || s.bytes_remaining <= 1e-6 {
                0.0
            } else {
                share.min(hardware.request_rate(s.req.len))
            }
        };

        let mut dt = f64::INFINITY;
        for s in &serving {
            if s.positioning_remaining > 0.0 {
                dt = dt.min(s.positioning_remaining);
            } else if s.bytes_remaining > 1e-6 {
                let r = rate_of(s);
                if r > 0.0 {
                    dt = dt.min(s.bytes_remaining / r);
                }
            } else if s.program_remaining > 0.0 {
                // Programming holds the die but moves no bytes.
                dt = dt.min(s.program_remaining);
            }
        }
        // A rate-limited stream may become eligible before anything in
        // service finishes; do not step past that moment.
        for i in 0..n {
            if !exhausted[i]
                && outstanding[i] < streams[i].offered_depth
                && next_submit_at[i] > clock
            {
                dt = dt.min(next_submit_at[i] - clock);
            }
        }
        if host_blocked {
            let next_core = host_free_at.iter().cloned().fold(f64::INFINITY, f64::min);
            if next_core > clock {
                dt = dt.min(next_core - clock);
            }
        }
        if memory_blocked && memory_free_at > clock {
            dt = dt.min(memory_free_at - clock);
        }
        if !dt.is_finite() || dt <= 0.0 {
            dt = 1e-12;
        }

        stats.busy_s += dt;
        if transferring > 0 {
            stats.transferring_s += dt;
            let moved: f64 = serving.iter().map(rate_of).sum::<f64>() * dt;
            stats.bytes_transferred += moved.round() as u64;
        }
        if blocked {
            stats.queue_blocked_s += dt;
        }
        if host_blocked {
            stats.host_blocked_s += dt;
        }
        if memory_blocked {
            stats.memory_blocked_s += dt;
        }
        stats.service_occupancy_integral += serving.len() as f64 * dt;
        clock += dt;

        for s in serving.iter_mut() {
            if s.positioning_remaining > 0.0 {
                s.positioning_remaining -= dt;
            } else if s.bytes_remaining > 1e-6 {
                s.bytes_remaining -= rate_of(s) * dt;
            } else {
                s.program_remaining -= dt;
            }
        }

        // ---- Completion -------------------------------------------------
        let mut i = 0;
        while i < serving.len() {
            let s = serving[i];
            if s.positioning_remaining <= 0.0
                && s.bytes_remaining <= 1e-6
                && s.program_remaining <= 0.0
            {
                for i in 0..s.die_count {
                    die_busy[(s.first_die + i) % hardware.dies.max(1)] = false;
                }
                stats.requests_completed += 1;
                let latency = clock - s.req.submitted_at;
                stats.total_latency_s += latency;
                stats.max_latency_s = stats.max_latency_s.max(latency);

                let sidx = s.req.stream;
                per_stream[sidx].completed += 1;
                per_stream[sidx].bytes += s.req.len;
                per_stream[sidx].total_latency_s += latency;
                outstanding[sidx] -= 1;
                serving.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    stats.elapsed_s = clock;
    for s in per_stream.iter_mut() {
        s.elapsed_s = clock;
    }
    RunResult {
        total: stats,
        streams: per_stream,
    }
}

/// Reproduce a perfscripts random-read point: uniform random reads of
/// `block_bytes` over a 5 GiB span at `iodepth=10`, unbuffered.
///
/// This is the forward-simulation check. Nothing in [`run`] computes a
/// throughput; it advances a clock through positioning and transfer
/// phases under a shared bandwidth ceiling. If the IOPS that fall out
/// match what fio measured, the mechanism is right.
pub fn fio_like(hardware: &Hardware, block_bytes: u64, requests: u64) -> IoStats {
    const SPAN: u64 = 5 * 1024 * 1024 * 1024;
    let mut scheduler = sched::Noop::default();
    let mut issuer = RandomAccess::new(SPAN, block_bytes, requests, 0xF10);
    run(
        hardware,
        &mut scheduler,
        &mut issuer,
        RunConfig::direct(10, SPAN),
    )
}

/// The sequential counterpart, for checking the streaming side.
pub fn fio_like_sequential(hardware: &Hardware, block_bytes: u64, requests: u64) -> IoStats {
    const SPAN: u64 = 5 * 1024 * 1024 * 1024;
    let mut scheduler = sched::Noop::default();
    let mut issuer = SequentialAccess::new(SPAN, block_bytes, requests, false);
    run(
        hardware,
        &mut scheduler,
        &mut issuer,
        RunConfig::direct(10, SPAN),
    )
}

#[cfg(test)]
mod dump {
    use super::*;
    use crate::regime::ALL;

    #[test]
    #[ignore = "diagnostic dump, not an assertion"]
    fn print_forward_simulation_fit() {
        for (hardware, regime) in hw::ALL_HARDWARE.iter().zip(ALL.iter()) {
            println!("\n{} vs {}", hardware.name, regime.device);
            println!(
                "  {:>9}  {:>10}  {:>10}  {:>7}  {:>6}  {:>6}",
                "block", "sim iops", "fio iops", "error", "util", "posn"
            );
            for p in regime.random_read {
                let n = if p.block_bytes >= 1 << 20 { 300 } else { 4_000 };
                let s = fio_like(hardware, p.block_bytes, n);
                let err = (s.iops() - p.iops as f64) / p.iops as f64;
                println!(
                    "  {:>9}  {:>10.0}  {:>10}  {:>6.1}%  {:>5.0}%  {:>5.0}%",
                    p.block_bytes,
                    s.iops(),
                    p.iops,
                    err * 100.0,
                    s.bandwidth_utilization() * 100.0,
                    s.positioning_fraction() * 100.0
                );
            }
            let seq = fio_like_sequential(hardware, 1 << 20, 2_000);
            println!(
                "  {:>9}  {:>10.0}  {:>10.0}  (seq MB/s)  posn {:.0}%  util {:.0}%  occ {:.2}",
                "seq 1M",
                seq.throughput() / 1e6,
                regime.seq_read.bytes_per_s() as f64 / 1e6,
                seq.positioning_fraction() * 100.0,
                seq.bandwidth_utilization() * 100.0,
                seq.mean_service_occupancy()
            );
        }
    }
}

/// Two contending streams: scattered reads alongside a streaming writer,
/// which is the shape of a rewrite's Transfer stage and of the `mixed`
/// fio jobs.
pub fn contended(
    hardware: &Hardware,
    reader_block: u64,
    writer_cap: Option<f64>,
    requests: u64,
) -> RunResult {
    const SPAN: u64 = 5 * 1024 * 1024 * 1024;
    let mut scheduler = sched::Noop::default();
    let mut reads = RandomAccess::new(SPAN, reader_block, requests, 0xBEEF);
    // The writer runs for the whole measurement, as fio's does.
    let mut writes = SequentialAccess::new(SPAN, 1 << 20, u64::MAX, true);
    let writer = Stream::new("writer", &mut writes, 10).background();
    let mut streams = [
        Stream::new("reader", &mut reads, 10),
        match writer_cap {
            Some(cap) => writer.capped(cap),
            None => writer,
        },
    ];
    run_streams(
        hardware,
        &mut scheduler,
        &mut streams,
        RunConfig::direct(10, SPAN),
    )
}

/// A faithful reproduction of the perfscripts `mixed` job: a random
/// reader at `bsrange=8k-16k` alongside a sequential reader and a
/// sequential writer at `bs=1m`, each at `iodepth=10`, with both
/// sequential jobs held to `cap` bytes per second (uncapped when `None`).
///
/// This exists so simulated contention can be compared against the
/// measured [`crate::regime::ContentionPoint`] sweep directly, rather
/// than only in direction.
pub fn mixed_job(hardware: &Hardware, cap: Option<f64>, reader_requests: u64) -> RunResult {
    const SPAN: u64 = 5 * 1024 * 1024 * 1024;
    let mut scheduler = sched::Noop::default();
    let mut rand_reads = RandomAccess::ranged(SPAN, 8 * 1024, 16 * 1024, reader_requests, 0xBEEF);
    let mut seq_reads = SequentialAccess::new(SPAN, 1 << 20, u64::MAX, false);
    let mut seq_writes = SequentialAccess::new(SPAN, 1 << 20, u64::MAX, true);

    let sr = Stream::new("seqread", &mut seq_reads, 10).background();
    let sw = Stream::new("seqwrite", &mut seq_writes, 10).background();
    let mut streams = [
        Stream::new("randread", &mut rand_reads, 10),
        match cap {
            Some(c) => sr.capped(c),
            None => sr,
        },
        match cap {
            Some(c) => sw.capped(c),
            None => sw,
        },
    ];
    run_streams(
        hardware,
        &mut scheduler,
        &mut streams,
        RunConfig::direct(10, SPAN),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regime::{ALL, NVME_CONSUMER, SATA_SSD, SPINNING_SATA};

    /// Requests to simulate per point. Enough for steady state without
    /// making the suite slow; the run is deterministic, so this is a
    /// precision knob, not a flakiness one.
    fn sample_count(block_bytes: u64) -> u64 {
        if block_bytes >= 1 << 21 { 200 } else { 2_000 }
    }

    /// **The central claim of this module.** Nothing in [`run_streams`]
    /// computes a throughput: it advances a clock through positioning and
    /// transfer phases against a shared bandwidth ceiling, a finite
    /// command queue and a serial controller. If the IOPS that fall out
    /// of that match what fio measured on three unrelated devices across
    /// five orders of magnitude of block size, the mechanism is right.
    #[test]
    fn the_simulator_reproduces_the_measured_random_curves() {
        for (hardware, regime) in hw::ALL_HARDWARE.iter().zip(ALL.iter()) {
            for p in regime
                .random_read
                .iter()
                .filter(|p| p.block_bytes <= 1 << 20)
            {
                let s = fio_like(hardware, p.block_bytes, sample_count(p.block_bytes));
                let error = (s.iops() - p.iops as f64).abs() / p.iops as f64;
                assert!(
                    error < 0.20,
                    "{} at {} B: simulated {:.0} IOPS, fio measured {} ({:.0}% off)",
                    hardware.name,
                    p.block_bytes,
                    s.iops(),
                    p.iops,
                    error * 100.0
                );
            }
        }
    }

    /// The spinning disk, whose curve is the hardest to reproduce because
    /// it is entirely positional, is the one that fits best.
    #[test]
    fn the_disk_curve_is_reproduced_closely() {
        for p in SPINNING_SATA
            .random_read
            .iter()
            .filter(|p| p.block_bytes <= 1 << 20)
        {
            let s = fio_like(
                &hw::SPINNING_SATA_HW,
                p.block_bytes,
                sample_count(p.block_bytes),
            );
            let error = (s.iops() - p.iops as f64).abs() / p.iops as f64;
            assert!(
                error < 0.06,
                "{} B: {:.0}% off",
                p.block_bytes,
                error * 100.0
            );
        }
    }

    /// Sequential throughput is not fitted anywhere — it is what happens
    /// when consecutive requests need no repositioning.
    #[test]
    fn the_simulator_reproduces_measured_sequential_throughput() {
        for (hardware, regime) in hw::ALL_HARDWARE.iter().zip(ALL.iter()) {
            let s = fio_like_sequential(hardware, 1 << 20, 1_500);
            let measured = regime.seq_read.bytes_per_s() as f64;
            let error = (s.throughput() - measured).abs() / measured;
            assert!(
                error < 0.05,
                "{}: simulated {:.0} MB/s, measured {:.0} MB/s",
                hardware.name,
                s.throughput() / 1e6,
                measured / 1e6
            );
            assert!(
                s.positioning_fraction() < 0.02,
                "sequential should barely position"
            );
        }
    }

    /// **Where a disk's time actually goes.** Under 4 KiB random reads it
    /// spends 99% of its busy time moving the head and waiting for the
    /// platter, and 1% moving bytes. Reading in order converts almost all
    /// of that back into transfer.
    ///
    /// This is the SPLAT thesis stated as a measurement rather than an
    /// argument, and it is why bandwidth utilization is the number to
    /// watch: the same device is at 1% of its bandwidth in one case and
    /// 100% in the other.
    #[test]
    fn a_disk_under_random_access_spends_its_life_positioning() {
        let random = fio_like(&hw::SPINNING_SATA_HW, 4_096, 2_000);
        let ordered = fio_like_sequential(&hw::SPINNING_SATA_HW, 4_096, 20_000);

        assert!(
            random.positioning_fraction() > 0.95,
            "{:.2}",
            random.positioning_fraction()
        );
        assert!(
            ordered.positioning_fraction() < 0.05,
            "{:.2}",
            ordered.positioning_fraction()
        );
        assert!(random.bandwidth_utilization() < 0.02);
        assert!(ordered.bandwidth_utilization() > 0.90);
    }

    /// Flash has no geometry, so its small-block limit is the controller
    /// and its large-block limit is the bus — and utilization shows which
    /// one is binding without being told.
    #[test]
    fn utilization_reveals_which_ceiling_binds() {
        let small = fio_like(&hw::NVME_CONSUMER_HW, 512, 4_000);
        let large = fio_like(&hw::NVME_CONSUMER_HW, 1 << 20, 300);

        assert!(
            small.bandwidth_utilization() < 0.10,
            "small blocks waste bandwidth"
        );
        assert!(
            large.bandwidth_utilization() > 0.90,
            "large blocks saturate it"
        );
        // And the small case is pinned to the command rate.
        let expected = hw::NVME_CONSUMER_HW.max_command_rate;
        assert!((small.iops() - expected).abs() / expected < 0.05);
    }

    /// Queue depth helps a disk only because it gives the firmware
    /// something to choose between. One command in hand is one seek; ten
    /// is a choice of ten, and the shortest of them is much shorter.
    #[test]
    fn queue_depth_helps_a_disk_only_by_giving_it_a_choice() {
        const SPAN: u64 = 5 * 1024 * 1024 * 1024;
        let at = |depth: usize| {
            let mut sched = sched::Noop::default();
            let mut issuer = RandomAccess::new(SPAN, 4_096, 2_000, 0xF10);
            run(
                &hw::SPINNING_SATA_HW,
                &mut sched,
                &mut issuer,
                RunConfig::direct(depth, SPAN),
            )
            .iops()
        };
        let shallow = at(1);
        let deep = at(10);
        assert!(
            deep > shallow * 1.5,
            "reordering should pay: {shallow:.0} → {deep:.0} IOPS"
        );
        // But it is still one head: the gain is nothing like linear.
        assert!(deep < shallow * 10.0, "ten slots cannot mean ten heads");
    }

    /// Flash gains from depth for the opposite reason — parallel channels,
    /// not reordering — and stops gaining once the ceiling binds.
    #[test]
    fn queue_depth_helps_flash_by_filling_channels() {
        const SPAN: u64 = 5 * 1024 * 1024 * 1024;
        let at = |depth: usize| {
            let mut sched = sched::Noop::default();
            let mut issuer = RandomAccess::new(SPAN, 4_096, 4_000, 0xF10);
            run(
                &hw::NVME_CONSUMER_HW,
                &mut sched,
                &mut issuer,
                RunConfig::direct(depth, SPAN),
            )
            .iops()
        };
        assert!(at(8) > at(1) * 5.0, "channels should fill");
        assert!(at(64) < at(16) * 1.2, "and then the command rate binds");
    }

    /// **The starvation the mixed fio sweep measured, derived.** An
    /// unthrottled sequential writer keeps the command queue full of its
    /// own work, and a concurrent random reader is left with whatever
    /// slots it can win. Capping the writer restores the reader.
    #[test]
    fn an_uncapped_writer_starves_a_concurrent_reader() {
        for hardware in hw::ALL_HARDWARE {
            let capped = contended(hardware, 8_192, Some(40.0e6), 3_000);
            let free = contended(hardware, 8_192, None, 3_000);

            let capped_reader = capped.stream("reader").iops();
            let free_reader = free.stream("reader").iops();
            assert!(
                free_reader < capped_reader,
                "{}: an uncapped writer must cost the reader ({capped_reader:.0} → {free_reader:.0} IOPS)",
                hardware.name
            );
            assert!(
                free.stream("writer").throughput() > capped.stream("writer").throughput(),
                "{}: and the writer must be the one gaining",
                hardware.name
            );
        }
    }

    /// The cap is honoured when the device can sustain it, and quietly is
    /// not when the device cannot — which is exactly what the spinning
    /// disk does in the measured sweep.
    #[test]
    fn a_saturated_device_cannot_honour_a_rate_cap() {
        let cap = 160.0e6;
        // The reader's request count sets the measurement window, so it
        // has to be long enough for a rate-limited writer to demonstrate
        // its rate. A few thousand reads is thirty seconds on the disk
        // and thirty milliseconds on the NVMe drive.
        let disk = contended(&hw::SPINNING_SATA_HW, 8_192, Some(cap), 1_500);
        let nvme = contended(&hw::NVME_CONSUMER_HW, 8_192, Some(cap), 40_000);

        assert!(
            disk.stream("writer").throughput() < cap * 0.9,
            "the disk should fall short of a 160 MB/s cap while also reading"
        );
        assert!(
            nvme.stream("writer").throughput() > cap * 0.9,
            "the NVMe drive should meet it comfortably"
        );
    }

    /// The page cache belongs in the request path, not beside it: a hit
    /// must never become a device request.
    #[test]
    fn cached_reads_never_reach_the_device() {
        const SPAN: u64 = 64 * 1024 * 1024;
        let mut sched = sched::Noop::default();
        // Re-read one small region repeatedly, so everything after the
        // first pass is resident.
        let mut issuer = SequentialAccess::new(1 << 20, 4_096, 4_000, false);
        let config = RunConfig {
            seed: 1,
            ..RunConfig::buffered(10, SPAN, crate::cache::CacheConfig::new(8 << 20, 4_096))
        };
        let stats = run(&hw::SPINNING_SATA_HW, &mut sched, &mut issuer, config);

        assert!(
            stats.cache_hits > 3_000,
            "most reads should hit: {}",
            stats.cache_hits
        );
        assert!(
            stats.requests_completed < 300,
            "only the first pass should reach the disk: {}",
            stats.requests_completed
        );
    }

    /// Two devices with nearly identical bandwidth behave completely
    /// differently under scattered access, and the simulator has to show
    /// that without being told which is which.
    #[test]
    fn the_regimes_separate_without_being_told_to() {
        let disk = fio_like(&hw::SPINNING_SATA_HW, 4_096, 2_000);
        let ssd = fio_like(&hw::SATA_SSD_HW, 4_096, 4_000);
        assert!(
            ssd.iops() > disk.iops() * 100.0,
            "{:.0} vs {:.0} IOPS",
            ssd.iops(),
            disk.iops()
        );
        let _ = (SATA_SSD.name, NVME_CONSUMER.name);
    }
}

#[cfg(test)]
mod contention_fit {
    use super::*;
    use crate::regime::ALL;

    #[test]
    #[ignore = "diagnostic dump, not an assertion"]
    fn print_contention_fit() {
        for (hardware, regime) in hw::HISTORICAL_HARDWARE.iter().zip(ALL.iter()) {
            println!("\n{} vs {}", hardware.name, regime.device);
            println!(
                "  {:>10}  {:>12}  {:>12}  {:>8}",
                "seq cap", "sim randread", "fio randread", "error"
            );
            let n = if hardware.name == "spinning-sata" {
                400
            } else {
                12_000
            };
            for point in regime.contention {
                let cap = point.seq_cap.map(|c| c.bytes_per_s() as f64);
                let r = mixed_job(hardware, cap, n);
                let sim = r.stream("randread").iops();
                let measured = point.random_iops as f64;
                println!(
                    "  {:>10}  {:>12.0}  {:>12}  {:>7.0}%",
                    cap.map(|c| format!("{:.0}M", c / 1e6))
                        .unwrap_or("none".into()),
                    sim,
                    point.random_iops,
                    (sim - measured) / measured * 100.0
                );
            }
        }
    }
}

#[cfg(test)]
mod modern {
    use super::*;
    use crate::regime::ALL;

    fn at_depth(
        hardware: &Hardware,
        block: u64,
        depth: usize,
        host: hw::HostModel,
        n: u64,
    ) -> IoStats {
        const SPAN: u64 = 5 * 1024 * 1024 * 1024;
        let mut sched = sched::Noop::default();
        let mut issuer = RandomAccess::new(SPAN, block, n, 0xF10);
        let config = RunConfig {
            host,
            ..RunConfig::direct(depth, SPAN)
        };
        run(hardware, &mut sched, &mut issuer, config)
    }

    /// **The host is the bottleneck on a modern device**, which is the
    /// headline finding of Ren et al. and something a device-only model
    /// cannot express. With one core issuing, throughput pins to the
    /// host's ceiling while the device sits far from saturated.
    #[test]
    fn a_single_core_bottlenecks_a_modern_drive() {
        let host = hw::HostModel::DEFAULT;
        let s = at_depth(&hw::NVME_MODERN_HW, 4_096, 256, host, 60_000);

        let ceiling = host.ceiling_iops();
        assert!(
            (s.iops() - ceiling).abs() / ceiling < 0.10,
            "expected to pin near the host ceiling of {ceiling:.0} IOPS, got {:.0}",
            s.iops()
        );
        assert!(
            s.bandwidth_utilization() < 0.55,
            "and the device should be far from saturated, at {:.0}%",
            s.bandwidth_utilization() * 100.0
        );
        assert!(
            s.host_saturation() > 0.5,
            "most of the run should be host-blocked"
        );
    }

    /// Give it cores and the device becomes the limit again.
    #[test]
    fn more_cores_move_the_bottleneck_back_to_the_device() {
        let one = at_depth(
            &hw::NVME_MODERN_HW,
            4_096,
            256,
            hw::HostModel::DEFAULT,
            60_000,
        );
        let many = at_depth(
            &hw::NVME_MODERN_HW,
            4_096,
            256,
            hw::HostModel::cores(8),
            200_000,
        );

        assert!(
            many.iops() > one.iops() * 1.5,
            "cores should buy throughput"
        );
        assert!(many.host_saturation() < one.host_saturation());
        let ceiling = hw::NVME_MODERN_HW.max_command_rate;
        assert!(
            many.iops() > ceiling * 0.8,
            "eight cores should reach the device's command rate: {:.0} vs {ceiling:.0}",
            many.iops()
        );
    }

    /// The host cost is small enough that it does not disturb the
    /// historical fits — those devices top out an order of magnitude
    /// below where a single core runs out.
    #[test]
    fn the_host_does_not_bind_on_the_historical_devices() {
        for hardware in hw::HISTORICAL_HARDWARE {
            let s = fio_like(hardware, 4_096, 2_000);
            assert!(
                s.host_saturation() < 0.05,
                "{}: should be device-bound, host-blocked {:.0}% of the run",
                hardware.name,
                s.host_saturation() * 100.0
            );
        }
    }

    /// The modern regime has to hit the published anchors it was
    /// calibrated to: ~1M 4 KiB random read IOPS and 7 GB/s sequential.
    #[test]
    fn the_modern_regime_reproduces_its_published_anchors() {
        let random = at_depth(
            &hw::NVME_MODERN_HW,
            4_096,
            256,
            hw::HostModel::cores(16),
            300_000,
        );
        assert!(
            (0.9e6..=1.3e6).contains(&random.iops()),
            "expected ~1M 4 KiB IOPS, got {:.0}",
            random.iops()
        );

        const SPAN: u64 = 5 * 1024 * 1024 * 1024;
        let mut sched = sched::Noop::default();
        let mut issuer = SequentialAccess::new(SPAN, 1 << 20, 4_000, false);
        let seq = run(
            &hw::NVME_MODERN_HW,
            &mut sched,
            &mut issuer,
            RunConfig {
                host: hw::HostModel::cores(4),
                ..RunConfig::direct(32, SPAN)
            },
        );
        assert!(
            (6.3e9..=7.1e9).contains(&seq.throughput()),
            "expected ~7 GB/s sequential, got {:.2} GB/s",
            seq.throughput() / 1e9
        );
    }

    /// A modern drive is roughly an order of magnitude past the 2016 one
    /// in operation rate — which is exactly why keeping only the
    /// historical regimes would understate current hardware.
    #[test]
    fn the_modern_drive_is_an_order_of_magnitude_past_the_historical_one() {
        let old = at_depth(
            &hw::NVME_CONSUMER_HW,
            4_096,
            256,
            hw::HostModel::FREE,
            40_000,
        );
        let new = at_depth(
            &hw::NVME_MODERN_HW,
            4_096,
            256,
            hw::HostModel::FREE,
            300_000,
        );
        assert!(
            new.iops() > old.iops() * 8.0,
            "{:.0} vs {:.0} IOPS",
            new.iops(),
            old.iops()
        );
    }

    /// **Simulated contention against measured contention.** Die-level
    /// blocking is what makes this comparable at all: without it the
    /// model produced roughly 2× starvation where the measurement shows
    /// nearly 200×.
    ///
    /// The residual is honest and stated: under a cap the model gives the
    /// random reader about 30% less than measured, and uncapped it
    /// starves it about half as hard as reality does.
    #[test]
    fn simulated_contention_tracks_the_measured_mixed_sweep() {
        for (hardware, regime) in hw::HISTORICAL_HARDWARE.iter().zip(ALL.iter()).skip(1) {
            let n = 8_000;
            for point in regime.capped_contention().take(3) {
                let cap = point.seq_cap.map(|c| c.bytes_per_s() as f64);
                let sim = mixed_job(hardware, cap, n).stream("randread").iops();
                let measured = point.random_iops as f64;
                let error = (sim - measured).abs() / measured;
                assert!(
                    error < 0.40,
                    "{}: capped contention {sim:.0} vs measured {measured:.0}",
                    hardware.name
                );
            }

            let free = mixed_job(hardware, None, n).stream("randread").iops();
            let capped = regime.capped_contention().next().unwrap().random_iops as f64;
            let collapse = capped / free;
            assert!(
                collapse > 20.0,
                "{}: uncapped sequential should collapse the reader by far more than \
                 an order of magnitude, got {collapse:.0}×",
                hardware.name
            );
        }
    }
}

#[cfg(test)]
mod platform {
    use super::*;
    use crate::cache::CacheConfig;

    const SPAN: u64 = 1 << 30;

    fn ordered(config: RunConfig, hardware: &Hardware, n: u64) -> IoStats {
        let mut sched = sched::Noop::default();
        let mut issuer = SequentialAccess::new(SPAN, 4_096, n, false);
        run(hardware, &mut sched, &mut issuer, config)
    }

    fn scattered(config: RunConfig, hardware: &Hardware, n: u64) -> IoStats {
        let mut sched = sched::Noop::default();
        let mut issuer = RandomAccess::new(SPAN, 4_096, n, 0x9E37);
        run(hardware, &mut sched, &mut issuer, config)
    }

    fn buffered(ra: Readahead) -> RunConfig {
        RunConfig {
            readahead: ra,
            ..RunConfig::buffered(32, SPAN, CacheConfig::new(8 << 20, 4_096))
        }
    }

    // ---- Readahead ----------------------------------------------------

    /// **Readahead is asymmetric, and that asymmetry is the point.** It
    /// engages for an ascending reader and disengages for a scattered
    /// one, so the two do not merely differ in locality — they get
    /// different fetch granularity from the kernel.
    #[test]
    fn readahead_engages_for_ordered_access_and_not_for_scattered() {
        let cfg = buffered(Readahead::DEFAULT);
        let o = ordered(cfg, &hw::SPINNING_SATA_HW, 20_000);
        let s = scattered(cfg, &hw::SPINNING_SATA_HW, 2_000);

        let ordered_share = o.readahead_requests as f64 / o.requests_completed.max(1) as f64;
        let scattered_share = s.readahead_requests as f64 / s.requests_completed.max(1) as f64;
        assert!(
            ordered_share > 0.5,
            "most of an ordered reader's device traffic should be readahead, got {ordered_share:.2}"
        );
        assert!(
            scattered_share < 0.05,
            "a scattered reader should get essentially none, got {scattered_share:.2}"
        );
    }

    /// Readahead coalesces: one window-sized request replaces many page
    /// faults. Without it an ordered reader issues one request per page.
    #[test]
    fn readahead_coalesces_an_ordered_reader() {
        let with = ordered(buffered(Readahead::DEFAULT), &hw::SPINNING_SATA_HW, 20_000);
        let without = ordered(buffered(Readahead::OFF), &hw::SPINNING_SATA_HW, 20_000);

        assert!(
            with.requests_completed < without.requests_completed / 4,
            "readahead should collapse the request count: {} vs {}",
            with.requests_completed,
            without.requests_completed
        );
        // It need not collapse the *time*. Contiguous 4 KiB reads on a
        // disk cost almost nothing extra even issued one at a time, so
        // there is little for coalescing to recover. What readahead buys
        // is spent where a per-request cost binds — see
        // `readahead_pays_where_per_request_cost_binds`.
        assert!(with.elapsed_s <= without.elapsed_s * 1.05);
    }

    /// Where coalescing actually pays: a device fast enough that the
    /// per-request cost is the constraint. Turning thousands of page
    /// faults into dozens of window fetches takes the host out of the
    /// way.
    #[test]
    fn readahead_pays_where_per_request_cost_binds() {
        let at = |ra: Readahead| {
            let mut sched = sched::Noop::default();
            let mut issuer = SequentialAccess::new(SPAN, 4_096, 200_000, false);
            run(
                &hw::NVME_MODERN_HW,
                &mut sched,
                &mut issuer,
                RunConfig {
                    readahead: ra,
                    host: hw::HostModel::DEFAULT,
                    ..RunConfig::buffered(32, SPAN, CacheConfig::new(256 << 20, 4_096))
                },
            )
        };
        let without = at(Readahead::OFF);
        let with = at(Readahead::DEFAULT);
        assert!(
            with.stream_access_rate() > without.stream_access_rate() * 2.0,
            "coalescing should lift the application rate: {:.0} vs {:.0} accesses/s",
            with.stream_access_rate(),
            without.stream_access_rate()
        );
    }

    /// `POSIX_FADV_SEQUENTIAL` doubles the ceiling, and the model should
    /// show the larger window doing fewer, bigger fetches.
    #[test]
    fn sequential_advice_widens_the_window() {
        let normal = ordered(buffered(Readahead::DEFAULT), &hw::SPINNING_SATA_HW, 40_000);
        let advised = ordered(
            buffered(Readahead::SEQUENTIAL_ADVICE),
            &hw::SPINNING_SATA_HW,
            40_000,
        );
        assert!(
            advised.readahead_requests < normal.readahead_requests,
            "wider windows mean fewer of them: {} vs {}",
            advised.readahead_requests,
            normal.readahead_requests
        );
    }

    /// **The correction readahead forces.** Without it, page size is the
    /// only fetch-granularity knob and an ordered reader looks barely
    /// better than a scattered one. With it, the kernel gives the ordered
    /// reader 128 KiB requests and the scattered reader nothing, and the
    /// gap widens by more than an order of magnitude.
    ///
    /// A model without readahead understates what ordering is worth.
    #[test]
    fn readahead_rescues_ascending_access_that_skips() {
        // A pass of an ordered rewrite does not read every record: it
        // ascends and skips. Fully contiguous access is cheap with or
        // without readahead, so it is this pattern that shows what
        // readahead is for.
        let strided = |ra: Readahead| {
            let mut sched = sched::Noop::default();
            let accesses: Vec<(u64, u64, bool)> =
                (0..8_000u64).map(|i| (i * 4_096 * 4, 512, false)).collect();
            let mut issuer = Recorded::new(accesses);
            run(
                &hw::SPINNING_SATA_HW,
                &mut sched,
                &mut issuer,
                RunConfig {
                    readahead: ra,
                    ..RunConfig::buffered(32, SPAN, CacheConfig::new(8 << 20, 4_096))
                },
            )
        };
        let without = strided(Readahead::OFF);
        let with = strided(Readahead::DEFAULT);

        assert!(
            with.requests_completed < without.requests_completed / 3,
            "readahead should coalesce a skipping ascent: {} vs {} requests",
            with.requests_completed,
            without.requests_completed
        );
        // The price is real and this is where it shows: readahead fills
        // the gaps the reader was skipping, so it moves more bytes for
        // the same time. On seek-bound media that trade is roughly
        // neutral; it is bad whenever bandwidth is the constraint.
        assert!(
            with.bytes_transferred > without.bytes_transferred * 2,
            "readahead should move materially more bytes: {} vs {}",
            with.bytes_transferred,
            without.bytes_transferred
        );
        assert!(with.readahead_requests > 0);
    }

    // ---- Memory bandwidth ---------------------------------------------

    /// Memory is a ceiling on storage throughput, because every byte that
    /// arrives is touched several times before anyone uses it.
    #[test]
    fn memory_bandwidth_caps_device_throughput() {
        let plenty = hw::HostModel {
            memory_bandwidth: 200.0e9,
            ..hw::HostModel::cores(16)
        };
        let scarce = hw::HostModel {
            memory_bandwidth: 6.0e9,
            ..hw::HostModel::cores(16)
        };

        let at = |host: hw::HostModel| {
            let mut sched = sched::Noop::default();
            let mut issuer = SequentialAccess::new(SPAN, 1 << 20, 3_000, false);
            run(
                &hw::NVME_MODERN_HW,
                &mut sched,
                &mut issuer,
                RunConfig {
                    host,
                    ..RunConfig::direct(64, SPAN)
                },
            )
        };

        let fast = at(plenty);
        let slow = at(scarce);
        assert!(
            slow.throughput() < fast.throughput() / 2.0,
            "scarce memory should halve storage throughput: {:.1} vs {:.1} GB/s",
            slow.throughput() / 1e9,
            fast.throughput() / 1e9
        );
        assert!(
            slow.platform_headroom() < 0.5,
            "and the platform, not the device, should be the constraint"
        );
        assert!(fast.platform_headroom() > 0.9);
    }

    /// A cache hit is a memcpy. It is far cheaper than a device read, but
    /// it is not free, and a model that treats it as free will predict
    /// unreachable throughput for a well-cached workload.
    #[test]
    fn cache_hits_consume_memory_bandwidth() {
        let hot = CacheConfig::new(64 << 20, 4_096);
        let at = |bw: f64| {
            let mut sched = sched::Noop::default();
            // A small region read repeatedly, so almost everything hits.
            let mut issuer = SequentialAccess::new(1 << 20, 4_096, 20_000, false);
            run(
                &hw::NVME_MODERN_HW,
                &mut sched,
                &mut issuer,
                RunConfig {
                    host: hw::HostModel {
                        memory_bandwidth: bw,
                        ..hw::HostModel::cores(16)
                    },
                    ..RunConfig::buffered(32, SPAN, hot)
                },
            )
        };
        let fast = at(200.0e9);
        let slow = at(2.0e9);
        assert!(
            fast.cache_hits > 15_000,
            "the workload must actually be hitting: {} hits, {} device requests",
            fast.cache_hits,
            fast.requests_completed
        );
        assert!(
            slow.elapsed_s > fast.elapsed_s * 2.0,
            "hits should cost memory time: {:.4}s vs {:.4}s",
            slow.elapsed_s,
            fast.elapsed_s
        );
        assert!(slow.memory_saturation() > 0.5);
    }

    // ---- PCIe fabric ---------------------------------------------------

    /// A device's own link is not the whole story. Eight drives behind
    /// one root port share it, and the per-device ceiling cannot say so.
    #[test]
    fn sharing_an_upstream_link_divides_it() {
        let at = |fabric: hw::Fabric| {
            let mut sched = sched::Noop::default();
            let mut issuer = SequentialAccess::new(SPAN, 1 << 20, 2_000, false);
            run(
                &hw::NVME_MODERN_HW,
                &mut sched,
                &mut issuer,
                RunConfig {
                    fabric,
                    host: hw::HostModel {
                        memory_bandwidth: 400.0e9,
                        ..hw::HostModel::cores(16)
                    },
                    ..RunConfig::direct(64, SPAN)
                },
            )
        };

        let alone = at(hw::Fabric::DEDICATED);
        let shared = at(hw::Fabric::pcie4_x16(8));
        assert!(
            shared.throughput() < alone.throughput() / 1.5,
            "eight drives on one x16 link should each get much less: {:.1} vs {:.1} GB/s",
            shared.throughput() / 1e9,
            alone.throughput() / 1e9
        );
        // 28 GB/s over 8 devices is 3.5 GB/s apiece.
        assert!((shared.throughput() - 3.5e9).abs() / 3.5e9 < 0.15);
    }

    // ---- NUMA -----------------------------------------------------------

    /// Issuing from the wrong socket costs latency per request and a
    /// large share of memory bandwidth. Neither is visible in a
    /// single-socket measurement, which is what every device figure this
    /// crate calibrates against happens to be.
    #[test]
    fn a_remote_node_costs_both_latency_and_bandwidth() {
        let at = |numa: hw::Numa| {
            let mut sched = sched::Noop::default();
            let mut issuer = RandomAccess::new(SPAN, 4_096, 40_000, 0x9E37);
            run(
                &hw::NVME_MODERN_HW,
                &mut sched,
                &mut issuer,
                RunConfig {
                    numa,
                    host: hw::HostModel::cores(16),
                    ..RunConfig::direct(128, SPAN)
                },
            )
        };

        let local = at(hw::Numa::LOCAL);
        let remote = at(hw::Numa::REMOTE);

        // Latency always costs something, but concurrency hides most of
        // it — the same mechanism that erodes the case for ordering.
        assert!(
            remote.mean_latency_s() > local.mean_latency_s(),
            "remote issue must cost latency"
        );
        assert!(
            remote.iops() < local.iops(),
            "and some throughput: {:.0} vs {:.0} IOPS",
            remote.iops(),
            local.iops()
        );
        assert!(
            remote.iops() > local.iops() * 0.9,
            "but at depth 128 the latency is largely hidden, so the loss should be \
             small — it is the bandwidth that bites, and only near the ceiling"
        );
        // With ample memory the interconnect share still exceeds what the
        // device can produce, so the platform ceiling has not moved at
        // all. That is the point: NUMA is invisible until bandwidth is
        // the constraint, which the next test arranges.
        assert_eq!(remote.platform_headroom(), local.platform_headroom());
    }

    /// Where being on the wrong node actually hurts: a bandwidth-hungry
    /// stream, where the interconnect and not the request rate is what
    /// runs out.
    #[test]
    fn a_remote_node_bites_when_bandwidth_is_the_constraint() {
        let at = |numa: hw::Numa| {
            let mut sched = sched::Noop::default();
            let mut issuer = SequentialAccess::new(SPAN, 1 << 20, 3_000, false);
            run(
                &hw::NVME_MODERN_HW,
                &mut sched,
                &mut issuer,
                RunConfig {
                    numa,
                    host: hw::HostModel {
                        // One channel pair, which is what a socket on a
                        // two-socket box commonly has to itself.
                        memory_bandwidth: 24.0e9,
                        ..hw::HostModel::cores(16)
                    },
                    ..RunConfig::direct(64, SPAN)
                },
            )
        };
        let local = at(hw::Numa::LOCAL);
        let remote = at(hw::Numa::REMOTE);
        assert!(
            remote.throughput() < local.throughput() * 0.75,
            "a streaming reader should lose a quarter or more across the interconnect: \
             {:.1} vs {:.1} GB/s",
            remote.throughput() / 1e9,
            local.throughput() / 1e9
        );
    }

    /// The historical fits are unaffected by any of this, because the
    /// defaults describe the single-socket, dedicated-link, unbuffered
    /// conditions those measurements were taken under.
    #[test]
    fn the_defaults_leave_the_calibrated_fits_alone() {
        for hardware in hw::HISTORICAL_HARDWARE {
            let s = fio_like(hardware, 4_096, 2_000);
            assert!(
                (s.platform_headroom() - 1.0).abs() < 1e-9,
                "{}: the platform should not bind for a 2016 device",
                hardware.name
            );
            assert_eq!(s.readahead_requests, 0, "direct I/O has no readahead");
            assert!(s.memory_saturation() < 0.01);
        }
    }
}

#[cfg(test)]
mod accounting {
    use super::*;
    use crate::cache::CacheConfig;

    /// Every access an issuer makes must be accounted for exactly once,
    /// whether the device served it or the cache did. A run that silently
    /// drops accesses would make every throughput figure derived from it
    /// meaningless.
    #[test]
    fn every_issued_access_is_accounted_for() {
        const SPAN: u64 = 1 << 30;
        for (label, cache, ra) in [
            ("direct", None, Readahead::OFF),
            (
                "buffered",
                Some(CacheConfig::new(64 << 20, 4_096)),
                Readahead::DEFAULT,
            ),
        ] {
            let accesses = 5_000u64;
            let mut sched = sched::Noop::default();
            let mut issuer = SequentialAccess::new(1 << 20, 4_096, accesses, false);
            let r = run_streams(
                &hw::NVME_MODERN_HW,
                &mut sched,
                &mut [Stream::new("main", &mut issuer, 32)],
                RunConfig {
                    cache,
                    readahead: ra,
                    host: hw::HostModel::cores(8),
                    ..RunConfig::direct(32, SPAN)
                },
            );
            assert_eq!(
                r.stream("main").accesses,
                accesses,
                "{label}: accesses went missing"
            );
        }
    }
}
