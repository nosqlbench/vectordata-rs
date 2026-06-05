// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Bandwidth rate limiting for the object transfer paths.
//!
//! Two independent caps shape throughput, each split by direction so a
//! download limit and an upload limit are tuned separately:
//!
//! - **per-connection** — keyed by the remote socket pair (`IP:port`).
//!   Each TCP connection is shaped on its own bucket, so opening *more*
//!   connections multiplies aggregate throughput. This is the knob that
//!   makes auto-concurrent streaming visibly effective: cap a single
//!   connection, and a client that fans out across N connections gets
//!   ~N× the bandwidth.
//! - **per-client** — keyed by the remote host (`IP`). Every connection
//!   from the same address shares one bucket, so the aggregate is capped
//!   regardless of how many connections the client opens — concurrency
//!   buys nothing past the cap.
//!
//! Both default to `0` (**unlimited** — no shaping, zero overhead). When a
//! limit is active a transfer is paced by a [token bucket]: tokens accrue
//! at the configured bytes/sec up to a small burst, and a chunk waits
//! until its byte count is covered. A transfer subject to *both* a
//! per-connection and a per-client cap waits on both buckets, so the
//! tighter one governs. Buckets are created lazily per key and swept when
//! idle and unreferenced, so the registries stay bounded under connection
//! churn.
//!
//! [token bucket]: https://en.wikipedia.org/wiki/Token_bucket

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use futures_util::StreamExt;

/// Granularity of pacing: a transfer is shaped in pieces no larger than
/// this, so a single large body chunk is metered smoothly rather than as
/// one long stall. 64 KiB matches the typical socket read size.
const PACE_CHUNK: usize = 64 * 1024;

/// Idle period after which an unreferenced bucket is eligible for sweeping
/// from the registry. Generous so a brief lull between a client's
/// connections doesn't drop its shared per-client bucket mid-session.
const SWEEP_IDLE: Duration = Duration::from_secs(60);

/// Minimum interval between opportunistic registry sweeps.
const SWEEP_EVERY: Duration = Duration::from_secs(30);

/// The four bandwidth caps, in **bytes per second**. `0` means unlimited
/// (the bucket for that axis is never created and no pacing is applied).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RateLimits {
    /// Per-connection (`IP:port`) download cap.
    pub connection_download: u64,
    /// Per-connection (`IP:port`) upload cap.
    pub connection_upload: u64,
    /// Per-client (`IP`) download cap.
    pub client_download: u64,
    /// Per-client (`IP`) upload cap.
    pub client_upload: u64,
}

impl RateLimits {
    /// True when no axis is limited — the server can skip all rate-limit
    /// bookkeeping (no registry, no sweeper, no body wrapping).
    pub fn is_unlimited(&self) -> bool {
        *self == RateLimits::default()
    }
}

/// A token bucket: tokens accrue at `rate` bytes/sec up to `burst`, and a
/// reservation that overdraws leaves a debt the next accrual pays down.
///
/// `acquire` computes the wait under the lock then sleeps *without* it
/// held, so concurrent transfers sharing a bucket (the per-client case)
/// serialize only briefly on the arithmetic. Debt-carry means two
/// transfers each reserving `n` see the bucket drain by `2n` and each
/// wait their share — the aggregate converges to `rate`.
struct Bucket {
    state: Mutex<BucketState>,
    /// Wall-clock millis of the last reservation, for the idle sweep.
    last_used_ms: AtomicU64,
    start: Instant,
}

struct BucketState {
    tokens: f64,
    last: Instant,
    rate: f64,
    burst: f64,
}

impl Bucket {
    fn new(rate_bps: u64, now: Instant) -> Self {
        // Burst is small so shaping is tight and observable, but at least
        // one pace-chunk so a transfer can always make forward progress
        // without per-byte sleeps. Capped at the per-second rate so a
        // slow link can't bank a multi-second head start.
        let burst = (rate_bps as f64).min(256.0 * 1024.0).max(PACE_CHUNK as f64);
        Bucket {
            state: Mutex::new(BucketState {
                tokens: burst,
                last: now,
                rate: rate_bps as f64,
                burst,
            }),
            last_used_ms: AtomicU64::new(0),
            start: now,
        }
    }

    /// Reserve `n` bytes, returning how long the caller must wait before
    /// the bytes are "paid for" (`ZERO` if tokens covered them).
    fn reserve(&self, n: usize) -> Duration {
        let now = Instant::now();
        self.last_used_ms
            .store(now.duration_since(self.start).as_millis() as u64, Ordering::Relaxed);
        let mut s = self.state.lock().unwrap();
        let elapsed = now.duration_since(s.last).as_secs_f64();
        s.tokens = (s.tokens + elapsed * s.rate).min(s.burst);
        s.last = now;
        s.tokens -= n as f64;
        if s.tokens < 0.0 {
            Duration::from_secs_f64(-s.tokens / s.rate)
        } else {
            Duration::ZERO
        }
    }

    async fn acquire(&self, n: usize) {
        let wait = self.reserve(n);
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
        }
    }

    /// Seconds since the last reservation — input to the idle sweep.
    fn idle(&self) -> Duration {
        let last = self.last_used_ms.load(Ordering::Relaxed);
        let nowm = self.start.elapsed().as_millis() as u64;
        Duration::from_millis(nowm.saturating_sub(last))
    }
}

/// Per-key bucket pair (download + upload), each lazily created only for an
/// axis that is actually limited.
#[derive(Default)]
struct Buckets {
    download: Option<Arc<Bucket>>,
    upload: Option<Arc<Bucket>>,
}

/// The process-wide limiter: the configured caps plus the two lazily
/// populated registries. Cheap to share behind an `Arc`.
pub struct RateLimiter {
    limits: RateLimits,
    /// Per-connection buckets, keyed by remote `IP:port`.
    by_connection: Mutex<HashMap<SocketAddr, Buckets>>,
    /// Per-client buckets, keyed by remote `IP`.
    by_client: Mutex<HashMap<IpAddr, Buckets>>,
    last_sweep: Mutex<Instant>,
}

/// Which way bytes flow — selects the download vs upload cap on each axis.
#[derive(Clone, Copy)]
enum Direction {
    Download,
    Upload,
}

impl RateLimiter {
    /// Build a limiter for `limits`. When `limits.is_unlimited()` the
    /// returned limiter hands out no shapers and touches no registry.
    pub fn new(limits: RateLimits) -> Arc<Self> {
        Arc::new(RateLimiter {
            limits,
            by_connection: Mutex::new(HashMap::new()),
            by_client: Mutex::new(HashMap::new()),
            last_sweep: Mutex::new(Instant::now()),
        })
    }

    /// A shaper for an upload (client→server) from `addr`, or `None` if no
    /// upload axis is limited.
    pub fn upload(&self, addr: SocketAddr) -> Option<Shaper> {
        self.shaper(addr, Direction::Upload)
    }

    /// A shaper for a download (server→client) to `addr`, or `None` if no
    /// download axis is limited.
    pub fn download(&self, addr: SocketAddr) -> Option<Shaper> {
        self.shaper(addr, Direction::Download)
    }

    fn shaper(&self, addr: SocketAddr, dir: Direction) -> Option<Shaper> {
        let (conn_rate, client_rate) = match dir {
            Direction::Download => (self.limits.connection_download, self.limits.client_download),
            Direction::Upload => (self.limits.connection_upload, self.limits.client_upload),
        };
        if conn_rate == 0 && client_rate == 0 {
            return None;
        }
        self.maybe_sweep();
        let mut buckets = Vec::with_capacity(2);
        if conn_rate != 0 {
            buckets.push(bucket_for(&self.by_connection, addr, dir, conn_rate));
        }
        if client_rate != 0 {
            buckets.push(bucket_for(&self.by_client, addr.ip(), dir, client_rate));
        }
        Some(Shaper { buckets })
    }

    /// Wrap a download response body so it is paced to `addr`'s caps. A
    /// no-op (returns `body` unchanged) when no download axis is limited.
    pub fn throttle_download(&self, addr: SocketAddr, body: Body) -> Body {
        match self.download(addr) {
            Some(shaper) => throttle_body(body, shaper),
            None => body,
        }
    }

    /// Opportunistically drop idle, unreferenced buckets so the registries
    /// don't grow without bound under connection churn. Runs at most once
    /// per [`SWEEP_EVERY`]; only entries whose buckets are unreferenced
    /// (`strong_count == 1`, i.e. no live shaper holds them) and idle past
    /// [`SWEEP_IDLE`] are removed, so an in-flight transfer is never
    /// disturbed.
    fn maybe_sweep(&self) {
        {
            let mut last = self.last_sweep.lock().unwrap();
            if last.elapsed() < SWEEP_EVERY {
                return;
            }
            *last = Instant::now();
        }
        sweep(&self.by_connection);
        sweep(&self.by_client);
    }
}

/// Fetch-or-create the bucket for `key` on the given direction, sized to
/// `rate`. The bucket persists across requests on the same key (e.g.
/// sequential keep-alive requests on one connection share it).
fn bucket_for<K: std::hash::Hash + Eq + Clone>(
    map: &Mutex<HashMap<K, Buckets>>,
    key: K,
    dir: Direction,
    rate: u64,
) -> Arc<Bucket> {
    let mut m = map.lock().unwrap();
    let entry = m.entry(key).or_default();
    let slot = match dir {
        Direction::Download => &mut entry.download,
        Direction::Upload => &mut entry.upload,
    };
    slot.get_or_insert_with(|| Arc::new(Bucket::new(rate, Instant::now())))
        .clone()
}

fn sweep<K: std::hash::Hash + Eq + Clone>(map: &Mutex<HashMap<K, Buckets>>) {
    let mut m = map.lock().unwrap();
    m.retain(|_, b| {
        let live = |o: &Option<Arc<Bucket>>| match o {
            // Keep a bucket that a shaper still references, or one used
            // recently enough that a follow-on transfer is plausible.
            Some(arc) => Arc::strong_count(arc) > 1 || arc.idle() < SWEEP_IDLE,
            None => false,
        };
        live(&b.download) || live(&b.upload)
    });
}

/// A handle that paces a single transfer against one or two buckets (the
/// per-connection and/or per-client caps for its direction).
pub struct Shaper {
    buckets: Vec<Arc<Bucket>>,
}

impl Shaper {
    /// Block until `n` bytes are permitted by every governing bucket.
    pub async fn pace(&self, n: usize) {
        for b in &self.buckets {
            b.acquire(n).await;
        }
    }
}

/// Re-stream `body`, splitting each frame into [`PACE_CHUNK`]-sized pieces
/// and pacing each piece through `shaper`. The total byte count and frame
/// contents are unchanged — only the timing is shaped — so `Content-Length`
/// stays correct.
fn throttle_body(body: Body, shaper: Shaper) -> Body {
    let data = body.into_data_stream();
    let stream = futures_util::stream::unfold(
        ThrottleState { data, pending: Bytes::new(), shaper },
        |mut st| async move {
            loop {
                if !st.pending.is_empty() {
                    let take = st.pending.len().min(PACE_CHUNK);
                    let piece = st.pending.split_to(take);
                    st.shaper.pace(piece.len()).await;
                    return Some((Ok::<Bytes, std::io::Error>(piece), st));
                }
                match st.data.next().await {
                    Some(Ok(chunk)) => {
                        st.pending = chunk;
                        // loop: emit the first paced piece of this chunk
                    }
                    Some(Err(e)) => return Some((Err(std::io::Error::other(e)), st)),
                    None => return None,
                }
            }
        },
    );
    Body::from_stream(stream)
}

struct ThrottleState {
    data: axum::body::BodyDataStream,
    pending: Bytes,
    shaper: Shaper,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(s: &str) -> SocketAddr {
        s.parse().unwrap()
    }

    #[test]
    fn unlimited_hands_out_no_shaper() {
        let rl = RateLimiter::new(RateLimits::default());
        assert!(rl.limits.is_unlimited());
        assert!(rl.download(addr("10.0.0.1:5000")).is_none());
        assert!(rl.upload(addr("10.0.0.1:5000")).is_none());
    }

    #[test]
    fn download_axis_creates_only_download_shaper() {
        let rl = RateLimiter::new(RateLimits { connection_download: 1_000_000, ..Default::default() });
        assert!(rl.download(addr("10.0.0.1:5000")).is_some());
        // Upload axis is unlimited → no shaper.
        assert!(rl.upload(addr("10.0.0.1:5000")).is_none());
    }

    /// A bucket meters the rate: reserving more than the burst returns a
    /// wait of about `overflow / rate`.
    #[test]
    fn bucket_meters_overflow() {
        let now = Instant::now();
        // 1 MiB/s, burst clamped to 256 KiB.
        let b = Bucket::new(1024 * 1024, now);
        // First reservation within burst → no wait.
        assert_eq!(b.reserve(64 * 1024), Duration::ZERO);
        // Drain well past burst → must wait ~ (debt / rate).
        let w = b.reserve(1024 * 1024);
        assert!(w > Duration::ZERO, "overdraw must induce a wait");
        // Debt ≈ (64Ki + 1Mi - 256Ki) bytes over 1 MiB/s ≈ 0.81 s.
        assert!(w.as_millis() >= 500 && w.as_millis() <= 1200, "got {w:?}");
    }

    /// Per-connection keys (IP:port) are distinct buckets; per-client keys
    /// (IP) coalesce across ports.
    #[test]
    fn connection_vs_client_keying() {
        let rl = RateLimiter::new(RateLimits {
            connection_download: 1_000_000,
            client_download: 1_000_000,
            ..Default::default()
        });
        let _ = rl.download(addr("10.0.0.1:1111"));
        let _ = rl.download(addr("10.0.0.1:2222"));
        let _ = rl.download(addr("10.0.0.2:1111"));
        // Two distinct ports on .1 plus one on .2 → 3 connection buckets.
        assert_eq!(rl.by_connection.lock().unwrap().len(), 3);
        // .1 (two ports coalesce) and .2 → 2 client buckets.
        assert_eq!(rl.by_client.lock().unwrap().len(), 2);
    }
}
