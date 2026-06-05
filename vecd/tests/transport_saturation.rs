// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Transport **saturation** scenarios — opt-in operational tests gated behind
//! the `extended-tests` feature (run with `cargo test -p vecd --features
//! extended-tests` or the workspace `cargo ta` alias; excluded from a normal
//! `cargo test`). They extend the vecd-end-to-end flow (push → read) with a
//! per-connection bandwidth cap to prove the client scales across many
//! connections when *each* connection is rate-limited but the transport as a
//! whole is not:
//!
//! - **Upload** rides parallel sparse `PATCH` chunks (vecd's resumable path),
//!   so one large object saturates aggregate bandwidth across `concurrency`
//!   connections.
//! - **Download** rides parallel ranged `GET`s (vecd's `Range` support + the
//!   client's chunked prebuffer), scaled by `VECTORDATA_DOWNLOAD_CONCURRENCY`.
//! - **Resume** after a mid-flight connection drop re-sends only the chunks
//!   that were never acknowledged (not the whole remainder), and a dropped
//!   download connection is retried in place.
//!
//! The harness is a transparent TCP proxy in front of vecd that rate-limits
//! each connection (and can drop one connection once); the client points at
//! the proxy. No production code has test-only hooks.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use vecd::admin;
use vecd::db::Db;
use vecd::model::{Level, Listable};
use vecd::ratelimit::RateLimits;
use vecd::server::{self, AppState};

/// Serializes the env-mutating download scenarios (`VECTORDATA_HOME` /
/// `VECTORDATA_DOWNLOAD_CONCURRENCY` are process-global) so they never race
/// each other when the harness runs tests in parallel.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

use vectordata::push::transport::TransportOptions;
use vectordata::push::{execute, ChecksumPolicy, Options};

// ───────────────────────── in-process vecd ─────────────────────────

struct Vecd {
    addr: SocketAddr,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Vecd {
    fn start(db: Db) -> Self {
        Self::start_with_limits(db, RateLimits::default())
    }
    fn start_with_limits(db: Db, limits: RateLimits) -> Self {
        let state = AppState::new(db).unwrap().with_rate_limits(limits);
        let (addr_tx, addr_rx) = mpsc::channel();
        let (sd_tx, sd_rx) = tokio::sync::oneshot::channel::<()>();
        let thread = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                addr_tx.send(listener.local_addr().unwrap()).unwrap();
                server::serve_listener(state, listener, async move {
                    let _ = sd_rx.await;
                })
                .await
                .unwrap();
            });
        });
        let addr = addr_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        Vecd { addr, shutdown: Some(sd_tx), thread: Some(thread) }
    }
}
impl Drop for Vecd {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

// ────────────────────── throttling TCP proxy ───────────────────────

/// A transparent TCP proxy that rate-limits **each** connection to
/// `per_conn_bytes_per_sec` in each direction, while imposing no aggregate
/// cap (unlimited concurrent connections). It tallies bytes forwarded in the
/// upload (client→backend) direction, and can drop the first connection once
/// after it has forwarded `drop_after` upload bytes (to force a client
/// resume). Dropping `shutdown` stops accepting new connections.
struct ThrottleProxy {
    addr: SocketAddr,
    uploaded: Arc<AtomicU64>,
    downloaded: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
}

struct ProxyConfig {
    backend: SocketAddr,
    /// Per-connection, per-direction cap. `u64::MAX` ≈ unlimited.
    per_conn_bytes_per_sec: u64,
    /// If > 0, the first connection is force-closed once it has relayed this
    /// many upload bytes — simulating a mid-flight drop.
    drop_first_after: u64,
}

impl ThrottleProxy {
    fn start(cfg: ProxyConfig) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let uploaded = Arc::new(AtomicU64::new(0));
        let downloaded = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let conn_seq = Arc::new(AtomicUsize::new(0));

        let cfg = Arc::new(cfg);
        let up = uploaded.clone();
        let down = downloaded.clone();
        let st = stop.clone();
        listener.set_nonblocking(true).unwrap();
        thread::spawn(move || {
            loop {
                if st.load(Ordering::Relaxed) {
                    return;
                }
                match listener.accept() {
                    Ok((client, _)) => {
                        let cfg = cfg.clone();
                        let up = up.clone();
                        let down = down.clone();
                        let n = conn_seq.fetch_add(1, Ordering::Relaxed);
                        thread::spawn(move || handle_conn(client, cfg, up, down, n));
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => return,
                }
            }
        });
        ThrottleProxy { addr, uploaded, downloaded, stop }
    }

    fn base_url(&self) -> String {
        format!("http://{}/", self.addr)
    }
    fn uploaded_bytes(&self) -> u64 {
        self.uploaded.load(Ordering::Relaxed)
    }
    fn downloaded_bytes(&self) -> u64 {
        self.downloaded.load(Ordering::Relaxed)
    }
}
impl Drop for ThrottleProxy {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn handle_conn(
    client: TcpStream,
    cfg: Arc<ProxyConfig>,
    uploaded: Arc<AtomicU64>,
    downloaded: Arc<AtomicU64>,
    conn_index: usize,
) {
    let Ok(backend) = TcpStream::connect(cfg.backend) else { return };
    client.set_nodelay(true).ok();
    backend.set_nodelay(true).ok();

    let drop_after = if conn_index == 0 { cfg.drop_first_after } else { 0 };

    // client → backend (upload): counted, and optionally drop-after.
    let c2b_client = client.try_clone().unwrap();
    let c2b_backend = backend.try_clone().unwrap();
    let rate = cfg.per_conn_bytes_per_sec;
    let up = uploaded.clone();
    let up_thread = thread::spawn(move || {
        pipe(c2b_client, c2b_backend, rate, Some(up), drop_after);
    });

    // backend → client (download): counted.
    pipe(backend, client, rate, Some(downloaded), 0);
    let _ = up_thread.join();
}

/// Token-bucket burst per connection/direction. A connection that sits idle
/// (e.g. while the worker driving it is busy on other connections, or between
/// keep-alive requests) banks at most this much capacity, after which it is
/// shaped to `rate`. Bounding the burst is the whole point: a `start`-relative
/// shaper banks *unlimited* idle credit, so a connection opened early but used
/// late delivers its whole payload unthrottled — which let a 1-worker download
/// that round-robins across many keep-alive clients run at aggregate speed and
/// defeated the per-connection cap. 256 KiB ≈ a real link's small burst
/// tolerance while keeping idle banking negligible against a 32 MiB object.
const BURST_BYTES: u64 = 256 * 1024;

/// Copy `from` → `to`, shaping throughput to `rate` bytes/sec via a
/// **token bucket** (refill = `rate`, capacity = [`BURST_BYTES`]). If `counter`
/// is set, every relayed byte is tallied. If `drop_after > 0`, the copy stops
/// (closing both halves) once that many bytes have been relayed.
fn pipe(mut from: TcpStream, mut to: TcpStream, rate: u64, counter: Option<Arc<AtomicU64>>, drop_after: u64) {
    let mut buf = vec![0u8; 64 * 1024];
    // Token bucket: `tokens` is available byte-budget, refilled at `rate` and
    // capped at `BURST_BYTES`, so idle time accrues at most one burst. May go
    // negative (a debt the next refill pays down) — we sleep off the deficit.
    let mut tokens: f64 = BURST_BYTES as f64;
    let mut last = Instant::now();
    let mut relayed: u64 = 0;
    loop {
        let n = match from.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        if to.write_all(&buf[..n]).is_err() {
            break;
        }
        relayed += n as u64;
        if let Some(c) = &counter {
            c.fetch_add(n as u64, Ordering::Relaxed);
        }
        // Force the drop mid-flight to exercise resume.
        if drop_after > 0 && relayed >= drop_after {
            let _ = to.shutdown(std::net::Shutdown::Both);
            let _ = from.shutdown(std::net::Shutdown::Both);
            break;
        }
        // Rate-shape against the bucket. Refill for wall-clock elapsed (capped
        // at one burst so idle banking can't defeat the cap), spend `n`, and
        // sleep off any deficit. The sleep is undone by the next refill, so the
        // average rate converges to `rate` regardless of read sizes or gaps.
        if rate != u64::MAX {
            let now = Instant::now();
            tokens = (tokens + now.duration_since(last).as_secs_f64() * rate as f64)
                .min(BURST_BYTES as f64);
            last = now;
            tokens -= n as f64;
            if tokens < 0.0 {
                thread::sleep(Duration::from_secs_f64(-tokens / rate as f64));
            }
        }
    }
    let _ = to.flush();
    // Half-close so the peer sees EOF and the HTTP exchange can complete.
    let _ = to.shutdown(std::net::Shutdown::Write);
}

// ─────────────────────────── fixtures ──────────────────────────────

const PER_CONN_BPS: u64 = 8 * 1024 * 1024; // 8 MiB/s/connection cap.
const STREAMS: u32 = 8;
const DIM: usize = 128;
const VEC_BYTES: usize = 4 + DIM * 4; // i32 dim header + DIM f32s
const FACET: &str = "base_vectors.fvec";

/// A ~32 MiB valid `.fvec` (≈65 000 dim-128 vectors) — large enough to span
/// many upload (4 MiB) and download (8 MiB) chunks. Deterministic so the
/// resume read-back can byte-compare.
fn fvec_blob() -> Vec<u8> {
    let target = 32 * 1024 * 1024;
    let n = target / VEC_BYTES;
    let mut out = Vec::with_capacity(n * VEC_BYTES);
    for i in 0..n {
        out.extend_from_slice(&(DIM as i32).to_le_bytes());
        for j in 0..DIM {
            out.extend_from_slice(&(((i * DIM + j) % 997) as f32).to_le_bytes());
        }
    }
    out
}

/// Facet transport the reader will select for a dataset, decided purely by
/// whether a `.mref` merkle sidecar is published next to the facet file.
#[derive(Clone, Copy)]
enum FacetMode {
    /// Publish a `.mref` → reader takes the chunked, merkle-**verified**,
    /// parallel-prebuffer `Storage::Cached` path (1 MiB merkle chunks).
    Mref,
    /// No `.mref` → reader takes the chunked, **unverified** (TLS-trusted)
    /// parallel-prebuffer `Storage::Http` / `ChunkStore` path (8 MiB chunks).
    /// Proves chunked streaming + saturation is afforded to non-mref modes.
    Http,
}

/// A dataset dir: a `dataset.yaml` declaring one `base_vectors` facet, the
/// facet file, and — for [`FacetMode::Mref`] — a `.mref` merkle sidecar. The
/// presence of the sidecar is the *only* thing that steers the reader between
/// the `Storage::Cached` and `Storage::Http` parallel-prebuffer paths.
fn write_dataset(dir: &std::path::Path, data: &[u8], mode: FacetMode) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("dataset.yaml"),
        format!("name: sat\nattributes:\n  distance_function: L2\nprofiles:\n  default:\n    base_vectors: {FACET}\n"),
    )
    .unwrap();
    std::fs::write(dir.join(FACET), data).unwrap();
    if let FacetMode::Mref = mode {
        // 1 MiB merkle chunks → ~32 chunks for the 32 MiB facet, plenty to
        // spread across STREAMS parallel ranged GETs.
        let mref = vectordata::merkle::MerkleRef::from_content(data, 1024 * 1024);
        std::fs::write(dir.join(format!("{FACET}.mref")), mref.to_bytes()).unwrap();
    }
}

fn push_opts(src: &std::path::Path, to: String, token: &str, concurrency: u32) -> Options {
    Options {
        path: src.to_path_buf(),
        to: Some(to),
        message: None,
        raw: false,
        checksums: ChecksumPolicy::Auto,
        dry_run: false,
        no_check: true,
        assume_yes: true,
        delete: false,
        abort_incomplete: false,
        concurrency,
        files: None,
        transport: TransportOptions { token: Some(token.to_string()), profile: None, endpoint_url: None },
        cmd: "push".into(),
        actor: "sat".into(),
    }
}

/// Build the vecd DB with a public `datasets` namespace + a push token.
fn build_db() -> (Db, String) {
    let dbdir = tempfile::tempdir().unwrap();
    let mut db = Db::init(&dbdir.path().join("vecd.db")).unwrap();
    std::mem::forget(dbdir); // keep the DB dir alive for the test process
    admin::add_backend(&mut db, "store", "mem", "mem:saturation", None, None, None, true).unwrap();
    admin::add_user(&mut db, "alice", Level::User, None, None).unwrap();
    admin::add_namespace(&mut db, "datasets", "alice", Some("store"), true, Listable::Grantees, None, None).unwrap();
    admin::bind(&mut db, "alice", "curate", "datasets").unwrap();
    admin::bind(&mut db, "PUBLIC", "reader", "datasets").unwrap();
    let token = admin::create_token(&mut db, "alice", "key", Some("30d"), None).unwrap().plaintext;
    (db, token)
}

/// Stand up vecd with a public namespace + a push token, behind a
/// per-connection-throttling proxy (the proxy shapes bandwidth; vecd runs
/// unthrottled).
fn fixture(drop_first_after: u64) -> (Vecd, ThrottleProxy, String) {
    let (db, token) = build_db();
    let vecd = Vecd::start(db);
    let proxy = ThrottleProxy::start(ProxyConfig {
        backend: vecd.addr,
        per_conn_bytes_per_sec: PER_CONN_BPS,
        drop_first_after,
    });
    (vecd, proxy, token)
}

/// Stand up vecd with its own bandwidth rate limits and **no** proxy — the
/// client connects straight to vecd, so vecd's limiter is what shapes the
/// transfer. Used to prove the per-connection vs per-client distinction.
fn fixture_rate_limited(limits: RateLimits) -> (Vecd, String) {
    let (db, token) = build_db();
    (Vecd::start_with_limits(db, limits), token)
}

fn secs(d: Duration) -> f64 {
    d.as_secs_f64()
}

// ─────────────────────────── scenarios ─────────────────────────────

#[test]
fn upload_saturates_beyond_one_capped_connection() {
    let (_vecd, proxy, token) = fixture(0);
    let data = fvec_blob();
    // A fresh source dir per push: the push engine records a `.publish_url`
    // binding in the source, so the same dir can't target two namespaces.
    let seq_src = tempfile::tempdir().unwrap();
    let par_src = tempfile::tempdir().unwrap();
    write_dataset(seq_src.path(), &data, FacetMode::Mref);
    write_dataset(par_src.path(), &data, FacetMode::Mref);

    // Sequential: one connection, capped at PER_CONN_BPS.
    let t0 = Instant::now();
    execute(&push_opts(seq_src.path(), format!("{}datasets/seq/", proxy.base_url()), &token, 1))
        .expect("sequential push");
    let seq = t0.elapsed();

    // Parallel: STREAMS connections, each capped — aggregate should scale.
    let t1 = Instant::now();
    execute(&push_opts(par_src.path(), format!("{}datasets/par/", proxy.base_url()), &token, STREAMS))
        .expect("parallel push");
    let par = t1.elapsed();

    let speedup = secs(seq) / secs(par);
    eprintln!(
        "upload: seq={:.2}s par(x{})={:.2}s speedup={:.1}x",
        secs(seq), STREAMS, secs(par), speedup
    );
    assert!(
        speedup >= 2.0,
        "parallel upload must scale beyond one capped connection (got {speedup:.1}x: seq {:.2}s, par {:.2}s)",
        secs(seq), secs(par)
    );
}

#[test]
fn download_saturates_beyond_one_capped_connection() {
    // Both download transports — merkle-verified `Storage::Cached` (`.mref`)
    // and TLS-trusted `Storage::Http` (no `.mref`) — share one parallel
    // chunk-download driver, so both must scale across connections. Covering
    // both in a single test keeps the process-global env mutation
    // (VECTORDATA_HOME / VECTORDATA_DOWNLOAD_CONCURRENCY, read fresh per open)
    // in one place so it can't race the env-free upload/resume scenarios.
    let _env = ENV_LOCK.lock().unwrap();
    let (_vecd, proxy, token) = fixture(0);
    let data = fvec_blob();

    // Isolate the client cache under a throwaway home (see config_isolation).
    let home = tempfile::tempdir().unwrap();
    unsafe { std::env::set_var("VECTORDATA_HOME", home.path()) };

    let precache = |ns: &str, concurrency: u32| -> Duration {
        unsafe { std::env::set_var("VECTORDATA_DOWNLOAD_CONCURRENCY", concurrency.to_string()) };
        let url = format!("{}datasets/{ns}/dataset.yaml", proxy.base_url());
        let t = Instant::now();
        let code = vectordata::datasets::precache::run(&url, "", &[], &[], None);
        assert_eq!(code, 0, "precache of {ns} failed");
        t.elapsed()
    };

    for (mode, label) in [(FacetMode::Mref, "mref/Cached"), (FacetMode::Http, "http/ChunkStore")] {
        // Push the same dataset to two namespaces so each download is a fresh
        // (uncached) fetch — a fresh source dir per push (the `.publish_url`
        // binding is per-source). A `_h` suffix keeps the Http run's
        // namespaces (and client cache keys) distinct from the mref run's.
        let suffix = match mode { FacetMode::Mref => "", FacetMode::Http => "_h" };
        let seq_ns = format!("dl_seq{suffix}");
        let par_ns = format!("dl_par{suffix}");
        for ns in [&seq_ns, &par_ns] {
            let s = tempfile::tempdir().unwrap();
            write_dataset(s.path(), &data, mode);
            execute(&push_opts(s.path(), format!("{}datasets/{ns}/", proxy.base_url()), &token, STREAMS))
                .expect("seed push");
        }

        let before = proxy.downloaded_bytes();
        let seq = precache(&seq_ns, 1);
        let par = precache(&par_ns, STREAMS);
        let relayed = proxy.downloaded_bytes() - before;

        let speedup = secs(seq) / secs(par);
        eprintln!(
            "download[{label}]: seq={:.2}s par(x{})={:.2}s speedup={:.1}x  (proxy relayed {} MiB downstream)",
            secs(seq), STREAMS, secs(par), speedup, relayed / (1024 * 1024)
        );
        assert!(
            speedup >= 2.0,
            "[{label}] parallel download must scale beyond one capped connection \
             (got {speedup:.1}x: seq {:.2}s, par {:.2}s)",
            secs(seq), secs(par)
        );
    }

    unsafe {
        std::env::remove_var("VECTORDATA_HOME");
        std::env::remove_var("VECTORDATA_DOWNLOAD_CONCURRENCY");
    }
}

#[test]
fn vecd_rate_limits_distinguish_per_connection_from_per_client() {
    // Drive vecd's OWN bandwidth limiter (no proxy — the client connects
    // straight to vecd). The contrast is the whole point of the feature:
    //   • a per-CONNECTION cap shapes each TCP connection, so fanning out
    //     across more connections multiplies aggregate throughput —
    //     auto-concurrent streaming is *effective*.
    //   • a per-CLIENT cap shapes the sum across one host's connections, so
    //     opening more connections buys nothing past the cap.
    // Same process-global env mutation as the proxy download test →
    // serialize via ENV_LOCK.
    let _env = ENV_LOCK.lock().unwrap();
    let data = fvec_blob();
    const CAP: u64 = 16 * 1024 * 1024; // 16 MiB/s

    // Stand up a vecd with `limits`, seed two namespaces, then return the
    // ratio of a 1-connection download to a STREAMS-connection download. A
    // fresh client HOME keeps both legs uncached.
    let measure = |limits: RateLimits| -> f64 {
        let (vecd, token) = fixture_rate_limited(limits);
        let base = format!("http://{}/", vecd.addr);
        for ns in ["rl_seq", "rl_par"] {
            let s = tempfile::tempdir().unwrap();
            write_dataset(s.path(), &data, FacetMode::Mref);
            execute(&push_opts(s.path(), format!("{base}datasets/{ns}/"), &token, STREAMS))
                .expect("seed push");
        }
        let home = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("VECTORDATA_HOME", home.path()) };
        let precache = |ns: &str, concurrency: u32| -> Duration {
            unsafe { std::env::set_var("VECTORDATA_DOWNLOAD_CONCURRENCY", concurrency.to_string()) };
            let url = format!("{base}datasets/{ns}/dataset.yaml");
            let t = Instant::now();
            assert_eq!(
                vectordata::datasets::precache::run(&url, "", &[], &[], None),
                0,
                "precache of {ns} failed"
            );
            t.elapsed()
        };
        let seq = precache("rl_seq", 1);
        let par = precache("rl_par", STREAMS);
        unsafe {
            std::env::remove_var("VECTORDATA_HOME");
            std::env::remove_var("VECTORDATA_DOWNLOAD_CONCURRENCY");
        }
        secs(seq) / secs(par)
    };

    let per_connection = measure(RateLimits { connection_download: CAP, ..Default::default() });
    let per_client = measure(RateLimits { client_download: CAP, ..Default::default() });

    eprintln!(
        "vecd rate-limit: per-connection cap → speedup={per_connection:.1}x (concurrency helps); \
         per-client cap → speedup={per_client:.1}x (concurrency bounded)"
    );
    assert!(
        per_connection >= 2.0,
        "a per-connection cap must let auto-concurrent streaming scale beyond one connection \
         (got {per_connection:.1}x)"
    );
    assert!(
        per_client < 1.6,
        "a per-client cap must bound aggregate throughput regardless of connection count \
         (got {per_client:.1}x)"
    );
}

#[test]
fn resume_resends_only_unacked_chunks_after_a_drop() {
    // Drop the first connection once, partway through the upload, forcing the
    // client to resume. With per-chunk ack tracking it should re-send only the
    // chunk(s) lost on that connection — not the whole remainder — so the
    // total bytes the proxy relays upstream stays close to the object size.
    let data = fvec_blob();
    let obj_bytes = data.len() as u64;
    let (_vecd, proxy, token) = fixture(obj_bytes / 4);
    let src = tempfile::tempdir().unwrap();
    write_dataset(src.path(), &data, FacetMode::Mref);

    execute(&push_opts(src.path(), format!("{}datasets/resumed/", proxy.base_url()), &token, STREAMS))
        .expect("push survives a mid-flight connection drop");

    // Re-sends are bounded: the object plus a small overhead (the dropped
    // chunk re-sent, control files, headers) — NOT a whole second copy.
    let uploaded = proxy.uploaded_bytes();
    let budget = obj_bytes + 12 * 1024 * 1024; // object + a few 4 MiB chunks + overhead
    eprintln!(
        "resume: object={} MiB, relayed-upstream={} MiB (budget {} MiB)",
        obj_bytes / (1024 * 1024),
        uploaded / (1024 * 1024),
        budget / (1024 * 1024),
    );
    assert!(
        uploaded < budget,
        "resume re-sent too much: relayed {uploaded} bytes for a {obj_bytes}-byte object \
         (budget {budget}) — only the unacked chunks should be re-sent"
    );

    // And the object is intact: read it back through the proxy and compare.
    let url = format!("{}datasets/resumed/{FACET}", proxy.base_url());
    let got = fetch_all(&url, &token);
    assert_eq!(got.len(), data.len(), "resumed object truncated");
    assert!(got == data, "resumed object content mismatch");
}

/// Plain full GET of a URL (through the proxy), bearer-authenticated.
fn fetch_all(url: &str, token: &str) -> Vec<u8> {
    let client = reqwest::blocking::Client::new();
    let resp = client.get(url).bearer_auth(token).send().expect("GET");
    assert!(resp.status().is_success(), "GET {url} -> {}", resp.status());
    resp.bytes().expect("body").to_vec()
}
