// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Chunked byte-range transport for downloading data with retry and progress.
//!
//! This module abstracts the mechanics of fetching byte ranges from local or
//! remote sources, with retry logic, connection pooling, and parallel download
//! support. It is used by the cache layer to download merkle-verified chunks.

#![allow(dead_code)] // pub(crate) module — many helpers are kept for Storage and tests
pub mod http;
mod progress;
pub(crate) mod pump;
mod retry;

pub use http::HttpTransport;
pub use progress::DownloadProgress;
pub use retry::RetryPolicy;

use std::io;
use std::sync::OnceLock;
use std::time::Duration;

/// Default number of distinct `reqwest::blocking::Client` instances
/// in the process-wide pool. Each Client wraps its own internal
/// `tokio::runtime::Runtime` built with `new_current_thread()` — one
/// runtime thread per Client — and `Client::clone` shares everything
/// (runtime, connection pool, DNS cache) with the original. So *N
/// workers calling `shared_client()` on a singleton all funnel HTTP
/// completion processing through the same runtime thread*, capping
/// aggregate throughput at whatever one core can drive (TLS decrypt
/// of 8 MiB chunks dominates well before that even matters).
///
/// A pool of separate Clients gives us N parallel runtime threads.
/// Workers picking round-robin spreads decryption across cores, so
/// `download_concurrency` actually translates into multi-stream
/// aggregate throughput instead of capped by one Tokio thread.
///
/// Default parallel-stream fanout — used as the default for both the
/// HTTP runtime pool (one runtime per stream so TLS decryption
/// spreads across cores) and the chunk-download worker pool (one
/// worker per stream so all runtimes stay busy). The 1:1 coupling
/// is load-bearing: with N runtimes and M workers, the per-runtime
/// fanout is M/N, and any imbalance leaves cores idle or
/// re-introduces the single-Tokio-thread bottleneck.
///
/// On hosts with fewer than 32 cores the threads schedule onto the
/// available cores; the overhead of "extra" idle Tokio threads
/// (~MB of stack each, no wakeups when idle) is negligible
/// compared to the throughput floor of one-runtime-per-stream.
///
/// Tunable independently per layer if needed:
///   - `VECTORDATA_HTTP_RUNTIMES`         (transport / runtime pool)
///   - `VECTORDATA_DOWNLOAD_CONCURRENCY`  (cache / chunk worker count)
///
/// Both fall back to this constant when unset, so the default
/// stays matched without two places to update.
pub(crate) const DEFAULT_PARALLEL_STREAMS: usize = 32;

fn http_runtime_count() -> usize {
    std::env::var("VECTORDATA_HTTP_RUNTIMES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_PARALLEL_STREAMS)
}

/// Process-wide pool of `reqwest::blocking::Client` instances.
/// Constructed exactly once on first access. Each entry is a
/// separate Client with a separate internal current-thread Tokio
/// runtime — round-robin pick spreads HTTP I/O + TLS work across
/// the pool's runtime threads so concurrent chunk downloads
/// actually scale on multi-core hosts.
/// The process-wide TLS-trust policy from `settings.yaml`, read once.
fn tls_trust() -> &'static crate::settings::TlsTrust {
    static T: OnceLock<crate::settings::TlsTrust> = OnceLock::new();
    T.get_or_init(crate::settings::tls_trust)
}

/// The secure pool (default) and the insecure pool (built only if an endpoint
/// is configured `trust_self_signed`). Both add any configured `trusted_ca_certs`
/// to their roots; the insecure one additionally skips cert verification.
fn client_pool(insecure: bool) -> &'static [reqwest::blocking::Client] {
    static SECURE: OnceLock<Vec<reqwest::blocking::Client>> = OnceLock::new();
    static INSECURE: OnceLock<Vec<reqwest::blocking::Client>> = OnceLock::new();
    let slot = if insecure { &INSECURE } else { &SECURE };
    slot.get_or_init(|| (0..http_runtime_count()).map(|_| build_client(insecure)).collect())
}

fn build_client(insecure: bool) -> reqwest::blocking::Client {
    let mut b = reqwest::blocking::Client::builder()
        .user_agent(concat!("vectordata/", env!("CARGO_PKG_VERSION")))
        .pool_max_idle_per_host(64)
        .redirect(reqwest::redirect::Policy::limited(10))
        .timeout(Duration::from_secs(60 * 60)); // 1 h ceiling on long-running large fetches
    // Augment the system trust roots with any configured CA/leaf certs (e.g. a
    // private vecd's exported cert). Verification stays ON — this is the secure
    // way to trust a self-signed deployment.
    for path in &tls_trust().trusted_ca_certs {
        match std::fs::read(path).map(|pem| reqwest::Certificate::from_pem(&pem)) {
            Ok(Ok(cert)) => b = b.add_root_certificate(cert),
            Ok(Err(e)) => log::warn!("trusted_ca_certs: {} is not a valid PEM cert: {e}", path.display()),
            Err(e) => log::warn!("trusted_ca_certs: cannot read {}: {e}", path.display()),
        }
    }
    // Only the insecure pool skips verification, and only for endpoints the user
    // explicitly listed in `trust_self_signed` (see `shared_client_for`).
    if insecure {
        b = b.danger_accept_invalid_certs(true);
    }
    b.build().expect("vectordata shared HTTP client")
}

fn pick(pool: &'static [reqwest::blocking::Client]) -> reqwest::blocking::Client {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let idx = COUNTER.fetch_add(1, Ordering::Relaxed) % pool.len();
    pool[idx].clone()
}

/// A round-robin clone of a process-wide **verifying** client. Cheap (the
/// `Client` is `Arc`-wrapped); successive calls spread work across runtime
/// threads. This is the default — use it unless you have the target URL.
pub(crate) fn shared_client() -> reqwest::blocking::Client {
    pick(client_pool(false))
}

/// Like [`shared_client`] but honors the per-endpoint `trust_self_signed` policy:
/// if `url`'s origin is listed there, returns a client that skips cert
/// verification; otherwise the normal verifying client. Use this for any request
/// to a user-configured endpoint so a self-signed local vecd is reachable.
pub(crate) fn shared_client_for(url: &str) -> reqwest::blocking::Client {
    pick(client_pool(trusts_self_signed(url)))
}

/// Whether `url`'s origin is configured (`settings.yaml` `trust_self_signed`) to
/// accept a self-signed/invalid server cert. Origins are compared normalized.
fn trusts_self_signed(url: &str) -> bool {
    let Some(origin) = crate::credentials::origin_of_str(url) else { return false };
    tls_trust()
        .trust_self_signed
        .iter()
        .any(|c| crate::credentials::origin_of_str(c).as_deref() == Some(origin.as_str()))
}

/// Number of independent HTTP runtimes the process-wide client pool
/// is using. Exposed so progress drivers can show the effective
/// "(N runtimes × C concurrent streams)" plan to the user.
pub(crate) fn http_runtimes() -> usize {
    client_pool(false).len()
}

/// Returns `true` for any URL whose transport vectordata speaks
/// directly: `http://`, `https://`, or `s3://`. The `s3://` form
/// is dispatched as a drop-in for HTTPS via
/// [`normalize_remote_url`] — the actual fetch uses S3's
/// virtual-hosted-style HTTPS endpoint, so by the time bytes hit
/// the wire the URL has been rewritten. Anything else is treated
/// as a local filesystem path by callers.
pub(crate) fn is_remote_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://") || s.starts_with("s3://")
}

/// Translate any remote URL to the actual HTTPS form the shared
/// HTTP client should fetch. `http(s)://` URLs pass through;
/// `s3://bucket/key` is rewritten to
/// `https://<bucket>.s3.<region>.amazonaws.com/<key>`.
///
/// Region selection (in priority order):
/// 1. `AWS_REGION` environment variable.
/// 2. `AWS_DEFAULT_REGION` environment variable.
/// 3. `us-east-1` fallback. If the bucket lives elsewhere, S3
///    returns an HTTP 301 with a `Location` header pointing at
///    the correct regional endpoint; the shared `reqwest::Client`
///    follows up to 10 redirects, so the wrong-region default
///    still works at the cost of one extra round trip.
///
/// This is the entire "S3 transport" — anonymous public buckets
/// (the only kind the protected catalog uses) have no auth
/// requirement, so a plain virtual-hosted-style HTTPS GET is all
/// we need. Signed requests / private buckets would require a
/// proper AWS SDK integration; explicitly out of scope here.
pub(crate) fn normalize_remote_url(s: &str) -> std::borrow::Cow<'_, str> {
    if let Some(rest) = s.strip_prefix("s3://") {
        let (bucket, key) = match rest.split_once('/') {
            Some(pair) => pair,
            None => (rest, ""),
        };
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());
        std::borrow::Cow::Owned(format!("https://{bucket}.s3.{region}.amazonaws.com/{key}"))
    } else {
        std::borrow::Cow::Borrowed(s)
    }
}

#[cfg(test)]
mod s3_normalisation_tests {
    use super::*;

    #[test]
    fn s3_url_rewrites_to_virtual_hosted_https() {
        // Pin a region so the test is independent of the
        // surrounding env (`AWS_REGION` may be set in the dev
        // shell). Pin / restore via a guard so concurrent tests
        // don't see the mutation.
        let prev = std::env::var("AWS_REGION").ok();
        unsafe { std::env::set_var("AWS_REGION", "us-east-2") };
        let translated = normalize_remote_url("s3://example-bucket/path/to/file.bin");
        assert_eq!(translated.as_ref(), "https://example-bucket.s3.us-east-2.amazonaws.com/path/to/file.bin");
        // Bucket-only URL (no key path) yields a trailing slash.
        let bare = normalize_remote_url("s3://only-bucket");
        assert_eq!(bare.as_ref(), "https://only-bucket.s3.us-east-2.amazonaws.com/");
        match prev {
            Some(v) => unsafe { std::env::set_var("AWS_REGION", v) },
            None => unsafe { std::env::remove_var("AWS_REGION") },
        }
    }

    #[test]
    fn non_s3_urls_pass_through() {
        let https = normalize_remote_url("https://example.com/x");
        assert_eq!(https.as_ref(), "https://example.com/x");
        let http = normalize_remote_url("http://example.com/x");
        assert_eq!(http.as_ref(), "http://example.com/x");
        let path = normalize_remote_url("/abs/path");
        assert_eq!(path.as_ref(), "/abs/path");
    }

    #[test]
    fn is_remote_url_covers_all_dispatched_schemes() {
        assert!(is_remote_url("http://x/y"));
        assert!(is_remote_url("https://x/y"));
        assert!(is_remote_url("s3://bucket/key"));
        assert!(!is_remote_url("file:///abs"));
        assert!(!is_remote_url("/abs/path"));
        assert!(!is_remote_url("relative/path"));
    }
}

#[cfg(test)]
mod shared_client_tests {
    use super::*;

    /// `client_pool()` must return the same `&'static [Client]`
    /// slice on every call — that's the pool-singleton invariant
    /// that prevents `load_native_certs` from running per request.
    #[test]
    fn client_pool_returns_singleton() {
        let a = client_pool(false).as_ptr();
        let b = client_pool(false).as_ptr();
        assert_eq!(a, b, "client_pool must return the same singleton slice");
        // The secure and insecure pools are distinct singletons.
        assert_ne!(client_pool(false).as_ptr(), client_pool(true).as_ptr());
    }

    /// `shared_client()` rotates through pool entries, but the
    /// observable cost is still negligible — no per-clone cert
    /// load. Many round-robin clones should complete in well
    /// under a second.
    #[test]
    fn shared_client_clone_is_cheap() {
        let _ = shared_client();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            let _ = shared_client();
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 500,
            "10_000 shared_client() calls took {elapsed:?}; if this exceeds \
             500 ms a per-clone load_native_certs has reappeared somewhere");
    }
}

/// Fetch a small remote file to a local path, preserving the remote
/// `Last-Modified` timestamp as the local mtime.
///
/// On subsequent calls, issues a HEAD request to check if the remote
/// file has been modified. Only re-downloads if the remote timestamp
/// is newer than the local copy. Returns `Ok(true)` if the file was
/// downloaded/updated, `Ok(false)` if the local copy is current.
pub fn fetch_if_modified(url: &str, local_path: &std::path::Path) -> io::Result<bool> {
    let client = shared_client();
    let parsed = url::Url::parse(url).ok();

    // If we have a local copy, do a quick HEAD check
    if local_path.exists() {
        let local_mtime = std::fs::metadata(local_path)?
            .modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

        if let Ok(resp) = apply_read_auth(client.head(url), parsed.as_ref()).send()
            && resp.status().is_success()
                && let Some(remote_mtime) = parse_last_modified(&resp)
                    && remote_mtime <= local_mtime {
                        return Ok(false); // local is current
                    }
        // HEAD failed or no Last-Modified — fall through to re-download
    }

    // Download the file
    let resp = apply_read_auth(client.get(url), parsed.as_ref()).send()
        .map_err(io::Error::other)?;
    if !resp.status().is_success() {
        return Err(io::Error::other(
            format!("HTTP {} fetching {}", resp.status(), url),
        ));
    }

    let remote_mtime = parse_last_modified(&resp);
    let bytes = resp.bytes()
        .map_err(io::Error::other)?;

    std::fs::write(local_path, &bytes)?;

    // Set local mtime to match remote Last-Modified
    if let Some(mtime) = remote_mtime {
        let ft = filetime::FileTime::from_system_time(mtime);
        let _ = filetime::set_file_mtime(local_path, ft);
    }

    Ok(true)
}

/// Attach a bearer token resolved by request origin (read-side auth), when
/// one is configured (`$VECTORDATA_TOKEN` or the `vectordata login` store).
/// **Additive**: with no token resolvable — or no parseable URL — the
/// request is returned unchanged, so anonymous/public reads are unaffected.
pub(crate) fn apply_read_auth(
    rb: reqwest::blocking::RequestBuilder,
    url: Option<&url::Url>,
) -> reqwest::blocking::RequestBuilder {
    match url.and_then(crate::credentials::resolve_read_token) {
        Some(token) => rb.bearer_auth(token),
        None => rb,
    }
}

/// Parse `Last-Modified` header into a `SystemTime`.
fn parse_last_modified(resp: &reqwest::blocking::Response) -> Option<std::time::SystemTime> {
    let header = resp.headers().get("last-modified")?.to_str().ok()?;
    // HTTP date format: "Thu, 01 Jan 2025 00:00:00 GMT"
    let parsed = httpdate::parse_http_date(header).ok()?;
    Some(parsed)
}

use semaphore::Semaphore;

/// Byte-range data fetcher — abstracts HTTP vs local file access.
pub trait ChunkedTransport: Send + Sync {
    /// Fetch bytes in range `[start, start+len)` from the resource.
    fn fetch_range(&self, start: u64, len: u64) -> io::Result<Vec<u8>>;

    /// Total size of the resource in bytes.
    fn content_length(&self) -> io::Result<u64>;

    /// Whether the source supports byte-range requests.
    fn supports_range(&self) -> bool;
}

/// A chunk descriptor for parallel download scheduling.
#[derive(Debug, Clone, Copy)]
pub struct ChunkRequest {
    /// Chunk index (for merkle verification).
    pub index: u32,
    /// Byte offset within the content.
    pub start: u64,
    /// Byte length of this chunk.
    pub len: u64,
}

/// Fetch multiple chunks in parallel using a thread pool.
///
/// Returns a vec of `(chunk_index, data)` for successfully fetched chunks.
/// Stops early if the progress tracker signals failure.
pub fn fetch_chunks_parallel(
    transport: &dyn ChunkedTransport,
    chunks: &[ChunkRequest],
    retry_policy: &RetryPolicy,
    progress: &DownloadProgress,
    concurrency: usize,
) -> Vec<io::Result<(u32, Vec<u8>)>> {
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let semaphore = std::sync::Arc::new(Semaphore::new(concurrency));

        for chunk in chunks {
            if progress.is_failed() {
                break;
            }

            let permit = semaphore.clone();
            let chunk = *chunk;

            let handle = scope.spawn(move || {
                let _permit = permit.acquire();

                if progress.is_failed() {
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "download aborted",
                    ));
                }

                let result = retry_policy.execute(|| {
                    transport.fetch_range(chunk.start, chunk.len)
                });

                match &result {
                    Ok(data) => {
                        progress.add_downloaded_bytes(data.len() as u64);
                        progress.increment_completed();
                    }
                    Err(_) => {
                        progress.mark_failed();
                    }
                }

                result.map(|data| (chunk.index, data))
            });

            handles.push(handle);
        }

        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| {
                Err(io::Error::other("thread panicked"))
            }))
            .collect()
    })
}

/// Simple counting semaphore for bounding concurrency.
pub(crate) mod semaphore {
    use std::sync::{Condvar, Mutex};

    pub struct Semaphore {
        state: Mutex<usize>,
        cond: Condvar,
    }

    pub struct SemaphoreGuard<'a> {
        sem: &'a Semaphore,
    }

    impl Semaphore {
        pub fn new(permits: usize) -> Self {
            Semaphore {
                state: Mutex::new(permits),
                cond: Condvar::new(),
            }
        }

        pub fn acquire(&self) -> SemaphoreGuard<'_> {
            let mut count = self.state.lock().unwrap();
            while *count == 0 {
                count = self.cond.wait(count).unwrap();
            }
            *count -= 1;
            SemaphoreGuard { sem: self }
        }
    }

    impl Drop for SemaphoreGuard<'_> {
        fn drop(&mut self) {
            let mut count = self.sem.state.lock().unwrap();
            *count += 1;
            self.sem.cond.notify_one();
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory transport for testing.
    struct MemTransport {
        data: Vec<u8>,
    }

    impl ChunkedTransport for MemTransport {
        fn fetch_range(&self, start: u64, len: u64) -> io::Result<Vec<u8>> {
            let start = start as usize;
            let end = start + len as usize;
            if end > self.data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "range exceeds content",
                ));
            }
            Ok(self.data[start..end].to_vec())
        }

        fn content_length(&self) -> io::Result<u64> {
            Ok(self.data.len() as u64)
        }

        fn supports_range(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_fetch_chunks_parallel_basic() {
        let data = vec![0u8; 4096];
        let transport = MemTransport { data };
        let policy = RetryPolicy::default();
        let progress = DownloadProgress::new(4096, 4);

        let chunks: Vec<ChunkRequest> = (0..4)
            .map(|i| ChunkRequest {
                index: i,
                start: i as u64 * 1024,
                len: 1024,
            })
            .collect();

        let results = fetch_chunks_parallel(&transport, &chunks, &policy, &progress, 4);
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(r.is_ok());
        }
        assert_eq!(progress.completed_chunks(), 4);
        assert_eq!(progress.downloaded_bytes(), 4096);
    }

    #[test]
    fn test_fetch_chunks_parallel_early_abort() {
        // Transport that fails on chunk 1
        struct FailTransport;
        impl ChunkedTransport for FailTransport {
            fn fetch_range(&self, start: u64, _len: u64) -> io::Result<Vec<u8>> {
                if start == 1024 {
                    Err(io::Error::new(io::ErrorKind::ConnectionReset, "boom"))
                } else {
                    Ok(vec![0u8; 1024])
                }
            }
            fn content_length(&self) -> io::Result<u64> { Ok(4096) }
            fn supports_range(&self) -> bool { true }
        }

        let transport = FailTransport;
        let policy = RetryPolicy { max_retries: 1, base_delay_ms: 1, max_delay_ms: 1, jitter_fraction: 0.0 };
        let progress = DownloadProgress::new(4096, 4);

        let chunks: Vec<ChunkRequest> = (0..4)
            .map(|i| ChunkRequest {
                index: i,
                start: i as u64 * 1024,
                len: 1024,
            })
            .collect();

        let results = fetch_chunks_parallel(&transport, &chunks, &policy, &progress, 2);
        assert!(progress.is_failed());
        // At least one result should be an error
        assert!(results.iter().any(|r| r.is_err()));
    }
}
