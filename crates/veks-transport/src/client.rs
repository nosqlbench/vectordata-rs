// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The client: [`Transport`], one configured pool of HTTP clients with
//! its TLS trust, offline gate, user agent, timeouts and read
//! credentials, and the [`Response`] a sent [`Request`] answers with.
//!
//! A transport is built from a [`TransportConfig`] value and a
//! [`Credentials`] implementation, and reads nothing else: no
//! environment variable, no settings file (`srd-crate-boundaries.md`
//! TR-3, TR-4). Cloning it is cheap and shares the pool.
//!
//! ```
//! use veks_apicore::transport::{Request, TransportConfig};
//! use veks_transport::Transport;
//!
//! let offline = Transport::new(TransportConfig { offline: true, ..TransportConfig::default() });
//! let err = offline.send(Request::get("https://data.example/x.fvecs")).unwrap_err();
//! assert!(err.is_offline());
//! assert!(err.to_string().contains("offline mode is on"), "{err}");
//! ```

use std::borrow::Cow;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use reqwest::blocking::{Client, RequestBuilder};
use veks_apicore::transport::{Authorization, Body, Credentials, Method, NoCredentials, Request, TransportConfig};
use veks_apicore::{TransportError, TransportErrorKind};

pub use veks_apicore::location::is_remote_url;

/// A configured HTTP transport: a pool of clients, the TLS trust, the
/// offline gate, the user agent and timeouts, and the read credentials.
///
/// The pool holds [`TransportConfig::parallel_streams`] separate
/// clients. Each blocking client wraps its own current-thread runtime,
/// and clones of one client share it, so N workers on one client
/// funnel their completion processing — TLS decryption of every chunk —
/// through one thread. Requests pick clients round-robin, which spreads
/// that work over the pool's threads, so a parallel download actually
/// scales with its stream count.
///
/// Two pools exist: the verifying one every request uses, and — built
/// only when an origin is listed in [`TlsTrust::trust_self_signed`](veks_apicore::transport::TlsTrust::trust_self_signed)
/// and asked for — one that accepts that origin's certificate without
/// verification. Both add [`TlsTrust::trusted_ca_certs`](veks_apicore::transport::TlsTrust::trusted_ca_certs)
/// to their roots. Every request picks its pool by its own origin.
#[derive(Clone)]
pub struct Transport {
    inner: Arc<Inner>,
}

struct Inner {
    config: TransportConfig,
    credentials: Arc<dyn Credentials>,
    /// The origins of `config.tls.trust_self_signed`, normalized once.
    self_signed: Vec<String>,
    secure: OnceLock<Result<Vec<Client>, String>>,
    insecure: OnceLock<Result<Vec<Client>, String>>,
    next: AtomicUsize,
}

impl std::fmt::Debug for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transport").field("config", &self.inner.config).finish_non_exhaustive()
    }
}

impl Transport {
    /// A transport configured by `config`, whose reads are anonymous.
    pub fn new(config: TransportConfig) -> Self {
        Self::with_credentials(config, Arc::new(NoCredentials))
    }

    /// A transport configured by `config`, whose reads carry the token
    /// `credentials` gives for their URL.
    pub fn with_credentials(config: TransportConfig, credentials: Arc<dyn Credentials>) -> Self {
        let self_signed = config.tls.trust_self_signed.iter().filter_map(|o| origin_of(o)).collect();
        Transport {
            inner: Arc::new(Inner {
                config,
                credentials,
                self_signed,
                secure: OnceLock::new(),
                insecure: OnceLock::new(),
                next: AtomicUsize::new(0),
            }),
        }
    }

    /// The configuration this transport was built with.
    pub fn config(&self) -> &TransportConfig {
        &self.inner.config
    }

    /// How many independent clients requests are spread over.
    pub fn parallel_streams(&self) -> usize {
        self.inner.config.parallel_streams.max(1)
    }

    /// Workers a chunked download runs at once (at least one).
    pub fn download_concurrency(&self) -> usize {
        self.inner.config.download_concurrency.max(1)
    }

    /// Whether offline mode is on.
    pub fn is_offline(&self) -> bool {
        self.inner.config.offline
    }

    /// Refuse a request to `url` in offline mode, naming what is missing
    /// and the ways out. Called at every point a read would talk to a
    /// server; everything that has a local copy is served before
    /// reaching one. The error carries a [`TransportError`] of kind
    /// [`Offline`](TransportErrorKind::Offline), which no retry
    /// repeats.
    pub fn ensure_online(&self, url: &str) -> io::Result<()> {
        if self.inner.config.offline {
            return Err(io::Error::new(io::ErrorKind::NotConnected, self.offline_refusal(url)));
        }
        Ok(())
    }

    /// The error offline mode refuses a request to `url` with.
    fn offline_refusal(&self, url: &str) -> TransportError {
        let how = match &self.inner.config.offline_hint {
            Some(hint) => format!("offline mode is on ({hint})"),
            None => "offline mode is on".to_string(),
        };
        TransportError {
            url: Some(url.to_string()),
            status: None,
            kind: TransportErrorKind::Offline,
            message: format!(
                "{how}: {url} is not available locally. Fetch it while online, or turn offline mode off."
            ),
        }
    }

    /// `url` in the form the wire takes: `http(s)://` as given, and
    /// `s3://bucket/key` as its virtual-hosted HTTPS endpoint in the
    /// configured region (see [`normalize_remote_url`]).
    pub fn normalize_url<'a>(&self, url: &'a str) -> Cow<'a, str> {
        normalize_remote_url(url, &self.inner.config.s3_region)
    }

    /// Send `req` and return the server's answer, whatever its status.
    ///
    /// Offline mode refuses it unless it is [`explicit`](Request::explicit).
    /// An `s3://` URL is sent to its HTTPS endpoint. Its authorisation
    /// is its own ([`Request::auth`]); by default the read credentials,
    /// asked with the URL as the request names it. An error is a request
    /// that got no answer — a refused or timed-out connection, an
    /// untrusted certificate, a body that could not be sent.
    pub fn send(&self, req: Request) -> Result<Response, TransportError> {
        if self.inner.config.offline && !req.explicit {
            return Err(self.offline_refusal(&req.url));
        }
        let mut req = req;
        let mut wire = self.normalize_url(&req.url).into_owned();
        if let Authorization::Signed(signer) = &req.auth {
            // A signature covers the host and path the wire sees, so
            // the signer is handed the request in its wire form, and
            // what it signed is what is sent.
            let signer = Arc::clone(signer);
            req.url = wire;
            signer.sign(&mut req)?;
            wire = req.url.clone();
        }
        let parsed = url::Url::parse(&wire).map_err(|e| permanent(&req.url, format!("invalid URL '{}': {e}", req.url)))?;
        let client = self.client_for(&wire).map_err(|e| permanent(&req.url, e.to_string()))?;
        let method = match req.method {
            Method::Get => reqwest::Method::GET,
            Method::Head => reqwest::Method::HEAD,
            Method::Put => reqwest::Method::PUT,
            Method::Post => reqwest::Method::POST,
            Method::Patch => reqwest::Method::PATCH,
            Method::Delete => reqwest::Method::DELETE,
        };
        let mut rb = client.request(method, parsed);
        for (name, value) in &req.headers {
            rb = rb.header(name.as_str(), value.as_str());
        }
        rb = match std::mem::take(&mut req.body) {
            Body::Empty => rb,
            Body::Bytes(bytes) => rb.body(bytes),
            Body::File(path) => {
                let file = std::fs::File::open(&path)
                    .map_err(|e| permanent(&req.url, format!("{}: {e}", path.display())))?;
                rb.body(reqwest::blocking::Body::from(file))
            }
        };
        if let Some(t) = req.timeout {
            rb = rb.timeout(t);
        }
        rb = match &req.auth {
            Authorization::ReadCredentials => self.apply_read_auth(rb, &req.url),
            Authorization::Bearer(token) => rb.bearer_auth(token),
            Authorization::Anonymous | Authorization::Signed(_) => rb,
        };
        let inner = rb.send().map_err(transport_error)?;
        Ok(Response { inner, url: req.url })
    }

    /// A small remote file as text, with the read credentials: what a
    /// caller reads beside a dataset (a catalog, a list). An answer that
    /// is not a success is an error naming its status.
    pub fn fetch_text(&self, url: &str) -> io::Result<String> {
        let parsed =
            url::Url::parse(url).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("{url}: {e}")))?;
        let client = self.client_for(url)?;
        let resp = self.apply_read_auth(client.get(parsed), url).send().map_err(io::Error::other)?;
        if !resp.status().is_success() {
            return Err(io::Error::other(format!("HTTP {} fetching {url}", resp.status())));
        }
        resp.text().map_err(io::Error::other)
    }

    /// Fetch a small remote file to `local_path`, keeping the server's
    /// `Last-Modified` as the local mtime.
    ///
    /// With a local copy, a `HEAD` asks whether the server's copy is
    /// newer, and only then is it fetched again. `Ok(true)` when the
    /// file was fetched, `Ok(false)` when the local copy is current.
    /// Every request goes through the client for `url`'s origin, so a
    /// self-signed origin the configuration trusts is reachable here as
    /// everywhere (`srd-crate-boundaries.md` TR-12).
    pub fn fetch_if_modified(&self, url: &str, local_path: &std::path::Path) -> io::Result<bool> {
        let client = self.client_for(url)?;

        // With a local copy, a quick HEAD.
        if local_path.exists() {
            let local_mtime = std::fs::metadata(local_path)?.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            if let Ok(resp) = self.apply_read_auth(client.head(url), url).send()
                && resp.status().is_success()
                && let Some(remote_mtime) = last_modified(resp.headers())
                && remote_mtime <= local_mtime
            {
                return Ok(false); // the local copy is current
            }
            // HEAD failed or no Last-Modified: fetch again.
        }

        let resp = self.apply_read_auth(client.get(url), url).send().map_err(io::Error::other)?;
        if !resp.status().is_success() {
            return Err(io::Error::other(format!("HTTP {} fetching {}", resp.status(), url)));
        }
        let remote_mtime = last_modified(resp.headers());
        let bytes = resp.bytes().map_err(io::Error::other)?;
        std::fs::write(local_path, &bytes)?;
        if let Some(mtime) = remote_mtime {
            let ft = filetime::FileTime::from_system_time(mtime);
            let _ = filetime::set_file_mtime(local_path, ft);
        }
        Ok(true)
    }

    /// A round-robin clone of a client for `url`: the one that accepts a
    /// self-signed certificate when `url`'s origin is listed in
    /// `trust_self_signed`, the verifying one otherwise. Cheap (a client
    /// is shared state behind an `Arc`).
    pub(crate) fn client_for(&self, url: &str) -> io::Result<Client> {
        self.pick(self.trusts_self_signed(url))
    }

    /// A round-robin clone of a verifying client.
    #[cfg(test)]
    pub(crate) fn client(&self) -> io::Result<Client> {
        self.pick(false)
    }

    fn pick(&self, insecure: bool) -> io::Result<Client> {
        let pool = self.pool(insecure).map_err(|e| io::Error::other(e.to_string()))?;
        let idx = self.inner.next.fetch_add(1, Ordering::Relaxed) % pool.len();
        Ok(pool[idx].clone())
    }

    /// The verifying pool, or the self-signed one, built on first use.
    fn pool(&self, insecure: bool) -> Result<&[Client], &str> {
        let slot = if insecure { &self.inner.insecure } else { &self.inner.secure };
        slot.get_or_init(|| (0..self.parallel_streams()).map(|_| self.build_client(insecure)).collect())
            .as_deref()
            .map_err(String::as_str)
    }

    fn build_client(&self, insecure: bool) -> Result<Client, String> {
        let config = &self.inner.config;
        let redirects = if config.redirects == 0 {
            reqwest::redirect::Policy::none()
        } else {
            reqwest::redirect::Policy::limited(config.redirects)
        };
        let mut b = Client::builder()
            .user_agent(config.user_agent.as_str())
            .pool_max_idle_per_host(64)
            .redirect(redirects)
            .timeout(config.timeouts.request)
            .connect_timeout(config.timeouts.connect);
        // Augment the system roots with the configured CA/leaf certs (e.g.
        // a private vecd's exported cert). Verification stays on: this is
        // the secure way to trust a self-signed deployment.
        for path in &config.tls.trusted_ca_certs {
            match std::fs::read(path).map(|pem| reqwest::Certificate::from_pem(&pem)) {
                Ok(Ok(cert)) => b = b.add_root_certificate(cert),
                Ok(Err(e)) => log::warn!("trusted_ca_certs: {} is not a valid PEM cert: {e}", path.display()),
                Err(e) => log::warn!("trusted_ca_certs: cannot read {}: {e}", path.display()),
            }
        }
        // Only this pool skips verification, and only origins listed in
        // `trust_self_signed` reach it (`client_for`).
        if insecure {
            b = b.danger_accept_invalid_certs(true);
        }
        b.build().map_err(|e| format!("building the HTTP client: {e}"))
    }

    /// Whether `url`'s origin is listed in `trust_self_signed`. Origins
    /// are compared normalized.
    fn trusts_self_signed(&self, url: &str) -> bool {
        if self.inner.self_signed.is_empty() {
            return false;
        }
        origin_of(url).is_some_and(|o| self.inner.self_signed.contains(&o))
    }

    /// Attach the bearer token the read credentials give for `url`, when
    /// they give one. Additive: with no token the request is unchanged,
    /// so anonymous reads are unaffected.
    pub(crate) fn apply_read_auth(&self, rb: RequestBuilder, url: &str) -> RequestBuilder {
        match self.inner.credentials.read_token(url) {
            Some(token) => rb.bearer_auth(token),
            None => rb,
        }
    }
}

/// `scheme://host[:port]` of a URL, the form trust is listed by.
fn origin_of(url: &str) -> Option<String> {
    let u = url::Url::parse(url).ok()?;
    let host = u.host_str()?;
    Some(match u.port() {
        Some(p) => format!("{}://{host}:{p}", u.scheme()),
        None => format!("{}://{host}", u.scheme()),
    })
}

/// A permanent error for `url`, decided before any request.
fn permanent(url: &str, message: String) -> TransportError {
    TransportError { url: Some(url.to_string()), status: None, kind: TransportErrorKind::Permanent, message }
}

/// The server's answer to a [`Request`]: its status, its headers, and
/// its body, read whole or streamed.
pub struct Response {
    inner: reqwest::blocking::Response,
    url: String,
}

impl std::fmt::Debug for Response {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Response").field("url", &self.url).field("status", &self.status()).finish_non_exhaustive()
    }
}

impl Response {
    /// The HTTP status.
    pub fn status(&self) -> u16 {
        self.inner.status().as_u16()
    }

    /// Whether the status is a success (2xx).
    pub fn is_success(&self) -> bool {
        self.inner.status().is_success()
    }

    /// Whether the status is a redirect (3xx).
    pub fn is_redirection(&self) -> bool {
        self.inner.status().is_redirection()
    }

    /// The status with its reason, as `404 Not Found`.
    pub fn status_line(&self) -> String {
        self.inner.status().to_string()
    }

    /// The URL the request named.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Header `name`'s value, when it is present and text.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.inner.headers().get(name).and_then(|v| v.to_str().ok())
    }

    /// The `Content-Length` header, parsed.
    pub fn content_length(&self) -> Option<u64> {
        self.header("content-length").and_then(|v| v.parse().ok())
    }

    /// The `Last-Modified` header, parsed.
    pub fn last_modified(&self) -> Option<std::time::SystemTime> {
        last_modified(self.inner.headers())
    }

    /// The body, whole.
    pub fn bytes(self) -> Result<Vec<u8>, TransportError> {
        self.inner.bytes().map(|b| b.to_vec()).map_err(transport_error)
    }

    /// The body as text.
    pub fn text(self) -> Result<String, TransportError> {
        self.inner.text().map_err(transport_error)
    }

    /// Stream the body into `out`, returning the bytes written.
    pub fn copy_to(mut self, out: &mut dyn io::Write) -> io::Result<u64> {
        io::copy(&mut self.inner, out)
    }

    /// This response, or the error its status is when it is not a
    /// success.
    pub fn error_for_status(self) -> Result<Self, TransportError> {
        match self.inner.error_for_status() {
            Ok(inner) => Ok(Response { inner, url: self.url }),
            Err(e) => Err(transport_error(e)),
        }
    }
}

/// A reqwest failure as the API's [`TransportError`]: its URL, its
/// status when the server answered, and what a caller does next by the
/// same rule [`is_permanent`] applies.
pub(crate) fn transport_error(e: reqwest::Error) -> TransportError {
    let status = e.status().map(|s| s.as_u16());
    let kind = if status.is_some_and(status_is_permanent) || e.is_builder() || e.is_redirect() {
        TransportErrorKind::Permanent
    } else if e.is_connect() || e.is_timeout() || (e.is_request() && status.is_none()) {
        // No answer at all: refused, timed out, or the connection closed
        // before the server said anything (a pooled connection to a
        // server that has gone away fails this way, not as a connect
        // error). A local copy is what there is.
        TransportErrorKind::Unreachable
    } else {
        TransportErrorKind::Transient
    };
    TransportError { url: e.url().map(|u| u.to_string()), status, kind, message: e.to_string() }
}

/// Whether `error` is one that no retry can change, found anywhere in
/// its source chain: an offline-mode refusal, or an HTTP response whose
/// status says the request itself is refused ([`status_is_permanent`]).
pub(crate) fn is_permanent(error: &io::Error) -> bool {
    let mut cause: Option<&(dyn std::error::Error + 'static)> = error.get_ref().map(|e| e as _);
    while let Some(e) = cause {
        if let Some(t) = e.downcast_ref::<TransportError>() {
            return t.is_permanent();
        }
        if let Some(status) = e.downcast_ref::<reqwest::Error>().and_then(reqwest::Error::status) {
            return status_is_permanent(status.as_u16());
        }
        if let Some(inner) = e.downcast_ref::<io::Error>() {
            cause = inner.get_ref().map(|e| e as _);
            continue;
        }
        cause = e.source();
    }
    false
}

/// Whether an HTTP status answers every repeat of the request the same
/// way. A client error does — a missing file (404), a refused or
/// expired credential (401, 403), a range past the end (416) — except
/// the ones that mean "not now": a request timeout (408), too early
/// (425), or too many requests (429). Server errors (5xx) are
/// transient and retried.
fn status_is_permanent(status: u16) -> bool {
    (400..500).contains(&status) && !matches!(status, 408 | 425 | 429)
}

/// `Last-Modified` as a time.
fn last_modified(headers: &reqwest::header::HeaderMap) -> Option<std::time::SystemTime> {
    let header = headers.get("last-modified")?.to_str().ok()?;
    // HTTP date format: "Thu, 01 Jan 2025 00:00:00 GMT"
    httpdate::parse_http_date(header).ok()
}

/// Translate a remote URL to the HTTPS form the client fetches.
/// `http(s)://` URLs pass through; `s3://bucket/key` becomes
/// `https://<bucket>.s3.<region>.amazonaws.com/<key>`.
///
/// If the bucket lives in another region, S3 answers with a redirect to
/// its regional endpoint, which the client follows (or, for a `301`
/// with only `x-amz-bucket-region`, [`HttpTransport`](crate::HttpTransport)
/// rewrites the URL), so a wrong region costs one round trip.
///
/// This is the whole of the S3 read transport: an anonymous public
/// bucket needs no signature, so a virtual-hosted GET is all there is.
pub fn normalize_remote_url<'a>(s: &'a str, region: &str) -> Cow<'a, str> {
    if let Some(rest) = s.strip_prefix("s3://") {
        let (bucket, key) = match rest.split_once('/') {
            Some(pair) => pair,
            None => (rest, ""),
        };
        Cow::Owned(format!("https://{bucket}.s3.{region}.amazonaws.com/{key}"))
    } else {
        Cow::Borrowed(s)
    }
}

#[cfg(test)]
mod s3_normalisation_tests {
    use super::*;

    #[test]
    fn s3_url_rewrites_to_virtual_hosted_https() {
        let translated = normalize_remote_url("s3://example-bucket/path/to/file.bin", "us-east-2");
        assert_eq!(translated.as_ref(), "https://example-bucket.s3.us-east-2.amazonaws.com/path/to/file.bin");
        // Bucket-only URL (no key path) yields a trailing slash.
        let bare = normalize_remote_url("s3://only-bucket", "us-east-2");
        assert_eq!(bare.as_ref(), "https://only-bucket.s3.us-east-2.amazonaws.com/");
    }

    #[test]
    fn a_transport_rewrites_in_its_configured_region() {
        let t = Transport::new(TransportConfig { s3_region: "eu-west-1".into(), ..TransportConfig::default() });
        assert_eq!(t.normalize_url("s3://b/k").as_ref(), "https://b.s3.eu-west-1.amazonaws.com/k");
    }

    #[test]
    fn non_s3_urls_pass_through() {
        let https = normalize_remote_url("https://example.com/x", "us-east-1");
        assert_eq!(https.as_ref(), "https://example.com/x");
        let http = normalize_remote_url("http://example.com/x", "us-east-1");
        assert_eq!(http.as_ref(), "http://example.com/x");
        let path = normalize_remote_url("/abs/path", "us-east-1");
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

    /// A transport builds each pool once and keeps it: the singleton
    /// invariant that keeps `load_native_certs` from running per request.
    #[test]
    fn a_transport_builds_its_pools_once() {
        let t = Transport::new(TransportConfig::default());
        let a = t.pool(false).unwrap().as_ptr();
        let b = t.pool(false).unwrap().as_ptr();
        assert_eq!(a, b, "the pool must be built once");
        // The secure and insecure pools are distinct.
        assert_ne!(t.pool(false).unwrap().as_ptr(), t.pool(true).unwrap().as_ptr());
        // And a clone shares them.
        assert_eq!(t.clone().pool(false).unwrap().as_ptr(), a);
    }

    /// `client()` rotates through pool entries, but the observable cost
    /// is still negligible — no per-clone cert load. Many round-robin
    /// clones should complete in well under a second.
    #[test]
    fn shared_client_clone_is_cheap() {
        let t = Transport::new(TransportConfig::default());
        let _ = t.client().unwrap();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            let _ = t.client().unwrap();
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 500,
            "10_000 client() calls took {elapsed:?}; if this exceeds \
             500 ms a per-clone load_native_certs has reappeared somewhere"
        );
    }

    /// Trust is per origin: the listed one, normalized, and no other.
    #[test]
    fn self_signed_trust_is_per_origin() {
        let mut config = TransportConfig::default();
        config.tls.trust_self_signed = vec!["https://127.0.0.1:18443/".into()];
        let t = Transport::new(config);
        assert!(t.trusts_self_signed("https://127.0.0.1:18443/ds/base.fvecs"));
        assert!(!t.trusts_self_signed("https://127.0.0.1:18444/ds/base.fvecs"));
        assert!(!t.trusts_self_signed("http://127.0.0.1:18443/ds/base.fvecs"));
        assert!(!Transport::new(TransportConfig::default()).trusts_self_signed("https://127.0.0.1:18443/x"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Client errors are permanent except the "not now" ones; server
    /// errors and anything outside 4xx are retried.
    #[test]
    fn client_errors_are_permanent_but_not_now_ones() {
        for s in [400, 401, 403, 404, 410, 416] {
            assert!(status_is_permanent(s), "{s} is permanent");
        }
        for s in [408, 425, 429, 500, 502, 503, 504, 301] {
            assert!(!status_is_permanent(s), "{s} is retried");
        }
    }

    /// The offline refusal names the URL and the configured way out, and
    /// is permanent however deeply it is wrapped.
    #[test]
    fn the_offline_refusal_is_permanent_and_says_how_out() {
        let t = Transport::new(TransportConfig {
            offline: true,
            offline_hint: Some("VECTORDATA_OFFLINE / settings `offline`".into()),
            ..TransportConfig::default()
        });
        let err = t.ensure_online("https://h/f").unwrap_err();
        assert_eq!(
            err.to_string(),
            "offline mode is on (VECTORDATA_OFFLINE / settings `offline`): https://h/f is not available \
             locally. Fetch it while online, or turn offline mode off."
        );
        assert!(is_permanent(&err));
        assert!(is_permanent(&io::Error::other(err)));
        // An explicit request is not refused by it (it fails to connect
        // instead, which is not an offline error).
        let sent = t.send(Request::get("http://127.0.0.1:9/never").explicit());
        assert!(sent.is_err_and(|e| !e.is_offline()));
    }

    /// **A server that closes without answering is unreachable**, as a
    /// refused connection is: what a pooled connection to a server that
    /// has gone away sees, and what decides falling back to a kept copy.
    #[test]
    fn a_connection_closed_without_an_answer_is_unreachable() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/x", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                drop(stream);
            }
        });
        let t = Transport::new(TransportConfig::default());
        let err = t.send(Request::get(url.as_str())).unwrap_err();
        assert!(err.is_unreachable(), "{err:?}");
        let refused = t.send(Request::get("http://127.0.0.1:9/x")).unwrap_err();
        assert!(refused.is_unreachable(), "{refused:?}");
    }
}
