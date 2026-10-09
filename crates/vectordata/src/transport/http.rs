// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Internal. HTTP byte-range transport using reqwest with connection pooling.

use std::io;
use std::sync::OnceLock;

use reqwest::blocking::Client;
use reqwest::header::{ACCEPT_RANGES, CONTENT_LENGTH, RANGE};
use url::Url;

use super::ChunkedTransport;

/// HTTP transport that fetches byte ranges from a URL.
///
/// Uses a shared `reqwest::blocking::Client` for connection pooling.
/// Content length and range support are detected lazily on first use.
#[derive(Debug)]
pub struct HttpTransport {
    client: Client,
    url: Url,
    /// URL after any first-probe redirect correction (e.g. S3
    /// wrong-region rewrite). When set, takes precedence over
    /// `url` for every subsequent request. Read with
    /// [`Self::effective_url`].
    effective_url: OnceLock<Url>,
    content_length: OnceLock<u64>,
    supports_range: OnceLock<bool>,
}

impl HttpTransport {
    /// Create a new HTTP transport for the given URL. Reuses the
    /// process-wide shared `reqwest::blocking::Client` rather than
    /// constructing a fresh one — `Client::new()` triggers a full
    /// native-cert load that dominates per-request CPU when called
    /// repeatedly.
    pub fn new(url: Url) -> Self {
        HttpTransport {
            // url-aware: a self-signed endpoint listed in `trust_self_signed`
            // gets the non-verifying client; everyone else the verifying one.
            client: super::shared_client_for(url.as_str()),
            url,
            effective_url: OnceLock::new(),
            content_length: OnceLock::new(),
            supports_range: OnceLock::new(),
        }
    }

    /// Create with a shared client (for connection pooling across transports).
    pub fn with_client(client: Client, url: Url) -> Self {
        HttpTransport {
            client,
            url,
            effective_url: OnceLock::new(),
            content_length: OnceLock::new(),
            supports_range: OnceLock::new(),
        }
    }

    /// The remote URL this transport reads from. Used in error
    /// messages to point operators at the source that needs a
    /// `.mref` published.
    pub fn url(&self) -> &Url { &self.url }

    /// Returns the URL that should actually be hit. This is the
    /// initial URL unless the first probe corrected it (e.g. an
    /// S3 cross-region rewrite triggered by an `x-amz-bucket-
    /// region` redirect header — see [`Self::probe`]).
    fn effective_url(&self) -> &Url {
        self.effective_url.get().unwrap_or(&self.url)
    }

    /// Probe the remote resource via HEAD to determine size and
    /// range support. If S3 returns a wrong-region redirect
    /// (`HTTP 301` carrying `x-amz-bucket-region` instead of a
    /// `Location` header — reqwest's auto-follow can't help
    /// there), the bucket-hosted URL is rewritten to the correct
    /// regional endpoint, cached in `effective_url`, and the
    /// probe retried once. All subsequent `fetch_range` calls
    /// pick up the corrected URL.
    fn probe(&self) -> io::Result<(u64, bool)> {
        super::ensure_online(self.url.as_str())?;
        let target = self.effective_url().clone();
        let resp = super::apply_read_auth(self.client.head(target.clone()), Some(&target))
            .send()
            .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, e))?;

        // S3 cross-region redirect: 301 with `x-amz-bucket-region`
        // header and no `Location`. Rewrite the URL and retry
        // exactly once. We don't loop — if the second probe also
        // misbehaves we surface the error.
        if resp.status().as_u16() == 301
            && let Some(region) = resp
                .headers()
                .get("x-amz-bucket-region")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
                && let Some(corrected) = rewrite_s3_url_region(self.effective_url(), &region)
            {
                let _ = self.effective_url.set(corrected.clone());
                let retry = super::apply_read_auth(self.client.head(corrected.clone()), Some(&corrected))
                    .send()
                    .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, e))?
                    .error_for_status()
                    .map_err(io::Error::other)?;
                return read_probe_headers(&retry);
            }

        // Any other unfollowed 3xx is a misconfiguration — reqwest
        // follows up to 10 redirects when there's a `Location`
        // header, so a 3xx surfacing here means the response is
        // missing `Location` (or the limit was exceeded). Surface
        // it as a clear error rather than letting the "missing
        // Content-Length" read further down hide the real cause.
        if resp.status().is_redirection() {
            return Err(io::Error::other(
                format!(
                    "unfollowed {} from {} (no Location header — \
                     bucket may be in a different region; set AWS_REGION \
                     or use the regional endpoint)",
                    resp.status(),
                    self.effective_url(),
                ),
            ));
        }
        let resp = resp
            .error_for_status()
            .map_err(io::Error::other)?;
        read_probe_headers(&resp)
    }

    fn ensure_probed(&self) -> io::Result<()> {
        if self.content_length.get().is_none() {
            let (length, ranges) = self.probe()?;
            let _ = self.content_length.set(length);
            let _ = self.supports_range.set(ranges);
        }
        Ok(())
    }

    /// Stream the entire resource body — one plain GET, no `Range`
    /// header — into `out`. Returns the byte count written.
    ///
    /// This is the FullTransfer path for servers that don't support
    /// byte ranges: chunked access is impossible, so the documented
    /// fallback downloads the whole file once into the local cache
    /// and serves every read from there.
    pub(crate) fn fetch_full_to(&self, out: &mut dyn io::Write) -> io::Result<u64> {
        self.ensure_probed()?;
        let target = self.effective_url().clone();
        let client = super::shared_client_for(target.as_str());
        let mut resp = super::apply_read_auth(client.get(target.clone()), Some(&target))
            .send()
            .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, e))?
            .error_for_status()
            .map_err(io::Error::other)?;
        std::io::copy(&mut resp, out)
    }
}

/// Read `Content-Length` and `Accept-Ranges` from a successful
/// HEAD/GET response. Factored out so the wrong-region retry path
/// can re-use it.
fn read_probe_headers(resp: &reqwest::blocking::Response) -> io::Result<(u64, bool)> {
    let length = resp
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing Content-Length header")
        })?;
    let ranges = resp
        .headers()
        .get(ACCEPT_RANGES)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("bytes"));
    Ok((length, ranges))
}

/// Rewrite the host of a virtual-hosted-style S3 URL
/// (`<bucket>.s3.<region>.amazonaws.com`) so the region segment
/// matches `correct_region`. Returns `None` for URLs that don't
/// look like S3 — those are passed through unchanged at the call
/// site.
pub(crate) fn rewrite_s3_url_region(url: &Url, correct_region: &str) -> Option<Url> {
    let host = url.host_str()?;
    // Virtual-hosted-style: `<bucket>.s3.<region>.amazonaws.com`
    // or `<bucket>.s3.amazonaws.com` (legacy global). The dot-
    // split shape we recognise: at least 4 segments ending in
    // `amazonaws.com` and containing an `s3` literal at position 1.
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() < 4 { return None; }
    if !parts.ends_with(&["amazonaws", "com"]) { return None; }
    let bucket = parts[0];
    if parts[1] != "s3" { return None; }
    let new_host = format!("{bucket}.s3.{correct_region}.amazonaws.com");
    let mut corrected = url.clone();
    corrected.set_host(Some(&new_host)).ok()?;
    Some(corrected)
}

impl ChunkedTransport for HttpTransport {
    fn fetch_range(&self, start: u64, len: u64) -> io::Result<Vec<u8>> {
        // Ensure the probe has run — it may have discovered a
        // wrong-region S3 redirect and corrected `effective_url`.
        // Without this, the very first range request goes to the
        // original (wrong) URL and S3 returns a 301 XML body that
        // confuses the byte-count assertion downstream.
        self.ensure_probed()?;
        super::ensure_online(self.url.as_str())?;
        // Pick a client off the round-robin pool **per call**, not
        // per transport. A single shared `Client` puts every worker's
        // HTTP completion + TLS decryption on the same internal
        // Tokio runtime thread, capping aggregate throughput at one
        // core regardless of `download_concurrency`. Per-call pick
        // distributes N workers across the pool's N runtime threads
        // so chunks actually arrive in parallel.
        let end = start + len - 1;
        let target = self.effective_url().clone();
        let client = super::shared_client_for(target.as_str());
        let resp = super::apply_read_auth(client.get(target.clone()), Some(&target))
            .header(RANGE, format!("bytes={}-{}", start, end))
            .send()
            .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, e))?
            .error_for_status()
            .map_err(io::Error::other)?;

        let bytes = resp
            .bytes()
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e))?;

        if bytes.len() != len as usize {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "expected {} bytes, got {} (range {}-{})",
                    len,
                    bytes.len(),
                    start,
                    end
                ),
            ));
        }

        Ok(bytes.to_vec())
    }

    fn content_length(&self) -> io::Result<u64> {
        self.ensure_probed()?;
        Ok(*self.content_length.get().unwrap())
    }

    fn supports_range(&self) -> bool {
        let _ = self.ensure_probed();
        self.supports_range.get().copied().unwrap_or(false)
    }
}

// HTTP transport integration tests live in tests/transport.rs (using testserver).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrite_s3_region_swaps_the_region_segment() {
        let original = Url::parse("https://my-bucket.s3.us-east-1.amazonaws.com/path/to/x").unwrap();
        let corrected = rewrite_s3_url_region(&original, "us-east-2").unwrap();
        assert_eq!(corrected.as_str(),
            "https://my-bucket.s3.us-east-2.amazonaws.com/path/to/x");
    }

    #[test]
    fn rewrite_s3_region_returns_none_for_non_s3_hosts() {
        let other = Url::parse("https://example.com/x").unwrap();
        assert!(rewrite_s3_url_region(&other, "us-east-2").is_none());
        let github = Url::parse("https://api.github.com/repos/x").unwrap();
        assert!(rewrite_s3_url_region(&github, "us-east-2").is_none());
    }

    #[test]
    fn rewrite_s3_region_handles_three_part_region_codes() {
        // S3 regions like `ap-southeast-3` have hyphens; the host
        // split on dots still treats the whole `ap-southeast-3`
        // segment as one element.
        let original = Url::parse("https://b.s3.us-east-1.amazonaws.com/k").unwrap();
        let corrected = rewrite_s3_url_region(&original, "ap-southeast-3").unwrap();
        assert_eq!(corrected.host_str().unwrap(), "b.s3.ap-southeast-3.amazonaws.com");
    }
}

/// What reaches the retry policy from a real HTTP response: the status
/// survives `fetch_range`'s error mapping, so a client error — a
/// refused credential, a missing file — is given back at once, and a
/// server error or throttle is retried.
#[cfg(test)]
mod retry_classification {
    use super::*;
    use crate::transport::RetryPolicy;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Serve one object of 8 bytes: HEAD answers its size, and each GET
    /// is answered with the next status in `gets` (206 sends the range).
    /// Returns the URL and the count of GETs seen.
    fn scripted_server(gets: Vec<u16>) -> (Url, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = Url::parse(&format!("http://{}/object", listener.local_addr().unwrap())).unwrap();
        let seen = Arc::new(AtomicUsize::new(0));
        let count = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut request = String::new();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    request.push_str(&line);
                }
                let response = if request.starts_with("HEAD") {
                    "HTTP/1.1 200 OK\r\nContent-Length: 8\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n".to_string()
                } else {
                    let n = count.fetch_add(1, Ordering::SeqCst);
                    match gets.get(n).copied().unwrap_or(500) {
                        206 => "HTTP/1.1 206 Partial Content\r\nContent-Length: 4\r\n\
                                Content-Range: bytes 0-3/8\r\nConnection: close\r\n\r\nabcd"
                            .to_string(),
                        s => format!("HTTP/1.1 {s} Scripted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"),
                    }
                };
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (url, seen)
    }

    fn fast_policy() -> RetryPolicy {
        RetryPolicy { max_retries: 3, base_delay_ms: 1, max_delay_ms: 1, jitter_fraction: 0.0 }
    }

    #[test]
    fn a_client_error_is_asked_once() {
        for status in [401, 403, 404] {
            let (url, gets) = scripted_server(vec![status; 4]);
            let transport = HttpTransport::with_client(Client::new(), url);
            let err = fast_policy().execute(|| transport.fetch_range(0, 4)).unwrap_err();
            assert!(err.to_string().contains(&status.to_string()), "{status}: {err}");
            assert_eq!(gets.load(Ordering::SeqCst), 1, "{status} was retried");
        }
    }

    #[test]
    fn a_server_error_or_throttle_is_retried_until_it_passes() {
        for status in [503, 429, 408] {
            let (url, gets) = scripted_server(vec![status, status, 206]);
            let transport = HttpTransport::with_client(Client::new(), url);
            let bytes = fast_policy().execute(|| transport.fetch_range(0, 4)).unwrap();
            assert_eq!(bytes, b"abcd");
            assert_eq!(gets.load(Ordering::SeqCst), 3, "{status}: two failures, then the range");
        }
    }
}
