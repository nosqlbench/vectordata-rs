// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! HTTP byte-range transport using reqwest with connection pooling.

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
