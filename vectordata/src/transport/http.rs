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
            client: super::shared_client(),
            url,
            content_length: OnceLock::new(),
            supports_range: OnceLock::new(),
        }
    }

    /// Create with a shared client (for connection pooling across transports).
    pub fn with_client(client: Client, url: Url) -> Self {
        HttpTransport {
            client,
            url,
            content_length: OnceLock::new(),
            supports_range: OnceLock::new(),
        }
    }

    /// The remote URL this transport reads from. Used in error
    /// messages to point operators at the source that needs a
    /// `.mref` published.
    pub fn url(&self) -> &Url { &self.url }

    /// Probe the remote resource via HEAD to determine size and range support.
    fn probe(&self) -> io::Result<(u64, bool)> {
        let resp = self
            .client
            .head(self.url.clone())
            .send()
            .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, e))?
            .error_for_status()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

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

    fn ensure_probed(&self) -> io::Result<()> {
        if self.content_length.get().is_none() {
            let (length, ranges) = self.probe()?;
            let _ = self.content_length.set(length);
            let _ = self.supports_range.set(ranges);
        }
        Ok(())
    }
}

impl ChunkedTransport for HttpTransport {
    fn fetch_range(&self, start: u64, len: u64) -> io::Result<Vec<u8>> {
        let end = start + len - 1;
        let resp = self
            .client
            .get(self.url.clone())
            .header(RANGE, format!("bytes={}-{}", start, end))
            .send()
            .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, e))?
            .error_for_status()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

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
