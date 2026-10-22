// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! HTTP download primitives over the process's transport.
//!
//! Provides [`download_file`] for streaming one request's answer to disk
//! with progress callbacks, and [`head_content_length_retry`] for querying
//! the remote file size without downloading. Every request goes through a
//! [`Transport`] — in the commands, `vectordata::transport::shared()`, so
//! downloads share its client pool, TLS trust, offline gate and user agent
//! — and carries only the authorisation its caller gave it (anonymous, an
//! API key, a SigV4 signature), never vectordata's read credentials
//! (`docs/design/srd-crate-boundaries.md` TR-9).

use std::fs;
use std::io::{self, Read as _, Write as _};
use std::path::Path;

use vectordata::transport::{Request, RetryPolicy, Transport};

/// The retry policy of one file's download, `tries` attempts in all: the
/// transport's backoff (1 s doubling to at most 30 s, with jitter), and
/// no retry of an answer that no retry can change — offline mode, or a
/// client error such as 404 or 403.
pub fn download_retry_policy(tries: u32) -> RetryPolicy {
    RetryPolicy { max_retries: tries.max(1) - 1, ..RetryPolicy::default() }
}

/// Query remote Content-Length with retries.
///
/// Up to `max_attempts` anonymous HEAD requests, half a second apart.
/// One that the server answers without a `Content-Length`, with a server
/// error, or not at all is retried; offline mode and a client error (such
/// as 404) end the attempts at once. The error names the URL, the
/// attempts made and the last failure.
pub fn head_content_length_retry(transport: &Transport, url: &str, max_attempts: u32) -> Result<u64, String> {
    let policy =
        RetryPolicy { max_retries: max_attempts.max(1) - 1, base_delay_ms: 500, max_delay_ms: 500, jitter_fraction: 0.0 };
    let mut made = 0u32;
    policy
        .execute(|| {
            made += 1;
            let resp = transport.send(Request::head(url).anonymous()).map_err(io::Error::other)?;
            if !resp.is_success() {
                return Err(io::Error::other(resp.status_error()));
            }
            resp.content_length().ok_or_else(|| io::Error::other("no Content-Length in the answer"))
        })
        .map_err(|e| format!("failed to get Content-Length for {} after {} attempts: {}", url, made, e))
}

/// Send `request` and stream its answer to `dest`.
///
/// Always downloads fresh (no resume from partial files — partials are
/// deleted first to avoid append-on-complete bugs). The body is read in
/// 256 KiB increments straight to the file, and `on_bytes` is called after
/// each with the cumulative bytes written, so a caller's progress bar
/// moves in real time.
///
/// `listed_size` is the size the file is known to have (a bucket listing's);
/// the download is verified against it, and the answer's `Content-Length`
/// must agree with it. Without one, the download is verified against the
/// `Content-Length`. A partial file is removed on every failure but a
/// `Content-Length` disagreement.
///
/// A non-success answer is an error carrying its status and the start of
/// its body (an S3 error's XML), permanent or transient by
/// [`Response::status_error`](vectordata::transport::Response::status_error)'s
/// rule, so a [`RetryPolicy`] stops on what no retry can change.
pub fn download_file(
    transport: &Transport,
    request: Request,
    dest: &Path,
    listed_size: Option<u64>,
    on_bytes: &dyn Fn(u64),
) -> io::Result<()> {
    if dest.exists() {
        let _ = fs::remove_file(dest);
    }
    let url = request.url.clone();

    // The transport's error as it is, so a retry policy sees an offline
    // refusal or an unreachable server for what it is.
    let mut response = transport.send(request).map_err(io::Error::other)?;

    if !response.is_success() {
        return Err(io::Error::other(response.status_error()));
    }

    let content_length = response.content_length();

    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(dest)
        .map_err(|e| io::Error::other(format!("Failed to create {}: {}", dest.display(), e)))?;

    let mut written: u64 = 0;
    let mut buf = vec![0u8; 256 * 1024];

    loop {
        let n = response.read(&mut buf).map_err(|e| {
            let _ = fs::remove_file(dest);
            io::Error::other(format!("Download failed for {}: {}", url, e))
        })?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| {
            let _ = fs::remove_file(dest);
            io::Error::other(format!("Write failed for {}: {}", dest.display(), e))
        })?;
        written += n as u64;
        on_bytes(written);
    }

    let actual = fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
    match listed_size {
        Some(listed) => {
            if actual != listed {
                let _ = fs::remove_file(dest);
                return Err(io::Error::other(format!(
                    "Size mismatch for {}: expected {} bytes (from listing), got {}",
                    url, listed, actual,
                )));
            }
            if let Some(cl) = content_length
                && cl != listed
            {
                return Err(io::Error::other(format!(
                    "Content-Length {} does not match listed size {} for {}",
                    cl, listed, url,
                )));
            }
        }
        // Verify the downloaded size against this response's Content-Length.
        None => {
            if let Some(expected) = content_length
                && actual != expected
            {
                let _ = fs::remove_file(dest);
                return Err(io::Error::other(format!(
                    "Size mismatch for {}: expected {} bytes, got {}",
                    url, expected, actual
                )));
            }
        }
    }

    Ok(())
}
