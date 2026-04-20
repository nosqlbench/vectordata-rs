// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! HTTP download primitives backed by reqwest (blocking).
//!
//! Provides [`download_file`] for downloading a single URL to disk with
//! progress callbacks, and [`head_content_length`] for querying the remote
//! file size without downloading. [`download_signed_file`] is the same
//! download path but accepts caller-supplied request headers (used for
//! SigV4-signed COS GETs).

use std::fs;
use std::io::{Read as _, Write as _};
use std::path::Path;

use reqwest::blocking::Client;
use reqwest::header::HeaderMap;

/// Query the `Content-Length` of a remote resource via an HTTP HEAD request.
///
/// Returns `None` if the server does not provide a content length or the
/// request fails after a single attempt.
pub fn head_content_length(url: &str) -> Option<u64> {
    head_content_length_once(url)
}

fn head_content_length_once(url: &str) -> Option<u64> {
    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build().ok()?;
    let resp = client.head(url).send().ok()?;
    resp.content_length()
}

/// Query remote Content-Length with retries.
///
/// Tries up to `max_attempts` HEAD requests. Returns an error if all
/// attempts fail to produce a Content-Length.
pub fn head_content_length_retry(url: &str, max_attempts: u32) -> Result<u64, String> {
    for attempt in 1..=max_attempts {
        if let Some(len) = head_content_length_once(url) {
            return Ok(len);
        }
        if attempt < max_attempts {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
    Err(format!("failed to get Content-Length for {} after {} attempts", url, max_attempts))
}

/// Download a file from `url` to `dest` using reqwest.
///
/// Always downloads fresh (no resume from partial files — partials are
/// deleted first to avoid append-on-complete bugs). After download,
/// verifies file size against Content-Length.
///
/// `on_bytes` is called with the cumulative bytes written so far,
/// allowing the caller to update a progress bar in real time.
pub fn download_file(
    url: &str,
    dest: &Path,
    on_bytes: &dyn Fn(u64),
) -> Result<(), String> {
    if dest.exists() {
        let _ = fs::remove_file(dest);
    }

    let client = reqwest::blocking::Client::builder()
        .user_agent("veks/0.14")
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))?;

    let mut response = client.get(url).send()
        .map_err(|e| format!("Download failed for {}: {}", url, e))?;

    let status = response.status();
    if !status.is_success() {
        return Err(format!("HTTP {} for {}", status.as_u16(), url));
    }

    let expected_len = response.content_length();

    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(dest)
        .map_err(|e| format!("Failed to create {}: {}", dest.display(), e))?;

    let mut written: u64 = 0;
    let mut buf = vec![0u8; 256 * 1024];

    loop {
        let n = response.read(&mut buf)
            .map_err(|e| {
                let _ = fs::remove_file(dest);
                format!("Download failed for {}: {}", url, e)
            })?;
        if n == 0 { break; }
        file.write_all(&buf[..n])
            .map_err(|e| {
                let _ = fs::remove_file(dest);
                format!("Write failed for {}: {}", dest.display(), e)
            })?;
        written += n as u64;
        on_bytes(written);
    }

    // Verify downloaded size against Content-Length from this response.
    if let Some(expected) = expected_len {
        let actual = fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
        if actual != expected {
            let _ = fs::remove_file(dest);
            return Err(format!(
                "Size mismatch for {}: expected {} bytes, got {}",
                url, expected, actual
            ));
        }
    }

    Ok(())
}

/// Download a file using a caller-supplied client and request headers.
///
/// Used for COS / S3-compatible GETs where the `Authorization`, `x-amz-date`,
/// and related SigV4 headers must accompany the request. The caller signs
/// each request immediately before calling this function — SigV4 signatures
/// are time-bound (15-minute window) so the signing must happen at the moment
/// of the request, not when the job was enqueued.
///
/// `expected_size` is the size from the bucket listing; the file is verified
/// against it after download. (`Content-Length` from the response is used as a
/// secondary check.)
pub fn download_signed_file(
    client: &Client,
    url: &str,
    headers: HeaderMap,
    dest: &Path,
    expected_size: u64,
    on_bytes: &dyn Fn(u64),
) -> Result<(), String> {
    if dest.exists() {
        let _ = fs::remove_file(dest);
    }

    let mut response = client
        .get(url)
        .headers(headers)
        .send()
        .map_err(|e| format!("Download failed for {}: {}", url, e))?;

    let status = response.status();
    if !status.is_success() {
        // Body of an S3 error is XML — read it for diagnostics.
        let body = response.text().unwrap_or_default();
        return Err(format!(
            "HTTP {} for {}{}",
            status.as_u16(),
            url,
            if body.is_empty() { String::new() } else { format!(": {}", body) }
        ));
    }

    let content_length = response.content_length();

    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(dest)
        .map_err(|e| format!("Failed to create {}: {}", dest.display(), e))?;

    let mut written: u64 = 0;
    let mut buf = vec![0u8; 256 * 1024];

    loop {
        let n = response.read(&mut buf).map_err(|e| {
            let _ = fs::remove_file(dest);
            format!("Download failed for {}: {}", url, e)
        })?;
        if n == 0 { break; }
        file.write_all(&buf[..n]).map_err(|e| {
            let _ = fs::remove_file(dest);
            format!("Write failed for {}: {}", dest.display(), e)
        })?;
        written += n as u64;
        on_bytes(written);
    }

    let actual = fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
    if actual != expected_size {
        let _ = fs::remove_file(dest);
        return Err(format!(
            "Size mismatch for {}: expected {} bytes (from listing), got {}",
            url, expected_size, actual,
        ));
    }
    if let Some(cl) = content_length {
        if cl != expected_size {
            return Err(format!(
                "Content-Length {} does not match listed size {} for {}",
                cl, expected_size, url,
            ));
        }
    }

    Ok(())
}
