// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! HTTP download primitives backed by libcurl.
//!
//! Provides [`download_file`] for downloading a single URL to disk with
//! automatic resume support, and [`head_content_length`] for querying
//! the remote file size without downloading.

use std::fs;
use std::io::Write as IoWrite;
use std::path::Path;
use std::sync::{Arc, Mutex};

use curl::easy::Easy;

/// Query the `Content-Length` of a remote resource via an HTTP HEAD request.
///
/// Returns `None` if the server does not provide a content length or the
/// request fails after a single attempt.
pub fn head_content_length(url: &str) -> Option<u64> {
    head_content_length_once(url)
}

fn head_content_length_once(url: &str) -> Option<u64> {
    let mut easy = Easy::new();
    easy.url(url).ok()?;
    easy.nobody(true).ok()?;
    easy.follow_location(true).ok()?;
    easy.perform().ok()?;
    easy.content_length_download()
        .ok()
        .and_then(|len| if len >= 0.0 { Some(len as u64) } else { None })
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

/// Download a file from `url` to `dest` using libcurl.
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

    let mut easy = Easy::new();
    easy.url(url).map_err(|e| e.to_string())?;
    easy.follow_location(true).map_err(|e| e.to_string())?;
    easy.fail_on_error(true).map_err(|e| e.to_string())?;
    easy.progress(true).map_err(|e| e.to_string())?;

    let written = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let written_clone = written.clone();

    let file = Arc::new(Mutex::new(
        fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(dest)
            .map_err(|e| format!("Failed to create {}: {}", dest.display(), e))?,
    ));

    let file_clone = Arc::clone(&file);
    let written_for_progress = written.clone();
    let mut transfer = easy.transfer();
    transfer
        .write_function(move |data| {
            let mut f = file_clone.lock().unwrap();
            match f.write_all(data) {
                Ok(()) => {
                    written_clone.fetch_add(data.len() as u64, std::sync::atomic::Ordering::Relaxed);
                    Ok(data.len())
                }
                Err(_) => Err(curl::easy::WriteError::Pause),
            }
        })
        .map_err(|e| e.to_string())?;

    transfer
        .progress_function(move |_dl_total, _dl_now, _ul_total, _ul_now| {
            on_bytes(written_for_progress.load(std::sync::atomic::Ordering::Relaxed));
            true
        })
        .map_err(|e| e.to_string())?;

    transfer.perform().map_err(|e| {
        let _ = fs::remove_file(dest);
        format!("Download failed for {}: {}", url, e)
    })?;

    drop(transfer);

    let response_code = easy.response_code().map_err(|e| e.to_string())?;
    if response_code != 200 {
        let _ = fs::remove_file(dest);
        return Err(format!("HTTP {} for {}", response_code, url));
    }

    // Verify downloaded size against Content-Length from this response.
    // For HTTP 200 (full download), content_length_download() is the
    // total file size — safe to compare against local file.
    let expected_len = easy.content_length_download()
        .ok()
        .and_then(|len| if len > 0.0 { Some(len as u64) } else { None });

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
