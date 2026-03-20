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
use log::info;

/// Query the `Content-Length` of a remote resource via an HTTP HEAD request.
///
/// Returns `None` if the server does not provide a content length or the
/// request fails.
pub fn head_content_length(url: &str) -> Option<u64> {
    let mut easy = Easy::new();
    easy.url(url).ok()?;
    easy.nobody(true).ok()?;
    easy.follow_location(true).ok()?;
    easy.perform().ok()?;
    easy.content_length_download()
        .ok()
        .and_then(|len| if len >= 0.0 { Some(len as u64) } else { None })
}

/// Download a file from `url` to `dest` using libcurl.
///
/// If `dest` already exists as a partial file, the download resumes from
/// the existing byte offset. On HTTP error or transfer failure, the
/// partial file is removed so the next retry starts fresh.
pub fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    let mut easy = Easy::new();
    easy.url(url).map_err(|e| e.to_string())?;
    easy.follow_location(true).map_err(|e| e.to_string())?;

    // Resume support: if partial file exists, resume from its length
    let resume_from = if dest.exists() {
        let meta = fs::metadata(dest).map_err(|e| e.to_string())?;
        let len = meta.len();
        if len > 0 {
            easy.resume_from(len).map_err(|e| e.to_string())?;
            info!("Resuming {} from byte {}", url, len);
            len
        } else {
            0
        }
    } else {
        0
    };

    let file = Arc::new(Mutex::new(
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dest)
            .map_err(|e| format!("Failed to open {}: {}", dest.display(), e))?,
    ));

    let file_clone = Arc::clone(&file);
    let mut transfer = easy.transfer();
    transfer
        .write_function(move |data| {
            let mut f = file_clone.lock().unwrap();
            f.write_all(data)
                .map(|_| data.len())
                .map_err(|_| curl::easy::WriteError::Pause)
        })
        .map_err(|e| e.to_string())?;

    transfer.perform().map_err(|e| {
        // If resume failed, remove partial file so next retry starts fresh
        if resume_from > 0 {
            let _ = fs::remove_file(dest);
        }
        format!("Download failed for {}: {}", url, e)
    })?;

    drop(transfer);

    let response_code = easy.response_code().map_err(|e| e.to_string())?;
    // 200 = full download, 206 = partial content (resume)
    if response_code != 200 && response_code != 206 {
        let _ = fs::remove_file(dest);
        return Err(format!("HTTP {} for {}", response_code, url));
    }

    Ok(())
}
