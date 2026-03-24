// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Merkle coverage check.
//!
//! Verifies that all publishable data files above a size threshold have
//! companion `.mref` merkle reference files.

use std::path::{Path, PathBuf};

use super::CheckResult;
use crate::filters;

/// Check merkle (.mref) coverage for publishable data files.
pub fn check(_root: &Path, publishable: &[PathBuf], threshold: u64) -> CheckResult {
    let mut missing: Vec<String> = Vec::new();
    let mut stale: Vec<String> = Vec::new();
    let mut covered = 0u64;
    let mut checked = 0u64;

    for file in publishable {
        let fname = file.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if filters::is_merkle_exempt(&fname) {
            continue;
        }

        let size = match std::fs::metadata(file) {
            Ok(m) => m.len(),
            Err(_) => continue,
        };

        if size < threshold {
            continue;
        }

        checked += 1;
        let mref_path = PathBuf::from(format!("{}.mref", file.display()));

        if !mref_path.exists() {
            missing.push(format!(
                "{} ({}) — no .mref",
                super::rel_display(file),
                format_size(size),
            ));
            continue;
        }

        // Check if .mref is older than the data file
        let data_mtime = std::fs::metadata(file).ok().and_then(|m| m.modified().ok());
        let mref_mtime = std::fs::metadata(&mref_path).ok().and_then(|m| m.modified().ok());

        if let (Some(dm), Some(mm)) = (data_mtime, mref_mtime) {
            if dm > mm {
                stale.push(format!(
                    "{} ({}) — .mref stale (data newer)",
                    super::rel_display(file),
                    format_size(size),
                ));
                continue;
            }
        }

        covered += 1;
    }

    if missing.is_empty() && stale.is_empty() {
        let mut result = CheckResult::ok("merkle");
        if checked > 0 {
            result.messages.push(format!(
                "{} file(s) >= {}, all have current .mref",
                checked,
                format_size(threshold),
            ));
        } else {
            result.messages.push(format!(
                "no files >= {} threshold",
                format_size(threshold),
            ));
        }
        result
    } else {
        let mut messages: Vec<String> = Vec::new();
        for m in &missing {
            messages.push(m.clone());
        }
        for s in &stale {
            messages.push(s.clone());
        }
        messages.push(format!(
            "{} covered, {} missing/stale out of {} checked",
            covered,
            missing.len() + stale.len(),
            checked,
        ));
        CheckResult::fail("merkle", messages)
    }
}

/// Return the list of publishable files that are missing or have stale `.mref` files.
pub fn missing_mref_files(publishable: &[PathBuf], threshold: u64) -> Vec<PathBuf> {
    let mut result = Vec::new();
    for file in publishable {
        if file.extension().map(|e| e == "mref").unwrap_or(false) {
            continue;
        }
        let size = match std::fs::metadata(file) {
            Ok(m) => m.len(),
            Err(_) => continue,
        };
        if size < threshold {
            continue;
        }
        let mref_path = PathBuf::from(format!("{}.mref", file.display()));
        if !mref_path.exists() {
            result.push(file.clone());
            continue;
        }
        let data_mtime = std::fs::metadata(file).ok().and_then(|m| m.modified().ok());
        let mref_mtime = std::fs::metadata(&mref_path).ok().and_then(|m| m.modified().ok());
        if let (Some(dm), Some(mm)) = (data_mtime, mref_mtime) {
            if dm > mm {
                result.push(file.clone());
            }
        }
    }
    result
}

/// Format a byte count as a human-readable string.
fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
