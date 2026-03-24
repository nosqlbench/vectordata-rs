// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Path utility functions shared across veks crates.

use std::path::{Component, Path, PathBuf};

/// Compute a relative path from `base` directory to `target`.
///
/// Both paths are normalized to absolute before comparison.
/// Returns `"."` if the paths are identical.
pub fn relative_path(base: &Path, target: &Path) -> PathBuf {
    let abs_base = if base.is_absolute() {
        base.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(base)
    };
    let abs_target = if target.is_absolute() {
        target.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(target)
    };

    let base_parts: Vec<_> = abs_base.components()
        .filter(|c| !matches!(c, Component::CurDir))
        .collect();
    let target_parts: Vec<_> = abs_target.components()
        .filter(|c| !matches!(c, Component::CurDir))
        .collect();

    let common = base_parts.iter().zip(target_parts.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let mut result = PathBuf::new();
    for _ in common..base_parts.len() {
        result.push("..");
    }
    for part in &target_parts[common..] {
        result.push(part);
    }

    if result.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        result
    }
}

/// Display a path relative to the current working directory.
/// Falls back to the full path if stripping fails.
pub fn rel_display(path: &Path) -> String {
    if let Ok(cwd) = std::env::current_dir() {
        path.strip_prefix(&cwd)
            .map(|r| {
                let s = r.to_string_lossy().to_string();
                if s.is_empty() { ".".to_string() } else { s }
            })
            .unwrap_or_else(|_| path.to_string_lossy().to_string())
    } else {
        path.to_string_lossy().to_string()
    }
}

/// Parse a size string with optional suffix (K, M, G, T, KiB, MiB, etc.).
pub fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() { return None; }

    let digit_end = s.find(|c: char| !c.is_ascii_digit() && c != '_' && c != '.')
        .unwrap_or(s.len());

    let num_str: String = s[..digit_end].chars().filter(|c| *c != '_').collect();
    let suffix = &s[digit_end..];

    let num: f64 = num_str.parse().ok()?;

    let multiplier: u64 = match suffix.to_uppercase().as_str() {
        "" => 1,
        "K" => 1_000,
        "M" => 1_000_000,
        "G" | "B" => 1_000_000_000,
        "T" => 1_000_000_000_000,
        "KB" => 1_000,
        "MB" => 1_000_000,
        "GB" => 1_000_000_000,
        "TB" => 1_000_000_000_000,
        "KIB" => 1_024,
        "MIB" => 1_048_576,
        "GIB" => 1_073_741_824,
        "TIB" => 1_099_511_627_776,
        _ => return None,
    };

    Some((num * multiplier as f64) as u64)
}
