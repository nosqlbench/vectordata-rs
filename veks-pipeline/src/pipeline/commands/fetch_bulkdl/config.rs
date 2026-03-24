// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! YAML configuration types for `veks bulkdl`.
//!
//! A config file lists one or more [`Dataset`] entries, each with a URL
//! template, token ranges, and download parameters. See [`Config`] for
//! the top-level structure.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Top-level YAML configuration for `veks bulkdl`.
///
/// Contains one or more dataset entries to download.
#[derive(Debug, Deserialize)]
pub struct Config {
    /// Datasets to download, processed sequentially.
    pub datasets: Vec<Dataset>,
}

/// A single dataset to download.
///
/// The `baseurl` may contain `${token}` placeholders that are expanded
/// using the corresponding [`TokenSpec`] ranges to produce the full
/// list of URLs.
#[derive(Debug, Deserialize)]
pub struct Dataset {
    /// Human-readable dataset name (used in progress display).
    pub name: String,
    /// URL template with `${token}` placeholders.
    pub baseurl: String,
    /// Named token ranges for URL expansion.
    pub tokens: HashMap<String, TokenSpec>,
    /// Local directory where downloaded files are saved.
    pub savedir: String,
    /// Maximum download attempts per file (default: 3).
    #[serde(default = "default_tries")]
    pub tries: u32,
    /// Maximum concurrent downloads for this dataset (default: 4).
    #[serde(default = "default_concurrency")]
    pub concurrency: u32,
}

fn default_tries() -> u32 {
    3
}

fn default_concurrency() -> u32 {
    4
}

/// Inclusive integer range for URL template expansion.
///
/// Parsed from bracket notation like `[0..409]`. Both `start` and `end`
/// are inclusive, so `[0..2]` yields values `0, 1, 2`.
#[derive(Debug)]
pub struct TokenSpec {
    /// First value in the range (inclusive).
    pub start: i64,
    /// Last value in the range (inclusive).
    pub end: i64,
}

impl<'de> Deserialize<'de> for TokenSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        parse_token_spec(&s).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "invalid token spec '{}', expected format like [0..409]",
                s
            ))
        })
    }
}

/// Parse a token range string like `[0..409]` into an inclusive [`TokenSpec`].
///
/// Returns `None` if the string is not in `[start..end]` format.
pub fn parse_token_spec(s: &str) -> Option<TokenSpec> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return None;
    }
    let inner = &s[1..s.len() - 1];
    let parts: Vec<&str> = inner.splitn(2, "..").collect();
    if parts.len() != 2 {
        return None;
    }
    let start: i64 = parts[0].trim().parse().ok()?;
    let end: i64 = parts[1].trim().parse().ok()?;
    Some(TokenSpec { start, end })
}

/// Persistent JSON file tracking which files have been downloaded.
///
/// Stored as `.down-rs-status.json` in the dataset's save directory.
/// Used to skip already-completed files on subsequent runs.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct StatusFile {
    /// Filenames (not full paths) that have been successfully downloaded.
    pub completed: Vec<String>,
}

impl StatusFile {
    /// Load from disk, returning an empty status if the file is missing or corrupt.
    pub fn load(path: &Path) -> Self {
        if path.exists() {
            match fs::read_to_string(path) {
                Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
                Err(_) => Self::default(),
            }
        } else {
            Self::default()
        }
    }

    /// Write the current status to disk as pretty-printed JSON.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }
}
