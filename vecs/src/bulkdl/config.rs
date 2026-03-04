// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Top-level YAML configuration
#[derive(Debug, Deserialize)]
pub struct Config {
    pub datasets: Vec<Dataset>,
}

/// A single dataset to download
#[derive(Debug, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub baseurl: String,
    pub tokens: HashMap<String, TokenSpec>,
    pub savedir: String,
    #[serde(default = "default_tries")]
    pub tries: u32,
    #[serde(default = "default_concurrency")]
    pub concurrency: u32,
}

fn default_tries() -> u32 {
    3
}

fn default_concurrency() -> u32 {
    4
}

/// Token range specification, e.g. `[0..409]`
#[derive(Debug)]
pub struct TokenSpec {
    pub start: i64,
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

/// Parses a token range like `[0..409]` into start/end inclusive
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

/// Status file tracking completed downloads
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct StatusFile {
    pub completed: Vec<String>,
}

impl StatusFile {
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

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }
}
