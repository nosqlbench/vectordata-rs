// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Data source types with sugar parsing for dataset profiles.
//!
//! A `DSSource` identifies a file path plus optional namespace and window
//! (record range). Sources can be specified as bare strings with inline
//! sugar notation or as explicit maps in YAML.
//!
//! ## Sugar forms
//!
//! | Form | Example |
//! |------|---------|
//! | Bare string | `"file.fvec"` |
//! | String + bracket window | `"file.fvec[0..1M]"` |
//! | String + paren window | `"file.fvec(0..1000)"` |
//! | String + namespace | `"file.slab:content"` |
//! | String + namespace + window | `"file.slab:ns:[0..1K]"` |
//! | Map with `source` | `{ source: "f.fvec" }` |
//! | Map with `path` | `{ path: "f.fvec" }` |
//! | Map with window | `{ source: "f.fvec", window: "0..1000" }` |

use std::fmt;

use serde::{Deserialize, Serialize};
use serde::de::{self, Visitor};

/// Data source with optional namespace and window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DSSource {
    /// Path to the source file.
    pub path: String,
    /// Optional namespace within the source (e.g., slab page name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// Window (record range) restriction.
    #[serde(skip_serializing_if = "DSWindow::is_empty")]
    pub window: DSWindow,
}

/// A window is a list of intervals (empty = ALL data).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct DSWindow(pub Vec<DSInterval>);

/// Half-open interval `[min_incl, max_excl)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DSInterval {
    /// Inclusive lower bound.
    pub min_incl: u64,
    /// Exclusive upper bound.
    pub max_excl: u64,
}

impl DSWindow {
    /// Returns `true` if no intervals are specified (meaning all data).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for DSInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.min_incl, self.max_excl)
    }
}

impl fmt::Display for DSWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            write!(f, "ALL")
        } else {
            let parts: Vec<String> = self.0.iter().map(|i| i.to_string()).collect();
            write!(f, "[{}]", parts.join(", "))
        }
    }
}

impl fmt::Display for DSSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path)?;
        if let Some(ref ns) = self.namespace {
            write!(f, ":{}", ns)?;
        }
        if !self.window.is_empty() {
            write!(f, "[{}]", self.window)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Interval parsing
// ---------------------------------------------------------------------------

/// Parse a unit suffix and return the multiplier.
///
/// Supports `K` (×1,000), `M` (×1,000,000), `B`/`G` (×1,000,000,000),
/// matching Java's `UnitConversions.longCountFor`.
fn parse_number_with_suffix(s: &str) -> Result<u64, String> {
    let s = s.replace('_', "");
    if s.is_empty() {
        return Err("empty number".to_string());
    }

    let (num_part, multiplier) = match s.as_bytes().last() {
        Some(b'K' | b'k') => (&s[..s.len() - 1], 1_000u64),
        Some(b'M' | b'm') => (&s[..s.len() - 1], 1_000_000u64),
        Some(b'B' | b'b' | b'G' | b'g') => (&s[..s.len() - 1], 1_000_000_000u64),
        _ => (s.as_str(), 1u64),
    };

    let n: u64 = num_part
        .parse()
        .map_err(|e| format!("invalid number '{}': {}", num_part, e))?;
    Ok(n * multiplier)
}

/// Parse a single interval from a string.
///
/// Accepted forms:
/// - `"1M"` → `[0, 1_000_000)`
/// - `"0..1000"` → `[0, 1000)`
/// - `"[0..1000)"` → `[0, 1000)`
/// - `"0..1K"` → `[0, 1000)`
fn parse_interval(s: &str) -> Result<DSInterval, String> {
    let s = s.trim();
    // Strip optional surrounding brackets: [0..1000) or [0..1000]
    let s = s.strip_prefix('[').unwrap_or(s);
    let s = s.strip_suffix(')').or_else(|| s.strip_suffix(']')).unwrap_or(s);
    let s = s.trim();

    if let Some((left, right)) = s.split_once("..") {
        let min_incl = parse_number_with_suffix(left.trim())?;
        let max_excl = parse_number_with_suffix(right.trim())?;
        Ok(DSInterval { min_incl, max_excl })
    } else {
        // Single number = shorthand for 0..N
        let max_excl = parse_number_with_suffix(s)?;
        Ok(DSInterval {
            min_incl: 0,
            max_excl,
        })
    }
}

/// Parse a window specification from a string.
///
/// Can be a single interval or comma-separated list of intervals.
fn parse_window(s: &str) -> Result<DSWindow, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(DSWindow(vec![]));
    }

    // Strip outer brackets if present: "[0..1K, 2K..3K]"
    let inner = if s.starts_with('[') && s.ends_with(']') {
        &s[1..s.len() - 1]
    } else {
        s
    };

    let intervals: Result<Vec<DSInterval>, String> =
        inner.split(',').map(|part| parse_interval(part.trim())).collect();
    Ok(DSWindow(intervals?))
}

// ---------------------------------------------------------------------------
// Source string sugar parsing
// ---------------------------------------------------------------------------

/// Parse a source string with sugar notation.
///
/// Forms:
/// - `"file.fvec"` → path only
/// - `"file.fvec[0..1M]"` → path + bracket window
/// - `"file.fvec(0..1000)"` → path + paren window
/// - `"file.slab:content"` → path + namespace
/// - `"file.slab:ns:[0..1K]"` → path + namespace + window
pub(crate) fn parse_source_string(s: &str) -> Result<DSSource, String> {
    let s = s.trim();

    // Check for bracket window: "file.fvec[0..1M]" or "file.slab:ns:[0..1K]"
    if let Some(bracket_start) = s.find('[') {
        if s.ends_with(']') {
            let mut path_part = &s[..bracket_start];
            let window_part = &s[bracket_start + 1..s.len() - 1];

            // Strip trailing colon before bracket (ns separator in "path:ns:[window]")
            if path_part.ends_with(':') {
                path_part = &path_part[..path_part.len() - 1];
            }

            let (path, namespace) = split_namespace(path_part);
            let window = parse_window(window_part)?;
            return Ok(DSSource {
                path: path.to_string(),
                namespace,
                window,
            });
        }
    }

    // Check for paren window: "file.fvec(0..1000)"
    if let Some(paren_start) = s.find('(') {
        if s.ends_with(')') {
            let mut path_part = &s[..paren_start];
            let window_part = &s[paren_start + 1..s.len() - 1];

            if path_part.ends_with(':') {
                path_part = &path_part[..path_part.len() - 1];
            }

            let (path, namespace) = split_namespace(path_part);
            let window = parse_window(window_part)?;
            return Ok(DSSource {
                path: path.to_string(),
                namespace,
                window,
            });
        }
    }

    // No window — check for namespace: "file.slab:content"
    let (path, namespace) = split_namespace(s);
    Ok(DSSource {
        path: path.to_string(),
        namespace,
        window: DSWindow::default(),
    })
}

/// Split a path string on the last colon that follows a dot-extension,
/// treating the part after the colon as a namespace.
///
/// `"file.slab:content"` → `("file.slab", Some("content"))`
/// `"file.fvec"` → `("file.fvec", None)`
/// `"../path/file.slab:ns"` → `("../path/file.slab", Some("ns"))`
fn split_namespace(s: &str) -> (&str, Option<String>) {
    // Find the last colon. If the part before it contains a dot (extension),
    // treat it as namespace separator. Skip Windows drive letters like C:
    if let Some(colon_pos) = s.rfind(':') {
        let before = &s[..colon_pos];
        let after = &s[colon_pos + 1..];
        // Only treat as namespace if there's a dot before the colon (extension)
        // and the part after colon is non-empty and doesn't look like a path
        if before.contains('.') && !after.is_empty() && !after.contains('/') && !after.contains('\\') {
            return (before, Some(after.to_string()));
        }
    }
    (s, None)
}

// ---------------------------------------------------------------------------
// Serde implementations
// ---------------------------------------------------------------------------

impl<'de> Deserialize<'de> for DSInterval {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct IntervalVisitor;

        impl<'de> Visitor<'de> for IntervalVisitor {
            type Value = DSInterval;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("an interval string like '0..1000' or a number")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<DSInterval, E> {
                parse_interval(v).map_err(de::Error::custom)
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<DSInterval, E> {
                Ok(DSInterval {
                    min_incl: 0,
                    max_excl: v,
                })
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<DSInterval, E> {
                if v < 0 {
                    return Err(de::Error::custom("interval bound cannot be negative"));
                }
                Ok(DSInterval {
                    min_incl: 0,
                    max_excl: v as u64,
                })
            }
        }

        deserializer.deserialize_any(IntervalVisitor)
    }
}

impl<'de> Deserialize<'de> for DSWindow {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct WindowVisitor;

        impl<'de> Visitor<'de> for WindowVisitor {
            type Value = DSWindow;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a window string, number, or list of intervals")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<DSWindow, E> {
                parse_window(v).map_err(de::Error::custom)
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<DSWindow, E> {
                Ok(DSWindow(vec![DSInterval {
                    min_incl: 0,
                    max_excl: v,
                }]))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<DSWindow, E> {
                if v < 0 {
                    return Err(de::Error::custom("window bound cannot be negative"));
                }
                Ok(DSWindow(vec![DSInterval {
                    min_incl: 0,
                    max_excl: v as u64,
                }]))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<DSWindow, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut intervals = Vec::new();
                while let Some(interval) = seq.next_element::<DSInterval>()? {
                    intervals.push(interval);
                }
                Ok(DSWindow(intervals))
            }
        }

        deserializer.deserialize_any(WindowVisitor)
    }
}

impl<'de> Deserialize<'de> for DSSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct SourceVisitor;

        impl<'de> Visitor<'de> for SourceVisitor {
            type Value = DSSource;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a source string or map with 'source'/'path' key")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<DSSource, E> {
                parse_source_string(v).map_err(de::Error::custom)
            }

            fn visit_map<M>(self, mut map: M) -> Result<DSSource, M::Error>
            where
                M: de::MapAccess<'de>,
            {
                let mut path: Option<String> = None;
                let mut namespace: Option<String> = None;
                let mut window: Option<DSWindow> = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "source" | "path" => {
                            path = Some(map.next_value()?);
                        }
                        "namespace" | "ns" => {
                            namespace = Some(map.next_value()?);
                        }
                        "window" => {
                            window = Some(map.next_value()?);
                        }
                        _ => {
                            // Ignore unknown keys
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                let path = path.ok_or_else(|| {
                    de::Error::custom("source map must have 'source' or 'path' key")
                })?;

                Ok(DSSource {
                    path,
                    namespace,
                    window: window.unwrap_or_default(),
                })
            }
        }

        deserializer.deserialize_any(SourceVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_number_with_suffix --

    #[test]
    fn test_parse_plain_number() {
        assert_eq!(parse_number_with_suffix("1000").unwrap(), 1000);
    }

    #[test]
    fn test_parse_k_suffix() {
        assert_eq!(parse_number_with_suffix("1K").unwrap(), 1_000);
        assert_eq!(parse_number_with_suffix("5k").unwrap(), 5_000);
    }

    #[test]
    fn test_parse_m_suffix() {
        assert_eq!(parse_number_with_suffix("1M").unwrap(), 1_000_000);
        assert_eq!(parse_number_with_suffix("10m").unwrap(), 10_000_000);
    }

    #[test]
    fn test_parse_b_g_suffix() {
        assert_eq!(parse_number_with_suffix("1B").unwrap(), 1_000_000_000);
        assert_eq!(parse_number_with_suffix("1G").unwrap(), 1_000_000_000);
    }

    #[test]
    fn test_parse_underscore_separator() {
        assert_eq!(parse_number_with_suffix("1_000_000").unwrap(), 1_000_000);
        assert_eq!(parse_number_with_suffix("1_000").unwrap(), 1_000);
    }

    // -- parse_interval --

    #[test]
    fn test_parse_interval_range() {
        let i = parse_interval("0..1000").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 1000);
    }

    #[test]
    fn test_parse_interval_with_suffix() {
        let i = parse_interval("0..1M").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 1_000_000);
    }

    #[test]
    fn test_parse_interval_bracketed() {
        let i = parse_interval("[0..1000)").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 1000);
    }

    #[test]
    fn test_parse_interval_single_number() {
        let i = parse_interval("1M").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 1_000_000);
    }

    #[test]
    fn test_parse_interval_with_underscores() {
        let i = parse_interval("0..1_000").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 1_000);
    }

    // -- parse_window --

    #[test]
    fn test_parse_window_single() {
        let w = parse_window("0..1000").unwrap();
        assert_eq!(w.0.len(), 1);
        assert_eq!(w.0[0].min_incl, 0);
        assert_eq!(w.0[0].max_excl, 1000);
    }

    #[test]
    fn test_parse_window_bracketed_list() {
        let w = parse_window("[0..1K, 2K..3K]").unwrap();
        assert_eq!(w.0.len(), 2);
        assert_eq!(w.0[0].max_excl, 1000);
        assert_eq!(w.0[1].min_incl, 2000);
        assert_eq!(w.0[1].max_excl, 3000);
    }

    #[test]
    fn test_parse_window_empty() {
        let w = parse_window("").unwrap();
        assert!(w.is_empty());
    }

    // -- parse_source_string --

    #[test]
    fn test_source_bare_string() {
        let s = parse_source_string("file.fvec").unwrap();
        assert_eq!(s.path, "file.fvec");
        assert_eq!(s.namespace, None);
        assert!(s.window.is_empty());
    }

    #[test]
    fn test_source_bracket_window() {
        let s = parse_source_string("file.fvec[0..1M]").unwrap();
        assert_eq!(s.path, "file.fvec");
        assert_eq!(s.namespace, None);
        assert_eq!(s.window.0.len(), 1);
        assert_eq!(s.window.0[0].max_excl, 1_000_000);
    }

    #[test]
    fn test_source_paren_window() {
        let s = parse_source_string("file.fvec(0..1000)").unwrap();
        assert_eq!(s.path, "file.fvec");
        assert_eq!(s.window.0[0].max_excl, 1000);
    }

    #[test]
    fn test_source_namespace() {
        let s = parse_source_string("file.slab:content").unwrap();
        assert_eq!(s.path, "file.slab");
        assert_eq!(s.namespace.as_deref(), Some("content"));
        assert!(s.window.is_empty());
    }

    #[test]
    fn test_source_namespace_and_window() {
        let s = parse_source_string("file.slab:ns:[0..1K]").unwrap();
        assert_eq!(s.path, "file.slab");
        assert_eq!(s.namespace.as_deref(), Some("ns"));
        assert_eq!(s.window.0.len(), 1);
        assert_eq!(s.window.0[0].max_excl, 1000);
    }

    #[test]
    fn test_source_yaml_bare_string() {
        let s: DSSource = serde_yaml::from_str("\"file.fvec\"").unwrap();
        assert_eq!(s.path, "file.fvec");
        assert!(s.window.is_empty());
    }

    #[test]
    fn test_source_yaml_map_with_source() {
        let yaml = r#"
source: f.fvec
"#;
        let s: DSSource = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(s.path, "f.fvec");
    }

    #[test]
    fn test_source_yaml_map_with_path() {
        let yaml = r#"
path: f.fvec
"#;
        let s: DSSource = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(s.path, "f.fvec");
    }

    #[test]
    fn test_source_yaml_map_with_window() {
        let yaml = r#"
source: f.fvec
window: "0..1000"
"#;
        let s: DSSource = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(s.path, "f.fvec");
        assert_eq!(s.window.0.len(), 1);
        assert_eq!(s.window.0[0].max_excl, 1000);
    }

    #[test]
    fn test_window_yaml_number() {
        let w: DSWindow = serde_yaml::from_str("1000000").unwrap();
        assert_eq!(w.0.len(), 1);
        assert_eq!(w.0[0].min_incl, 0);
        assert_eq!(w.0[0].max_excl, 1_000_000);
    }

    #[test]
    fn test_window_yaml_list() {
        let yaml = r#"
- "0..1000"
- "2000..3000"
"#;
        let w: DSWindow = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(w.0.len(), 2);
    }

    #[test]
    fn test_interval_yaml_string() {
        let i: DSInterval = serde_yaml::from_str("\"0..1K\"").unwrap();
        assert_eq!(i.min_incl, 0);
        assert_eq!(i.max_excl, 1000);
    }
}
