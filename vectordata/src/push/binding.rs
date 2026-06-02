// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `.publish_url` binding — the contract that ties a local dataset
//! directory to the remote endpoint it belongs to.
//!
//! This is the canonical implementation for the workspace. `veks`
//! depends on `vectordata` (not the other way round), so the binding
//! logic lives here and `veks` can delegate to it. It generalizes the
//! original `veks` `.publish_url` check from S3-only to every transport
//! `vectordata push` speaks: `s3://`, `https://`, `http://`, `file://`.
//!
//! See `docs/design/push-command.md` — *Binding: `.publish_url` is the
//! contract*.

use std::path::{Path, PathBuf};

/// Name of the publish-URL binding file.
pub const PUBLISH_FILE: &str = ".publish_url";

/// Transport schemes a `.publish_url` may name. This is the write side
/// of the read transport's scheme set, widened beyond the original
/// S3-only `veks` check.
pub const SUPPORTED_SCHEMES: &[&str] = &["s3", "https", "http", "file"];

/// A parsed, validated `.publish_url`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedPublishUrl {
    /// The normalized URL (always with a trailing slash).
    pub url: String,
    /// The scheme (e.g. `s3`, `https`, `file`).
    pub scheme: String,
}

/// Parse and validate the contents of a `.publish_url` file.
///
/// Blank lines and `#` comments are ignored; the remaining text is the
/// URL. The URL is normalized to end in a single `/` so it composes
/// cleanly with relative keys.
pub fn parse_publish_url(content: &str) -> Result<ParsedPublishUrl, String> {
    let url: String = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join("");
    let url = url.trim().to_string();

    if url.is_empty() {
        return Err("file is empty".to_string());
    }

    let scheme_end = url
        .find("://")
        .ok_or_else(|| format!("no scheme found in URL: {url}"))?;
    let scheme = url[..scheme_end].to_lowercase();

    if !SUPPORTED_SCHEMES.contains(&scheme.as_str()) {
        return Err(format!(
            "unsupported transport scheme '{}' (supported: {})",
            scheme,
            SUPPORTED_SCHEMES.join(", "),
        ));
    }

    // Validate scheme-specific host/authority. `file://` is allowed to
    // address an absolute path (`file:///srv/data/`) whose authority is
    // empty, so only the network schemes require a non-empty host.
    let after_scheme = &url[scheme_end + 3..];
    let host = after_scheme.split('/').next().unwrap_or("");
    if scheme != "file" && host.is_empty() {
        return Err(format!("empty host/bucket in URL: {url}"));
    }
    if after_scheme.is_empty() {
        return Err(format!("empty path in URL: {url}"));
    }

    let normalized = if url.ends_with('/') {
        url
    } else {
        format!("{url}/")
    };

    Ok(ParsedPublishUrl { url: normalized, scheme })
}

/// Search for `.publish_url` starting at `dir` and walking up parent
/// directories. Returns a path relative to `dir` (so the caller can
/// display it the way the user typed it).
///
/// Uses the logical (non-canonicalized) path so traversal stays within
/// the user's view of the tree even across symlinks.
pub fn find_publish_file(dir: &Path) -> Option<PathBuf> {
    let abs = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    let mut current = abs;
    let mut levels_up: usize = 0;
    loop {
        if current.join(PUBLISH_FILE).is_file() {
            let mut rel = dir.to_path_buf();
            for _ in 0..levels_up {
                rel = rel.join("..");
            }
            return Some(rel.join(PUBLISH_FILE));
        }
        if !current.pop() {
            return None;
        }
        levels_up += 1;
    }
}

/// Read and parse the `.publish_url` at or above `dir`, if any.
///
/// Returns `Ok(None)` when no binding file exists, `Ok(Some(..))` for a
/// valid one, and `Err(..)` when a file exists but is malformed (which
/// the caller should treat as a hard stop rather than "unbound").
pub fn read_binding(dir: &Path) -> Result<Option<(PathBuf, ParsedPublishUrl)>, String> {
    let Some(path) = find_publish_file(dir) else {
        return Ok(None);
    };
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let parsed = parse_publish_url(&content)
        .map_err(|e| format!("invalid {}: {e}", path.display()))?;
    Ok(Some((path, parsed)))
}

/// Outcome of reconciling an existing local binding with a `--to`
/// override (or its absence).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Binding {
    /// A binding already existed and is what we will use. `to_write` is
    /// `None` — nothing to persist.
    Existing(ParsedPublishUrl),
    /// No binding existed; `--to` supplied one that must be persisted
    /// into the source directory.
    Staged(ParsedPublishUrl),
}

impl Binding {
    /// The endpoint this binding resolves to.
    pub fn url(&self) -> &ParsedPublishUrl {
        match self {
            Binding::Existing(u) | Binding::Staged(u) => u,
        }
    }
}

/// Reconcile a local `.publish_url` (if present) with an optional
/// `--to` override, per the resolution rules in the design:
///
/// - neither present → error (refuse to invent a destination);
/// - binding present, no `--to` → use the binding;
/// - `--to` present, no binding → stage it for persistence;
/// - both present and equal → use the binding;
/// - both present and different → conflict (hard stop).
pub fn reconcile(
    existing: Option<ParsedPublishUrl>,
    to: Option<&str>,
) -> Result<Binding, String> {
    let to = to.map(parse_publish_url).transpose()?;
    match (existing, to) {
        (None, None) => Err(
            "no destination: there is no .publish_url here (or in a parent) and no --to was given.\n\
             Bind this directory with, e.g.:\n  \
             echo 'https://host/path/' > .publish_url"
                .to_string(),
        ),
        (Some(b), None) => Ok(Binding::Existing(b)),
        (None, Some(t)) => Ok(Binding::Staged(t)),
        (Some(b), Some(t)) => {
            if b.url == t.url {
                Ok(Binding::Existing(b))
            } else {
                Err(format!(
                    "binding conflict: this data is already bound to\n  {}\nbut --to names\n  {}\n\
                     Re-binding is deliberate; remove or edit the .publish_url to change it.",
                    b.url, t.url,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_supported_scheme() {
        for (raw, scheme) in [
            ("s3://bucket/prefix", "s3"),
            ("https://host/path", "https"),
            ("http://host/path", "http"),
            ("file:///srv/data", "file"),
        ] {
            let p = parse_publish_url(raw).unwrap();
            assert_eq!(p.scheme, scheme);
            assert!(p.url.ends_with('/'), "{} not normalized", p.url);
        }
    }

    #[test]
    fn normalizes_trailing_slash() {
        assert_eq!(parse_publish_url("s3://b/p").unwrap().url, "s3://b/p/");
        assert_eq!(parse_publish_url("s3://b/p/").unwrap().url, "s3://b/p/");
    }

    #[test]
    fn rejects_unsupported_and_empty() {
        assert!(parse_publish_url("ftp://h/p").is_err());
        assert!(parse_publish_url("").is_err());
        assert!(parse_publish_url("# only a comment\n").is_err());
        assert!(parse_publish_url("no-scheme").is_err());
        assert!(parse_publish_url("s3://").is_err());
    }

    #[test]
    fn file_scheme_allows_empty_authority() {
        // file:///abs/path has an empty authority but a real path.
        let p = parse_publish_url("file:///abs/path").unwrap();
        assert_eq!(p.url, "file:///abs/path/");
    }

    #[test]
    fn reconcile_rules() {
        let s3 = parse_publish_url("s3://b/p").unwrap();
        // neither → error
        assert!(reconcile(None, None).is_err());
        // binding only → existing
        assert_eq!(
            reconcile(Some(s3.clone()), None).unwrap(),
            Binding::Existing(s3.clone())
        );
        // --to only → staged
        assert_eq!(
            reconcile(None, Some("s3://b/p")).unwrap(),
            Binding::Staged(s3.clone())
        );
        // both equal → existing
        assert_eq!(
            reconcile(Some(s3.clone()), Some("s3://b/p/")).unwrap(),
            Binding::Existing(s3)
        );
        // both different → conflict
        assert!(reconcile(
            Some(parse_publish_url("s3://b/p").unwrap()),
            Some("s3://b/other")
        )
        .is_err());
    }

    #[test]
    fn finds_binding_walking_up() {
        let tmp = std::env::temp_dir().join(format!(
            "vd-binding-test-{}",
            std::process::id()
        ));
        let nested = tmp.join("a/b/c");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(tmp.join(PUBLISH_FILE), "s3://bucket/root/\n").unwrap();
        let found = find_publish_file(&nested).expect("should find ancestor binding");
        assert!(found.ends_with(PUBLISH_FILE));
        let (_, parsed) = read_binding(&nested).unwrap().unwrap();
        assert_eq!(parsed.url, "s3://bucket/root/");
        std::fs::remove_dir_all(&tmp).ok();
    }
}
