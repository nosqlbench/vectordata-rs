// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Publish URL check.
//!
//! Verifies that a valid `.publish_url` file exists in the directory hierarchy,
//! that it contains a well-formed URL, and that the URL's scheme maps to a
//! supported publish transport.

use std::path::{Path, PathBuf};

use super::CheckResult;

/// Name of the publish URL binding file.
const PUBLISH_FILE: &str = ".publish_url";

/// Supported publish transport schemes.
const SUPPORTED_SCHEMES: &[&str] = &["s3"];

/// Check that a valid `.publish_url` file is reachable and uses a supported transport.
pub fn check(root: &Path, dataset_files: &[PathBuf]) -> CheckResult {
    let publish_path = find_publish_file(root);

    let publish_path = match publish_path {
        Some(p) => p,
        None => {
            return CheckResult::fail("publish", vec![
                format!(
                    "no .publish_url file found in '{}' or any parent directory",
                    super::rel_display(root),
                ),
            ]);
        }
    };

    let content = match std::fs::read_to_string(&publish_path) {
        Ok(c) => c,
        Err(e) => {
            return CheckResult::fail("publish", vec![
                format!("failed to read {}: {}", super::rel_display(&publish_path), e),
            ]);
        }
    };

    match parse_publish_url(&content) {
        Ok(parsed) => {
            let publish_root = publish_path.parent().unwrap_or(Path::new("."));
            let mut messages = vec![
                format!(".publish_url: {} (transport: {})", parsed.url, parsed.scheme),
            ];
            for ds in dataset_files {
                let ds_dir = ds.parent().unwrap_or(Path::new("."));
                // Path from publish root to the dataset directory
                // Compute path from publish root to dataset directory
                let rel = if let Ok(r) = ds_dir.strip_prefix(publish_root) {
                    r.to_path_buf()
                } else {
                    // Canonicalize both for cross-relative/absolute comparison
                    let abs_ds = std::fs::canonicalize(ds_dir).unwrap_or_else(|_| ds_dir.to_path_buf());
                    let abs_root = std::fs::canonicalize(publish_root).unwrap_or_else(|_| publish_root.to_path_buf());
                    abs_ds.strip_prefix(&abs_root)
                        .map(|r| r.to_path_buf())
                        .unwrap_or_else(|_| ds_dir.to_path_buf())
                };
                let rel_str = rel.to_string_lossy();
                let target = if rel_str.is_empty() || rel_str == "." {
                    parsed.url.clone()
                } else {
                    let base = parsed.url.trim_end_matches('/');
                    format!("{}/{}/", base, rel_str)
                };
                messages.push(format!("  {} -> {}", rel_str, target));
            }
            let mut result = CheckResult::ok("publish");
            result.messages = messages;
            result
        }
        Err(e) => {
            CheckResult::fail("publish", vec![
                format!("{}: {}", super::rel_display(&publish_path), e),
            ])
        }
    }
}

/// Search for `.publish_url` starting at `dir` and walking up parent directories.
///
/// Also checks for the legacy `.s3-bucket` filename for backward compatibility.
pub fn find_publish_file(dir: &Path) -> Option<PathBuf> {
    // Walk up using absolute paths internally (pop() needs them),
    // but return a relative path from the original dir.
    let abs = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    let mut current = abs.clone();
    let mut levels_up: usize = 0;

    loop {
        let candidate = current.join(PUBLISH_FILE);
        if candidate.is_file() {
            // Reconstruct as relative path: dir + "../" * levels_up + filename
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

/// Parsed publish URL with scheme validation.
#[derive(Debug)]
pub struct ParsedPublishUrl {
    /// The normalized URL (with trailing slash).
    pub url: String,
    /// The scheme (e.g., "s3").
    pub scheme: String,
}

/// Parse and validate the contents of a `.publish_url` file.
///
/// Returns the normalized URL and validated scheme on success.
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

    // Extract scheme
    let scheme_end = url.find("://")
        .ok_or_else(|| format!("no scheme found in URL: {}", url))?;
    let scheme = url[..scheme_end].to_lowercase();

    if !SUPPORTED_SCHEMES.contains(&scheme.as_str()) {
        return Err(format!(
            "unsupported transport scheme '{}' (supported: {})",
            scheme,
            SUPPORTED_SCHEMES.join(", "),
        ));
    }

    // Validate scheme-specific structure
    let after_scheme = &url[scheme_end + 3..];
    if after_scheme.is_empty() || after_scheme == "/" {
        return Err(format!("empty host/bucket in URL: {}", url));
    }

    let host = after_scheme.split('/').next().unwrap_or("");
    if host.is_empty() {
        return Err(format!("empty host/bucket in URL: {}", url));
    }

    // Normalize: ensure trailing slash
    let normalized = if url.ends_with('/') {
        url
    } else {
        format!("{}/", url)
    };

    Ok(ParsedPublishUrl { url: normalized, scheme })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_s3_url() {
        let parsed = parse_publish_url("s3://my-bucket/prefix/").unwrap();
        assert_eq!(parsed.url, "s3://my-bucket/prefix/");
        assert_eq!(parsed.scheme, "s3");
    }

    #[test]
    fn test_parse_s3_no_trailing_slash() {
        let parsed = parse_publish_url("s3://my-bucket/prefix").unwrap();
        assert_eq!(parsed.url, "s3://my-bucket/prefix/");
    }

    #[test]
    fn test_parse_s3_bucket_only() {
        let parsed = parse_publish_url("s3://my-bucket").unwrap();
        assert_eq!(parsed.url, "s3://my-bucket/");
        assert_eq!(parsed.scheme, "s3");
    }

    #[test]
    fn test_parse_with_comments() {
        let content = "# production\ns3://prod-bucket/data/\n";
        let parsed = parse_publish_url(content).unwrap();
        assert_eq!(parsed.url, "s3://prod-bucket/data/");
    }

    #[test]
    fn test_unsupported_scheme() {
        let err = parse_publish_url("ftp://server/path").unwrap_err();
        assert!(err.contains("unsupported transport"), "{}", err);
    }

    #[test]
    fn test_https_unsupported() {
        let err = parse_publish_url("https://server/path").unwrap_err();
        assert!(err.contains("unsupported transport"), "{}", err);
    }

    #[test]
    fn test_empty() {
        assert!(parse_publish_url("").is_err());
        assert!(parse_publish_url("# just a comment\n").is_err());
    }

    #[test]
    fn test_empty_host() {
        assert!(parse_publish_url("s3://").is_err());
        assert!(parse_publish_url("s3:///prefix/").is_err());
    }

    #[test]
    fn test_no_scheme() {
        assert!(parse_publish_url("just-a-path").is_err());
    }
}
