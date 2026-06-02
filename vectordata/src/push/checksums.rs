// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `SHA256SUMS` — the standard, externally-verifiable content checksum
//! file `vectordata push` materializes per directory level.
//!
//! This is separate from the internal `.mrkl` merkle sidecars: the
//! merkle tree is the streaming, chunk-level read-time verifier; the
//! `SHA256SUMS` file is the whole-file, directory-level, tool-
//! interoperable provenance artifact that anyone can verify with stock
//! `sha256sum -c SHA256SUMS`.
//!
//! SHA-256 is chosen deliberately: the repo already hashes content with
//! `sha2`/SHA-256 (the merkle scheme), so this shares one hash family
//! and adds no dependency; for the GiB-scale facet files involved,
//! checksumming is IO-bound and any SHA-1 edge is irrelevant. We
//! *generate* digests natively but *format* the file in the normative
//! `sha256sum` layout (`<hex>  <name>`).
//!
//! See `docs/design/push-command.md` — *Content checksums*.

use std::io::Read;
use std::path::Path;
use std::time::SystemTime;

use sha2::{Digest, Sha256};

use super::binding::PUBLISH_FILE;
use super::pushlog::PUSHLOG_FILE;

/// Name of the per-directory checksum file.
pub const CHECKSUMS_FILE: &str = "SHA256SUMS";

/// Files that are never themselves *content* and so never appear inside
/// a `SHA256SUMS` listing: the checksum file, the binding file, and the
/// provenance log. Everything else at a directory level (including the
/// `.mref`/`.mrkl` merkle sidecars, which readers fetch) is content.
pub fn is_sentinel(name: &str) -> bool {
    name == CHECKSUMS_FILE || name == PUBLISH_FILE || name == PUSHLOG_FILE
}

/// One `<hex>  <name>` row of a `SHA256SUMS` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChecksumEntry {
    /// Lowercase hex SHA-256 digest.
    pub hex: String,
    /// File name (relative to the directory the `SHA256SUMS` lives in).
    pub name: String,
}

/// A parsed `SHA256SUMS` file, sorted by name for deterministic output.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChecksumFile {
    pub entries: Vec<ChecksumEntry>,
}

impl ChecksumFile {
    /// Parse the normative `sha256sum` format. Tolerates the optional
    /// binary-mode `*` marker (`<hex> *<name>`) and blank lines.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut entries = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim_end();
            if line.trim().is_empty() {
                continue;
            }
            // hex, then whitespace, then (optionally '*') name.
            let (hex, rest) = line
                .split_once(char::is_whitespace)
                .ok_or_else(|| format!("malformed SHA256SUMS line {}: {line:?}", n + 1))?;
            let name = rest.trim_start().trim_start_matches('*').to_string();
            if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("malformed digest on line {}: {hex:?}", n + 1));
            }
            entries.push(ChecksumEntry { hex: hex.to_lowercase(), name });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(ChecksumFile { entries })
    }

    /// Render in normative `sha256sum` format (two spaces between digest
    /// and name), names sorted for byte-stable output.
    pub fn render(&self) -> String {
        let mut sorted = self.entries.clone();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        let mut out = String::new();
        for e in &sorted {
            out.push_str(&e.hex);
            out.push_str("  ");
            out.push_str(&e.name);
            out.push('\n');
        }
        out
    }

    /// The recorded digest for `name`, if listed.
    pub fn digest_of(&self, name: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.hex.as_str())
    }

    /// The set of names this file describes.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }
}

/// Stream a file through SHA-256 without loading it into memory.
/// Returns the lowercase hex digest.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// SHA-256 hex of an in-memory byte slice.
pub fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// The content files at a single directory level (non-recursive),
/// excluding sentinels and subdirectories. Names are returned sorted.
pub fn content_files(dir: &Path) -> std::io::Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_sentinel(&name) {
            continue;
        }
        names.push(name);
    }
    names.sort();
    Ok(names)
}

/// Whether the directory has any content files (and therefore needs a
/// `SHA256SUMS`).
pub fn has_content(dir: &Path) -> std::io::Result<bool> {
    Ok(!content_files(dir)?.is_empty())
}

/// Freshness verdict for a directory's `SHA256SUMS` against the content
/// at that level, per the mtime invariant: the checksum file must be at
/// least as new as every file it describes, and its listed set must
/// match the content present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Freshness {
    /// No `SHA256SUMS` exists yet at this level.
    Missing,
    /// Present and current — safe to ship as-is.
    Current,
    /// Present but stale; `reason` explains why.
    Stale { reason: String },
}

/// Evaluate the `SHA256SUMS` freshness for `dir` against the exact set
/// of files `expected` to be published from it (sorted). The expected
/// set is supplied by the caller — `content_files(dir)` for a standalone
/// scan, or a filtered subset when a producer (e.g. `veks`) selects what
/// to publish — so the checksum file always describes precisely what is
/// shipped.
pub fn freshness(dir: &Path, expected: &[String]) -> std::io::Result<Freshness> {
    let sums_path = dir.join(CHECKSUMS_FILE);
    if !sums_path.is_file() {
        return Ok(Freshness::Missing);
    }
    let listed = match ChecksumFile::parse(&std::fs::read_to_string(&sums_path)?) {
        Ok(c) => c,
        Err(e) => return Ok(Freshness::Stale { reason: e }),
    };
    let listed_names: Vec<String> = listed.names().iter().map(|s| s.to_string()).collect();
    if listed_names != expected {
        return Ok(Freshness::Stale {
            reason: "listed file set does not match the published set".to_string(),
        });
    }
    let sums_mtime = mtime(&sums_path)?;
    for name in expected {
        let m = mtime(&dir.join(name))?;
        if m > sums_mtime {
            return Ok(Freshness::Stale {
                reason: format!("'{name}' is newer than {CHECKSUMS_FILE}"),
            });
        }
    }
    Ok(Freshness::Current)
}

/// Recompute and write `SHA256SUMS` for `dir` over exactly `names`, then
/// enforce the mtime invariant (checksum file ≥ every described file).
/// Returns the parsed, freshly written checksum file.
pub fn generate(dir: &Path, names: &[String]) -> std::io::Result<ChecksumFile> {
    let mut entries = Vec::with_capacity(names.len());
    let mut newest = SystemTime::UNIX_EPOCH;
    for name in names {
        let path = dir.join(name);
        entries.push(ChecksumEntry { hex: sha256_file(&path)?, name: name.clone() });
        newest = newest.max(mtime(&path)?);
    }
    let present = names;
    let cf = ChecksumFile { entries };
    let sums_path = dir.join(CHECKSUMS_FILE);
    std::fs::write(&sums_path, cf.render())?;
    // Anchor the checksum file's mtime to the newest *described file*,
    // not to `SystemTime::now()`. The mtime invariant (checksums >=
    // every described file) is satisfied by equality with the newest,
    // and — critically — staleness is then judged file-clock against
    // file-clock. Using `now()` here is unreliable: on hosts whose
    // realtime clock jitters against the filesystem mtime clock, a
    // freshly stamped `now()` can land *after* a content write that
    // happens later in program order, masking a real change.
    let target = if present.is_empty() { SystemTime::now() } else { newest };
    let _ = filetime::set_file_mtime(&sums_path, filetime::FileTime::from_system_time(target));
    Ok(cf)
}

fn mtime(path: &Path) -> std::io::Result<SystemTime> {
    std::fs::metadata(path)?.modified()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("vd-sums-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn parse_render_roundtrip_normative() {
        let text = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  a.bin\n\
                    da39a3ee5e6b4b0d3255bfef95601890afd80709  bad\n";
        // second line has a 40-char (sha1) digest → should be rejected
        assert!(ChecksumFile::parse(text).is_err());

        let good = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  a.bin\n";
        let cf = ChecksumFile::parse(good).unwrap();
        assert_eq!(cf.digest_of("a.bin").unwrap().len(), 64);
        assert_eq!(cf.render(), good);
    }

    #[test]
    fn tolerates_binary_marker() {
        let cf = ChecksumFile::parse(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 *a.bin\n",
        )
        .unwrap();
        assert_eq!(cf.entries[0].name, "a.bin");
    }

    #[test]
    fn generate_then_current_then_stale() {
        let d = tmpdir("gen");
        std::fs::write(d.join("a.bin"), b"hello").unwrap();
        std::fs::write(d.join("b.bin"), b"world").unwrap();
        let names = content_files(&d).unwrap();
        let cf = generate(&d, &names).unwrap();
        assert_eq!(cf.names(), vec!["a.bin", "b.bin"]);
        assert_eq!(cf.digest_of("a.bin").unwrap(), &sha256_bytes(b"hello"));
        assert_eq!(freshness(&d, &names).unwrap(), Freshness::Current);

        // Touch a described file into the future → stale.
        let future = filetime::FileTime::from_unix_time(
            filetime::FileTime::now().unix_seconds() + 100,
            0,
        );
        filetime::set_file_mtime(d.join("a.bin"), future).unwrap();
        assert!(matches!(freshness(&d, &names).unwrap(), Freshness::Stale { .. }));

        // Add a new file not in the listing → stale (set mismatch).
        generate(&d, &names).unwrap();
        std::fs::write(d.join("c.bin"), b"new").unwrap();
        let names2 = content_files(&d).unwrap();
        assert!(matches!(freshness(&d, &names2).unwrap(), Freshness::Stale { .. }));

        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn sentinels_excluded_from_content() {
        let d = tmpdir("sentinel");
        std::fs::write(d.join("data.fvec"), b"x").unwrap();
        std::fs::write(d.join(PUBLISH_FILE), b"s3://b/p/").unwrap();
        std::fs::write(d.join(PUSHLOG_FILE), b"").unwrap();
        std::fs::write(d.join(CHECKSUMS_FILE), b"").unwrap();
        assert_eq!(content_files(&d).unwrap(), vec!["data.fvec"]);
        std::fs::remove_dir_all(&d).ok();
    }
}
