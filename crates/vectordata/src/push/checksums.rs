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
    /// binary-mode `*` marker (`<hex> *<name>`) and blank lines, and
    /// understands the GNU coreutils backslash-escaped form (a leading
    /// `\` on the line, with `\\` and `\n` escapes in the name) used when
    /// a file name contains a backslash or newline.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut entries = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim_end_matches(['\r', '\n']);
            if line.trim().is_empty() {
                continue;
            }
            // A leading backslash marks an escaped line (the name held a
            // backslash or newline); strip it before reading the digest.
            let (escaped, body) = match line.strip_prefix('\\') {
                Some(rest) => (true, rest),
                None => (false, line),
            };
            let (hex, rest) = body
                .split_once(char::is_whitespace)
                .ok_or_else(|| format!("malformed SHA256SUMS line {}: {line:?}", n + 1))?;
            if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("malformed digest on line {}: {hex:?}", n + 1));
            }
            let raw = rest.trim_start().trim_start_matches('*');
            let name = if escaped { unescape_name(raw) } else { raw.to_string() };
            entries.push(ChecksumEntry { hex: hex.to_lowercase(), name });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(ChecksumFile { entries })
    }

    /// Render in normative `sha256sum` format (two spaces between digest
    /// and name), names sorted for byte-stable output. Names containing a
    /// backslash or newline are emitted in the coreutils escaped form so
    /// the manifest stays a faithful, line-oriented, injective mapping.
    pub fn render(&self) -> String {
        let mut sorted = self.entries.clone();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        let mut out = String::new();
        for e in &sorted {
            if name_needs_escape(&e.name) {
                out.push('\\');
                out.push_str(&e.hex);
                out.push_str("  ");
                out.push_str(&escape_name(&e.name));
            } else {
                out.push_str(&e.hex);
                out.push_str("  ");
                out.push_str(&e.name);
            }
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

/// Whether a file name must use the escaped line form.
fn name_needs_escape(name: &str) -> bool {
    name.contains('\\') || name.contains('\n') || name.contains('\r')
}

/// Escape a name for the coreutils `\`-prefixed line form.
fn escape_name(name: &str) -> String {
    name.replace('\\', "\\\\").replace('\n', "\\n").replace('\r', "\\r")
}

/// Inverse of [`escape_name`].
fn unescape_name(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
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

/// What bringing `SHA256SUMS` up to date over `names` would take:
/// the digests that can be kept and the files that must be hashed.
#[derive(Debug, Clone)]
pub struct HashPlan {
    /// Entries kept from the existing sums: files the sums list whose
    /// mtime is not newer than the sums — by the mtime invariant the
    /// sums were written after them, so what was hashed is what is
    /// there.
    pub reuse: Vec<ChecksumEntry>,
    /// `(name, len)` of every file to hash: newer than the sums, or
    /// absent from them.
    pub hash: Vec<(String, u64)>,
    /// The newest described file's mtime, which the written sums are
    /// anchored to.
    pub newest: SystemTime,
}

impl HashPlan {
    /// Bytes the plan would read.
    pub fn bytes(&self) -> u64 {
        self.hash.iter().map(|(_, len)| *len).sum()
    }
}

/// Plan the update of `SHA256SUMS` in `dir` over exactly `names`
/// without reading any content: which digests are kept, which files
/// are hashed. A dry run reports this; [`generate`] executes it.
pub fn plan(dir: &Path, names: &[String]) -> std::io::Result<HashPlan> {
    let sums_path = dir.join(CHECKSUMS_FILE);
    let prior: Option<(ChecksumFile, SystemTime)> = if sums_path.is_file() {
        match ChecksumFile::parse(&std::fs::read_to_string(&sums_path)?) {
            Ok(cf) => Some((cf, mtime(&sums_path)?)),
            Err(_) => None,
        }
    } else {
        None
    };
    let mut out = HashPlan { reuse: Vec::new(), hash: Vec::new(), newest: SystemTime::UNIX_EPOCH };
    for name in names {
        let path = dir.join(name);
        let (len, m) = len_mtime(&path)?;
        out.newest = out.newest.max(m);
        if let Some((cf, sums_mtime)) = &prior
            && m <= *sums_mtime
            && let Some(hex) = cf.digest_of(name)
        {
            out.reuse.push(ChecksumEntry { hex: hex.to_string(), name: name.clone() });
            continue;
        }
        out.hash.push((name.clone(), len));
    }
    Ok(out)
}

/// Bring `SHA256SUMS` in `dir` up to date over exactly `names`: plan,
/// then execute. Returns the parsed, freshly written checksum file.
pub fn generate(dir: &Path, names: &[String]) -> std::io::Result<ChecksumFile> {
    let hash_plan = plan(dir, names)?;
    execute(dir, names, hash_plan)
}

/// Execute a [`HashPlan`]: hash the files it names, write `SHA256SUMS`
/// over exactly `names`, and enforce the mtime invariant (checksum
/// file ≥ every described file). This is the effector; the planning
/// that decides what it does is [`plan`], which a dry run shares.
///
/// Kept digests are kept, and only files newer than the sums, or
/// absent from them, are hashed. One new file in a directory no
/// longer sends every other file in it back through SHA-256: on
/// tessera that was two terabytes of base shards untouched for days,
/// rehashed for one report. Hashing runs across files in parallel and
/// reports its progress on stderr.
pub fn execute(dir: &Path, names: &[String], hash_plan: HashPlan) -> std::io::Result<ChecksumFile> {
    let mut by_name: std::collections::BTreeMap<&str, String> =
        hash_plan.reuse.iter().map(|e| (e.name.as_str(), e.hex.clone())).collect();
    if !hash_plan.hash.is_empty() {
        let work: Vec<(usize, std::path::PathBuf, u64)> = hash_plan
            .hash
            .iter()
            .enumerate()
            .map(|(i, (name, len))| (i, dir.join(name), *len))
            .collect();
        let hashed = hash_files(&work, hash_plan.reuse.len(), dir)?;
        for ((name, _), hex) in hash_plan.hash.iter().zip(hashed) {
            by_name.insert(name.as_str(), hex);
        }
    }
    let entries: Vec<ChecksumEntry> = names
        .iter()
        .map(|name| ChecksumEntry { hex: by_name.remove(name.as_str()).expect("every name resolved"), name: name.clone() })
        .collect();
    let sums_path = dir.join(CHECKSUMS_FILE);
    let cf = ChecksumFile { entries };
    std::fs::write(&sums_path, cf.render())?;
    // Anchor the checksum file's mtime to the newest *described file*,
    // not to `SystemTime::now()`. The mtime invariant (checksums >=
    // every described file) is satisfied by equality with the newest,
    // and — critically — staleness is then judged file-clock against
    // file-clock. Using `now()` here is unreliable: on hosts whose
    // realtime clock jitters against the filesystem mtime clock, a
    // freshly stamped `now()` can land *after* a content write that
    // happens later in program order, masking a real change.
    let target = if names.is_empty() { SystemTime::now() } else { hash_plan.newest };
    let _ = filetime::set_file_mtime(&sums_path, filetime::FileTime::from_system_time(target));
    Ok(cf)
}

/// SHA-256 of every file in `work`, in `work`'s order, hashed across
/// files in parallel with progress on stderr: bytes done of bytes to
/// do, the rate, and the file count. `reused` is only for the summary.
fn hash_files(
    work: &[(usize, std::path::PathBuf, u64)],
    reused: usize,
    dir: &Path,
) -> std::io::Result<Vec<String>> {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;
    let total: u64 = work.iter().map(|(_, _, len)| *len).sum();
    let done = AtomicU64::new(0);
    let next = AtomicUsize::new(0);
    let finished = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<std::io::Result<String>>>> = Mutex::new((0..work.len()).map(|_| None).collect());
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8).min(work.len()).max(1);
    let started = std::time::Instant::now();
    let report = |final_line: bool| {
        let d = done.load(Ordering::Relaxed);
        let secs = started.elapsed().as_secs_f64().max(1e-3);
        let line = format!(
            "  hashing {}: {} / {} ({}/s), {} of {} files{}",
            dir.display(),
            fmt_bytes(d),
            fmt_bytes(total),
            fmt_bytes((d as f64 / secs) as u64),
            finished.load(Ordering::Relaxed),
            work.len(),
            if reused > 0 { format!(", {} reused", reused) } else { String::new() },
        );
        // Clear to the end of the line: a shorter line over a longer
        // one otherwise leaves the tail of the old one standing.
        if final_line {
            eprintln!("\r{line}\x1b[K");
        } else {
            eprint!("\r{line}\x1b[K");
        }
    };
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= work.len() {
                    break;
                }
                let r = sha256_file_counting(&work[i].1, &done);
                results.lock().unwrap()[i] = Some(r);
                finished.fetch_add(1, Ordering::Relaxed);
            });
        }
        // The calling thread reports while the workers hash.
        while finished.load(Ordering::Relaxed) < work.len() {
            std::thread::sleep(std::time::Duration::from_millis(500));
            if total >= 1 << 30 {
                report(false);
            }
        }
    });
    if total >= 1 << 30 {
        report(true);
    }
    let results = results.into_inner().unwrap();
    results.into_iter().map(|r| r.expect("every file hashed")).collect()
}

/// SHA-256 hex of a file, adding every byte read to `done`.
fn sha256_file_counting(path: &Path, done: &std::sync::atomic::AtomicU64) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        done.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(hex(&hasher.finalize()))
}

fn fmt_bytes(b: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 { format!("{} {}", b, UNITS[u]) } else { format!("{:.1} {}", v, UNITS[u]) }
}

fn mtime(path: &Path) -> std::io::Result<SystemTime> {
    std::fs::metadata(path)?.modified()
}

/// `(len, mtime)` of a file — a cheap fingerprint for detecting that a
/// file changed between when it was checksummed and when it is uploaded
/// (the TOCTOU window).
pub fn len_mtime(path: &Path) -> std::io::Result<(u64, SystemTime)> {
    let m = std::fs::metadata(path)?;
    Ok((m.len(), m.modified()?))
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
    fn escapes_names_with_backslash_or_newline() {
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        let c = "c".repeat(64);
        let cf = ChecksumFile {
            entries: vec![
                ChecksumEntry { hex: a.clone(), name: "weird\nname".into() },
                ChecksumEntry { hex: b.clone(), name: "back\\slash".into() },
                ChecksumEntry { hex: c.clone(), name: "normal.bin".into() },
            ],
        };
        let text = cf.render();
        // The two awkward names use the escaped line form; the plain one doesn't.
        assert_eq!(text.lines().filter(|l| l.starts_with('\\')).count(), 2);
        assert!(!text.contains("weird\nname")); // the literal newline is escaped away

        // Round-trips back to the exact original names.
        let parsed = ChecksumFile::parse(&text).unwrap();
        assert_eq!(parsed.digest_of("weird\nname"), Some(a.as_str()));
        assert_eq!(parsed.digest_of("back\\slash"), Some(b.as_str()));
        assert_eq!(parsed.digest_of("normal.bin"), Some(c.as_str()));
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
