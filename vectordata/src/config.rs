// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Canonical implementation of the user-facing `config` admin
//! commands.
//!
//! Every operation lives here once. Both the in-tree `vectordata`
//! binary and the `veks` CLI dispatch into this module; there is no
//! parallel implementation. Functions print directly to stdout/stderr
//! and return a process-style exit code (`0` on success, non-zero on
//! failure) so callers can simply `std::process::exit(code)`.
//!
//! Operations:
//!
//! - [`show`] — print the active configuration (settings file path,
//!   `cache_dir`, status, used space, protect flag).
//! - [`set_cache`] — write a new `cache_dir` into `settings.yaml`,
//!   honoring `protect_settings: true` unless `force=true`.
//! - [`list_mounts`] — enumerate writable mount points with available
//!   and total space; the standard "where should I put my cache"
//!   helper.
//! - [`add_catalog`], [`remove_catalog`], [`list_catalogs`] —
//!   manage `catalogs.yaml`, the list of catalog sources used by
//!   [`crate::catalog::resolver::Catalog::of`].

use std::path::{Path, PathBuf};

use crate::catalog::resolver::Catalog;
use crate::catalog::sources::{
    CatalogSources, DEFAULT_CONFIG_DIR, expand_tilde, raw_catalog_entries,
};
use crate::mounts;
use crate::settings::{self, SettingsWriteError, WriteCacheOutcome};

// ─── show ────────────────────────────────────────────────────────────

/// Print the active vectordata configuration. Mirrors the historical
/// `veks datasets config show` output. Returns 0 unconditionally —
/// "no configuration" is informational, not an error.
pub fn show() -> i32 {
    let path = settings::settings_path();
    println!("Configuration: {}", path.display());

    if !path.exists() {
        println!("  (settings file does not exist)");
        println!();
        println!("Set up a cache directory with:");
        println!("  vectordata config set-cache <path>");
        return 0;
    }

    match settings::cache_dir() {
        Ok(cache) => {
            println!("  cache_dir: {}", cache.display());
            if cache.is_dir() {
                println!("  Status:    Active");
                let used = dir_size(&cache);
                println!("  Used:      {}", fmt_bytes(used));
            } else if cache.exists() {
                println!("  Status:    Error — path exists but is not a directory");
            } else {
                println!("  Status:    Not yet created");
            }
        }
        Err(_) => println!("  cache_dir: (not set)"),
    }
    println!("  protect_settings: {}", settings::protect_settings());
    0
}

// ─── get-cache ───────────────────────────────────────────────────────

/// Print the configured `cache_dir` to stdout (one line, no
/// decoration) so it can be captured by `$(vectordata config
/// get-cache)`. Returns 1 with the actionable configuration error on
/// stderr when no `cache_dir:` is configured — distinct from
/// [`show`], which is purely informational.
pub fn get_cache() -> i32 {
    match settings::cache_dir() {
        Ok(p) => { println!("{}", p.display()); 0 }
        Err(e) => { eprintln!("{e}"); 1 }
    }
}

// ─── set-cache ───────────────────────────────────────────────────────

/// Sentinel values for [`set_cache`] that trigger auto-resolution.
/// Both spellings are accepted; `largest-writable-mount` is what the
/// help text advertises and `auto` is the shorter alias.
pub const AUTO_CACHE_SENTINEL_LONG: &str = "largest-writable-mount";
pub const AUTO_CACHE_SENTINEL_SHORT: &str = "auto";

/// Write `cache_dir` into `settings.yaml`. Honors
/// `protect_settings: true` unless `force=true`. Creates the target
/// directory if it does not exist.
///
/// When `cache_dir` is `largest-writable-mount` (or the `auto`
/// alias), picks a sensible default:
///
/// - If the largest writable mount is on a different filesystem than
///   `$HOME`, uses `<that-mount>/vectordata-cache/`.
/// - Otherwise (largest mount is the same filesystem as `$HOME`),
///   uses `$HOME/.cache/vectordata/`.
///
/// The "largest" rule looks at available bytes, not total — so a
/// nearly-full big disk loses to a smaller-but-mostly-empty one.
pub fn set_cache(cache_dir: &Path, force: bool) -> i32 {
    let resolved = match resolve_cache_spec(cache_dir) {
        Ok(p) => p,
        Err(e) => { eprintln!("{e}"); return 1; }
    };
    let cache_dir = resolved.as_path();
    match settings::write_cache_dir(cache_dir, force) {
        Ok(WriteCacheOutcome::Wrote) => {
            println!("Configuration updated:");
            println!("  settings.yaml: {}", settings::settings_path().display());
            println!("  cache_dir:     {}", cache_dir.display());
            0
        }
        Ok(WriteCacheOutcome::AlreadySet) => {
            println!("cache_dir already set to {}; settings.yaml unchanged.",
                cache_dir.display());
            0
        }
        Err(SettingsWriteError::Protected { settings_path, existing_cache_dir }) => {
            let existing = existing_cache_dir.as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(not set)".to_string());
            eprintln!("Refusing to overwrite protected settings at {}.",
                settings_path.display());
            eprintln!("Current cache_dir: {existing}");
            eprintln!("Re-run with --force to overwrite.");
            1
        }
        Err(SettingsWriteError::Io(e)) => {
            eprintln!("Error writing settings: {e}");
            1
        }
    }
}

/// Resolve a `set_cache` input. Pass-through for normal paths;
/// expands [`AUTO_CACHE_SENTINEL_LONG`] / [`AUTO_CACHE_SENTINEL_SHORT`]
/// to a concrete path picked from the live mount table.
fn resolve_cache_spec(input: &Path) -> Result<PathBuf, String> {
    let s = input.to_string_lossy();
    if s == AUTO_CACHE_SENTINEL_LONG || s == AUTO_CACHE_SENTINEL_SHORT {
        let resolved = pick_auto_cache_dir()?;
        println!("auto-resolved cache_dir → {}", resolved.display());
        return Ok(resolved);
    }
    Ok(input.to_path_buf())
}

/// Pick a cache directory based on the live mount table. Delegates
/// to [`settings::auto_resolved_cache_dir`] so the same selection
/// rule applies whether the user types `set-cache auto` or the
/// `cache_dir()` auto-bootstrap kicks in implicitly.
fn pick_auto_cache_dir() -> Result<PathBuf, String> {
    settings::auto_resolved_cache_dir().map(|r| r.path)
}

// ─── list-mounts ─────────────────────────────────────────────────────

pub use crate::mounts::MountInfo;

/// List writable mount points suitable for `cache_dir`. By default
/// hides mounts with less than 100 MiB free; pass `show_all=true` to
/// include them. Returns 0 unconditionally.
pub fn list_mounts(show_all: bool) -> i32 {
    let mounts = mounts::enumerate();
    let min = if show_all { 0 } else { 100 * 1024 * 1024 };

    println!("{:<40} {:>12} {:>12} {:>8}", "Mount Point", "Available", "Total", "Writable");
    println!("{}", "-".repeat(76));

    let mut count = 0;
    for m in &mounts {
        if m.available < min { continue; }
        let display = if m.path.len() > 38 {
            format!("...{}", &m.path[m.path.len() - 35..])
        } else { m.path.clone() };
        println!("{:<40} {:>12} {:>12} {:>8}",
            display, fmt_bytes(m.available), fmt_bytes(m.total),
            if m.writable { "Yes" } else { "No " });
        count += 1;
    }
    if count == 0 {
        println!("No suitable mount points found.");
        if !show_all {
            println!("Pass --all to include mounts with < 100 MiB free.");
        }
    } else {
        println!();
        println!("To set the cache dir: vectordata config set-cache <path>");
    }
    0
}

// ─── catalog management ─────────────────────────────────────────────

/// Specifier for [`remove_catalog`]: caller passes either a literal
/// URL/path or a 1-based index from [`list_catalogs`] output.
pub enum RemoveCatalogSpec<'a> {
    /// Match against the recorded URL/path string.
    Source(&'a str),
    /// 1-based index into the catalogs.yaml list.
    Index(usize),
}

fn catalogs_path() -> PathBuf {
    let dir = expand_tilde(DEFAULT_CONFIG_DIR);
    PathBuf::from(dir).join("catalogs.yaml")
}

fn load_catalog_entries() -> Vec<String> {
    let path = catalogs_path();
    if !path.is_file() { return Vec::new(); }
    let content = std::fs::read_to_string(&path).unwrap_or_default();
    serde_yaml::from_str(&content).unwrap_or_default()
}

fn save_catalog_entries(entries: &[String]) -> Result<PathBuf, String> {
    let path = catalogs_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }
    let yaml = serde_yaml::to_string(entries)
        .map_err(|e| format!("serializing catalogs.yaml: {e}"))?;
    std::fs::write(&path, yaml)
        .map_err(|e| format!("writing {}: {e}", path.display()))?;
    Ok(path)
}

/// Append a catalog source. Probes the source first to confirm it
/// resolves to at least one dataset — refusing to record an
/// unreachable or empty source is the single biggest paper-cut this
/// command has. Returns 1 if the source is unreachable or already
/// present.
pub fn add_catalog(source: &str) -> i32 {
    let mut entries = load_catalog_entries();
    if entries.iter().any(|e| e == source) {
        println!("Already configured: {source}");
        return 0;
    }

    let sources = CatalogSources::new().add_catalogs(&[source.to_string()]);
    let catalog = Catalog::of(&sources);
    let count = catalog.datasets().len();
    if count == 0 {
        eprintln!("FAIL {source} — no datasets found at this location");
        return 1;
    }
    println!("OK   {source} ({count} dataset(s))");

    entries.push(source.to_string());
    match save_catalog_entries(&entries) {
        Ok(path) => { println!("Saved to {}", path.display()); 0 }
        Err(e) => { eprintln!("Error: {e}"); 1 }
    }
}

/// Remove a catalog source by URL/path or 1-based index. Returns 1
/// if the spec doesn't match anything.
pub fn remove_catalog(spec: RemoveCatalogSpec<'_>) -> i32 {
    let mut entries = load_catalog_entries();
    let source = match spec {
        RemoveCatalogSpec::Source(s) => s.to_string(),
        RemoveCatalogSpec::Index(n) => {
            if n == 0 || n > entries.len() {
                eprintln!("Error: index {n} out of range (1..{})", entries.len());
                return 1;
            }
            let s = entries[n - 1].clone();
            println!("Removing catalog #{n}: {s}");
            s
        }
    };
    let before = entries.len();
    entries.retain(|e| e != &source);
    if entries.len() == before {
        eprintln!("Not found: {source}");
        return 1;
    }
    match save_catalog_entries(&entries) {
        Ok(_) => { println!("Removed: {source}"); 0 }
        Err(e) => { eprintln!("Error: {e}"); 1 }
    }
}

/// List the configured catalog sources.
pub fn list_catalogs() -> i32 {
    let path = catalogs_path();
    let config_dir = expand_tilde(DEFAULT_CONFIG_DIR);
    let entries = raw_catalog_entries(&config_dir);
    if entries.is_empty() {
        println!("No catalog sources configured.");
        println!();
        println!("Add one with:");
        println!("  vectordata config add-catalog <URL-or-path>");
    } else {
        println!("Configured catalog sources ({}):", entries.len());
        for (i, e) in entries.iter().enumerate() {
            println!("  {:>3}. {}", i + 1, e);
        }
        println!();
        println!("Catalogs file: {}", path.display());
    }
    0
}

// ─── helpers ─────────────────────────────────────────────────────────

fn fmt_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    const TIB: u64 = 1024 * GIB;
    if bytes >= TIB { format!("{:.1} TiB", bytes as f64 / TIB as f64) }
    else if bytes >= GIB { format!("{:.1} GiB", bytes as f64 / GIB as f64) }
    else if bytes >= MIB { format!("{:.1} MiB", bytes as f64 / MIB as f64) }
    else if bytes >= KIB { format!("{:.1} KiB", bytes as f64 / KIB as f64) }
    else { format!("{} B", bytes) }
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) { Ok(e) => e, Err(_) => continue };
        for e in entries.flatten() {
            let p = e.path();
            match e.file_type() {
                Ok(ft) if ft.is_dir() => stack.push(p),
                Ok(_) => {
                    if let Ok(m) = e.metadata() { total += m.len(); }
                }
                _ => {}
            }
        }
    }
    total
}

