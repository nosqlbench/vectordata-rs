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
use crate::catalog::sources::{CatalogSources, raw_catalog_entries};
use crate::mounts;
use crate::settings::{self, SettingsWriteError, WriteCacheOutcome};

// ─── show ────────────────────────────────────────────────────────────

/// Print the active vectordata configuration. Mirrors the historical
/// `veks datasets config get` output. Returns 0 unconditionally —
/// "no configuration" is informational, not an error.
pub fn show() -> i32 {
    let settings_path = settings::settings_path();
    let cats_path = catalogs_path();
    let config_dir = crate::catalog::sources::config_dir();
    let cats_configured = !raw_catalog_entries(&config_dir).is_empty();

    println!("Configuration: {}", settings_path.display());

    if !settings_path.exists() {
        println!("  (settings file does not exist)");
        println!();
        println!("First-run setup — three quick steps:");
        println!("  1. Pick a cache directory:");
        println!("       vectordata config set cache <path>");
        println!("       (or `vectordata config set cache auto` to choose the largest writable mount)");
        println!("  2. Subscribe to a catalog of published datasets:");
        println!("       vectordata config catalog add <URL-or-path>");
        println!("  3. Browse what's available:");
        println!("       vectordata datasets       # TUI");
        println!("       vectordata datasets list  # text");
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

    println!();
    println!("Catalogs: {}", cats_path.display());
    if cats_configured {
        // Defer to `list-catalogs` for the actual enumeration — keeps
        // a single source of truth for catalog output.
        let entries = raw_catalog_entries(&config_dir);
        for (i, e) in entries.iter().enumerate() {
            println!("  {:>3}. {}", i + 1, e);
        }
    } else {
        println!("  (no catalogs configured)");
        println!();
        println!("Add one to discover published datasets:");
        println!("  vectordata config catalog add <URL-or-path>");
    }
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
            eprintln!("error: refusing to overwrite protected settings at {}.",
                settings_path.display());
            eprintln!("Current cache_dir: {existing}");
            eprintln!("Re-run with --force to overwrite.");
            1
        }
        Err(SettingsWriteError::Io(e)) => {
            eprintln!("error: writing settings: {e}");
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
        println!("To set the cache dir: vectordata config set cache <path>");
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
    PathBuf::from(crate::catalog::sources::config_dir()).join("catalogs.yaml")
}

fn load_catalog_entries() -> Vec<crate::catalog::sources::NamedCatalogSource> {
    let path = catalogs_path();
    if !path.is_file() { return Vec::new(); }
    let content = std::fs::read_to_string(&path).unwrap_or_default();
    crate::catalog::sources::parse_named_catalogs(&content).unwrap_or_default()
}

// ── Comment-preserving textual edits ────────────────────────────────
//
// `catalogs.yaml` is a hand-edited file: commenting an entry out (and
// back in) is a normal workflow. Writes therefore NEVER round-trip
// through a YAML serializer — that would discard every comment and
// reflow the user's formatting. Instead, `add` appends one entry line
// and `remove` deletes exactly the matching entry line; every other
// byte of the file is preserved.

/// True when a line is a block-sequence entry (`- value`) — the
/// legacy list form.
fn is_list_entry_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("- ") || t == "-"
}

/// `(name, location)` of a top-level map entry line (`name: url`),
/// the default form going forward. Comments and list lines are not
/// map entries.
fn map_entry_line(line: &str) -> Option<(String, String)> {
    let t = line.trim_start();
    if t.starts_with('#') || is_list_entry_line(line) { return None; }
    let (key, value) = t.split_once(':')?;
    let key = key.trim();
    let value = value.trim().trim_matches('"').trim_matches('\'');
    if key.is_empty() || value.is_empty() { return None; }
    Some((key.to_string(), value.to_string()))
}

/// Parse the scalar value of a legacy list entry line, unquoting via
/// the YAML scalar parser so `- "url"` and `- url` compare equal.
fn entry_line_value(line: &str) -> Option<String> {
    let t = line.trim_start().strip_prefix('-')?.trim();
    if t.is_empty() { return None; }
    serde_yaml::from_str::<String>(t).ok()
}

/// Append a catalog entry, preserving the rest of the file
/// byte-for-byte. New entries use the map form (`name: url`) — the
/// default going forward — unless the existing file is a legacy list,
/// where a list line keeps the document a valid sequence. Refuses
/// structures it can't edit line-wise.
fn append_entry_text(content: &str, name: &str, source: &str) -> Result<String, String> {
    let mut has_list = false;
    let mut has_map = false;
    for line in content.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') { continue; }
        if is_list_entry_line(line) { has_list = true; continue; }
        if map_entry_line(line).is_some() { has_map = true; continue; }
        return Err(format!(
            "catalogs.yaml contains a line that is neither a list entry nor a \
             name: url mapping ({t:?}); refusing to edit it automatically"));
    }
    if has_list && has_map {
        return Err("catalogs.yaml mixes list and map entries; fix it manually".to_string());
    }
    let mut out = content.to_string();
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    if has_list {
        out.push_str(&format!("- {source}\n"));
    } else {
        out.push_str(&format!("{name}: {source}\n"));
    }
    Ok(out)
}

/// Remove the entry line(s) matching `target` — a legacy list value,
/// a map entry's location, or a map entry's NAME — preserving every
/// other byte (comments and commented-out entries included).
/// Returns `(new_content, removed_any)`.
fn remove_entry_text(content: &str, target: &str) -> (String, bool) {
    let mut removed = false;
    let kept: Vec<&str> = content.lines()
        .filter(|line| {
            let is_match = entry_line_value(line).as_deref() == Some(target)
                || map_entry_line(line)
                    .is_some_and(|(k, v)| k == target || v == target);
            if is_match { removed = true; }
            !is_match
        })
        .collect();
    let mut out = kept.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    (out, removed)
}

/// Derive a symbolic catalog name from its location: the last path
/// segment's stem, lowercased and sanitized to `[a-z0-9-]`. Empty
/// when nothing usable remains — the caller falls back to the
/// made-up enumeration.
fn derive_catalog_name(location: &str, taken: &[String]) -> String {
    let stem = location
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("")
        .rsplit_once('.')
        .map(|(s, _)| s)
        .unwrap_or_else(|| location.trim_end_matches('/').rsplit('/').next().unwrap_or(""));
    let slug: String = stem.to_lowercase().chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    let base = if slug.is_empty() {
        // Same promotion rule as legacy list files: stringified index.
        taken.len().to_string()
    } else {
        slug
    };
    if !taken.iter().any(|t| t == &base) {
        return base;
    }
    let mut n = 2;
    loop {
        let candidate = format!("{base}-{n}");
        if !taken.iter().any(|t| t == &candidate) {
            return candidate;
        }
        n += 1;
    }
}

fn append_catalog_entry(name: &str, source: &str) -> Result<PathBuf, String> {
    let path = catalogs_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }
    let content = if path.is_file() {
        std::fs::read_to_string(&path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?
    } else {
        String::new()
    };
    let new_content = append_entry_text(&content, name, source)?;
    std::fs::write(&path, new_content)
        .map_err(|e| format!("writing {}: {e}", path.display()))?;
    Ok(path)
}

fn remove_catalog_entry(source: &str) -> Result<bool, String> {
    let path = catalogs_path();
    if !path.is_file() { return Ok(false); }
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("reading {}: {e}", path.display()))?;
    let (new_content, removed) = remove_entry_text(&content, source);
    if removed {
        std::fs::write(&path, new_content)
            .map_err(|e| format!("writing {}: {e}", path.display()))?;
    }
    Ok(removed)
}

// The palette/curve settings configure the `explore` TUI's
// appearance, so the whole surface lives behind the `explore`
// feature — its single authority for the name lists and defaults is
// `crate::explore::palette`, which only compiles with that feature.
// Library consumers without `explore` (e.g. `vecd`) don't see these.

/// Canonical palette names for `config set palette` validation and
/// tab-completion. Delegates to the palette module's single
/// authority so the lists can't drift.
#[cfg(feature = "explore")]
pub fn ui_palette_names() -> Vec<&'static str> {
    crate::explore::palette::ALL_PALETTES.iter().map(|p| p.name()).collect()
}

/// Canonical curve names; see [`ui_palette_names`].
#[cfg(feature = "explore")]
pub fn ui_curve_names() -> Vec<&'static str> {
    crate::explore::palette::ALL_CURVES.iter().map(|c| c.name()).collect()
}

/// Print one UI setting (`palette` / `curve`): the configured value
/// or the project standard when unset.
#[cfg(feature = "explore")]
pub fn get_ui_setting(key: &str) -> i32 {
    let standard = match key {
        "palette" => crate::explore::palette::Palette::default().name(),
        "curve" => crate::explore::palette::Curve::default().name(),
        _ => { eprintln!("unknown UI setting '{key}'"); return 2; }
    };
    match crate::settings::setting_value(key) {
        Some(v) => println!("{v}"),
        None => println!("{standard}"),
    }
    0
}

/// Validate and persist a UI setting (`palette` / `curve`) to
/// `settings.yaml` via the comment-preserving line editor.
#[cfg(feature = "explore")]
pub fn set_ui_setting(key: &str, value: &str) -> i32 {
    let valid = match key {
        "palette" => ui_palette_names(),
        "curve" => ui_curve_names(),
        _ => { eprintln!("unknown UI setting '{key}'"); return 2; }
    };
    let value = value.trim().to_lowercase();
    if !valid.contains(&value.as_str()) {
        eprintln!("unknown {key} '{value}' (valid: {})", valid.join(", "));
        return 1;
    }
    match crate::settings::write_setting(key, &value) {
        Ok(path) => { println!("{key} = {value}\nSaved to {}", path.display()); 0 }
        Err(e) => { eprintln!("error: {e}"); 1 }
    }
}

/// Print the update-check switch: `on` / `off` (standard: on).
pub fn get_update_check() -> i32 {
    let enabled = crate::update_check::enabled_setting(
        crate::settings::setting_value(crate::update_check::SETTING_KEY).as_deref());
    println!("{}", if enabled { "on" } else { "off" });
    0
}

/// Validate and persist the update-check switch to `settings.yaml`
/// via the comment-preserving line editor.
pub fn set_update_check(value: &str) -> i32 {
    let norm = match value.trim().to_lowercase().as_str() {
        "on" | "true" | "yes" | "1" | "enabled" => "on",
        "off" | "false" | "no" | "0" | "disabled" => "off",
        other => {
            eprintln!("invalid update_check value '{other}' (use on or off)");
            return 1;
        }
    };
    match crate::settings::write_setting(crate::update_check::SETTING_KEY, norm) {
        Ok(path) => {
            println!("update_check = {norm}\nSaved to {}", path.display());
            0
        }
        Err(e) => { eprintln!("error: {e}"); 1 }
    }
}

/// Verify a catalog source end-to-end before it is accepted:
///
/// 1. The location must parse into at least one dataset entry.
/// 2. The first dataset must pass a **catalog ping** — every facet
///    of its default (or first) profile reachable through the same
///    access layer the runtime uses for real reads. A catalog whose
///    document parses but whose data endpoints are dead or
///    mis-based is refused here instead of failing later at first
///    use.
///
/// Returns the dataset count on success, a human-readable refusal
/// otherwise. Pure with respect to configuration — nothing is read
/// from or written to `catalogs.yaml` — so it is directly testable
/// against a fixture server.
pub fn verify_catalog_source(source: &str) -> Result<usize, String> {
    let sources = CatalogSources::new().add_catalogs(&[source.to_string()]);
    let catalog = Catalog::of(&sources);
    let count = catalog.datasets().len();
    if count == 0 {
        return Err("no datasets found at this location".to_string());
    }
    let first = &catalog.datasets()[0];
    let name = first.name.clone();
    let profiles = first.profile_names();
    let profile = if profiles.contains(&"default") {
        "default".to_string()
    } else {
        profiles.first().map(|p| p.to_string()).unwrap_or_else(|| "default".to_string())
    };
    println!("Verifying endpoint with a catalog ping — dataset '{name}', profile '{profile}':");
    let code = crate::datasets::ping::run_via_catalog(&catalog, &name, &profile);
    if code != 0 {
        return Err(format!(
            "catalog ping failed for dataset '{name}' (profile '{profile}') —              the catalog parses but its data endpoint is not serving"));
    }
    Ok(count)
}

/// Append a catalog source. The source must pass
/// [`verify_catalog_source`] — parse to at least one dataset AND
/// answer a catalog ping — before anything is recorded. Returns 1
/// if verification fails; nothing is saved in that case.
pub fn add_catalog(source: &str, name: Option<&str>) -> i32 {
    let entries = load_catalog_entries();
    if entries.iter().any(|e| e.location == source) {
        println!("Already configured: {source}");
        return 0;
    }
    let taken: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();
    let name = match name {
        Some(n) => {
            let n = n.trim();
            if n.is_empty() || taken.iter().any(|t| t == n) {
                eprintln!("error: catalog name '{n}' is empty or already in use");
                return 1;
            }
            n.to_string()
        }
        None => derive_catalog_name(source, &taken),
    };

    let count = match verify_catalog_source(source) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("FAIL {source} — {e}");
            eprintln!("Not saved.");
            return 1;
        }
    };
    println!("OK   {source} ({count} dataset(s))");

    match append_catalog_entry(&name, source) {
        Ok(path) => { println!("Saved as '{name}' to {}", path.display()); 0 }
        Err(e) => { eprintln!("error: {e}"); 1 }
    }
}

/// Remove a catalog source by name, URL/path, or 1-based index.
/// Returns 1 if the spec doesn't match anything.
pub fn remove_catalog(spec: RemoveCatalogSpec<'_>) -> i32 {
    let entries = load_catalog_entries();
    let target = match spec {
        RemoveCatalogSpec::Source(s) => s.to_string(),
        RemoveCatalogSpec::Index(n) => {
            if n == 0 || n > entries.len() {
                eprintln!("error: index {n} out of range (1..{})", entries.len());
                return 1;
            }
            let e = &entries[n - 1];
            println!("Removing catalog #{n}: {} ({})", e.name, e.location);
            e.location.clone()
        }
    };
    match remove_catalog_entry(&target) {
        Ok(true) => { println!("Removed: {target}"); 0 }
        Ok(false) => { eprintln!("Not found: {target}"); 1 }
        Err(e) => { eprintln!("error: {e}"); 1 }
    }
}

/// List the configured catalog sources with their symbolic names.
pub fn list_catalogs() -> i32 {
    let path = catalogs_path();
    let config_dir = crate::catalog::sources::config_dir();
    let entries = crate::catalog::sources::named_catalog_entries(&config_dir);
    if entries.is_empty() {
        println!("No catalog sources configured.");
        println!();
        println!("Add one with:");
        println!("  vectordata config catalog add <URL-or-path> [--name <name>]");
    } else {
        println!("Configured catalog sources ({}):", entries.len());
        let name_w = entries.iter().map(|e| e.name.len()).max().unwrap_or(8);
        for (i, e) in entries.iter().enumerate() {
            println!("  {:>3}. {:<name_w$}  {}", i + 1, e.name, e.location);
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

#[cfg(test)]
mod tests {
    use super::*;

    const HAND_EDITED: &str = "\
# my catalogs — keep sorted
- https://a.example/cat/
# temporarily disabled:
# - https://b.example/cat/

- https://c.example/cat/
";

    /// `config catalog add` must never disturb existing bytes —
    /// comments and commented-out entries are a normal hand-edit
    /// workflow and were previously destroyed by a serde round-trip.
    #[test]
    fn append_preserves_comments_and_formatting() {
        // Legacy list file → stays a list (a map line would break the
        // YAML document); comments untouched.
        let out = append_entry_text(HAND_EDITED, "dx", "https://d.example/cat/").unwrap();
        assert!(out.starts_with(HAND_EDITED),
            "existing content must be byte-identical:\n{out}");
        assert!(out.ends_with("- https://d.example/cat/\n"));
    }

    #[test]
    fn append_uses_map_form_going_forward() {
        // Empty file → map form (the default going forward).
        assert_eq!(append_entry_text("", "prot", "https://x/").unwrap(),
            "prot: https://x/\n");
        // Existing map file → map form, comments preserved.
        let existing = "# main\nprot: https://a/\n";
        let out = append_entry_text(existing, "lab", "https://b/").unwrap();
        assert_eq!(out, "# main\nprot: https://a/\nlab: https://b/\n");
        // Existing list file → list line for document validity.
        let out = append_entry_text("- https://a/", "x", "https://x/").unwrap();
        assert_eq!(out, "- https://a/\n- https://x/\n");
    }

    #[test]
    fn remove_matches_map_name_or_location() {
        let content = "# keep\nprot: https://a/\nlab: https://b/\n";
        let (out, removed) = remove_entry_text(content, "prot");
        assert!(removed);
        assert!(out.contains("# keep") && out.contains("lab:") && !out.contains("prot"));
        let (out2, removed2) = remove_entry_text(content, "https://b/");
        assert!(removed2);
        assert!(out2.contains("prot:") && !out2.contains("lab"));
    }

    #[test]
    fn derive_catalog_name_slugs_and_enumerates() {
        assert_eq!(derive_catalog_name("https://h/x/protected-catalog.yaml", &[]),
            "protected-catalog");
        assert_eq!(derive_catalog_name("/data/catalogs/lab/", &[]), "lab");
        // Collision → numbered suffix.
        assert_eq!(derive_catalog_name("/x/lab", &["lab".to_string()]), "lab-2");
        // Nothing usable → stringified index (the promotion rule).
        assert_eq!(derive_catalog_name("///", &["a".to_string()]), "1");
    }

    #[test]
    fn append_refuses_unrecognized_structure() {
        let err = append_entry_text("catalogs:\n  - https://a/\n", "x", "https://x/")
            .unwrap_err();
        assert!(err.contains("mixes") || err.contains("refusing"), "{err}");
    }

    /// Removal drops exactly the matching entry line; comments —
    /// including a commented-out copy of the same URL — survive.
    #[test]
    fn remove_preserves_comments_including_commented_twin() {
        let content = "\
# header comment
- https://a.example/cat/
# - https://a.example/cat/   (disabled mirror)
- \"https://b.example/cat/\"
";
        let (out, removed) = remove_entry_text(content, "https://a.example/cat/");
        assert!(removed);
        assert!(out.contains("# header comment"));
        assert!(out.contains("# - https://a.example/cat/"),
            "commented twin must survive: {out}");
        assert!(!out.lines().any(|l| l == "- https://a.example/cat/"));
        // Quoted entries compare by parsed value.
        let (out2, removed2) = remove_entry_text(&out, "https://b.example/cat/");
        assert!(removed2, "quoted entry must match by value");
        assert!(!out2.contains("b.example"));
    }

    #[test]
    fn remove_of_absent_entry_changes_nothing() {
        let (out, removed) = remove_entry_text(HAND_EDITED, "https://nope/");
        assert!(!removed);
        assert_eq!(out, HAND_EDITED);
    }
}
