// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `<binary> datasets precache` — drive a dataset profile to
//! fully-resident state through the canonical reader API.
//!
//! Reachable as `vectordata datasets precache` or `veks datasets
//! precache` — both binaries dispatch into this module.
//!
//! Source resolution: catalog name, `name:profile` pair, local path
//! / `dataset.yaml`, or HTTP URL. The reader layer dispatches
//! per-facet:
//!
//! - **local file** → `Storage::Mmap`, no copy, no merkle, no work.
//! - **remote URL with `.mref`** → `Storage::Cached`; download +
//!   merkle-verify chunks into the configured cache directory,
//!   promote to mmap on completion.
//! - **remote URL without `.mref`** → `Storage::Http`; download the
//!   full file via parallel fixed-size HTTP RANGE chunks (same
//!   `download_concurrency` worker pool + retry policy as the
//!   `.mref` path, but trusting TLS rather than a per-chunk hash
//!   chain) into the cache directory, promote to mmap on completion.
//!
//! The driver prints a live single-line status meter with per-facet
//! and aggregate progress (carriage-return-overwritten on stderr).
//! Pre-walks the facet manifest once to know the total download
//! size upfront, then streams chunk-level updates from
//! [`crate::view::TestDataView::prebuffer_all_with_progress`].

use std::path::Path;

use super::build_sources;
use crate::catalog::resolver::Catalog;
use crate::{PrebufferProgress, TestDataView};

/// Entry point.
///
/// `dataset_spec` is one of:
/// - `name:profile` resolved via the catalog (e.g. `glove-100:default`)
/// - `name` resolved via the catalog (uses *all* profiles)
/// - a local directory containing a `dataset.yaml`
/// - a path to a `dataset.yaml` file
/// - an HTTP URL to a dataset directory or `dataset.yaml`
///
/// `configdir`, `extra_catalogs`, and `at` are the catalog-source
/// inputs (same shape both binaries pass). `cache_dir` is purely
/// informational; the actual cache root is resolved via
/// [`crate::settings::cache_dir`].
///
/// Returns a process exit code (0 = success).
pub fn run(
    dataset_spec: &str,
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
    cache_dir: Option<&Path>,
) -> i32 {
    let configured = match crate::settings::cache_dir() {
        Ok(p) => Some(p),
        Err(e) => {
            // Only fatal if we'll actually need the cache. Local-only
            // datasets precache fine without one. Defer the fatal
            // until we know the dispatch outcome.
            eprintln!("note: {e}");
            eprintln!();
            None
        }
    };
    if let Some(override_) = cache_dir {
        eprintln!("note: --cache-dir {} is recorded but the active cache root is {}",
            override_.display(),
            configured.as_deref().map(|p| p.display().to_string())
                .unwrap_or_else(|| "(unconfigured)".to_string()));
    }

    let (resolution, profile_sel) = match resolve_spec(
        dataset_spec, configdir, extra_catalogs, at,
    ) {
        Some(r) => r,
        None => return 1,
    };

    // Open through whichever path knows how to materialise this
    // shape. Catalog-resolved entries MUST go through
    // `Catalog::open` so the knn_entries-shape synthesis path is
    // taken when applicable — `TestDataGroup::load(entry.path)`
    // would point at the catalog base URL for those entries (there
    // is no per-dataset `dataset.yaml` to load) and fail.
    let (group, descriptor) = match resolution {
        Resolved::CatalogEntry { catalog, name } => {
            let group = match catalog.open(&name) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("error: failed to open dataset '{name}': {e}");
                    return 1;
                }
            };
            (group, name)
        }
        Resolved::Local(path) | Resolved::Url(path) => {
            let group = match crate::TestDataGroup::load(&path) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("error: failed to open dataset at {path}: {e}");
                    return 1;
                }
            };
            (group, path)
        }
    };

    if let Some(c) = &configured {
        eprintln!("  Cache root: {}", c.display());
    }

    match profile_sel {
        ProfileSelection::Named(profile_name) => {
            let view = match group.profile(&profile_name) {
                Some(v) => v,
                None => {
                    eprintln!("Profile '{profile_name}' not found at {descriptor}.");
                    eprintln!("Available profiles: {}",
                        group.profile_names().join(", "));
                    return 1;
                }
            };
            eprintln!("Prebuffering {descriptor}:{profile_name}");
            drive_prebuffer(&*view)
        }
        ProfileSelection::AllProfiles => {
            let names = group.profile_names();
            eprintln!("Prebuffering {descriptor} — all profiles ({})",
                names.join(", "));
            drive_prebuffer_all(&group)
        }
    }
}

enum Resolved {
    /// Catalog-resolved entry. Carries the catalog itself so the
    /// caller goes through `Catalog::open(name)` — that's the only
    /// path that handles `knn_entries.yaml`-shape catalogs
    /// correctly (those entries have no per-dataset `dataset.yaml`;
    /// the catalog's own embedded layout *is* the dataset
    /// description, and `Catalog::open` synthesises the group from
    /// it).
    CatalogEntry { catalog: Catalog, name: String },
    Local(String),
    Url(String),
}

/// Whether the user named a specific profile or asked for all of
/// them. A bare `dataset` spec (no `:profile` suffix) selects all
/// profiles; an explicit `dataset:profile` selects just that one.
enum ProfileSelection {
    Named(String),
    AllProfiles,
}

/// Resolve a user-supplied spec to a (path-or-url, profile)
/// pair. Returns `None` when resolution fails after writing a
/// diagnostic to stderr — the caller surfaces the exit code.
fn resolve_spec(
    dataset_spec: &str,
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
) -> Option<(Resolved, ProfileSelection)> {
    let (head, profile_sel) = if dataset_spec.contains('/')
        || dataset_spec.starts_with("http://")
        || dataset_spec.starts_with("https://")
    {
        (dataset_spec, ProfileSelection::AllProfiles)
    } else if let Some(pos) = dataset_spec.find(':') {
        (&dataset_spec[..pos],
         ProfileSelection::Named(dataset_spec[pos + 1..].to_string()))
    } else {
        (dataset_spec, ProfileSelection::AllProfiles)
    };

    if head.starts_with("http://") || head.starts_with("https://") {
        return Some((Resolved::Url(head.to_string()), profile_sel));
    }
    let as_path = Path::new(head);
    if as_path.exists() {
        return Some((Resolved::Local(head.to_string()), profile_sel));
    }

    let sources = build_sources(configdir, extra_catalogs, at);
    if sources.is_empty() {
        eprintln!("'{}' is not a local path, not a URL, and no catalog is configured.",
            head);
        eprintln!("Add a catalog with:");
        eprintln!("  vectordata config catalog add <URL-or-path>");
        eprintln!("Or use --catalog/--at for one-off access.");
        return None;
    }
    let catalog = Catalog::of(&sources);
    let entry = match catalog.find_exact(head) {
        Some(e) => e,
        None => {
            eprintln!("Dataset '{head}' not found.");
            catalog.list_datasets(head);
            return None;
        }
    };
    if let ProfileSelection::Named(ref p) = profile_sel
        && entry.layout.profiles.profile(p).is_none() {
            eprintln!("Profile '{p}' not found in dataset '{}'. Available: {}",
                entry.name, entry.profile_names().join(", "));
            return None;
        }
    let name = entry.name.clone();
    Some((Resolved::CatalogEntry { catalog, name }, profile_sel))
}

// ─── Drivers ─────────────────────────────────────────────────────────

fn drive_prebuffer(view: &dyn TestDataView) -> i32 {
    let plan = plan_prebuffer(view);
    if plan.facets.is_empty() {
        println!("Precache: profile declared no facets.");
        return 0;
    }
    eprintln!("Prebuffering {} facet(s), {} to download. ({} streams × {} HTTP runtimes)",
        plan.facets.len(), fmt_bytes(plan.total_bytes),
        crate::cache::download_concurrency(),
        crate::transport::http_runtimes());
    let mut ctx = LiveCtx::new(plan.facets.len(), plan.total_bytes);
    let result = view.prebuffer_all_with_progress(&mut |facet, p| ctx.on_progress(facet, p));
    ctx.finalize(&result.as_ref().map(|_| ()).map_err(|e| e.to_string()));
    if result.is_err() { 1 } else { 0 }
}

fn drive_prebuffer_all(group: &crate::TestDataGroup) -> i32 {
    let mut all_facets: Vec<FacetPlanRow> = Vec::new();
    let mut total_bytes = 0u64;
    for profile_name in group.profile_names() {
        if let Some(view) = group.profile(&profile_name) {
            let plan = plan_prebuffer(&*view);
            total_bytes += plan.total_bytes;
            for row in plan.facets {
                all_facets.push(FacetPlanRow {
                    qualified_name: format!("{profile_name}/{}", row.qualified_name),
                });
            }
        }
    }
    if all_facets.is_empty() {
        println!("Precache: no facets across any profile.");
        return 0;
    }
    if total_bytes >= crate::PREBUFFER_LARGE_WARNING_BYTES {
        eprintln!("warning: precache announced {} across all profiles \
                   (above the {} advisory threshold).",
            fmt_bytes(total_bytes),
            fmt_bytes(crate::PREBUFFER_LARGE_WARNING_BYTES));
        eprintln!("Continuing — pass an explicit `dataset:profile` to limit \
                   which profiles are downloaded.");
    }
    eprintln!("Prebuffering {} facet(s) across all profiles, {} to download.",
        all_facets.len(), fmt_bytes(total_bytes));

    let mut ctx = LiveCtx::new(all_facets.len(), total_bytes);
    let result = group.prebuffer_all_profiles_with_progress(
        &mut |profile, facet, p| {
            let qualified = format!("{profile}/{facet}");
            ctx.on_progress(&qualified, p);
        },
        &mut |_total| { /* warning already issued above */ },
    );
    ctx.finalize(&result.as_ref().map(|_| ()).map_err(|e| e.to_string()));
    if result.is_err() { 1 } else { 0 }
}

// ─── Plan + Live-update renderer ─────────────────────────────────────

#[derive(Clone, Debug)]
struct FacetPlanRow {
    qualified_name: String,
}

struct PrebufferPlan {
    facets: Vec<FacetPlanRow>,
    /// Sum of `total_bytes` across *remote* facets only — local
    /// facets are already resident so they don't contribute to the
    /// download tally.
    total_bytes: u64,
}

fn plan_prebuffer(view: &dyn TestDataView) -> PrebufferPlan {
    let mut facets = Vec::new();
    let mut total_bytes = 0u64;
    for (name, desc) in view.facet_manifest() {
        if view.facet_element_type(&name).is_err() { continue; }
        if let Ok(storage) = view.open_facet_storage(&name) {
            // Windowed facets contribute their window size, not the
            // shared base file's full size — otherwise a sized
            // profile against a 1.3 TiB base announces "1.3 TiB to
            // download" even though the windowed precache only
            // pulls a fraction. `facet_download_bytes` handles the
            // local/remote and windowed/full split in one call.
            total_bytes += crate::view::facet_download_bytes(
                desc.source_path.as_deref(), &storage);
            facets.push(FacetPlanRow { qualified_name: name });
        }
    }
    PrebufferPlan { facets, total_bytes }
}

/// In-place stderr renderer for precache progress.
///
/// Tracks per-facet `verified_bytes` (last callback wins) and
/// aggregates across all facets. Updates the status line at ~4 Hz
/// via carriage-return so the output stays on a single line. When
/// stderr isn't a tty (piped, captured), the updates still write
/// but the terminal won't reflow them — acceptable, since piped
/// output usually wants a log rather than a meter.
pub(super) struct LiveCtx {
    facet_count: usize,
    total_bytes: u64,
    bytes_per_facet: std::collections::HashMap<String, u64>,
    current_facet: String,
    facet_index: usize,
    last_render: std::time::Instant,
    started: std::time::Instant,
}

impl LiveCtx {
    pub(super) fn new(facet_count: usize, total_bytes: u64) -> Self {
        Self {
            facet_count,
            total_bytes,
            bytes_per_facet: std::collections::HashMap::new(),
            current_facet: String::new(),
            facet_index: 0,
            last_render: std::time::Instant::now()
                - std::time::Duration::from_secs(1),
            started: std::time::Instant::now(),
        }
    }

    pub(super) fn on_progress(&mut self, facet: &str, p: &PrebufferProgress) {
        if facet != self.current_facet {
            self.flush_facet_summary();
            self.current_facet = facet.to_string();
            self.facet_index += 1;
            self.last_render = std::time::Instant::now()
                - std::time::Duration::from_secs(1);
            // Print the in-place line immediately on facet switch
            // so users see the meter flip to the new facet *before*
            // the .mref network round trip completes.
            self.render(facet, p);
            self.last_render = std::time::Instant::now();
        }
        // Pre-open events arrive with `total_bytes == 0`; they only
        // exist to flush the previous facet's summary and surface
        // the new facet name. Skip the byte-accounting update for
        // those — otherwise the post-open size briefly appears as
        // a "regression" (0 bytes verified out of N total bytes).
        if p.total_bytes > 0 {
            self.bytes_per_facet.insert(facet.to_string(), p.verified_bytes);
        }
        if self.last_render.elapsed().as_millis() >= 250 {
            self.render(facet, p);
            self.last_render = std::time::Instant::now();
        }
    }

    fn render(&self, facet: &str, p: &PrebufferProgress) {
        let aggregate_done: u64 = self.bytes_per_facet.values().sum();
        let pct_total = pct(aggregate_done, self.total_bytes);
        let facet_state = if p.total_bytes == 0 {
            // Pre-open — `.mref` fetch in flight, size still unknown.
            "opening…".to_string()
        } else {
            format!("{}% ({}/{})",
                pct(p.verified_bytes, p.total_bytes),
                fmt_bytes(p.verified_bytes),
                fmt_bytes(p.total_bytes))
        };
        // Throughput + ETA. Held back until we've been downloading
        // long enough for the rate to be meaningful — the first
        // second is dominated by TLS handshake + initial chunk
        // bring-up, so the implied "bytes / elapsed" would suggest
        // an absurdly long ETA right when the user is most likely
        // to look at it.
        let elapsed = self.started.elapsed().as_secs_f64();
        let trailing = if elapsed > 1.5 && aggregate_done > 0 && self.total_bytes > aggregate_done {
            let rate = aggregate_done as f64 / elapsed;
            let remaining = self.total_bytes - aggregate_done;
            let eta_secs = (remaining as f64 / rate.max(1.0)) as u64;
            format!(" \u{2022} {}/s \u{2022} ETA {}",
                fmt_bytes(rate as u64), fmt_duration(eta_secs))
        } else {
            String::new()
        };
        use std::io::Write;
        eprint!(
            "\r  [{}/{}] {}: {} \u{2022} total {}% ({}/{}){}\u{1b}[K",
            self.facet_index, self.facet_count, facet,
            facet_state,
            pct_total, fmt_bytes(aggregate_done), fmt_bytes(self.total_bytes),
            trailing);
        let _ = std::io::stderr().flush();
    }

    /// Print a permanent "✓" line for the just-finished facet,
    /// erasing the in-place progress line first.
    fn flush_facet_summary(&self) {
        if self.current_facet.is_empty() { return; }
        let bytes = self.bytes_per_facet.get(&self.current_facet)
            .copied().unwrap_or(0);
        eprintln!("\r  [{}/{}] {} \u{2713} {}\u{1b}[K",
            self.facet_index, self.facet_count, self.current_facet,
            fmt_bytes(bytes));
    }

    pub(super) fn finalize<T, E: std::fmt::Display>(&self, result: &Result<T, E>) {
        self.flush_facet_summary();
        let elapsed = self.started.elapsed().as_secs_f64();
        let done: u64 = self.bytes_per_facet.values().sum();
        match result {
            Ok(_) => {
                eprintln!("Precache done: {} facet(s), {} in {:.1}s ({}/s).",
                    self.facet_count, fmt_bytes(done), elapsed,
                    fmt_bytes((done as f64 / elapsed.max(0.001)) as u64));
            }
            Err(e) => {
                eprintln!("Precache: failed — {e}");
            }
        }
    }
}

pub(super) fn pct(done: u64, total: u64) -> u32 {
    if total == 0 { return 100; }
    ((done as u128 * 100) / total as u128) as u32
}

pub(super) fn fmt_bytes(bytes: u64) -> String {
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

/// Format a duration in seconds as a compact human string. Picks
/// the largest unit pair: `45s`, `3m 22s`, `1h 12m`, `2d 04h`. The
/// double-unit form keeps the resolution useful at the boundary
/// (so a 60m ETA doesn't display as "1h 00m" right next to a 59s
/// ETA without showing the seconds context).
pub(super) fn fmt_duration(secs: u64) -> String {
    const M: u64 = 60;
    const H: u64 = 60 * M;
    const D: u64 = 24 * H;
    if secs < M { format!("{secs}s") }
    else if secs < H { format!("{}m {:02}s", secs / M, secs % M) }
    else if secs < D { format!("{}h {:02}m", secs / H, (secs % H) / M) }
    else             { format!("{}d {:02}h", secs / D, (secs % D) / H) }
}
