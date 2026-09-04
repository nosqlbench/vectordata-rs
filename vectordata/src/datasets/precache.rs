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

use std::path::{Path, PathBuf};

use super::build_sources;
use crate::catalog::resolver::Catalog;
use crate::{PrebufferProgress, TestDataView};

/// Everything a precache run needs.
///
/// A struct rather than a widening positional list: the call already
/// carried five arguments before windows and facet selection, and four
/// call sites had to be edited in lockstep every time one was added.
#[derive(Debug, Clone, Default)]
pub struct PrecacheRequest {
    /// `name`, `name:profile`, a local path, a `dataset.yaml`, or a URL.
    pub dataset_spec: String,
    pub configdir: String,
    pub extra_catalogs: Vec<String>,
    pub at: Vec<String>,
    /// Recorded and reported; the active cache root comes from settings.
    pub cache_dir: Option<PathBuf>,
    /// Profile to use, when the spec does not name one.
    ///
    /// Its own field rather than a `spec:profile` suffix because that
    /// suffix cannot work for a local path: `resolve_spec` treats
    /// anything containing `/` as naming every profile, so a directory
    /// has no way to spell a profile inside the spec at all.
    pub profile: Option<String>,
    /// Facets to fetch. Empty means every facet the profile declares —
    /// which is what precache has always done.
    pub facets: Vec<String>,
    /// Record window, in the dataset-source window grammar. `None`
    /// means the whole facet, so a windowless run is the original
    /// behaviour rather than a special case of it.
    pub window: Option<String>,
    /// Print what would be fetched and stop.
    pub plan_only: bool,
    /// Accept fetching a whole facet when the window cannot be resolved
    /// for its format.
    ///
    /// Off by default. Asking for a window and silently receiving a
    /// terabyte is the surprise windowed precache exists to prevent, so
    /// the fallback is something the caller says yes to rather than
    /// something they discover afterwards.
    pub allow_whole_facet: bool,
}

impl PrecacheRequest {
    /// The plain form: everything, no window. What every caller that
    /// predates windowed precache wants.
    pub fn all(dataset_spec: &str, configdir: &str) -> Self {
        PrecacheRequest {
            dataset_spec: dataset_spec.to_string(),
            configdir: configdir.to_string(),
            ..PrecacheRequest::default()
        }
    }

    /// Whether this run selects a subset — of facets, of records, or
    /// only wants to be told what it would do.
    fn is_selective(&self) -> bool {
        !self.facets.is_empty() || self.window.is_some() || self.plan_only
    }
}

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
pub fn run(req: PrecacheRequest) -> i32 {
    let dataset_spec = req.dataset_spec.as_str();
    let configdir = req.configdir.as_str();
    let extra_catalogs = req.extra_catalogs.as_slice();
    let at = req.at.as_slice();
    let cache_dir = req.cache_dir.as_deref();

    // A window has to parse before anything is opened or downloaded.
    // Discovering it is malformed after the catalog round-trip wastes
    // the user's time for no reason.
    let window = match req.window.as_deref() {
        Some(w) => match crate::dataset::source::parse_window(w) {
            Ok(parsed) => Some(parsed),
            Err(e) => {
                eprintln!("error: --window '{w}': {e}");
                return 2;
            }
        },
        None => None,
    };

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
        eprintln!(
            "note: --cache-dir {} is recorded but the active cache root is {}",
            override_.display(),
            configured
                .as_deref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(unconfigured)".to_string())
        );
    }

    let (resolution, spec_profile) = match resolve_spec(dataset_spec, configdir, extra_catalogs, at)
    {
        Some(r) => r,
        None => return 1,
    };
    // An explicit --profile outranks whatever the spec implied.
    let profile_sel = match req.profile.as_deref() {
        Some(p) => ProfileSelection::Named(p.to_string()),
        None => spec_profile,
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
                    eprintln!("Available profiles: {}", group.profile_names().join(", "));
                    return 1;
                }
            };
            if req.is_selective() {
                return drive_selective(
                    &*view,
                    &format!("{descriptor}:{profile_name}"),
                    &req.facets,
                    window.as_ref(),
                    req.plan_only,
                    req.allow_whole_facet,
                );
            }
            eprintln!("Prebuffering {descriptor}:{profile_name}");
            drive_prebuffer(&*view, req.allow_whole_facet)
        }
        ProfileSelection::AllProfiles if req.is_selective() => {
            // A facet or window selection needs one profile to resolve
            // against — the same facet name means different bytes in
            // different profiles, and silently picking one would be a
            // guess presented as a result.
            let names = group.profile_names();
            if names.len() == 1 {
                let view = group.profile(&names[0]).expect("profile just listed");
                return drive_selective(
                    &*view,
                    &format!("{descriptor}:{}", names[0]),
                    &req.facets,
                    window.as_ref(),
                    req.plan_only,
                    req.allow_whole_facet,
                );
            }
            eprintln!(
                "error: --facet/--window/--plan need a single profile, but \
                 '{descriptor}' has {}: {}",
                names.len(),
                names.join(", ")
            );
            eprintln!("Choose one with `--profile <name>`.");
            2
        }
        ProfileSelection::AllProfiles => {
            let names = group.profile_names();
            eprintln!(
                "Prebuffering {descriptor} — all profiles ({})",
                names.join(", ")
            );
            drive_prebuffer_all(&group, req.allow_whole_facet)
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
    CatalogEntry {
        catalog: Catalog,
        name: String,
    },
    Local(String),
    Url(String),
}

/// Split a spec into the part that names a dataset and the profile
/// selection it implies.
///
/// Pure, so the punctuation rules can be tested on any platform —
/// they are exactly the kind that look obvious and are wrong somewhere
/// else.
///
/// The order matters:
///
/// 1. A **URL** is unambiguous.
/// 2. A **path that exists** is a path, whatever punctuation it
///    contains. This is what makes Windows work: a spec there looks
///    like `C:\\data\\ds`, and splitting on the first colon would take
///    the drive letter for a dataset name and the rest for a profile.
///    That is how `precache -d C:\\data\\ds` came to report that
///    *'C' is not a local path*.
/// 3. A **separator** means it was meant as a path even if it does not
///    exist, so the diagnostic names the path rather than hunting a
///    catalog for a fragment of it. Backslash counts, not only slash.
/// 4. Otherwise a **colon** separates a catalog name from a profile.
///
/// A local path still cannot carry a `:profile` suffix on any platform
/// — use the `--profile` flag, which is why it exists.
fn classify_spec(dataset_spec: &str) -> (&str, ProfileSelection) {
    if dataset_spec.starts_with("http://") || dataset_spec.starts_with("https://") {
        return (dataset_spec, ProfileSelection::AllProfiles);
    }
    if Path::new(dataset_spec).exists() {
        return (dataset_spec, ProfileSelection::AllProfiles);
    }
    if dataset_spec.contains('/') || dataset_spec.contains('\\') {
        return (dataset_spec, ProfileSelection::AllProfiles);
    }
    match dataset_spec.find(':') {
        Some(pos) => (
            &dataset_spec[..pos],
            ProfileSelection::Named(dataset_spec[pos + 1..].to_string()),
        ),
        None => (dataset_spec, ProfileSelection::AllProfiles),
    }
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
    let (head, profile_sel) = classify_spec(dataset_spec);

    if head.starts_with("http://") || head.starts_with("https://") {
        return Some((Resolved::Url(head.to_string()), profile_sel));
    }
    let as_path = Path::new(head);
    if as_path.exists() {
        return Some((Resolved::Local(head.to_string()), profile_sel));
    }

    let sources = build_sources(configdir, extra_catalogs, at);
    if sources.is_empty() {
        eprintln!(
            "'{}' is not a local path, not a URL, and no catalog is configured.",
            head
        );
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
        && entry.layout.profiles.profile(p).is_none()
    {
        eprintln!(
            "Profile '{p}' not found in dataset '{}'. Available: {}",
            entry.name,
            entry.profile_names().join(", ")
        );
        return None;
    }
    let name = entry.name.clone();
    Some((Resolved::CatalogEntry { catalog, name }, profile_sel))
}

// ─── Drivers ─────────────────────────────────────────────────────────

fn drive_prebuffer(view: &dyn TestDataView, allow_whole_facet: bool) -> i32 {
    let plan = match plan_prebuffer(view) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Precache: {e}");
            return 1;
        }
    };
    if plan.facets.is_empty() {
        println!("Precache: profile declared no facets.");
        return 0;
    }
    if !allow_whole_facet && !plan.unresolvable.is_empty() {
        report_unresolvable(&plan.unresolvable);
        return 2;
    }
    eprintln!(
        "Prebuffering {} facet(s), {} to download. ({} streams × {} HTTP runtimes)",
        plan.facets.len(),
        fmt_bytes(plan.total_bytes),
        crate::cache::download_concurrency(),
        crate::transport::http_runtimes()
    );
    let mut ctx = LiveCtx::new(plan.facets.len(), plan.total_bytes);
    let result = view.prebuffer_all_with_progress(
        whole_facet_fallback(allow_whole_facet),
        &mut |facet, p| ctx.on_progress(facet, p),
    );
    ctx.finalize(&result.as_ref().map(|_| ()).map_err(|e| e.to_string()));
    if result.is_err() { 1 } else { 0 }
}

/// Fetch a chosen window of chosen facets, or just say what that would
/// cost.
///
/// The plan is always printed, whether or not the fetch follows. What a
/// prefetch is about to do is the thing worth knowing: the unit of
/// fetch is a chunk, so a small window can be a large download, and a
/// window whose chunks are already resident is free. Printing the plan
/// only under `--plan` would hide that from every run that did not ask.
fn drive_selective(
    view: &dyn TestDataView,
    label: &str,
    facets: &[String],
    window: Option<&crate::dataset::source::DSWindow>,
    plan_only: bool,
    allow_whole_facet: bool,
) -> i32 {
    let manifest = view.facet_manifest();
    let selected: Vec<String> = if facets.is_empty() {
        let mut names: Vec<String> = manifest.keys().cloned().collect();
        names.sort();
        names
    } else {
        // Name a facet that does not exist and the run stops, rather
        // than quietly fetching the ones that do and reporting success.
        let mut missing: Vec<&String> = facets
            .iter()
            .filter(|f| !manifest.contains_key(f.as_str()))
            .collect();
        if !missing.is_empty() {
            missing.sort();
            let mut known: Vec<&String> = manifest.keys().collect();
            known.sort();
            eprintln!(
                "error: no such facet(s): {}",
                missing
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            eprintln!(
                "This profile declares: {}",
                known
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            return 2;
        }
        facets.to_vec()
    };

    let empty = crate::dataset::source::DSWindow(Vec::new());
    let window = window.unwrap_or(&empty);
    let window_label = if window.is_empty() {
        "whole facet".to_string()
    } else {
        format!("records {window}")
    };
    eprintln!("Precache {label} — {window_label}");

    let mut plans = Vec::new();
    for name in &selected {
        match view.prefetch_plan(name, window) {
            Ok(plan) => plans.push((name.clone(), plan)),
            Err(e) => {
                eprintln!("error: facet '{name}': {e}");
                return 1;
            }
        }
    }

    print!("{}", render_plan(&plans));
    if plan_only {
        return 0;
    }

    // Refuse the whole set before fetching any of it. Fetching the
    // facets that can be windowed and then failing on one that cannot
    // would leave the run half done for a reason the user could have
    // been told up front.
    let fallback = whole_facet_fallback(allow_whole_facet);
    if !allow_whole_facet {
        let refused: Vec<String> = plans
            .iter()
            .filter(|(_, p)| p.degrades_to_full_download)
            .map(|(n, _)| n.clone())
            .collect();
        if !refused.is_empty() {
            report_unresolvable(&refused);
            return 2;
        }
    }

    for (name, plan) in &plans {
        if plan.is_resident() {
            eprintln!("  {name}: already resident");
            continue;
        }
        let mut ctx = LiveCtx::new(1, plan.bytes_to_fetch());
        let result = view.prefetch_with_progress(name, window, fallback, &mut |p| {
            // The prefetch callback carries transport progress; the
            // renderer speaks the per-facet shape, so adapt.
            ctx.on_progress(
                name,
                &PrebufferProgress {
                    verified_chunks: p.completed_chunks(),
                    total_chunks: p.total_chunks(),
                    verified_bytes: p.downloaded_bytes(),
                    total_bytes: p.total_bytes(),
                },
            );
        });
        ctx.finalize(&result.as_ref().map(|_| ()).map_err(|e| e.to_string()));
        if let Err(e) = result {
            eprintln!("error: facet '{name}': {e}");
            return 1;
        }
    }
    0
}

/// Render one row per facet: what was asked for, what it costs, and
/// what is already there.
fn render_plan(plans: &[(String, crate::PrefetchPlan)]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "\n  {:<28} {:>10} {:>10} {:>8} {:>10} {:>8}  note",
        "facet", "to fetch", "overfetch", "requests", "resident", "index"
    );
    let mut total = 0u64;
    for (name, plan) in plans {
        total += plan.bytes_to_fetch();
        let resident = plan.fills.iter().map(|f| f.chunks_resident).sum::<u32>();
        let chunks = plan.fills.iter().map(|f| f.chunks).sum::<u32>();
        let note = if plan.degrades_to_full_download {
            "no ordinal mapping — whole facet"
        } else if plan.is_resident() {
            "already resident"
        } else {
            ""
        };
        let _ = writeln!(
            s,
            "  {:<28} {:>10} {:>10} {:>8} {:>10} {:>8}  {}",
            name,
            fmt_bytes(plan.bytes_to_fetch()),
            fmt_bytes(plan.overfetch_bytes()),
            if plan.requested_ranges.len() == plan.requests() {
                format!("{}", plan.requests())
            } else {
                // Merging happened; show what was asked for beside it.
                format!("{}/{}", plan.requests(), plan.requested_ranges.len())
            },
            if chunks == 0 {
                "local".to_string()
            } else {
                format!("{resident}/{chunks}")
            },
            if plan.prerequisite_bytes == 0 {
                "—".to_string()
            } else {
                fmt_bytes(plan.prerequisite_bytes)
            },
            note
        );
    }
    let _ = writeln!(s, "\n  {} to fetch\n", fmt_bytes(total));
    s
}

fn drive_prebuffer_all(group: &crate::TestDataGroup, allow_whole_facet: bool) -> i32 {
    let mut all_facets: Vec<FacetPlanRow> = Vec::new();
    let mut total_bytes = 0u64;
    let mut unresolvable: Vec<String> = Vec::new();
    for profile_name in group.profile_names() {
        if let Some(view) = group.profile(&profile_name) {
            let plan = match plan_prebuffer(&*view) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Precache: profile '{profile_name}': {e}");
                    return 1;
                }
            };
            total_bytes += plan.total_bytes;
            unresolvable.extend(plan.unresolvable.iter().map(|f| format!("{profile_name}/{f}")));
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
    if !allow_whole_facet && !unresolvable.is_empty() {
        report_unresolvable(&unresolvable);
        return 2;
    }
    if total_bytes >= crate::PREBUFFER_LARGE_WARNING_BYTES {
        eprintln!(
            "warning: precache announced {} across all profiles \
                   (above the {} advisory threshold).",
            fmt_bytes(total_bytes),
            fmt_bytes(crate::PREBUFFER_LARGE_WARNING_BYTES)
        );
        eprintln!(
            "Continuing — pass an explicit `dataset:profile` to limit \
                   which profiles are downloaded."
        );
    }
    eprintln!(
        "Prebuffering {} facet(s) across all profiles, {} to download.",
        all_facets.len(),
        fmt_bytes(total_bytes)
    );

    let mut ctx = LiveCtx::new(all_facets.len(), total_bytes);
    let result = group.prebuffer_all_profiles_with_progress(
        whole_facet_fallback(allow_whole_facet),
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
    /// Bytes the precache will fetch: each facet's declared window,
    /// decomposed across its shards, net of chunks already resident.
    total_bytes: u64,
    /// Facets whose declared window the format cannot map, so
    /// honouring it means fetching the facet whole. Refused up front
    /// unless the caller accepts that.
    unresolvable: Vec<String>,
}

/// The fallback a caller's `--allow-whole-facet` selects.
fn whole_facet_fallback(allow_whole_facet: bool) -> crate::view::WholeFacetFallback {
    if allow_whole_facet {
        crate::view::WholeFacetFallback::Allow
    } else {
        crate::view::WholeFacetFallback::Refuse
    }
}

/// Say which facets a window cannot be honoured for, and what to do
/// about it, before anything is fetched. Fetching the facets that can
/// be windowed and then failing on one that cannot would leave the run
/// half done for a reason the user could have been told up front.
fn report_unresolvable(refused: &[String]) {
    eprintln!(
        "error: the window cannot be resolved for {}, so honouring it \
         means fetching {} whole.",
        refused
            .iter()
            .map(|s| format!("'{s}'"))
            .collect::<Vec<_>>()
            .join(", "),
        if refused.len() == 1 {
            "that facet"
        } else {
            "those facets"
        }
    );
    eprintln!("Pass --allow-whole-facet to accept that, or drop --window.");
}

/// Size a prebuffer before any of it runs, with the plan the precache
/// will execute: each facet against the window it declares, a series
/// decomposed across its shards. One planner for the headline and the
/// fetch, so the number printed is the number downloaded.
///
/// Fails rather than guessing when a facet's window cannot be
/// interpreted. Reporting the whole file for a malformed window would
/// announce a terabyte, download it, and only then fail on the read —
/// the plan is the last cheap place to catch it.
fn plan_prebuffer(view: &dyn TestDataView) -> crate::Result<PrebufferPlan> {
    let mut facets = Vec::new();
    let mut total_bytes = 0u64;
    let mut unresolvable = Vec::new();
    for (name, desc) in view.facet_manifest() {
        // The spec's formats, not an element width: a slab holds
        // data and has no element type, and asking the wrong question
        // left metadata out of every precache.
        if !view.facet_holds_data(&name) {
            continue;
        }
        let window = crate::view::facet_declared_window(&desc)
            .map_err(|e| crate::Error::Other(format!("facet '{name}': {e}")))?;
        if let Ok(plan) = view.prefetch_plan(&name, &window) {
            if plan.degrades_to_full_download {
                total_bytes += plan.facet_bytes;
                unresolvable.push(name.clone());
            } else {
                total_bytes += plan.bytes_to_fetch();
            }
            facets.push(FacetPlanRow {
                qualified_name: name,
            });
        }
    }
    Ok(PrebufferPlan {
        facets,
        total_bytes,
        unresolvable,
    })
}

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
            last_render: std::time::Instant::now() - std::time::Duration::from_secs(1),
            started: std::time::Instant::now(),
        }
    }

    pub(super) fn on_progress(&mut self, facet: &str, p: &PrebufferProgress) {
        if facet != self.current_facet {
            self.flush_facet_summary();
            self.current_facet = facet.to_string();
            self.facet_index += 1;
            self.last_render = std::time::Instant::now() - std::time::Duration::from_secs(1);
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
            self.bytes_per_facet
                .insert(facet.to_string(), p.verified_bytes);
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
            format!(
                "{}% ({}/{})",
                pct(p.verified_bytes, p.total_bytes),
                fmt_bytes(p.verified_bytes),
                fmt_bytes(p.total_bytes)
            )
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
            format!(
                " \u{2022} {}/s \u{2022} ETA {}",
                fmt_bytes(rate as u64),
                fmt_duration(eta_secs)
            )
        } else {
            String::new()
        };
        use std::io::Write;
        eprint!(
            "\r  [{}/{}] {}: {} \u{2022} total {}% ({}/{}){}\u{1b}[K",
            self.facet_index,
            self.facet_count,
            facet,
            facet_state,
            pct_total,
            fmt_bytes(aggregate_done),
            fmt_bytes(self.total_bytes),
            trailing
        );
        let _ = std::io::stderr().flush();
    }

    /// Print a permanent "✓" line for the just-finished facet,
    /// erasing the in-place progress line first.
    fn flush_facet_summary(&self) {
        if self.current_facet.is_empty() {
            return;
        }
        let bytes = self
            .bytes_per_facet
            .get(&self.current_facet)
            .copied()
            .unwrap_or(0);
        eprintln!(
            "\r  [{}/{}] {} \u{2713} {}\u{1b}[K",
            self.facet_index,
            self.facet_count,
            self.current_facet,
            fmt_bytes(bytes)
        );
    }

    pub(super) fn finalize<T, E: std::fmt::Display>(&self, result: &Result<T, E>) {
        self.flush_facet_summary();
        let elapsed = self.started.elapsed().as_secs_f64();
        let done: u64 = self.bytes_per_facet.values().sum();
        match result {
            Ok(_) => {
                eprintln!(
                    "Precache done: {} facet(s), {} in {:.1}s ({}/s).",
                    self.facet_count,
                    fmt_bytes(done),
                    elapsed,
                    fmt_bytes((done as f64 / elapsed.max(0.001)) as u64)
                );
            }
            Err(e) => {
                eprintln!("Precache: failed — {e}");
            }
        }
    }
}

pub(super) fn pct(done: u64, total: u64) -> u32 {
    if total == 0 {
        return 100;
    }
    ((done as u128 * 100) / total as u128) as u32
}

pub(super) fn fmt_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    const TIB: u64 = 1024 * GIB;
    if bytes >= TIB {
        format!("{:.1} TiB", bytes as f64 / TIB as f64)
    } else if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
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
    if secs < M {
        format!("{secs}s")
    } else if secs < H {
        format!("{}m {:02}s", secs / M, secs % M)
    } else if secs < D {
        format!("{}h {:02}m", secs / H, (secs % H) / M)
    } else {
        format!("{}d {:02}h", secs / D, (secs % D) / H)
    }
}

#[cfg(test)]
mod spec_classification {
    use super::*;

    fn named(sel: &ProfileSelection) -> Option<&str> {
        match sel {
            ProfileSelection::Named(p) => Some(p.as_str()),
            ProfileSelection::AllProfiles => None,
        }
    }

    /// **A Windows drive letter is not a dataset name.**
    ///
    /// `C:\data\ds` has no forward slash, so the colon rule used to
    /// claim the drive letter as the dataset and the rest as a profile
    /// — `precache -d C:\data\ds` reported that *'C' is not a local
    /// path*. Six CI tests failed on Windows and nowhere else.
    #[test]
    fn a_windows_path_is_not_split_at_its_drive_letter() {
        for spec in [
            r"C:\data\ds",
            r"D:\a\b\c",
            r"C:\Users\runner\AppData\Local\Temp\x\ds",
        ] {
            let (head, sel) = classify_spec(spec);
            assert_eq!(head, spec, "the whole path is the spec: {spec}");
            assert_eq!(named(&sel), None, "a drive letter is not a profile: {spec}");
        }
    }

    /// A UNC path has no drive letter but is still a path.
    #[test]
    fn a_unc_path_is_a_path() {
        let (head, sel) = classify_spec(r"\\server\share\ds");
        assert_eq!(head, r"\\server\share\ds");
        assert_eq!(named(&sel), None);
    }

    /// The catalog form still splits — that is the whole reason the
    /// colon rule exists, and it must survive the fix.
    #[test]
    fn a_catalog_name_still_carries_its_profile() {
        let (head, sel) = classify_spec("glove-100:default");
        assert_eq!(head, "glove-100");
        assert_eq!(named(&sel), Some("default"));

        let (head, sel) = classify_spec("glove-100");
        assert_eq!(head, "glove-100");
        assert_eq!(named(&sel), None);
    }

    /// URLs are taken whole, colons and all.
    #[test]
    fn a_url_is_never_split() {
        for spec in [
            "https://example.com/data/ds",
            "http://example.com:8080/data/ds",
        ] {
            let (head, sel) = classify_spec(spec);
            assert_eq!(head, spec);
            assert_eq!(named(&sel), None, "a port is not a profile: {spec}");
        }
    }

    /// A posix path is unchanged, whether or not it exists.
    #[test]
    fn a_posix_path_is_a_path() {
        for spec in ["/tmp/ds", "./ds", "some/dir/ds"] {
            let (head, sel) = classify_spec(spec);
            assert_eq!(head, spec);
            assert_eq!(named(&sel), None);
        }
    }

    /// An existing path wins over every punctuation rule — the case
    /// that makes a real Windows spec resolve.
    #[test]
    fn an_existing_path_is_taken_whole() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = tmp.path().to_str().unwrap();
        let (head, sel) = classify_spec(spec);
        assert_eq!(head, spec);
        assert_eq!(named(&sel), None);
    }
}
