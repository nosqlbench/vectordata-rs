// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! CLI support. `<binary> datasets precache`: the command-line adapter
//! over the library fetch.
//!
//! Reachable as `vectordata datasets precache` or `veks datasets
//! precache` — both binaries dispatch into this module. **Everything it
//! does is library API:** it resolves the spec with
//! [`Catalog::open_selection`](crate::catalog::Catalog::open_selection),
//! plans with [`TestDataView::plan_fetch`]
//! or [`TestDataGroup::plan_fetch`](crate::TestDataGroup::plan_fetch),
//! and executes the plan with a [`TextMeter`](crate::fetch::TextMeter)
//! attached on a terminal, a [`LogMeter`](crate::fetch::LogMeter)
//! otherwise. What this module adds is only what a command line needs:
//! string-shaped input, headings, the `--plan` table, and an exit code.
//! A program should call [`crate::fetch`] directly.
//!
//! Per facet, the reader layer decides how bytes arrive:
//!
//! - **local file** → mmap, nothing to fetch.
//! - **remote URL with `.mref`** → download and merkle-verify chunks
//!   into the cache directory, promote to mmap on completion.
//! - **remote URL without `.mref`** → download over parallel HTTP range
//!   requests, trusting TLS rather than a per-chunk hash chain, into the
//!   cache directory, promote to mmap on completion.

use std::path::{Path, PathBuf};

use super::build_sources;
use crate::catalog::resolver::Catalog;
use crate::dataset::selector::{DatasetSpec, SelectorError};
use crate::fetch::meter::fmt_bytes;
use crate::fetch::{FetchPlan, FetchProgress, FetchRequest};
use crate::{Error, TestDataView};

/// Everything a precache run needs, in command-line shape. A program
/// builds a [`FetchRequest`] instead.
///
/// A struct rather than a widening positional list: the call already
/// carried five arguments before windows and facet selection, and four
/// call sites had to be edited in lockstep every time one was added.
#[derive(Debug, Clone, Default)]
pub struct PrecacheRequest {
    /// `name`, `name:profile`, a local path, a `dataset.yaml`, or a URL.
    pub dataset_spec: String,
    /// Directory holding `catalogs.yaml` (`--configdir`).
    pub configdir: String,
    /// Catalog locations added to the configured ones (`--catalog`).
    pub extra_catalogs: Vec<String>,
    /// Catalog locations used instead of the configured ones (`--at`).
    pub at: Vec<String>,
    /// Cache directory for this run in place of the configured one
    /// (`--cache-dir`), set with [`crate::settings::set_cache_dir`].
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
/// `dataset_spec` is `<head>[:<selector>]` (PS-1), where the head is:
/// - a catalog name (e.g. `glove-100:default`, `tessera:size=10m`)
/// - a local directory containing a `dataset.yaml`
/// - a path to a `dataset.yaml` file
/// - an HTTP URL to a dataset directory or `dataset.yaml`
///
/// The selector names the profiles to fetch; precache acts on every
/// match (PS-9). A spec with no selector and no `--profile` is refused
/// naming `head:profile=*`, which is how "every profile" is spelled
/// now (PS-1).
///
/// `configdir`, `extra_catalogs`, and `at` are the catalog-source
/// inputs (same shape both binaries pass). `cache_dir`, when given, is
/// the cache this run fetches into; otherwise the configured one
/// ([`crate::settings::cache_dir`]).
///
/// Returns a process exit code: 0 on success, 1 when the fetch or a
/// lookup fails, 2 when the request itself is refused (a malformed
/// window or selector, an unknown facet, a window that would become a
/// whole-facet download).
pub fn run(req: PrecacheRequest) -> i32 {
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

    if let Some(dir) = req.cache_dir.as_deref()
        && let Err(e) = crate::settings::set_cache_dir(dir)
    {
        eprintln!("error: --cache-dir: {e}");
        return 2;
    }
    let configured = match crate::settings::cache_dir() {
        Ok(p) => Some(p),
        Err(e) => {
            // Only fatal if we'll actually need the cache. Local-only
            // datasets precache fine without one.
            eprintln!("note: {e}");
            eprintln!();
            None
        }
    };

    let (head, spec_selector) = match classify_spec(&req.dataset_spec) {
        Ok(split) => split,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    // An explicit --profile outranks whatever the spec implied; it is
    // a selector too (PS-14).
    let Some(selector) = req.profile.clone().or(spec_selector) else {
        eprintln!("error: {}", bare_spec_refusal(&head));
        return 2;
    };

    // Catalogs are loaded only for a head that names a dataset: a path
    // or URL opens directly, and fetching remote catalogs to then not
    // use them would be wasted round trips.
    let catalog = if crate::transport::is_remote_url(&head) || Path::new(&head).exists() {
        Catalog::default()
    } else {
        let sources = build_sources(&req.configdir, &req.extra_catalogs, &req.at);
        if sources.is_empty() {
            eprintln!("'{head}' is not a local path, not a URL, and no catalog is configured.");
            eprintln!("Add a catalog with:");
            eprintln!("  vectordata config catalog add <URL-or-path>");
            eprintln!("Or use --catalog/--at for one-off access.");
            return 1;
        }
        super::open_catalog(&sources)
    };
    let selection = match catalog.open_selection(&head, Some(&selector)) {
        Ok(s) => s,
        Err(e) => {
            super::report_lookup_failure(&catalog, &e);
            return 1;
        }
    };
    let descriptor = selection.dataset().to_string();

    if let Some(c) = &configured {
        eprintln!("  Cache root: {}", c.display());
    }

    let fetch = FetchRequest::facets(req.facets.iter().cloned())
        .fallback(whole_facet_fallback(req.allow_whole_facet));
    let fetch = match window {
        Some(w) => fetch.window(w),
        None => fetch,
    };
    let mut meter = crate::fetch::meter::stderr_meter("Precache");

    let names = selection.profiles();
    if let [profile_name] = names {
        let view = match selection.view() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return 1;
            }
        };
        let label = format!("{descriptor}:{profile_name}");
        if req.is_selective() {
            return drive_selective(&*view, &label, &fetch, req.plan_only, &mut *meter);
        }
        eprintln!("Prebuffering {label}");
        return drive_prebuffer(&*view, &fetch, &mut *meter);
    }
    if req.is_selective() {
        // A facet or window selection needs one profile to resolve
        // against — the same facet name means different bytes in
        // different profiles, and silently picking one would be a
        // guess presented as a result.
        eprintln!(
            "error: --facet/--window/--plan need a single profile, but \
             '{descriptor}:{selector}' matches {}: {}",
            names.len(),
            names.join(", ")
        );
        eprintln!("Narrow the selector, or choose one with `--profile <name>`.");
        return 2;
    }
    eprintln!(
        "Prebuffering {descriptor}:{selector} — {} profiles ({})",
        names.len(),
        names.join(", ")
    );
    drive_prebuffer_all(&selection, &fetch, &mut *meter)
}

/// Split a spec into the part that names a dataset and the selector
/// after it (PS-2).
///
/// Pure, so the punctuation rules can be tested on any platform —
/// they are exactly the kind that look obvious and are wrong somewhere
/// else. A **path that exists** is a path whatever punctuation it
/// contains, and the head of anything else is found by its shape: a
/// URL by scheme and authority, a path by its separators or a drive
/// letter, and a catalog name otherwise. This is what makes Windows
/// work: a spec there looks like `C:\\data\\ds`, and splitting on the
/// first colon would take the drive letter for a dataset name and the
/// rest for a profile. That is how `precache -d C:\\data\\ds` came to
/// report that *'C' is not a local path*.
///
/// A malformed selector is a selector error, never a dataset lookup.
fn classify_spec(dataset_spec: &str) -> Result<(String, Option<String>), SelectorError> {
    if Path::new(dataset_spec).exists() {
        return Ok((dataset_spec.to_string(), None));
    }
    let spec = DatasetSpec::parse(dataset_spec)?;
    Ok((spec.head, spec.selector.map(|s| s.text().to_string())))
}

/// The message a spec with no selector gets (PS-1).
///
/// A bare dataset used to precache every profile. Every other surface
/// reads no selector as `default`, and quietly meaning less than it
/// used to would be worse than either, so the old form is refused
/// naming both spellings.
fn bare_spec_refusal(head: &str) -> String {
    format!(
        "'{head}' names no profile. A bare dataset used to precache every profile; \
         say which:\n  {head}:profile=*   every profile\n  {head}:default     the default \
         profile\n  {head}:<selector>  a selection, e.g. {head}:size=10m"
    )
}

// ─── Drivers ─────────────────────────────────────────────────────────

/// Fetch one whole profile.
fn drive_prebuffer(view: &dyn TestDataView, fetch: &FetchRequest, meter: &mut dyn FetchProgress) -> i32 {
    let plan = match view.plan_fetch(fetch, meter) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Precache: {e}");
            return 1;
        }
    };
    if plan.is_empty() {
        println!("Precache: profile declared no facets.");
        return 0;
    }
    if let Some(code) = refuse(&plan) {
        return code;
    }
    eprintln!(
        "Prebuffering {} facet(s), {} to download. ({} streams × {} HTTP runtimes)",
        plan.facets().len(),
        fmt_bytes(plan.bytes_to_fetch()),
        crate::cache::download_concurrency(),
        crate::transport::http_runtimes()
    );
    execute(plan, meter)
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
    fetch: &FetchRequest,
    plan_only: bool,
    meter: &mut dyn FetchProgress,
) -> i32 {
    // A requested window overrides every selected facet's own. Absent
    // one, each facet is planned against the window it declares — the
    // same plan the whole-profile precache runs — so `--plan` and
    // `--facet` describe and fetch what the profile addresses, never
    // a sized profile's whole base.
    let window_label = match fetch.record_window() {
        Some(w) => format!("records {w}"),
        None => "each facet's declared window".to_string(),
    };
    eprintln!("Precache {label} — {window_label}");

    let plan = match view.plan_fetch(fetch, meter) {
        Ok(p) => p,
        Err(Error::UnknownFacets { missing, declared, .. }) => {
            // Name a facet that does not exist and the run stops, rather
            // than quietly fetching the ones that do and reporting success.
            eprintln!("error: no such facet(s): {}", missing.join(", "));
            eprintln!("This profile declares: {}", declared.join(", "));
            return 2;
        }
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };

    print!("{}", render_plan(&plan));
    if plan_only {
        return 0;
    }
    if let Some(code) = refuse(&plan) {
        return code;
    }
    execute(plan, meter)
}

/// Render one row per facet: what was asked for, what it costs, and
/// what is already there.
fn render_plan(plan: &FetchPlan) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "\n  {:<28} {:>10} {:>10} {:>8} {:>10} {:>8}  note",
        "facet", "to fetch", "overfetch", "requests", "resident", "index"
    );
    for facet in plan.facets() {
        let name = facet.id().to_string();
        let plan = facet.plan();
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
    let _ = writeln!(s, "\n  {} to fetch\n", fmt_bytes(plan.bytes_to_fetch()));
    s
}

/// Fetch every profile a selector matched.
fn drive_prebuffer_all(
    selection: &crate::catalog::DatasetSelection,
    fetch: &FetchRequest,
    meter: &mut dyn FetchProgress,
) -> i32 {
    let plan = match selection.plan_fetch(fetch, meter) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Precache: {e}");
            return 1;
        }
    };
    if plan.is_empty() {
        println!("Precache: no facets across the selected profiles.");
        return 0;
    }
    if let Some(code) = refuse(&plan) {
        return code;
    }
    let total = plan.bytes_to_fetch();
    if total >= crate::PREBUFFER_LARGE_WARNING_BYTES {
        eprintln!(
            "warning: precache announced {} across the selected profiles \
                   (above the {} advisory threshold).",
            fmt_bytes(total),
            fmt_bytes(crate::PREBUFFER_LARGE_WARNING_BYTES)
        );
        eprintln!(
            "Continuing — narrow the selector to limit which profiles \
                   are downloaded."
        );
    }
    eprintln!(
        "Prebuffering {} facet(s) across {} profiles, {} to download.",
        plan.facets().len(),
        selection.profiles().len(),
        fmt_bytes(total)
    );
    execute(plan, meter)
}

/// The exit code for a plan the library would refuse, before any
/// heading announces a download that will not happen. `None` when the
/// plan may run.
fn refuse(plan: &FetchPlan) -> Option<i32> {
    match plan.check() {
        Ok(()) => None,
        Err(Error::WindowUnresolvable { facets, .. }) => {
            report_unresolvable(&facets);
            Some(2)
        }
        Err(e) => {
            eprintln!("Precache: failed — {e}");
            Some(1)
        }
    }
}

/// Run a checked plan; the meter reports success or failure.
fn execute(plan: FetchPlan, meter: &mut dyn FetchProgress) -> i32 {
    match plan.execute(meter) {
        Ok(_) => 0,
        Err(_) => 1,
    }
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
/// about it, before anything is fetched.
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

#[cfg(test)]
mod spec_classification {
    use super::*;

    fn split(spec: &str) -> (String, Option<String>) {
        classify_spec(spec).unwrap_or_else(|e| panic!("{spec}: {e}"))
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
            let (head, sel) = split(spec);
            assert_eq!(head, spec, "the whole path is the spec: {spec}");
            assert_eq!(sel, None, "a drive letter is not a profile: {spec}");
        }
    }

    /// A UNC path has no drive letter but is still a path.
    #[test]
    fn a_unc_path_is_a_path() {
        let (head, sel) = split(r"\\server\share\ds");
        assert_eq!(head, r"\\server\share\ds");
        assert_eq!(sel, None);
    }

    /// The catalog form still splits — that is the whole reason the
    /// colon rule exists, and it must survive the fix — and what
    /// follows the colon is a selector now (PS-3).
    #[test]
    fn a_catalog_name_still_carries_its_profile() {
        let (head, sel) = split("glove-100:default");
        assert_eq!(head, "glove-100");
        assert_eq!(sel.as_deref(), Some("default"));

        let (head, sel) = split("glove-100");
        assert_eq!(head, "glove-100");
        assert_eq!(sel, None);

        let (head, sel) = split("tessera:size=10m,predicates=uniform*");
        assert_eq!(head, "tessera");
        assert_eq!(sel.as_deref(), Some("size=10m,predicates=uniform*"));

        assert!(classify_spec("tessera:size==10m").is_err(), "a malformed selector is an error");
    }

    /// URLs are taken whole, colons and all, and a selector may follow
    /// the path.
    #[test]
    fn a_url_is_never_split() {
        for spec in [
            "https://example.com/data/ds",
            "http://example.com:8080/data/ds",
        ] {
            let (head, sel) = split(spec);
            assert_eq!(head, spec);
            assert_eq!(sel, None, "a port is not a profile: {spec}");
        }
        let (head, sel) = split("http://example.com:8080/data/ds:10m");
        assert_eq!(head, "http://example.com:8080/data/ds");
        assert_eq!(sel.as_deref(), Some("10m"));
    }

    /// **A bare spec is refused naming the new spelling** (PS-1), so
    /// `precache ds` neither fetches everything as it used to nor
    /// quietly fetches only `default`.
    #[test]
    fn a_bare_spec_is_refused_naming_both_spellings() {
        let msg = bare_spec_refusal("tessera");
        assert!(msg.contains("tessera:profile=*"), "{msg}");
        assert!(msg.contains("tessera:default"), "{msg}");
    }

    /// A posix path is unchanged, whether or not it exists.
    #[test]
    fn a_posix_path_is_a_path() {
        for spec in ["/tmp/ds", "./ds", "some/dir/ds"] {
            let (head, sel) = split(spec);
            assert_eq!(head, spec);
            assert_eq!(sel.as_deref(), None);
        }
    }

    /// An existing path wins over every punctuation rule — the case
    /// that makes a real Windows spec resolve.
    #[test]
    fn an_existing_path_is_taken_whole() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = tmp.path().to_str().unwrap();
        let (head, sel) = split(spec);
        assert_eq!(head, spec);
        assert_eq!(sel.as_deref(), None);
    }
}

