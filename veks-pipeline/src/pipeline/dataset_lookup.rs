// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dataset-aware option resolution for standalone verify commands.
//!
//! When a verify-style command is invoked outside `veks run`, it can
//! still recover its inputs by reading the dataset's structure
//! through the vectordata `DatasetConfig` API — `profiles.<name>` for
//! the active profile and that profile's `views` keyed by canonical
//! facet name (`base_vectors`, `query_vectors`, `neighbor_indices`,
//! …). This is the right view: a dataset's *contents* live in the
//! `profiles:` block, not in the build-time `steps:` block.
//!
//! Resolution chain for a path option:
//!   1. Explicit `--<option>` on the command line.
//!   2. The given canonical facet on the resolved profile of the
//!      dataset's `dataset.yaml`.
//!   3. Original "required option not set" error, augmented to point
//!      the user at `--dataset`, `--profile`, and the explicit option.
//!
//! Dataset selection:
//!   - `--dataset <path>` if given. May be either a directory
//!     containing `dataset.yaml`, or a `dataset.yaml` file directly.
//!   - Otherwise `ctx.workspace` (i.e., the cwd when invoked through
//!     the CLI without `--dataset`).
//!
//! Profile selection: `--profile <name>`, default `"default"`.
//!
//! Output-style options (`--output`, `--report`) are not facets and
//! aren't handled here — they remain `options.require`-driven.

use std::path::{Path, PathBuf};

use vectordata::dataset::DatasetConfig;
use vectordata::dataset::facet::resolve_standard_key;

use super::command::{Options, StreamContext};

/// Resolve a path option, falling back to the matching facet on the
/// resolved profile of the dataset's `dataset.yaml`.
///
/// `option_key` is the CLI/option-map name the user might pass (e.g.,
/// `"base"`, `"indices"`, `"metadata"`). `facet_alias` is a canonical
/// facet name or recognized shorthand alias accepted by
/// `vectordata::dataset::facet::resolve_standard_key`
/// (`"base_vectors"`, `"base"`, `"neighbor_indices"`, `"gt"`, …).
pub fn resolve_path_option(
    ctx: &StreamContext,
    options: &Options,
    option_key: &str,
    facet_alias: &str,
) -> Result<String, String> {
    if let Some(v) = options.get(option_key) {
        return Ok(v.to_string());
    }
    match lookup_facet(ctx, options, facet_alias) {
        Ok(Some(value)) => return Ok(value),
        // The facet is declared, but as a series this command cannot
        // read. That is a different answer from "not declared", and
        // saying so is the whole point of refusing it (SH-74).
        Err(e) => return Err(e),
        Ok(None) => {}
    }
    let canonical = resolve_standard_key(facet_alias)
        .unwrap_or_else(|| facet_alias.to_string());
    let profile_name = options
        .get("profile")
        .or_else(|| (!ctx.profile.is_empty() && ctx.profile != "all").then_some(ctx.profile.as_str()))
        .unwrap_or("default");
    Err(format!(
        "required option '{}' not set. The dataset's `{}` profile does not expose a `{}` facet — \
         either pass `--{} <path>`, choose a different profile with `--profile <name>`, \
         or point at a different dataset with `--dataset <dir|dataset.yaml>`.",
        option_key, profile_name, canonical, option_key,
    ))
}

/// `Ok(Some(path))` when the profile names one file for the facet,
/// `Ok(None)` when it does not name the facet at all, and `Err` when it
/// names it as something this lookup cannot reduce to a path.
fn lookup_facet(
    ctx: &StreamContext,
    options: &Options,
    facet_alias: &str,
) -> Result<Option<String>, String> {
    let Some((yaml_path, dataset_root)) = resolve_dataset_paths(ctx, options) else {
        return Ok(None);
    };
    if !yaml_path.exists() {
        return Ok(None);
    }
    // A dataset that does not load is an error to report, not a facet
    // that happens to be missing.
    let cfg = DatasetConfig::load_and_resolve(&yaml_path)
        .map_err(|e| format!("cannot load {}: {}", yaml_path.display(), e))?;
    // An explicit `profile` option wins; otherwise the profile the
    // step is scoped to — the runner narrows `ctx.profile` to a
    // per-profile step's own profile — when the dataset has it; and
    // `default` last. Resolving a sized profile's step against
    // `default` gave every sized profile an E facet intersected with
    // the census profile's neighbours (TS-176).
    let profile_name = options
        .get("profile")
        .or_else(|| cfg.profiles.profile(&ctx.profile).map(|_| ctx.profile.as_str()))
        .unwrap_or("default");
    let Some(profile) = cfg.profiles.profile(profile_name) else {
        return Ok(None);
    };
    let Some(canonical) = resolve_standard_key(facet_alias) else {
        return Ok(None);
    };
    // Resolve the declared view: prefer a view keyed by the canonical
    // name, but also accept one whose (possibly legacy/alias) key
    // normalizes to the same canonical facet — e.g. a `metadata_indices`
    // view satisfies a `metadata_results` lookup. A facet is identified by
    // its canonical identity, not by a single literal key.
    let Some(view) = profile.view(&canonical).or_else(|| {
        profile
            .views()
            .find(|(k, _)| resolve_standard_key(k).as_deref() == Some(canonical.as_str()))
            .map(|(_, v)| v)
    }) else {
        return Ok(None);
    };
    // Commands take a path and read one file. A series has no single
    // one, and its first entry is a real file — so resolving through
    // `path()` would hand `compute knn` shard 0 and produce neighbours
    // over a fraction of the base with no error anywhere (SH-74,
    // SH-79). `None` here surfaces as `resolve_path_option`'s "does not
    // expose this facet" message, which stops the run.
    let Some(raw) = view.single_path().map(str::to_string) else {
        return Err(format!(
            "the dataset's `{profile_name}` profile declares `{canonical}` as a \
             multi-file series ({} shards), and this command's kernel reads one \
             mmapped file. Derive an unsharded copy first — `veks datasets derive \
             <dataset>:{profile_name} -o <dir>` with no --shard-stride writes the \
             series back as a single file (SH-38) — then run against that. Or pass \
             an explicit path. See docs/design/srd-multifile-facet-shards.md.",
            view.sources().len()
        ));
    };
    // A view path may address a slab namespace (`file#namespace`); resolve
    // the file part against the dataset root and preserve the namespace.
    let (file_part, ns_part) = match raw.split_once('#') {
        Some((f, n)) => (f, Some(n)),
        None => (raw.as_str(), None),
    };
    let p = Path::new(file_part);
    let resolved = if p.is_absolute() { p.to_path_buf() } else { dataset_root.join(p) };
    let resolved = resolved.to_string_lossy().into_owned();
    Ok(Some(match ns_part {
        Some(ns) => format!("{resolved}#{ns}"),
        None => resolved,
    }))
}

/// Resolve `--neighbors` (k), falling back to the active profile's
/// `maxk` field in `dataset.yaml`. Same dataset/profile selection as
/// `resolve_path_option`.
pub fn resolve_neighbors(
    ctx: &StreamContext,
    options: &Options,
) -> Result<usize, String> {
    if let Some(s) = options.get("neighbors") {
        let n: usize = s.parse().map_err(|e| format!("invalid neighbors '{}': {}", s, e))?;
        if n == 0 { return Err("neighbors must be > 0".into()); }
        return Ok(n);
    }
    if let Some(k) = lookup_profile_maxk(ctx, options) {
        return Ok(k);
    }
    let profile_name = options.get("profile").unwrap_or("default");
    Err(format!(
        "required option 'neighbors' not set and the `{}` profile in dataset.yaml \
         has no `maxk` to default from. Pass `--neighbors <k>` explicitly.",
        profile_name,
    ))
}

fn lookup_profile_maxk(ctx: &StreamContext, options: &Options) -> Option<usize> {
    let (yaml_path, _) = resolve_dataset_paths(ctx, options)?;
    if !yaml_path.exists() { return None; }
    let cfg = DatasetConfig::load_and_resolve(&yaml_path).ok()?;
    let profile_name = options.get("profile").unwrap_or("default");
    let profile = cfg.profiles.profile(profile_name)?;
    profile.maxk.map(|k| k as usize)
}

/// Returns `(dataset.yaml file, dataset root dir)`.
///
/// `--dataset` may point at either a directory (we append
/// `dataset.yaml`) or a file (we use it directly and treat its parent
/// as the root). Without `--dataset`, both come from `ctx.workspace`.
fn resolve_dataset_paths(ctx: &StreamContext, options: &Options) -> Option<(PathBuf, PathBuf)> {
    if let Some(p) = options.get("dataset") {
        let path = PathBuf::from(p);
        if path.is_dir() {
            let yaml = path.join("dataset.yaml");
            return Some((yaml, path));
        }
        if path.is_file() {
            let parent = path.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
            return Some((path, parent));
        }
        // Path doesn't exist yet — let the caller see the missing-file
        // error rather than silently falling through to ctx.workspace.
        return Some((path.clone(), path));
    }
    let yaml = ctx.workspace.join("dataset.yaml");
    Some((yaml, ctx.workspace.clone()))
}

// ── Facet manifest + scope validation ────────────────────────────────
//
// Each verify command declares two sets of canonical facet names:
//   - anchor_facets: required on the active profile (the "shared
//     inputs" — typically base_vectors, query_vectors, metadata).
//   - per_profile_facets: required on EACH non-partition profile
//     that this command iterates (the per-profile artifacts —
//     neighbor_indices, filtered_neighbor_indices, metadata_results).
//
// Discovery uses the vectordata `DatasetConfig` / `DSProfile` /
// `DSView` API exclusively (no YAML parsing here). File-existence
// confirmation uses the local filesystem directly via `Path::exists`,
// which is the right pattern for "what's actually here right now."

/// Identifies one of the verify commands so we can look up its
/// required-facets manifest. New commands extend this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyKind {
    KnnGroundtruth,
    KnnConsolidated,
    KnnFaissConsolidated,
    /// Legacy `verify filtered-knn-consolidated` (kept for
    /// backwards-compat; produces F-facet verification). New pipelines
    /// should use [`Self::PrefilteredKnnConsolidated`] or
    /// [`Self::PostfilteredKnnConsolidated`] as appropriate.
    FilteredKnnConsolidated,
    /// F-facet verifier — pre-filter ground truth (ACORN G_K).
    PrefilteredKnnConsolidated,
    /// E-facet verifier — post-filter ground truth (G ∩ R).
    PostfilteredKnnConsolidated,
    DatasetKnnutils,
    PredicateResults,
    PredicatesConsolidated,
    PredicatesSqlite,
}

impl VerifyKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::KnnGroundtruth          => "verify knn-groundtruth",
            Self::KnnConsolidated         => "verify knn-consolidated",
            Self::KnnFaissConsolidated    => "verify knn-faiss-consolidated",
            Self::FilteredKnnConsolidated => "verify filtered-knn-consolidated",
            Self::PrefilteredKnnConsolidated => "verify prefiltered-knn-consolidated",
            Self::PostfilteredKnnConsolidated => "verify postfiltered-knn-consolidated",
            Self::DatasetKnnutils         => "verify dataset-knnutils",
            Self::PredicateResults        => "verify predicate-results",
            Self::PredicatesConsolidated  => "verify predicates-consolidated",
            Self::PredicatesSqlite        => "verify predicates-sqlite",
        }
    }

    /// `(option_key, canonical_facet)` tuples — the shared anchor
    /// inputs this kind needs. Validation skips facets whose
    /// `option_key` is already set in `Options` (the user passed
    /// `--<option_key>` explicitly, or the pipeline runner copied
    /// the value from `dataset.yaml`'s step options).
    pub fn anchor_facets(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::KnnGroundtruth | Self::DatasetKnnutils => &[
                ("base", "base_vectors"),
                ("query", "query_vectors"),
                ("indices", "neighbor_indices"),
            ],
            Self::KnnConsolidated
                | Self::KnnFaissConsolidated
                | Self::FilteredKnnConsolidated
                | Self::PrefilteredKnnConsolidated => &[
                ("base", "base_vectors"),
                ("query", "query_vectors"),
            ],
            // E (postfilter) is derived from G + R alone; no base/query
            // anchor needed. The R input is the per-query predicate-match
            // index — the `metadata_results` facet (its on-disk file is
            // named `metadata_results.{ivvecs,slab,…}`, or `metadata_indices.*`
            // on legacy datasets; `canonical_basenames_for` resolves both).
            Self::PostfilteredKnnConsolidated => &[
                ("ground-truth", "neighbor_indices"),
                ("metadata-indices", "metadata_results"),
            ],
            Self::PredicateResults => &[
                ("metadata", "metadata_content"),
                ("predicates", "metadata_predicates"),
                ("metadata-indices", "metadata_results"),
            ],
            Self::PredicatesConsolidated => &[
                ("metadata", "metadata_content"),
                ("predicates", "metadata_predicates"),
            ],
            Self::PredicatesSqlite => &[
                ("metadata", "metadata_content"),
                ("predicates", "metadata_predicates"),
                ("results", "metadata_results"),
            ],
        }
    }

    /// Canonical facet names that must exist on EACH per-profile
    /// entry in scope. Empty for single-profile commands.
    ///
    /// `PredicatesConsolidated` iterates by `metadata_results` (the
    /// per-profile predicate-match index file), matching the existing
    /// internal scan logic in `verify_consolidated.rs` — it's the
    /// presence of `profiles/<name>/metadata_results.{ivvecs,slab,…}`
    /// (or legacy `metadata_indices.*`) that gates a profile's inclusion.
    pub fn per_profile_facets(self) -> &'static [&'static str] {
        match self {
            Self::KnnConsolidated | Self::KnnFaissConsolidated => &["neighbor_indices"],
            // F facet verifier — gated by presence of the prefiltered
            // (or legacy `filtered_`) facet on the profile.
            Self::FilteredKnnConsolidated
                | Self::PrefilteredKnnConsolidated => &["prefiltered_neighbor_indices"],
            Self::PostfilteredKnnConsolidated => &["postfiltered_neighbor_indices"],
            Self::PredicatesConsolidated  => &["metadata_results"],
            _ => &[],
        }
    }

    /// Whether this command iterates all non-partition profiles or
    /// runs against a single (anchor) profile.
    pub fn iterates_profiles(self) -> bool {
        !self.per_profile_facets().is_empty()
    }
}

/// One profile's status for the active verify command.
#[derive(Debug, Clone)]
pub struct ProfileStatus {
    pub name: String,
    /// True iff every per-profile facet is declared on the profile
    /// AND the corresponding file exists on disk.
    pub in_scope: bool,
    /// When `in_scope` is false, the canonical facet names that were
    /// missing or whose declared file doesn't exist on disk.
    pub missing: Vec<String>,
}

/// Pre-flight summary for a verify command: which dataset, which
/// anchor profile, which profiles are in scope, which are skipped
/// and why. Emitted via [`ScopeReport::log_to`] so the user sees
/// up-front exactly what will be verified.
#[derive(Debug, Clone)]
pub struct ScopeReport {
    pub kind: VerifyKind,
    pub dataset_yaml: PathBuf,
    pub anchor_profile: String,
    pub profiles: Vec<ProfileStatus>,
}

impl ScopeReport {
    pub fn in_scope(&self) -> usize {
        self.profiles.iter().filter(|p| p.in_scope).count()
    }

    /// Emit the report to the UI log so the user sees the scope
    /// before any work runs.
    pub fn log_to(&self, ctx: &mut StreamContext) {
        let n = self.in_scope();
        let total = self.profiles.len();
        ctx.ui.log(&format!(
            "{}: dataset={} anchor={} scope={}/{} profile{}",
            self.kind.label(),
            self.dataset_yaml.display(),
            self.anchor_profile,
            n, total,
            if total == 1 { "" } else { "s" },
        ));
        for p in &self.profiles {
            if p.in_scope {
                ctx.ui.log(&format!("    {} (in scope)", p.name));
            } else {
                ctx.ui.log(&format!(
                    "    {} (skipped — missing {})",
                    p.name, p.missing.join(", "),
                ));
            }
        }
    }
}

/// Validate that the dataset has the minimum facets the given verify
/// command requires, and report the per-profile scope.
///
/// Anchor-facet failures are immediate hard errors (the command
/// can't run at all). Per-profile-facet failures mark the profile
/// as out of scope and are aggregated; if NO profile is in scope
/// (for an iterating command) that's also a hard error.
///
/// If `dataset.yaml` doesn't exist (e.g., the user is invoking the
/// command on bare files via explicit `--base`/`--query`/etc.),
/// returns `Ok` with an empty profile list — validation is
/// dataset-driven and can't apply when there's no dataset.
pub fn validate_scope(
    ctx: &StreamContext,
    options: &Options,
    kind: VerifyKind,
) -> Result<ScopeReport, String> {
    let (yaml_path, dataset_root) = resolve_dataset_paths(ctx, options)
        .ok_or_else(|| format!("{}: could not resolve dataset paths", kind.label()))?;

    if !yaml_path.exists() {
        // No dataset.yaml — caller must supply paths explicitly.
        // Return an empty report so the verify command can proceed
        // and let `resolve_path_option` enforce explicit-or-fail.
        return Ok(ScopeReport {
            kind,
            dataset_yaml: yaml_path,
            anchor_profile: options.get("profile").unwrap_or("default").to_string(),
            profiles: Vec::new(),
        });
    }

    // Load via the vectordata API — same lens used everywhere else.
    let cfg = DatasetConfig::load_and_resolve(&yaml_path).map_err(|e| format!(
        "{}: failed to load {}: {}", kind.label(), yaml_path.display(), e,
    ))?;

    let anchor_name = options.get("profile").unwrap_or("default").to_string();
    let anchor = cfg.profiles.profile(&anchor_name).ok_or_else(|| format!(
        "{}: profile '{}' not found in {}. Available profiles: {}",
        kind.label(), anchor_name, yaml_path.display(),
        cfg.profiles.profiles.keys().cloned().collect::<Vec<_>>().join(", "),
    ))?;

    // Anchor-facet check: must all be present + on-disk.
    // Two sources are accepted, in priority order:
    //   (a) The profile's declared view (dataset.yaml `views:` entry).
    //   (b) A file at the canonical filesystem location
    //       `profiles/<profile_name>/<canonical_filename>.<ext>`.
    // (b) is needed because the bootstrap historically writes some
    // facets to canonical paths without registering them as views;
    // matching the existing in-tree iteration logic which probes
    // canonical paths directly.
    let mut anchor_missing: Vec<String> = Vec::new();
    for (option_key, facet) in kind.anchor_facets() {
        // Explicit `--<option_key>` (or pipeline-passed step option)
        // wins. The user has told us where to look; no need to
        // verify a facet view declaration.
        if options.get(option_key).is_some() { continue; }
        let canonical = match resolve_standard_key(facet) {
            Some(c) => c,
            None => { anchor_missing.push(format!("{} (unknown facet)", facet)); continue; }
        };
        if facet_present(&dataset_root, &anchor_name, anchor, &canonical) { continue; }
        anchor_missing.push(canonical);
    }
    if !anchor_missing.is_empty() {
        return Err(format!(
            "{}: anchor profile '{}' in {} is missing required facet{}:\n  - {}",
            kind.label(),
            anchor_name,
            yaml_path.display(),
            if anchor_missing.len() == 1 { "" } else { "s" },
            anchor_missing.join("\n  - "),
        ));
    }

    // Per-profile scope.
    let mut profiles: Vec<ProfileStatus> = Vec::new();
    if kind.iterates_profiles() {
        for (name, profile) in &cfg.profiles.profiles {
            // Partition profiles have their own per-partition
            // verification flow; they're never part of consolidated
            // scope (this matches the existing scan logic).
            if profile.partition { continue; }
            let mut missing = Vec::new();
            for facet in kind.per_profile_facets() {
                let canonical = match resolve_standard_key(facet) {
                    Some(c) => c,
                    None => { missing.push(format!("{} (unknown facet)", facet)); continue; }
                };
                if !facet_present(&dataset_root, name, profile, &canonical) {
                    missing.push(canonical);
                }
            }
            profiles.push(ProfileStatus {
                name: name.clone(),
                in_scope: missing.is_empty(),
                missing,
            });
        }
        // Stable ordering: in-scope first, then alphabetical.
        profiles.sort_by(|a, b| b.in_scope.cmp(&a.in_scope).then(a.name.cmp(&b.name)));

        if profiles.iter().all(|p| !p.in_scope) {
            let summary: Vec<String> = profiles.iter()
                .map(|p| format!("    {} — missing {}", p.name, p.missing.join(", ")))
                .collect();
            return Err(format!(
                "{}: no profiles in {} have the required per-profile facet{} ({}). \
                 Profiles checked:\n{}",
                kind.label(),
                yaml_path.display(),
                if kind.per_profile_facets().len() == 1 { "" } else { "s" },
                kind.per_profile_facets().join(", "),
                summary.join("\n"),
            ));
        }
    } else {
        profiles.push(ProfileStatus {
            name: anchor_name.clone(),
            in_scope: true,
            missing: Vec::new(),
        });
    }

    Ok(ScopeReport {
        kind,
        dataset_yaml: yaml_path,
        anchor_profile: anchor_name,
        profiles,
    })
}

/// Convenience wrapper: validate the scope and immediately log the
/// report to the UI. Returns the report so callers can inspect or
/// defer further work to it.
pub fn validate_and_log(
    ctx: &mut StreamContext,
    options: &Options,
    kind: VerifyKind,
) -> Result<ScopeReport, String> {
    let report = validate_scope(ctx, options, kind)?;
    report.log_to(ctx);
    Ok(report)
}

/// Resolve a view's path against the dataset root and check that the
/// file exists locally. Direct filesystem access is correct here:
/// "does this file exist right now, on this machine?" is a local
/// question, not a dataset-API one.
fn facet_file_exists(dataset_root: &Path, raw_path: &str) -> bool {
    let p = Path::new(raw_path);
    let resolved = if p.is_absolute() { p.to_path_buf() } else { dataset_root.join(p) };
    resolved.exists()
}

/// Is the canonical facet `canonical` present for `profile_name`?
///
/// Two acceptance paths, in priority:
///   1. The profile declares a view for `canonical` AND the
///      view's file exists on disk.
///   2. The default canonical filesystem layout
///      `profiles/<profile_name>/<basename>.<ext>` exists, for any
///      basename/format the facet permits — driven by the `vectordata`
///      facet spec (`StandardFacet::basenames`/`formats`), the single
///      authority for a facet's valid resources.
///
/// (2) exists for backward compat with bootstrap-generated datasets
/// that write artifacts to canonical paths without registering them
/// as views. The existing internal iteration in
/// `verify_consolidated.rs` already probes the canonical paths;
/// this keeps validation behavior aligned with the iteration.
fn facet_present(
    dataset_root: &Path,
    profile_name: &str,
    profile: &vectordata::dataset::profile::DSProfile,
    canonical: &str,
) -> bool {
    if let Some(view) = profile.view(canonical) {
        // A view path may address a namespace within a slab
        // (`file#namespace`); existence is a property of the file.
        if facet_file_exists(dataset_root, strip_namespace(view.path())) {
            return true;
        }
    }
    // Fallback: probe the canonical filesystem layout, driven entirely by
    // the `vectordata` facet spec (the single authority for a facet's valid
    // basenames + formats/extensions). A facet may own its file under more
    // than one basename (canonical + legacy) and several formats.
    if let Some(facet) = vectordata::dataset::facet::StandardFacet::from_key(canonical) {
        for basename in facet.basenames() {
            for format in facet.formats() {
                for ext in format.extensions() {
                    let candidate = dataset_root.join(format!(
                        "profiles/{}/{}.{}",
                        profile_name, basename, ext,
                    ));
                    if candidate.exists() {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Strip a `#namespace` suffix from a view path, leaving the file path.
/// `metadata_content.slab#layout` → `metadata_content.slab`.
fn strip_namespace(path: &str) -> &str {
    path.split('#').next().unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::{Options, StreamContext};
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn ctx_at(workspace: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            cache: workspace.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    fn workspace_with(yaml: &str) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("dataset.yaml"), yaml).unwrap();
        tmp
    }

    /// A step scoped to a sized profile resolves that profile's facet;
    /// a run-level selection that names no profile, or nothing, falls
    /// to `default`; an explicit `profile` option wins over both.
    #[test]
    fn a_scoped_step_resolves_its_own_profiles_facet() {
        let tmp = workspace_with(
            "name: d\nprofiles:\n  default:\n    neighbor_indices: profiles/default/g.ivecs\n  100k:\n    base_count: 100000\n    neighbor_indices: profiles/100k/g.ivecs\n",
        );
        let mut ctx = ctx_at(tmp.path());
        ctx.profile = "100k".into();
        let got = resolve_path_option(&ctx, &Options::new(), "ground-truth", "neighbor_indices").unwrap();
        assert!(got.ends_with("profiles/100k/g.ivecs"), "{got}");
        ctx.profile = "all".into();
        let got = resolve_path_option(&ctx, &Options::new(), "ground-truth", "neighbor_indices").unwrap();
        assert!(got.ends_with("profiles/default/g.ivecs"), "{got}");
        ctx.profile = String::new();
        let got = resolve_path_option(&ctx, &Options::new(), "ground-truth", "neighbor_indices").unwrap();
        assert!(got.ends_with("profiles/default/g.ivecs"), "{got}");
        ctx.profile = "100k".into();
        let mut o = Options::new();
        o.set("profile", "default");
        let got = resolve_path_option(&ctx, &o, "ground-truth", "neighbor_indices").unwrap();
        assert!(got.ends_with("profiles/default/g.ivecs"), "{got}");
    }

    /// A profile materialised by `veks prepare stratify` — numeric
    /// name, windowed base and metadata views — resolves its own
    /// per-profile facets.
    #[test]
    fn a_materialised_sized_profile_resolves_its_own_facets() {
        let tmp = workspace_with(
            "name: d\nstrata:\n  one:\n    spec: \"100\"\n    series: [\"100\"]\nprofiles:\n  default:\n    maxk: 5\n    base_vectors: profiles/base/base_vectors.fvecs\n    neighbor_indices: profiles/default/neighbor_indices.ivecs\n    metadata_results: profiles/default/metadata_results.slab\n  100:\n    maxk: 5\n    base_count: 100\n    base_vectors:\n      source: profiles/base/base_vectors.fvecs\n      window: \"[0..100]\"\n    metadata_content:\n      source: profiles/base/metadata_content.slab\n      window: \"[0..100]\"\n    neighbor_indices: profiles/100/neighbor_indices.ivecs\n    neighbor_distances: profiles/100/neighbor_distances.fvecs\n",
        );
        let cfg = vectordata::dataset::DatasetConfig::load_and_resolve(&tmp.path().join("dataset.yaml"))
            .unwrap_or_else(|e| panic!("load_and_resolve: {e}"));
        let names: Vec<&String> = cfg.profiles.profiles.keys().collect();
        assert!(cfg.profiles.profile("100").is_some(), "profile 100 missing; profiles are {names:?}");
        let views: Vec<String> = cfg.profiles.profile("100").unwrap().views().map(|(k, _)| k.to_string()).collect();
        assert!(views.iter().any(|k| k == "neighbor_indices"), "views of 100: {views:?}");
        let mut ctx = ctx_at(tmp.path());
        ctx.profile = "100".into();
        let got = resolve_path_option(&ctx, &Options::new(), "ground-truth", "neighbor_indices").unwrap();
        assert!(got.ends_with("profiles/100/neighbor_indices.ivecs"), "{got}");
        // The materialised entry declares no predicate results; they
        // are not default's either.
        let err = resolve_path_option(&ctx, &Options::new(), "metadata-indices", "metadata_results").unwrap_err();
        assert!(err.contains("`100` profile does not expose"), "{err}");
    }

    /// A single-file facet resolves to that file, as it always has.
    #[test]
    fn a_single_file_facet_resolves_to_its_path() {
        let tmp = workspace_with(
            "name: d\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
        );
        let got = resolve_path_option(&ctx_at(tmp.path()), &Options::new(), "base", "base_vectors")
            .expect("a single file resolves");
        assert!(got.ends_with("base.fvec"), "{got}");
    }

    /// **An explicit series is refused, not resolved to shard 0**
    /// (SH-74, SH-79).
    ///
    /// This is the dangerous case: `part_a.fvec` is a real file, so a
    /// lookup that reached for `path()` would hand a single-file
    /// command a fifth of the base and it would compute neighbours over
    /// that fraction and report success. The refusal has to name
    /// sharding, or the operator reads "facet not exposed" about a
    /// facet that plainly is.
    #[test]
    fn an_explicit_series_is_refused_rather_than_resolved_to_its_first_shard() {
        let tmp = workspace_with(
            "name: d\nprofiles:\n  default:\n    base_vectors:\n      source:\n\
             \x20       - part_a.fvec=100\n        - part_b.fvec=100\n      record_count: 200\n",
        );
        let err = resolve_path_option(&ctx_at(tmp.path()), &Options::new(), "base", "base_vectors")
            .expect_err("a series must not resolve to a path");
        assert!(err.contains("multi-file series"), "{err}");
        assert!(err.contains("2 shards"), "{err}");
        assert!(!err.contains("part_a"), "the first shard must not be offered: {err}");
    }

    /// A uniform series is refused for the same reason, even though its
    /// pattern names no file — the diagnosis should be the same one.
    #[test]
    fn a_uniform_series_is_refused_with_the_same_diagnosis() {
        let tmp = workspace_with(
            "name: d\nprofiles:\n  default:\n    base_vectors:\n      source: base__NNNN.fvec\n\
             \x20     shard_stride: 100\n      shard_count: 5\n      record_count: 500\n",
        );
        let err = resolve_path_option(&ctx_at(tmp.path()), &Options::new(), "base", "base_vectors")
            .expect_err("a series must not resolve to a path");
        assert!(err.contains("multi-file series"), "{err}");
        assert!(!err.contains("NNNN"), "the pattern must not be offered as a path: {err}");
    }

    /// An explicit `--base` still wins: the refusal is about what the
    /// dataset declares, not a veto on the command.
    #[test]
    fn an_explicit_option_overrides_a_sharded_declaration() {
        let tmp = workspace_with(
            "name: d\nprofiles:\n  default:\n    base_vectors:\n      source: base__NNNN.fvec\n\
             \x20     shard_stride: 100\n      shard_count: 5\n      record_count: 500\n",
        );
        let mut opts = Options::new();
        opts.set("base", "/elsewhere/base.fvec");
        let got = resolve_path_option(&ctx_at(tmp.path()), &opts, "base", "base_vectors").unwrap();
        assert_eq!(got, "/elsewhere/base.fvec");
    }

    /// A facet the profile does not declare still reports as missing,
    /// with the guidance it always had.
    #[test]
    fn an_absent_facet_still_reports_as_absent() {
        let tmp = workspace_with(
            "name: d\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
        );
        let err = resolve_path_option(&ctx_at(tmp.path()), &Options::new(), "gt", "neighbor_indices")
            .expect_err("an undeclared facet has no path");
        assert!(err.contains("does not expose"), "{err}");
        assert!(!err.contains("series"), "{err}");
    }
}
