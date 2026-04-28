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
    if let Some(value) = lookup_facet(ctx, options, facet_alias) {
        return Ok(value);
    }
    let canonical = resolve_standard_key(facet_alias)
        .unwrap_or_else(|| facet_alias.to_string());
    let profile_name = options.get("profile").unwrap_or("default");
    Err(format!(
        "required option '{}' not set. The dataset's `{}` profile does not expose a `{}` facet — \
         either pass `--{} <path>`, choose a different profile with `--profile <name>`, \
         or point at a different dataset with `--dataset <dir|dataset.yaml>`.",
        option_key, profile_name, canonical, option_key,
    ))
}

fn lookup_facet(
    ctx: &StreamContext,
    options: &Options,
    facet_alias: &str,
) -> Option<String> {
    let (yaml_path, dataset_root) = resolve_dataset_paths(ctx, options)?;
    if !yaml_path.exists() {
        return None;
    }
    let cfg = DatasetConfig::load_and_resolve(&yaml_path).ok()?;
    let profile_name = options.get("profile").unwrap_or("default");
    let profile = cfg.profiles.profile(profile_name)?;
    let canonical = resolve_standard_key(facet_alias)?;
    let view = profile.view(&canonical)?;
    let raw = view.path();
    let p = Path::new(raw);
    let resolved = if p.is_absolute() { p.to_path_buf() } else { dataset_root.join(p) };
    Some(resolved.to_string_lossy().into_owned())
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
    FilteredKnnConsolidated,
    DatasetKnnutils,
    PredicateResults,
    PredicatesConsolidated,
    PredicatesSqlite,
}

impl VerifyKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::KnnGroundtruth        => "verify knn-groundtruth",
            Self::KnnConsolidated       => "verify knn-consolidated",
            Self::KnnFaissConsolidated  => "verify knn-faiss-consolidated",
            Self::FilteredKnnConsolidated => "verify filtered-knn-consolidated",
            Self::DatasetKnnutils       => "verify dataset-knnutils",
            Self::PredicateResults      => "verify predicate-results",
            Self::PredicatesConsolidated => "verify predicates-consolidated",
            Self::PredicatesSqlite      => "verify predicates-sqlite",
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
            Self::KnnConsolidated | Self::KnnFaissConsolidated | Self::FilteredKnnConsolidated => &[
                ("base", "base_vectors"),
                ("query", "query_vectors"),
            ],
            Self::PredicateResults => &[
                ("metadata", "metadata_content"),
                ("predicates", "metadata_predicates"),
                ("metadata-indices", "metadata_layout"),
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
    /// `PredicatesConsolidated` iterates by `metadata_layout` (the
    /// per-profile metadata-index file), matching the existing
    /// internal scan logic in `verify_consolidated.rs` — it's the
    /// presence of `profiles/<name>/metadata_indices.{slab,ivvec,ivec}`
    /// that gates a profile's inclusion, not the eval results.
    pub fn per_profile_facets(self) -> &'static [&'static str] {
        match self {
            Self::KnnConsolidated | Self::KnnFaissConsolidated => &["neighbor_indices"],
            Self::FilteredKnnConsolidated => &["filtered_neighbor_indices"],
            Self::PredicatesConsolidated  => &["metadata_layout"],
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
///      `profiles/<profile_name>/<canonical_basename>.<ext>` exists,
///      where `canonical_basename` matches the bootstrap-generated
///      naming convention (see `canonical_basename_for`).
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
        if facet_file_exists(dataset_root, view.path()) {
            return true;
        }
    }
    // Fallback: probe canonical filesystem layout.
    let basename = canonical_basename_for(canonical);
    let extensions = canonical_extensions_for(canonical);
    for ext in extensions {
        let candidate = dataset_root.join(format!(
            "profiles/{}/{}.{}",
            profile_name, basename, ext,
        ));
        if candidate.exists() {
            return true;
        }
    }
    false
}

/// The on-disk basename the bootstrap uses for a given canonical
/// facet. Most facets use the same name as their canonical key; the
/// historical exception is `metadata_layout` which writes as
/// `metadata_indices` on disk (matching the alias users typed).
fn canonical_basename_for(canonical: &str) -> &str {
    match canonical {
        "metadata_layout" => "metadata_indices",
        other => other, // canonical key matches on-disk basename
    }
}

/// File extensions to try when probing canonical filesystem layout
/// for a facet. The extension depends on the element type the
/// bootstrap chose, so we try the common ones.
fn canonical_extensions_for(canonical: &str) -> &'static [&'static str] {
    match canonical {
        "base_vectors" | "query_vectors" => &["fvecs", "fvec", "mvec", "dvec"],
        "neighbor_indices" | "filtered_neighbor_indices" => &["ivecs", "ivec", "slab"],
        "neighbor_distances" | "filtered_neighbor_distances" => &["fvecs", "fvec"],
        "metadata_content" | "metadata_predicates" | "metadata_results" => &["slab"],
        "metadata_layout" => &["slab", "ivvec", "ivec"],
        _ => &[],
    }
}
