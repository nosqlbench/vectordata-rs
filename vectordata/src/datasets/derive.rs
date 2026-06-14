// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `<binary> datasets derive` — materialize a profile of an
//! existing dataset as a self-standing dataset.
//!
//! Where `precache` brings a profile's bytes into the local cache
//! (still resolved through the parent dataset.yaml), `derive` copies
//! those bytes into a new directory and emits a fresh dataset.yaml
//! whose `default` profile points at local files only — no parent,
//! no windowed references, no shared facets.
//!
//! The whole point is to flatten windowed views into their own
//! files. A profile like `vecs1m:25` has `base_vectors` declared as
//! `profiles/base/base_vectors.fvecs[0..25)` — a 25-vector window
//! into the full base. After `derive` the output directory holds a
//! 25-vector `base_vectors.fvecs` file that any consumer can open
//! directly with no awareness of the windowing.
//!
//! Per-facet plan:
//!
//! - **Scalar packed** (`.u8`, `.i32`, …): record size = element
//!   byte width. Window intervals are byte ranges; copy them
//!   sequentially.
//! - **Uniform xvec** (`.fvecs`, `.ivecs`, …): record size = `4 +
//!   dim * byte_width(elem)` where `dim` comes from the first
//!   record's i32 header. Window intervals are record ranges;
//!   copy `record_count * record_size` bytes per interval.
//! - **Slab** (`.slab`): an empty window byte-copies the file as-is
//!   (the slab is self-describing — its embedded pages page stays
//!   valid). A windowed slab is sliced on record boundaries via
//!   slabtastic, preserving the optional `:schema` sidecar.
//! - **Variable-length vvec**: not yet supported — emits an
//!   actionable error pointing the user at `transform extract`.
//!
//! Each materialized file gets a fresh `.mref` sibling via
//! [`crate::merkle::MerkleRef::from_content`] so the derived
//! dataset is immediately publishable.

use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::build_sources;
use crate::catalog::resolver::Catalog;
use crate::dataset::config::DatasetConfig as RichDatasetConfig;
use crate::dataset::source::DSWindow;
use crate::merkle::MerkleRef;
use crate::typed_access::ElementType;

/// Default chunk size for merkle-tree generation on the derived
/// files. 1 MiB matches `merkle create`'s default.
const MERKLE_CHUNK_SIZE: u64 = 1024 * 1024;

/// Kind of source container the facet stores. Drives which
/// `materialize_*` path runs and how the planner computes the
/// expected output size.
#[derive(Debug, Clone, Copy)]
enum FacetKind {
    /// Packed scalar payload (`.u8`, `.i32`, …). Record size is the
    /// element byte width.
    Scalar(ElementType),
    /// Uniform xvec payload (`.fvecs`, `.ivecs`, …). Record size is
    /// `4 + dim * byte_width(elem)` once the dim header is read.
    UniformXvec(ElementType),
    /// Variable-length vvec — not yet materializable here; surfaces
    /// an actionable error at materialize time.
    VariableVvec,
    /// Slabtastic `.slab` container — variable-length typed records
    /// addressed by ordinal. Window slicing goes through slabtastic
    /// rather than byte-range copy.
    Slab,
}

/// Per-facet plan row produced before any I/O. Lets the live
/// meter show "[N/M]" and an overall percentage instead of just
/// a running byte counter.
struct PlanRow {
    facet: String,
    src: std::path::PathBuf,
    dest_filename: String,
    kind: FacetKind,
    window: DSWindow,
    /// Expected number of output bytes for this facet, used to
    /// drive the aggregate progress meter. Computed from the
    /// source file (dim header) + window before any writes.
    expected_bytes: u64,
}

/// Compute the expected output size of a facet given its window.
/// Used during planning so the meter has a real total.
fn plan_output_size(src: &Path, kind: FacetKind, window: &DSWindow) -> io::Result<u64> {
    // For slabs and variable-length vvecs we don't have a cheap
    // record-size formula; an empty window still byte-copies, but
    // anything else is best counted at materialize time.
    if window.is_empty() {
        return fs::metadata(src).map(|m| m.len());
    }
    let record_size = match kind {
        FacetKind::Scalar(elem) => elem.byte_width() as u64,
        FacetKind::UniformXvec(elem) => {
            let mut f = fs::File::open(src)?;
            if f.metadata()?.len() < 4 { return Ok(0); }
            let mut dim_bytes = [0u8; 4];
            f.read_exact(&mut dim_bytes)?;
            let dim = i32::from_le_bytes(dim_bytes) as u64;
            if dim == 0 { return Ok(0); }
            4 + dim * elem.byte_width() as u64
        }
        // Slab and vvec are variable-length — return 0 so the meter
        // ticks against the running byte count. The actual write is
        // bounded by the materialize routine.
        FacetKind::VariableVvec | FacetKind::Slab => return Ok(0),
    };
    let mut total = 0u64;
    for iv in &window.0 {
        total = total.saturating_add(
            (iv.max_excl - iv.min_incl).saturating_mul(record_size));
    }
    Ok(total)
}

/// Entry point for `<binary> datasets derive`.
///
/// `dataset` selects the source dataset (catalog name, local
/// directory containing `dataset.yaml`, path to a `dataset.yaml`
/// file, or HTTPS URL). `profile` names the profile to flatten.
///
/// Local sources (directory or `.yaml` file) take a **fast path**
/// that loads the dataset config + materialises facet files
/// directly, with no `TestDataGroup` / `Storage` / cache involved
/// — useful for derivations on workspaces that aren't fully
/// wired into the runtime access layer (no `.mref`, no
/// `settings.yaml` cache_dir, etc.). Catalog / URL sources go
/// through the runtime access layer (precache-then-copy).
///
/// Returns a process exit code (0 on success).
pub fn run(
    dataset: &str,
    profile: &str,
    output: &Path,
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
    name_override: Option<&str>,
    force: bool,
) -> i32 {
    if let Err(e) = preflight_output(output, force) {
        eprintln!("{e}");
        return 1;
    }

    // Fast local path: a directory containing dataset.yaml, or a
    // direct path to a dataset.yaml file. No catalog lookup, no
    // runtime access layer, no precache — just read the YAML
    // and slice the files in place.
    if let Some(yaml_path) = local_dataset_yaml(dataset) {
        return derive_local(&yaml_path, profile, output, name_override);
    }

    // Otherwise: catalog / URL → runtime access layer.
    derive_via_access_layer(
        dataset, profile, output, configdir, extra_catalogs, at, name_override)
}

/// If `dataset` points at a local directory containing a
/// `dataset.yaml`, or directly at a `dataset.yaml` file, return
/// the resolved path. Returns `None` for URLs, catalog names,
/// directories with only a `knn_entries.yaml` (those flow through
/// the access-layer path because `derive_local` consumes the rich
/// `DatasetConfig` schema directly), or anything not on disk.
fn local_dataset_yaml(dataset: &str) -> Option<std::path::PathBuf> {
    if dataset.starts_with("http://") || dataset.starts_with("https://") {
        return None;
    }
    let p = Path::new(dataset);
    if !p.exists() { return None; }
    if p.is_dir() {
        let yaml = p.join("dataset.yaml");
        if yaml.is_file() { return Some(yaml); }
        return None;
    }
    if p.extension().is_some_and(|e| e == "yaml" || e == "yml") {
        return Some(p.to_path_buf());
    }
    None
}

fn preflight_output(output: &Path, force: bool) -> Result<(), String> {
    if output.exists() && !force {
        return Err(format!("Output {} already exists. Pass --force to overwrite.",
            output.display()));
    }
    if output.exists() {
        fs::remove_dir_all(output)
            .map_err(|e| format!("Failed to remove existing {}: {e}", output.display()))?;
    }
    fs::create_dir_all(output)
        .map_err(|e| format!("Failed to create {}: {e}", output.display()))?;
    Ok(())
}

/// Pure-local derive: load `dataset.yaml` directly, walk views,
/// materialise from `<dataset-dir>/<view.path>` to `<output>/…`.
/// Bypasses `TestDataGroup`, `Storage`, the cache, and
/// `settings.yaml` entirely.
fn derive_local(
    yaml_path: &Path,
    profile_name: &str,
    output: &Path,
    name_override: Option<&str>,
) -> i32 {
    let base_dir = yaml_path.parent().unwrap_or(Path::new("."));
    let config = match RichDatasetConfig::load(yaml_path) {
        Ok(c) => c,
        Err(e) => { eprintln!("error: failed to load {}: {e}", yaml_path.display()); return 1; }
    };
    let ds_profile = match config.profiles.profile(profile_name) {
        Some(p) => p,
        None => {
            eprintln!("Profile '{profile_name}' not found in {}.", yaml_path.display());
            let names: Vec<&str> = config.profiles.profiles.keys()
                .map(|s| s.as_str()).collect();
            eprintln!("Available: {}", names.join(", "));
            return 1;
        }
    };

    let plan = match build_plan_local(base_dir, ds_profile, output) {
        Ok(p) => p,
        Err(e) => { eprintln!("{e}"); return 1; }
    };
    let donor_name = yaml_path.parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("derived")
        .to_string();
    run_plan(&plan, output, &donor_name, profile_name,
        &yaml_path.display().to_string(), ds_profile, name_override,
        /* local_fast_path = */ true)
}

/// Slow path: catalog or URL source. Goes through the runtime
/// access layer so remote bytes are fetched + verified into the
/// cache, then read from there.
fn derive_via_access_layer(
    dataset: &str,
    profile_name: &str,
    output: &Path,
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
    name_override: Option<&str>,
) -> i32 {
    let (resolution, derived_default_name) =
        match resolve_spec(dataset, configdir, extra_catalogs, at) {
            Some(t) => t,
            None => return 1,
        };

    // Catalog-resolved entries open through `Catalog::open(name)` so
    // the knn_entries-shape synthesis path is taken when applicable.
    // (resolve_spec already rejected knn_entries-shape entries above —
    // derive needs the per-dataset dataset.yaml for window metadata,
    // and those catalogs don't publish one.)
    let (group, yaml_url) = match resolution {
        Resolved::CatalogEntry { catalog, name, yaml_url } => {
            let g = match catalog.open(&name) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("error: failed to open dataset '{name}': {e}");
                    return 1;
                }
            };
            (g, yaml_url)
        }
        Resolved::Local(path) | Resolved::Url(path) => {
            let g = match crate::TestDataGroup::load(&path) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("error: failed to open dataset at {path}: {e}");
                    return 1;
                }
            };
            (g, path)
        }
    };
    let view = match group.profile(profile_name) {
        Some(v) => v,
        None => {
            eprintln!("Profile '{profile_name}' not found at {yaml_url}.");
            eprintln!("Available: {}", group.profile_names().join(", "));
            return 1;
        }
    };
    let rich = match load_rich_config(&yaml_url) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to re-parse dataset.yaml for window info: {e}");
            return 1;
        }
    };
    let ds_profile = match rich.profiles.profile(profile_name) {
        Some(p) => p,
        None => {
            eprintln!("Profile '{profile_name}' missing from rich config (internal).");
            return 1;
        }
    };

    eprintln!("Prebuffering source profile so windows can be sliced locally…");
    if let Err(e) = view.prebuffer_all() {
        eprintln!("error: failed to precache source: {e}");
        return 1;
    }

    let plan = match build_plan_via_view(&*view, ds_profile, output) {
        Ok(p) => p,
        Err(e) => { eprintln!("{e}"); return 1; }
    };
    run_plan(&plan, output, &derived_default_name, profile_name,
        dataset, ds_profile, name_override,
        /* local_fast_path = */ false)
}

// ─── Planning ─────────────────────────────────────────────────────

fn build_plan_local(
    base_dir: &Path,
    ds_profile: &crate::dataset::profile::DSProfile,
    output: &Path,
) -> Result<Vec<PlanRow>, String> {
    let mut rows = Vec::new();
    for (facet_name, dview) in ds_profile.views() {
        let raw_path = dview.path();
        let src_path = base_dir.join(raw_path);
        if !src_path.is_file() {
            return Err(format!("Facet '{facet_name}': source {} not found.",
                src_path.display()));
        }
        rows.push(plan_row_for(facet_name, src_path, dview.effective_window().clone(), output)?);
    }
    Ok(rows)
}

fn build_plan_via_view(
    view: &dyn crate::TestDataView,
    ds_profile: &crate::dataset::profile::DSProfile,
    output: &Path,
) -> Result<Vec<PlanRow>, String> {
    let mut rows = Vec::new();
    for (facet_name, dview) in ds_profile.views() {
        if view.facet_element_type(facet_name).is_err() {
            // Skip non-data facets.
            continue;
        }
        let storage = view.open_facet_storage(facet_name)
            .map_err(|e| format!("Failed to open facet '{facet_name}': {e}"))?;
        let src_path = if let Some(p) = storage.cache_path() {
            p
        } else if let Some(s) = view.facet_source(facet_name) {
            if s.starts_with("http://") || s.starts_with("https://") {
                return Err(format!("Facet '{facet_name}': source is direct HTTP ({s}) \
                    — no `.mref` published, so derive has no integrity-checked \
                    snapshot to copy from."));
            }
            std::path::PathBuf::from(s)
        } else {
            return Err(format!("Facet '{facet_name}': cannot resolve source path."));
        };
        rows.push(plan_row_for(facet_name, src_path, dview.effective_window().clone(), output)?);
    }
    Ok(rows)
}

fn plan_row_for(
    facet_name: &str,
    src_path: std::path::PathBuf,
    window: DSWindow,
    output: &Path,
) -> Result<PlanRow, String> {
    let src_ext = src_path.extension()
        .and_then(|e| e.to_str()).unwrap_or("").to_string();
    let kind = classify_facet(facet_name, &src_path, &src_ext)?;
    // Canonical layout: every materialised facet lives under
    // `profiles/base/`, matching the source-side convention used by
    // `precache` and every dataset.yaml fixture in the workspace.
    // The dest filename stored here is the YAML-visible path
    // relative to the dataset root, so dataset.yaml and the on-disk
    // location stay in sync automatically.
    let dest_filename = format!("profiles/base/{facet_name}.{src_ext}");
    let _dest_path = output.join(&dest_filename); // computed lazily downstream
    let expected_bytes = plan_output_size(&src_path, kind, &window)
        .map_err(|e| format!("Facet '{facet_name}': cannot plan output size: {e}"))?;
    Ok(PlanRow {
        facet: facet_name.to_string(),
        src: src_path,
        dest_filename,
        kind,
        window,
        expected_bytes,
    })
}

/// Map a source extension to a [`FacetKind`]. Slab gets its own
/// branch — it's a typed-record container, not a fixed-width
/// numeric file, so it has no [`ElementType`].
fn classify_facet(
    facet_name: &str,
    src_path: &Path,
    src_ext: &str,
) -> Result<FacetKind, String> {
    if src_ext.eq_ignore_ascii_case("slab") {
        return Ok(FacetKind::Slab);
    }
    // Validate that the extension maps to a known element type — both
    // vvec and uniform xvec / scalar paths need this check.
    if src_ext.contains("vvec") {
        if ElementType::from_extension(src_ext).is_none() {
            return Err(format!(
                "Facet '{facet_name}': unknown element type for extension '{src_ext}'."
            ));
        }
        return Ok(FacetKind::VariableVvec);
    }
    let element_type = ElementType::from_extension(src_ext).ok_or_else(|| {
        format!(
            "Facet '{facet_name}': unknown element type for extension '{src_ext}'."
        )
    })?;
    if ElementType::is_scalar_format(src_path) {
        Ok(FacetKind::Scalar(element_type))
    } else {
        Ok(FacetKind::UniformXvec(element_type))
    }
}

// ─── Plan execution + progress ────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_plan(
    plan: &[PlanRow],
    output: &Path,
    source_label: &str,
    source_profile: &str,
    derived_from_attr: &str,
    src_profile: &crate::dataset::profile::DSProfile,
    name_override: Option<&str>,
    local_fast_path: bool,
) -> i32 {
    let total_bytes: u64 = plan.iter().map(|r| r.expected_bytes).sum();
    eprintln!("Materializing {} facet(s), {} to write.",
        plan.len(), super::precache::fmt_bytes(total_bytes));

    let mut meter = DeriveMeter::new(plan.len(), total_bytes);
    let mut derived_facets: Vec<(String, String)> = Vec::new();

    for row in plan {
        let dest_path = output.join(&row.dest_filename);
        // `dest_filename` is `profiles/base/<facet>.<ext>` — ensure
        // the containing directory exists before any sink opens it
        // for writing. Idempotent across facets.
        if let Some(parent) = dest_path.parent()
            && let Err(e) = fs::create_dir_all(parent) {
                meter.fail(
                    &row.facet,
                    &format!("create dir {}: {e}", parent.display()),
                );
                return 1;
            }
        meter.begin_facet(&row.facet, row.expected_bytes);

        let mut written: u64 = 0;
        let res = materialize_facet(
            &row.facet, &row.src, &dest_path,
            row.kind, &row.window,
            |delta| {
                written = written.saturating_add(delta);
                meter.tick_copy(written);
            });
        if let Err(e) = res {
            meter.fail(&row.facet, &e.to_string());
            return 1;
        }

        // Merkle generation. The .mref is computed from the
        // (windowed) output bytes — the donor's mref doesn't
        // apply once the content changes. Stream the file so 10+
        // GiB facets don't allocate a giant Vec, and tick the meter
        // by hashed bytes so users see real progress instead of a
        // frozen "computing merkle…" line.
        let merkle_total = std::fs::metadata(&dest_path)
            .map(|m| m.len())
            .unwrap_or(0);
        meter.begin_merkle(merkle_total);
        let merkle_res = generate_mref(&dest_path, |hashed| meter.tick_merkle(hashed));
        if let Err(e) = merkle_res {
            meter.fail(&row.facet, &format!("mref: {e}"));
            return 1;
        }
        meter.end_facet(&row.facet, row.expected_bytes);
        derived_facets.push((row.facet.clone(), row.dest_filename.clone()));
    }

    let derived_name = name_override
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{source_label}-{source_profile}"));
    if let Err(e) = write_dataset_yaml(output, &derived_name,
        derived_from_attr, source_profile, &derived_facets, src_profile)
    {
        eprintln!("error: failed to write dataset.yaml: {e}");
        return 1;
    }

    meter.summary();
    println!();
    println!("Derived dataset '{derived_name}' at {}:", output.display());
    if local_fast_path {
        println!("  {} facet(s)  (local fast path — no cache/access-layer)",
            derived_facets.len());
    } else {
        println!("  {} facet(s)  (via runtime access layer)",
            derived_facets.len());
    }
    println!("  dataset.yaml");
    0
}

/// Which phase of a facet's lifecycle the meter is currently
/// rendering. Each phase has its own progress numerator/denominator,
/// so e.g. a 10 GiB merkle hash on top of a finished copy doesn't
/// sit at "computing merkle…" without movement.
#[derive(Debug, Clone, Copy)]
enum Phase {
    Copy,
    Merkle,
}

/// Single-line stderr meter for derive. Each facet goes through
/// two phases — *copy* (live byte progress as data is written) and
/// *merkle* (live byte progress as the `.mref` is streamed). The
/// aggregate counter on the right of the line shows total bytes
/// copied across all facets against the planned total.
struct DeriveMeter {
    facet_count: usize,
    total_bytes: u64,
    bytes_done_in_prior_facets: u64,
    current_facet: String,
    /// Bytes copied for the current facet (used by the aggregate
    /// total% and by the per-facet line during the copy phase).
    current_facet_bytes: u64,
    /// Expected number of bytes for the current facet's copy phase.
    current_facet_total: u64,
    /// Bytes hashed so far for the current facet's `.mref`.
    current_merkle_bytes: u64,
    /// Total file bytes the current facet's `.mref` will hash.
    current_merkle_total: u64,
    phase: Phase,
    facet_index: usize,
    last_render: std::time::Instant,
    started: std::time::Instant,
}

impl DeriveMeter {
    fn new(facet_count: usize, total_bytes: u64) -> Self {
        Self {
            facet_count, total_bytes,
            bytes_done_in_prior_facets: 0,
            current_facet: String::new(),
            current_facet_bytes: 0,
            current_facet_total: 0,
            current_merkle_bytes: 0,
            current_merkle_total: 0,
            phase: Phase::Copy,
            facet_index: 0,
            last_render: std::time::Instant::now() - std::time::Duration::from_secs(1),
            started: std::time::Instant::now(),
        }
    }

    fn begin_facet(&mut self, facet: &str, expected_bytes: u64) {
        self.current_facet = facet.to_string();
        self.current_facet_bytes = 0;
        self.current_facet_total = expected_bytes;
        self.current_merkle_bytes = 0;
        self.current_merkle_total = 0;
        self.phase = Phase::Copy;
        self.facet_index += 1;
        // Force first render so users see the facet flip immediately.
        self.last_render = std::time::Instant::now() - std::time::Duration::from_secs(1);
        self.render();
    }

    fn tick_copy(&mut self, bytes_so_far: u64) {
        self.current_facet_bytes = bytes_so_far;
        if self.last_render.elapsed().as_millis() >= 250 {
            self.render();
            self.last_render = std::time::Instant::now();
        }
    }

    fn begin_merkle(&mut self, total_bytes: u64) {
        self.phase = Phase::Merkle;
        self.current_merkle_total = total_bytes;
        self.current_merkle_bytes = 0;
        // Force an immediate render so the phase transition is visible.
        self.last_render = std::time::Instant::now() - std::time::Duration::from_secs(1);
        self.render();
    }

    fn tick_merkle(&mut self, bytes_hashed: u64) {
        self.current_merkle_bytes = bytes_hashed;
        if self.last_render.elapsed().as_millis() >= 250 {
            self.render();
            self.last_render = std::time::Instant::now();
        }
    }

    fn end_facet(&mut self, facet: &str, expected_bytes: u64) {
        // Clear the live line and print a permanent ✓ row.
        eprintln!("\r  [{}/{}] {} \u{2713} {}\u{1b}[K",
            self.facet_index, self.facet_count, facet,
            super::precache::fmt_bytes(expected_bytes));
        self.bytes_done_in_prior_facets =
            self.bytes_done_in_prior_facets.saturating_add(expected_bytes);
        self.current_facet.clear();
        self.current_facet_bytes = 0;
        self.current_facet_total = 0;
        self.current_merkle_bytes = 0;
        self.current_merkle_total = 0;
    }

    fn fail(&self, facet: &str, msg: &str) {
        eprintln!("\rFacet '{facet}': {msg}\u{1b}[K");
    }

    fn render(&self) {
        use std::io::Write;
        let aggregate_done = self.bytes_done_in_prior_facets
            .saturating_add(self.current_facet_bytes);
        let pct_total = super::precache::pct(aggregate_done, self.total_bytes);
        let facet_state = match self.phase {
            Phase::Copy => {
                if self.current_facet_total == 0 {
                    "scanning…".to_string()
                } else {
                    format!(
                        "copy {}% ({}/{})",
                        super::precache::pct(self.current_facet_bytes, self.current_facet_total),
                        super::precache::fmt_bytes(self.current_facet_bytes),
                        super::precache::fmt_bytes(self.current_facet_total),
                    )
                }
            }
            Phase::Merkle => {
                if self.current_merkle_total == 0 {
                    "merkle …".to_string()
                } else {
                    format!(
                        "merkle {}% ({}/{})",
                        super::precache::pct(self.current_merkle_bytes, self.current_merkle_total),
                        super::precache::fmt_bytes(self.current_merkle_bytes),
                        super::precache::fmt_bytes(self.current_merkle_total),
                    )
                }
            }
        };
        eprint!(
            "\r  [{}/{}] {}: {} \u{2022} total {}% ({}/{})\u{1b}[K",
            self.facet_index, self.facet_count, self.current_facet,
            facet_state,
            pct_total,
            super::precache::fmt_bytes(aggregate_done),
            super::precache::fmt_bytes(self.total_bytes));
        let _ = std::io::stderr().flush();
    }

    fn summary(&self) {
        let elapsed = self.started.elapsed().as_secs_f64();
        let done = self.bytes_done_in_prior_facets;
        eprintln!("Derive done: {} facet(s), {} in {:.1}s ({}/s).",
            self.facet_count,
            super::precache::fmt_bytes(done),
            elapsed,
            super::precache::fmt_bytes((done as f64 / elapsed.max(0.001)) as u64));
    }
}

// ─── Materialization ────────────────────────────────────────────

fn materialize_facet<F: FnMut(u64)>(
    facet_name: &str,
    src: &Path,
    dest: &Path,
    kind: FacetKind,
    window: &DSWindow,
    on_bytes_written: F,
) -> io::Result<()> {
    match kind {
        FacetKind::VariableVvec => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "variable-length vvec facets ('{facet_name}') are not yet \
                 materializable by `derive`; use `transform extract` \
                 to slice the source file manually for now"
            ),
        )),
        FacetKind::Scalar(elem) => materialize_scalar(src, dest, elem, window, on_bytes_written),
        FacetKind::UniformXvec(elem) => {
            materialize_uniform_xvec(src, dest, elem, window, on_bytes_written)
        }
        FacetKind::Slab => materialize_slab(src, dest, window, on_bytes_written),
    }
}

/// Materialize a `.slab` facet.
///
/// An empty window byte-copies the source — slab files are
/// self-describing (their trailing pages/namespaces page already
/// references the right offsets), so a verbatim copy preserves all
/// namespaces, including any `:schema` sidecar emitted at import.
/// A windowed slab is sliced ordinal-by-ordinal through slabtastic;
/// the resulting file holds only the content namespace's selected
/// range, and any sibling namespaces (e.g. `:schema`) are carried
/// forward verbatim with their original entries.
fn materialize_slab<F: FnMut(u64)>(
    src: &Path,
    dest: &Path,
    window: &DSWindow,
    mut cb: F,
) -> io::Result<()> {
    if window.is_empty() {
        let mut src_f = fs::File::open(src)?;
        let len = src_f.metadata()?.len();
        let mut dst_f = fs::File::create(dest)?;
        copy_with_callback(&mut src_f, &mut dst_f, len, &mut cb)?;
        return Ok(());
    }

    // Open the source via slabtastic so we can address records by
    // ordinal in each window interval. Use a fresh writer on the
    // destination and stream the windowed records through it.
    let reader = slabtastic::SlabReader::open(src).map_err(|e| {
        io::Error::other(format!("open slab {}: {e}", src.display()))
    })?;
    let config = slabtastic::WriterConfig::default();
    let mut writer = slabtastic::SlabWriter::new(dest, config).map_err(|e| {
        io::Error::other(format!("create slab {}: {e}", dest.display()))
    })?;
    for iv in &window.0 {
        for ord in iv.min_incl..iv.max_excl {
            let data = reader.get(ord as i64).map_err(|e| {
                io::Error::other(format!(
                    "read ordinal {ord} from {}: {e}",
                    src.display()
                ))
            })?;
            writer.add_record(&data).map_err(|e| {
                io::Error::other(format!(
                    "write ordinal {ord} to {}: {e}",
                    dest.display()
                ))
            })?;
            cb(data.len() as u64);
        }
    }

    // Carry sibling namespaces (e.g. the metadata `layout` schema) forward
    // verbatim. Windowing applies only to the default content namespace;
    // sibling namespaces are metadata (not per-row), so they are copied
    // whole rather than sliced. Without this, a windowed derive would drop
    // the embedded layout copy.
    let namespaces = slabtastic::SlabReader::list_namespaces(src).map_err(|e| {
        io::Error::other(format!("list namespaces of {}: {e}", src.display()))
    })?;
    for ns in &namespaces {
        if ns.name.is_empty() {
            continue; // default namespace — already written (windowed) above
        }
        let ns_reader =
            slabtastic::SlabReader::open_namespace(src, Some(&ns.name)).map_err(|e| {
                io::Error::other(format!(
                    "open namespace '{}' of {}: {e}",
                    ns.name,
                    src.display()
                ))
            })?;
        writer.start_namespace(&ns.name).map_err(|e| {
            io::Error::other(format!(
                "start namespace '{}' in {}: {e}",
                ns.name,
                dest.display()
            ))
        })?;
        let total = ns_reader.total_records() as i64;
        for ord in 0..total {
            let data = ns_reader.get(ord).map_err(|e| {
                io::Error::other(format!(
                    "read namespace '{}' ordinal {ord} from {}: {e}",
                    ns.name,
                    src.display()
                ))
            })?;
            writer.add_record(&data).map_err(|e| {
                io::Error::other(format!(
                    "write namespace '{}' ordinal {ord} to {}: {e}",
                    ns.name,
                    dest.display()
                ))
            })?;
            cb(data.len() as u64);
        }
    }

    writer
        .finish()
        .map_err(|e| io::Error::other(format!("finalize slab {}: {e}", dest.display())))?;
    Ok(())
}

/// Scalar packed files have no per-record header; a "record" is one
/// element of `byte_width(element_type)` bytes. The window
/// intervals address records (= bytes for u8/i8, two-byte words
/// for u16/i16/f16, etc.).
fn materialize_scalar<F: FnMut(u64)>(
    src: &Path,
    dest: &Path,
    elem: ElementType,
    window: &DSWindow,
    mut cb: F,
) -> io::Result<()> {
    let mut src_f = fs::File::open(src)?;
    let src_len = src_f.metadata()?.len();
    if window.is_empty() {
        let mut dst_f = fs::File::create(dest)?;
        copy_with_callback(&mut src_f, &mut dst_f, src_len, &mut cb)?;
        return Ok(());
    }
    let record_size = elem.byte_width() as u64;
    if record_size == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            "scalar element type has zero byte width"));
    }
    let mut dst_f = fs::File::create(dest)?;
    let mut buf = vec![0u8; 1024 * 1024];
    for iv in &window.0 {
        let mut from = iv.min_incl * record_size;
        let to = iv.max_excl * record_size;
        if to > src_len {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof,
                format!("window [{}..{}) past EOF ({} bytes)",
                    iv.min_incl, iv.max_excl, src_len)));
        }
        src_f.seek(SeekFrom::Start(from))?;
        while from < to {
            let want = ((to - from).min(buf.len() as u64)) as usize;
            src_f.read_exact(&mut buf[..want])?;
            dst_f.write_all(&buf[..want])?;
            from += want as u64;
            cb(want as u64);
        }
    }
    Ok(())
}

/// Stream from src to dst, ticking `cb` after each block so the
/// caller can render a live byte meter. `total` is informational —
/// not enforced, just provided so a one-block whole-file copy
/// fires cb exactly once.
fn copy_with_callback<F: FnMut(u64)>(
    src: &mut fs::File, dst: &mut fs::File, total: u64, cb: &mut F,
) -> io::Result<()> {
    let mut buf = vec![0u8; 1024 * 1024];
    let mut remaining = total;
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        src.read_exact(&mut buf[..want])?;
        dst.write_all(&buf[..want])?;
        remaining -= want as u64;
        cb(want as u64);
    }
    Ok(())
}

/// Uniform xvec layout: each record is `<i32 dim><dim*byte_width
/// bytes>`. We assume all records have the same `dim` (which is the
/// xvec format's *uniform* contract — variable-length data uses
/// the `vvec` extensions). Read `dim` from the first record's
/// header, compute the stride, and copy whole records.
fn materialize_uniform_xvec<F: FnMut(u64)>(
    src: &Path,
    dest: &Path,
    elem: ElementType,
    window: &DSWindow,
    mut cb: F,
) -> io::Result<()> {
    let mut src_f = fs::File::open(src)?;
    let src_len = src_f.metadata()?.len();

    if src_len < 4 {
        let mut dst_f = fs::File::create(dest)?;
        copy_with_callback(&mut src_f, &mut dst_f, src_len, &mut cb)?;
        return Ok(());
    }
    let mut dim_bytes = [0u8; 4];
    src_f.read_exact(&mut dim_bytes)?;
    let dim = i32::from_le_bytes(dim_bytes) as u64;
    if dim == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            "first xvec record header reports dim=0"));
    }
    let record_size = 4 + dim * elem.byte_width() as u64;

    let mut dst_f = fs::File::create(dest)?;
    if window.is_empty() {
        // Whole-file copy — replay dim_bytes (already read) and
        // stream the rest. Tick cb after each block.
        dst_f.write_all(&dim_bytes)?;
        cb(4);
        copy_with_callback(&mut src_f, &mut dst_f, src_len.saturating_sub(4), &mut cb)?;
        return Ok(());
    }

    let mut buf = vec![0u8; (record_size as usize).max(1024 * 1024)];
    for iv in &window.0 {
        let from_bytes = iv.min_incl * record_size;
        let to_bytes = iv.max_excl * record_size;
        if to_bytes > src_len {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof,
                format!("window [{}..{}) of {}-byte records past EOF ({} bytes)",
                    iv.min_incl, iv.max_excl, record_size, src_len)));
        }
        src_f.seek(SeekFrom::Start(from_bytes))?;
        let mut remaining = to_bytes - from_bytes;
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            src_f.read_exact(&mut buf[..want])?;
            dst_f.write_all(&buf[..want])?;
            remaining -= want as u64;
            cb(want as u64);
        }
    }
    Ok(())
}

/// Build a `.mref` next to `path` so the derived dataset can be
/// fetched merkle-verified by other clients. Streams the file
/// chunk-by-chunk so 10+ GiB facets don't have to be loaded into
/// memory in one shot, and so the caller can drive a progress
/// meter via `on_progress`. The source's `.mref` doesn't apply
/// because the content differs after windowing.
fn generate_mref<F: FnMut(u64)>(path: &Path, on_progress: F) -> io::Result<()> {
    let mref = MerkleRef::from_path_with_progress(path, MERKLE_CHUNK_SIZE, on_progress)?;
    let mut mref_path = path.as_os_str().to_owned();
    mref_path.push(".mref");
    mref.save(Path::new(&mref_path))
}

// ─── dataset.yaml emission ──────────────────────────────────────

fn write_dataset_yaml(
    output: &Path,
    derived_name: &str,
    source_spec: &str,
    source_profile: &str,
    facets: &[(String, String)],
    src_profile: &crate::dataset::profile::DSProfile,
) -> io::Result<()> {
    // Keep the format hand-written rather than going through
    // serde — the derived YAML is intentionally minimal and
    // human-readable. Three top-level keys: `name`, `attributes`,
    // `profiles`.
    let mut out = String::new();
    out.push_str("# Generated by `vectordata datasets derive`.\n");
    out.push_str(&format!("name: {derived_name}\n\n"));
    out.push_str("attributes:\n");
    out.push_str(&format!("  derived_from: {source_spec}:{source_profile}\n"));
    out.push_str(&format!("  derived_at: {}\n",
        httpdate::fmt_http_date(std::time::SystemTime::now())));
    out.push('\n');
    out.push_str("profiles:\n");
    out.push_str("  default:\n");
    if let Some(maxk) = src_profile.maxk {
        out.push_str(&format!("    maxk: {maxk}\n"));
    }
    if let Some(bc) = src_profile.base_count {
        out.push_str(&format!("    base_count: {bc}\n"));
    }
    for (facet_name, filename) in facets {
        out.push_str(&format!("    {facet_name}: {filename}\n"));
    }
    fs::write(output.join("dataset.yaml"), out)
}

// ─── Spec resolution (shared shape with precache.rs) ───────────

/// Resolve a head spec (no profile suffix) to (resolution, derived
/// default name). The derived-default name is used by derive as the
/// fallback name of the new dataset directory when `--name` is not
/// passed.
fn resolve_spec(
    head: &str,
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
) -> Option<(Resolved, String)> {
    if head.starts_with("http://") || head.starts_with("https://") {
        let name = head.rsplit('/').find(|s| !s.is_empty())
            .unwrap_or("derived").to_string();
        return Some((Resolved::Url(head.to_string()), name));
    }
    let as_path = Path::new(head);
    if as_path.exists() {
        let name = as_path.file_stem().and_then(|s| s.to_str())
            .unwrap_or("derived").to_string();
        return Some((Resolved::Local(head.to_string()), name));
    }
    let sources = build_sources(configdir, extra_catalogs, at);
    if sources.is_empty() {
        eprintln!("'{head}' is not a local path, not a URL, and no catalog is configured.");
        eprintln!("Add a catalog with: vectordata config catalog add <URL-or-path>");
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
    if entry.dataset_type == "knn_entries.yaml" {
        // derive needs the per-dataset `dataset.yaml` to extract
        // window metadata (see [`load_rich_config`]). knn_entries-
        // shape catalogs don't publish one — the catalog's embedded
        // layout *is* the dataset description but lacks the rich
        // window info derive depends on. Fail early with a clear
        // diagnostic instead of erroring deep inside the open path.
        eprintln!("error: derive does not support knn_entries.yaml-shape catalogs ({head})");
        eprintln!("       (those catalogs have no per-dataset dataset.yaml with window metadata)");
        return None;
    }
    let name = entry.name.clone();
    let yaml_url = entry.path.clone();
    Some((
        Resolved::CatalogEntry { catalog, name: name.clone(), yaml_url },
        name,
    ))
}

enum Resolved {
    /// Catalog-resolved canonical entry. `yaml_url` is the absolute
    /// URL or path of the per-dataset `dataset.yaml` (needed by
    /// `load_rich_config` for window metadata).
    CatalogEntry { catalog: Catalog, name: String, yaml_url: String },
    Local(String),
    Url(String),
}

/// Fetch the dataset.yaml as raw text and parse it as the rich
/// [`crate::dataset::config::DatasetConfig`] (the one that exposes
/// per-view windows). `group_path` is either an HTTPS URL or a
/// local filesystem path — the same shape `TestDataGroup::load`
/// accepts.
fn load_rich_config(group_path: &str) -> Result<RichDatasetConfig, String> {
    let yaml = if group_path.starts_with("http://") || group_path.starts_with("https://") {
        fetch_yaml_url(group_path).map_err(|e| e.to_string())?
    } else {
        let p = Path::new(group_path);
        let yaml_path = if p.is_dir() {
            p.join("dataset.yaml")
        } else if p.extension().is_some_and(|e| e == "yaml" || e == "yml") {
            p.to_path_buf()
        } else {
            p.join("dataset.yaml")
        };
        fs::read_to_string(&yaml_path).map_err(|e| format!("{}: {e}", yaml_path.display()))?
    };
    let mut config: RichDatasetConfig = serde_yaml::from_str(&yaml)
        .map_err(|e| format!("parsing dataset.yaml: {e}"))?;
    // Mirror DatasetConfig::load's strata-vs-profiles.raw_sized
    // unification so downstream window resolution works the same
    // way it does on a `load()`ed config.
    if config.strata.is_empty() && !config.profiles.raw_sized.is_empty() {
        config.strata = config.profiles.raw_sized.clone();
    } else if !config.strata.is_empty() && config.profiles.raw_sized.is_empty() {
        config.profiles.raw_sized = config.strata.clone();
        config.profiles.deferred_sized = config.strata.clone();
    }
    Ok(config)
}

fn fetch_yaml_url(url: &str) -> io::Result<String> {
    use crate::transport::shared_client_for;
    let mut u = url::Url::parse(url)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
    if !u.path().ends_with(".yaml") && !u.path().ends_with(".yml") {
        if !u.path().ends_with('/') {
            u.set_path(&(u.path().to_owned() + "/"));
        }
        u = u.join("dataset.yaml")
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
    }
    let resp = shared_client_for(u.as_str()).get(u).send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| io::Error::other(e.to_string()))?;
    resp.text().map_err(|e| io::Error::other(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::source::{DSInterval, DSWindow};

    /// Build a slab with `n` records (each `b"r-{i}"`) plus a single
    /// `:schema` namespace record so we can verify the sidecar
    /// survives the byte-copy path.
    fn write_test_slab(path: &Path, n: u64) {
        let cfg = slabtastic::WriterConfig::default();
        let mut w = slabtastic::SlabWriter::new(path, cfg).unwrap();
        for i in 0..n {
            w.add_record(format!("r-{i}").as_bytes()).unwrap();
        }
        w.start_namespace("schema").unwrap();
        w.add_record(b"{\"v\":1}").unwrap();
        w.finish().unwrap();
    }

    /// Empty-window derive of a slab is a byte-copy that preserves
    /// every namespace (including `:schema`).
    #[test]
    fn materialize_slab_empty_window_preserves_all_namespaces() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("m.slab");
        let dst = tmp.path().join("m-derived.slab");
        write_test_slab(&src, 4);

        let mut written = 0u64;
        materialize_slab(&src, &dst, &DSWindow(vec![]), |d| written += d).unwrap();

        // Default namespace: 4 content records.
        let r = slabtastic::SlabReader::open(&dst).unwrap();
        for i in 0..4 {
            assert_eq!(r.get(i).unwrap(), format!("r-{i}").as_bytes());
        }
        // Schema namespace: still there, single record verbatim.
        let s = slabtastic::SlabReader::open_namespace(&dst, Some("schema")).unwrap();
        assert_eq!(s.get(0).unwrap(), b"{\"v\":1}");
        assert!(written > 0, "byte copy should have ticked progress");
    }

    /// Windowed derive of a slab keeps only the selected ordinals in the
    /// content namespace, **and** carries sibling namespaces (e.g. the
    /// `:schema`/`layout` sidecar) forward verbatim — they are metadata, not
    /// per-row, so they are copied whole rather than sliced.
    #[test]
    fn materialize_slab_with_window_slices_content_and_keeps_namespaces() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("m.slab");
        let dst = tmp.path().join("m-derived.slab");
        write_test_slab(&src, 10); // also writes a `schema` namespace record

        let window = DSWindow(vec![DSInterval { min_incl: 2, max_excl: 5 }]);
        let mut written = 0u64;
        materialize_slab(&src, &dst, &window, |d| written += d).unwrap();

        // Content namespace: exactly the windowed range.
        let r = slabtastic::SlabReader::open(&dst).unwrap();
        assert_eq!(r.get(0).unwrap(), b"r-2");
        assert_eq!(r.get(1).unwrap(), b"r-3");
        assert_eq!(r.get(2).unwrap(), b"r-4");
        assert!(r.get(3).is_err(), "window should produce exactly 3 records");
        assert!(written > 0);

        // Sibling namespace: carried across verbatim (the prior limitation).
        let s = slabtastic::SlabReader::open_namespace(&dst, Some("schema")).unwrap();
        assert_eq!(s.get(0).unwrap(), b"{\"v\":1}");
    }

    /// `classify_facet` recognizes `.slab` without demanding an
    /// ElementType mapping — the bug that prompted this whole change.
    #[test]
    fn classify_facet_accepts_slab() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("x.slab");
        std::fs::write(&p, b"placeholder").unwrap();
        let kind = classify_facet("metadata_content", &p, "slab").unwrap();
        assert!(matches!(kind, FacetKind::Slab));
    }

    /// End-to-end: build a tiny source dataset on disk, derive it,
    /// and verify the output lands at `profiles/base/<facet>.<ext>`
    /// with companion `.mref` files, the dataset.yaml references
    /// those same paths, and the derived dataset reloads cleanly via
    /// `TestDataGroup::load`.
    #[test]
    fn derive_local_emits_profiles_base_layout() {
        use crate::TestDataGroup;

        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();

        // Write a 3-vector fvec at the canonical source location.
        let src_base = src.path().join("profiles/base");
        std::fs::create_dir_all(&src_base).unwrap();
        let base_fvec = src_base.join("base_vectors.fvec");
        let mut buf = Vec::new();
        for i in 0..3i32 {
            buf.extend_from_slice(&2i32.to_le_bytes()); // dim header
            buf.extend_from_slice(&(i as f32).to_le_bytes());
            buf.extend_from_slice(&((i + 1) as f32).to_le_bytes());
        }
        std::fs::write(&base_fvec, &buf).unwrap();

        // Minimal dataset.yaml with one profile referencing the
        // canonical source path.
        std::fs::write(
            src.path().join("dataset.yaml"),
            "name: src\nprofiles:\n  default:\n    base_vectors: profiles/base/base_vectors.fvec\n",
        )
        .unwrap();

        // Drive a local derive on it.
        let yaml_path = src.path().join("dataset.yaml");
        let rc = derive_local(&yaml_path, "default", dst.path(), Some("derived"));
        assert_eq!(rc, 0, "derive_local should succeed");

        // The output must use the profiles/base layout, not flat.
        let derived_base = dst.path().join("profiles/base/base_vectors.fvec");
        assert!(
            derived_base.is_file(),
            "expected derived facet at {} but found nothing",
            derived_base.display(),
        );
        let derived_mref = dst.path().join("profiles/base/base_vectors.fvec.mref");
        assert!(derived_mref.is_file(), "expected companion .mref");
        // No flat fallback at the root.
        assert!(
            !dst.path().join("base_vectors.fvec").exists(),
            "derive should not also write a flat copy"
        );

        // Reload the derived dataset and confirm the profile points
        // at the canonical path.
        let group = TestDataGroup::load(dst.path().to_str().unwrap())
            .expect("derived dataset must reload");
        let yaml = std::fs::read_to_string(dst.path().join("dataset.yaml")).unwrap();
        assert!(
            yaml.contains("profiles/base/base_vectors.fvec"),
            "dataset.yaml should reference the profiles/base path: {yaml}"
        );
        assert!(group.profile("default").is_some());
    }
}
