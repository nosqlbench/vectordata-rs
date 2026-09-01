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
use crate::dataset::Sharding;
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
    /// The source's bytes, as one stream. A single file is one span; a
    /// series is one per shard, in ordinal order (SH-38).
    src: SourceSpans,
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
fn plan_output_size(src: &SourceSpans, kind: FacetKind, window: &DSWindow) -> io::Result<u64> {
    // For slabs and variable-length vvecs we don't have a cheap
    // record-size formula; an empty window still byte-copies, but
    // anything else is best counted at materialize time.
    if window.is_empty() {
        return Ok(src.len());
    }
    let record_size = match kind {
        FacetKind::Scalar(elem) => elem.byte_width() as u64,
        FacetKind::UniformXvec(elem) => {
            if src.len() < 4 { return Ok(0); }
            let mut f = src.open()?;
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
/// Derive a dataset.
///
/// `sharding` decides how each facet is laid out:
/// [`Sharding::Whole`] writes one file per facet,
/// [`Sharding::Stride`] rolls over every `n` records, and
/// [`Sharding::MaxBytes`] picks whatever stride keeps one shard under
/// a size cap. A run that fits in one shard collapses to the
/// single-file form in every case (SH-35, SH-83).
pub fn run(
    dataset: &str,
    profile: &str,
    output: &Path,
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
    name_override: Option<&str>,
    force: bool,
    sharding: Sharding,
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
        return derive_local(&yaml_path, profile, output, name_override, sharding);
    }

    // Otherwise: catalog / URL → runtime access layer.
    derive_via_access_layer(
        dataset, profile, output, configdir, extra_catalogs, at, name_override, sharding)
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
    sharding: Sharding,
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
        /* local_fast_path = */ true, sharding)
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
    sharding: Sharding,
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

    // Where a series' shard paths resolve from. Relative sources in a
    // `dataset.yaml` are relative to the file that declares them, and
    // that is the only anchor a shard has (SH-78).
    let yaml_base = std::path::Path::new(&yaml_url)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let plan = match build_plan_via_view(&*view, ds_profile, output, &yaml_base) {
        Ok(p) => p,
        Err(e) => { eprintln!("{e}"); return 1; }
    };
    run_plan(&plan, output, &derived_default_name, profile_name,
        dataset, ds_profile, name_override,
        /* local_fast_path = */ false, sharding)
}

// ─── Planning ─────────────────────────────────────────────────────

fn build_plan_local(
    base_dir: &Path,
    ds_profile: &crate::dataset::profile::DSProfile,
    output: &Path,
) -> Result<Vec<PlanRow>, String> {
    let mut rows = Vec::new();
    for (facet_name, dview) in ds_profile.views() {
        let spans = spans_for_view(facet_name, dview, base_dir)?;
        rows.push(plan_row_for(facet_name, spans, dview.effective_window().clone(), output)?);
    }
    Ok(rows)
}

/// The bytes a declared view presents, whether it names one file or a
/// series (SH-38).
///
/// A series is realized through the **same** code the loader runs, so
/// derive cannot disagree with the reader about which files a facet is
/// or what order they are in (SH-90). Deriving from a series is a copy
/// across that ordinal space; the output's own stride is decided
/// independently, which is what makes re-striding possible at all.
fn spans_for_view(
    facet_name: &str,
    dview: &crate::dataset::profile::DSView,
    base_dir: &Path,
) -> Result<SourceSpans, String> {
    if !dview.is_series() {
        let src_path = base_dir.join(dview.path());
        if !src_path.is_file() {
            return Err(format!(
                "Facet '{facet_name}': source {} not found.",
                src_path.display()
            ));
        }
        return SourceSpans::single(src_path)
            .map_err(|e| format!("Facet '{facet_name}': {e}"));
    }

    let sources = dview.declaration_sources();
    let probe = |s: &crate::dataset::source::DSSource| -> Result<u64, String> {
        let path = base_dir.join(&s.path);
        let resolved = path.to_str().ok_or_else(|| "non-UTF-8 path".to_string())?;
        let storage = crate::storage::Storage::open(resolved).map_err(|e| e.to_string())?;
        crate::view::records_in(resolved, &storage)
            .ok_or_else(|| format!("cannot count records in {}", path.display()))
    };
    let shards =
        crate::dataset::shards::realize(facet_name, &dview.declaration(&sources), &probe)
            .map_err(|e| format!("Facet '{facet_name}': {e}"))?;

    // The record size the spans are measured in. Reading it from the
    // first shard is reading the facet's: every shard shares one
    // format and one dimension.
    let first = shards
        .entries()
        .first()
        .ok_or_else(|| format!("Facet '{facet_name}': series declares no shards"))?;
    let first_path = base_dir.join(&first.source.path);
    let ext = first_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_string();
    let kind = classify_facet(facet_name, &first_path, &ext)?;
    let single = SourceSpans::single(first_path.clone())
        .map_err(|e| format!("Facet '{facet_name}': {e}"))?;
    match fixed_record_size(&single, kind)
        .map_err(|e| format!("Facet '{facet_name}': {e}"))?
    {
        // A fixed stride turns an ordinal extent into a byte extent, so
        // a sliced shard contributes exactly the bytes it addresses.
        Some(record_size) => SourceSpans::from_shards(&shards, base_dir, record_size),
        // Without one, where a record starts is only knowable from the
        // records. A vvec is a self-describing stream, so walking it
        // answers that — which is the same walk `materialize_vvec` does
        // for a window, and refusing a sliced shard here while doing it
        // there would be a claim this file disproves.
        None if matches!(kind, FacetKind::VariableVvec) => {
            SourceSpans::from_variable_shards(&shards, base_dir, elem_width(&ext))
        }
        // A slab's records are indexed by its own pages, not by a
        // position in the byte stream, so a sliced shard cannot be
        // reduced to a byte extent here. The composer walks slab shards
        // record by record instead (`materialize_slab_series`), and it
        // applies the *series* window rather than a per-shard one — so
        // a sliced entry would be silently ignored, which is worse than
        // refused.
        None => SourceSpans::from_whole_shards(&shards, base_dir),
    }
    .map_err(|e| format!("Facet '{facet_name}': {e}"))
}

fn build_plan_via_view(
    view: &dyn crate::TestDataView,
    ds_profile: &crate::dataset::profile::DSProfile,
    output: &Path,
    base_dir: &Path,
) -> Result<Vec<PlanRow>, String> {
    let mut rows = Vec::new();
    for (facet_name, dview) in ds_profile.views() {
        // Skip non-data facets. Asked as "does the spec name this
        // format?" rather than "what element width does it have?" —
        // the second dropped slab facets here while the local plan
        // builder kept them, so one dataset derived two ways.
        if !view.facet_holds_data(facet_name) {
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
        } else if dview.is_series() {
            // A series has no single cache path. Its shards resolve
            // through the declaration against the dataset's own
            // directory, which is what the reader does too (SH-78).
            let spans = spans_for_view(facet_name, dview, base_dir)?;
            rows.push(plan_row_for(
                facet_name,
                spans,
                dview.effective_window().clone(),
                output,
            )?);
            continue;
        } else {
            return Err(format!("Facet '{facet_name}': cannot resolve source path."));
        };
        let spans = SourceSpans::single(src_path)
            .map_err(|e| format!("Facet '{facet_name}': {e}"))?;
        rows.push(plan_row_for(facet_name, spans, dview.effective_window().clone(), output)?);
    }
    Ok(rows)
}

fn plan_row_for(
    facet_name: &str,
    src: SourceSpans,
    window: DSWindow,
    output: &Path,
) -> Result<PlanRow, String> {
    // Every span of a series shares one format, so the first names the
    // facet's extension — and for a single file it is the only one.
    let src_path = src
        .first_path()
        .ok_or_else(|| format!("Facet '{facet_name}': source names no file"))?
        .to_path_buf();
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
    let expected_bytes = plan_output_size(&src, kind, &window)
        .map_err(|e| format!("Facet '{facet_name}': cannot plan output size: {e}"))?;
    Ok(PlanRow {
        facet: facet_name.to_string(),
        src,
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
    sharding: Sharding,
) -> i32 {
    let total_bytes: u64 = plan.iter().map(|r| r.expected_bytes).sum();
    eprintln!("Materializing {} facet(s), {} to write.",
        plan.len(), super::precache::fmt_bytes(total_bytes));

    let mut meter = DeriveMeter::new(plan.len(), total_bytes);
    let mut derived_facets: Vec<DerivedFacet> = Vec::new();

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
            row.kind, &row.window, sharding,
            |delta| {
                written = written.saturating_add(delta);
                meter.tick_copy(written);
            });
        let produced = match res {
            Ok(p) => p,
            Err(e) => {
                meter.fail(&row.facet, &e.to_string());
                return 1;
            }
        };

        // Merkle generation. The .mref is computed from the
        // (windowed) output bytes — the donor's mref doesn't
        // apply once the content changes. Stream the file so 10+
        // GiB facets don't allocate a giant Vec, and tick the meter
        // by hashed bytes so users see real progress instead of a
        // frozen "computing merkle…" line.
        // One `.mref` per **file**, never per facet (SH-20): each shard
        // is independently verifiable and independently re-fetchable,
        // which is most of the point of splitting them.
        let merkle_total: u64 = produced
            .files
            .iter()
            .filter_map(|f| std::fs::metadata(f).ok())
            .map(|m| m.len())
            .sum();
        meter.begin_merkle(merkle_total);
        for f in &produced.files {
            if let Err(e) = generate_mref(f, |hashed| meter.tick_merkle(hashed)) {
                meter.fail(&row.facet, &format!("mref: {e}"));
                return 1;
            }
        }
        meter.end_facet(&row.facet, row.expected_bytes);

        // The declaration names what was written. `dest_filename`
        // carries the profile directory; the materializer reports only
        // the file's own name, so rejoin them.
        let dir = std::path::Path::new(&row.dest_filename)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .filter(|p| !p.is_empty());
        let source = match dir {
            Some(d) => format!("{d}/{}", produced.source_spec),
            None => produced.source_spec.clone(),
        };
        derived_facets.push(DerivedFacet {
            facet: row.facet.clone(),
            source,
            shard_stride: produced.shard_stride,
            shard_count: produced.shard_count,
            record_count: produced.record_count,
        });
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

/// Bytes per record for a fixed-stride facet, or `None` when the format
/// has no fixed stride.
///
/// Only fixed-stride formats can be sharded by the writer, because
/// rolling over at a record boundary means knowing where one is.
fn fixed_record_size(src: &SourceSpans, kind: FacetKind) -> io::Result<Option<u64>> {
    Ok(match kind {
        FacetKind::Scalar(elem) => Some(elem.byte_width() as u64),
        FacetKind::UniformXvec(elem) => {
            if src.len() < 4 {
                return Ok(None);
            }
            // The dim header of the first record. Every shard of a
            // series shares one dimension, so reading the first is
            // reading the facet's.
            let mut f = src.open()?;
            let mut dim_bytes = [0u8; 4];
            f.read_exact(&mut dim_bytes)?;
            let dim = i32::from_le_bytes(dim_bytes) as u64;
            (dim > 0).then_some(4 + dim * elem.byte_width() as u64)
        }
        FacetKind::VariableVvec | FacetKind::Slab => None,
    })
}

/// Materialize a facet as a **series**, rolling over every `stride`
/// records (SH-35).
///
/// Records are copied whole, so a record never spans a shard boundary
/// (SH-13) and the output is byte-identical for the same input and
/// stride (SH-36). A run that fits in one shard collapses to the
/// single-file form (SH-83) — the writer decides that, not the caller.
fn materialize_sharded<F: FnMut(u64)>(
    src: &SourceSpans,
    dir: &Path,
    basename: &str,
    ext: &str,
    record_size: u64,
    window: &DSWindow,
    stride: u64,
    mut cb: F,
) -> io::Result<crate::datasets::shard_writer::ShardOutcome> {
    use crate::datasets::shard_writer::ShardWriter;

    // Re-striding is a copy (SH-38): the source's shard boundaries and
    // the output's have nothing to do with each other, so the read side
    // presents one ordinal space and the writer rolls over at its own
    // stride.
    let mut src_f = src.open()?;
    let src_len = src.len();
    if record_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cannot shard a facet whose record size is zero",
        ));
    }
    let total_records = src_len / record_size;

    // An empty window is every record; otherwise the window's
    // intervals, in order.
    let intervals: Vec<(u64, u64)> = if window.is_empty() {
        vec![(0, total_records)]
    } else {
        window.0.iter().map(|iv| (iv.min_incl, iv.max_excl)).collect()
    };

    let mut writer = ShardWriter::new(dir, basename, ext, stride)?;
    let mut buf = vec![0u8; record_size as usize];
    for (lo, hi) in intervals {
        if hi * record_size > src_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "window [{lo}..{hi}) of {record_size}-byte records past EOF \
                     ({src_len} bytes)"
                ),
            ));
        }
        src_f.seek(SeekFrom::Start(lo * record_size))?;
        for _ in lo..hi {
            src_f.read_exact(&mut buf)?;
            writer.write_record(&buf)?;
            cb(record_size);
        }
    }
    writer.finish()
}

/// The bytes a facet's source presents, as one stream (SH-38).
///
/// A facet's source is one file or a series of them, and the copy that
/// derives a new dataset should not care which. Every span is a byte
/// range of a real file, in ordinal order; reading across them in
/// sequence is exactly the facet's own ordinal space, because that is
/// what the shard model already says a series is.
///
/// **Only sound where records have a fixed stride.** For scalar and
/// uniform-xvec facets a series is the concatenation of its shards'
/// bytes, so a byte-level read across spans is a record-level read
/// across shards. A vvec or slab shard carries its own index or page
/// structure, so concatenating two of them produces neither format —
/// those ask for [`Self::single_path`] and refuse a series by name.
#[derive(Debug, Clone)]
pub(crate) struct SourceSpans {
    spans: Vec<Span>,
    total: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct Span {
    pub(crate) path: std::path::PathBuf,
    /// First byte of the file this span presents.
    offset: u64,
    len: u64,
}

impl SourceSpans {
    /// One whole file — every facet written before series existed.
    pub(crate) fn single(path: std::path::PathBuf) -> io::Result<Self> {
        let len = fs::metadata(&path)?.len();
        Ok(Self {
            total: len,
            spans: vec![Span { path, offset: 0, len }],
        })
    }

    /// A series, from the shard model the loader realized.
    ///
    /// Byte extents come from the entries' ordinal extents and the
    /// record size, so a sliced entry contributes only the bytes it
    /// addresses — the same rule residency uses (SH-92).
    pub(crate) fn from_shards(
        shards: &crate::dataset::shards::Shards,
        base_dir: &Path,
        record_size: u64,
    ) -> io::Result<Self> {
        if record_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot span a facet whose record size is zero",
            ));
        }
        let mut spans = Vec::with_capacity(shards.entries().len());
        let mut total = 0;
        for entry in shards.entries() {
            let path = base_dir.join(&entry.source.path);
            let file_len = fs::metadata(&path)?.len();
            let offset = entry.file_base * record_size;
            let len = entry.len * record_size;
            if offset + len > file_len {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "shard {} addresses bytes [{offset}..{}) of a {file_len}-byte file",
                        path.display(),
                        offset + len
                    ),
                ));
            }
            total += len;
            spans.push(Span { path, offset, len });
        }
        Ok(Self { spans, total })
    }

    /// A series of variable-length shards, honouring each entry's
    /// window.
    ///
    /// Each file is walked to find where its records start, so an entry
    /// reading part of a file contributes exactly the bytes its
    /// ordinals cover. The walk is local — derive has already brought
    /// its source into the cache — and is the same one a windowed
    /// single-file copy performs.
    pub(crate) fn from_variable_shards(
        shards: &crate::dataset::shards::Shards,
        base_dir: &Path,
        elem: usize,
    ) -> io::Result<Self> {
        if elem == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot span a variable-length facet of unknown element width",
            ));
        }
        let mut spans = Vec::with_capacity(shards.entries().len());
        let mut total = 0;
        for entry in shards.entries() {
            let path = base_dir.join(&entry.source.path);
            let whole = Self::single(path.clone())?;
            // A whole-file entry needs no walk: the span is the file.
            if entry.file_base == 0 && entry.source.window.is_empty() {
                total += whole.len();
                spans.push(Span { path, offset: 0, len: whole.len() });
                continue;
            }
            let offsets = vvec_record_offsets(&whole, elem)?;
            let records = offsets.len().saturating_sub(1) as u64;
            let (lo, hi) = (entry.file_base, entry.file_base + entry.len);
            if hi > records {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "shard '{}' reads records [{lo}..{hi}) of a {records}-record file",
                        entry.source.path
                    ),
                ));
            }
            let (offset, end) = (offsets[lo as usize], offsets[hi as usize]);
            total += end - offset;
            spans.push(Span { path, offset, len: end - offset });
        }
        Ok(Self { spans, total })
    }

    /// A series of **whole** files, for a format with no fixed stride.
    ///
    /// A vvec's records are self-describing and a slab's are indexed by
    /// its own pages; neither offers a byte offset for an ordinal
    /// without reading. Whole files sidestep the question — the span is
    /// the file — and that is what the explicit form declares, since it
    /// composes files rather than splitting them (SH-50).
    ///
    /// A sliced entry is refused by name. Its ordinal window is
    /// meaningful, but resolving it needs the index this format keeps
    /// somewhere other than in the byte stream.
    pub(crate) fn from_whole_shards(
        shards: &crate::dataset::shards::Shards,
        base_dir: &Path,
    ) -> io::Result<Self> {
        let mut spans = Vec::with_capacity(shards.entries().len());
        let mut total = 0;
        for entry in shards.entries() {
            if entry.file_base != 0 || !entry.source.window.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "shard '{}' reads part of its file, and this format has no \
                         fixed record size to resolve that against — a variable-length \
                         or container facet composes whole files (SH-50)",
                        entry.source.path
                    ),
                ));
            }
            let path = base_dir.join(&entry.source.path);
            let len = fs::metadata(&path)?.len();
            total += len;
            spans.push(Span { path, offset: 0, len });
        }
        Ok(Self { spans, total })
    }

    /// Total bytes across every span.
    pub(crate) fn len(&self) -> u64 {
        self.total
    }

    /// The one file behind these spans, when there is one.
    ///
    /// `None` for a series — the answer a format that cannot be read
    /// across files needs, so it refuses rather than reading the first
    /// shard as the whole facet (SH-74).
    pub(crate) fn single_path(&self) -> Option<&Path> {
        match self.spans.as_slice() {
            [only] if only.offset == 0 => Some(&only.path),
            _ => None,
        }
    }

    /// The files these spans read from, in ordinal order.
    ///
    /// For a format that cannot be read as one byte stream — a slab,
    /// whose shards each carry their own page index — composition has
    /// to happen file by file, and this is what it walks.
    pub(crate) fn shards(&self) -> &[Span] {
        &self.spans
    }

    /// The first file these spans read from.
    ///
    /// Sound for format questions only — extension, dimension, element
    /// type — which every shard of a series shares. Anything about
    /// *content* must go through [`Self::open`].
    pub(crate) fn first_path(&self) -> Option<&Path> {
        self.spans.first().map(|s| s.path.as_path())
    }

    /// A cursor over the concatenation.
    pub(crate) fn open(&self) -> io::Result<SpanReader<'_>> {
        Ok(SpanReader {
            spans: &self.spans,
            pos: 0,
            open: None,
        })
    }
}

/// A `Read + Seek` view of [`SourceSpans`] as one contiguous stream.
///
/// Implementing the standard traits rather than a bespoke interface is
/// what lets the materializers stay as they were: they seek to
/// `record * stride` and read whole records, and whether that lands in
/// one file or walks three is not their concern.
pub(crate) struct SpanReader<'a> {
    spans: &'a [Span],
    pos: u64,
    /// The span currently open, and its file.
    open: Option<(usize, fs::File)>,
}

impl SpanReader<'_> {
    /// Which span holds stream position `pos`, and how far into it.
    fn locate(&self, pos: u64) -> Option<(usize, u64)> {
        let mut base = 0u64;
        for (i, span) in self.spans.iter().enumerate() {
            if pos < base + span.len {
                return Some((i, pos - base));
            }
            base += span.len;
        }
        None
    }
}

impl io::Read for SpanReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let Some((index, within)) = self.locate(self.pos) else {
            // Past the last span: end of stream, exactly as a file
            // read past EOF answers.
            return Ok(0);
        };
        let span = &self.spans[index];
        // Reopen only when the span changes. A sequential pass over a
        // series therefore opens each file once, in order.
        let reopen = !matches!(self.open, Some((i, _)) if i == index);
        if reopen {
            let file = fs::File::open(&span.path)?;
            self.open = Some((index, file));
        }
        let (_, file) = self.open.as_mut().expect("span file");
        file.seek(SeekFrom::Start(span.offset + within))?;
        // Never read past this span's extent: the next bytes of the
        // file may belong to another facet's window, or to nothing.
        let want = buf.len().min((span.len - within) as usize);
        let got = file.read(&mut buf[..want])?;
        self.pos += got as u64;
        Ok(got)
    }
}

impl io::Seek for SpanReader<'_> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let total: u64 = self.spans.iter().map(|s| s.len).sum();
        let target = match from {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::End(n) => total as i128 + n as i128,
            SeekFrom::Current(n) => self.pos as i128 + n as i128,
        };
        if target < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek before the start of the stream",
            ));
        }
        self.pos = target as u64;
        Ok(self.pos)
    }
}

/// Compose a slab series into one slab (SH-18, SH-38).
///
/// Each shard is an ordinary slab based at zero, and the global base
/// comes from the shard map (SH-96) — so the series' ordinal `o` is
/// shard `s`'s local ordinal `o - base(s)`. Records are read through
/// slabtastic and written through a fresh writer, which is the only
/// composition a page-structured container admits.
///
/// Sibling namespaces are **not** carried across. An embedded `layout`
/// namespace does not travel into a sharded content slab (SH-98): the
/// standalone `metadata_layout.slab` is authoritative and the embedded
/// copy is a convenience, so choosing one shard's copy to promote would
/// invent a rule about where a schema lives that the unsharded case
/// never needed.
fn materialize_slab_series<F: FnMut(u64)>(
    src: &SourceSpans,
    dest: &Path,
    window: &DSWindow,
    mut cb: F,
) -> io::Result<()> {
    let config = slabtastic::WriterConfig::default();
    let mut writer = slabtastic::SlabWriter::new(dest, config)
        .map_err(|e| io::Error::other(format!("create slab {}: {e}", dest.display())))?;

    // Ordinal extents of each shard within the series, so a window in
    // the series' space resolves to a range in a shard's own.
    let mut base = 0u64;
    for shard in src.shards() {
        let reader = slabtastic::SlabReader::open(&shard.path).map_err(|e| {
            io::Error::other(format!("open slab {}: {e}", shard.path.display()))
        })?;
        let count = reader.total_records();
        let end = base + count;
        // What of this shard the window asks for, in local ordinals.
        let wanted: Vec<(u64, u64)> = if window.is_empty() {
            vec![(0, count)]
        } else {
            window
                .0
                .iter()
                .filter_map(|iv| {
                    let lo = iv.min_incl.max(base);
                    let hi = iv.max_excl.min(end);
                    (lo < hi).then(|| (lo - base, hi - base))
                })
                .collect()
        };
        for (lo, hi) in wanted {
            for ord in lo..hi {
                let data = reader.get(ord as i64).map_err(|e| {
                    io::Error::other(format!(
                        "read ordinal {ord} from {}: {e}",
                        shard.path.display()
                    ))
                })?;
                writer.add_record(&data).map_err(|e| {
                    io::Error::other(format!("write to {}: {e}", dest.display()))
                })?;
                cb(data.len() as u64);
            }
        }
        base = end;
    }
    writer
        .finish()
        .map_err(|e| io::Error::other(format!("finish slab {}: {e}", dest.display())))?;
    Ok(())
}

/// Element width for a variable-length extension, or zero.
fn elem_width(ext: &str) -> usize {
    crate::io::infer_elem_size(ext)
}

/// Copy a variable-length facet, whole or windowed.
///
/// **A vvec is a self-describing stream**: each record is an `i32`
/// dimension followed by that many elements, and the offset index is a
/// sidecar rather than part of the file. Concatenating two vvec shards
/// therefore produces a valid vvec, which is what lets a series be read
/// as one stream here exactly as a fixed-stride facet is (SH-38).
///
/// A window has to walk to find its bounds, because only the records
/// themselves say where they start. The walk is over the spans, so a
/// windowed series is windowed in the *series'* ordinal space and not
/// in any one shard's.
fn materialize_vvec<F: FnMut(u64)>(
    src: &SourceSpans,
    dest: &Path,
    elem: usize,
    window: &DSWindow,
    mut cb: F,
) -> io::Result<()> {
    let mut src_f = src.open()?;
    let mut dst_f = fs::File::create(dest)?;
    if window.is_empty() {
        copy_with_callback(&mut src_f, &mut dst_f, src.len(), &mut cb)?;
        return Ok(());
    }

    let offsets = vvec_record_offsets(src, elem)?;
    // `offsets` has one entry per record plus a final end sentinel, so
    // a record's extent is always a pair of neighbours.
    let records = offsets.len().saturating_sub(1);
    let mut buf = vec![0u8; 1024 * 1024];
    for iv in &window.0 {
        let (lo, hi) = (iv.min_incl as usize, iv.max_excl as usize);
        if hi > records {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("window [{lo}..{hi}) past the end of a {records}-record facet"),
            ));
        }
        let (mut from, to) = (offsets[lo], offsets[hi]);
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

/// Byte offsets of every record in a variable-length stream, plus a
/// final sentinel at the end.
///
/// Built by walking the records, which is the only thing that knows
/// where they are. The published `IDXFOR__` sidecar answers the same
/// question for a single file, but a series has one per shard and none
/// for the concatenation — and rebuilding is a local read here, since
/// derive has already brought its source into the cache.
fn vvec_record_offsets(src: &SourceSpans, elem: usize) -> io::Result<Vec<u64>> {
    let mut f = src.open()?;
    let total = src.len();
    let mut offsets = Vec::new();
    let mut at = 0u64;
    let mut header = [0u8; 4];
    while at + 4 <= total {
        offsets.push(at);
        f.seek(SeekFrom::Start(at))?;
        f.read_exact(&mut header)?;
        let dim = i32::from_le_bytes(header);
        if dim < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("record at byte {at} declares a negative dimension ({dim})"),
            ));
        }
        let len = 4 + dim as u64 * elem as u64;
        if at + len > total {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("record at byte {at} claims {len} bytes, past the end at {total}"),
            ));
        }
        at += len;
    }
    if at != total {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} trailing bytes after the last whole record", total - at),
        ));
    }
    offsets.push(total);
    Ok(offsets)
}

/// A facet as written, for the emitted declaration.
#[derive(Debug, Clone)]
pub(crate) struct DerivedFacet {
    pub facet: String,
    pub source: String,
    pub shard_stride: Option<u64>,
    pub shard_count: Option<u32>,
    pub record_count: Option<u64>,
}

/// What materializing one facet produced.
///
/// Carries the declaration the output needs, so the emitted
/// `dataset.yaml` describes the files that were actually written rather
/// than the ones the plan expected (SH-37).
#[derive(Debug, Clone)]
pub(crate) struct MaterializedFacet {
    /// Files written, in ordinal order.
    pub files: Vec<std::path::PathBuf>,
    /// The `source:` value for the declaration — a filename, or the
    /// `NNNN` pattern for a series.
    pub source_spec: String,
    /// Set only for a series (SH-83 collapses a one-shard run).
    pub shard_stride: Option<u64>,
    pub shard_count: Option<u32>,
    pub record_count: Option<u64>,
}

fn materialize_facet<F: FnMut(u64)>(
    facet_name: &str,
    src: &SourceSpans,
    dest: &Path,
    kind: FacetKind,
    window: &DSWindow,
    sharding: Sharding,
    on_bytes_written: F,
) -> io::Result<MaterializedFacet> {
    // Sharding here is available only where records have a fixed
    // stride: this materializer copies bytes, so rolling over at a
    // record boundary means knowing where one is without decoding. A
    // format without one is refused rather than silently written as a
    // single file the caller did not ask for.
    if sharding.is_requested() {
        match fixed_record_size(src, kind)? {
            Some(record_size) => {
                let Some(stride) = sharding.stride_for_fixed(record_size) else {
                    // Only reachable from a cap: an explicit stride is
                    // used verbatim. A cap that cannot hold ten
                    // records of this facet is a misconfiguration, not
                    // a layout, and saying so beats emitting a series
                    // with a file per handful of records.
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "facet '{facet_name}': a shard cap of {} bytes cannot hold ten \
                             {record_size}-byte records; raise the cap",
                            sharding.max_bytes().unwrap_or(0),
                        ),
                    ));
                };
                let dir = dest.parent().unwrap_or(Path::new("."));
                let file = dest
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                let (basename, ext) = file.rsplit_once('.').unwrap_or((file, ""));
                let out = materialize_sharded(
                    src,
                    dir,
                    basename,
                    ext,
                    record_size,
                    window,
                    stride,
                    on_bytes_written,
                )?;
                return Ok(MaterializedFacet {
                    source_spec: out.source_spec(),
                    shard_stride: out.is_series().then_some(out.stride),
                    shard_count: out.is_series().then(|| out.shard_count()),
                    record_count: out.is_series().then_some(out.records),
                    files: out.files,
                });
            }
            None => {
                // A slab or a vvec carries per-record extents, so a
                // byte-range copy cannot find a record boundary. The
                // *pipeline* writes these as series through a
                // record-oriented sink; this materializer does not,
                // and refuses rather than pretending.
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!(
                        "facet '{facet_name}' has no fixed record stride, so this \
                         derivation cannot shard it; write it whole, or produce it \
                         through the pipeline, which shards record-oriented formats"
                    ),
                ));
            }
        }
    }
    let single = |files: Vec<std::path::PathBuf>| MaterializedFacet {
        source_spec: dest
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string(),
        files,
        shard_stride: None,
        shard_count: None,
        record_count: None,
    };
    let done = |r: io::Result<()>| r.map(|()| single(vec![dest.to_path_buf()]));
    match kind {
        FacetKind::VariableVvec => {
            let ext = dest
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default();
            let elem = crate::io::infer_elem_size(ext);
            if elem == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("facet '{facet_name}': unknown element width for '.{ext}'"),
                ));
            }
            done(materialize_vvec(src, dest, elem, window, on_bytes_written))
        }
        FacetKind::Scalar(elem) => {
            done(materialize_scalar(src, dest, elem, window, on_bytes_written))
        }
        FacetKind::UniformXvec(elem) => {
            done(materialize_uniform_xvec(src, dest, elem, window, on_bytes_written))
        }
        FacetKind::Slab => done(materialize_slab(src, dest, window, on_bytes_written)),
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
    src: &SourceSpans,
    dest: &Path,
    window: &DSWindow,
    mut cb: F,
) -> io::Result<()> {
    // A slab carries its own page index, so two shards concatenated are
    // neither slab nor readable — a byte copy is not available here.
    // Re-striding one is a slab-format operation: read records by
    // ordinal from each shard in turn and stream them through one
    // writer, which is what the windowed path below already does for a
    // single file (SH-18, SH-38).
    let Some(src) = src.single_path() else {
        return materialize_slab_series(src, dest, window, cb);
    };
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
    src: &SourceSpans,
    dest: &Path,
    elem: ElementType,
    window: &DSWindow,
    mut cb: F,
) -> io::Result<()> {
    let mut src_f = src.open()?;
    let src_len = src.len();
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
fn copy_with_callback<F: FnMut(u64), R: Read>(
    src: &mut R, dst: &mut fs::File, total: u64, cb: &mut F,
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
    src: &SourceSpans,
    dest: &Path,
    elem: ElementType,
    window: &DSWindow,
    mut cb: F,
) -> io::Result<()> {
    let mut src_f = src.open()?;
    let src_len = src.len();

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
    facets: &[DerivedFacet],
    src_profile: &crate::dataset::profile::DSProfile,
) -> io::Result<()> {
    // Keep the format hand-written rather than going through
    // serde — the derived YAML is intentionally minimal and
    // human-readable. Three top-level keys: `name`, `attributes`,
    // `profiles`.
    let mut out = String::new();
    out.push_str("# Generated by `vectordata datasets derive`.\n");
    out.push_str(&format!("name: {derived_name}\n"));
    // The *lowest* version that describes what was written, not the
    // highest this build knows (V-5) — folded from what each facet
    // needs rather than restated here, so it cannot drift from the rule
    // (V-19). An unsharded output emits nothing and stays readable by
    // every build that ever existed.
    let required = facets
        .iter()
        .map(|f| {
            if f.shard_count.is_some() {
                crate::model::FORMAT_VERSION_SHARDED
            } else {
                crate::model::FORMAT_VERSION_BASE
            }
        })
        .max()
        .unwrap_or(crate::model::FORMAT_VERSION_BASE);
    if required > crate::model::FORMAT_VERSION_BASE {
        out.push_str(&format!("format_version: {required}\n"));
    }
    out.push('\n');
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
    for f in facets {
        // A single file keeps the plain one-line spelling — the shape
        // every dataset written before sharding is in, and the one a
        // reader predating it can open (SH-4, SH-83). Only a genuine
        // series takes the mapping form.
        match (f.shard_stride, f.shard_count, f.record_count) {
            (Some(stride), Some(count), Some(records)) => {
                out.push_str(&format!("    {}:\n", f.facet));
                out.push_str(&format!("      source: {}\n", f.source));
                out.push_str(&format!("      shard_stride: {stride}\n"));
                out.push_str(&format!("      shard_count: {count}\n"));
                out.push_str(&format!("      record_count: {records}\n"));
            }
            _ => out.push_str(&format!("    {}: {}\n", f.facet, f.source)),
        }
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
    // Same strata-vs-profiles.raw_sized unification `DatasetConfig::load`
    // applies, so downstream window resolution works the same way it
    // does on a `load()`ed config.
    config.unify_strata();
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
    /// A single file as the one-span stream every materializer now
    /// takes. The tests below are about formats, not about series, so
    /// this keeps them reading exactly as they did.
    fn spans_of(path: &std::path::Path) -> super::SourceSpans {
        super::SourceSpans::single(path.to_path_buf()).expect("source exists")
    }

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

    /// Two slab shards compose into one slab, in ordinal order
    /// (SH-18, SH-38).
    ///
    /// A slab carries its own page index, so the shards cannot be
    /// concatenated as bytes. They are read record by record and
    /// written through one writer — the only composition a
    /// page-structured container admits.
    #[test]
    fn a_slab_series_composes_into_one_slab() {
        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("part_a.slab");
        let b = tmp.path().join("part_b.slab");
        let dst = tmp.path().join("joined.slab");
        write_test_slab_range(&a, 0, 3);
        write_test_slab_range(&b, 3, 3);

        let spans = SourceSpans::from_whole_shards(
            &whole_file_shards(&["part_a.slab", "part_b.slab"]),
            tmp.path(),
        )
        .unwrap();

        let mut written = 0u64;
        materialize_slab(&spans, &dst, &DSWindow(vec![]), |d| written += d).unwrap();

        let r = slabtastic::SlabReader::open(&dst).unwrap();
        assert_eq!(r.total_records(), 6, "both shards' records");
        for i in 0..6i64 {
            assert_eq!(
                r.get(i).unwrap(),
                format!("r-{i}").as_bytes(),
                "ordinal {i} keeps its place in the series"
            );
        }
        assert!(written > 0);
    }

    /// A window over a slab series is in the **series'** ordinal space:
    /// it clips each shard to the part of the window that falls in it,
    /// rather than applying the same range to every shard.
    #[test]
    fn a_windowed_slab_series_clips_the_series_not_each_shard() {
        let tmp = tempfile::tempdir().unwrap();
        write_test_slab_range(&tmp.path().join("part_a.slab"), 0, 4);
        write_test_slab_range(&tmp.path().join("part_b.slab"), 4, 4);
        let dst = tmp.path().join("windowed.slab");

        let spans = SourceSpans::from_whole_shards(
            &whole_file_shards(&["part_a.slab", "part_b.slab"]),
            tmp.path(),
        )
        .unwrap();

        // [2..6) spans the seam: the last two of shard 0 and the first
        // two of shard 1.
        let window = DSWindow(vec![crate::dataset::source::DSInterval {
            min_incl: 2,
            max_excl: 6,
        }]);
        materialize_slab(&spans, &dst, &window, |_| {}).unwrap();

        let r = slabtastic::SlabReader::open(&dst).unwrap();
        assert_eq!(r.total_records(), 4);
        for (out, src) in (0i64..4).zip(2i64..6) {
            assert_eq!(r.get(out).unwrap(), format!("r-{src}").as_bytes());
        }
    }

    fn write_test_slab_range(path: &Path, first: u64, n: u64) {
        let cfg = slabtastic::WriterConfig::default();
        let mut w = slabtastic::SlabWriter::new(path, cfg).unwrap();
        for i in first..first + n {
            w.add_record(format!("r-{i}").as_bytes()).unwrap();
        }
        w.finish().unwrap();
    }

    /// Whole-file shards for the named files, in order.
    fn whole_file_shards(names: &[&str]) -> crate::dataset::shards::Shards {
        crate::dataset::shards::Shards::new(
            names
                .iter()
                .map(|n| crate::dataset::shards::Entry {
                    source: crate::dataset::source::parse_source_string(n).unwrap(),
                    file_base: 0,
                    // Lengths are not consulted by the whole-file path;
                    // the file is the span.
                    len: 1,
                })
                .collect(),
        )
        .unwrap()
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
        materialize_slab(&spans_of(&src), &dst, &DSWindow(vec![]), |d| written += d).unwrap();

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
        materialize_slab(&spans_of(&src), &dst, &window, |d| written += d).unwrap();

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
        let rc = derive_local(
            &yaml_path,
            "default",
            dst.path(),
            Some("derived"),
            Sharding::Whole,
        );
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

#[cfg(test)]
mod sharded_output {
    use super::*;

    /// A single file as the one-span stream the materializers take.
    fn spans_of(path: &std::path::Path) -> SourceSpans {
        SourceSpans::single(path.to_path_buf()).expect("source exists")
    }

    fn tmpdir() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
        fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    /// A uniform fvec of `n` records, dim 2 → 12 bytes each.
    fn write_fvec(path: &Path, n: usize) {
        let mut buf = Vec::new();
        for i in 0..n {
            buf.extend(&2i32.to_le_bytes());
            buf.extend(&(i as f32).to_le_bytes());
            buf.extend(&((i as f32) + 0.5).to_le_bytes());
        }
        fs::write(path, buf).unwrap();
    }

    /// Record size comes from the format, not from a guess.
    #[test]
    fn the_record_size_comes_from_the_format() {
        let d = tmpdir();
        let src = d.path().join("in.fvec");
        write_fvec(&src, 3);
        assert_eq!(
            fixed_record_size(&spans_of(&src), FacetKind::UniformXvec(ElementType::F32)).unwrap(),
            Some(12)
        );
        let s = d.path().join("in.u32");
        fs::write(&s, [0u8; 16]).unwrap();
        assert_eq!(
            fixed_record_size(&spans_of(&s), FacetKind::Scalar(ElementType::U32)).unwrap(),
            Some(4)
        );
        // Variable-length and slab have no fixed stride to roll over on.
        assert_eq!(
            fixed_record_size(&spans_of(&src), FacetKind::VariableVvec).unwrap(),
            None
        );
        assert_eq!(fixed_record_size(&spans_of(&src), FacetKind::Slab).unwrap(), None);
    }

    /// **The shards concatenate back to the source.** Splitting is a
    /// layout change, not a content change (SH-48).
    #[test]
    fn a_sharded_copy_concatenates_back_to_the_source() {
        let d = tmpdir();
        let src = d.path().join("in.fvec");
        write_fvec(&src, 10);
        let out = materialize_sharded(
            &spans_of(&src),
            d.path(),
            "base_vectors",
            "fvec",
            12,
            &DSWindow::default(),
            4,
            |_| {},
        )
        .unwrap();

        assert_eq!(out.records, 10);
        assert_eq!(out.shard_count(), 3);
        assert_eq!(out.source_spec(), "base_vectors__NNNN.fvec");
        let joined: Vec<u8> = out
            .files
            .iter()
            .flat_map(|f| fs::read(f).unwrap())
            .collect();
        assert_eq!(joined, fs::read(&src).unwrap(), "content must be identical");
    }

    /// A window selects records before sharding, and the shards hold
    /// exactly those.
    #[test]
    fn a_window_selects_records_before_they_are_sharded() {
        let d = tmpdir();
        let src = d.path().join("in.fvec");
        write_fvec(&src, 20);
        let window = crate::dataset::source::parse_window("4..14").unwrap();
        let out =
            materialize_sharded(&spans_of(&src), d.path(), "b", "fvec", 12, &window, 4, |_| {}).unwrap();

        assert_eq!(out.records, 10);
        let joined: Vec<u8> = out
            .files
            .iter()
            .flat_map(|f| fs::read(f).unwrap())
            .collect();
        let all = fs::read(&src).unwrap();
        assert_eq!(joined, all[4 * 12..14 * 12], "exactly the windowed records");
    }

    /// **Output that fits in one shard collapses** — so a derive run
    /// that happened to fit does not emit a declaration older readers
    /// cannot open (SH-83).
    #[test]
    fn an_output_that_fits_one_shard_is_written_as_a_single_file() {
        let d = tmpdir();
        let src = d.path().join("in.fvec");
        write_fvec(&src, 5);
        let out = materialize_sharded(
            &spans_of(&src),
            d.path(),
            "base_vectors",
            "fvec",
            12,
            &DSWindow::default(),
            1000,
            |_| {},
        )
        .unwrap();
        assert!(out.collapsed);
        assert_eq!(out.source_spec(), "base_vectors.fvec");
        assert!(d.path().join("base_vectors.fvec").exists());
    }

    /// A window past the end is refused rather than silently truncated.
    #[test]
    fn a_window_past_the_end_is_refused() {
        let d = tmpdir();
        let src = d.path().join("in.fvec");
        write_fvec(&src, 5);
        let window = crate::dataset::source::parse_window("0..99").unwrap();
        assert!(
            materialize_sharded(&spans_of(&src), d.path(), "b", "fvec", 12, &window, 4, |_| {}).is_err()
        );
    }
}

#[cfg(test)]
mod span_reader_tests {
    use super::*;
    use crate::dataset::shards::{Entry, Shards};
    use std::io::{Read, Seek, SeekFrom};

    /// A file of `n` bytes whose byte `i` is `i as u8` — so any read
    /// answers *where in the file* it came from, not merely how much.
    fn ramp(path: &Path, n: usize) {
        let bytes: Vec<u8> = (0..n).map(|i| i as u8).collect();
        fs::write(path, bytes).unwrap();
    }

    /// Spans over `parts` whole files, in order.
    fn series(dir: &Path, parts: &[(&str, usize)]) -> SourceSpans {
        for (name, n) in parts {
            ramp(&dir.join(name), *n);
        }
        let shards = Shards::new(
            parts
                .iter()
                .map(|(name, n)| Entry {
                    source: crate::dataset::source::parse_source_string(name).unwrap(),
                    file_base: 0,
                    len: *n as u64,
                })
                .collect(),
        )
        .unwrap();
        SourceSpans::from_shards(&shards, dir, 1).unwrap()
    }

    /// **The concatenation is the stream.** Reading a series to the end
    /// yields every shard's bytes, in shard order, once.
    #[test]
    fn a_series_reads_as_one_stream() {
        let tmp = tempfile::tempdir().unwrap();
        let spans = series(tmp.path(), &[("a", 10), ("b", 5), ("c", 7)]);
        assert_eq!(spans.len(), 22);

        let mut all = Vec::new();
        spans.open().unwrap().read_to_end(&mut all).unwrap();

        let expect: Vec<u8> = (0..10u8).chain(0..5).chain(0..7).collect();
        assert_eq!(all, expect);
    }

    /// A read that would cross a shard seam stops at it and returns a
    /// short read — the contract `Read` allows, and what keeps a read
    /// from splicing two files in one syscall. `read_exact` loops over
    /// that seam, which is what the materializers rely on.
    #[test]
    fn a_read_stops_at_a_seam_and_read_exact_crosses_it() {
        let tmp = tempfile::tempdir().unwrap();
        let spans = series(tmp.path(), &[("a", 10), ("b", 10)]);
        let mut r = spans.open().unwrap();

        r.seek(SeekFrom::Start(8)).unwrap();
        let mut buf = [0u8; 8];
        let got = r.read(&mut buf).unwrap();
        assert_eq!(got, 2, "the read stops at the seam");
        assert_eq!(&buf[..2], &[8, 9]);

        // read_exact spans it, and the bytes come from both files.
        r.seek(SeekFrom::Start(8)).unwrap();
        let mut exact = [0u8; 6];
        r.read_exact(&mut exact).unwrap();
        assert_eq!(exact, [8, 9, 0, 1, 2, 3]);
    }

    /// A sliced shard contributes only the bytes it addresses. The
    /// stream must never read past a span's extent into whatever the
    /// file holds next (SH-92).
    #[test]
    fn a_sliced_shard_contributes_only_its_window() {
        let tmp = tempfile::tempdir().unwrap();
        ramp(&tmp.path().join("a"), 100);
        ramp(&tmp.path().join("b"), 100);
        let shards = Shards::new(vec![
            Entry {
                source: crate::dataset::source::parse_source_string("a").unwrap(),
                file_base: 20,
                len: 5,
            },
            Entry {
                source: crate::dataset::source::parse_source_string("b").unwrap(),
                file_base: 90,
                len: 4,
            },
        ])
        .unwrap();
        let spans = SourceSpans::from_shards(&shards, tmp.path(), 1).unwrap();
        assert_eq!(spans.len(), 9);

        let mut all = Vec::new();
        spans.open().unwrap().read_to_end(&mut all).unwrap();
        assert_eq!(all, [20, 21, 22, 23, 24, 90, 91, 92, 93]);
    }

    /// Seeks address the stream, not a file: from the start, from the
    /// end, and relative — each landing in whichever shard holds that
    /// stream position.
    #[test]
    fn seeks_address_the_stream_not_a_file() {
        let tmp = tempfile::tempdir().unwrap();
        let spans = series(tmp.path(), &[("a", 10), ("b", 10), ("c", 10)]);
        let mut r = spans.open().unwrap();
        let mut one = [0u8; 1];

        // Into the third shard from the start.
        assert_eq!(r.seek(SeekFrom::Start(25)).unwrap(), 25);
        r.read_exact(&mut one).unwrap();
        assert_eq!(one[0], 5);

        // Backwards into the first, relatively.
        assert_eq!(r.seek(SeekFrom::Current(-24)).unwrap(), 2);
        r.read_exact(&mut one).unwrap();
        assert_eq!(one[0], 2);

        // The last byte, from the end.
        assert_eq!(r.seek(SeekFrom::End(-1)).unwrap(), 29);
        r.read_exact(&mut one).unwrap();
        assert_eq!(one[0], 9);
    }

    /// Past the end is end-of-stream, exactly as a file read past EOF
    /// answers — not an error, and not a wrap to another shard.
    #[test]
    fn reading_past_the_end_is_end_of_stream() {
        let tmp = tempfile::tempdir().unwrap();
        let spans = series(tmp.path(), &[("a", 4), ("b", 4)]);
        let mut r = spans.open().unwrap();

        assert_eq!(r.seek(SeekFrom::Start(8)).unwrap(), 8);
        let mut buf = [0u8; 4];
        assert_eq!(r.read(&mut buf).unwrap(), 0);
        assert_eq!(r.seek(SeekFrom::Start(1000)).unwrap(), 1000);
        assert_eq!(r.read(&mut buf).unwrap(), 0);
    }

    /// Seeking before the start is refused rather than wrapping to a
    /// huge unsigned position.
    #[test]
    fn seeking_before_the_start_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let spans = series(tmp.path(), &[("a", 4)]);
        let mut r = spans.open().unwrap();
        assert_eq!(
            r.seek(SeekFrom::Start(2)).unwrap(),
            2,
            "a legal seek first, so the error below is about the sign"
        );
        assert!(r.seek(SeekFrom::Current(-3)).is_err());
        assert!(r.seek(SeekFrom::End(-5)).is_err());
        // The refused seek left the position alone.
        let mut one = [0u8; 1];
        r.read_exact(&mut one).unwrap();
        assert_eq!(one[0], 2);
    }

    /// A zero-length read is zero bytes, not an attempt to open a file
    /// or a claim of end-of-stream at a valid position.
    #[test]
    fn an_empty_buffer_reads_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let spans = series(tmp.path(), &[("a", 4)]);
        let mut r = spans.open().unwrap();
        assert_eq!(r.read(&mut []).unwrap(), 0);
        // And the position is untouched.
        let mut one = [0u8; 1];
        r.read_exact(&mut one).unwrap();
        assert_eq!(one[0], 0);
    }

    /// `single_path` is the question "is this one whole file?" — the
    /// answer a format that cannot be read across files needs (SH-74).
    /// A series and a sliced single file both answer no.
    #[test]
    fn single_path_answers_only_for_one_whole_file() {
        let tmp = tempfile::tempdir().unwrap();
        ramp(&tmp.path().join("a"), 100);

        let whole = SourceSpans::single(tmp.path().join("a")).unwrap();
        assert_eq!(whole.single_path(), Some(tmp.path().join("a").as_path()));

        let two = series(tmp.path(), &[("a", 10), ("b", 10)]);
        assert_eq!(two.single_path(), None, "a series is not one file");
        assert_eq!(two.shards().len(), 2);
        assert_eq!(two.first_path(), Some(tmp.path().join("a").as_path()));

        ramp(&tmp.path().join("c"), 100);
        let sliced = Shards::new(vec![Entry {
            source: crate::dataset::source::parse_source_string("c").unwrap(),
            file_base: 10,
            len: 5,
        }])
        .unwrap();
        let sliced = SourceSpans::from_shards(&sliced, tmp.path(), 1).unwrap();
        assert_eq!(
            sliced.single_path(),
            None,
            "a file read from an offset is not the whole file"
        );
    }

    /// A shard that addresses bytes the file does not have is refused
    /// when the spans are built, not discovered as a short read halfway
    /// through a derive.
    #[test]
    fn a_shard_that_overruns_its_file_is_refused_up_front() {
        let tmp = tempfile::tempdir().unwrap();
        ramp(&tmp.path().join("a"), 40);
        let shards = Shards::new(vec![Entry {
            source: crate::dataset::source::parse_source_string("a").unwrap(),
            file_base: 0,
            len: 10,
        }])
        .unwrap();

        // Ten 4-byte records fit exactly.
        assert_eq!(SourceSpans::from_shards(&shards, tmp.path(), 4).unwrap().len(), 40);
        // Ten 8-byte records do not.
        let err = SourceSpans::from_shards(&shards, tmp.path(), 8).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
        // And a record size of zero is not a stride at all.
        let err = SourceSpans::from_shards(&shards, tmp.path(), 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}
