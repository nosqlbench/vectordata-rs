// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Import command — converts source data into the preferred internal format
//! for each dataset facet type.
//!
//! Unlike `convert`, which performs arbitrary format-to-format conversion,
//! `import` automatically selects the output format based on the facet type:
//!
//! - `base_vectors`, `query_vectors`, `neighbor_distances` -> fvec
//! - `neighbor_indices`, `filtered_neighbor_indices` -> ivec
//! - `metadata_content` -> slab (MNode encoding)
//! - `metadata_predicates` -> slab (PNode encoding)
//! - `predicate_results`, `metadata_layout` -> slab

pub mod args;
pub mod dataset_ext;
pub mod facet;
pub mod profile;
pub mod source;

pub use args::ImportArgs;
pub use facet::Facet;

use std::path::Path;

use log::info;

use crate::formats::VecFormat;
use crate::formats::reader;
use crate::formats::writer::{self, SinkConfig};

use vectordata::dataset::DatasetConfig;
use dataset_ext::DatasetConfigExt;

/// Entry point for the import subcommand
pub fn run(args: ImportArgs) {
    let ui = crate::ui::auto_ui_handle();

    // Handle scaffold generation (no source needed)
    if let Some(name) = &args.scaffold {
        let config = DatasetConfig::scaffold(name);
        let yaml = serde_yaml::to_string(&config).expect("failed to serialize scaffold");
        let out_path = args
            .output
            .clone()
            .unwrap_or_else(|| std::path::PathBuf::from("dataset.yaml"));
        std::fs::write(&out_path, &yaml).unwrap_or_else(|e| {
            ui.emitln(format!("Failed to write {}: {}", out_path.display(), e));
            std::process::exit(1);
        });
        ui.log(&format!("Scaffold written to {}", out_path.display()));
        return;
    }

    // Dataset mode: upstream sources come from the YAML, not the CLI
    if let Some(dataset_path) = &args.dataset {
        run_dataset_import(dataset_path, &args, &ui);
        return;
    }

    // Single facet mode: source is required
    let source = args.source.as_ref().unwrap_or_else(|| {
        ui.emitln(format!("Source path is required for single-facet import. Use --dataset for multi-facet mode."));
        std::process::exit(1);
    });

    let facet = args.facet.unwrap_or_else(|| {
        ui.emitln(format!("--facet is required for single-facet import. Use --dataset for multi-facet mode."));
        std::process::exit(1);
    });

    import_single_facet(source, facet, args.output.as_deref(), args.from.as_deref(), &args, &ui);
}

/// Import views from a dataset.yaml using the default profile.
///
/// Iterates the default profile's views and imports each source file
/// as the corresponding facet. Datasets with pipeline-based upstream
/// (`upstream.steps`) should use `veks run` instead.
fn run_dataset_import(dataset_path: &Path, args: &ImportArgs, ui: &crate::ui::UiHandle) {
    let config = DatasetConfig::load(dataset_path).unwrap_or_else(|e| {
        ui.emitln(format!("{}", e));
        std::process::exit(1);
    });

    let base_dir = dataset_path.parent().unwrap_or(Path::new("."));

    // Phase 1: Fail-fast validation
    ui.log(&format!("Validating dataset configuration..."));
    let errors = config.validate(base_dir);
    if !errors.is_empty() {
        for err in &errors {
            ui.emitln(format!("  ERROR: {}", err));
        }
        ui.emitln(format!("Validation failed with {} error(s)", errors.len()));
        std::process::exit(1);
    }

    let default_profile = match config.default_profile() {
        Some(p) => p,
        None => {
            ui.log(&format!("No default profile defined — nothing to import."));
            ui.log(&format!("Use 'veks run' for pipeline-based datasets."));
            return;
        }
    };

    if default_profile.views.is_empty() {
        ui.log(&format!("Default profile has no views — nothing to import"));
        return;
    }

    // Collect importable views: those whose source paths exist on disk
    let importable: Vec<(&str, &profile::DSView)> = default_profile
        .views
        .iter()
        .filter(|(_, view)| {
            let source_path = Path::new(view.path());
            let resolved = if source_path.is_absolute() {
                source_path.to_path_buf()
            } else {
                base_dir.join(source_path)
            };
            resolved.exists()
        })
        .map(|(key, view)| (key.as_str(), view))
        .collect();

    if importable.is_empty() {
        ui.log(&format!("No view sources found on disk — nothing to import."));
        ui.log(&format!("Use 'veks run' for pipeline-based datasets."));
        return;
    }

    // Phase 2: Import each view
    ui.log(&format!("\nImporting {} view(s)...", importable.len()));
    for (key, view) in &importable {
        let facet = match Facet::from_key(key) {
            Some(f) => f,
            None => {
                ui.log(&format!("  Warning: unknown view key '{}', skipping", key));
                continue;
            }
        };

        let source_path = {
            let p = Path::new(view.path());
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                base_dir.join(p)
            }
        };

        ui.emitln(format!("\n--- {} ---", key));
        import_single_facet(
            &source_path,
            facet,
            None,
            None,
            args,
            ui,
        );
    }

    ui.log(&format!("\nAll views imported successfully."));
}

/// Import a single facet from source to output.
///
/// When `output_path` is `None`, the output filename is derived from the facet
/// key and the preferred format (which depends on source element size).
fn import_single_facet(
    source_path: &Path,
    facet: Facet,
    output_path: Option<&Path>,
    format_override: Option<&str>,
    args: &ImportArgs,
    ui: &crate::ui::UiHandle,
) {
    let source_format = resolve_source_format(format_override, source_path, facet, ui);

    // Probe source first to get metadata (lightweight — no data loading)
    ui.log(&format!("  probing {} ({})...", source_path.display(), source_format));
    let probe = reader::probe_source_for_facet(source_path, source_format, facet).unwrap_or_else(|e| {
        ui.emitln(format!("Failed to probe source: {}", e));
        std::process::exit(1);
    });
    let dimension = probe.dimension;
    let element_size = probe.element_size;
    let record_count = probe.record_count;
    let target_format = facet.preferred_format(element_size);

    ui.log(&format!(
        "  probed: dimension={}, element_size={}, records={}, target={}",
        dimension,
        element_size,
        record_count.map_or("unknown".to_string(), |n| n.to_string()),
        target_format,
    ));

    // Derive output path if not provided
    let derived_output;
    let output_path = match output_path {
        Some(p) => p,
        None => {
            derived_output = std::path::PathBuf::from(format!(
                "{}.{}",
                facet.key(),
                target_format.name()
            ));
            &derived_output
        }
    };

    // Skip if output already exists and is verified complete, unless --force
    if !args.force && output_path.exists() {
        if let Ok(meta) = std::fs::metadata(output_path) {
            let actual_size = meta.len();
            if actual_size > 0 {
                let status = check_output_completeness(
                    output_path,
                    actual_size,
                    record_count,
                    dimension,
                    target_format,
                );
                match status {
                    OutputStatus::Complete => {
                        ui.log(&format!(
                            "  skipping {} — output complete ({} bytes). Use --force to re-import.",
                            facet.key(),
                            actual_size
                        ));
                        return;
                    }
                    OutputStatus::Incomplete { actual, expected } => {
                        let unit = if target_format == VecFormat::Slab { "records" } else { "bytes" };
                        if actual == 0 {
                            ui.log(&format!(
                                "  overwriting {} — output corrupt or unfinished ({} bytes). Expected {} {}.",
                                facet.key(),
                                actual_size,
                                expected,
                                unit,
                            ));
                        } else {
                            ui.log(&format!(
                                "  re-importing {} — output incomplete: {} of {} {} ({:.1}%).",
                                facet.key(),
                                actual,
                                expected,
                                unit,
                                (actual as f64 / expected as f64) * 100.0
                            ));
                        }
                        // Fall through to re-import
                    }
                    OutputStatus::Unknown => {
                        // Cannot determine completeness — treat non-empty as complete (conservative)
                        ui.log(&format!(
                            "  skipping {} — output exists ({} bytes, completeness unverifiable). Use --force to re-import.",
                            facet.key(),
                            actual_size
                        ));
                        return;
                    }
                }
            }
        }
    }

    if facet.is_mnode() {
        ui.log(&format!(
            "Importing {} -> {} ({})",
            source_path.display(),
            output_path.display(),
            target_format,
        ));
        ui.log(&format!(
            "  records: {} (MNode metadata)",
            record_count.map_or("unknown".to_string(), |n| n.to_string())
        ));
    } else {
        ui.log(&format!(
            "Importing {} -> {} ({}, {} bytes/element)",
            source_path.display(),
            output_path.display(),
            target_format,
            element_size,
        ));
        ui.log(&format!(
            "  dimension: {}, records: {}",
            dimension,
            record_count.map_or("unknown".to_string(), |n| n.to_string())
        ));
    }

    // Open the full source reader (may spawn threads, load data)
    ui.log(&format!("  opening source reader..."));
    let mut source = reader::open_source_for_facet(source_path, source_format, facet, args.threads, None).unwrap_or_else(|e| {
        ui.emitln(format!("Failed to open source: {}", e));
        std::process::exit(1);
    });

    // Create output directory if needed
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                ui.emitln(format!("Failed to create directory {}: {}", parent.display(), e));
                std::process::exit(1);
            });
        }
    }

    // Warn the user when we're about to truncate a large existing file,
    // since the filesystem must deallocate all blocks before File::create
    // returns — this can stall for many seconds on multi-GB files.
    if output_path.exists() {
        if let Ok(meta) = std::fs::metadata(output_path) {
            let size = meta.len();
            if size > 1 << 30 {
                ui.log(&format!(
                    "  removing existing output ({:.1} GB) — this may take a moment...",
                    size as f64 / (1u64 << 30) as f64
                ));
            } else if size > 1 << 20 {
                ui.log(&format!(
                    "  removing existing output ({:.1} MB)...",
                    size as f64 / (1u64 << 20) as f64
                ));
            }
        }
    }

    // Open sink
    let sink_config = SinkConfig {
        dimension,
        source_format,
        slab_page_size: args.slab_page_size,
        slab_namespace: args.slab_namespace,
    };
    let mut sink =
        writer::open_sink(output_path, target_format, &sink_config).unwrap_or_else(|e| {
            ui.emitln(format!("Failed to open sink: {}", e));
            std::process::exit(1);
        });

    // Progress via UI handle
    let pb = if let Some(total) = record_count {
        ui.bar(total, "importing records")
    } else {
        ui.spinner("importing records")
    };

    // Import loop
    let mut ordinal: i64 = 0;
    while let Some(data) = source.next_record() {
        sink.write_record(ordinal, &data);
        ordinal += 1;
        pb.inc(1);
    }

    pb.finish();

    sink.finish().unwrap_or_else(|e| {
        ui.emitln(format!("Failed to finalize output: {}", e));
        std::process::exit(1);
    });

    info!("Imported {} records for facet {}", ordinal, facet);
    ui.log(&format!("  wrote {} records to {}", ordinal, output_path.display()));
}

/// Result of comparing an existing output file against the upstream source bounds
enum OutputStatus {
    /// Output file size or record count matches the expected upstream count
    Complete,
    /// Output is smaller than expected — partial/interrupted import.
    ///
    /// For xvec formats, `actual` and `expected` are byte counts.
    /// For slab, they are record counts.
    Incomplete { actual: u64, expected: u64 },
    /// Cannot determine expected size or record count
    Unknown,
}

/// Check whether an existing output file is complete by comparing against
/// the upstream record count.
///
/// For xvec formats, compares file size against the expected byte count.
/// For slab, opens the file and sums page-entry record counts against the
/// upstream total.
fn check_output_completeness(
    output_path: &Path,
    actual_size: u64,
    record_count: Option<u64>,
    dimension: u32,
    target_format: VecFormat,
) -> OutputStatus {
    let Some(expected_records) = record_count else {
        return OutputStatus::Unknown;
    };

    if target_format == VecFormat::Slab {
        return check_slab_completeness(output_path, expected_records);
    }

    let Some(expected_bytes) = target_format.expected_file_size(expected_records, dimension) else {
        return OutputStatus::Unknown;
    };
    if actual_size >= expected_bytes {
        OutputStatus::Complete
    } else {
        OutputStatus::Incomplete {
            actual: actual_size,
            expected: expected_bytes,
        }
    }
}

/// Check slab completeness by counting records across all data pages.
///
/// Opens the slab file read-only, reads each data page's footer to get its
/// record count, and compares the total against the expected upstream count.
///
/// Returns `Incomplete { actual: 0, expected }` if the slab cannot be opened
/// (e.g. no pages-page from an interrupted write), since a corrupt/unfinished
/// slab is definitively incomplete.
fn check_slab_completeness(output_path: &Path, expected_records: u64) -> OutputStatus {
    // Use lightweight probe to get total records from cached page metadata
    // without reading every data page (which would be extremely slow for
    // large slab files).
    let stats = match slabtastic::SlabReader::probe(output_path) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("    (slab unreadable: {} — treating as incomplete)", e);
            return OutputStatus::Incomplete {
                actual: 0,
                expected: expected_records,
            };
        }
    };
    let actual_records = stats.total_records;
    if actual_records >= expected_records {
        OutputStatus::Complete
    } else {
        OutputStatus::Incomplete {
            actual: actual_records,
            expected: expected_records,
        }
    }
}

/// Format records/sec for the progress bar.
///
/// Format a count with thousands separators.
fn format_count(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}

/// Resolve the source format from an override string, auto-detection, or facet default
fn resolve_source_format(
    format_override: Option<&str>,
    source_path: &Path,
    _facet: Facet,
    ui: &crate::ui::UiHandle,
) -> VecFormat {
    if let Some(fmt_str) = format_override {
        VecFormat::from_extension(fmt_str).unwrap_or_else(|| {
            ui.emitln(format!("Unknown source format: '{}'", fmt_str));
            std::process::exit(1);
        })
    } else {
        VecFormat::detect(source_path).unwrap_or_else(|| {
            ui.emitln(format!(
                "Could not auto-detect source format for '{}'. Use --from to specify.",
                source_path.display()
            ));
            std::process::exit(1);
        })
    }
}
