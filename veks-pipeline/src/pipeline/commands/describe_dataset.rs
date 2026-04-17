// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate comprehensive dataset documentation.
//!
//! Reads `dataset.yaml`, `variables.yaml`, and pipeline steps to produce
//! a `dataset.md` that explains the dataset's structure, data flow, ordinal
//! relationships, profiles, facets, and partition model. Optionally generates
//! `exemplars.md` with sample outputs from the explain commands.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::VectorReader;
use vectordata::dataset::DatasetConfig;
use vectordata::io::MmapVectorReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

pub struct DescribeDatasetOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(DescribeDatasetOp)
}

impl CommandOp for DescribeDatasetOp {
    fn command_path(&self) -> &str {
        "analyze describe-dataset"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate comprehensive dataset documentation (dataset.md)".into(),
            body: format!(
                r#"# analyze describe-dataset

Generate a comprehensive markdown document describing a dataset.

## Description

Reads `dataset.yaml` and `variables.yaml` to produce `dataset.md` with:

- **Overview** — name, metric, dimensions, vector counts
- **Profiles** — all profiles with their views, base counts, and facets
- **Data Processing Pipeline** — step-by-step processing chain
- **Ordinal Relationships** — how base, query, metadata, and partition ordinals relate
- **Oracle Partitions** — per-label query set model (when partitions exist)
- **File Inventory** — all data files with sizes and formats

When `--exemplars` is set, also generates `exemplars.md` with sample
outputs from the explain commands for representative queries.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let workspace = if let Some(dir) = options.get("directory") {
            PathBuf::from(dir)
        } else {
            ctx.workspace.clone()
        };

        let ds_path = workspace.join("dataset.yaml");
        let config = match DatasetConfig::load_and_resolve(&ds_path) {
            Ok(c) => c,
            Err(e) => return error_result(format!("load dataset.yaml: {}", e), start),
        };

        let docs_dir = workspace.join("docs");
        let _ = std::fs::create_dir_all(&docs_dir);
        let output_path = docs_dir.join("dataset.md");
        let gen_exemplars = options.get("exemplars")
            .map(|s| s == "true")
            .unwrap_or(false);

        ctx.ui.log(&format!("  generating dataset.md for '{}'...", config.name));

        let mut doc = String::with_capacity(16384);

        // ── Header ──────────────────────────────────────────────────
        write_header(&config, &workspace, &mut doc);

        // ── Source Provenance ───────────────────────────────────────
        write_provenance(&config, &workspace, &mut doc);

        // ── Facet Legend ────────────────────────────────────────────
        write_facet_legend(&config, &mut doc);

        // ── Profiles ────────────────────────────────────────────────
        write_profiles(&config, &workspace, &mut doc);

        // ── Data Processing Pipeline ────────────────────────────────
        write_pipeline(&config, &workspace, &mut doc);

        // ── Ordinal Relationships ───────────────────────────────────
        write_ordinals(&config, &workspace, &mut doc);

        // ── Oracle Partitions ───────────────────────────────────────
        write_partitions(&config, &workspace, &mut doc);

        // ── Verification Summary ────────────────────────────────────
        write_verification(&config, &mut doc);

        // ── File Inventory ──────────────────────────────────────────
        write_inventory(&config, &workspace, &mut doc);

        // ── Write output ────────────────────────────────────────────
        match std::fs::write(&output_path, &doc) {
            Ok(()) => ctx.ui.log(&format!("  wrote {}", output_path.display())),
            Err(e) => return error_result(format!("write {}: {}", output_path.display(), e), start),
        }

        // ── Exemplars ───────────────────────────────────────────────
        if gen_exemplars {
            let exemplar_path = docs_dir.join("exemplars.md");
            ctx.ui.log("  generating exemplars.md...");
            let exemplar_doc = generate_exemplars(&config, &workspace, ctx);
            match std::fs::write(&exemplar_path, &exemplar_doc) {
                Ok(()) => ctx.ui.log(&format!("  wrote {}", exemplar_path.display())),
                Err(e) => ctx.ui.log(&format!("  WARNING: write exemplars.md: {}", e)),
            }
        }

        CommandResult {
            status: Status::Ok,
            message: format!("generated dataset.md for '{}'", config.name),
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("directory", "Path", false, Some("."),
                "Dataset directory (must contain dataset.yaml)", OptionRole::Input),
            opt("exemplars", "bool", false, Some("false"),
                "Also generate exemplars.md with sample explain outputs", OptionRole::Config),
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Section generators
// ═══════════════════════════════════════════════════════════════════════

fn write_header(config: &DatasetConfig, workspace: &Path, doc: &mut String) {
    doc.push_str(&format!("# {}\n\n", config.name));

    if let Some(ref desc) = config.description {
        doc.push_str(&format!("{}\n\n", desc));
    }

    doc.push_str("## Overview\n\n");
    doc.push_str("| Property | Value |\n");
    doc.push_str("|----------|-------|\n");
    doc.push_str(&format!("| Name | `{}` |\n", config.name));

    if let Some(df) = config.distance_function() {
        doc.push_str(&format!("| Distance function | {} |\n", df));
    }
    if let Some(norm) = config.is_normalized() {
        doc.push_str(&format!("| Normalized | {} |\n", norm));
    }

    // Probe base vectors for dimension/count/type
    if let Some(profile) = config.default_profile() {
        if let Some(view) = profile.view("base_vectors") {
            let path = workspace.join(view.path());
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("fvec");
            let etype = match ext {
                "fvec" | "fvecs" => "f32",
                "mvec" => "f16",
                "bvec" | "bvecs" => "u8",
                "ivec" | "ivecs" => "i32",
                _ => ext,
            };
            if let Ok(reader) = MmapVectorReader::<f32>::open_fvec(&path) {
                let count = VectorReader::<f32>::count(&reader);
                let dim = VectorReader::<f32>::dim(&reader);
                doc.push_str(&format!("| Base vectors | {} ({} format) |\n", format_count(count), etype));
                doc.push_str(&format!("| Dimensions | {} |\n", dim));
            }
        }
        if let Some(view) = profile.view("query_vectors") {
            let path = workspace.join(view.path());
            if let Ok(reader) = MmapVectorReader::<f32>::open_fvec(&path) {
                let count = VectorReader::<f32>::count(&reader);
                doc.push_str(&format!("| Query vectors | {} |\n", format_count(count)));
            }
        }
        if let Some(maxk) = profile.maxk {
            doc.push_str(&format!("| k (neighbors) | {} |\n", maxk));
        }
    }

    // Variables — format numbers consistently
    if let Some(v) = config.variable("partition_count") {
        doc.push_str(&format!("| Oracle partitions | {} |\n", v));
    }
    if let Some(v) = config.variable("duplicate_count") {
        if let Ok(n) = v.parse::<u64>() {
            let base = config.variable("vector_count").and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
            if base > 0 {
                doc.push_str(&format!("| Duplicate vectors | {} ({:.2}% of {}) |\n",
                    format_count(n as usize), 100.0 * n as f64 / base as f64, format_count(base as usize)));
            } else {
                doc.push_str(&format!("| Duplicate vectors | {} |\n", format_count(n as usize)));
            }
        }
    }
    if let Some(v) = config.variable("zero_count") {
        doc.push_str(&format!("| Zero vectors | {} |\n", v));
    }

    // Total dataset size
    let total_size = total_dir_size(workspace);
    if total_size > 0 {
        doc.push_str(&format!("| Total size on disk | {} |\n", format_size(total_size)));
    }

    // Build provenance
    let attrs = &config.attributes;
    if let Some(a) = attrs {
        if let Some(ref ver) = a.veks_version {
            doc.push_str(&format!("| Built with veks | {} |\n", ver));
        }
        if let Some(ref build) = a.veks_build {
            // Decode build number to timestamp if possible
            let decoded = if let Some(num_str) = build.rsplit('.').next() {
                if let Ok(epoch) = num_str.parse::<u64>() {
                    format!("`{}` ({})", build, epoch_to_utc(epoch))
                } else {
                    format!("`{}`", build)
                }
            } else {
                format!("`{}`", build)
            };
            doc.push_str(&format!("| Build | {} |\n", decoded));
        }
    }

    doc.push_str("\n");
}

fn write_provenance(_config: &DatasetConfig, workspace: &Path, doc: &mut String) {
    // Find source files (symlink targets or files starting with _)
    let mut source_files: Vec<(String, String, u64)> = Vec::new(); // (name, target, size)

    if let Ok(entries) = std::fs::read_dir(workspace) {
        let mut files: Vec<_> = entries.flatten().collect();
        files.sort_by_key(|e| e.file_name());
        for entry in files {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            if name.starts_with('_') || name.ends_with(".fvecs") || name.ends_with(".ivecs") {
                let target = if path.symlink_metadata().map(|m| m.file_type().is_symlink()).unwrap_or(false) {
                    std::fs::read_link(&path)
                        .map(|t| t.to_string_lossy().to_string())
                        .unwrap_or_default()
                } else { String::new() };
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                source_files.push((name, target, size));
            }
        }
    }

    if source_files.is_empty() { return; }

    doc.push_str("## Source Data\n\n");
    doc.push_str("The dataset was built from these source files:\n\n");
    doc.push_str("| File | Size | Origin |\n");
    doc.push_str("|------|------|--------|\n");
    for (name, target, size) in &source_files {
        let origin = if target.is_empty() { "local".to_string() }
            else { format!("symlink → `{}`", target) };
        doc.push_str(&format!("| `{}` | {} | {} |\n", name, format_size(*size), origin));
    }
    doc.push_str("\n");
}

fn write_facet_legend(config: &DatasetConfig, doc: &mut String) {
    doc.push_str("## Facet Legend\n\n");
    doc.push_str("Facets identify the role of each data file in the benchmark:\n\n");
    doc.push_str("| Code | Facet | Description | Format |\n");
    doc.push_str("|------|-------|-------------|--------|\n");
    doc.push_str("| B | `base_vectors` | Vectors to search against | fvec (f32) |\n");
    doc.push_str("| Q | `query_vectors` | Vectors to search for | fvec (f32) |\n");
    doc.push_str("| G | `neighbor_indices` | Ground-truth k-nearest neighbor ordinals | ivec (i32) |\n");
    doc.push_str("| D | `neighbor_distances` | Ground-truth distances | fvec (f32) |\n");

    let has_metadata = config.default_profile()
        .map(|p| p.view("metadata_content").is_some())
        .unwrap_or(false);
    let has_filtered = config.default_profile()
        .map(|p| p.view("filtered_neighbor_indices").is_some())
        .unwrap_or(false);
    let has_partitions = config.profiles.profiles.values().any(|p| p.partition);

    if has_metadata {
        doc.push_str("| M | `metadata_content` | Per-base-vector metadata labels | scalar (u8/i32/etc.) |\n");
        doc.push_str("| P | `metadata_predicates` | Per-query filter conditions | scalar or slab |\n");
        doc.push_str("| R | `metadata_indices` | Per-query matching base ordinals | ivvec (variable-length i32) |\n");
    }
    if has_filtered {
        doc.push_str("| F | `filtered_neighbor_indices` | k-nearest among predicate-matching base vectors | ivec (i32) |\n");
    }
    if has_partitions {
        doc.push_str("| O | oracle partition profiles | Per-label subsets with independent KNN | per-partition directory |\n");
    }
    doc.push_str("\n");

    doc.push_str("**File formats:** ");
    doc.push_str("`fvec`/`ivec` = dimension-prefixed vectors (`[dim:i32, v0, v1, ..., v_dim-1]` per record). ");
    doc.push_str("`ivvec` = variable-length integer vectors (same prefix, varying dimensions). ");
    doc.push_str("Scalar files (`.u8`, `.i32`) = flat packed values, one element per record.\n\n");
}

fn write_profiles(config: &DatasetConfig, workspace: &Path, doc: &mut String) {
    doc.push_str("## Profiles\n\n");

    let partition_count = config.profiles.profiles.values()
        .filter(|p| p.partition).count();
    let non_partition: Vec<(&str, &vectordata::dataset::profile::DSProfile)> = config.profiles.profiles.iter()
        .filter(|(_, p)| !p.partition)
        .map(|(n, p)| (n.as_str(), p))
        .collect();
    let sized_count = non_partition.len().saturating_sub(
        if config.profiles.profiles.contains_key("default") { 1 } else { 0 });

    doc.push_str(&format!("**{}** profiles total: ", config.profiles.profiles.len()));
    let mut parts = Vec::new();
    parts.push("1 default".to_string());
    if sized_count > 0 { parts.push(format!("{} sized", sized_count)); }
    if partition_count > 0 { parts.push(format!("{} partition", partition_count)); }
    doc.push_str(&parts.join(", "));
    doc.push_str("\n\n");

    // Show non-partition profiles in detail
    let mut ordered: Vec<&str> = non_partition.iter().map(|(n, _)| *n).collect();
    ordered.sort();
    if let Some(pos) = ordered.iter().position(|n| *n == "default") {
        ordered.remove(pos);
        ordered.insert(0, "default");
    }

    for name in &ordered {
        let profile = &config.profiles.profiles[*name];
        let kind = if *name == "default" { " (default)" } else { " (sized)" };

        doc.push_str(&format!("### `{}`{}\n\n", name, kind));

        if let Some(bc) = profile.base_count {
            doc.push_str(&format!("- Base count: {}\n", format_count(bc as usize)));
        } else if *name == "default" {
            // Probe actual count for default profile
            if let Some(view) = profile.view("base_vectors") {
                let path = workspace.join(view.path());
                if let Ok(reader) = MmapVectorReader::<f32>::open_fvec(&path) {
                    doc.push_str(&format!("- Base count: {} (full dataset)\n",
                        format_count(VectorReader::<f32>::count(&reader))));
                }
            }
        }
        if let Some(maxk) = profile.maxk {
            doc.push_str(&format!("- maxk: {}\n", maxk));
        }

        if !profile.views.is_empty() {
            doc.push_str("\n| Facet | Path | Size | Notes |\n");
            doc.push_str("|-------|------|------|-------|\n");

            for (facet_name, view) in &profile.views {
                let path = workspace.join(view.path());
                let exists = path.exists();
                let size = if exists {
                    std::fs::metadata(&path)
                        .map(|m| format_size(m.len()))
                        .unwrap_or_else(|_| "—".to_string())
                } else {
                    "missing".to_string()
                };
                let is_symlink = path.symlink_metadata()
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false);
                let note = if !exists { "not yet computed".to_string() }
                    else if is_symlink {
                        let target = std::fs::read_link(&path)
                            .map(|t| t.to_string_lossy().to_string())
                            .unwrap_or_default();
                        format!("symlink → `{}`", target)
                    } else { String::new() };
                doc.push_str(&format!("| {} | `{}` | {} | {} |\n",
                    facet_name, view.path(), size, note));
            }
            doc.push_str("\n");
        }
    }

    // Partition profiles as a compact summary (not individual sections)
    if partition_count > 0 {
        doc.push_str(&format!("### Partition profiles ({} partitions)\n\n", partition_count));
        doc.push_str("Each partition has: `base_vectors`, `query_vectors`, `neighbor_indices`, `neighbor_distances`\n\n");
    }
}

fn write_pipeline(config: &DatasetConfig, workspace: &Path, doc: &mut String) {
    doc.push_str("## Data Processing Pipeline\n\n");

    let steps = match config.upstream {
        Some(ref pipeline) => match pipeline.steps {
            Some(ref s) => s,
            None => { doc.push_str("No pipeline steps recorded.\n\n"); return; }
        },
        None => { doc.push_str("No pipeline configuration recorded.\n\n"); return; }
    };

    // Separate state-set initialization from real processing steps
    let state_steps: Vec<_> = steps.iter()
        .filter(|s| s.run == "state set")
        .collect();
    let processing_steps: Vec<_> = steps.iter()
        .filter(|s| s.run != "state set")
        .collect();

    if !state_steps.is_empty() {
        doc.push_str(&format!("**Initialization:** {} state variables set", state_steps.len()));
        let named: Vec<String> = state_steps.iter()
            .filter_map(|s| s.options.get("name")
                .and_then(|v| v.as_str())
                .map(|n| format!("`{}`", n)))
            .collect();
        if !named.is_empty() {
            doc.push_str(&format!(" ({})", named.join(", ")));
        }
        doc.push_str("\n\n");
    }

    doc.push_str("### Processing Steps\n\n");
    doc.push_str("| # | Step | Command | Description |\n");
    doc.push_str("|---|------|---------|-------------|\n");

    for (i, step) in processing_steps.iter().enumerate() {
        let id = step.effective_id();
        let desc = step.description.as_deref().unwrap_or("");
        let mut tags = Vec::new();
        if step.per_profile { tags.push("per-profile"); }
        if step.finalize { tags.push("finalize"); }
        let tag_str = if tags.is_empty() { String::new() }
            else { format!(" *({})*", tags.join(", ")) };
        doc.push_str(&format!("| {} | `{}` | `{}`{} | {} |\n",
            i + 1, id, step.run, tag_str, desc));
    }
    doc.push_str("\n");

    // Show key step parameters for important processing steps
    let key_steps = ["generate-metadata", "generate-predicates", "partition-profiles"];
    let mut shown_params = false;
    for step in steps {
        let id = step.effective_id();
        if !key_steps.contains(&id.as_str()) { continue; }
        let params: Vec<String> = step.options.iter()
            .filter(|(k, _)| !["output", "source", "after"].contains(&k.as_str()))
            .map(|(k, v)| {
                let val = v.as_str().map(|s| s.to_string())
                    .or_else(|| v.as_i64().map(|n| n.to_string()))
                    .unwrap_or_else(|| format!("{:?}", v));
                format!("`{}`={}", k, val)
            })
            .collect();
        if !params.is_empty() {
            if !shown_params {
                doc.push_str("### Key Step Parameters\n\n");
                shown_params = true;
            }
            doc.push_str(&format!("- **{}**: {}\n", id, params.join(", ")));
        }
    }
    if shown_params { doc.push_str("\n"); }

    // Defaults
    if let Some(ref pipeline) = config.upstream {
        if let Some(ref defaults) = pipeline.defaults {
            if !defaults.is_empty() {
                doc.push_str("### Pipeline Defaults\n\n");
                for (k, v) in defaults {
                    doc.push_str(&format!("- `{}` = `{}`\n", k, v));
                }
                doc.push_str("\n");
            }
        }
    }

    // Data flow — derived from actual pipeline steps and source files.
    doc.push_str("### Data Flow\n\n");

    // Determine which facets are provided (symlinks to source) vs computed (pipeline steps)
    let profile = config.default_profile();
    let is_provided = |facet: &str| -> bool {
        profile.and_then(|p| p.view(facet))
            .map(|v| {
                let path = workspace.join(v.path());
                path.symlink_metadata()
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    };

    doc.push_str("```\n");

    // Source/provided facets
    let mut provided = Vec::new();
    if is_provided("base_vectors") { provided.push("B (base_vectors)"); }
    if is_provided("query_vectors") { provided.push("Q (query_vectors)"); }
    if is_provided("neighbor_indices") { provided.push("G (neighbor_indices)"); }
    if is_provided("neighbor_distances") { provided.push("D (neighbor_distances)"); }
    if !provided.is_empty() {
        doc.push_str(&format!("provided:  {}\n", provided.join(", ")));
    }

    // Computed facets — walk the actual processing steps
    let mut prev_indent = false;
    for step in &processing_steps {
        let id = step.effective_id();

        // Skip verify/finalize steps in the data flow — they don't produce data
        if id.starts_with("verify") || id.starts_with("scan") || step.finalize { continue; }

        // Derive facet code from the step's command or output files
        let facet_code = infer_facet_code(&id, &step.run);

        // Collect all output-role paths from this step's options.
        let output_paths: Vec<String> = step.output_paths().into_iter()
            .map(|p| {
                // Strip variable prefixes like ${profile_dir} for display
                if let Some(rest) = p.strip_prefix("${profile_dir}") {
                    rest.to_string()
                } else if let Some(rest) = p.strip_prefix("${cache}/") {
                    rest.to_string()
                } else if p.contains("${") {
                    // Other variable — show the filename portion
                    p.rsplit('/').next().unwrap_or(&p).to_string()
                } else {
                    p
                }
            })
            .filter(|p| !p.is_empty())
            .collect();

        let outputs_str = if !output_paths.is_empty() {
            output_paths.join(", ")
        } else {
            // No explicit output — use step description as fallback
            step.description.as_deref().unwrap_or("").to_string()
        };

        let out_label = if let Some(code) = facet_code {
            format!("({}) {}", code, outputs_str)
        } else if !outputs_str.is_empty() {
            outputs_str
        } else {
            String::new()
        };

        let connector = if prev_indent { "    │\n" } else { "" };
        if !out_label.is_empty() {
            doc.push_str(&format!("{}    {} ─→ {}\n", connector, id, out_label));
        } else {
            doc.push_str(&format!("{}    {}\n", connector, id));
        }
        prev_indent = true;
    }

    doc.push_str("```\n\n");
}

fn write_ordinals(config: &DatasetConfig, workspace: &Path, doc: &mut String) {
    doc.push_str("## Ordinal Relationships\n\n");

    doc.push_str("Vector datasets use ordinal indices to relate data across files. ");
    doc.push_str("An ordinal is a 0-based position that identifies one record ");
    doc.push_str("across all files sharing that ordinal space.\n\n");

    let profile = match config.default_profile() {
        Some(p) => p,
        None => { doc.push_str("No default profile available.\n\n"); return; }
    };

    // Probe actual counts
    let base_count = probe_count(workspace, profile, "base_vectors");
    let query_count = probe_count(workspace, profile, "query_vectors");
    let gt_count = probe_ivec_count(workspace, profile, "neighbor_indices");

    doc.push_str("### Base vector ordinals\n\n");
    if let Some(n) = base_count {
        doc.push_str(&format!("- Range: `0..{}` ({} vectors)\n", n, format_count(n)));
    }
    doc.push_str("- Shared by: `base_vectors[i]`, `metadata_content[i]` — same ordinal, same vector\n");
    doc.push_str("- `base_vectors[42]` is the 43rd vector; `metadata_content[42]` is its label\n\n");

    doc.push_str("### Query ordinals\n\n");
    if let Some(n) = query_count {
        doc.push_str(&format!("- Range: `0..{}` ({} queries)\n", n, format_count(n)));
    }
    doc.push_str("- Shared by: `query_vectors[q]`, `metadata_predicates[q]`, `neighbor_indices[q]`, ");
    doc.push_str("`filtered_neighbor_indices[q]`\n");
    doc.push_str("- All files indexed by the same `q` describe the same query\n\n");

    // Concrete example
    if let (Some(_qc), Some(k)) = (query_count, gt_count) {
        doc.push_str("### Worked example: query ordinal 42\n\n");
        doc.push_str("```\n");
        doc.push_str("query_vectors[42]              → the 128-dim f32 search vector\n");
        doc.push_str(&format!("neighbor_indices[42]           → i32[{}] nearest base ordinals (G facet)\n", k));
        if profile.view("metadata_predicates").is_some() {
            doc.push_str("metadata_predicates[42]        → the filter condition for this query (P facet)\n");
            doc.push_str("metadata_indices[42]           → variable-length list of matching base ordinals (R facet)\n");
            doc.push_str(&format!("filtered_neighbor_indices[42]  → i32[{}] nearest among matching bases (F facet)\n", k));
        }
        doc.push_str("```\n\n");
        doc.push_str(&format!(
            "The values inside `neighbor_indices[42]` are **base ordinals** (range `0..{}`), \
            not query ordinals. They reference positions in `base_vectors`.\n\n",
            base_count.unwrap_or(0)));
    }

    doc.push_str("### Ground truth (G facet)\n\n");
    if let (Some(qc), Some(k)) = (query_count, gt_count) {
        doc.push_str(&format!("- `neighbor_indices[q]` → `i32[{}]` — the {} nearest base ordinals to query `q`\n",
            k, k));
        doc.push_str(&format!("- {} queries x {} neighbors = {} ordinal pairs\n", format_count(qc), k, format_count(qc * k)));
        doc.push_str("- Sorted by ascending distance (nearest first)\n\n");
    }

    if profile.view("metadata_predicates").is_some() {
        doc.push_str("### Predicate evaluation chain\n\n");
        doc.push_str("```\n");
        doc.push_str("metadata_predicates[q]    \"field_0 == 5\"     (one predicate per query)\n");
        doc.push_str("        │\n");
        doc.push_str("        ▼  evaluate against metadata_content[0..N]\n");
        doc.push_str("metadata_indices[q]       [12, 45, 89, ...]  (variable-length: all matching base ordinals)\n");
        doc.push_str("        │\n");
        doc.push_str("        ▼  brute-force KNN restricted to matching ordinals\n");
        doc.push_str("filtered_neighbor_indices[q]  [89, 12, 45, ...]  (top-k nearest among matches)\n");
        doc.push_str("```\n\n");
        doc.push_str("Note: `metadata_indices` uses ivvec format (variable-length records) because \
            the number of matching base vectors varies per query. \
            `filtered_neighbor_indices` is fixed-length ivec (always k neighbors).\n\n");
    }
}

fn write_partitions(config: &DatasetConfig, workspace: &Path, doc: &mut String) {
    let partition_profiles: Vec<(&str, &vectordata::dataset::profile::DSProfile)> = config.profiles.profiles.iter()
        .filter(|(_, p)| p.partition)
        .map(|(n, p)| (n.as_str(), p))
        .collect();

    if partition_profiles.is_empty() { return; }

    doc.push_str("## Oracle Partitions\n\n");

    doc.push_str(&format!("This dataset has **{}** oracle partition profiles.\n\n", partition_profiles.len()));

    doc.push_str("### Partition Model\n\n");
    doc.push_str("Each partition isolates the base vectors with a specific metadata label value. ");
    doc.push_str("Within a partition:\n\n");
    doc.push_str("- **Base vectors** are the subset of global base vectors with matching label\n");
    doc.push_str("- **Query vectors** are limited to queries whose predicate matches the partition label\n");
    doc.push_str("- **KNN results** are computed independently within the partition's ordinal space\n\n");

    doc.push_str("### Ordinal Remapping\n\n");
    doc.push_str("Partition ordinals are **not** the same as global ordinals:\n\n");
    doc.push_str("- **Base ordinal 0** in a partition is the first global base vector with the matching label\n");
    doc.push_str("- **Query ordinal 0** in a partition is the first global query whose predicate matches\n");
    doc.push_str("- Neighbor indices in partition KNN reference partition base ordinals, not global ones\n\n");

    doc.push_str("### Per-Partition Query Sets\n\n");
    doc.push_str("A query with `predicate == label_N` appears **only** in partition `label_N`. ");
    doc.push_str("It has no entry in any other partition's KNN results. ");
    doc.push_str("The sum of per-partition query counts equals the total query count.\n\n");

    // Show partition summary table
    doc.push_str("### Partition Summary\n\n");

    // Compute total query count from default profile
    let total_queries = config.default_profile()
        .and_then(|p| p.view("query_vectors"))
        .and_then(|v| {
            let path = workspace.join(v.path());
            MmapVectorReader::<f32>::open_fvec(&path).ok()
                .map(|r| VectorReader::<f32>::count(&r))
        })
        .unwrap_or(0);

    doc.push_str("| Partition | Base vectors | Queries | Query source |\n");
    doc.push_str("|-----------|-------------|---------|---------------|\n");

    let mut sorted: Vec<_> = partition_profiles.iter().collect();
    sorted.sort_by_key(|(n, _)| *n);

    let mut total_partition_queries: usize = 0;

    for (name, profile) in &sorted {
        let bc = profile.base_count
            .map(|c| format_count(c as usize))
            .unwrap_or_else(|| "—".to_string());

        let qv_path = workspace.join(format!("profiles/{}/query_vectors.fvec", name));
        let is_symlink = qv_path.symlink_metadata()
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);

        let (qc_str, qc_num) = if qv_path.exists() {
            match MmapVectorReader::<f32>::open_fvec(&qv_path) {
                Ok(r) => {
                    let n = VectorReader::<f32>::count(&r);
                    (format_count(n), n)
                }
                Err(_) => ("—".to_string(), 0),
            }
        } else { ("—".to_string(), 0) };

        let source = if is_symlink {
            format!("shared (all {} queries)", format_count(total_queries))
        } else {
            total_partition_queries += qc_num;
            format!("per-label (predicate == {})",
                name.strip_prefix("label_").unwrap_or(name))
        };

        doc.push_str(&format!("| `{}` | {} | {} | {} |\n", name, bc, qc_str, source));
    }

    doc.push_str("\n");

    if total_partition_queries > 0 && total_queries > 0 {
        doc.push_str(&format!(
            "Sum of per-partition queries: **{}** (total query set: {})\n\n",
            format_count(total_partition_queries),
            format_count(total_queries)));

        if total_partition_queries == total_queries {
            doc.push_str("Every query appears in exactly one partition (complete coverage).\n\n");
        } else if total_partition_queries < total_queries {
            let missing = total_queries - total_partition_queries;
            doc.push_str(&format!(
                "{} queries ({:.1}%) have predicates that don't match any partition label.\n\n",
                missing, 100.0 * missing as f64 / total_queries as f64));
        }
    }
}

fn write_verification(config: &DatasetConfig, doc: &mut String) {
    // Extract verification info from variables
    let vars = &config.variables;
    let verify_vars: Vec<(&str, &str)> = vars.iter()
        .filter(|(k, _)| k.starts_with("verified_count:") || k.contains("verify"))
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    if verify_vars.is_empty() { return; }

    doc.push_str("## Verification\n\n");
    doc.push_str("The pipeline verified these artifacts:\n\n");
    doc.push_str("| Artifact | Verified records |\n");
    doc.push_str("|----------|------------------|\n");

    for (k, v) in &verify_vars {
        if let Some(name) = k.strip_prefix("verified_count:") {
            doc.push_str(&format!("| `{}` | {} |\n", name, v));
        }
    }

    // KNN tie info
    if let (Some(tied_q), Some(tied_n)) = (
        config.variable("knn_queries_with_ties"),
        config.variable("knn_tied_neighbors"),
    ) {
        doc.push_str(&format!("\nKNN ties: {} queries with tied neighbors, {} total tied neighbors\n",
            tied_q, tied_n));
    }

    doc.push_str("\n");
}

fn write_inventory(config: &DatasetConfig, workspace: &Path, doc: &mut String) {
    doc.push_str("## File Inventory\n\n");

    // (relative_path, size, symlink_target)
    let mut files: Vec<(String, u64, Option<String>)> = Vec::new();

    for (_pname, profile) in &config.profiles.profiles {
        for (_facet, view) in &profile.views {
            let rel = view.path();
            let path = workspace.join(rel);

            let is_symlink = path.symlink_metadata()
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false);
            let target = if is_symlink {
                std::fs::read_link(&path)
                    .ok()
                    .map(|t| t.to_string_lossy().to_string())
            } else { None };

            // Use the actual file size (follows symlinks)
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

            // Deduplicate by relative path (multiple profiles may reference same file)
            if !files.iter().any(|(r, _, _)| r == rel) {
                files.push((rel.to_string(), size, target));
            }
        }
    }

    if files.is_empty() {
        doc.push_str("No data files found.\n\n");
        return;
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));

    let total: u64 = files.iter().map(|(_, s, _)| *s).sum();

    doc.push_str(&format!("**{}** unique data files, **{}** total\n\n", files.len(), format_size(total)));
    doc.push_str("| Path | Size | Notes |\n");
    doc.push_str("|------|------|-------|\n");
    for (rel, size, target) in &files {
        let note = match target {
            Some(t) => format!("→ `{}`", t),
            None => String::new(),
        };
        let sz = if *size > 0 { format_size(*size) } else { "missing".to_string() };
        doc.push_str(&format!("| `{}` | {} | {} |\n", rel, sz, note));
    }
    doc.push_str("\n");
}

// ═══════════════════════════════════════════════════════════════════════
// Exemplars
// ═══════════════════════════════════════════════════════════════════════

fn generate_exemplars(
    config: &DatasetConfig,
    workspace: &Path,
    ctx: &mut StreamContext,
) -> String {
    let mut doc = String::with_capacity(8192);
    doc.push_str(&format!("# {} — Exemplar Explain Outputs\n\n", config.name));
    doc.push_str("These are sample outputs from the explain commands, showing how data ");
    doc.push_str("flows through the pipeline for representative queries.\n\n");

    let registry = crate::pipeline::registry::CommandRegistry::with_builtins();

    // Pick a representative query ordinal
    let exemplar_ordinal = 0;

    // explain-predicates
    if let Some(factory) = registry.get("analyze explain-predicates") {
        doc.push_str("## Predicate Explanation (query 0)\n\n");
        doc.push_str("```\n");
        let output = run_explain_command(*factory, workspace, exemplar_ordinal, ctx);
        doc.push_str(&output);
        doc.push_str("```\n\n");
    }

    // explain-filtered-knn
    if config.default_profile()
        .and_then(|p| p.view("filtered_neighbor_indices"))
        .is_some()
    {
        if let Some(factory) = registry.get("analyze explain-filtered-knn") {
            doc.push_str("## Filtered KNN Explanation (query 0)\n\n");
            doc.push_str("```\n");
            let output = run_explain_command(*factory, workspace, exemplar_ordinal, ctx);
            doc.push_str(&output);
            doc.push_str("```\n\n");
        }
    }

    // explain-partitions (if partitions exist)
    let has_partitions = config.profiles.profiles.values().any(|p| p.partition);
    if has_partitions {
        if let Some(factory) = registry.get("analyze explain-partitions") {
            doc.push_str("## Partition Explanation (query 0)\n\n");
            doc.push_str("```\n");
            let output = run_explain_command(*factory, workspace, exemplar_ordinal, ctx);
            doc.push_str(&output);
            doc.push_str("```\n\n");
        }
    }

    doc
}

/// Run an explain command and capture its log output.
fn run_explain_command(
    factory: fn() -> Box<dyn CommandOp>,
    workspace: &Path,
    ordinal: usize,
    ctx: &mut StreamContext,
) -> String {
    let mut cmd = factory();
    let mut opts = Options::new();
    opts.set("ordinal", &ordinal.to_string());

    // Capture log output via a collecting sink
    let collector = std::sync::Arc::new(LogCollector::new());
    let old_workspace = ctx.workspace.clone();
    ctx.workspace = workspace.to_path_buf();

    // Save the original UI and swap in a collecting one
    let collecting_ui = veks_core::ui::UiHandle::new(collector.clone());
    let original_ui = std::mem::replace(&mut ctx.ui, collecting_ui);

    let _result = cmd.execute(&opts, ctx);

    // Restore
    ctx.ui = original_ui;
    ctx.workspace = old_workspace;

    collector.take()
}

/// Simple log collector that captures all log messages.
struct LogCollector {
    lines: std::sync::Mutex<Vec<String>>,
}

impl LogCollector {
    fn new() -> Self {
        Self { lines: std::sync::Mutex::new(Vec::new()) }
    }
    fn take(&self) -> String {
        self.lines.lock().unwrap().join("\n")
    }
}

impl veks_core::ui::UiSink for LogCollector {
    fn send(&self, event: veks_core::ui::UiEvent) {
        if let veks_core::ui::UiEvent::Log { message, .. } = event {
            self.lines.lock().unwrap().push(message);
        }
    }
    fn next_progress_id(&self) -> veks_core::ui::ProgressId {
        veks_core::ui::ProgressId(0)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

fn probe_count(workspace: &Path, profile: &vectordata::dataset::profile::DSProfile, facet: &str) -> Option<usize> {
    let view = profile.view(facet)?;
    let path = workspace.join(view.path());
    let reader = MmapVectorReader::<f32>::open_fvec(&path).ok()?;
    Some(VectorReader::<f32>::count(&reader))
}

fn probe_ivec_count(workspace: &Path, profile: &vectordata::dataset::profile::DSProfile, facet: &str) -> Option<usize> {
    let view = profile.view(facet)?;
    let path = workspace.join(view.path());
    let reader = MmapVectorReader::<i32>::open_ivec(&path).ok()?;
    Some(VectorReader::<i32>::dim(&reader))
}

fn total_dir_size(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                total += total_dir_size(&p);
            } else {
                total += std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

fn epoch_to_utc(epoch: u64) -> String {
    let s = epoch;
    let days = s / 86400;
    let time = s % 86400;
    let h = time / 3600;
    let m = (time % 3600) / 60;
    let sec = time % 60;
    let mut y = 1970i64;
    let mut rem = days as i64;
    loop {
        let ydays = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if rem < ydays { break; }
        rem -= ydays;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let mdays = [31, if leap {29} else {28}, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 0usize;
    while mo < 12 && rem >= mdays[mo] { rem -= mdays[mo]; mo += 1; }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo + 1, rem + 1, h, m, sec)
}

/// Infer a facet code from a step's ID and command path.
///
/// Uses the command semantics rather than hardcoded step IDs, so this
/// works for any pipeline configuration.
fn infer_facet_code(step_id: &str, run: &str) -> Option<&'static str> {
    // Match on the command being run (not the step ID, which is user-chosen)
    match run {
        "generate metadata" => Some("M"),
        "generate predicates" => Some("P"),
        "compute evaluate-predicates" => Some("R"),
        "compute filtered-knn" => Some("F"),
        "compute partition-profiles" => Some("O"),
        "compute knn" => {
            if step_id.contains("partition") { Some("partition G/D") }
            else { Some("G, D") }
        }
        "compute knn-faiss" | "compute knn-blas" => Some("G, D"),
        _ => None,
    }
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000_000 { format!("{:.1}B", n as f64 / 1e9) }
    else if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) }
    else if n >= 1_000 { format!("{:.1}K", n as f64 / 1e3) }
    else { format!("{}", n) }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 { format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0) }
    else if bytes >= 1_048_576 { format!("{:.1} MiB", bytes as f64 / 1_048_576.0) }
    else if bytes >= 1024 { format!("{:.1} KiB", bytes as f64 / 1024.0) }
    else { format!("{} B", bytes) }
}

fn error_result(msg: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message: msg.into(),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, desc: &str, role: OptionRole) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
        role,
    }
}
