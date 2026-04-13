// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate catalog index files.
//!
//! Scans the workspace directory for `dataset.yaml` files and writes
//! `catalog.json` / `catalog.yaml` at each directory level. This is
//! the pipeline-integrated form of `veks prepare catalog generate`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::dataset::{CatalogEntry, CatalogLayout, DatasetConfig};

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status,
    StreamContext, render_options_table,
};

pub struct CatalogGenerateOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(CatalogGenerateOp)
}

impl CommandOp for CatalogGenerateOp {
    fn command_path(&self) -> &str {
        "catalog generate"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate catalog index for the dataset directory".into(),
            body: format!(
                "# catalog generate\n\n\
                Generate catalog.json and catalog.yaml index files.\n\n\
                ## Description\n\n\
                Scans the input directory for dataset.yaml files and writes hierarchical \
                catalog index files (catalog.json and catalog.yaml) at each directory level. \
                Each catalog entry contains the dataset name, relative path, and embedded \
                layout (profiles and attributes).\n\n\
                ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = options.get("source").unwrap_or(".");
        let basename = options.get("basename").unwrap_or("catalog");

        let input_path = if Path::new(input_str).is_absolute() {
            PathBuf::from(input_str)
        } else {
            ctx.workspace.join(input_str)
        };

        if !input_path.is_dir() {
            return error_result(format!("{} is not a directory", input_path.display()), start);
        }

        // Canonicalize so find_catalog_boundary can walk above a relative "."
        // workspace to find .catalog_root / .publish_url in ancestor directories.
        let input_abs = std::fs::canonicalize(&input_path).unwrap_or(input_path.clone());

        // Find the catalog boundary: the closest ancestor (or self) containing
        // .catalog_root or .publish_url. This is the top of the catalog hierarchy.
        let boundary = find_catalog_boundary(&input_abs);
        let scan_root = boundary.clone().unwrap_or(input_abs.clone());

        // If a publish tree exists, ensure the current dataset has a .publish
        // sentinel so it is included in the catalog and ready for publishing.
        if boundary.is_some() {
            let publish_sentinel = input_abs.join(".publish");
            if !publish_sentinel.exists() {
                if let Err(e) = std::fs::write(&publish_sentinel, "") {
                    ctx.ui.log(&format!("  warning: could not create .publish: {}", e));
                } else {
                    ctx.ui.log("  created .publish sentinel");
                }
            }
        }

        ctx.ui.log(&format!("  scanning {} for datasets...", scan_root.display()));

        let mut all_datasets = Vec::new();
        walk_for_datasets(&scan_root, &mut all_datasets);
        // Only include publishable datasets (those with .publish sentinel)
        let mut datasets: Vec<DiscoveredDataset> = all_datasets.into_iter()
            .filter(|ds| {
                let ds_dir = ds.yaml_path.parent().unwrap_or(std::path::Path::new("."));
                ds_dir.join(".publish").exists()
            })
            .collect();
        datasets.sort_by(|a, b| a.yaml_path.cmp(&b.yaml_path));

        if datasets.is_empty() {
            return CommandResult {
                status: Status::Ok,
                message: "no datasets found — no catalogs written".into(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        for ds in &datasets {
            ctx.ui.log(&format!("  dataset: {} [{}]", ds.config.name, ds.yaml_path.display()));
        }

        let mut catalog_dirs: BTreeSet<PathBuf> = BTreeSet::new();
        catalog_dirs.insert(scan_root.clone());
        for ds in &datasets {
            let ds_dir = ds.yaml_path.parent().unwrap_or(&ds.yaml_path);
            let mut dir = ds_dir.to_path_buf();
            while dir.starts_with(&scan_root) {
                catalog_dirs.insert(dir.clone());
                match dir.parent() {
                    Some(parent) => dir = parent.to_path_buf(),
                    None => break,
                }
            }
        }

        let mut total_files = 0usize;
        let mut produced = Vec::new();

        // Write catalogs depth-first (deepest directories first) so that
        // parent catalog mtimes are always newer than their children.
        let mut sorted_dirs: Vec<&PathBuf> = catalog_dirs.iter().collect();
        sorted_dirs.sort_by(|a, b| b.components().count().cmp(&a.components().count())
            .then_with(|| a.cmp(b)));

        for catalog_dir in &sorted_dirs {
            if catalog_dir.join(DO_NOT_CATALOG_FILE).exists() {
                continue;
            }

            let entries: Vec<CatalogEntry> = datasets
                .iter()
                .filter(|ds| {
                    ds.yaml_path.starts_with(catalog_dir)
                })
                .map(|ds| ds.to_entry(catalog_dir))
                .collect();

            if entries.is_empty() {
                continue;
            }

            // Generate knn_entries.yaml FIRST — catalog files must be
            // newer for publish pre-flight mtime checks.
            let knn_path = catalog_dir.join("knn_entries.yaml");
            let knn_content = generate_knn_entries(&datasets, catalog_dir, &scan_root);
            if !knn_content.is_empty() {
                if let Err(e) = std::fs::write(&knn_path, &knn_content) {
                    ctx.ui.log(&format!("  warning: failed to write {}: {}", knn_path.display(), e));
                } else {
                    produced.push(knn_path);
                    total_files += 1;
                }
            }

            let json_path = catalog_dir.join(format!("{}.json", basename));
            let json = match serde_json::to_string_pretty(&entries) {
                Ok(j) => j,
                Err(e) => return error_result(format!("JSON serialization failed: {}", e), start),
            };
            if let Err(e) = std::fs::write(&json_path, &json) {
                return error_result(format!("failed to write {}: {}", json_path.display(), e), start);
            }

            let yaml_path = catalog_dir.join(format!("{}.yaml", basename));
            let yaml = match serde_yaml::to_string(&entries) {
                Ok(y) => y,
                Err(e) => return error_result(format!("YAML serialization failed: {}", e), start),
            };
            if let Err(e) = std::fs::write(&yaml_path, &yaml) {
                return error_result(format!("failed to write {}: {}", yaml_path.display(), e), start);
            }

            ctx.ui.log(&format!("  wrote {} entries to {} (knn_entries.yaml + catalog.json + catalog.yaml)", entries.len(), catalog_dir.display()));
            produced.push(json_path);
            produced.push(yaml_path);
            total_files += 2;
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} datasets cataloged across {} directories ({} files written)",
                datasets.len(), catalog_dirs.len(), total_files,
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".into(),
                type_name: "Path".into(),
                required: false,
                default: Some(".".into()),
                description: "Root directory to scan for datasets".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "basename".into(),
                type_name: "String".into(),
                required: false,
                default: Some("catalog".into()),
                description: "Base filename for catalog files (without extension)".into(),
                role: OptionRole::Config,
            },
        ]
    }
}

// ---------------------------------------------------------------------------
// Internal helpers (duplicated from catalog/generate.rs to avoid process::exit)
// ---------------------------------------------------------------------------

const CATALOG_ROOT_FILE: &str = ".catalog_root";
const PUBLISH_URL_FILE: &str = ".publish_url";
const DO_NOT_CATALOG_FILE: &str = ".do_not_catalog";

/// Walk up from `dir` to find the catalog boundary — the closest ancestor
/// (or self) containing `.catalog_root` or `.publish_url`. This is the
/// top of the catalog hierarchy for a pipeline-driven catalog rebuild.
fn find_catalog_boundary(dir: &Path) -> Option<PathBuf> {
    let mut current = dir.to_path_buf();
    loop {
        if current.join(CATALOG_ROOT_FILE).is_file()
            || current.join(PUBLISH_URL_FILE).is_file()
        {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

struct DiscoveredDataset {
    yaml_path: PathBuf,
    config: DatasetConfig,
}

impl DiscoveredDataset {
    fn to_entry(&self, catalog_dir: &Path) -> CatalogEntry {
        let rel_path = self.yaml_path.strip_prefix(catalog_dir)
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|_| self.yaml_path.clone());
        CatalogEntry {
            name: self.config.name.clone(),
            path: rel_path.to_string_lossy().to_string(),
            dataset_type: "dataset.yaml".to_string(),
            layout: CatalogLayout {
                attributes: self.config.attributes.clone(),
                profiles: self.config.profiles.clone(),
            },
        }
    }
}

fn walk_for_datasets(dir: &Path, datasets: &mut Vec<DiscoveredDataset>) {
    if dir.join(DO_NOT_CATALOG_FILE).exists() {
        return;
    }

    let yaml_path = dir.join("dataset.yaml");
    if yaml_path.exists() {
        match DatasetConfig::load(&yaml_path) {
            Ok(config) => {
                datasets.push(DiscoveredDataset { yaml_path, config });
            }
            Err(e) => {
                log::warn!("failed to load {}: {}", yaml_path.display(), e);
            }
        }
        return;
    }

    if let Ok(read_dir) = std::fs::read_dir(dir) {
        let mut subdirs: Vec<PathBuf> = read_dir.flatten()
            .filter(|e| e.path().is_dir())
            .map(|e| e.path())
            .collect();
        subdirs.sort();
        for subdir in subdirs {
            let name = subdir.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if veks_core::filters::is_excluded_dir(name) {
                continue;
            }
            walk_for_datasets(&subdir, datasets);
        }
    }
}

/// Generate knn_entries.yaml content in jvector-compatible format.
///
/// Produces one entry per dataset:profile pair with `base`, `query`, `gt`
/// filenames relative to the dataset directory. The `_defaults.base_url`
/// is derived from the .publish_url if available.
fn generate_knn_entries(
    datasets: &[DiscoveredDataset],
    catalog_dir: &Path,
    scan_root: &Path,
) -> String {
    let mut out = String::new();

    // Derive base_url from .publish_url if present
    let publish_url = find_publish_url(scan_root);
    if let Some(ref url) = publish_url {
        out.push_str("_defaults:\n");
        out.push_str(&format!("  base_url: {}\n\n", url.trim_end_matches('/')));
    }

    for ds in datasets {
        if !ds.yaml_path.starts_with(catalog_dir) {
            continue;
        }

        let ds_dir = ds.yaml_path.parent().unwrap_or(Path::new("."));
        let ds_rel = ds_dir.strip_prefix(catalog_dir)
            .map(|r| r.to_string_lossy().to_string())
            .unwrap_or_default();
        let ds_prefix = if ds_rel.is_empty() { String::new() } else { format!("{}/", ds_rel) };

        // Load profiles from the dataset config
        let config = &ds.config;

        for (profile_name, profile) in &config.profiles.profiles {
            let entry_name = format!("{}:{}", config.name, profile_name);

            // Find base_vectors, query_vectors, neighbor_indices view paths
            let base_path = profile.view("base_vectors").map(|v| v.path());
            let query_path = profile.view("query_vectors").map(|v| v.path());
            let gt_path = profile.view("neighbor_indices").map(|v| v.path());

            // Skip profiles without the essential BQG facets
            if base_path.is_none() || query_path.is_none() || gt_path.is_none() {
                continue;
            }

            // Quote the entry name since it contains ':'
            out.push_str(&format!("\"{}\":\n", entry_name));
            out.push_str(&format!("  base: {}{}\n", ds_prefix, base_path.unwrap()));
            out.push_str(&format!("  query: {}{}\n", ds_prefix, query_path.unwrap()));
            out.push_str(&format!("  gt: {}{}\n", ds_prefix, gt_path.unwrap()));
            out.push('\n');
        }
    }

    out
}

/// Find .publish_url content by walking up from the given directory.
fn find_publish_url(dir: &Path) -> Option<String> {
    let mut current = dir.to_path_buf();
    loop {
        let candidate = current.join(PUBLISH_URL_FILE);
        if candidate.is_file() {
            if let Ok(content) = std::fs::read_to_string(&candidate) {
                let url = content.lines()
                    .map(|l| l.trim())
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .collect::<String>();
                if !url.is_empty() {
                    return Some(url);
                }
            }
        }
        if !current.pop() {
            return None;
        }
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}
