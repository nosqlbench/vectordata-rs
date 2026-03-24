// Copyright (c) DataStax, Inc.
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

        let input_str = options.get("input").unwrap_or(".");
        let basename = options.get("basename").unwrap_or("catalog");

        let input_path = if Path::new(input_str).is_absolute() {
            PathBuf::from(input_str)
        } else {
            ctx.workspace.join(input_str)
        };

        if !input_path.is_dir() {
            return error_result(format!("{} is not a directory", input_path.display()), start);
        }

        // Use .catalog_root if present, otherwise scan from input
        let scan_root = find_catalog_root(&input_path).unwrap_or(input_path.clone());

        ctx.ui.log(&format!("  scanning {} for datasets...", scan_root.display()));

        let mut datasets = Vec::new();
        walk_for_datasets(&scan_root, &mut datasets);
        datasets.sort_by(|a: &DiscoveredDataset, b: &DiscoveredDataset| a.yaml_path.cmp(&b.yaml_path));

        if datasets.is_empty() {
            return CommandResult {
                status: Status::Ok,
                message: "no datasets found — no catalogs written".into(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        ctx.ui.log(&format!("  found {} dataset(s)", datasets.len()));

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

        for catalog_dir in &catalog_dirs {
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

            ctx.ui.log(&format!("  wrote {} entries to {}", entries.len(), catalog_dir.display()));
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
                name: "input".into(),
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
const DO_NOT_CATALOG_FILE: &str = ".do_not_catalog";

fn find_catalog_root(dir: &Path) -> Option<PathBuf> {
    let mut current = dir.to_path_buf();
    loop {
        if current.join(CATALOG_ROOT_FILE).is_file() {
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

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}
