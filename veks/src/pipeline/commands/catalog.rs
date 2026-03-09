// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate catalog files from dataset directories.
//!
//! Recursively walks directory trees, locates `dataset.yaml` files, and
//! writes `catalog.json` and `catalog.yaml` index files.
//!
//! Equivalent to the Java `CMD_catalog` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: generate catalog from dataset directories.
pub struct CatalogGenerateOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(CatalogGenerateOp)
}

/// A catalog entry representing one discovered dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CatalogEntry {
    name: String,
    path: String,
    description: Option<String>,
    views: Vec<String>,
}

impl CommandOp for CatalogGenerateOp {
    fn command_path(&self) -> &str {
        "catalog generate"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate a dataset catalog index".into(),
            body: format!(
                "# catalog generate\n\n\
                 Generate a dataset catalog index.\n\n\
                 ## Description\n\n\
                 Recursively walks directory trees, locates `dataset.yaml` files, and \
                 writes `catalog.json` and `catalog.yaml` index files.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = options.get("input").unwrap_or(".");
        let input_path = resolve_path(input_str, &ctx.workspace);
        let basename = options.get("basename").unwrap_or("catalog");

        if !input_path.is_dir() {
            return error_result(
                format!("{} is not a directory", input_path.display()),
                start,
            );
        }

        ctx.ui.log(&format!("Scanning {} for datasets...", input_path.display()));

        // Walk directory tree and find dataset.yaml files
        let mut entries: Vec<CatalogEntry> = Vec::new();
        walk_for_datasets(&input_path, &input_path, &mut entries);

        entries.sort_by(|a, b| a.path.cmp(&b.path));

        ctx.ui.log(&format!("Found {} dataset(s)", entries.len()));

        if entries.is_empty() {
            return CommandResult {
                status: Status::Warning,
                message: "no datasets found".to_string(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        for entry in &entries {
            ctx.ui.log(&format!(
                "  {} — {} view(s) [{}]",
                entry.name,
                entry.views.len(),
                entry.path
            ));
        }

        // Write catalog.json
        let json_path = input_path.join(format!("{}.json", basename));
        let json = match serde_json::to_string_pretty(&entries) {
            Ok(j) => j,
            Err(e) => return error_result(format!("JSON serialization error: {}", e), start),
        };
        if let Err(e) = std::fs::write(&json_path, &json) {
            return error_result(format!("failed to write {}: {}", json_path.display(), e), start);
        }

        // Write catalog.yaml
        let yaml_path = input_path.join(format!("{}.yaml", basename));
        let yaml = match serde_yaml::to_string(&entries) {
            Ok(y) => y,
            Err(e) => return error_result(format!("YAML serialization error: {}", e), start),
        };
        if let Err(e) = std::fs::write(&yaml_path, &yaml) {
            return error_result(format!("failed to write {}: {}", yaml_path.display(), e), start);
        }

        ctx.ui.log(&format!(
            "Wrote {} entries to {} and {}",
            entries.len(),
            json_path.display(),
            yaml_path.display()
        ));

        CommandResult {
            status: Status::Ok,
            message: format!("{} datasets cataloged", entries.len()),
            produced: vec![json_path, yaml_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "input".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: Some(".".to_string()),
                description: "Root directory to scan for datasets".to_string(),
            },
            OptionDesc {
                name: "basename".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("catalog".to_string()),
                description: "Base filename for catalog files (without extension)".to_string(),
            },
        ]
    }
}

/// Recursively walk directories to find dataset.yaml files.
fn walk_for_datasets(dir: &Path, root: &Path, entries: &mut Vec<CatalogEntry>) {
    let yaml_path = dir.join("dataset.yaml");
    if yaml_path.exists() {
        if let Some(entry) = load_dataset_entry(&yaml_path, root) {
            entries.push(entry);
        }
        // Don't descend into dataset directories
        return;
    }

    if let Ok(read_dir) = std::fs::read_dir(dir) {
        let mut subdirs: Vec<PathBuf> = read_dir
            .flatten()
            .filter(|e| e.path().is_dir())
            .map(|e| e.path())
            .collect();
        subdirs.sort();

        for subdir in subdirs {
            // Skip hidden directories and common non-dataset dirs
            let name = subdir.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') || name == "node_modules" || name == "target" {
                continue;
            }
            walk_for_datasets(&subdir, root, entries);
        }
    }
}

/// Load a dataset.yaml and create a catalog entry.
fn load_dataset_entry(yaml_path: &Path, root: &Path) -> Option<CatalogEntry> {
    let content = std::fs::read_to_string(yaml_path).ok()?;
    let config: serde_yaml::Value = serde_yaml::from_str(&content).ok()?;

    let name = config
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let description = config
        .get("description")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Extract view names from profiles (default profile's keys)
    let mut views = Vec::new();
    if let Some(profiles_map) = config.get("profiles").and_then(|v| v.as_mapping()) {
        // Use default profile's keys as views, or first profile if no default
        let profile = profiles_map
            .get(&serde_yaml::Value::String("default".to_string()))
            .or_else(|| profiles_map.values().next());
        if let Some(profile_val) = profile.and_then(|v| v.as_mapping()) {
            for key in profile_val.keys() {
                if let Some(name) = key.as_str() {
                    if name != "maxk" {
                        views.push(name.to_string());
                    }
                }
            }
        }
    }
    views.sort();

    let rel_path = yaml_path
        .parent()?
        .strip_prefix(root)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| yaml_path.parent().unwrap().to_string_lossy().to_string());

    Some(CatalogEntry {
        name,
        path: rel_path,
        description,
        views,
    })
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: crate::ui::UiHandle::new(std::sync::Arc::new(crate::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
        }
    }

    #[test]
    fn test_catalog_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let mut opts = Options::new();
        opts.set("input", ws.to_string_lossy().to_string());
        let mut op = CatalogGenerateOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Warning);
        assert!(result.message.contains("no datasets"));
    }

    #[test]
    fn test_catalog_with_datasets() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Create two dataset directories
        let ds1 = ws.join("dataset-a");
        std::fs::create_dir(&ds1).unwrap();
        std::fs::write(
            ds1.join("dataset.yaml"),
            "name: alpha\ndescription: First\nprofiles:\n  default:\n    base_vectors: base.fvec\n",
        ).unwrap();

        let ds2 = ws.join("dataset-b");
        std::fs::create_dir(&ds2).unwrap();
        std::fs::write(
            ds2.join("dataset.yaml"),
            "name: beta\nprofiles:\n  default:\n    base_vectors: base.fvec\n    query_vectors: query.fvec\n",
        ).unwrap();

        let mut opts = Options::new();
        opts.set("input", ws.to_string_lossy().to_string());
        let mut op = CatalogGenerateOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "catalog failed: {}", result.message);
        assert!(result.message.contains("2 datasets"));

        // Verify output files exist
        assert!(ws.join("catalog.json").exists());
        assert!(ws.join("catalog.yaml").exists());

        // Parse and verify JSON
        let json_str = std::fs::read_to_string(ws.join("catalog.json")).unwrap();
        let entries: Vec<CatalogEntry> = serde_json::from_str(&json_str).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "alpha");
        assert_eq!(entries[1].name, "beta");
        assert_eq!(entries[1].views.len(), 2);
    }
}
