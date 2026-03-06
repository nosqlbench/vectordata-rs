// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: list available datasets from a catalog or directory.
//!
//! Scans a directory for `dataset.yaml` files and lists them with their
//! metadata (name, description, facets, profiles).
//!
//! Supports output formats: text (default), json, yaml.
//!
//! Equivalent to the Java `CMD_datasets_list` command (simplified: local
//! directory scanning only, no remote catalog support).

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::import::dataset::DatasetConfig;
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: list datasets.
pub struct DatasetsListOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(DatasetsListOp)
}

/// Summary info for a discovered dataset.
#[derive(Debug, serde::Serialize)]
struct DatasetSummary {
    name: String,
    path: String,
    description: Option<String>,
    views: Vec<String>,
    profile_count: usize,
    has_pipeline: bool,
}

impl CommandOp for DatasetsListOp {
    fn command_path(&self) -> &str {
        "datasets list"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "List available datasets from configured sources".into(),
            body: format!(
                "# datasets list\n\n\
                 List available datasets from configured sources.\n\n\
                 ## Description\n\n\
                 Scans a directory for `dataset.yaml` files and lists them with their \
                 metadata (name, description, facets, profiles). Supports text, JSON, \
                 and YAML output formats.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let workspace_str = ctx.workspace.to_string_lossy().to_string();
        let catalog_str = options
            .get("catalog")
            .unwrap_or(&workspace_str);

        let format = options.get("format").unwrap_or("text");
        let verbose = options.get("verbose").map_or(false, |s| s == "true");

        let catalog_path = resolve_path(catalog_str, &ctx.workspace);

        if !catalog_path.is_dir() {
            return error_result(
                format!("{} is not a directory", catalog_path.display()),
                start,
            );
        }

        // Find all dataset.yaml files
        let datasets = match find_datasets(&catalog_path) {
            Ok(d) => d,
            Err(e) => return error_result(e, start),
        };

        match format {
            "json" => {
                let json = serde_json::to_string_pretty(&datasets).unwrap_or_default();
                ctx.display.log(&format!("{}", json));
            }
            "yaml" => {
                let yaml = serde_yaml::to_string(&datasets).unwrap_or_default();
                ctx.display.log(&format!("{}", yaml));
            }
            _ => {
                // Text format
                if datasets.is_empty() {
                    ctx.display.log(&format!("No datasets found in {}", catalog_path.display()));
                } else {
                    ctx.display.log(&format!("Datasets in {}:", catalog_path.display()));
                    ctx.display.log("");
                    for ds in &datasets {
                        ctx.display.log(&format!("  {}", ds.name));
                        if verbose {
                            ctx.display.log(&format!("    Path: {}", ds.path));
                            if let Some(ref desc) = ds.description {
                                ctx.display.log(&format!("    Description: {}", desc));
                            }
                            ctx.display.log(&format!("    Views: {}", ds.views.join(", ")));
                            ctx.display.log(&format!("    Profiles: {}", ds.profile_count));
                            ctx.display.log(&format!(
                                "    Pipeline: {}",
                                if ds.has_pipeline { "yes" } else { "no" }
                            ));
                            ctx.display.log("");
                        }
                    }
                    if !verbose {
                        ctx.display.log("");
                        ctx.display.log(&format!("  {} datasets found. Use verbose=true for details.", datasets.len()));
                    }
                }
            }
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "found {} datasets in {}",
                datasets.len(),
                catalog_path.display()
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "catalog".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Directory to scan for datasets (default: workspace)".to_string(),
            },
            OptionDesc {
                name: "format".to_string(),
                type_name: "enum".to_string(),
                required: false,
                default: Some("text".to_string()),
                description: "Output format: text, json, yaml".to_string(),
            },
            OptionDesc {
                name: "verbose".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Show detailed information".to_string(),
            },
        ]
    }
}

/// Recursively find dataset.yaml files and parse them.
fn find_datasets(dir: &Path) -> Result<Vec<DatasetSummary>, String> {
    let mut results = Vec::new();
    find_datasets_recursive(dir, &mut results, 0)?;
    results.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(results)
}

fn find_datasets_recursive(
    dir: &Path,
    results: &mut Vec<DatasetSummary>,
    depth: usize,
) -> Result<(), String> {
    if depth > 5 {
        return Ok(()); // Limit recursion depth
    }

    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read directory {}: {}", dir.display(), e))?;

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();

        if path.is_file() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name == "dataset.yaml" || name == "dataset.yml" {
                match DatasetConfig::load(&path) {
                    Ok(config) => {
                        let views: Vec<String> = config.view_names().into_iter().map(|s| s.to_string()).collect();
                        let profile_count = config.profile_names().len();
                        results.push(DatasetSummary {
                            name: config.name.clone(),
                            path: path.to_string_lossy().to_string(),
                            description: config.description.clone(),
                            views,
                            profile_count,
                            has_pipeline: config.upstream.is_some(),
                        });
                    }
                    Err(e) => {
                        eprintln!(
                            "  Warning: failed to parse {}: {}",
                            path.display(),
                            e
                        );
                    }
                }
            }
        } else if path.is_dir() {
            // Skip hidden directories and common non-dataset directories
            let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !dir_name.starts_with('.') && dir_name != "target" && dir_name != "node_modules" {
                find_datasets_recursive(&path, results, depth + 1)?;
            }
        }
    }

    Ok(())
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            display: crate::pipeline::display::ProgressDisplay::new(),
        }
    }

    #[test]
    fn test_datasets_list_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let opts = Options::new();
        let mut op = DatasetsListOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("0 datasets"));
    }

    #[test]
    fn test_datasets_list_with_dataset() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Create a minimal dataset.yaml
        let yaml = r#"
name: test-dataset
description: A test dataset
profiles:
  default:
    base_vectors: base.fvec
    query_vectors: query.fvec
"#;
        std::fs::write(ws.join("dataset.yaml"), yaml).unwrap();

        let opts = Options::new();
        let mut op = DatasetsListOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("1 datasets"));
    }

    #[test]
    fn test_datasets_list_json_format() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let yaml = r#"
name: json-test
profiles:
  default:
    base_vectors: vectors.fvec
"#;
        std::fs::write(ws.join("dataset.yaml"), yaml).unwrap();

        let mut opts = Options::new();
        opts.set("format", "json");
        let mut op = DatasetsListOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_datasets_list_nested() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Create nested dataset
        let subdir = ws.join("sub");
        std::fs::create_dir(&subdir).unwrap();
        let yaml = r#"
name: nested-dataset
profiles:
  default:
    base_vectors: data.fvec
"#;
        std::fs::write(subdir.join("dataset.yaml"), yaml).unwrap();

        let opts = Options::new();
        let mut op = DatasetsListOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("1 datasets"));
    }
}
