// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: analyze a dataset.yaml and suggest commands for missing facets.
//!
//! Reads a dataset configuration, checks which referenced files exist, and
//! generates command suggestions to create the missing ones.
//!
//! Equivalent to the Java `CMD_datasets_plan` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
};

/// Pipeline command: plan missing facets for a dataset.
pub struct DatasetsPlanOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(DatasetsPlanOp)
}

impl CommandOp for DatasetsPlanOp {
    fn command_path(&self) -> &str {
        "datasets plan"
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let path_str = options.get("path").unwrap_or(".");
        let path = resolve_path(path_str, &ctx.workspace);

        // Find dataset.yaml
        let yaml_path = if path.is_file() {
            path.clone()
        } else {
            path.join("dataset.yaml")
        };

        if !yaml_path.exists() {
            return error_result(
                format!("dataset.yaml not found at {}", yaml_path.display()),
                start,
            );
        }

        let content = match std::fs::read_to_string(&yaml_path) {
            Ok(s) => s,
            Err(e) => {
                return error_result(
                    format!("failed to read {}: {}", yaml_path.display(), e),
                    start,
                )
            }
        };

        let config: serde_yaml::Value = match serde_yaml::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                return error_result(
                    format!("failed to parse {}: {}", yaml_path.display(), e),
                    start,
                )
            }
        };

        let dataset_dir = yaml_path.parent().unwrap_or(Path::new("."));
        let dataset_name = config
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        eprintln!("Dataset: {} ({})", dataset_name, yaml_path.display());
        eprintln!();

        let mut missing = Vec::new();
        let mut present = Vec::new();

        // Check views from the default profile
        if let Some(profiles) = config.get("profiles").and_then(|v| v.as_mapping()) {
            let default_key = serde_yaml::Value::String("default".to_string());
            let profile = profiles.get(&default_key).or_else(|| profiles.values().next());
            if let Some(profile_map) = profile.and_then(|v| v.as_mapping()) {
                for (name, entry) in profile_map {
                    let view_name = name.as_str().unwrap_or("?");
                    if view_name == "maxk" {
                        continue;
                    }
                    // View entries can be bare strings or maps with a source/path key
                    let rel_path = if let Some(s) = entry.as_str() {
                        // Strip window/namespace sugar for path check
                        let s = s.split('[').next().unwrap_or(s);
                        let s = s.split('(').next().unwrap_or(s);
                        s.to_string()
                    } else if let Some(m) = entry.as_mapping() {
                        m.get(&serde_yaml::Value::String("source".to_string()))
                            .or_else(|| m.get(&serde_yaml::Value::String("path".to_string())))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string()
                    } else {
                        continue;
                    };
                    if rel_path.is_empty() {
                        continue;
                    }
                    let full_path = dataset_dir.join(&rel_path);
                    if full_path.exists() {
                        let size = std::fs::metadata(&full_path)
                            .map(|m| m.len())
                            .unwrap_or(0);
                        present.push((view_name.to_string(), rel_path, size));
                    } else {
                        missing.push((view_name.to_string(), rel_path));
                    }
                }
            }
        }

        // Report present facets
        if !present.is_empty() {
            eprintln!("Present views:");
            for (name, path, size) in &present {
                eprintln!(
                    "  {} — {} ({} bytes)",
                    name, path, size
                );
            }
            eprintln!();
        }

        // Report missing facets with suggestions
        if !missing.is_empty() {
            eprintln!("Missing views:");
            for (name, path) in &missing {
                eprintln!("  {} — {}", name, path);
            }
            eprintln!();

            // Detect dimension from existing files
            let dim = detect_dimension(dataset_dir, &present);

            eprintln!("Suggested commands:");
            for (name, path) in &missing {
                let ext = Path::new(path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                let full = dataset_dir.join(path);
                match (name.as_str(), ext) {
                    (n, "fvec") if n.contains("base") || n.contains("vector") => {
                        eprintln!(
                            "  vecs run generate vectors output={} dimension={} count=<COUNT> seed=42",
                            full.display(),
                            dim.unwrap_or(128)
                        );
                    }
                    (n, "fvec") if n.contains("query") => {
                        eprintln!(
                            "  vecs run generate vectors output={} dimension={} count=<QUERY_COUNT> seed=43",
                            full.display(),
                            dim.unwrap_or(128)
                        );
                    }
                    (n, "ivec") if n.contains("neighbor") || n.contains("indic") => {
                        eprintln!(
                            "  vecs run compute knn source=<BASE_VECTORS> queries=<QUERY_VECTORS> output-indices={} k=100 metric=EUCLIDEAN",
                            full.display()
                        );
                    }
                    (n, "fvec") if n.contains("dist") => {
                        eprintln!(
                            "  vecs run compute knn source=<BASE_VECTORS> queries=<QUERY_VECTORS> output-distances={} k=100 metric=EUCLIDEAN",
                            full.display()
                        );
                    }
                    _ => {
                        eprintln!("  # {} — no automatic suggestion for '{}'", name, path);
                    }
                }
            }
        } else {
            eprintln!("All facets are present.");
        }

        let msg = format!(
            "{} present, {} missing",
            present.len(),
            missing.len()
        );
        let status = if missing.is_empty() {
            Status::Ok
        } else {
            Status::Warning
        };

        CommandResult {
            status,
            message: msg,
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![OptionDesc {
            name: "path".to_string(),
            type_name: "Path".to_string(),
            required: false,
            default: Some(".".to_string()),
            description: "Directory containing dataset.yaml or path to the YAML file".to_string(),
        }]
    }
}

/// Detect vector dimension from existing .fvec files by reading the first 4 bytes.
fn detect_dimension(dir: &Path, present: &[(String, String, u64)]) -> Option<usize> {
    for (_, rel_path, _) in present {
        if rel_path.ends_with(".fvec") {
            let full = dir.join(rel_path);
            if let Ok(data) = std::fs::read(&full) {
                if data.len() >= 4 {
                    let dim = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
                    if dim > 0 && dim < 100_000 {
                        return Some(dim);
                    }
                }
            }
        }
    }
    None
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
        }
    }

    #[test]
    fn test_plan_missing_yaml() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let mut opts = Options::new();
        opts.set("path", ws.to_string_lossy().to_string());
        let mut op = DatasetsPlanOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("not found"));
    }

    #[test]
    fn test_plan_all_present() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Create a minimal dataset.yaml with an existing file
        std::fs::write(ws.join("test.fvec"), &[4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        let yaml = "name: test\nprofiles:\n  default:\n    base_vectors: test.fvec\n";
        std::fs::write(ws.join("dataset.yaml"), yaml).unwrap();

        let mut opts = Options::new();
        opts.set("path", ws.to_string_lossy().to_string());
        let mut op = DatasetsPlanOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("1 present"));
        assert!(result.message.contains("0 missing"));
    }

    #[test]
    fn test_plan_missing_facet() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let yaml = "name: test\nprofiles:\n  default:\n    base_vectors: missing.fvec\n";
        std::fs::write(ws.join("dataset.yaml"), yaml).unwrap();

        let mut opts = Options::new();
        opts.set("path", ws.to_string_lossy().to_string());
        let mut op = DatasetsPlanOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Warning);
        assert!(result.message.contains("1 missing"));
    }
}
