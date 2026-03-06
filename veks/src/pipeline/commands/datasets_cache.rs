// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: list locally cached datasets.
//!
//! Scans the local cache directory for dataset directories that contain
//! `dataset.yaml` and reports their status.
//!
//! Equivalent to the Java `CMD_datasets_cache` command.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: list locally cached datasets.
pub struct DatasetsCacheOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(DatasetsCacheOp)
}

impl CommandOp for DatasetsCacheOp {
    fn command_path(&self) -> &str {
        "datasets cache"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Manage the dataset download cache".into(),
            body: format!(
                "# datasets cache\n\n\
                 Manage the dataset download cache.\n\n\
                 ## Description\n\n\
                 Scans the local cache directory for dataset directories that contain \
                 `dataset.yaml` and reports their status, file counts, and sizes.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let cache_dir = options
            .get("cache-dir")
            .map(|s| resolve_path(s, &ctx.workspace))
            .unwrap_or_else(default_cache_dir);

        let verbose = options.get("verbose").map_or(false, |s| s == "true");

        if !cache_dir.exists() {
            ctx.display.log(&format!("Cache directory does not exist: {}", cache_dir.display()));
            return CommandResult {
                status: Status::Ok,
                message: "no cache directory".to_string(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        ctx.display.log(&format!("Cache directory: {}", cache_dir.display()));
        ctx.display.log("");

        let mut datasets = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let yaml = path.join("dataset.yaml");
                    if yaml.exists() {
                        datasets.push(path);
                    }
                }
            }
        }

        datasets.sort();

        if datasets.is_empty() {
            ctx.display.log("No datasets found in cache.");
        } else {
            ctx.display.log(&format!(
                "{:<30} {:>6} {:>12}",
                "Dataset", "Files", "Size"
            ));
            ctx.display.log(&format!("{}", "-".repeat(52)));

            for ds_dir in &datasets {
                let name = ds_dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?");
                let (file_count, total_size) = scan_directory(ds_dir);

                if verbose {
                    ctx.display.log(&format!("{:<30} {:>6} {:>12}", name, file_count, format_size(total_size)));
                    // List facet files
                    if let Ok(files) = std::fs::read_dir(ds_dir) {
                        for f in files.flatten() {
                            let fp = f.path();
                            if fp.is_file() && fp.file_name().and_then(|n| n.to_str()) != Some("dataset.yaml") {
                                let fname = fp.file_name().and_then(|n| n.to_str()).unwrap_or("?");
                                let fsize = std::fs::metadata(&fp).map(|m| m.len()).unwrap_or(0);
                                ctx.display.log(&format!("  {:<28} {:>12}", fname, format_size(fsize)));
                            }
                        }
                    }
                } else {
                    ctx.display.log(&format!("{:<30} {:>6} {:>12}", name, file_count, format_size(total_size)));
                }
            }
        }

        ctx.display.log("");
        ctx.display.log(&format!("{} dataset(s) cached", datasets.len()));

        CommandResult {
            status: Status::Ok,
            message: format!("{} datasets cached", datasets.len()),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "cache-dir".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Override cache directory location".to_string(),
            },
            OptionDesc {
                name: "verbose".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Show per-file details".to_string(),
            },
        ]
    }
}

/// Default cache directory location.
fn default_cache_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".config").join("vectordata").join("cache")
    } else {
        PathBuf::from("/tmp/vectordata/cache")
    }
}

/// Scan a directory and return (file_count, total_bytes).
fn scan_directory(dir: &Path) -> (usize, u64) {
    let mut count = 0;
    let mut size = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                count += 1;
                size += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    (count, size)
}

/// Format bytes as human-readable size.
fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
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
    fn test_cache_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let cache = ws.join("cache");
        std::fs::create_dir(&cache).unwrap();
        let mut ctx = test_ctx(ws);

        let mut opts = Options::new();
        opts.set("cache-dir", cache.to_string_lossy().to_string());
        let mut op = DatasetsCacheOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("0 datasets"));
    }

    #[test]
    fn test_cache_with_dataset() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let cache = ws.join("cache");
        let ds = cache.join("my-dataset");
        std::fs::create_dir_all(&ds).unwrap();
        std::fs::write(ds.join("dataset.yaml"), "name: my-dataset\nprofiles: {}\n").unwrap();
        std::fs::write(ds.join("base_vectors.fvec"), &[0u8; 100]).unwrap();
        let mut ctx = test_ctx(ws);

        let mut opts = Options::new();
        opts.set("cache-dir", cache.to_string_lossy().to_string());
        let mut op = DatasetsCacheOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("1 datasets"));
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(1048576), "1.0 MB");
    }
}
