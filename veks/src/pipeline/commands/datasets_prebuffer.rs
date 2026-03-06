// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: download and cache dataset facets locally.
//!
//! Reads a dataset.yaml, downloads facets from their upstream URLs into
//! the local cache directory, and reports progress.
//!
//! Equivalent to the Java `CMD_datasets_prebuffer` command.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: prebuffer (download) dataset facets.
pub struct DatasetsPrebufferOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(DatasetsPrebufferOp)
}

impl CommandOp for DatasetsPrebufferOp {
    fn command_path(&self) -> &str {
        "datasets prebuffer"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Pre-load dataset files into page cache".into(),
            body: format!(
                "# datasets prebuffer\n\n\
                 Pre-load dataset files into page cache.\n\n\
                 ## Description\n\n\
                 Reads a dataset.yaml, downloads facets from their upstream URLs into \
                 the local cache directory, and reports progress.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "readahead".into(), description: "Page cache prefetch window".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let path_str = options.get("path").unwrap_or(".");
        let path = resolve_path(path_str, &ctx.workspace);

        let cache_dir = options
            .get("cache-dir")
            .map(|s| resolve_path(s, &ctx.workspace))
            .unwrap_or_else(|| {
                if let Ok(home) = std::env::var("HOME") {
                    PathBuf::from(home).join(".config/vectordata/cache")
                } else {
                    PathBuf::from("/tmp/vectordata/cache")
                }
            });

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
            Err(e) => return error_result(format!("failed to read: {}", e), start),
        };

        let config: serde_yaml::Value = match serde_yaml::from_str(&content) {
            Ok(v) => v,
            Err(e) => return error_result(format!("failed to parse: {}", e), start),
        };

        let dataset_name = config
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("dataset");

        let ds_cache = cache_dir.join(dataset_name);
        if let Err(e) = std::fs::create_dir_all(&ds_cache) {
            return error_result(format!("failed to create cache dir: {}", e), start);
        }

        // Copy dataset.yaml to cache
        let cached_yaml = ds_cache.join("dataset.yaml");
        if let Err(e) = std::fs::copy(&yaml_path, &cached_yaml) {
            ctx.display.log(&format!("  warning: failed to copy dataset.yaml: {}", e));
        }

        let mut downloaded = 0u32;
        let mut skipped = 0u32;
        let mut failed = 0u32;

        // Process views from the default profile
        if let Some(profiles) = config.get("profiles").and_then(|v| v.as_mapping()) {
            let default_key = serde_yaml::Value::String("default".to_string());
            let profile = profiles.get(&default_key).or_else(|| profiles.values().next());
            if let Some(profile_map) = profile.and_then(|v| v.as_mapping()) {
                for (name, entry) in profile_map {
                    let view_name = name.as_str().unwrap_or("?");
                    if view_name == "maxk" {
                        continue;
                    }

                    // Extract source path from view entry (string or map)
                    let rel_path = if let Some(s) = entry.as_str() {
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

                    let target = ds_cache.join(&rel_path);

                    // Check if already cached
                    if target.exists() {
                        let size = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
                        ctx.display.log(&format!("  {} — already cached ({} bytes)", view_name, size));
                        skipped += 1;
                        continue;
                    }

                    // Check if source is a URL
                    if rel_path.starts_with("http://") || rel_path.starts_with("https://") {
                        ctx.display.log(&format!("  {} — downloading from {}", view_name, rel_path));
                        match download_file(&rel_path, &target) {
                            Ok(size) => {
                                ctx.display.log(&format!("  {} — downloaded {} bytes", view_name, size));
                                downloaded += 1;
                            }
                            Err(e) => {
                                ctx.display.log(&format!("  {} — FAILED: {}", view_name, e));
                                failed += 1;
                            }
                        }
                    } else {
                        // Try copying from local source
                        let dataset_dir = yaml_path.parent().unwrap_or(Path::new("."));
                        let local_src = dataset_dir.join(&rel_path);
                        if local_src.exists() && local_src != target {
                            if let Err(e) = std::fs::copy(&local_src, &target) {
                                ctx.display.log(&format!("  {} — FAILED to copy: {}", view_name, e));
                                failed += 1;
                            } else {
                                ctx.display.log(&format!("  {} — copied from local source", view_name));
                                downloaded += 1;
                            }
                        } else {
                            ctx.display.log(&format!("  {} — no source available", view_name));
                        }
                    }
                }
            }
        }

        ctx.display.log("");
        ctx.display.log(&format!(
            "Prebuffer: {} downloaded, {} skipped, {} failed",
            downloaded, skipped, failed
        ));

        let status = if failed > 0 {
            Status::Error
        } else {
            Status::Ok
        };

        CommandResult {
            status,
            message: format!(
                "{} downloaded, {} skipped, {} failed",
                downloaded, skipped, failed
            ),
            produced: vec![ds_cache],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "path".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: Some(".".to_string()),
                description: "Dataset directory or dataset.yaml path".to_string(),
            },
            OptionDesc {
                name: "cache-dir".to_string(),
                type_name: "Path".to_string(),
                required: false,
                default: None,
                description: "Override cache directory location".to_string(),
            },
        ]
    }
}

/// Download a URL to a local file using the curl crate.
fn download_file(url: &str, dest: &Path) -> Result<u64, String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create dir: {}", e))?;
    }

    let mut file = std::fs::File::create(dest)
        .map_err(|e| format!("failed to create {}: {}", dest.display(), e))?;

    let mut easy = curl::easy::Easy::new();
    easy.url(url).map_err(|e| format!("invalid URL: {}", e))?;
    easy.follow_location(true).ok();
    easy.fail_on_error(true).ok();

    let mut transfer = easy.transfer();
    transfer
        .write_function(|data| {
            file.write_all(data).map_or(Ok(0), |()| Ok(data.len()))
        })
        .map_err(|e| format!("transfer setup error: {}", e))?;
    transfer
        .perform()
        .map_err(|e| format!("download failed: {}", e))?;
    drop(transfer);

    let size = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
    Ok(size)
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
    fn test_prebuffer_local_copy() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Create source file and dataset.yaml
        std::fs::write(ws.join("data.fvec"), &[0u8; 100]).unwrap();
        let yaml = "name: test\nprofiles:\n  default:\n    base_vectors: data.fvec\n";
        std::fs::write(ws.join("dataset.yaml"), yaml).unwrap();

        let cache = ws.join("cache");
        let mut opts = Options::new();
        opts.set("path", ws.to_string_lossy().to_string());
        opts.set("cache-dir", cache.to_string_lossy().to_string());
        let mut op = DatasetsPrebufferOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "prebuffer failed: {}", result.message);

        // Verify cache structure
        assert!(cache.join("test").join("dataset.yaml").exists());
        assert!(cache.join("test").join("data.fvec").exists());
    }
}
