// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `veks datasets prebuffer` — download and cache dataset facets locally.

use std::io::Write;
use std::path::{Path, PathBuf};

pub fn run(path: &Path, cache_dir: Option<&Path>) {
    let yaml_path = if path.is_file() {
        path.to_path_buf()
    } else {
        path.join("dataset.yaml")
    };

    if !yaml_path.exists() {
        println!("dataset.yaml not found at {}", yaml_path.display());
        std::process::exit(1);
    }

    let content = std::fs::read_to_string(&yaml_path).unwrap_or_else(|e| {
        println!("Failed to read: {}", e);
        std::process::exit(1);
    });

    let config: serde_yaml::Value = serde_yaml::from_str(&content).unwrap_or_else(|e| {
        println!("Failed to parse: {}", e);
        std::process::exit(1);
    });

    let dataset_name = config
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("dataset");

    let cache = cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_cache_dir);

    let ds_cache = cache.join(dataset_name);
    std::fs::create_dir_all(&ds_cache).unwrap_or_else(|e| {
        println!("Failed to create cache dir: {}", e);
        std::process::exit(1);
    });

    // Copy dataset.yaml to cache
    let cached_yaml = ds_cache.join("dataset.yaml");
    if let Err(e) = std::fs::copy(&yaml_path, &cached_yaml) {
        println!("  warning: failed to copy dataset.yaml: {}", e);
    }

    let mut downloaded = 0u32;
    let mut skipped = 0u32;
    let mut failed = 0u32;

    if let Some(profiles) = config.get("profiles").and_then(|v| v.as_mapping()) {
        let default_key = serde_yaml::Value::String("default".to_string());
        let profile = profiles
            .get(&default_key)
            .or_else(|| profiles.values().next());
        if let Some(profile_map) = profile.and_then(|v| v.as_mapping()) {
            for (name, entry) in profile_map {
                let view_name = name.as_str().unwrap_or("?");
                if view_name == "maxk" {
                    continue;
                }

                let rel_path = extract_view_path(entry);
                if rel_path.is_empty() {
                    continue;
                }

                let target = ds_cache.join(&rel_path);

                if target.exists() {
                    let size = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
                    println!("  {} — already cached ({} bytes)", view_name, size);
                    skipped += 1;
                    continue;
                }

                if rel_path.starts_with("http://") || rel_path.starts_with("https://") {
                    println!("  {} — downloading from {}", view_name, rel_path);
                    match download_file(&rel_path, &target) {
                        Ok(size) => {
                            println!("  {} — downloaded {} bytes", view_name, size);
                            downloaded += 1;
                        }
                        Err(e) => {
                            println!("  {} — FAILED: {}", view_name, e);
                            failed += 1;
                        }
                    }
                } else {
                    let dataset_dir = yaml_path.parent().unwrap_or(Path::new("."));
                    let local_src = dataset_dir.join(&rel_path);
                    if local_src.exists() && local_src != target {
                        if let Err(e) = std::fs::copy(&local_src, &target) {
                            println!("  {} — FAILED to copy: {}", view_name, e);
                            failed += 1;
                        } else {
                            println!("  {} — copied from local source", view_name);
                            downloaded += 1;
                        }
                    } else {
                        println!("  {} — no source available", view_name);
                    }
                }
            }
        }
    }

    println!();
    println!(
        "Prebuffer: {} downloaded, {} skipped, {} failed",
        downloaded, skipped, failed
    );

    if failed > 0 {
        std::process::exit(1);
    }
}

fn extract_view_path(entry: &serde_yaml::Value) -> String {
    if let Some(s) = entry.as_str() {
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
        String::new()
    }
}

fn default_cache_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".config/vectordata/cache")
    } else {
        PathBuf::from("/tmp/vectordata/cache")
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prebuffer_local_copy() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();

        std::fs::write(ws.join("data.fvec"), &[0u8; 100]).unwrap();
        let yaml = "name: test\nprofiles:\n  default:\n    base_vectors: data.fvec\n";
        std::fs::write(ws.join("dataset.yaml"), yaml).unwrap();

        let cache = ws.join("cache");
        run(ws, Some(&cache));

        assert!(cache.join("test").join("dataset.yaml").exists());
        assert!(cache.join("test").join("data.fvec").exists());
    }
}
