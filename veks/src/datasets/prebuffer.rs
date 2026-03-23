// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `veks datasets prebuffer` — download and cache dataset facets locally.
//!
//! Accepts either a catalog dataset specifier (`dataset:profile`) or a local
//! path to a dataset directory / `dataset.yaml`. When a catalog specifier is
//! given, the dataset is resolved through the configured catalog chain and
//! facets are downloaded from the remote source.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::catalog::resolver::Catalog;
use crate::catalog::sources::CatalogSources;

/// Entry point for `veks datasets prebuffer`.
///
/// `dataset_spec` is either:
/// - A `name:profile` pair resolved via the catalog (e.g. `sift-128:default`)
/// - A `name` resolved via the catalog (uses the default profile)
/// - A local directory or path to `dataset.yaml`
pub fn run(
    dataset_spec: &str,
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
    cache_dir: Option<&Path>,
) {
    let cache = cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_cache_dir);

    // Try local path first
    let local = Path::new(dataset_spec);
    if local.exists() {
        run_local(local, &cache);
        return;
    }

    // Parse dataset:profile specifier
    let (dataset_name, profile_name) = if let Some(pos) = dataset_spec.find(':') {
        (&dataset_spec[..pos], &dataset_spec[pos + 1..])
    } else {
        (dataset_spec, "default")
    };

    // Resolve via catalog
    let sources = build_sources(configdir, extra_catalogs, at);
    if sources.is_empty() {
        eprintln!("No catalog sources configured.");
        eprintln!("Create ~/.config/vectordata/catalogs.yaml or use --catalog/--at to specify locations.");
        std::process::exit(1);
    }

    let catalog = Catalog::of(&sources);
    let entry = match catalog.find_exact(dataset_name) {
        Some(e) => e,
        None => {
            eprintln!("Dataset '{}' not found in catalog.", dataset_name);
            catalog.list_datasets(dataset_name);
            std::process::exit(1);
        }
    };

    let profile = match entry.layout.profiles.profile(profile_name) {
        Some(p) => p,
        None => {
            eprintln!(
                "Profile '{}' not found in dataset '{}'. Available: {}",
                profile_name,
                entry.name,
                entry.profile_names().join(", ")
            );
            std::process::exit(1);
        }
    };

    // Determine base URL for the dataset (entry.path points to dataset.yaml)
    let base_url = entry.path.rsplit_once('/').map(|(base, _)| base).unwrap_or("");

    let ds_cache = cache.join(&entry.name);
    std::fs::create_dir_all(&ds_cache).unwrap_or_else(|e| {
        eprintln!("Failed to create cache dir: {}", e);
        std::process::exit(1);
    });

    println!(
        "Prebuffering {}:{} ({} views)",
        entry.name,
        profile_name,
        profile.view_names().len()
    );

    let mut downloaded = 0u32;
    let mut skipped = 0u32;
    let mut failed = 0u32;

    for view_name in profile.view_names() {
        let view = profile.view(view_name).unwrap();
        let source_path = &view.source.path;

        if source_path.is_empty() {
            continue;
        }

        // Resolve the full URL for the view source
        let full_url = if source_path.starts_with("http://") || source_path.starts_with("https://") {
            source_path.clone()
        } else {
            format!("{}/{}", base_url, source_path)
        };

        let target = ds_cache.join(source_path);

        if target.exists() {
            let size = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
            println!("  {} — already cached ({} bytes)", view_name, size);
            skipped += 1;
            continue;
        }

        if full_url.starts_with("http://") || full_url.starts_with("https://") {
            println!("  {} — downloading from {}", view_name, full_url);
            match download_file(&full_url, &target) {
                Ok(size) => {
                    println!("  {} — downloaded {} bytes", view_name, size);
                    downloaded += 1;
                }
                Err(e) => {
                    eprintln!("  {} — FAILED: {}", view_name, e);
                    failed += 1;
                }
            }
        } else {
            // Local source relative to the entry path
            let local_src = Path::new(&full_url);
            if local_src.exists() && *local_src != target {
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                if let Err(e) = std::fs::copy(local_src, &target) {
                    eprintln!("  {} — FAILED to copy: {}", view_name, e);
                    failed += 1;
                } else {
                    println!("  {} — copied from local source", view_name);
                    downloaded += 1;
                }
            } else {
                eprintln!("  {} — source not available: {}", view_name, full_url);
                failed += 1;
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

/// Handle a local dataset directory or yaml path (original behavior).
fn run_local(path: &Path, cache: &Path) {
    let yaml_path = if path.is_file() {
        path.to_path_buf()
    } else {
        path.join("dataset.yaml")
    };

    if !yaml_path.exists() {
        eprintln!("dataset.yaml not found at {}", yaml_path.display());
        std::process::exit(1);
    }

    let content = std::fs::read_to_string(&yaml_path).unwrap_or_else(|e| {
        eprintln!("Failed to read: {}", e);
        std::process::exit(1);
    });

    let config: serde_yaml::Value = serde_yaml::from_str(&content).unwrap_or_else(|e| {
        eprintln!("Failed to parse: {}", e);
        std::process::exit(1);
    });

    let dataset_name = config
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("dataset");

    let ds_cache = cache.join(dataset_name);
    std::fs::create_dir_all(&ds_cache).unwrap_or_else(|e| {
        eprintln!("Failed to create cache dir: {}", e);
        std::process::exit(1);
    });

    // Copy dataset.yaml to cache
    let cached_yaml = ds_cache.join("dataset.yaml");
    if let Err(e) = std::fs::copy(&yaml_path, &cached_yaml) {
        eprintln!("  warning: failed to copy dataset.yaml: {}", e);
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
                            eprintln!("  {} — FAILED: {}", view_name, e);
                            failed += 1;
                        }
                    }
                } else {
                    let dataset_dir = yaml_path.parent().unwrap_or(Path::new("."));
                    let local_src = dataset_dir.join(&rel_path);
                    if local_src.exists() && local_src != target {
                        if let Err(e) = std::fs::copy(&local_src, &target) {
                            eprintln!("  {} — FAILED to copy: {}", view_name, e);
                            failed += 1;
                        } else {
                            println!("  {} — copied from local source", view_name);
                            downloaded += 1;
                        }
                    } else {
                        eprintln!("  {} — no source available", view_name);
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

/// Build catalog sources from CLI args (same logic as `datasets list`).
fn build_sources(configdir: &str, extra_catalogs: &[String], at: &[String]) -> CatalogSources {
    let mut sources = CatalogSources::new();

    if !at.is_empty() {
        sources = sources.add_catalogs(at);
    } else {
        sources = sources.configure(configdir);
        if !extra_catalogs.is_empty() {
            sources = sources.add_catalogs(extra_catalogs);
        }
    }

    sources
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

pub(crate) fn download_file(url: &str, dest: &Path) -> Result<u64, String> {
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
        run_local(ws, &cache);

        assert!(cache.join("test").join("dataset.yaml").exists());
        assert!(cache.join("test").join("data.fvec").exists());
    }
}
