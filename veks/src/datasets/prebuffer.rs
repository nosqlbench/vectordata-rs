// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! `veks datasets prebuffer` — download and cache dataset facets locally.
//!
//! Accepts either a catalog dataset specifier (`dataset:profile`) or a local
//! path to a dataset directory / `dataset.yaml`. When a catalog specifier is
//! given, the dataset is resolved through the configured catalog chain and
//! facets are downloaded from the remote source.

use std::io::Write;
use std::path::Path;

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
        .unwrap_or_else(|| crate::pipeline::commands::config::configured_cache_dir());

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
        eprintln!("Add a catalog with:");
        eprintln!("  veks datasets config add-catalog <URL-or-path>");
        eprintln!();
        eprintln!("Or use --catalog/--at for one-off access.");
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

    let ds_cache = cache.join(&entry.name);

    println!(
        "Prebuffering {}:{} ({} views)",
        entry.name,
        profile_name,
        profile.view_names().len()
    );
    println!("  Cache: {}", ds_cache.display());

    // Use the data access layer's RemoteDatasetView which sets up
    // merkle-verified CachedChannels per facet. The prebuffer method
    // only downloads chunks covering the profile's windowed range.
    use vectordata::dataset::remote::RemoteDatasetView;
    let dataset_view = match RemoteDatasetView::open(entry, profile_name, &cache) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to open remote dataset: {}", e);
            std::process::exit(1);
        }
    };

    let facets: &[(&str, fn(&RemoteDatasetView) -> Option<&dyn vectordata::dataset::view::TypedVectorView>)] = &[
        ("base_vectors", |v| v.base_vectors()),
        ("query_vectors", |v| v.query_vectors()),
        ("neighbor_indices", |v| v.neighbor_indices()),
        ("neighbor_distances", |v| v.neighbor_distances()),
        ("filtered_neighbor_indices", |v| v.filtered_neighbor_indices()),
        ("filtered_neighbor_distances", |v| v.filtered_neighbor_distances()),
        ("metadata_content", |v| v.metadata_content()),
    ];

    let mut prebuffered = 0u32;
    let mut skipped = 0u32;
    let mut failed = 0u32;

    for &(name, accessor) in facets {
        if let Some(view) = accessor(&dataset_view) {
            let count = view.count();
            if count == 0 { continue; }

            if let Some(stats) = view.cache_stats() {
                if stats.is_complete {
                    println!("  {} — complete ({} chunks)", name, stats.total_chunks);
                    skipped += 1;
                    continue;
                }
                println!("  {} — prebuffering {} vectors ({}/{} chunks cached)...",
                    name, count, stats.valid_chunks, stats.total_chunks);
            } else {
                println!("  {} — prebuffering {} vectors...", name, count);
            }

            match view.prebuffer(0, count) {
                Ok(()) => {
                    println!("  {} — done", name);
                    prebuffered += 1;
                }
                Err(e) => {
                    eprintln!("  {} — FAILED: {}", name, e);
                    failed += 1;
                }
            }
        }
    }

    println!();
    println!(
        "Prebuffer: {} completed, {} already cached, {} failed",
        prebuffered, skipped, failed
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
        eprintln!("dataset.yaml not found at {}", crate::check::rel_display(&yaml_path));
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
    easy.progress(true).ok();

    let last_print = std::sync::Arc::new(std::sync::Mutex::new(std::time::Instant::now()));
    let lp = last_print.clone();

    let mut transfer = easy.transfer();
    transfer
        .progress_function(move |dl_total, dl_now, _, _| {
            let mut last = lp.lock().unwrap();
            let now = std::time::Instant::now();
            if now.duration_since(*last).as_millis() >= 500 && dl_total > 0.0 {
                let pct = 100.0 * dl_now / dl_total;
                let dl_mib = dl_now / 1_048_576.0;
                let total_mib = dl_total / 1_048_576.0;
                eprint!("\r    {:.1} / {:.1} MiB ({:.0}%)   ", dl_mib, total_mib, pct);
                let _ = std::io::stderr().flush();
                *last = now;
            }
            true
        })
        .map_err(|e| format!("progress setup error: {}", e))?;
    transfer
        .write_function(|data| {
            file.write_all(data).map_or(Ok(0), |()| Ok(data.len()))
        })
        .map_err(|e| format!("transfer setup error: {}", e))?;
    transfer
        .perform()
        .map_err(|e| format!("download failed: {}", e))?;
    drop(transfer);
    eprintln!(); // newline after progress

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
