// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks datasets prebuffer` — download and cache dataset facets locally.
//!
//! Accepts either a catalog dataset specifier (`dataset:profile`) or a local
//! path to a dataset directory / `dataset.yaml`. When a catalog specifier is
//! given, the dataset is resolved through the configured catalog chain and
//! facets are downloaded from the remote source.

use std::io::{Read, Write};
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
        .unwrap_or_else(|| crate::pipeline::commands::config::configured_cache_dir_or_exit());

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

    // Open through the canonical catalog → TestDataView path. Every
    // facet's underlying Storage knows how to prebuffer itself —
    // local files no-op, cached-remote downloads + merkle-verifies,
    // direct-HTTP no-ops silently.
    let view = match catalog.open_profile(dataset_name, profile_name) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to open dataset: {}", e);
            std::process::exit(1);
        }
    };

    let mut last_facet = String::new();
    let mut total_facets = 0u32;
    let result = view.prebuffer_all_with_progress(&mut |facet, p| {
        if facet != last_facet {
            if !last_facet.is_empty() {
                println!("  {last_facet} — done");
            }
            println!(
                "  {facet} — prebuffering ({}/{} chunks, {:.1} MiB total)...",
                p.verified_chunks,
                p.total_chunks,
                p.total_bytes as f64 / 1_048_576.0,
            );
            last_facet = facet.to_string();
            total_facets += 1;
        }
    });
    if !last_facet.is_empty() {
        println!("  {last_facet} — done");
    }

    match result {
        Ok(()) => {
            println!();
            println!("Prebuffer: {total_facets} facets processed");
        }
        Err(e) => {
            eprintln!();
            eprintln!("Prebuffer: failed — {e}");
            std::process::exit(1);
        }
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

    let client = reqwest::blocking::Client::builder()
        .user_agent("veks/0.14")
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))?;

    let mut response = client.get(url).send()
        .map_err(|e| format!("download failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("HTTP {} from {}", response.status().as_u16(), url));
    }

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut last_print = std::time::Instant::now();
    let mut buf = vec![0u8; 256 * 1024];

    loop {
        let n = response.read(&mut buf)
            .map_err(|e| format!("read error: {}", e))?;
        if n == 0 { break; }
        file.write_all(&buf[..n])
            .map_err(|e| format!("write error: {}", e))?;
        downloaded += n as u64;

        let now = std::time::Instant::now();
        if now.duration_since(last_print).as_millis() >= 500 && total_size > 0 {
            let pct = 100.0 * downloaded as f64 / total_size as f64;
            let dl_mib = downloaded as f64 / 1_048_576.0;
            let total_mib = total_size as f64 / 1_048_576.0;
            eprint!("\r    {:.1} / {:.1} MiB ({:.0}%)   ", dl_mib, total_mib, pct);
            let _ = std::io::stderr().flush();
            last_print = now;
        }
    }
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
