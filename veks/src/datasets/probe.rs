// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `veks datasets probe` — verify remote dataset access.
//!
//! Probes a remote dataset by:
//! 1. Fetching dataset.yaml from the catalog URL
//! 2. Listing all profiles and facets
//! 3. For each facet, reading the first record via HTTP range request
//! 4. Reporting success/failure per facet

use vectordata::dataset::DatasetConfig;

/// Run the probe command.
pub fn run(catalog_base: &str, dataset_name: &str, profile_name: &str) {
    let base = catalog_base.trim_end_matches('/');

    println!("Probing dataset '{}' at {}", dataset_name, base);
    println!();

    // Step 1: Fetch catalog.json to find the dataset and its path
    let catalog_url = format!("{}/catalog.json", base);
    println!("  Fetching catalog: {}", catalog_url);
    let mut dataset_path_from_catalog: Option<String> = None;
    match fetch_text(&catalog_url) {
        Ok(text) => {
            let size = text.len();
            println!("    OK ({} bytes)", size);

            // Catalog can be either a top-level array or {"datasets": [...]}
            let entries: Option<Vec<serde_json::Value>> =
                serde_json::from_str::<Vec<serde_json::Value>>(&text).ok()
                    .or_else(|| {
                        serde_json::from_str::<serde_json::Value>(&text).ok()
                            .and_then(|v| v.get("datasets")?.as_array().cloned())
                    });

            if let Some(datasets) = entries {
                let entry = datasets.iter().find(|d| {
                    d.get("name").and_then(|n| n.as_str()) == Some(dataset_name)
                });
                if let Some(ds) = entry {
                    let path = ds.get("path").and_then(|p| p.as_str()).unwrap_or("");
                    println!("    Dataset '{}' found in catalog (path: {})", dataset_name, path);
                    dataset_path_from_catalog = Some(path.to_string());
                } else {
                    let names: Vec<&str> = datasets.iter()
                        .filter_map(|d| d.get("name").and_then(|n| n.as_str()))
                        .collect();
                    println!("    Dataset '{}' NOT found. Available: {:?}", dataset_name, names);
                }
            }
        }
        Err(e) => {
            println!("    FAILED: {}", e);
        }
    }
    println!();

    // Step 2: Fetch dataset.yaml using the path from the catalog
    let dataset_url = if let Some(ref cat_path) = dataset_path_from_catalog {
        // Path from catalog is relative to the catalog directory (e.g., "laion400b/img-search/dataset.yaml")
        format!("{}/{}", base, cat_path)
    } else {
        // Fallback: assume {base}/{name}/dataset.yaml
        format!("{}/{}/dataset.yaml", base, dataset_name)
    };
    println!("  Fetching dataset.yaml: {}", dataset_url);
    let config = match fetch_text(&dataset_url) {
        Ok(text) => {
            println!("    OK ({} bytes)", text.len());
            match serde_yaml::from_str::<DatasetConfig>(&text) {
                Ok(c) => {
                    println!("    Parsed: name='{}', {} profile(s)",
                        c.name, c.profiles.profiles.len());
                    Some(c)
                }
                Err(e) => {
                    println!("    Parse FAILED: {}", e);
                    None
                }
            }
        }
        Err(e) => {
            println!("    FAILED: {}", e);
            println!();
            println!("  Trying alternate paths...");

            // Try without dataset name in path (dataset.yaml at base)
            let alt_url = format!("{}/dataset.yaml", base);
            println!("  Fetching: {}", alt_url);
            match fetch_text(&alt_url) {
                Ok(text) => {
                    println!("    OK ({} bytes)", text.len());
                    match serde_yaml::from_str::<DatasetConfig>(&text) {
                        Ok(c) => {
                            println!("    Parsed: name='{}'", c.name);
                            Some(c)
                        }
                        Err(e) => { println!("    Parse FAILED: {}", e); None }
                    }
                }
                Err(e2) => { println!("    FAILED: {}", e2); None }
            }
        }
    };
    println!();

    let mut config = match config {
        Some(c) => c,
        None => {
            eprintln!("Could not load dataset configuration. Aborting probe.");
            std::process::exit(1);
        }
    };

    // Try to resolve deferred sized profiles from variables.yaml
    if config.profiles.has_deferred() {
        let vars_url = format!("{}/{}/variables.yaml", base, dataset_name);
        println!("  Fetching variables.yaml: {}", vars_url);
        match fetch_text(&vars_url) {
            Ok(text) => {
                println!("    OK ({} bytes)", text.len());
                if let Ok(vars) = serde_yaml::from_str::<indexmap::IndexMap<String, String>>(&text) {
                    let added = config.profiles.expand_deferred_sized(&vars);
                    if added > 0 {
                        println!("    Resolved {} deferred sized profiles", added);
                    }
                }
            }
            Err(e) => {
                println!("    Not available: {} (sized profiles will not be resolved)", e);
            }
        }
        println!();
    }

    // Step 3: List profiles and facets
    println!("  Profiles:");
    for (name, profile) in &config.profiles.profiles {
        let bc = profile.base_count
            .map(|n| format!("base_count={}", n))
            .unwrap_or_else(|| "full".into());
        let facets: Vec<&str> = profile.views.keys().map(|k| k.as_str()).collect();
        println!("    {} ({}): {}", name, bc, facets.join(", "));
    }
    println!();

    // Step 4: Probe each facet of the requested profile
    let profile = config.profiles.profile(profile_name);
    if profile.is_none() {
        let available: Vec<&str> = config.profiles.profiles.keys()
            .map(|k| k.as_str()).collect();
        eprintln!("  Profile '{}' not found. Available: {:?}", profile_name, available);
        std::process::exit(1);
    }
    let profile = profile.unwrap();

    println!("  Probing facets for profile '{}':", profile_name);
    let mut pass = 0;
    let mut fail = 0;

    for (facet_name, view) in &profile.views {
        let facet_path = &view.source.path;
        // Resolve the facet URL relative to the dataset
        let facet_url = if facet_path.starts_with("http://") || facet_path.starts_with("https://") {
            facet_path.clone()
        } else {
            format!("{}/{}/{}", base, dataset_name, facet_path)
        };

        print!("    {} ({})... ", facet_name, facet_path);

        // Try a HEAD request first to check existence and get size
        match probe_url(&facet_url) {
            Ok((code, size)) => {
                if code == 200 {
                    println!("OK (HTTP {}, {} bytes)", code, format_size(size));
                    pass += 1;

                    // Try reading the first few bytes
                    match fetch_range(&facet_url, 0, 64) {
                        Ok(data) => {
                            println!("      First 64 bytes readable ({} received)", data.len());
                        }
                        Err(e) => {
                            println!("      Range read failed: {}", e);
                        }
                    }
                } else {
                    println!("FAILED (HTTP {})", code);
                    fail += 1;
                }
            }
            Err(e) => {
                println!("FAILED: {}", e);
                fail += 1;
            }
        }
    }

    println!();
    println!("  Summary: {} facets OK, {} failed", pass, fail);
    if fail > 0 {
        std::process::exit(1);
    }
}

/// Fetch a text resource via HTTP GET.
fn fetch_text(url: &str) -> Result<String, String> {
    let mut data = Vec::new();
    let mut handle = curl::easy::Easy::new();
    handle.url(url).map_err(|e| format!("curl: {}", e))?;
    handle.follow_location(true).map_err(|e| format!("curl: {}", e))?;
    handle.useragent("veks/0.9").map_err(|e| format!("curl: {}", e))?;
    handle.timeout(std::time::Duration::from_secs(30)).map_err(|e| format!("curl: {}", e))?;
    {
        let mut transfer = handle.transfer();
        transfer.write_function(|buf| {
            data.extend_from_slice(buf);
            Ok(buf.len())
        }).map_err(|e| format!("curl: {}", e))?;
        transfer.perform().map_err(|e| format!("HTTP GET {} failed: {}", url, e))?;
    }
    let code = handle.response_code().map_err(|e| format!("curl: {}", e))?;
    if code != 200 {
        return Err(format!("HTTP {} from {}", code, url));
    }
    String::from_utf8(data).map_err(|e| format!("invalid UTF-8 from {}: {}", url, e))
}

/// HEAD request — returns (status_code, content_length).
fn probe_url(url: &str) -> Result<(u32, u64), String> {
    let mut handle = curl::easy::Easy::new();
    handle.url(url).map_err(|e| format!("curl: {}", e))?;
    handle.nobody(true).map_err(|e| format!("curl: {}", e))?; // HEAD
    handle.follow_location(true).map_err(|e| format!("curl: {}", e))?;
    handle.useragent("veks/0.9").map_err(|e| format!("curl: {}", e))?;
    handle.timeout(std::time::Duration::from_secs(15)).map_err(|e| format!("curl: {}", e))?;
    handle.perform().map_err(|e| format!("HEAD {} failed: {}", url, e))?;
    let code = handle.response_code().map_err(|e| format!("curl: {}", e))?;
    let size = handle.content_length_download().unwrap_or(-1.0);
    Ok((code, if size >= 0.0 { size as u64 } else { 0 }))
}

/// Fetch a byte range via HTTP Range request.
fn fetch_range(url: &str, offset: u64, length: u64) -> Result<Vec<u8>, String> {
    let mut data = Vec::new();
    let mut handle = curl::easy::Easy::new();
    handle.url(url).map_err(|e| format!("curl: {}", e))?;
    handle.follow_location(true).map_err(|e| format!("curl: {}", e))?;
    handle.useragent("veks/0.9").map_err(|e| format!("curl: {}", e))?;
    handle.range(&format!("{}-{}", offset, offset + length - 1))
        .map_err(|e| format!("curl: {}", e))?;
    handle.timeout(std::time::Duration::from_secs(15)).map_err(|e| format!("curl: {}", e))?;
    {
        let mut transfer = handle.transfer();
        transfer.write_function(|buf| {
            data.extend_from_slice(buf);
            Ok(buf.len())
        }).map_err(|e| format!("curl: {}", e))?;
        transfer.perform().map_err(|e| format!("Range GET {} failed: {}", url, e))?;
    }
    let code = handle.response_code().map_err(|e| format!("curl: {}", e))?;
    if code != 200 && code != 206 {
        return Err(format!("HTTP {} from {} (expected 200 or 206)", code, url));
    }
    Ok(data)
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
