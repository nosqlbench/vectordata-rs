// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `veks datasets plan` — check which facets are present or missing.

use std::path::Path;

pub fn run(path: &Path) {
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
        println!("Failed to read {}: {}", yaml_path.display(), e);
        std::process::exit(1);
    });

    let config: serde_yaml::Value = serde_yaml::from_str(&content).unwrap_or_else(|e| {
        println!("Failed to parse {}: {}", yaml_path.display(), e);
        std::process::exit(1);
    });

    let dataset_dir = yaml_path.parent().unwrap_or(Path::new("."));
    let dataset_name = config
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    println!("Dataset: {} ({})", dataset_name, yaml_path.display());
    println!();

    let mut missing = Vec::new();
    let mut present: Vec<(String, String, u64)> = Vec::new();

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

    if !present.is_empty() {
        println!("Present views:");
        for (name, path, size) in &present {
            println!("  {} — {} ({} bytes)", name, path, size);
        }
        println!();
    }

    if !missing.is_empty() {
        println!("Missing views:");
        for (name, path) in &missing {
            println!("  {} — {}", name, path);
        }
        println!();

        let dim = detect_dimension(dataset_dir, &present);

        println!("Suggested commands:");
        for (name, path) in &missing {
            let ext = Path::new(path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            let full = dataset_dir.join(path);
            match (name.as_str(), ext) {
                (n, "fvec") if n.contains("base") || n.contains("vector") => {
                    println!(
                        "  veks run generate vectors output={} dimension={} count=<COUNT> seed=42",
                        full.display(),
                        dim.unwrap_or(128)
                    );
                }
                (n, "fvec") if n.contains("query") => {
                    println!(
                        "  veks run generate vectors output={} dimension={} count=<QUERY_COUNT> seed=43",
                        full.display(),
                        dim.unwrap_or(128)
                    );
                }
                (n, "ivec") if n.contains("neighbor") || n.contains("indic") => {
                    println!(
                        "  veks run compute knn source=<BASE_VECTORS> queries=<QUERY_VECTORS> output-indices={} k=100 metric=EUCLIDEAN",
                        full.display()
                    );
                }
                (n, "fvec") if n.contains("dist") => {
                    println!(
                        "  veks run compute knn source=<BASE_VECTORS> queries=<QUERY_VECTORS> output-distances={} k=100 metric=EUCLIDEAN",
                        full.display()
                    );
                }
                _ => {
                    println!(
                        "  # {} — no automatic suggestion for '{}'",
                        name, path
                    );
                }
            }
        }
    } else {
        println!("All facets are present.");
    }

    println!();
    println!("{} present, {} missing", present.len(), missing.len());
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

fn detect_dimension(dir: &Path, present: &[(String, String, u64)]) -> Option<usize> {
    for (_, rel_path, _) in present {
        if rel_path.ends_with(".fvec") {
            let full = dir.join(rel_path);
            if let Ok(data) = std::fs::read(&full) {
                if data.len() >= 4 {
                    let dim =
                        u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
                    if dim > 0 && dim < 100_000 {
                        return Some(dim);
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_dimension_from_fvec() {
        let tmp = tempfile::tempdir().unwrap();
        // 4-byte LE dimension = 128
        let mut data = vec![128u8, 0, 0, 0];
        data.extend_from_slice(&[0u8; 512]); // 128 floats
        std::fs::write(tmp.path().join("test.fvec"), &data).unwrap();

        let present = vec![("base".to_string(), "test.fvec".to_string(), data.len() as u64)];
        assert_eq!(detect_dimension(tmp.path(), &present), Some(128));
    }
}
