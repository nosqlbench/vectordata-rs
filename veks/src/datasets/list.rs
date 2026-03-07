// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `veks datasets list` — scan a directory for dataset.yaml files and list them.

use std::path::Path;

use dataset::DatasetConfig;

#[derive(Debug, serde::Serialize)]
struct DatasetSummary {
    name: String,
    path: String,
    description: Option<String>,
    views: Vec<String>,
    profile_count: usize,
    has_pipeline: bool,
}

pub fn run(catalog: &Path, format: &str, verbose: bool) {
    let catalog = if catalog.is_absolute() {
        catalog.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join(catalog)
    };

    if !catalog.is_dir() {
        println!("{} is not a directory", catalog.display());
        std::process::exit(1);
    }

    let datasets = find_datasets(&catalog);

    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&datasets).unwrap_or_default();
            println!("{}", json);
        }
        "yaml" => {
            let yaml = serde_yaml::to_string(&datasets).unwrap_or_default();
            println!("{}", yaml);
        }
        _ => {
            if datasets.is_empty() {
                println!("No datasets found in {}", catalog.display());
            } else {
                println!("Datasets in {}:", catalog.display());
                println!();
                for ds in &datasets {
                    println!("  {}", ds.name);
                    if verbose {
                        println!("    Path: {}", ds.path);
                        if let Some(ref desc) = ds.description {
                            println!("    Description: {}", desc);
                        }
                        println!("    Views: {}", ds.views.join(", "));
                        println!("    Profiles: {}", ds.profile_count);
                        println!(
                            "    Pipeline: {}",
                            if ds.has_pipeline { "yes" } else { "no" }
                        );
                        println!();
                    }
                }
                if !verbose {
                    println!();
                    println!(
                        "  {} datasets found. Use --verbose for details.",
                        datasets.len()
                    );
                }
            }
        }
    }
}

fn find_datasets(dir: &Path) -> Vec<DatasetSummary> {
    let mut results = Vec::new();
    find_recursive(dir, &mut results, 0);
    results.sort_by(|a, b| a.name.cmp(&b.name));
    results
}

fn find_recursive(dir: &Path, results: &mut Vec<DatasetSummary>, depth: usize) {
    if depth > 5 {
        return;
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_file() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name == "dataset.yaml" || name == "dataset.yml" {
                match DatasetConfig::load(&path) {
                    Ok(config) => {
                        let views: Vec<String> =
                            config.view_names().into_iter().map(|s| s.to_string()).collect();
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
                        log::warn!("failed to parse {}: {}", path.display(), e);
                    }
                }
            }
        } else if path.is_dir() {
            let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !dir_name.starts_with('.')
                && dir_name != "target"
                && dir_name != "node_modules"
            {
                find_recursive(&path, results, depth + 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_datasets_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let datasets = find_datasets(tmp.path());
        assert!(datasets.is_empty());
    }

    #[test]
    fn find_datasets_with_yaml() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = "name: test-dataset\ndescription: A test\nprofiles:\n  default:\n    base_vectors: base.fvec\n";
        std::fs::write(tmp.path().join("dataset.yaml"), yaml).unwrap();

        let datasets = find_datasets(tmp.path());
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].name, "test-dataset");
    }

    #[test]
    fn find_datasets_nested() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        let yaml = "name: nested\nprofiles:\n  default:\n    base_vectors: data.fvec\n";
        std::fs::write(sub.join("dataset.yaml"), yaml).unwrap();

        let datasets = find_datasets(tmp.path());
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].name, "nested");
    }
}
