// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks prepare cleanup` — build a parallel dataset tree from loose files.
//!
//! Reads a YAML manifest mapping dataset names to file lists (either plain
//! lists or the richer `{files, layout}` format from `infer-manifest`),
//! detects base/query vector roles from filenames, creates a subdirectory
//! for each dataset under a user-specified output directory, and populates
//! it with symlinks pointing back to the original (donor) files. The donor
//! tree is never modified.
//!
//! A `_layout.yaml` is saved in each subdirectory recording the original
//! file paths and their detected roles, so cleaned-up artifacts can later
//! be relinked to the original names.
//!
//! The output tree can be incrementally rebuilt — re-running cleanup with
//! the same output directory will update symlinks and metadata without
//! touching the donor files.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;

use crate::formats::VecFormat;
use crate::prepare::import::{self, ImportArgs};
use crate::prepare::wizard;

/// Arguments for `veks prepare cleanup`.
pub struct CleanupArgs {
    /// Path to the YAML manifest file.
    pub manifest: PathBuf,
    /// Output directory for the parallel dataset tree.
    pub output: PathBuf,
    /// Optional path to a dataset metadata YAML file mapping dataset
    /// names to properties (e.g. `similarity_function`). Used as a
    /// fallback when the manifest itself does not include the metric.
    pub metadata: Option<PathBuf>,
    /// Distance metric override (default: auto-detect).
    pub metric: String,
    /// Number of neighbors for KNN ground truth.
    pub neighbors: u32,
    /// Print what would be done without making changes.
    pub dry_run: bool,
}

/// A parsed manifest entry — normalized from either simple or rich format.
struct ManifestEntry {
    files: Vec<String>,
    /// Original path → detected role (from infer-manifest). Empty if the
    /// manifest used the simple list format.
    layout: IndexMap<String, String>,
    /// Vector similarity function (e.g. "COSINE", "DOT_PRODUCT").
    similarity_function: Option<String>,
}

/// A single entry from `dataset_metadata.yaml`.
#[derive(serde::Deserialize)]
struct MetadataEntry {
    similarity_function: Option<String>,
}

/// Load dataset metadata from a YAML file. The expected format is:
///
/// ```yaml
/// dataset-name:
///   similarity_function: COSINE
/// ```
///
/// Returns an empty map on any error (with a warning).
fn load_metadata(path: &Path) -> IndexMap<String, MetadataEntry> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_yaml::from_str(&text).unwrap_or_else(|e| {
            eprintln!("Warning: failed to parse metadata {}: {}", path.display(), e);
            IndexMap::new()
        }),
        Err(e) => {
            eprintln!("Warning: failed to read metadata {}: {}", path.display(), e);
            IndexMap::new()
        }
    }
}

/// Wrapper keys from infer-manifest that contain nested group maps.
const SECTION_KEYS: &[&str] = &["complete", "incomplete"];

/// Parse the manifest YAML, accepting multiple formats:
///
/// - Simple: `name: [file1, file2]`
/// - Rich:   `name: {files: [file1, file2], layout: {file1: role1, ...}}`
/// - Sectioned (from infer-manifest):
///   ```yaml
///   complete:
///     name: {files: [...], layout: {...}}
///   incomplete:
///     name: {files: [...], layout: {...}, missing: [...]}
///   orphans: [...]
///   ```
///
/// Section wrappers (`complete`, `incomplete`) are flattened — their
/// nested groups are promoted to the top level. The `orphans` key is
/// skipped (those files need manual assignment first).
fn parse_manifest(text: &str) -> Result<IndexMap<String, ManifestEntry>, String> {
    let raw: IndexMap<String, serde_yaml::Value> = serde_yaml::from_str(text)
        .map_err(|e| format!("failed to parse YAML: {}", e))?;

    let mut result = IndexMap::new();
    for (name, value) in &raw {
        // Skip orphan lists
        if name == "orphans" { continue; }

        // Flatten section wrappers (complete, incomplete) — their
        // values are maps of group entries.
        if SECTION_KEYS.contains(&name.as_str()) {
            if let serde_yaml::Value::Mapping(section) = value {
                for (k, v) in section {
                    if let Some(group_name) = k.as_str() {
                        if let Some(entry) = parse_group_value(v) {
                            result.insert(group_name.to_string(), entry);
                        }
                    }
                }
            }
            continue;
        }

        if let Some(entry) = parse_group_value(value) {
            result.insert(name.clone(), entry);
        } else {
            eprintln!("Warning: skipping '{}' — unrecognized value format", name);
        }
    }
    Ok(result)
}

/// Parse a single group value (list or mapping with files/layout).
fn parse_group_value(value: &serde_yaml::Value) -> Option<ManifestEntry> {
    match value {
        serde_yaml::Value::Sequence(seq) => {
            let files: Vec<String> = seq.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            Some(ManifestEntry { files, layout: IndexMap::new(), similarity_function: None })
        }
        serde_yaml::Value::Mapping(map) => {
            let get_str = |key: &str| map.get(serde_yaml::Value::String(key.into()));
            let files = get_str("files")
                .and_then(|v| v.as_sequence())
                .map(|seq| seq.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect())
                .unwrap_or_default();
            let layout = get_str("layout")
                .and_then(|v| v.as_mapping())
                .map(|m| m.iter()
                    .filter_map(|(k, v)| {
                        Some((k.as_str()?.to_string(), v.as_str()?.to_string()))
                    })
                    .collect())
                .unwrap_or_default();
            let similarity_function = get_str("similarity_function")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            Some(ManifestEntry { files, layout, similarity_function })
        }
        _ => None,
    }
}

/// Entry point for the cleanup command.
pub fn run(args: CleanupArgs) {
    let manifest_path = &args.manifest;
    let manifest_text = match std::fs::read_to_string(manifest_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: failed to read manifest {}: {}", manifest_path.display(), e);
            std::process::exit(1);
        }
    };

    let manifest = match parse_manifest(&manifest_text) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    if manifest.is_empty() {
        eprintln!("Manifest is empty — nothing to do.");
        return;
    }

    // Load optional dataset metadata (similarity_function fallback).
    let metadata = args.metadata.as_deref().map(load_metadata).unwrap_or_default();

    // Resolve file paths relative to the manifest's parent directory.
    let manifest_dir = manifest_path.parent().unwrap_or(Path::new("."));

    for (name, entry) in &manifest {
        println!("── {} ──", name);

        if entry.files.is_empty() {
            eprintln!("  Skipping '{}': no files listed", name);
            continue;
        }

        // Resolve and validate paths. Canonicalize so that symlinks
        // from the output directory back to the donor tree are correct
        // even when the output is in a different directory subtree.
        let mut resolved: Vec<PathBuf> = Vec::new();
        let mut missing = false;
        for f in &entry.files {
            let p = manifest_dir.join(f);
            match p.canonicalize() {
                Ok(abs) => resolved.push(abs),
                Err(_) => {
                    eprintln!("  Error: file '{}' does not exist (resolved: {})", f, p.display());
                    missing = true;
                }
            }
        }
        if missing {
            eprintln!("  Skipping '{}' due to missing files", name);
            continue;
        }

        // Build candidates for role detection: (path, format_name, file_size)
        let candidates: Vec<(PathBuf, String, u64)> = resolved.iter()
            .filter_map(|p| {
                let fmt = VecFormat::detect_from_path(p)?;
                let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
                Some((p.clone(), fmt.name().to_string(), size))
            })
            .collect();

        if candidates.is_empty() {
            eprintln!("  Skipping '{}': no recognized vector formats in file list", name);
            continue;
        }

        let roles = wizard::detect_roles(&candidates);

        if roles.base_vectors.is_none() {
            eprintln!("  Skipping '{}': could not detect base vectors from filenames", name);
            eprintln!("  Files: {:?}", entry.files);
            continue;
        }

        // Create subdirectory under the user-specified output directory.
        // Canonicalize so that relative symlinks back to the donor tree
        // resolve correctly regardless of how the output path was given.
        let subdir = args.output.join(name);
        if args.dry_run {
            println!("  Would create directory: {}/", subdir.display());
        } else {
            if let Err(e) = std::fs::create_dir_all(&subdir) {
                eprintln!("  Error: failed to create directory {}: {}", subdir.display(), e);
                continue;
            }
        }
        let subdir = if !args.dry_run {
            match subdir.canonicalize() {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("  Error: failed to canonicalize {}: {}", subdir.display(), e);
                    continue;
                }
            }
        } else {
            subdir
        };

        // Symlink source files into the output dataset directory (donor tree is untouched)
        let base_link = place_source_file(
            roles.base_vectors.as_ref().unwrap(), &subdir, args.dry_run,
        );
        let query_link = roles.query_vectors.as_ref().map(|qp| {
            place_source_file(qp, &subdir, args.dry_run)
        });

        // Save layout for later relinking
        let layout = build_layout(&resolved, &roles, &entry.layout);
        if args.dry_run {
            println!("  Would save _layout.yaml with {} entries", layout.len());
            println!("  Would save _manifest.yaml excerpt for '{}'", name);
            println!("  Would bootstrap dataset '{}' in {}/", name, subdir.display());
            println!();
            continue;
        }

        save_layout(&subdir, &layout);
        save_manifest_excerpt(&subdir, name, entry);

        // Bootstrap the dataset in the subdirectory
        let base_link = match base_link {
            Some(p) => p,
            None => {
                eprintln!("  Error: failed to create base vectors symlink for '{}'", name);
                continue;
            }
        };
        let query_link = match query_link {
            Some(Some(p)) => Some(p),
            Some(None) => {
                eprintln!("  Error: failed to create query vectors symlink for '{}'", name);
                continue;
            }
            None => None,
        };

        // Resolve similarity function: manifest entry > metadata file >
        // CLI override > error. Normalize to the canonical uppercase form
        // (COSINE, DOT_PRODUCT, L2, L1) that the KNN pipeline expects.
        let raw_metric = if let Some(ref sf) = entry.similarity_function {
            sf.clone()
        } else if let Some(sf) = metadata.get(name).and_then(|m| m.similarity_function.as_ref()) {
            sf.clone()
        } else if args.metric != "auto" {
            args.metric.clone()
        } else {
            eprintln!("  Error: no similarity_function for '{}' — \
                       specify --metric, --metadata, or add it to the manifest", name);
            continue;
        };
        let metric = normalize_metric(&raw_metric).unwrap_or_else(|| {
            eprintln!(
                "  Error: unrecognized similarity_function '{}' for '{}'. \
                 Use COSINE, DOT_PRODUCT, L2, or L1.",
                raw_metric, name,
            );
            std::process::exit(1);
        });

        let has_query = query_link.is_some();
        let import_args = ImportArgs {
            name: name.clone(),
            output: subdir.clone(),
            base_vectors: Some(base_link),
            query_vectors: query_link,
            self_search: !has_query,
            query_count: 10000,
            metadata: None,
            ground_truth: None,
            ground_truth_distances: None,
            metric,
            neighbors: args.neighbors,
            seed: 42,
            description: None,
            no_dedup: false,
            no_zero_check: false,
            no_filtered: true,
            normalize: false,
            force: true,
            base_convert_format: None,
            query_convert_format: None,
            compress_cache: false,
            sized_profiles: None,
            base_fraction: 1.0,
            required_facets: None,
            provided_facets: None,
            round_digits: 2,
            pedantic_dedup: false,
            selectivity: 0.0001,
            predicate_count: 10000,
            predicate_strategy: "eq".to_string(),
            classic: true,
            personality: "native".to_string(),
            synthesize_metadata: false,
            synthesis_mode: "simple-int-eq".to_string(),
            synthesis_format: "slab".to_string(),
            metadata_fields: 3,
            metadata_range_min: 0,
            metadata_range_max: 1000,
            predicate_range_min: 0,
            predicate_range_max: 1000,
            verify_knn_sample: 0,
        };

        import::run(import_args);
        println!();
    }
}

/// Build the layout mapping for a dataset group.
///
/// Merges role detection results with any pre-existing layout from the
/// manifest (infer-manifest format). The layout maps original file paths
/// to their detected roles for later relinking.
fn build_layout(
    resolved: &[PathBuf],
    roles: &wizard::DetectedRoles,
    existing_layout: &IndexMap<String, String>,
) -> IndexMap<String, String> {
    let mut layout = IndexMap::new();

    for p in resolved {
        let path_str = p.to_string_lossy().to_string();
        // Use pre-existing layout role if available, otherwise detect
        let role = if let Some(r) = existing_layout.values()
            .zip(existing_layout.keys())
            .find(|(_, k)| path_str.ends_with(k.as_str()))
            .map(|(v, _)| v.clone())
        {
            r
        } else {
            detect_file_role(p, roles)
        };
        layout.insert(path_str, role);
    }
    layout
}

/// Determine which role a file was assigned to by detect_roles.
fn detect_file_role(path: &Path, roles: &wizard::DetectedRoles) -> String {
    if roles.base_vectors.as_deref() == Some(path) {
        "base_vectors".to_string()
    } else if roles.query_vectors.as_deref() == Some(path) {
        "query_vectors".to_string()
    } else if roles.neighbor_indices.as_deref() == Some(path) {
        "neighbor_indices".to_string()
    } else if roles.neighbor_distances.as_deref() == Some(path) {
        "neighbor_distances".to_string()
    } else {
        "unassigned".to_string()
    }
}

/// Save the layout mapping to `_layout.yaml` in the dataset subdirectory.
fn save_layout(subdir: &Path, layout: &IndexMap<String, String>) {
    let path = subdir.join("_layout.yaml");
    match serde_yaml::to_string(layout) {
        Ok(yaml) => {
            if let Err(e) = std::fs::write(&path, &yaml) {
                eprintln!("  Warning: failed to write {}: {}", path.display(), e);
            } else {
                println!("  Saved layout to {}", path.display());
            }
        }
        Err(e) => eprintln!("  Warning: failed to serialize layout: {}", e),
    }
}

/// Save the manifest excerpt for this dataset to `_manifest.yaml` in
/// the dataset subdirectory. This preserves the original grouping,
/// layout, and similarity function from the inferred manifest.
fn save_manifest_excerpt(subdir: &Path, name: &str, entry: &ManifestEntry) {
    let path = subdir.join("_manifest.yaml");
    let mut excerpt = IndexMap::new();
    let mut group = IndexMap::new();
    group.insert("files".to_string(), serde_yaml::to_value(&entry.files).unwrap_or_default());
    group.insert("layout".to_string(), serde_yaml::to_value(&entry.layout).unwrap_or_default());
    if let Some(ref sf) = entry.similarity_function {
        group.insert("similarity_function".to_string(), serde_yaml::Value::String(sf.clone()));
    }
    excerpt.insert(name.to_string(), group);

    match serde_yaml::to_string(&excerpt) {
        Ok(yaml) => {
            if let Err(e) = std::fs::write(&path, &yaml) {
                eprintln!("  Warning: failed to write {}: {}", path.display(), e);
            } else {
                println!("  Saved manifest excerpt to {}", path.display());
            }
        }
        Err(e) => eprintln!("  Warning: failed to serialize manifest excerpt: {}", e),
    }
}

/// Normalize a metric/similarity function string to the canonical
/// uppercase form used by the pipeline (`COSINE`, `DOT_PRODUCT`, `L2`,
/// `L1`). Returns `None` for unrecognized values.
fn normalize_metric(s: &str) -> Option<String> {
    match s.to_uppercase().as_str() {
        "COSINE" => Some("COSINE".into()),
        "DOT_PRODUCT" | "DOTPRODUCT" | "DOT" | "INNER_PRODUCT" | "IP" => Some("DOT_PRODUCT".into()),
        "L2" | "EUCLIDEAN" => Some("L2".into()),
        "L1" | "MANHATTAN" => Some("L1".into()),
        _ => None,
    }
}

/// Place a source file into `subdir` with an underscore-prefixed name
/// and canonicalized extension (e.g. `fvecs` → `fvec`).
///
/// Creates a relative symlink from `subdir` back to the source file.
/// The source file is never moved or modified.
///
/// Returns the path to the symlink, or `None` on failure.
fn place_source_file(
    source: &Path, subdir: &Path, dry_run: bool,
) -> Option<PathBuf> {
    let file_name = source.file_name()?.to_string_lossy();
    let stem = Path::new(file_name.as_ref()).file_stem()?.to_string_lossy();
    let ext = source.extension()
        .and_then(|e| e.to_str())
        .and_then(VecFormat::canonical_extension)
        .unwrap_or("fvec");
    let dest_name = format!("_{}.{}", stem, ext);
    let dest_path = subdir.join(&dest_name);

    if dry_run {
        println!("  Would symlink: {} → {}", source.display(), dest_path.display());
        return Some(dest_path);
    }

    import::create_symlink(source, &dest_path);

    if dest_path.exists() || dest_path.symlink_metadata().is_ok() {
        Some(dest_path)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parse_simple_manifest() {
        let yaml = r#"
sift-128:
  - sift/sift_base.fvecs
  - sift/sift_query.fvecs
glove-200:
  - glove/base.fvec
"#;
        let manifest = parse_manifest(yaml).unwrap();
        assert_eq!(manifest.len(), 2);
        assert_eq!(manifest["sift-128"].files.len(), 2);
        assert!(manifest["sift-128"].layout.is_empty());
        assert_eq!(manifest["glove-200"].files.len(), 1);
    }

    #[test]
    fn parse_rich_manifest() {
        let yaml = r#"
sift:
  files:
    - sift/sift_base.fvecs
    - sift/sift_query.fvecs
  layout:
    sift/sift_base.fvecs: base_vectors
    sift/sift_query.fvecs: query_vectors
"#;
        let manifest = parse_manifest(yaml).unwrap();
        assert_eq!(manifest.len(), 1);
        assert_eq!(manifest["sift"].files.len(), 2);
        assert_eq!(manifest["sift"].layout.len(), 2);
        assert_eq!(manifest["sift"].layout["sift/sift_base.fvecs"], "base_vectors");
    }

    #[test]
    fn parse_sectioned_manifest() {
        let yaml = r#"
complete:
  sift:
    files:
      - sift/sift_base.fvec
      - sift/sift_query.fvec
      - sift/sift_gt.ivec
    layout:
      sift/sift_base.fvec: base_vectors
      sift/sift_query.fvec: query_vectors
      sift/sift_gt.ivec: neighbor_indices
incomplete:
  glove:
    files:
      - glove/glove_base.fvec
    layout:
      glove/glove_base.fvec: base_vectors
    missing:
      - query_vectors
      - neighbor_indices
orphans:
  - stray/unknown.fvec
"#;
        let manifest = parse_manifest(yaml).unwrap();
        // complete and incomplete sections are flattened, orphans skipped
        assert_eq!(manifest.len(), 2);
        assert_eq!(manifest["sift"].files.len(), 3);
        assert_eq!(manifest["glove"].files.len(), 1);
    }

    #[test]
    fn symlink_name_canonicalizes_extension() {
        let source = Path::new("data/sift_base.fvecs");
        let stem = Path::new(source.file_name().unwrap()).file_stem().unwrap().to_string_lossy();
        let ext = source.extension()
            .and_then(|e| e.to_str())
            .and_then(VecFormat::canonical_extension)
            .unwrap_or("fvec");
        let link_name = format!("_{}.{}", stem, ext);
        assert_eq!(link_name, "_sift_base.fvec");
    }

    #[test]
    fn symlink_name_preserves_canonical_extension() {
        let source = Path::new("data/base_vectors.mvec");
        let stem = Path::new(source.file_name().unwrap()).file_stem().unwrap().to_string_lossy();
        let ext = source.extension()
            .and_then(|e| e.to_str())
            .and_then(VecFormat::canonical_extension)
            .unwrap_or("fvec");
        let link_name = format!("_{}.{}", stem, ext);
        assert_eq!(link_name, "_base_vectors.mvec");
    }

    #[test]
    fn dry_run_does_not_create_files() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest_path = tmp.path().join("manifest.yaml");
        let output_dir = tmp.path().join("output");
        let mut f = std::fs::File::create(&manifest_path).unwrap();
        writeln!(f, "test-ds:\n  - nonexistent.fvec").unwrap();

        let args = CleanupArgs {
            manifest: manifest_path,
            output: output_dir.clone(),
            metadata: None,
            metric: "auto".to_string(),
            neighbors: 100,
            dry_run: true,
        };

        // Should not panic even with nonexistent files — it warns and skips
        run(args);
        assert!(!output_dir.join("test-ds").exists());
    }
}
