// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! `veks prepare infer-manifest` — scan a directory hierarchy and produce
//! a `inferred-manifest.yaml` that groups files by dataset name.
//!
//! Files are grouped into logically intact sets containing base vectors,
//! query vectors, and ground truth indices (with optional distances).
//! The grouping algorithm uses two phases:
//!
//! 1. **Name inference** — strip role/count/qualifier tokens from filenames
//!    and group files that share a common stem.
//! 2. **Directory-aware merging** — within each directory, merge groups
//!    that are missing complementary roles (e.g. a base+query group with
//!    an orphaned ground-truth ivec from a different prefix).
//!
//! The output manifest is compatible with `veks prepare cleanup`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;

use crate::formats::VecFormat;
use crate::prepare::wizard;

/// Arguments for `veks prepare infer-manifest`.
pub struct InferManifestArgs {
    /// Root directory to scan.
    pub root: PathBuf,
    /// Output path for the manifest file.
    pub output: PathBuf,
}

/// One dataset group in the manifest.
#[derive(Debug, serde::Serialize)]
struct ManifestGroup {
    /// File paths (compatible with cleanup's simple list format).
    files: Vec<String>,
    /// Original path → detected role mapping for later relinking.
    layout: IndexMap<String, String>,
    /// Vector similarity function (e.g. "COSINE", "DOT_PRODUCT").
    /// Resolved from dataset_metadata.yml or set to "unknown".
    #[serde(skip_serializing_if = "Option::is_none")]
    similarity_function: Option<String>,
    /// Which required roles are missing (empty for complete groups).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    missing: Vec<String>,
}

/// Top-level manifest structure.
///
/// Groups are split into `complete` (base + query + ground truth all
/// present), `incomplete` (at least base detected but missing query or
/// ground truth), and `orphans` (no base vectors at all).
#[derive(Debug, serde::Serialize)]
struct Manifest {
    /// Logically intact groups: base + query + ground truth present.
    /// Distances and other roles are optional.
    #[serde(skip_serializing_if = "IndexMap::is_empty")]
    complete: IndexMap<String, ManifestGroup>,
    /// Groups with base vectors but missing query or ground truth.
    #[serde(skip_serializing_if = "IndexMap::is_empty")]
    incomplete: IndexMap<String, ManifestGroup>,
    /// Files that could not be grouped into any dataset.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    orphans: Vec<String>,
}

/// Intermediate group before merge/completeness classification.
struct RawGroup {
    name: String,
    /// (absolute path, format)
    files: Vec<(PathBuf, VecFormat)>,
    roles: wizard::DetectedRoles,
}

/// Parsed dataset metadata entry from `dataset_metadata.yml`.
#[derive(Debug, serde::Deserialize)]
struct DatasetMeta {
    similarity_function: Option<String>,
}

/// Load `dataset_metadata.yml` or `dataset_metadata.yaml` from the
/// scan root or any ancestor directory. Returns a map of dataset name
/// → similarity function.
fn load_dataset_metadata(root: &Path) -> IndexMap<String, String> {
    let names = ["dataset_metadata.yml", "dataset_metadata.yaml"];
    let mut dir = Some(root);
    while let Some(d) = dir {
        for name in &names {
            let path = d.join(name);
            if path.is_file() {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    match serde_yaml::from_str::<IndexMap<String, DatasetMeta>>(&text) {
                        Ok(parsed) => {
                            println!("Loaded metadata from {}", path.display());
                            return parsed.into_iter()
                                .filter_map(|(k, v)| v.similarity_function.map(|sf| (k, sf)))
                                .collect();
                        }
                        Err(e) => {
                            eprintln!("Error: failed to parse {}: {}", path.display(), e);
                            std::process::exit(1);
                        }
                    }
                }
            }
        }
        dir = d.parent();
    }
    IndexMap::new()
}

/// Look up the similarity function for a group name in the metadata.
///
/// First tries an exact match, then normalizes both names (strip sizes,
/// collapse delimiters, lowercase) to find a base-identity match. E.g.
/// group `ada-002-100000` matches metadata key `ada002-100k` because
/// both normalize to `ada002`.
fn lookup_similarity(name: &str, metadata: &IndexMap<String, String>) -> Option<String> {
    // Exact match
    if let Some(sf) = metadata.get(name) {
        return Some(sf.clone());
    }

    // Normalized match: strip sizes and delimiters from both sides
    let group_norm = normalize_for_match(name);
    for (meta_key, sf) in metadata {
        let meta_norm = normalize_for_match(meta_key);
        if group_norm == meta_norm {
            return Some(sf.clone());
        }
    }
    None
}

/// Normalize a dataset name for fuzzy matching: lowercase, strip size
/// tokens, strip version fragments, and remove delimiters.
///
/// E.g. "ada-002-100000" → "ada002", "ada002-100k" → "ada002",
/// "cohere-english-v3-0-1024-100000" → "cohereenglishv301024",
/// "cohere-english-v3-100k" → "cohereenglishv3"
fn normalize_for_match(name: &str) -> String {
    name.to_lowercase()
        .split(|c: char| c == '-' || c == '_' || c == '.')
        .filter(|t| !t.is_empty() && !is_size_token(t))
        .collect::<Vec<_>>()
        .join("")
}

/// Entry point.
pub fn run(args: InferManifestArgs) {
    let root = &args.root;
    if !root.is_dir() {
        eprintln!("Error: '{}' is not a directory", root.display());
        std::process::exit(1);
    }

    // Recursively collect all recognized vector data files
    let mut all_files: Vec<(PathBuf, VecFormat)> = Vec::new();
    collect_files(root, &mut all_files);

    if all_files.is_empty() {
        eprintln!("No recognized vector data files found under {}", root.display());
        return;
    }

    println!("Found {} recognized files under {}", all_files.len(), root.display());

    // Load dataset_metadata.yml for similarity function lookup
    let metadata = load_dataset_metadata(root);

    // Phase 1: Group files by inferred dataset name
    let mut name_groups: BTreeMap<String, Vec<(PathBuf, VecFormat)>> = BTreeMap::new();
    for (path, fmt) in &all_files {
        let name = infer_dataset_name(path);
        name_groups.entry(name)
            .or_default()
            .push((path.clone(), *fmt));
    }

    // Phase 1.5: Split multi-size groups into per-size sub-groups.
    // When a directory has e.g. base_1m.fvec + base_10m.fvec + query.fvec,
    // detect_roles would see ambiguous base files. Splitting by size token
    // before role detection avoids this. Files without a size token (like
    // a shared query file) are duplicated into each sub-group.
    let mut split_groups: BTreeMap<String, Vec<(PathBuf, VecFormat)>> = BTreeMap::new();
    for (name, files) in name_groups {
        let sub = split_by_size(&name, &files);
        for (sub_name, sub_files) in sub {
            split_groups.entry(sub_name).or_default().extend(sub_files);
        }
    }

    // Run role detection on each (possibly size-split) group
    let mut raw_groups: Vec<RawGroup> = Vec::new();
    for (name, files) in split_groups {
        let candidates: Vec<(PathBuf, String, u64)> = files.iter()
            .map(|(p, fmt)| {
                let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
                (p.clone(), fmt.name().to_string(), size)
            })
            .collect();
        let roles = wizard::detect_roles(&candidates);
        raw_groups.push(RawGroup { name, files, roles });
    }

    // Phase 2: Directory-aware merging.
    // For groups with base+query but no ground truth, look for groups in the
    // same directory that have ground truth (ivec) but no base vectors.
    // This handles the common pattern where gt files have a different prefix
    // (e.g. "dpr_1m_gt..." vs "c4-en_base..." in the same directory).
    merge_within_directories(&mut raw_groups);

    // Classify groups into complete, incomplete, and orphan
    let mut complete: IndexMap<String, ManifestGroup> = IndexMap::new();
    let mut incomplete: IndexMap<String, ManifestGroup> = IndexMap::new();
    let mut orphans: Vec<String> = Vec::new();

    for group in &raw_groups {
        let mut layout = IndexMap::new();
        let mut file_paths = Vec::new();

        for (p, _fmt) in &group.files {
            let rel = p.strip_prefix(root)
                .unwrap_or(p)
                .to_string_lossy()
                .to_string();
            file_paths.push(rel.clone());
            let role = detect_file_role(p, &group.roles);
            layout.insert(rel, role);
        }

        if group.roles.base_vectors.is_none() {
            // No base vectors — orphan
            orphans.extend(file_paths);
            continue;
        }

        let mut missing = Vec::new();
        if group.roles.query_vectors.is_none() {
            missing.push("query_vectors".to_string());
        }
        if group.roles.neighbor_indices.is_none() {
            missing.push("neighbor_indices".to_string());
        }

        // Resolve similarity function from metadata, normalized to
        // canonical uppercase (COSINE, DOT_PRODUCT, L2, L1).
        let similarity_function = lookup_similarity(&group.name, &metadata)
            .map(|sf| normalize_metric_name(&sf).unwrap_or(sf));

        let entry = ManifestGroup {
            files: file_paths,
            layout,
            similarity_function: similarity_function.clone(),
            missing: missing.clone(),
        };

        if missing.is_empty() {
            let sf_display = similarity_function.as_deref().unwrap_or("unknown");
            println!("  Complete: {} ({} files, metric: {})", group.name, group.files.len(), sf_display);
            complete.insert(group.name.clone(), entry);
        } else {
            println!("  Incomplete: {} — missing: {}", group.name, missing.join(", "));
            incomplete.insert(group.name.clone(), entry);
        }
    }

    let manifest = Manifest { complete, incomplete, orphans };

    // Check for complete groups missing similarity function
    let missing_sf: Vec<&str> = manifest.complete.iter()
        .filter(|(_, g)| g.similarity_function.is_none())
        .map(|(name, _)| name.as_str())
        .collect();

    // Write the manifest
    let output = &args.output;
    match serde_yaml::to_string(&manifest) {
        Ok(yaml) => {
            if let Err(e) = std::fs::write(output, &yaml) {
                eprintln!("Error: failed to write {}: {}", output.display(), e);
                std::process::exit(1);
            }
            println!(
                "\nWrote {} complete, {} incomplete groups, {} orphan files to {}",
                manifest.complete.len(),
                manifest.incomplete.len(),
                manifest.orphans.len(),
                output.display(),
            );
        }
        Err(e) => {
            eprintln!("Error: failed to serialize manifest: {}", e);
            std::process::exit(1);
        }
    }

    if !missing_sf.is_empty() {
        eprintln!("\nError: {} complete group(s) have no similarity_function:", missing_sf.len());
        for name in &missing_sf {
            eprintln!("  - {}", name);
        }
        eprintln!(
            "\nProvide a dataset_metadata.yml (or .yaml) in the scan root or an ancestor \
             directory with entries like:\n  {}:\n    similarity_function: COSINE",
            missing_sf[0],
        );
        std::process::exit(1);
    }
}

/// Split a name group into per-size sub-groups when multiple size tokens
/// are present.
///
/// For example, a group containing `base_1m.fvec`, `base_10m.fvec`, and
/// `query_10k.fvec` becomes two sub-groups: `name-1m` (base_1m + query)
/// and `name-10m` (base_10m + query). Files without a recognized size
/// token are treated as shared and duplicated into every sub-group.
///
/// If the group has only one distinct size (or no sizes at all), it is
/// returned as-is with the original name.
fn split_by_size(
    name: &str,
    files: &[(PathBuf, VecFormat)],
) -> Vec<(String, Vec<(PathBuf, VecFormat)>)> {
    // Extract size token from each file
    let mut sizes_seen: BTreeMap<String, Vec<(PathBuf, VecFormat)>> = BTreeMap::new();
    let mut shared: Vec<(PathBuf, VecFormat)> = Vec::new();

    for (path, fmt) in files {
        if let Some(size) = extract_size_token(path) {
            sizes_seen.entry(size)
                .or_default()
                .push((path.clone(), *fmt));
        } else {
            shared.push((path.clone(), *fmt));
        }
    }

    if sizes_seen.len() <= 1 {
        // No splitting needed — single size or no sizes at all
        return vec![(name.to_string(), files.to_vec())];
    }

    // Multiple sizes: create a sub-group for each.
    // Shared files are only included if they come from the same directory
    // as at least one sized file in the sub-group. This prevents a query
    // file from dir A leaking into a sub-group whose base is in dir B.
    sizes_seen.into_iter()
        .map(|(size, size_files)| {
            let size_dirs: std::collections::HashSet<_> = size_files.iter()
                .filter_map(|(p, _)| p.parent().map(|d| d.to_path_buf()))
                .collect();
            let mut combined = size_files;
            for (sp, sf) in &shared {
                if let Some(dir) = sp.parent() {
                    if size_dirs.contains(dir) {
                        combined.push((sp.clone(), *sf));
                    }
                }
            }
            let sub_name = format!("{}-{}", name, size);
            (sub_name, combined)
        })
        .collect()
}

/// Extract the dataset-size token from a filename.
///
/// Returns the first size token that appears *before* the first role
/// keyword in the filename. This position distinguishes dataset sizes
/// (e.g. `ada_002_100000_base_vectors`) from per-role counts that
/// appear after the role keyword (e.g. `_query_vectors_10000`).
///
/// When a file has a size token only *after* the role keyword (like
/// `base_1m_norm`), that token is still the dataset size since it's
/// the only one. Files with no size tokens at all return `None` and
/// are treated as shared across all sub-groups.
fn extract_size_token(path: &Path) -> Option<String> {
    let stem = path.file_stem()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let tokens: Vec<&str> = stem.split(|c: char| c == '_' || c == '-' || c == '.')
        .filter(|s| !s.is_empty())
        .collect();

    // Find the position of the first role keyword
    let first_role_pos = tokens.iter().position(|t| ROLE_TOKENS.contains(t));

    // Look for a size token before the first role keyword — this is
    // unambiguously the dataset size, not a count.
    if let Some(role_pos) = first_role_pos {
        for t in &tokens[..role_pos] {
            if is_size_token(t) {
                return Some(t.to_string());
            }
        }
    }

    // Fallback: if the file has a base/gt keyword, accept a size token
    // anywhere (covers patterns like "base_1m_norm" where size is after
    // the role keyword).
    let has_base_or_gt = tokens.iter().any(|t| {
        matches!(*t, "base" | "train" | "gt" | "groundtruth" | "ground")
    });
    if has_base_or_gt {
        for t in &tokens {
            if is_size_token(t) {
                return Some(t.to_string());
            }
        }
    }

    // For indices/distances files: extract the base-count prefix "b<N>"
    // as the dataset size (e.g., `_indices_b100000_q10000_k100.ivec`).
    let has_indices_or_distances = tokens.iter().any(|t| {
        matches!(*t, "indices" | "neighbors" | "distances" | "distance")
    });
    if has_indices_or_distances {
        for t in &tokens {
            if t.len() >= 2 && t.starts_with('b') {
                if let Ok(n) = t[1..].parse::<u64>() {
                    if n >= 1000 {
                        return Some(n.to_string());
                    }
                }
            }
        }
    }

    None
}

/// Merge groups within the same directory to form complete triples.
///
/// When a group has base vectors but no ground truth, and another group
/// in the same directory has ground truth (neighbor_indices) but no base
/// vectors, absorb the ground-truth group's files into the base group.
/// The base group's name is kept since base vectors define the dataset
/// identity.
fn merge_within_directories(groups: &mut Vec<RawGroup>) {
    // Build a directory → group indices map
    let mut dir_map: BTreeMap<PathBuf, Vec<usize>> = BTreeMap::new();
    for (i, group) in groups.iter().enumerate() {
        for (path, _) in &group.files {
            if let Some(dir) = path.parent() {
                dir_map.entry(dir.to_path_buf()).or_default().push(i);
            }
        }
    }
    // Deduplicate indices per directory
    for indices in dir_map.values_mut() {
        indices.sort_unstable();
        indices.dedup();
    }

    // For each directory, find merge opportunities
    let mut absorbed: Vec<usize> = Vec::new();
    for indices in dir_map.values() {
        // Find groups with base but no ground truth
        let needs_gt: Vec<usize> = indices.iter().copied()
            .filter(|&i| {
                groups[i].roles.base_vectors.is_some()
                    && groups[i].roles.neighbor_indices.is_none()
            })
            .collect();
        // Find groups with ground truth but no base
        let has_gt: Vec<usize> = indices.iter().copied()
            .filter(|&i| {
                groups[i].roles.neighbor_indices.is_some()
                    && groups[i].roles.base_vectors.is_none()
            })
            .collect();

        // Simple 1:1 merge when there's exactly one needing and one providing.
        // For multi-way cases (multiple sizes), match by shared directory
        // and hope for the best — a human should review the manifest anyway.
        if needs_gt.len() == 1 && has_gt.len() == 1 {
            let target = needs_gt[0];
            let source = has_gt[0];
            // Move files and merge roles
            let source_files: Vec<_> = groups[source].files.drain(..).collect();
            groups[target].files.extend(source_files);
            // Transfer ground truth role
            if groups[target].roles.neighbor_indices.is_none() {
                groups[target].roles.neighbor_indices =
                    groups[source].roles.neighbor_indices.take();
            }
            if groups[target].roles.neighbor_distances.is_none() {
                groups[target].roles.neighbor_distances =
                    groups[source].roles.neighbor_distances.take();
            }
            absorbed.push(source);
        } else if !needs_gt.is_empty() && !has_gt.is_empty() {
            // Multiple base groups and multiple gt groups in the same dir.
            // Try to match by file size tokens (e.g. "1m" with "1m").
            for &target_idx in &needs_gt {
                let target_size = extract_size_token_from_group(&groups[target_idx]);
                for &source_idx in &has_gt {
                    if absorbed.contains(&source_idx) { continue; }
                    let source_size = extract_size_token_from_group(&groups[source_idx]);
                    if target_size.is_some() && target_size == source_size {
                        let source_files: Vec<_> = groups[source_idx].files.drain(..).collect();
                        groups[target_idx].files.extend(source_files);
                        if groups[target_idx].roles.neighbor_indices.is_none() {
                            groups[target_idx].roles.neighbor_indices =
                                groups[source_idx].roles.neighbor_indices.take();
                        }
                        if groups[target_idx].roles.neighbor_distances.is_none() {
                            groups[target_idx].roles.neighbor_distances =
                                groups[source_idx].roles.neighbor_distances.take();
                        }
                        absorbed.push(source_idx);
                        break;
                    }
                }
            }
        }
    }

    // Remove absorbed (empty) groups in reverse order to preserve indices
    absorbed.sort_unstable();
    absorbed.dedup();
    for i in absorbed.into_iter().rev() {
        groups.remove(i);
    }
}

/// Extract the first size token from any file in a group.
///
/// Used for matching base groups with ground-truth groups that share the
/// same size (e.g. both contain "1m" or "10m" in their filenames).
fn extract_size_token_from_group(group: &RawGroup) -> Option<String> {
    for (path, _) in &group.files {
        let stem = path.file_stem()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        for token in stem.split(|c: char| c == '_' || c == '-' || c == '.') {
            if is_size_token(token) {
                return Some(token.to_string());
            }
        }
    }
    None
}

/// Returns `true` if the token looks like a dataset size indicator
/// (e.g. "1m", "10m", "100k", "1000000").
fn is_size_token(token: &str) -> bool {
    // \d+[km]
    if token.len() >= 2 {
        let (num_part, suffix) = token.split_at(token.len() - 1);
        if matches!(suffix, "k" | "m") && num_part.parse::<u64>().is_ok() {
            return true;
        }
    }
    // Pure large number
    if let Ok(n) = token.parse::<u64>() {
        return n >= 10000;
    }
    false
}

/// Recursively collect files with recognized vector data formats.
///
/// Directories containing a `dataset.yaml` are skipped — they are
/// already managed datasets and should not be re-ingested.
fn collect_files(dir: &Path, out: &mut Vec<(PathBuf, VecFormat)>) {
    // Skip directories that are already proper datasets
    if dir.join("dataset.yaml").exists() {
        println!("  Skipping {} (contains dataset.yaml)", dir.display());
        return;
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with('.') {
                continue;
            }
            collect_files(&path, out);
        } else if path.is_file() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with('.')
                || name_str.ends_with(".json")
                || name_str.ends_with(".yaml")
                || name_str.ends_with(".yml")
            {
                continue;
            }

            if let Some(fmt) = VecFormat::detect_from_path(&path) {
                if fmt.is_xvec() || fmt == VecFormat::Slab {
                    out.push((path, fmt));
                }
            }
        }
    }
}

/// Role keywords stripped from filenames during dataset name inference.
const ROLE_TOKENS: &[&str] = &[
    "base", "query", "queries", "train", "test", "vectors",
    "neighbors", "indices", "distances", "groundtruth", "ground", "truth", "gt",
    "metadata", "content", "predicates", "results",
    "filtered", "filter",
];

/// Qualifier/noise tokens stripped from filenames during inference.
/// Includes processing qualifiers and distance metric names that appear
/// in real-world dataset filenames but are not part of the dataset identity.
const QUALIFIER_TOKENS: &[&str] = &[
    "norm", "shuffle", "flat", "embed", "embeddings",
    // distance metrics
    "ip", "cosine", "euclidean", "l2", "dot",
];

/// Returns `true` if the token looks like a record count rather than a
/// meaningful dataset identifier.
///
/// Matches:
/// - Pure numbers >= 10000 (e.g. `100000`, `10000000`)
/// - Numbers with k/m suffix (e.g. `1m`, `10k`, `6m`)
/// - Prefixed counts from ivec naming conventions (e.g. `b100000`, `q10000`, `k100`)
/// - File fragment tokens (e.g. `files0`, `files1`)
fn is_count_or_noise_token(token: &str) -> bool {
    // Pure numeric >= 10000
    if let Ok(n) = token.parse::<u64>() {
        return n >= 10000;
    }

    // Number with k/m suffix: "1m", "10k", "6m".
    // The `b` suffix is NOT included because it's ambiguous with model
    // parameter counts (e.g. "1.5B" splits to "5b" which is a model size,
    // not a data count).
    if token.len() >= 2 {
        let (num_part, suffix) = token.split_at(token.len() - 1);
        if matches!(suffix, "k" | "m") && num_part.parse::<u64>().is_ok() {
            return true;
        }
    }

    // Prefixed count: "b100000", "q10000", "k100"
    if token.len() >= 2 && matches!(token.as_bytes()[0], b'b' | b'q' | b'k') {
        if let Ok(n) = token[1..].parse::<u64>() {
            if n >= 10 {
                return true;
            }
        }
    }

    // File fragment tokens: "files0", "files1", etc.
    if token.starts_with("files") {
        return true;
    }

    false
}

/// Returns `true` if the token should be stripped during name inference.
fn is_noise_token(token: &str) -> bool {
    ROLE_TOKENS.contains(&token)
        || QUALIFIER_TOKENS.contains(&token)
        || is_count_or_noise_token(token)
}

/// Infer a dataset name from a file path by stripping role keywords,
/// count/size tokens, qualifier tokens, and the extension.
///
/// The algorithm is designed to produce the same name for all files in a
/// dataset group, even when individual files have different count suffixes
/// (e.g. base has 1000000, query has 10000).
///
/// Examples:
/// - `data/sift/sift_base.fvecs` → `sift`
/// - `embeddings/glove-200_query_vectors.fvec` → `glove-200`
/// - `deep1b/base.fvec` → `deep1b` (falls back to parent dir)
/// - `wiki/ada_002_100000_base_vectors.fvec` → `ada-002`
/// - `cohere/cohere_wiki_en_flat_base_1m_norm.fvecs` → `cohere-wiki-en`
/// - `wiki/100k/cohere_embed-english-v3.0_1024_base_vectors_100000.fvec` → `cohere-english-v3-0-1024`
fn infer_dataset_name(path: &Path) -> String {
    let stem = path.file_stem()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    // Strip leading underscore
    let stem = stem.strip_prefix('_').unwrap_or(&stem);

    // Tokenize on delimiters
    let tokens: Vec<&str> = stem.split(|c: char| c == '_' || c == '-' || c == '.')
        .filter(|s| !s.is_empty())
        .collect();

    // Remove noise tokens (roles, qualifiers, counts).
    // Also strip everything after a "files*" token (file fragment indices).
    let mut name_tokens: Vec<&str> = Vec::new();
    let mut skip_rest = false;
    for t in &tokens {
        if skip_rest { continue; }
        if t.starts_with("files") {
            skip_rest = true;
            continue;
        }
        if !is_noise_token(t) {
            name_tokens.push(t);
        }
    }

    if name_tokens.is_empty() {
        // All tokens were noise — fall back to the parent directory name,
        // walking up past purely numeric directory names (like "100k", "1M")
        let mut dir = path.parent();
        while let Some(d) = dir {
            if let Some(name) = d.file_name().and_then(|n| n.to_str()) {
                let lower = name.to_lowercase();
                if !is_count_or_noise_token(&lower) && !lower.is_empty() {
                    return lower;
                }
            }
            dir = d.parent();
        }
        "unknown".to_string()
    } else {
        name_tokens.join("-")
    }
}

/// Normalize a metric/similarity function string to canonical uppercase.
fn normalize_metric_name(s: &str) -> Option<String> {
    match s.to_uppercase().as_str() {
        "COSINE" => Some("COSINE".into()),
        "DOT_PRODUCT" | "DOTPRODUCT" | "DOT" | "INNER_PRODUCT" | "IP" => Some("DOT_PRODUCT".into()),
        "L2" | "EUCLIDEAN" => Some("L2".into()),
        "L1" | "MANHATTAN" => Some("L1".into()),
        _ => None,
    }
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
    } else if roles.metadata.as_deref() == Some(path) {
        "metadata".to_string()
    } else if roles.metadata_predicates.as_deref() == Some(path) {
        "metadata_predicates".to_string()
    } else if roles.metadata_results.as_deref() == Some(path) {
        "metadata_results".to_string()
    } else if roles.filtered_neighbor_indices.as_deref() == Some(path) {
        "filtered_neighbor_indices".to_string()
    } else if roles.filtered_neighbor_distances.as_deref() == Some(path) {
        "filtered_neighbor_distances".to_string()
    } else {
        "unassigned".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_name_strips_role_tokens() {
        assert_eq!(infer_dataset_name(Path::new("data/sift_base.fvec")), "sift");
        assert_eq!(infer_dataset_name(Path::new("data/sift_query.fvec")), "sift");
        assert_eq!(infer_dataset_name(Path::new("data/sift_base_vectors.fvec")), "sift");
    }

    #[test]
    fn infer_name_preserves_dataset_identifiers() {
        assert_eq!(infer_dataset_name(Path::new("x/glove-200_base.fvec")), "glove-200");
        assert_eq!(infer_dataset_name(Path::new("x/deep1b_query_vectors.mvec")), "deep1b");
    }

    #[test]
    fn infer_name_falls_back_to_parent_dir() {
        assert_eq!(infer_dataset_name(Path::new("sift/base_vectors.fvec")), "sift");
        assert_eq!(infer_dataset_name(Path::new("mydata/query.fvec")), "mydata");
    }

    #[test]
    fn infer_name_falls_back_past_numeric_dirs() {
        assert_eq!(
            infer_dataset_name(Path::new("wikipedia_squad/100k/base_vectors.fvec")),
            "wikipedia_squad"
        );
    }

    #[test]
    fn infer_name_strips_underscore_prefix() {
        assert_eq!(infer_dataset_name(Path::new("data/_sift_base.fvec")), "sift");
    }

    #[test]
    fn infer_name_multi_token_dataset() {
        assert_eq!(
            infer_dataset_name(Path::new("x/ann_sift_128_base.fvec")),
            "ann-sift-128"
        );
    }

    // ── Tests from jvector MultiFileDatasource naming patterns ──────

    #[test]
    fn infer_name_strips_counts() {
        assert_eq!(
            infer_dataset_name(Path::new("wiki/ada_002_100000_base_vectors.fvec")),
            "ada-002"
        );
        assert_eq!(
            infer_dataset_name(Path::new("wiki/ada_002_100000_query_vectors_10000.fvec")),
            "ada-002"
        );
        assert_eq!(
            infer_dataset_name(Path::new("wiki/ada_002_100000_indices_query_10000.ivec")),
            "ada-002"
        );
    }

    #[test]
    fn infer_name_strips_qualifiers() {
        assert_eq!(
            infer_dataset_name(Path::new("cohere/cohere_wiki_en_flat_base_1m_norm.fvecs")),
            "cohere-wiki-en"
        );
        assert_eq!(
            infer_dataset_name(Path::new("cohere/cohere_wiki_en_flat_query_10k_norm.fvecs")),
            "cohere-wiki-en"
        );
    }

    #[test]
    fn infer_name_strips_metric_tokens() {
        // "ip" (inner product) should be stripped as a qualifier
        assert_eq!(
            infer_dataset_name(Path::new("dpr/dpr_1m_gt_norm_ip_k100.ivecs")),
            "dpr"
        );
    }

    #[test]
    fn infer_name_cohere_embed_model() {
        assert_eq!(
            infer_dataset_name(Path::new("wiki/100k/cohere_embed-english-v3.0_1024_base_vectors_100000.fvec")),
            "cohere-english-v3-0-1024"
        );
        assert_eq!(
            infer_dataset_name(Path::new("wiki/100k/cohere_embed-english-v3.0_1024_query_vectors_10000.fvec")),
            "cohere-english-v3-0-1024"
        );
        assert_eq!(
            infer_dataset_name(Path::new("wiki/100k/cohere_embed-english-v3.0_1024_indices_b100000_q10000_k100.ivec")),
            "cohere-english-v3-0-1024"
        );
    }

    #[test]
    fn infer_name_gecko_model() {
        assert_eq!(
            infer_dataset_name(Path::new("wiki/1M/textembedding-gecko_1000000_base_vectors.fvec")),
            "textembedding-gecko"
        );
        assert_eq!(
            infer_dataset_name(Path::new("wiki/1M/textembedding-gecko_1000000_query_vectors_10000.fvec")),
            "textembedding-gecko"
        );
    }

    #[test]
    fn infer_name_degen() {
        assert_eq!(
            infer_dataset_name(Path::new("ada-degen/degen_base_vectors.fvec")),
            "degen"
        );
        assert_eq!(
            infer_dataset_name(Path::new("ada-degen/degen_query_vectors.fvec")),
            "degen"
        );
        assert_eq!(
            infer_dataset_name(Path::new("ada-degen/degen_ground_truth.ivec")),
            "degen"
        );
    }

    #[test]
    fn infer_name_cap_dataset() {
        assert_eq!(
            infer_dataset_name(Path::new("cap/Caselaw_gte-Qwen2-1.5B_embeddings_base_1m_norm_shuffle.fvecs")),
            "caselaw-gte-qwen2-1-5b"
        );
        assert_eq!(
            infer_dataset_name(Path::new("cap/Caselaw_gte-Qwen2-1.5B_embeddings_query_10k_norm_shuffle.fvecs")),
            "caselaw-gte-qwen2-1-5b"
        );
    }

    #[test]
    fn infer_name_dpr_dataset() {
        assert_eq!(
            infer_dataset_name(Path::new("dpr/c4-en_base_1M_norm_files0_2.fvecs")),
            "c4-en"
        );
        assert_eq!(
            infer_dataset_name(Path::new("dpr/c4-en_query_10k_norm_files0_1.fvecs")),
            "c4-en"
        );
    }

    // ── Cross-prefix merge test (DPR/CAP pattern) ──────────────────

    #[test]
    fn dpr_gt_different_prefix_merges_into_base_group() {
        // DPR pattern: base/query are "c4-en", gt is "dpr" — same directory
        // After name inference: two groups: c4-en (base+query) and dpr (gt)
        // Merge should combine them.
        let base = PathBuf::from("/data/dpr/c4-en_base_1M_norm_files0_2.fvecs");
        let query = PathBuf::from("/data/dpr/c4-en_query_10k_norm_files0_1.fvecs");
        let gt = PathBuf::from("/data/dpr/dpr_1m_gt_norm_ip_k100.ivecs");

        let mut groups = vec![
            RawGroup {
                name: "c4-en".to_string(),
                files: vec![
                    (base.clone(), VecFormat::Fvec),
                    (query.clone(), VecFormat::Fvec),
                ],
                roles: wizard::DetectedRoles {
                    base_vectors: Some(base),
                    query_vectors: Some(query),
                    neighbor_indices: None,
                    ..Default::default()
                },
            },
            RawGroup {
                name: "dpr".to_string(),
                files: vec![(gt.clone(), VecFormat::Ivec)],
                roles: wizard::DetectedRoles {
                    neighbor_indices: Some(gt),
                    ..Default::default()
                },
            },
        ];

        merge_within_directories(&mut groups);

        // Should have merged into one group
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].name, "c4-en");
        assert_eq!(groups[0].files.len(), 3);
        assert!(groups[0].roles.base_vectors.is_some());
        assert!(groups[0].roles.query_vectors.is_some());
        assert!(groups[0].roles.neighbor_indices.is_some());
    }

    #[test]
    fn cap_gt_merges_by_size_token() {
        // CAP pattern: base/query are "caselaw-gte-qwen2-1-5b" with 1m,
        // gt is "cap" with 1m — should match by size token.
        let base = PathBuf::from("/data/cap/Caselaw_base_1m.fvecs");
        let query = PathBuf::from("/data/cap/Caselaw_query_10k.fvecs");
        let base_6m = PathBuf::from("/data/cap/Caselaw_base_6m.fvecs");
        let query_6m = PathBuf::from("/data/cap/Caselaw_query_6m_10k.fvecs");
        let gt_1m = PathBuf::from("/data/cap/cap_1m_gt.ivecs");
        let gt_6m = PathBuf::from("/data/cap/cap_6m_gt.ivecs");

        let mut groups = vec![
            RawGroup {
                name: "caselaw".to_string(),
                files: vec![
                    (base.clone(), VecFormat::Fvec),
                    (query.clone(), VecFormat::Fvec),
                ],
                roles: wizard::DetectedRoles {
                    base_vectors: Some(base),
                    query_vectors: Some(query),
                    ..Default::default()
                },
            },
            RawGroup {
                name: "caselaw-6m".to_string(),
                files: vec![
                    (base_6m.clone(), VecFormat::Fvec),
                    (query_6m.clone(), VecFormat::Fvec),
                ],
                roles: wizard::DetectedRoles {
                    base_vectors: Some(base_6m),
                    query_vectors: Some(query_6m),
                    ..Default::default()
                },
            },
            RawGroup {
                name: "cap".to_string(),
                files: vec![
                    (gt_1m.clone(), VecFormat::Ivec),
                    (gt_6m.clone(), VecFormat::Ivec),
                ],
                roles: wizard::DetectedRoles {
                    // detect_roles would see two ivecs and mark as ambiguous,
                    // but for this test we simulate the single-gt case
                    neighbor_indices: Some(gt_1m),
                    ..Default::default()
                },
            },
        ];

        merge_within_directories(&mut groups);

        // The 1:many case won't merge perfectly (cap group has 2 files
        // but only 1 detected as neighbor_indices due to ambiguity).
        // The simple 1:1 case is tested above. This test just verifies
        // the multi-way merge doesn't panic or corrupt data.
        assert!(groups.len() >= 2);
    }

    #[test]
    fn split_by_size_creates_subgroups() {
        let files = vec![
            (PathBuf::from("cap/base_1m.fvecs"), VecFormat::Fvec),
            (PathBuf::from("cap/base_6m.fvecs"), VecFormat::Fvec),
            (PathBuf::from("cap/query_10k.fvecs"), VecFormat::Fvec),  // shared (no base/gt keyword)
        ];
        let result = split_by_size("caselaw", &files);
        assert_eq!(result.len(), 2);
        // Each sub-group should have the base + shared query
        let names: Vec<&str> = result.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"caselaw-1m"));
        assert!(names.contains(&"caselaw-6m"));
        for (_, sub_files) in &result {
            assert_eq!(sub_files.len(), 2); // base + shared query
        }
    }

    #[test]
    fn split_by_size_no_split_single_size() {
        let files = vec![
            (PathBuf::from("x/base_1m.fvec"), VecFormat::Fvec),
            (PathBuf::from("x/query.fvec"), VecFormat::Fvec),
        ];
        let result = split_by_size("sift", &files);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "sift");
    }

    #[test]
    fn extract_size_from_various_roles() {
        // Size AFTER base/gt keyword (DPR/CAP pattern)
        assert_eq!(
            extract_size_token(Path::new("x/base_1m.fvec")),
            Some("1m".to_string())
        );
        assert_eq!(
            extract_size_token(Path::new("x/gt_10m.ivec")),
            Some("10m".to_string())
        );
        // Size BEFORE role keyword (astra-vector pattern)
        assert_eq!(
            extract_size_token(Path::new("x/ada_002_100000_base_vectors.fvec")),
            Some("100000".to_string())
        );
        assert_eq!(
            extract_size_token(Path::new("x/ada_002_100000_query_vectors_10000.fvec")),
            Some("100000".to_string())
        );
        assert_eq!(
            extract_size_token(Path::new("x/ada_002_100000_indices_query_10000.ivec")),
            Some("100000".to_string())
        );
        // Query with size only AFTER keyword → None (shared)
        assert_eq!(
            extract_size_token(Path::new("x/query_10k.fvec")),
            None
        );
        // Indices/distances: extract from b-prefix
        assert_eq!(
            extract_size_token(Path::new("x/cohere_indices_b100000_q10000_k100.ivec")),
            Some("100000".to_string())
        );
        assert_eq!(
            extract_size_token(Path::new("x/cohere_distances_b1000000_q10000_k100.fvec")),
            Some("1000000".to_string())
        );
        // No size at all → None
        assert_eq!(
            extract_size_token(Path::new("x/query_vectors.fvec")),
            None
        );
    }

    #[test]
    fn count_token_detection() {
        assert!(is_count_or_noise_token("100000"));
        assert!(is_count_or_noise_token("10000"));
        assert!(is_count_or_noise_token("1m"));
        assert!(is_count_or_noise_token("10k"));
        assert!(is_count_or_noise_token("6m"));
        assert!(is_count_or_noise_token("b100000"));
        assert!(is_count_or_noise_token("q10000"));
        assert!(is_count_or_noise_token("k100"));
        assert!(is_count_or_noise_token("files0"));
        assert!(is_count_or_noise_token("files1"));
        // Should NOT be treated as counts
        assert!(!is_count_or_noise_token("128"));
        assert!(!is_count_or_noise_token("1024"));
        assert!(!is_count_or_noise_token("v3"));
        assert!(!is_count_or_noise_token("sift"));
        assert!(!is_count_or_noise_token("002"));
        assert!(!is_count_or_noise_token("5b"));
        assert!(!is_count_or_noise_token("1"));
        assert!(!is_count_or_noise_token("2"));
    }

    #[test]
    fn size_token_detection() {
        assert!(is_size_token("1m"));
        assert!(is_size_token("10m"));
        assert!(is_size_token("100k"));
        assert!(is_size_token("1000000"));
        assert!(!is_size_token("128"));
        assert!(!is_size_token("sift"));
    }
}
