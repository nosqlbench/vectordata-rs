// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks prepare catalog generate` — write `catalog.json` /
//! `catalog.yaml` (and `knn_entries.yaml`) at each directory level of a
//! publish hierarchy.
//!
//! What a catalog lists and how its files are written is the shared
//! generator, [`vectordata::catalog::generate`], which the pipeline's
//! `catalog generate` step also runs, so a tree gets the same catalogs
//! from either. This command adds the choice of scan root (the publish
//! root above the input, or the publish roots below it), the
//! `.catalog_root` limit on how far up catalogs are written, and
//! `--update`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use vectordata::catalog::generate::{self, CatalogTree, GenerateEvent};

/// Run `veks catalog generate`.
///
/// Modes:
/// - `for_publish_url`: walk up to `.publish_url` root and generate catalogs
///   for the entire publish hierarchy.
/// - `update` (default true): only update catalog files that already exist in
///   the hierarchy. Directories with no existing catalog are skipped.
/// - Neither: generate catalogs at all hierarchy levels from `input` down.
///   If `.publish_url` is detected above `input` and `update` is false, warn
///   that partial catalogs may leave the publish hierarchy out of sync.
pub fn run(input: &Path, basename: &str, for_publish_url: bool, update: bool) {
    if basename.contains('.') {
        eprintln!("error: basename must not contain a dot");
        std::process::exit(1);
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // Display a path relative to cwd for user-facing output.
    let rel = |p: &Path| -> String {
        p.strip_prefix(&cwd)
            .map(|r| if r.as_os_str().is_empty() { ".".to_string() } else { r.to_string_lossy().to_string() })
            .unwrap_or_else(|_| p.to_string_lossy().to_string())
    };

    let input_path = generate::logical_path(input);

    if !input_path.is_dir() {
        eprintln!("error: {} is not a directory", rel(&input_path));
        std::process::exit(1);
    }

    // Resolve the effective scan root.
    //
    // 1. The closest `.publish_url` at or above input_path names it.
    // 2. Else each child directory holding a `.publish_url` gets its own
    //    run, and input_path a top-level catalog over them all.
    // 3. No `.publish_url` found → refuse to run.
    let scan_root = match crate::check::publish_url::find_publish_file(&input_path) {
        Some(pf) => {
            // Logical, not canonical: canonicalize would cross a
            // symlinked view into the physical path.
            let root = generate::logical_path(pf.parent().unwrap());
            eprintln!("Publish root: {} (from {})", rel(&root), rel(&pf));
            root
        }
        None => {
            let mut child_roots: Vec<PathBuf> = std::fs::read_dir(&input_path)
                .into_iter()
                .flatten()
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.is_dir() && path.join(".publish_url").exists())
                .collect();
            if child_roots.is_empty() {
                eprintln!("error: no .publish_url found at, within, or above {}", rel(&input_path));
                eprintln!("  Catalog generation requires a publish tree.");
                eprintln!("  Create one with: echo 's3://bucket/prefix/' > .publish_url");
                std::process::exit(1);
            }
            child_roots.sort();
            for child_root in &child_roots {
                eprintln!("Found child publish root: {}", rel(child_root));
                run(child_root, basename, for_publish_url, update);
            }
            eprintln!("Generating top-level catalog at {}", rel(&input_path));
            input_path.clone()
        }
    };

    // Scan from the publish root so that every dataset in the tree is
    // listed: running from one dataset directory must not drop its
    // siblings from the catalogs above it.
    let mut progress = |event: GenerateEvent| match event {
        GenerateEvent::Scanning(dir) => eprintln!("Scanning {} for datasets...", rel(dir)),
        GenerateEvent::Found(ds) => eprintln!(
            "  {} — {} profile(s) [{}]",
            ds.config().name,
            ds.config().profile_names().len(),
            rel(ds.yaml_path()),
        ),
        GenerateEvent::Unreadable { path, error } => eprintln!("WARNING: failed to load {}: {error}", rel(path)),
        GenerateEvent::Marked(dir) => eprintln!("  Skipping {} (.do_not_catalog)", rel(dir)),
        GenerateEvent::Wrote { entries, files, .. } => {
            let names: Vec<String> = files.iter().map(|f| rel(f)).collect();
            eprintln!("Wrote {entries} entries to {}", names.join(", "));
        }
    };
    let tree = CatalogTree::scan(&scan_root, &mut progress);
    eprintln!("Found {} publishable dataset(s)", tree.datasets().len());
    if tree.datasets().is_empty() {
        eprintln!("No publishable datasets found (no .publish sentinels).");
        return;
    }

    let mut catalog_dirs: BTreeSet<PathBuf> = tree.catalog_dirs();

    // Respect .catalog_root: don't write catalogs above it. The scan
    // root (from .publish_url) may be higher, but .catalog_root limits
    // how far up the catalog hierarchy extends.
    if let Some(catalog_root) = generate::find_catalog_root(&input_path) {
        let before = catalog_dirs.len();
        catalog_dirs.retain(|dir| dir.starts_with(&catalog_root));
        let skipped = before - catalog_dirs.len();
        if skipped > 0 {
            eprintln!("  .catalog_root at {} — skipping {} parent directories", rel(&catalog_root), skipped);
        }
    }

    // In --update mode, only directories that already have a catalog.
    if update && !for_publish_url {
        let before = catalog_dirs.len();
        catalog_dirs.retain(|dir| {
            dir.join(format!("{basename}.json")).exists() || dir.join(format!("{basename}.yaml")).exists()
        });
        let skipped = before - catalog_dirs.len();
        if skipped > 0 {
            eprintln!("  --update: skipping {} directories with no existing catalog", skipped);
        }
        if catalog_dirs.is_empty() {
            eprintln!("No existing catalog files found to update. Use --no-update to create new ones,");
            eprintln!("or --for-publish-url to generate the full publish hierarchy.");
            return;
        }
    }

    match tree.write(&catalog_dirs, basename, &mut progress) {
        Ok(files) => eprintln!(
            "{} datasets cataloged across {} directories ({} files written)",
            tree.datasets().len(),
            catalog_dirs.len(),
            files.len()
        ),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

