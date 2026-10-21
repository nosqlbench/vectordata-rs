// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! CLI support. `vectordata datasets init <dir>`: write the
//! `dataset.yaml` a directory of loose files implies
//! ([`crate::dataset::infer`]).

use std::path::{Path, PathBuf};

/// Arguments for `vectordata datasets init`.
#[cfg(feature = "cli")]
#[derive(Debug, veks_completion_derive::VeksCli)]
pub struct InitArgs {
    /// Replace a dataset.yaml that is already there.
    #[arg(long)]
    pub force: bool,
    /// The directory holding the files.
    pub dir: PathBuf,
}

/// What [`init`] wrote: the descriptor, and what it decided.
#[derive(Debug)]
pub struct Written {
    /// The file written.
    pub path: PathBuf,
    /// The inference it holds.
    pub inferred: crate::dataset::infer::Inferred,
}

/// Infer `dir`'s descriptor and write it as `dir/dataset.yaml`. An
/// existing one is replaced only with `force`; a `knn_entries.yaml`
/// already describes the directory and is refused, as the loader reads
/// it first.
pub fn init(dir: &Path, force: bool) -> Result<Written, String> {
    let path = dir.join("dataset.yaml");
    if path.exists() && !force {
        return Err(format!("{} exists; --force replaces it", path.display()));
    }
    if !path.exists() && dir.join("knn_entries.yaml").exists() {
        return Err(format!("{} already describes this directory", dir.join("knn_entries.yaml").display()));
    }
    let inferred = crate::dataset::infer::infer_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    crate::atomic_file::replace_config(&path, inferred.yaml.as_bytes(), false)
        .map_err(|e| format!("writing {}: {e}", path.display()))?;
    Ok(Written { path, inferred })
}

/// Run `vectordata datasets init`: write, then say what was decided.
#[cfg(feature = "cli")]
pub fn run(args: &InitArgs) -> i32 {
    match init(&args.dir, args.force) {
        Ok(w) => {
            for c in &w.inferred.classified {
                let profile = if c.profile == "default" { String::new() } else { format!("{}: ", c.profile) };
                println!("{profile}{} = {} (by {})", c.facet.key(), c.source, c.rule);
            }
            for note in &w.inferred.notes {
                println!("note: {note}");
            }
            println!("Wrote {}", w.path.display());
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}
