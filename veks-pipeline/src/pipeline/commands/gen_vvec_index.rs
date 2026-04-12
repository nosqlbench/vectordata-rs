// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate offset index files for variable-length
//! vector (vvec) files.
//!
//! Walks a directory tree, finds all `.vvec`/`.vvecs` files (including
//! legacy `.ivec` files that are variable-length), and creates
//! `IDXFOR__<name>.<i32|i64>` offset index files for each. These index
//! files enable O(1) random access on variable-length records and are
//! included in the published dataset for downstream consumers.

use std::path::{Path, PathBuf};
use std::time::Instant;

use veks_core::formats::VecFormat;
use vectordata::io::IndexedXvecReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};

/// Pipeline command: generate vvec offset indices.
pub struct GenerateVvecIndexOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateVvecIndexOp)
}

impl CommandOp for GenerateVvecIndexOp {
    fn command_path(&self) -> &str {
        "generate vvec-index"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Build offset indices for variable-length vector files".into(),
            body: format!(
                r#"# generate vvec-index

Build offset index files for variable-length vector (vvec) files.

## Description

Walks the source directory, finds all variable-length vector files
(`.ivvec`, `.fvvec`, `.bvvec`, etc., plus legacy `.ivec` files that
have non-uniform record dimensions), and creates `IDXFOR__<name>.<i32|i64>`
offset index files alongside each.

The index maps ordinal → byte offset, enabling O(1) random access on
files where records have different lengths. Index files are published
alongside the data so remote consumers can access records without
downloading the entire file.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let source_dir = if std::path::Path::new(source_str).is_absolute() {
            PathBuf::from(source_str)
        } else {
            ctx.workspace.join(source_str)
        };

        let mut created = 0u32;
        let mut skipped = 0u32;
        let mut errors = Vec::new();

        // Walk the directory tree
        let mut files_to_index = Vec::new();
        find_vvec_files(&source_dir, &mut files_to_index);

        for path in &files_to_index {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let format = match VecFormat::from_extension(ext) {
                Some(f) => f,
                None => continue,
            };

            let elem_size = format.element_size();
            if elem_size == 0 { continue; }

            // For explicitly vvec extensions, always build index
            // For legacy .ivec, check if actually variable-length
            if format.is_vvec() {
                match IndexedXvecReader::open(path, elem_size) {
                    Ok(r) => {
                        ctx.ui.log(&format!("  indexed {} ({} records)", rel_display(path, &source_dir), r.count()));
                        created += 1;
                    }
                    Err(e) => {
                        errors.push(format!("{}: {}", rel_display(path, &source_dir), e));
                    }
                }
            } else if format == VecFormat::Ivec {
                // Legacy .ivec — check if variable-length by trying uniform open
                let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                if file_size < 4 { continue; }

                // Read first dim to compute stride
                let dim_bytes = match std::fs::read(path).ok().and_then(|d| {
                    if d.len() >= 4 { Some(i32::from_le_bytes([d[0], d[1], d[2], d[3]])) } else { None }
                }) {
                    Some(d) if d > 0 => d as u64,
                    _ => continue,
                };
                let stride = 4 + dim_bytes * elem_size as u64;
                if file_size % stride != 0 {
                    // Variable-length — build index
                    match IndexedXvecReader::open(path, elem_size) {
                        Ok(r) => {
                            ctx.ui.log(&format!("  indexed {} ({} records, variable-length)",
                                rel_display(path, &source_dir), r.count()));
                            created += 1;
                        }
                        Err(e) => {
                            errors.push(format!("{}: {}", rel_display(path, &source_dir), e));
                        }
                    }
                } else {
                    skipped += 1;
                }
            }
        }

        if !errors.is_empty() {
            for e in &errors {
                ctx.ui.log(&format!("  ERROR: {}", e));
            }
        }

        let message = format!("{} index files created, {} uniform files skipped, {} errors",
            created, skipped, errors.len());

        CommandResult {
            status: if errors.is_empty() { Status::Ok } else { Status::Error },
            message,
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Directory to scan for vvec files".to_string(),
                role: OptionRole::Input,
            },
        ]
    }
}

/// Recursively find vvec and ivec files.
fn find_vvec_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if path.is_dir() {
            if name_str.starts_with('.') || name_str.starts_with('_') || name_str == ".cache" {
                continue;
            }
            find_vvec_files(&path, out);
        } else {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let format = VecFormat::from_extension(ext);
            if let Some(f) = format {
                if f.is_vvec() || f == VecFormat::Ivec {
                    out.push(path);
                }
            }
        }
    }
}

fn rel_display(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .map(|r| r.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string())
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}
