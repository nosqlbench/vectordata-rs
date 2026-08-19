// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: assert row-count (ordinal) alignment between artifacts.
//!
//! The passage pipeline's embed stage is an external contract: row i of the
//! vectors artifact must embed row i of `passages.parquet`. This command is
//! the enforcement point for that invariant — it compares the record counts
//! of a vectors artifact (`source`) and a reference artifact (`reference`),
//! optionally asserting the vector dimensionality, and fails the step on any
//! mismatch. Run it between the embed stage and `prepare bootstrap` so a
//! misaligned foreign artifact can never become a dataset's B facet.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
};
use veks_core::formats::VecFormat;
use veks_core::formats::passage_table::parquet_row_count;
use veks_core::formats::reader::probe_source;

/// Pipeline command: verify row-count alignment between two artifacts.
pub struct VerifyAlignmentOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(VerifyAlignmentOp)
}

impl CommandOp for VerifyAlignmentOp {
    fn command_path(&self) -> &str {
        "verify alignment"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_VERIFY
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Verify row-count alignment between two artifacts".into(),
            body: format!(
                r#"# verify alignment

Verify row-count alignment between two artifacts.

## Description

Asserts that a vectors artifact (`source` — npy file or directory, xvec,
hdf5, slab, or vector parquet) has exactly as many records as a reference
artifact (`reference` — any parquet table such as `passages.parquet`, or
another vectors artifact). Optionally asserts the vector dimensionality of
`source` with `dim`.

## Role in dataset pipelines

Ordinal identity between parallel artifacts is positional: row i of an
embedded vectors file corresponds to row i of the passage table it was
derived from. When embedding happens outside the pipeline (a foreign-input
contract), nothing else checks that correspondence — this command is the
gate. Place it after the embed stage and before `prepare bootstrap` so a
truncated, duplicated, or re-ordered-by-count vectors artifact fails fast
instead of becoming a dataset facet.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source = match options.require("source") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let reference = match options.require("reference") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let expect_dim: Option<u32> = match options.parse_opt("dim") {
            Ok(d) => d,
            Err(e) => return error_result(e, start),
        };

        let (source_count, source_dim) = match probe_vectors(&source) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let reference_count = match probe_reference(&reference) {
            Ok(c) => c,
            Err(e) => return error_result(e, start),
        };

        if source_count != reference_count {
            return error_result(
                format!(
                    "row-count misalignment: {} has {} record(s), {} has {} — \
                     ordinal identity is broken",
                    source.display(),
                    source_count,
                    reference.display(),
                    reference_count
                ),
                start,
            );
        }
        if let Some(expected) = expect_dim {
            match source_dim {
                Some(actual) if actual != expected => {
                    return error_result(
                        format!(
                            "dimension mismatch: {} has dim {}, expected {}",
                            source.display(),
                            actual,
                            expected
                        ),
                        start,
                    );
                }
                None => {
                    return error_result(
                        format!(
                            "dim assertion requested but {} carries no dimension metadata",
                            source.display()
                        ),
                        start,
                    );
                }
                Some(_) => {}
            }
        }

        let dim_note = match (expect_dim, source_dim) {
            (Some(d), _) => format!(", dim {}", d),
            (None, Some(d)) => format!(", dim {}", d),
            (None, None) => String::new(),
        };
        ctx.ui.log(&format!(
            "alignment verified: {} row(s){} in {} and {}",
            source_count,
            dim_note,
            source.display(),
            reference.display()
        ));

        CommandResult {
            status: Status::Ok,
            message: format!(
                "aligned: {} row(s){} in both artifacts",
                source_count, dim_note
            ),
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
                description: "Vectors artifact to check (npy, xvec, hdf5, slab, vector parquet)"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "reference".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Reference artifact whose row count must match (e.g. passages.parquet)"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "dim".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: None,
                description: "Also assert the source vector dimensionality".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id,
            self.command_path(),
            options,
            &["source", "reference"],
            &[],
        )
    }
}

/// Count records (and dimension, when the format carries one) of a vectors
/// artifact via the veks-core format probes.
fn probe_vectors(path: &Path) -> Result<(u64, Option<u32>), String> {
    let format = VecFormat::detect(path)
        .ok_or_else(|| format!("unrecognized artifact format: {}", path.display()))?;
    let meta = probe_source(path, format)?;
    let count = meta
        .record_count
        .ok_or_else(|| format!("cannot determine record count of {}", path.display()))?;
    let dim = if meta.dimension > 0 { Some(meta.dimension) } else { None };
    Ok((count, dim))
}

/// Count rows of the reference artifact. Parquet counts table rows straight
/// from footer metadata (works for non-vector tables like passages.parquet);
/// everything else goes through the vector probes.
fn probe_reference(path: &Path) -> Result<u64, String> {
    if path.extension().is_some_and(|e| e == "parquet") && path.is_file() {
        return parquet_row_count(path);
    }
    probe_vectors(path).map(|(count, _)| count)
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;
    use veks_core::formats::passage_table::{PassageRow, PassageTableWriter};

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    /// Write a minimal C-order f32 `.npy` of shape [rows, dim].
    fn write_npy(path: &Path, rows: usize, dim: usize) {
        use std::io::Write;
        let header_body = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
            rows, dim
        );
        // Pad with spaces so (magic 8 + 2 len bytes + header) % 64 == 0.
        let unpadded = 10 + header_body.len() + 1;
        let padding = (64 - unpadded % 64) % 64;
        let header = format!("{}{}\n", header_body, " ".repeat(padding));
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(b"\x93NUMPY\x01\x00").unwrap();
        f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        for i in 0..rows * dim {
            f.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
    }

    fn write_passages(path: &Path, rows: usize) {
        let mut w = PassageTableWriter::create(path).unwrap();
        for i in 0..rows {
            w.push(&PassageRow {
                corpusid: i as i64 / 10,
                section: "S".into(),
                ordinal: (i % 10) as i32,
                char_start: 0,
                char_end: 1,
                text: format!("t{}", i),
            })
            .unwrap();
        }
        w.finish().unwrap();
    }

    fn run(dir: &Path, source: &Path, reference: &Path, dim: Option<&str>) -> CommandResult {
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("reference", reference.to_string_lossy().to_string());
        if let Some(d) = dim {
            opts.set("dim", d);
        }
        let mut ctx = test_ctx(dir);
        VerifyAlignmentOp.execute(&opts, &mut ctx)
    }

    #[test]
    fn aligned_npy_and_passages_pass_with_dim() {
        let tmp = tempfile::tempdir().unwrap();
        let npy = tmp.path().join("base_all.npy");
        let pq = tmp.path().join("passages.parquet");
        write_npy(&npy, 25, 8);
        write_passages(&pq, 25);
        let result = run(tmp.path(), &npy, &pq, Some("8"));
        assert_eq!(result.status, Status::Ok, "{}", result.message);
        assert!(result.message.contains("25"));
    }

    #[test]
    fn row_count_mismatch_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let npy = tmp.path().join("base_all.npy");
        let pq = tmp.path().join("passages.parquet");
        write_npy(&npy, 24, 8);
        write_passages(&pq, 25);
        let result = run(tmp.path(), &npy, &pq, None);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("misalignment"), "{}", result.message);
        assert!(result.message.contains("24") && result.message.contains("25"));
    }

    #[test]
    fn dim_mismatch_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let npy = tmp.path().join("base_all.npy");
        let pq = tmp.path().join("passages.parquet");
        write_npy(&npy, 25, 8);
        write_passages(&pq, 25);
        let result = run(tmp.path(), &npy, &pq, Some("1024"));
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("dimension mismatch"), "{}", result.message);
    }

    #[test]
    fn missing_source_fails_cleanly() {
        let tmp = tempfile::tempdir().unwrap();
        let pq = tmp.path().join("passages.parquet");
        write_passages(&pq, 5);
        let result = run(tmp.path(), &tmp.path().join("absent.npy"), &pq, None);
        assert_eq!(result.status, Status::Error);
    }
}
