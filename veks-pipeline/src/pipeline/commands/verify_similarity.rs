// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compare two vector artifacts row-by-row.
//!
//! The embed stage's outputs are floating-point and implementation-shaped:
//! a kernel restructure, a dtype change, or a batching change reorders
//! reductions and moves every value a little. This command is the native
//! gate for "is the new artifact the same embedding, numerically": it
//! streams two same-shape f32 artifacts, reports per-row cosine statistics
//! and unit-norm deviation, and (optionally) fails the step when the worst
//! row drops below `min-cosine`. It replaces ad hoc out-of-repo scripting
//! for revision comparisons — pipeline verification stays in the pipeline.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
};
use veks_core::formats::VecFormat;
use veks_core::formats::reader::open_source;

/// Pipeline command: row-wise cosine comparison of two vector artifacts.
pub struct VerifySimilarityOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(VerifySimilarityOp)
}

impl CommandOp for VerifySimilarityOp {
    fn command_path(&self) -> &str {
        "verify similarity"
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
            summary: "Compare two vector artifacts row-by-row (cosine)".into(),
            body: format!(
                r#"# verify similarity

Compare two vector artifacts row-by-row (cosine).

## Description

Streams two f32 vector artifacts of identical shape (npy file or
directory, xvec, hdf5, slab, or vector parquet) and reports per-row
cosine statistics (min / mean, worst row) plus the maximum deviation of
the `source` rows from unit L2 norm. With `min-cosine` set, the step
fails when any row's cosine falls below the threshold.

## Role in dataset pipelines

Embedding outputs are floating-point: kernel fusion, dtype, batching, and
device changes all reorder reductions, so byte-identity is the wrong
equivalence for "same embedding". This command makes the right
equivalence — bounded per-row cosine drift — a first-class, recordable
pipeline step, e.g. gating a re-embed with a new binary revision against
the previous artifact before it replaces a dataset facet.

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
        let min_cosine: Option<f64> = match options.parse_opt("min-cosine") {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };

        let mut src = match open_f32(&source) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let mut refr = match open_f32(&reference) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        if src.dimension() != refr.dimension() {
            return error_result(
                format!(
                    "dimension mismatch: {} has dim {}, {} has dim {}",
                    source.display(),
                    src.dimension(),
                    reference.display(),
                    refr.dimension()
                ),
                start,
            );
        }

        let total = src.record_count().unwrap_or(0);
        let pb = ctx.ui.bar_with_unit(total, "compare", "row");
        let mut rows: u64 = 0;
        let mut cos_min = f64::INFINITY;
        let mut cos_min_row: u64 = 0;
        let mut cos_sum = 0.0f64;
        let mut norm_dev_max = 0.0f64;
        loop {
            match (src.next_record(), refr.next_record()) {
                (None, None) => break,
                (Some(a), Some(b)) => {
                    let a = as_f32(&a);
                    let b = as_f32(&b);
                    let mut dot = 0.0f64;
                    let mut na = 0.0f64;
                    let mut nb = 0.0f64;
                    for (x, y) in a.iter().zip(&b) {
                        let (x, y) = (*x as f64, *y as f64);
                        dot += x * y;
                        na += x * x;
                        nb += y * y;
                    }
                    let denom = (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE);
                    let cos = dot / denom;
                    if cos < cos_min {
                        cos_min = cos;
                        cos_min_row = rows;
                    }
                    cos_sum += cos;
                    norm_dev_max = norm_dev_max.max((na.sqrt() - 1.0).abs());
                    rows += 1;
                    if rows.is_multiple_of(4096) {
                        pb.set_position(rows);
                    }
                }
                _ => {
                    return error_result(
                        format!(
                            "row-count mismatch after {} row(s): {} and {} differ in length",
                            rows,
                            source.display(),
                            reference.display()
                        ),
                        start,
                    );
                }
            }
        }
        pb.finish();
        if rows == 0 {
            return error_result("no rows to compare".into(), start);
        }

        let summary = format!(
            "{} row(s): cosine min {:.6} (row {}), mean {:.7}; max |1-‖source‖| {:.2e}",
            rows,
            cos_min,
            cos_min_row,
            cos_sum / rows as f64,
            norm_dev_max
        );
        ctx.ui.log(&summary);

        if let Some(threshold) = min_cosine {
            if cos_min < threshold {
                return error_result(
                    format!("similarity below threshold {}: {}", threshold, summary),
                    start,
                );
            }
        }
        CommandResult {
            status: Status::Ok,
            message: summary,
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
                description: "Vector artifact under test (npy, xvec, hdf5, slab, vector parquet)"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "reference".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Golden vector artifact of identical shape".to_string(),
                extended_description: None,
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "min-cosine".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: None,
                description: "Fail when any row's cosine falls below this threshold".to_string(),
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

/// Open a vector artifact as a streaming f32 source.
fn open_f32(path: &Path) -> Result<Box<dyn veks_core::formats::reader::VecSource>, String> {
    let format = VecFormat::detect(path)
        .ok_or_else(|| format!("unrecognized artifact format: {}", path.display()))?;
    let src = open_source(path, format, 1, None)?;
    if src.element_size() != 4 {
        return Err(format!(
            "{}: expected f32 elements, found element size {}",
            path.display(),
            src.element_size()
        ));
    }
    Ok(src)
}

/// Reinterpret little-endian element bytes as f32 values.
fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
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

    #[test]
    fn as_f32_round_trips() {
        let vals = [1.0f32, -0.5, 3.25];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(as_f32(&bytes), vals);
    }
}
