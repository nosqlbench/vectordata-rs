// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: detect overlapping vectors between base and query sets.
//!
//! Hashes every base vector, then checks each query vector against the set.
//! Reports the number and fraction of query vectors that also appear in the
//! base set. Overlap makes KNN ground truth unreliable because a query's
//! nearest neighbor would be itself (distance ≈ 0).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::io::XvecReader;
use vectordata::io::VectorReader;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
};
use crate::pipeline::element_type::ElementType;

pub struct AnalyzeOverlapOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzeOverlapOp)
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = Path::new(path_str);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

impl CommandOp for AnalyzeOverlapOp {
    fn command_path(&self) -> &str {
        "analyze overlap"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_ANALYZE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Detect overlapping vectors between base and query sets".into(),
            body: format!(
                "# analyze overlap\n\n\
                 Detect vectors that appear in both the base and query sets.\n\n\
                 ## Description\n\n\
                 Hashes every base vector, then checks each query vector against \
                 the hash set. Reports the count and fraction of query vectors that \
                 also appear in the base set.\n\n\
                 ## Why Overlap Matters\n\n\
                 When a query vector is identical to a base vector, its nearest \
                 neighbor is itself at distance 0 (or near-0 for cosine). This makes \
                 KNN ground truth trivially correct for that query and distorts \
                 benchmark accuracy metrics. Datasets intended for evaluation \
                 should have disjoint base and query sets.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let base_str = match options.require("base") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let query_str = match options.require("query") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let base_path = resolve_path(base_str, &ctx.workspace);
        let query_path = resolve_path(query_str, &ctx.workspace);

        let base_etype = match ElementType::from_path(&base_path) {
            Ok(e) => e,
            Err(e) => return error_result(format!("base: {}", e), start),
        };
        let query_etype = match ElementType::from_path(&query_path) {
            Ok(e) => e,
            Err(e) => return error_result(format!("query: {}", e), start),
        };

        if base_etype != query_etype {
            return error_result(
                format!("element type mismatch: base={:?}, query={:?}", base_etype, query_etype),
                start,
            );
        }

        let result = match base_etype {
            ElementType::F32 => check_overlap_f32(&base_path, &query_path, &ctx.ui),
            ElementType::F16 => check_overlap_f16(&base_path, &query_path, &ctx.ui),
            _ => Err(format!("unsupported element type {:?} for overlap check", base_etype)),
        };

        let (base_count, query_count, overlap_count) = match result {
            Ok(r) => r,
            Err(msg) => return error_result(msg, start),
        };

        let overlap_frac = if query_count > 0 {
            overlap_count as f64 / query_count as f64
        } else {
            0.0
        };

        // Save overlap metrics as variables
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, "overlap_count", &overlap_count.to_string(),
        );
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, "overlap_fraction", &format!("{:.6}", overlap_frac),
        );
        ctx.defaults.insert("overlap_count".into(), overlap_count.to_string());
        ctx.defaults.insert("overlap_fraction".into(), format!("{:.6}", overlap_frac));

        let status = if overlap_count > 0 { Status::Error } else { Status::Ok };
        let message = format!(
            "base={} query={} overlap={} ({:.4}%)",
            base_count, query_count, overlap_count, overlap_frac * 100.0,
        );

        if overlap_count > 0 {
            ctx.ui.log(&format!(
                "  ERROR: {} of {} query vectors also appear in the base set",
                overlap_count, query_count,
            ));
            ctx.ui.log("  Overlapping vectors make KNN ground truth unreliable.");
        } else {
            ctx.ui.log(&format!(
                "  Base and query sets are disjoint ({} base, {} query vectors)",
                base_count, query_count,
            ));
        }

        CommandResult {
            status,
            message,
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "base".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Base vectors file".to_string(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "query".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Query vectors file".to_string(),
                role: OptionRole::Input,
            },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["base", "query"],
            &[],
        )
    }
}

/// Hash the raw bytes of a vector for fast set membership.
fn hash_vector(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for chunk in bytes.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let word = u64::from_le_bytes(buf);
        h ^= word;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    h
}

/// Hash a vector's raw float bytes for set membership.
fn hash_f32_vec(vec: &[f32]) -> u64 {
    let bytes = unsafe {
        std::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * 4)
    };
    hash_vector(bytes)
}

fn hash_f16_vec(vec: &[half::f16]) -> u64 {
    let bytes = unsafe {
        std::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * 2)
    };
    hash_vector(bytes)
}

fn check_overlap_f32(
    base_path: &Path,
    query_path: &Path,
    ui: &veks_core::ui::UiHandle,
) -> Result<(usize, usize, usize), String> {
    let base_reader = XvecReader::<f32>::open_path(base_path)
        .map_err(|e| format!("open base {}: {}", base_path.display(), e))?;
    let query_reader = XvecReader::<f32>::open_path(query_path)
        .map_err(|e| format!("open query {}: {}", query_path.display(), e))?;

    let base_count = <XvecReader<f32> as VectorReader<f32>>::count(&base_reader);
    let query_count = <XvecReader<f32> as VectorReader<f32>>::count(&query_reader);
    let dim = <XvecReader<f32> as VectorReader<f32>>::dim(&base_reader);

    ui.log(&format!("  hashing {} base vectors (dim={})", base_count, dim));
    let pb = ui.bar(base_count as u64, "hashing base vectors");

    let mut base_hashes: HashSet<u64> = HashSet::with_capacity(base_count);
    for i in 0..base_count {
        let vec = <XvecReader<f32> as VectorReader<f32>>::get(&base_reader, i)
            .map_err(|e| format!("read base[{}]: {}", i, e))?;
        base_hashes.insert(hash_f32_vec(&vec));
        pb.inc(1);
    }
    pb.finish();

    ui.log(&format!("  checking {} query vectors for overlap", query_count));
    let pb = ui.bar(query_count as u64, "checking query vectors");

    let mut overlap = 0usize;
    for i in 0..query_count {
        let vec = <XvecReader<f32> as VectorReader<f32>>::get(&query_reader, i)
            .map_err(|e| format!("read query[{}]: {}", i, e))?;
        if base_hashes.contains(&hash_f32_vec(&vec)) {
            overlap += 1;
        }
        pb.inc(1);
    }
    pb.finish();

    Ok((base_count, query_count, overlap))
}

fn check_overlap_f16(
    base_path: &Path,
    query_path: &Path,
    ui: &veks_core::ui::UiHandle,
) -> Result<(usize, usize, usize), String> {
    let base_reader = XvecReader::<half::f16>::open_path(base_path)
        .map_err(|e| format!("open base {}: {}", base_path.display(), e))?;
    let query_reader = XvecReader::<half::f16>::open_path(query_path)
        .map_err(|e| format!("open query {}: {}", query_path.display(), e))?;

    let base_count = <XvecReader<half::f16> as VectorReader<half::f16>>::count(&base_reader);
    let query_count = <XvecReader<half::f16> as VectorReader<half::f16>>::count(&query_reader);
    let dim = <XvecReader<half::f16> as VectorReader<half::f16>>::dim(&base_reader);

    ui.log(&format!("  hashing {} base vectors (dim={})", base_count, dim));
    let pb = ui.bar(base_count as u64, "hashing base vectors");

    let mut base_hashes: HashSet<u64> = HashSet::with_capacity(base_count);
    for i in 0..base_count {
        let vec = <XvecReader<half::f16> as VectorReader<half::f16>>::get(&base_reader, i)
            .map_err(|e| format!("read base[{}]: {}", i, e))?;
        base_hashes.insert(hash_f16_vec(&vec));
        pb.inc(1);
    }
    pb.finish();

    ui.log(&format!("  checking {} query vectors for overlap", query_count));
    let pb = ui.bar(query_count as u64, "checking query vectors");

    let mut overlap = 0usize;
    for i in 0..query_count {
        let vec = <XvecReader<half::f16> as VectorReader<half::f16>>::get(&query_reader, i)
            .map_err(|e| format!("read query[{}]: {}", i, e))?;
        if base_hashes.contains(&hash_f16_vec(&vec)) {
            overlap += 1;
        }
        pb.inc(1);
    }
    pb.finish();

    Ok((base_count, query_count, overlap))
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}
