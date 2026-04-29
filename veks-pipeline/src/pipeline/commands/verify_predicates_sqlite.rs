// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: SQLite oracle verification for simple-int-eq predicates.
//!
//! Loads metadata (M), predicates (P), and predicate results (R) into SQLite
//! and independently verifies every predicate evaluation. This provides a
//! ground-truth oracle that doesn't share code with the evaluation pipeline.

use std::path::Path;
use std::time::Instant;

use rusqlite::Connection;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
};

fn error_result(msg: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message: msg.into(),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn resolve_path(s: &str, workspace: &Path) -> std::path::PathBuf {
    let p = std::path::Path::new(s);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

/// Pipeline command: SQLite oracle verification for predicates.
pub struct VerifyPredicatesSqliteOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(VerifyPredicatesSqliteOp)
}

impl CommandOp for VerifyPredicatesSqliteOp {
    fn command_path(&self) -> &str {
        "verify predicates-sqlite"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_VERIFY
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Verify predicate evaluations using SQLite as an independent oracle".into(),
            body: format!(
                "# verify predicates-sqlite\n\n\
                Loads metadata values, predicate values, and computed predicate results \
                into SQLite tables, then independently evaluates each predicate and \
                compares against the stored results.\n\n\
                ## Options\n\n{}", render_options_table(&options)),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        // Up-front: confirm the local dataset has the minimum facets
        // this verify kind requires (see pipeline::dataset_lookup).
        if let Err(e) = crate::pipeline::dataset_lookup::validate_and_log(
            ctx, options, crate::pipeline::dataset_lookup::VerifyKind::PredicatesSqlite,
        ) {
            return error_result(e, start);
        }

        // Standalone-friendly: input paths come from the active
        // profile's metadata facets in dataset.yaml.
        let metadata_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "metadata", "metadata_content",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        let predicates_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "predicates", "metadata_predicates",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };
        let results_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "results", "metadata_results",
        ) { Ok(s) => s, Err(e) => return error_result(e, start) };

        let metadata_path = resolve_path(&metadata_str, &ctx.workspace);
        let predicates_path = resolve_path(&predicates_str, &ctx.workspace);
        let results_path = resolve_path(&results_str, &ctx.workspace);

        let fields: usize = options.parse_or("fields", 1u32).unwrap_or(1) as usize;
        let sample: usize = options.parse_or("sample", 0u32).unwrap_or(0) as usize;

        // Detect format from extension
        let meta_ext = metadata_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let pred_ext = predicates_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let results_ext = results_path.extension().and_then(|e| e.to_str()).unwrap_or("");

        ctx.ui.log(&format!("  verify predicates-sqlite: metadata={} predicates={} results={}",
            metadata_path.display(), predicates_path.display(), results_path.display()));

        // Create in-memory SQLite database
        let conn = match Connection::open_in_memory() {
            Ok(c) => c,
            Err(e) => return error_result(format!("SQLite: {}", e), start),
        };

        // Load metadata into table
        let meta_count = match load_scalar_metadata(&conn, &metadata_path, meta_ext, fields) {
            Ok(n) => n,
            Err(e) => return error_result(format!("load metadata: {}", e), start),
        };
        ctx.ui.log(&format!("  loaded {} metadata records into SQLite ({} fields)", meta_count, fields));

        // Load predicates
        let predicates = match load_scalar_predicates(&predicates_path, pred_ext, fields) {
            Ok(p) => p,
            Err(e) => return error_result(format!("load predicates: {}", e), start),
        };
        ctx.ui.log(&format!("  loaded {} predicates", predicates.len()));

        // Load stored results (R)
        let stored_results = match load_results(&results_path, results_ext) {
            Ok(r) => r,
            Err(e) => return error_result(format!("load results: {}", e), start),
        };
        ctx.ui.log(&format!("  loaded {} result records", stored_results.len()));

        if predicates.len() != stored_results.len() {
            return error_result(format!(
                "predicate count ({}) != result count ({})",
                predicates.len(), stored_results.len()
            ), start);
        }

        // Write in-memory DB to temp file for multi-threaded read access
        let db_path = ctx.workspace.join(".cache/verify_predicates.sqlite3");
        if let Some(parent) = db_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = conn.execute("VACUUM INTO ?1", [db_path.to_str().unwrap()]) {
            ctx.ui.log(&format!("  warning: export SQLite: {}", e));
        }
        drop(conn);

        // Build indices to check
        let check_count = if sample > 0 && sample < predicates.len() { sample } else { predicates.len() };
        let step = if sample > 0 && sample < predicates.len() { predicates.len() / sample } else { 1 };
        let indices_to_check: Vec<usize> = (0..predicates.len()).step_by(step).take(check_count).collect();
        let actual_check = indices_to_check.len();

        let threads = ctx.governor.current_or("threads", ctx.threads as u64).max(1) as usize;
        let verify_threads = threads.min(actual_check).max(1);
        ctx.ui.log(&format!(
            "verify predicates-sqlite: predicate evaluation results vs SQLite oracle (in-memory, \
             {} fields, {} metadata records); sample={} of {} predicates × {} threads",
            fields, meta_count, actual_check, predicates.len(), verify_threads,
        ));

        let pb = ctx.ui.bar(actual_check as u64, "verifying predicates");
        let progress = std::sync::atomic::AtomicU64::new(0);
        let pass_count = std::sync::atomic::AtomicU64::new(0);
        let fail_count = std::sync::atomic::AtomicU64::new(0);
        let exemplar_count = std::sync::atomic::AtomicU64::new(0);
        let errors: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

        let chunk_size = (actual_check + verify_threads - 1) / verify_threads;

        std::thread::scope(|scope| {
            for chunk_start in (0..actual_check).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(actual_check);
                let chunk_indices = &indices_to_check[chunk_start..chunk_end];
                let pb_ref = &pb;
                let progress_ref = &progress;
                let pass_ref = &pass_count;
                let fail_ref = &fail_count;
                let exemplar_ref = &exemplar_count;
                let errors_ref = &errors;
                let predicates_ref = &predicates;
                let stored_ref = &stored_results;
                let db_path_ref = &db_path;
                let ui_ref = &ctx.ui;

                scope.spawn(move || {
                    let conn = match Connection::open_with_flags(
                        db_path_ref,
                        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                            | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
                    ) {
                        Ok(c) => c,
                        Err(e) => {
                            if let Ok(mut errs) = errors_ref.lock() {
                                errs.push(format!("thread open SQLite: {}", e));
                            }
                            return;
                        }
                    };

                    let fmt_ords = |v: &[i32], n: usize| -> String {
                        let take: Vec<String> = v.iter().take(n).map(|x| x.to_string()).collect();
                        if v.len() > n { format!("[{}, ...]", take.join(", ")) }
                        else { format!("[{}]", take.join(", ")) }
                    };

                    for &idx in chunk_indices {
                        let pred = &predicates_ref[idx];
                        let stored = &stored_ref[idx];

                        let mut where_clauses = Vec::new();
                        for (fi, val) in pred.iter().enumerate() {
                            where_clauses.push(format!("field_{} = {}", fi, val));
                        }
                        let sql = format!("SELECT ordinal FROM metadata WHERE {} ORDER BY ordinal",
                            where_clauses.join(" AND "));

                        let expected: Vec<i32> = conn.prepare(&sql).ok()
                            .and_then(|mut stmt| {
                                stmt.query_map([], |row| row.get::<_, i32>(0)).ok()
                                    .map(|iter| iter.filter_map(|r| r.ok()).collect())
                            })
                            .unwrap_or_default();

                        if expected == *stored {
                            pass_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let ex = exemplar_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if ex < 3 {
                                ui_ref.log(&format!("  exemplar [{}]:", idx));
                                ui_ref.log(&format!("    SQL: {}", sql));
                                ui_ref.log(&format!("    SQL result: {} ordinals", expected.len()));
                                ui_ref.log(&format!("    stored R:   {} ordinals", stored.len()));
                                ui_ref.log(&format!("    SQL ordinals (first 5):    {}", fmt_ords(&expected, 5)));
                                ui_ref.log(&format!("    stored ordinals (first 5): {}", fmt_ords(stored, 5)));
                                ui_ref.log(&format!("    exact match ✓"));
                            }
                        } else {
                            fail_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if let Ok(mut errs) = errors_ref.lock() {
                                if errs.len() < 10 {
                                    errs.push(format!(
                                        "pred {} ({}): SQL={} stored={} Δ={}",
                                        idx, where_clauses.join(" AND "),
                                        expected.len(), stored.len(),
                                        (expected.len() as i64 - stored.len() as i64).abs(),
                                    ));
                                }
                            }
                        }

                        let done = progress_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if done % 100 == 0 { pb_ref.set_position(done); }
                    }
                });
            }
        });
        pb.set_position(actual_check as u64);
        pb.finish();

        let _ = std::fs::remove_file(&db_path);

        let pass = pass_count.load(std::sync::atomic::Ordering::Relaxed) as usize;
        let fail = fail_count.load(std::sync::atomic::Ordering::Relaxed) as usize;
        let checked = actual_check;

        if let Ok(errs) = errors.lock() {
            for err in errs.iter() {
                ctx.ui.log(&format!("  MISMATCH: {}", err));
            }
        }
        ctx.ui.log(&format!("  {} pass, {} fail ({} checked of {}, {} threads)",
            pass, fail, checked, predicates.len(), verify_threads));

        let output_path = options.get("output").map(|s| resolve_path(s, &ctx.workspace));
        if let Some(ref out) = output_path {
            if let Some(parent) = out.parent() { let _ = std::fs::create_dir_all(parent); }
            let report = serde_json::json!({
                "type": "predicates-sqlite",
                "pass": pass,
                "fail": fail,
                "checked": checked,
                "total": predicates.len(),
                "metadata_count": meta_count,
                "fields": fields,
            });
            let _ = std::fs::write(out, serde_json::to_string_pretty(&report).unwrap_or_default());
        }

        CommandResult {
            status: if fail == 0 { Status::Ok } else { Status::Error },
            message: format!("{} pass, {} fail ({} checked)", pass, fail, checked),
            produced: output_path.into_iter().collect(),
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc { name: "metadata".into(), type_name: "Path".into(), required: true,
                default: None, description: "Metadata file (u8, ivec, or slab)".into(),
                role: OptionRole::Input },
            OptionDesc { name: "predicates".into(), type_name: "Path".into(), required: true,
                default: None, description: "Predicate file (u8, ivec, or slab)".into(),
                role: OptionRole::Input },
            OptionDesc { name: "results".into(), type_name: "Path".into(), required: true,
                default: None, description: "Predicate results file (ivec of matching ordinals)".into(),
                role: OptionRole::Input },
            OptionDesc { name: "fields".into(), type_name: "int".into(), required: false,
                default: Some("1".into()), description: "Number of integer fields".into(),
                role: OptionRole::Config },
            OptionDesc { name: "sample".into(), type_name: "int".into(), required: false,
                default: Some("0".into()), description: "Number of predicates to sample (0 = all)".into(),
                role: OptionRole::Config },
            OptionDesc { name: "output".into(), type_name: "Path".into(), required: false,
                default: None, description: "JSON report output path".into(),
                role: OptionRole::Output },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> crate::pipeline::command::ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["metadata", "predicates", "results"],
            &["output"],
        )
    }
}

/// Load scalar metadata (u8 or ivec) into SQLite.
fn load_scalar_metadata(conn: &Connection, path: &Path, ext: &str, fields: usize) -> Result<usize, String> {
    // Create table
    let mut cols = vec!["ordinal INTEGER PRIMARY KEY".to_string()];
    for i in 0..fields {
        cols.push(format!("field_{} INTEGER", i));
    }
    let create_sql = format!("CREATE TABLE metadata ({})", cols.join(", "));
    conn.execute(&create_sql, []).map_err(|e| format!("create table: {}", e))?;

    let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;

    let elem_size = match ext {
        "u8" | "i8" => 1,
        "u16" | "i16" => 2,
        "u32" | "i32" => 4,
        "u64" | "i64" => 8,
        "ivec" | "ivecs" => 0, // special: xvec with dim header
        _ => return Err(format!("unsupported metadata format: {}", ext)),
    };

    let insert_sql = {
        let placeholders: Vec<&str> = (0..=fields).map(|_| "?").collect();
        format!("INSERT INTO metadata VALUES ({})", placeholders.join(", "))
    };

    conn.execute("BEGIN", []).map_err(|e| e.to_string())?;

    let mut count = 0;
    if elem_size > 0 {
        // Scalar format: flat packed
        let record_size = elem_size * fields;
        let total = data.len() / record_size;
        for i in 0..total {
            let offset = i * record_size;
            let mut params: Vec<i64> = vec![i as i64]; // ordinal
            for f in 0..fields {
                let fo = offset + f * elem_size;
                let val = match elem_size {
                    1 => data[fo] as i64,
                    2 => i16::from_le_bytes([data[fo], data[fo+1]]) as i64,
                    4 => i32::from_le_bytes(data[fo..fo+4].try_into().unwrap()) as i64,
                    8 => i64::from_le_bytes(data[fo..fo+8].try_into().unwrap()),
                    _ => 0,
                };
                params.push(val);
            }
            let refs: Vec<&dyn rusqlite::types::ToSql> = params.iter()
                .map(|v| v as &dyn rusqlite::types::ToSql).collect();
            conn.execute(&insert_sql, refs.as_slice()).map_err(|e| format!("insert: {}", e))?;
            count += 1;
        }
    } else {
        // ivec format: [dim:i32, val0:i32, val1:i32, ...]
        let mut offset = 0;
        let mut ordinal = 0i64;
        while offset + 4 <= data.len() {
            let dim = i32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            if offset + dim * 4 > data.len() { break; }
            let mut params: Vec<i64> = vec![ordinal];
            for f in 0..fields.min(dim) {
                let fo = offset + f * 4;
                params.push(i32::from_le_bytes(data[fo..fo+4].try_into().unwrap()) as i64);
            }
            // Pad missing fields with 0
            while params.len() <= fields {
                params.push(0);
            }
            let refs: Vec<&dyn rusqlite::types::ToSql> = params.iter()
                .map(|v| v as &dyn rusqlite::types::ToSql).collect();
            conn.execute(&insert_sql, refs.as_slice()).map_err(|e| format!("insert: {}", e))?;
            offset += dim * 4;
            ordinal += 1;
            count += 1;
        }
    }

    conn.execute("COMMIT", []).map_err(|e| e.to_string())?;

    // Create index for each field
    for i in 0..fields {
        let idx_sql = format!("CREATE INDEX idx_field_{} ON metadata (field_{})", i, i);
        conn.execute(&idx_sql, []).map_err(|e| format!("create index: {}", e))?;
    }

    Ok(count)
}

/// Load scalar predicates (u8, ivec, etc.) into Vec<Vec<i64>>.
fn load_scalar_predicates(path: &Path, ext: &str, fields: usize) -> Result<Vec<Vec<i64>>, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;

    let elem_size = match ext {
        "u8" | "i8" => 1,
        "u16" | "i16" => 2,
        "u32" | "i32" => 4,
        "u64" | "i64" => 8,
        "ivec" | "ivecs" | "ivvec" | "ivvecs" | "i32vvec" | "i32vvecs" => 0,
        _ => return Err(format!("unsupported predicate format: {}", ext)),
    };

    let mut predicates = Vec::new();

    if elem_size > 0 {
        let record_size = elem_size * fields;
        let total = data.len() / record_size;
        for i in 0..total {
            let offset = i * record_size;
            let mut vals = Vec::with_capacity(fields);
            for f in 0..fields {
                let fo = offset + f * elem_size;
                let val = match elem_size {
                    1 => data[fo] as i64,
                    2 => i16::from_le_bytes([data[fo], data[fo+1]]) as i64,
                    4 => i32::from_le_bytes(data[fo..fo+4].try_into().unwrap()) as i64,
                    8 => i64::from_le_bytes(data[fo..fo+8].try_into().unwrap()),
                    _ => 0,
                };
                vals.push(val);
            }
            predicates.push(vals);
        }
    } else {
        // ivec
        let mut offset = 0;
        while offset + 4 <= data.len() {
            let dim = i32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
            offset += 4;
            if offset + dim * 4 > data.len() { break; }
            let mut vals = Vec::with_capacity(fields.min(dim));
            for f in 0..fields.min(dim) {
                let fo = offset + f * 4;
                vals.push(i32::from_le_bytes(data[fo..fo+4].try_into().unwrap()) as i64);
            }
            predicates.push(vals);
            offset += dim * 4;
        }
    }

    Ok(predicates)
}

/// Load predicate results (ivec of matching ordinals per predicate).
fn load_results(path: &Path, ext: &str) -> Result<Vec<Vec<i32>>, String> {
    match ext {
        "ivec" | "ivecs" | "ivvec" | "ivvecs" | "i32vvec" | "i32vvecs" => {
            // Use IndexedXvecReader for both uniform and variable-length ivec
            let reader = vectordata::io::IndexedXvecReader::open_ivec(path)
                .map_err(|e| format!("{}: {}", path.display(), e))?;
            let mut results = Vec::with_capacity(reader.count());
            for i in 0..reader.count() {
                results.push(reader.get_i32(i).map_err(|e| format!("record {}: {}", i, e))?);
            }
            Ok(results)
        }
        "slab" => {
            let reader = slabtastic::SlabReader::open(path)
                .map_err(|e| format!("open slab {}: {}", path.display(), e))?;
            let count = reader.total_records();
            let mut results = Vec::with_capacity(count as usize);
            for i in 0..count {
                let data = reader.get(i as i64)
                    .map_err(|e| format!("read slab record {}: {}", i, e))?;
                let vals: Vec<i32> = data.chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                results.push(vals);
            }
            Ok(results)
        }
        _ => Err(format!("unsupported results format: {}", ext)),
    }
}
