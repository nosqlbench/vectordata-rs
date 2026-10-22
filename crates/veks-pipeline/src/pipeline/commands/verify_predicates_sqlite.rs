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

/// Creates a boxed `VerifyPredicatesSqliteOp` for command registration.
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

        let chunk_size = actual_check.div_ceil(verify_threads);

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
                                ui_ref.log("    exact match ✓");
                            }
                        } else {
                            fail_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if let Ok(mut errs) = errors_ref.lock()
                                && errs.len() < 10 {
                                    errs.push(format!(
                                        "pred {} ({}): SQL={} stored={} Δ={}",
                                        idx, where_clauses.join(" AND "),
                                        expected.len(), stored.len(),
                                        (expected.len() as i64 - stored.len() as i64).abs(),
                                    ));
                                }
                        }

                        let done = progress_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if done.is_multiple_of(100) { pb_ref.set_position(done); }
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
            extended_description: None,
                role: OptionRole::Input },
            OptionDesc { name: "predicates".into(), type_name: "Path".into(), required: true,
                default: None, description: "Predicate file (u8, ivec, or slab)".into(),
            extended_description: None,
                role: OptionRole::Input },
            OptionDesc { name: "results".into(), type_name: "Path".into(), required: true,
                default: None, description: "Predicate results file (ivec of matching ordinals)".into(),
            extended_description: None,
                role: OptionRole::Input },
            OptionDesc { name: "fields".into(), type_name: "int".into(), required: false,
                default: Some("1".into()), description: "Number of integer fields".into(),
            extended_description: None,
                role: OptionRole::Config },
            OptionDesc { name: "sample".into(), type_name: "int".into(), required: false,
                default: Some("0".into()), description: "Number of predicates to sample (0 = all)".into(),
            extended_description: None,
                role: OptionRole::Config },
            OptionDesc { name: "output".into(), type_name: "Path".into(), required: false,
                default: None, description: "JSON report output path".into(),
            extended_description: None,
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

/// An integer element as an i64, by `as`: exact for every type but
/// `u64`, whose values past `i64::MAX` keep their bits.
trait WidenI64: Copy {
    const ZERO: Self;
    fn widen_i64(self) -> i64;
}

macro_rules! widen_i64 {
    ($($t:ty),*) => {$(
        impl WidenI64 for $t {
            const ZERO: Self = 0;
            #[inline]
            fn widen_i64(self) -> i64 { self as i64 }
        }
    )*};
}
widen_i64!(u8, i8, u16, i16, u32, i32, u64, i64);

/// Integer records of a metadata or predicate file, in order, each
/// widened to i64 and handed to `each` with its ordinal.
///
/// The layout is the one the extension names, through veks-io's
/// readers: a header-free integer column (`u8` … `i64`) holds each
/// record as `fields` values of its type; an i32 vector file (`ivec`,
/// `ivvec`, …) holds each as one record, of which the first `fields`
/// values are given (fewer when the record is shorter). `what` names the
/// file in the unsupported-format error.
fn for_each_int_record(
    path: &Path,
    ext: &str,
    fields: usize,
    what: &str,
    mut each: impl FnMut(i64, &[i64]) -> Result<(), String>,
) -> Result<(), String> {
    use veks_io::{Container, ElementType as E};
    let open_err = |e: veks_io::IoError| format!("read {}: {}", path.display(), e);
    match E::parse_extension(ext) {
        Some((ty, Container::Scalar)) if ty.is_integer() => {
            fn column<T: veks_io::Element + WidenI64, F: FnMut(i64, &[i64]) -> Result<(), String>>(
                path: &Path,
                fields: usize,
                each: &mut F,
            ) -> Result<(), String> {
                use veks_io::VectorReader;
                let reader = veks_io::ScalarReader::<T>::open_path(path)
                    .map_err(|e| format!("read {}: {}", path.display(), e))?;
                let mut values = vec![T::ZERO; reader.count()];
                reader.read_into(0, &mut values).map_err(|e| format!("read {}: {}", path.display(), e))?;
                // A trailing partial record is not a record.
                let mut row = Vec::with_capacity(fields);
                for (i, rec) in values.chunks_exact(fields).enumerate() {
                    row.clear();
                    row.extend(rec.iter().map(|&v| v.widen_i64()));
                    each(i as i64, &row)?;
                }
                Ok(())
            }
            match ty {
                E::U8 => column::<u8, _>(path, fields, &mut each),
                E::I8 => column::<i8, _>(path, fields, &mut each),
                E::U16 => column::<u16, _>(path, fields, &mut each),
                E::I16 => column::<i16, _>(path, fields, &mut each),
                E::U32 => column::<u32, _>(path, fields, &mut each),
                E::I32 => column::<i32, _>(path, fields, &mut each),
                E::U64 => column::<u64, _>(path, fields, &mut each),
                _ => column::<i64, _>(path, fields, &mut each),
            }
        }
        Some((E::I32, _)) if veks_io::reader::is_vvec_ext(ext) => {
            // `ivec`/`ivvec`: records may differ in length. Walked front
            // to back over the mapped bytes, so verifying an input writes
            // no offset index beside it.
            let mapped = veks_io::MappedFile::map(path)
                .map_err(|e| format!("read {}: {}", path.display(), e))?;
            let mut row = Vec::with_capacity(fields);
            for (i, record) in veks_io::xvec::record::records(mapped.bytes(), 4).enumerate() {
                let (_, payload) = record.map_err(|e| format!("record {}: {}", i, e))?;
                row.clear();
                row.extend(payload.as_chunks::<4>().0.iter().take(fields)
                    .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64));
                each(i as i64, &row)?;
            }
            Ok(())
        }
        Some((E::I32, Container::Vec)) => {
            let reader = veks_io::XvecReader::<i32>::open_path(path).map_err(open_err)?;
            let mut row = Vec::with_capacity(fields);
            for i in 0..reader.count() {
                row.clear();
                row.extend(reader.get_slice(i).iter().take(fields).map(|&v| v as i64));
                each(i as i64, &row)?;
            }
            Ok(())
        }
        _ => Err(format!("unsupported {} format: {}", what, ext)),
    }
}

/// Load scalar metadata (an integer column or ivec) into SQLite.
fn load_scalar_metadata(conn: &Connection, path: &Path, ext: &str, fields: usize) -> Result<usize, String> {
    // Create table
    let mut cols = vec!["ordinal INTEGER PRIMARY KEY".to_string()];
    for i in 0..fields {
        cols.push(format!("field_{} INTEGER", i));
    }
    let create_sql = format!("CREATE TABLE metadata ({})", cols.join(", "));
    conn.execute(&create_sql, []).map_err(|e| format!("create table: {}", e))?;

    let insert_sql = {
        let placeholders: Vec<&str> = (0..=fields).map(|_| "?").collect();
        format!("INSERT INTO metadata VALUES ({})", placeholders.join(", "))
    };

    conn.execute("BEGIN", []).map_err(|e| e.to_string())?;

    let mut count = 0;
    let mut params: Vec<i64> = Vec::with_capacity(fields + 1);
    for_each_int_record(path, ext, fields, "metadata", |ordinal, vals| {
        params.clear();
        params.push(ordinal);
        params.extend_from_slice(vals);
        // Pad missing fields with 0
        params.resize(fields + 1, 0);
        let refs: Vec<&dyn rusqlite::types::ToSql> = params.iter()
            .map(|v| v as &dyn rusqlite::types::ToSql).collect();
        conn.execute(&insert_sql, refs.as_slice()).map_err(|e| format!("insert: {}", e))?;
        count += 1;
        Ok(())
    })?;

    conn.execute("COMMIT", []).map_err(|e| e.to_string())?;

    // Create index for each field
    for i in 0..fields {
        let idx_sql = format!("CREATE INDEX idx_field_{} ON metadata (field_{})", i, i);
        conn.execute(&idx_sql, []).map_err(|e| format!("create index: {}", e))?;
    }

    Ok(count)
}

/// Load scalar predicates (an integer column, ivec, ivvec) into Vec<Vec<i64>>.
fn load_scalar_predicates(path: &Path, ext: &str, fields: usize) -> Result<Vec<Vec<i64>>, String> {
    let mut predicates = Vec::new();
    for_each_int_record(path, ext, fields, "predicate", |_, vals| {
        predicates.push(vals.to_vec());
        Ok(())
    })?;
    Ok(predicates)
}

/// Load predicate results (ivec of matching ordinals per predicate).
fn load_results(path: &Path, ext: &str) -> Result<Vec<Vec<i32>>, String> {
    let is_i32_vvec = veks_io::reader::is_vvec_ext(ext)
        && veks_io::ElementType::from_extension(ext) == Some(veks_io::ElementType::I32);
    match ext {
        _ if is_i32_vvec => {
            // Use IndexedVvecReader::<i32> for both uniform and variable-length ivec
            let reader = veks_io::IndexedVvecReader::<i32>::open_path(path)
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
