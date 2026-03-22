// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: verify predicate evaluation results using SQLite.
//!
//! Loads metadata into an in-memory SQLite database, evaluates a sparse
//! sample of predicates as SQL queries, and compares the results against
//! the stored predicate evaluation indices. This provides a completely
//! independent verification path for predicate correctness.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rusqlite::Connection;
use slabtastic::SlabReader;

use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options, ResourceDesc, Status, StreamContext,
    render_options_table,
};
use vectordata::formats::mnode::MNode;
use vectordata::formats::pnode::{PNode, ConjugateType, OpType, Comparand};

/// Pipeline command: verify predicate results via SQLite.
pub struct VerifyPredicatesOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(VerifyPredicatesOp)
}

impl CommandOp for VerifyPredicatesOp {
    fn command_path(&self) -> &str {
        "verify predicates"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Verify predicate evaluation results using SQLite as source of truth".into(),
            body: format!(
                "# verify predicates\n\n\
                Verify predicate evaluation results using SQLite as source of truth.\n\n\
                ## Description\n\n\
                Loads metadata records from a slab file into an in-memory SQLite database, \
                translates a sparse random sample of PNode predicates to SQL WHERE clauses, \
                evaluates them against the SQLite database, and compares the resulting \
                ordinal sets against the stored predicate evaluation results. Any discrepancy \
                indicates a bug in the predicate evaluator or the metadata import.\n\n\
                ## Options\n\n{}",
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "mem".into(), description: "SQLite in-memory database".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let metadata_str = match options.require("metadata") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let predicates_str = match options.require("predicates") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let indices_str = match options.require("metadata-indices") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };

        let sample_count: usize = options.get("sample")
            .and_then(|s| s.parse().ok())
            .unwrap_or(50);
        let metadata_sample: usize = options.get("metadata-sample")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100_000);
        let seed: u64 = options.get("seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(42);

        let metadata_path = resolve_path(metadata_str, &ctx.workspace);
        let predicates_path = resolve_path(predicates_str, &ctx.workspace);
        let indices_path = resolve_path(indices_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        // Open metadata slab
        let meta_reader = match SlabReader::open(&metadata_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open metadata: {}", e), start),
        };
        let meta_count = meta_reader.total_records() as usize;
        ctx.ui.log(&format!("  metadata: {} records from {}", meta_count, metadata_path.display()));

        // Open predicates slab
        let pred_reader = match SlabReader::open(&predicates_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open predicates: {}", e), start),
        };
        let pred_count = pred_reader.total_records() as usize;
        ctx.ui.log(&format!("  predicates: {} from {}", pred_count, predicates_path.display()));

        // Open indices slab
        let idx_reader = match SlabReader::open(&indices_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open indices: {}", e), start),
        };

        // Phase 1: Load sampled metadata into SQLite
        let effective_meta_sample = metadata_sample.min(meta_count);
        ctx.ui.log(&format!("  loading {} sampled metadata records into SQLite (of {} total)...",
            effective_meta_sample, meta_count));
        let load_start = Instant::now();

        // Sample metadata ordinals uniformly
        use rand::SeedableRng;
        use rand::seq::index::sample;
        let mut meta_rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(1));
        let meta_ordinals = sample(&mut meta_rng, meta_count, effective_meta_sample);
        let mut meta_ordinals_sorted: Vec<usize> = meta_ordinals.into_iter().collect();
        meta_ordinals_sorted.sort();
        let meta_ordinal_set: std::collections::HashSet<usize> = meta_ordinals_sorted.iter().copied().collect();

        let db = match load_metadata_to_sqlite(&meta_reader, &meta_ordinals_sorted, &ctx.ui) {
            Ok(db) => db,
            Err(e) => return error_result(format!("SQLite load failed: {}", e), start),
        };
        let load_elapsed = load_start.elapsed();
        ctx.ui.log(&format!("  SQLite loaded {} rows in {:.1}s",
            effective_meta_sample, load_elapsed.as_secs_f64()));

        // Phase 2: Sample predicates
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        let effective_sample = sample_count.min(pred_count);
        let sampled_indices = sample(&mut rng, pred_count, effective_sample);

        ctx.ui.log(&format!("  verifying {} sampled predicates...", effective_sample));
        let pb = ctx.ui.bar_with_unit(effective_sample as u64, "verifying predicates", "predicates");

        let mut pass_count = 0usize;
        let mut fail_count = 0usize;
        let mut false_positives = 0usize;
        let mut false_negatives = 0usize;
        let mut failures: Vec<serde_json::Value> = Vec::new();

        for (_si, pred_idx) in sampled_indices.iter().enumerate() {
            // Read PNode predicate
            let pred_data = match pred_reader.get(pred_idx as i64) {
                Ok(d) => d,
                Err(e) => {
                    ctx.ui.log(&format!("  WARNING: failed to read predicate {}: {}", pred_idx, e));
                    continue;
                }
            };
            let pnode = match PNode::from_bytes_named(&pred_data) {
                Ok(p) => p,
                Err(e) => {
                    ctx.ui.log(&format!("  WARNING: failed to decode predicate {}: {}", pred_idx, e));
                    continue;
                }
            };

            // Translate PNode to SQL WHERE clause
            let sql_clause = match pnode_to_sql(&pnode) {
                Ok(s) => s,
                Err(e) => {
                    ctx.ui.log(&format!("  WARNING: predicate {} not translatable to SQL: {}", pred_idx, e));
                    continue;
                }
            };

            // Execute SQL query to get expected ordinals
            let sql = format!("SELECT ordinal FROM metadata WHERE {}", sql_clause);
            let sql_ordinals: Vec<i64> = match db.prepare(&sql)
                .and_then(|mut stmt| {
                    let rows = stmt.query_map([], |row| row.get(0))?;
                    rows.collect::<Result<Vec<i64>, _>>()
                }) {
                Ok(v) => v,
                Err(e) => {
                    ctx.ui.log(&format!("  WARNING: SQL error for predicate {}: {} [{}]", pred_idx, e, sql));
                    continue;
                }
            };
            let sql_set: std::collections::HashSet<i64> = sql_ordinals.iter().copied().collect();

            // Read stored ordinals from indices slab
            let stored_data = match idx_reader.get(pred_idx as i64) {
                Ok(d) => d,
                Err(e) => {
                    ctx.ui.log(&format!("  WARNING: failed to read stored indices {}: {}", pred_idx, e));
                    continue;
                }
            };
            let stored_ordinals = read_ordinals_from_slab(&stored_data);
            let stored_set: std::collections::HashSet<i64> = stored_ordinals.iter().copied().collect();

            // Compare only within our metadata sample.
            // Filter stored ordinals to those in our sample, then compare.
            let stored_in_sample: std::collections::HashSet<i64> = stored_set.iter()
                .filter(|&&o| meta_ordinal_set.contains(&(o as usize)))
                .copied()
                .collect();
            let fp: Vec<i64> = stored_in_sample.difference(&sql_set).copied().collect();
            let fn_: Vec<i64> = sql_set.difference(&stored_in_sample).copied().collect();

            if fp.is_empty() && fn_.is_empty() {
                pass_count += 1;
            } else {
                fail_count += 1;
                false_positives += fp.len();
                false_negatives += fn_.len();
                if failures.len() < 20 { // cap detail output
                    failures.push(serde_json::json!({
                        "predicate_index": pred_idx,
                        "sql": sql_clause,
                        "expected_count": sql_set.len(),
                        "stored_count": stored_set.len(),
                        "false_positives": fp.len(),
                        "false_negatives": fn_.len(),
                    }));
                }
            }

            pb.inc(1);
        }
        pb.finish();

        // Write report
        let report = serde_json::json!({
            "sample_count": effective_sample,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "sqlite_load_secs": load_elapsed.as_secs_f64(),
            "total_secs": start.elapsed().as_secs_f64(),
            "failures": failures,
        });

        {
            use crate::pipeline::atomic_write::AtomicWriter;
            use std::io::Write;
            let json = serde_json::to_string_pretty(&report).unwrap();
            let write_result = AtomicWriter::new(&output_path)
                .and_then(|mut w| { w.write_all(json.as_bytes())?; w.finish() });
            if let Err(e) = write_result {
                return error_result(format!("failed to write report: {}", e), start);
            }
        }

        let status = if fail_count == 0 { Status::Ok } else { Status::Error };
        CommandResult {
            status,
            message: format!(
                "predicate verification: {}/{} passed, {} failed ({} FP, {} FN)",
                pass_count, effective_sample, fail_count, false_positives, false_negatives,
            ),
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> crate::pipeline::command::ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["metadata", "predicates", "metadata-indices"],
            &["output"],
        )
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc { name: "metadata".into(), type_name: "Path".into(), required: true, default: None,
                description: "Metadata content slab".into(), role: OptionRole::Input },
            OptionDesc { name: "predicates".into(), type_name: "Path".into(), required: true, default: None,
                description: "Predicates slab (PNode records)".into(), role: OptionRole::Input },
            OptionDesc { name: "metadata-indices".into(), type_name: "Path".into(), required: true, default: None,
                description: "Precomputed predicate results slab".into(), role: OptionRole::Input },
            OptionDesc { name: "output".into(), type_name: "Path".into(), required: true, default: None,
                description: "Verification report output (JSON)".into(), role: OptionRole::Output },
            OptionDesc { name: "sample".into(), type_name: "int".into(), required: false, default: Some("50".into()),
                description: "Number of predicates to spot-check".into(), role: OptionRole::Config },
            OptionDesc { name: "metadata-sample".into(), type_name: "int".into(), required: false, default: Some("100000".into()),
                description: "Number of metadata records to load into SQLite (bounds memory usage)".into(), role: OptionRole::Config },
            OptionDesc { name: "seed".into(), type_name: "int".into(), required: false, default: Some("42".into()),
                description: "Random seed for sample selection".into(), role: OptionRole::Config },
        ]
    }
}

// ---------------------------------------------------------------------------
// SQLite loading
// ---------------------------------------------------------------------------

/// Load sampled metadata records from a slab into an in-memory SQLite database.
///
/// Only the ordinals in `sampled_ordinals` are loaded. Schema is derived
/// from the first record's field names. Each MNode field becomes a column.
/// The `ordinal` column preserves the original record index.
fn load_metadata_to_sqlite(
    reader: &SlabReader,
    sampled_ordinals: &[usize],
    ui: &crate::ui::UiHandle,
) -> Result<Connection, String> {
    let db = Connection::open_in_memory()
        .map_err(|e| format!("SQLite open: {}", e))?;

    if sampled_ordinals.is_empty() {
        return Err("no metadata ordinals to load".into());
    }

    // Read first sampled record to discover schema
    let first_data = reader.get(sampled_ordinals[0] as i64)
        .map_err(|e| format!("read record {}: {}", sampled_ordinals[0], e))?;
    let first_mnode = MNode::from_bytes(&first_data)
        .map_err(|e| format!("decode record {}: {}", sampled_ordinals[0], e))?;

    let field_names: Vec<String> = first_mnode.fields.keys().cloned().collect();

    // Build CREATE TABLE with all fields as TEXT (SQLite is dynamically typed,
    // so this works for all comparisons)
    let columns: Vec<String> = std::iter::once("ordinal INTEGER PRIMARY KEY".to_string())
        .chain(field_names.iter().map(|f| format!("\"{}\" TEXT", f)))
        .collect();
    let create_sql = format!("CREATE TABLE metadata ({})", columns.join(", "));
    db.execute(&create_sql, [])
        .map_err(|e| format!("CREATE TABLE: {}", e))?;

    // Prepare INSERT statement
    let placeholders: Vec<String> = (0..=field_names.len()).map(|_| "?".to_string()).collect();
    let insert_sql = format!(
        "INSERT INTO metadata (ordinal, {}) VALUES ({})",
        field_names.iter().map(|f| format!("\"{}\"", f)).collect::<Vec<_>>().join(", "),
        placeholders.join(", "),
    );

    // Batch insert with transaction
    db.execute("BEGIN TRANSACTION", [])
        .map_err(|e| format!("BEGIN: {}", e))?;

    let total = sampled_ordinals.len();
    let pb = ui.bar_with_unit(total as u64, "loading SQLite", "records");

    {
        let mut stmt = db.prepare(&insert_sql)
            .map_err(|e| format!("prepare INSERT: {}", e))?;

        for (i, &ordinal) in sampled_ordinals.iter().enumerate() {
            let data = match reader.get(ordinal as i64) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let mnode = match MNode::from_bytes(&data) {
                Ok(m) => m,
                Err(_) => continue,
            };

            let mut values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::with_capacity(field_names.len() + 1);
            values.push(Box::new(ordinal as i64));

            for field_name in &field_names {
                let val: Box<dyn rusqlite::types::ToSql> = match mnode.fields.get(field_name) {
                    Some(v) => mvalue_to_sql(v),
                    None => Box::new(rusqlite::types::Null),
                };
                values.push(val);
            }

            let params: Vec<&dyn rusqlite::types::ToSql> = values.iter().map(|v| v.as_ref()).collect();
            let _ = stmt.execute(params.as_slice());

            if (i + 1) % 10_000 == 0 {
                pb.set_position((i + 1) as u64);
            }
        }
    }
    pb.finish();

    db.execute("COMMIT", [])
        .map_err(|e| format!("COMMIT: {}", e))?;

    // Create indices on all columns for query performance
    for field_name in &field_names {
        let idx_sql = format!("CREATE INDEX IF NOT EXISTS idx_{} ON metadata(\"{}\")", field_name, field_name);
        let _ = db.execute(&idx_sql, []);
    }

    Ok(db)
}

/// Convert an MValue to a SQLite-compatible value.
fn mvalue_to_sql(v: &vectordata::formats::mnode::MValue) -> Box<dyn rusqlite::types::ToSql> {
    use vectordata::formats::mnode::MValue;
    match v {
        MValue::Text(s) | MValue::Ascii(s) | MValue::EnumStr(s) => Box::new(s.clone()),
        MValue::Int(i) => Box::new(*i),
        MValue::Int32(i) => Box::new(*i as i64),
        MValue::Short(i) => Box::new(*i as i64),
        MValue::Float(f) => Box::new(*f),
        MValue::Float32(f) => Box::new(*f as f64),
        MValue::Bool(b) => Box::new(if *b { 1i64 } else { 0i64 }),
        MValue::EnumOrd(i) => Box::new(*i as i64),
        MValue::Null => Box::new(rusqlite::types::Null),
        _ => Box::new(rusqlite::types::Null), // Bytes, List, Map, Half — not SQL-representable
    }
}

// ---------------------------------------------------------------------------
// PNode → SQL translation
// ---------------------------------------------------------------------------

/// Translate a PNode predicate tree to a SQL WHERE clause.
fn pnode_to_sql(pnode: &PNode) -> Result<String, String> {
    match pnode {
        PNode::Predicate(pred) => {
            let field = field_ref_to_sql(&pred.field);
            match pred.op {
                OpType::Gt => Ok(format!("{} > {}", field, comparand_to_sql(&pred.comparands[0])?)),
                OpType::Lt => Ok(format!("{} < {}", field, comparand_to_sql(&pred.comparands[0])?)),
                OpType::Eq => Ok(format!("{} = {}", field, comparand_to_sql(&pred.comparands[0])?)),
                OpType::Ne => Ok(format!("{} != {}", field, comparand_to_sql(&pred.comparands[0])?)),
                OpType::Ge => Ok(format!("{} >= {}", field, comparand_to_sql(&pred.comparands[0])?)),
                OpType::Le => Ok(format!("{} <= {}", field, comparand_to_sql(&pred.comparands[0])?)),
                OpType::In => {
                    let vals: Result<Vec<String>, String> = pred.comparands.iter()
                        .map(|c| comparand_to_sql(c))
                        .collect();
                    Ok(format!("{} IN ({})", field, vals?.join(", ")))
                }
                OpType::Matches => {
                    // MATCHES → LIKE with % wildcards
                    let pattern = comparand_to_sql_string(&pred.comparands[0])?;
                    Ok(format!("{} LIKE '%{}%'", field, pattern.replace('\'', "''")))
                }
            }
        }
        PNode::Conjugate(conj) => {
            let op_str = match conj.conjugate_type {
                ConjugateType::And => "AND",
                ConjugateType::Or => "OR",
                _ => return Err("unsupported conjugate type".into()),
            };
            let parts: Result<Vec<String>, String> = conj.children.iter()
                .map(|c| pnode_to_sql(c))
                .collect();
            let joined = parts?.join(&format!(" {} ", op_str));
            Ok(format!("({})", joined))
        }
    }
}

/// Convert a FieldRef to a SQL column name.
fn field_ref_to_sql(field: &vectordata::formats::pnode::FieldRef) -> String {
    match field {
        vectordata::formats::pnode::FieldRef::Named(s) => format!("\"{}\"", s),
        vectordata::formats::pnode::FieldRef::Index(i) => format!("\"field_{}\"", i),
    }
}

/// Convert a Comparand to a SQL literal.
fn comparand_to_sql(c: &Comparand) -> Result<String, String> {
    match c {
        Comparand::Text(s) => Ok(format!("'{}'", s.replace('\'', "''"))) ,
        Comparand::Int(i) => Ok(i.to_string()),
        Comparand::Float(f) => Ok(f.to_string()),
        Comparand::Bool(b) => Ok(if *b { "1".into() } else { "0".into() }),
        Comparand::Null => Ok("NULL".into()),
        Comparand::Bytes(_) => Err("byte comparands not supported in SQL".into()),
    }
}

/// Extract a string value from a Comparand (for LIKE patterns).
fn comparand_to_sql_string(c: &Comparand) -> Result<String, String> {
    match c {
        Comparand::Text(s) => Ok(s.clone()),
        _ => Err("MATCHES comparand must be a string".into()),
    }
}

/// Read ordinals from a slab record (stored as packed i32 LE).
fn read_ordinals_from_slab(data: &[u8]) -> Vec<i64> {
    let mut ordinals = Vec::with_capacity(data.len() / 4);
    let mut offset = 0;
    while offset + 4 <= data.len() {
        let val = i32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
        ordinals.push(val as i64);
        offset += 4;
    }
    ordinals
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
