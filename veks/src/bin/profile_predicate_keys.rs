// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Profiling harness for `compute predicates` hot path.
//!
//! Isolates the per-record evaluation loop so perf/flamegraph can pinpoint
//! allocation and scanning costs.
//!
//! ## Usage
//!
//! ```sh
//! # Build with optimizations (profiling without optimization is meaningless)
//! cargo build -p veks --release --bin profile_predicate_keys
//!
//! # Run with defaults (page-deserialize + scan, 50K records, first 100 predicates)
//! target/release/profile_predicate_keys \
//!     --metadata local/laion400b/img2text/metadata_content.slab \
//!     --predicates local/laion400b/img2text/predicates.slab
//!
//! # Profile a specific phase
//! target/release/profile_predicate_keys \
//!     --metadata ... --predicates ... \
//!     --records 100000 --max-preds 200 --phase scan-only
//!
//! # Capture flamegraph
//! perf record -g --call-graph dwarf target/release/profile_predicate_keys ...
//! perf script | inferno-collapse-perf | inferno-flamegraph > predkeys.svg
//! ```
//!
//! ## Phases
//!
//! - `page-deser` — only Page::deserialize + record access (no predicate eval)
//! - `scan-only`  — zero-alloc scan_record (skip page deser by using get_ref)
//! - `full`       — page deser + scan_record + fallback (current hot path)
//! - `get-ref`    — SlabReader::get_ref per record (zero-copy baseline)
//! - `mnode-parse` — MNode::from_bytes per record (allocation baseline)
//! - `dump`        — print predicates in SQL form + structural diagnostics

use std::collections::HashMap;
use std::time::Instant;

use slabtastic::{OpenProgress, SlabReader};

use veks::formats::mnode::scan::{
    discover_schema, flatten_and, scan_record, CompiledScanPredicates,
};
use veks::formats::mnode::MNode;
use veks::formats::pnode::{Comparand, OpType, PNode};
use veks::formats::pnode::vernacular::to_sql;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let metadata_path = get_arg(&args, "--metadata")
        .expect("--metadata <path> required");
    let predicates_path = get_arg(&args, "--predicates")
        .expect("--predicates <path> required");
    let max_records: u64 = get_arg(&args, "--records")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    let max_preds: usize = get_arg(&args, "--max-preds")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let phase = get_arg(&args, "--phase")
        .unwrap_or_else(|| "full".to_string());

    eprintln!("=== profile_predicate_keys ===");
    eprintln!("  metadata:   {}", metadata_path);
    eprintln!("  predicates: {}", predicates_path);
    eprintln!("  records:    {}", max_records);
    eprintln!("  max-preds:  {}", max_preds);
    eprintln!("  phase:      {}", phase);
    eprintln!();

    // ── Load predicates ──────────────────────────────────────────────────

    let t0 = Instant::now();
    let pred_reader = SlabReader::open(&predicates_path).expect("open predicates");
    let pred_total = pred_reader.total_records();
    let pred_count = (pred_total as usize).min(max_preds);
    let mut predicates: Vec<PNode> = Vec::with_capacity(pred_count);
    for ord in 0..pred_count {
        if let Ok(data) = pred_reader.get(ord as i64) {
            if let Ok(pnode) = PNode::from_bytes_named(&data) {
                predicates.push(pnode);
            }
        }
    }
    eprintln!(
        "loaded {} predicates in {:.2}s (of {} total)",
        predicates.len(),
        t0.elapsed().as_secs_f64(),
        pred_total,
    );

    // ── Memoize predicates ───────────────────────────────────────────────

    let pred_len = predicates.len();
    let mut field_conditions: HashMap<String, Vec<(usize, OpType, Vec<Comparand>)>> =
        HashMap::new();
    let mut condition_counts = vec![0u32; pred_len];
    let mut fallback: Vec<(usize, PNode)> = Vec::new();

    for (idx, pnode) in predicates.iter().enumerate() {
        match flatten_and(pnode) {
            Some(conditions) => {
                condition_counts[idx] = conditions.len() as u32;
                for (field_name, op, comparands) in conditions {
                    field_conditions
                        .entry(field_name)
                        .or_default()
                        .push((idx, op, comparands));
                }
            }
            None => {
                fallback.push((idx, pnode.clone()));
            }
        }
    }

    eprintln!(
        "memoized: {} field-indexed, {} fallback, {} unique fields",
        pred_len - fallback.len(),
        fallback.len(),
        field_conditions.len(),
    );

    // ── Open metadata slab ───────────────────────────────────────────────

    let t1 = Instant::now();
    let meta_reader = SlabReader::open_with_progress(&metadata_path, |p| match p {
        OpenProgress::PagesPageRead { page_count } => {
            eprintln!("  slab index: {} pages", page_count);
        }
        OpenProgress::IndexBuild { done, total } => {
            if done % 5000 == 0 || done == total {
                eprintln!("  slab index: {}/{} scanned", done, total);
            }
        }
        OpenProgress::IndexComplete { total_records } => {
            eprintln!("  slab index: complete, {} records", total_records);
        }
    })
    .expect("open metadata");
    eprintln!("metadata slab opened in {:.2}s", t1.elapsed().as_secs_f64());

    let total_records = meta_reader.total_records();
    let actual_records = (total_records).min(max_records);

    // ── Discover schema and compile ──────────────────────────────────────

    let first_rec = meta_reader.get_ref(0).expect("read first record");
    let schema = discover_schema(first_rec).expect("discover schema");
    eprintln!(
        "schema: {} fields: [{}]",
        schema.field_count,
        schema
            .field_names
            .iter()
            .map(|n| String::from_utf8_lossy(n).to_string())
            .collect::<Vec<_>>()
            .join(", "),
    );

    let compiled = CompiledScanPredicates::compile(
        pred_len,
        &field_conditions,
        &condition_counts,
        fallback.clone(),
        &schema,
    );
    eprintln!("predicates compiled against schema");
    eprintln!();

    // ── Get sorted page entries ──────────────────────────────────────────

    let mut page_entries = meta_reader.page_entries();
    page_entries.sort_by_key(|e| e.start_ordinal);

    // ── Run the selected phase ───────────────────────────────────────────

    match phase.as_str() {
        "dump" => {
            run_dump(&predicates, &field_conditions, &condition_counts, &fallback);
            return;
        }
        "page-deser" => run_page_deser(&meta_reader, &page_entries, actual_records),
        "scan-only" => run_scan_only(
            &meta_reader,
            actual_records,
            &compiled,
            pred_len,
        ),
        "full" => run_full(
            &meta_reader,
            &page_entries,
            actual_records,
            &compiled,
            pred_len,
        ),
        "get-ref" => run_get_ref(&meta_reader, actual_records),
        "mnode-parse" => run_mnode_parse(&meta_reader, actual_records),
        other => {
            eprintln!("unknown phase: {}", other);
            std::process::exit(1);
        }
    }
}

// ── Phase: page-deser ────────────────────────────────────────────────────

/// Measure raw Page::deserialize + get_record cost (no predicate eval).
fn run_page_deser(
    reader: &SlabReader,
    page_entries: &[slabtastic::PageEntry],
    max_records: u64,
) {
    eprintln!("--- phase: page-deser (Page::deserialize + get_record) ---");
    let start = Instant::now();
    let mut records_done = 0u64;
    let mut bytes_touched = 0u64;

    for entry in page_entries {
        if records_done >= max_records {
            break;
        }
        let page = match reader.read_data_page(entry) {
            Ok(p) => p,
            Err(_) => continue,
        };
        for i in 0..page.record_count() {
            if records_done >= max_records {
                break;
            }
            if let Some(data) = page.get_record(i) {
                bytes_touched += data.len() as u64;
            }
            records_done += 1;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let rate = records_done as f64 / elapsed;
    eprintln!(
        "page-deser: {} records in {:.2}s ({:.0} rec/s, {:.1} MB touched)",
        records_done,
        elapsed,
        rate,
        bytes_touched as f64 / 1_048_576.0,
    );
}

// ── Phase: get-ref ───────────────────────────────────────────────────────

/// Measure SlabReader::get_ref cost (zero-copy baseline).
fn run_get_ref(reader: &SlabReader, max_records: u64) {
    eprintln!("--- phase: get-ref (zero-copy per-record) ---");
    let start = Instant::now();
    let mut bytes_touched = 0u64;

    for ord in 0..max_records {
        match reader.get_ref(ord as i64) {
            Ok(data) => {
                bytes_touched += data.len() as u64;
            }
            Err(_) => {}
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let rate = max_records as f64 / elapsed;
    eprintln!(
        "get-ref: {} records in {:.2}s ({:.0} rec/s, {:.1} MB touched)",
        max_records,
        elapsed,
        rate,
        bytes_touched as f64 / 1_048_576.0,
    );
}

// ── Phase: mnode-parse ───────────────────────────────────────────────────

/// Measure MNode::from_bytes cost (allocation baseline).
fn run_mnode_parse(reader: &SlabReader, max_records: u64) {
    eprintln!("--- phase: mnode-parse (MNode::from_bytes per record) ---");
    let start = Instant::now();
    let mut fields_seen = 0u64;

    for ord in 0..max_records {
        match reader.get_ref(ord as i64) {
            Ok(data) => {
                if let Ok(mnode) = MNode::from_bytes(data) {
                    fields_seen += mnode.fields.len() as u64;
                }
            }
            Err(_) => {}
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let rate = max_records as f64 / elapsed;
    eprintln!(
        "mnode-parse: {} records in {:.2}s ({:.0} rec/s, {} fields seen)",
        max_records,
        elapsed,
        rate,
        fields_seen,
    );
}

// ── Phase: scan-only ─────────────────────────────────────────────────────

/// Measure scan_record cost using get_ref (zero-copy + zero-alloc scan).
fn run_scan_only(
    reader: &SlabReader,
    max_records: u64,
    compiled: &CompiledScanPredicates,
    pred_len: usize,
) {
    eprintln!("--- phase: scan-only (get_ref + scan_record, no page deser) ---");
    let start = Instant::now();
    let mut pass_counts = vec![0u32; pred_len];
    let mut total_matches = 0u64;

    for ord in 0..max_records {
        match reader.get_ref(ord as i64) {
            Ok(data) => {
                for c in pass_counts.iter_mut() {
                    *c = 0;
                }
                if let Ok(true) = scan_record(data, compiled, &mut pass_counts) {
                    for i in 0..pred_len {
                        let required = compiled.required_count(i);
                        if required != u32::MAX && pass_counts[i] == required {
                            total_matches += 1;
                        }
                    }
                }
            }
            Err(_) => {}
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let rate = max_records as f64 / elapsed;
    eprintln!(
        "scan-only: {} records in {:.2}s ({:.0} rec/s, {} total pred matches)",
        max_records,
        elapsed,
        rate,
        total_matches,
    );
}

// ── Phase: full ──────────────────────────────────────────────────────────

/// Measure the current hot path: Page::deserialize + scan_record.
fn run_full(
    reader: &SlabReader,
    page_entries: &[slabtastic::PageEntry],
    max_records: u64,
    compiled: &CompiledScanPredicates,
    pred_len: usize,
) {
    eprintln!("--- phase: full (page deser + scan_record, current hot path) ---");
    let start = Instant::now();
    let mut pass_counts = vec![0u32; pred_len];
    let mut records_done = 0u64;
    let mut total_matches = 0u64;

    for entry in page_entries {
        if records_done >= max_records {
            break;
        }
        let page = match reader.read_data_page(entry) {
            Ok(p) => p,
            Err(_) => continue,
        };

        for rec_idx in 0..page.record_count() {
            if records_done >= max_records {
                break;
            }
            let data = match page.get_record(rec_idx) {
                Some(d) => d,
                None => continue,
            };

            for c in pass_counts.iter_mut() {
                *c = 0;
            }

            if let Ok(true) = scan_record(data, compiled, &mut pass_counts) {
                for i in 0..pred_len {
                    let required = compiled.required_count(i);
                    if required != u32::MAX && pass_counts[i] == required {
                        total_matches += 1;
                    }
                }
            }

            records_done += 1;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let rate = records_done as f64 / elapsed;
    eprintln!(
        "full: {} records in {:.2}s ({:.0} rec/s, {} total pred matches)",
        records_done,
        elapsed,
        rate,
        total_matches,
    );
}

// ── Phase: dump ─────────────────────────────────────────────────────────

/// Print each predicate in SQL form, then print field/condition/comparand diagnostics.
fn run_dump(
    predicates: &[PNode],
    field_conditions: &HashMap<String, Vec<(usize, OpType, Vec<Comparand>)>>,
    condition_counts: &[u32],
    fallback: &[(usize, PNode)],
) {
    eprintln!("--- phase: dump (predicate structure) ---\n");

    // Print each predicate in SQL form
    for (i, pnode) in predicates.iter().enumerate() {
        eprintln!("  pred[{:>4}]: {}", i, to_sql(pnode));
    }

    // Per-field diagnostics
    eprintln!("\n=== FIELD DIAGNOSTICS ===\n");
    let mut fields: Vec<(&String, &Vec<(usize, OpType, Vec<Comparand>)>)> =
        field_conditions.iter().collect();
    fields.sort_by_key(|(name, _)| name.to_string());

    let mut total_conditions = 0usize;
    let mut total_comparands = 0usize;
    let mut global_op_counts: HashMap<&str, (usize, usize)> = HashMap::new();

    for (field_name, conditions) in &fields {
        let cond_count = conditions.len();
        let comp_count: usize = conditions.iter().map(|(_, _, c)| c.len()).sum();
        total_conditions += cond_count;
        total_comparands += comp_count;

        // Op distribution for this field
        let mut op_counts: HashMap<&str, usize> = HashMap::new();
        for (_, op, comparands) in *conditions {
            let name = op_name(op);
            *op_counts.entry(name).or_insert(0) += 1;
            let ge = global_op_counts.entry(name).or_insert((0, 0));
            ge.0 += 1;
            ge.1 += comparands.len();
        }

        let ops: String = op_counts
            .iter()
            .map(|(op, count)| format!("{}:{}", op, count))
            .collect::<Vec<_>>()
            .join(", ");

        eprintln!(
            "  {:>20}: {:>5} conditions, {:>7} comparands  [{}]",
            field_name, cond_count, comp_count, ops
        );
    }

    eprintln!("\n=== SUMMARY ===\n");
    eprintln!("  total predicates:   {}", predicates.len());
    eprintln!("  field-indexed:      {}", predicates.len() - fallback.len());
    eprintln!("  fallback:           {}", fallback.len());
    eprintln!("  unique fields:      {}", field_conditions.len());
    eprintln!("  total conditions:   {}", total_conditions);
    eprintln!("  total comparands:   {}", total_comparands);
    eprintln!(
        "  avg cond/pred:      {:.1}",
        total_conditions as f64 / predicates.len().max(1) as f64
    );
    eprintln!(
        "  avg comp/cond:      {:.1}",
        total_comparands as f64 / total_conditions.max(1) as f64
    );
    eprintln!(
        "  comparisons/record: {} (fields × conditions checked)",
        total_comparands
    );

    // Predicate complexity distribution
    eprintln!("\n=== CONDITION COUNT DISTRIBUTION ===\n");
    let mut count_dist: HashMap<u32, usize> = HashMap::new();
    for &c in condition_counts {
        *count_dist.entry(c).or_insert(0) += 1;
    }
    let mut dist_entries: Vec<_> = count_dist.iter().collect();
    dist_entries.sort_by_key(|e| *e.0);
    for (conds, count) in dist_entries {
        eprintln!("  {} condition(s): {} predicates", conds, count);
    }

    eprintln!("\n=== OPERATOR DISTRIBUTION ===\n");
    let mut ops: Vec<_> = global_op_counts.iter().collect();
    ops.sort_by_key(|(name, _)| name.to_string());
    for (name, (count, comps)) in ops {
        eprintln!(
            "  {:>8}: {:>5} conditions, {:>7} comparands ({:.1} avg)",
            name,
            count,
            comps,
            *comps as f64 / *count as f64
        );
    }
}

fn op_name(op: &OpType) -> &'static str {
    match op {
        OpType::Gt => "Gt",
        OpType::Lt => "Lt",
        OpType::Eq => "Eq",
        OpType::Ne => "Ne",
        OpType::Ge => "Ge",
        OpType::Le => "Le",
        OpType::In => "In",
        OpType::Matches => "Matches",
    }
}

// ── Arg parsing ──────────────────────────────────────────────────────────

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
