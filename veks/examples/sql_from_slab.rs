// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Generate SQL DDL, INSERT, and SELECT statements from slab-stored metadata
//! and predicate records using the ANode vernacular system.
//!
//! This example demonstrates the full round-trip:
//!
//! 1. Build MNode metadata records (image catalog entries)
//! 2. Build PNode predicate trees (search filters)
//! 3. Write both to slab files
//! 4. Read them back and decode
//! 5. Render SQL from the decoded records:
//!    - CREATE TABLE from the metadata schema
//!    - INSERT INTO from each metadata record
//!    - SELECT ... WHERE from each predicate
//! 6. Evaluate predicates against metadata to show which rows match
//!
//! Run with:
//!
//!     cargo run --example sql_from_slab

use vectordata::formats::anode::{self, ANode};
use vectordata::formats::anode_vernacular::{self, Vernacular};
use vectordata::formats::mnode::{MNode, MValue};
use vectordata::formats::pnode::{
    Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode,
};
use vectordata::formats::pnode::eval::evaluate;

use slabtastic::{SlabReader, SlabWriter, WriterConfig};

/// Build a catalog of image metadata records.
fn build_metadata() -> Vec<MNode> {
    let records = [
        ("Sunset over the Pacific Ocean",     false, 0.92, 1920, 1080, "landscape"),
        ("Portrait of a Siamese cat",         false, 0.87, 800,  1200, "animals"),
        ("Abstract neon art installation",    false, 0.71, 3840, 2160, "art"),
        ("Beach volleyball tournament",       false, 0.65, 1280, 720,  "sports"),
        ("Explicit content flagged",          true,  0.44, 640,  480,  "flagged"),
        ("Mountain trail at dawn",            false, 0.95, 2560, 1440, "landscape"),
        ("City skyline at night",             false, 0.88, 4096, 2160, "landscape"),
        ("Macro photograph of a bee",         false, 0.79, 1024, 1024, "animals"),
        ("Renaissance painting reproduction", false, 0.33, 600,  800,  "art"),
        ("Stadium crowd panorama",            false, 0.58, 5120, 1440, "sports"),
    ];

    records
        .iter()
        .map(|(caption, nsfw, sim, w, h, category)| {
            let mut node = MNode::new();
            node.insert("caption".into(), MValue::Text(caption.to_string()));
            node.insert("nsfw".into(), MValue::Bool(*nsfw));
            node.insert("similarity".into(), MValue::Float(*sim));
            node.insert("width".into(), MValue::Int(*w));
            node.insert("height".into(), MValue::Int(*h));
            node.insert("category".into(), MValue::Text(category.to_string()));
            node
        })
        .collect()
}

/// Build a set of predicate trees representing different search filters.
fn build_predicates() -> Vec<(&'static str, PNode)> {
    vec![
        // High-quality landscape images
        (
            "high-quality landscapes",
            PNode::Conjugate(ConjugateNode {
                conjugate_type: ConjugateType::And,
                children: vec![
                    PNode::Predicate(PredicateNode {
                        field: FieldRef::Named("similarity".into()),
                        op: OpType::Ge,
                        comparands: vec![Comparand::Float(0.85)],
                    }),
                    PNode::Predicate(PredicateNode {
                        field: FieldRef::Named("category".into()),
                        op: OpType::Eq,
                        comparands: vec![Comparand::Text("landscape".into())],
                    }),
                    PNode::Predicate(PredicateNode {
                        field: FieldRef::Named("nsfw".into()),
                        op: OpType::Eq,
                        comparands: vec![Comparand::Bool(false)],
                    }),
                ],
            }),
        ),
        // Wide images (panoramas)
        (
            "wide images (aspect > 2:1)",
            PNode::Conjugate(ConjugateNode {
                conjugate_type: ConjugateType::And,
                children: vec![
                    PNode::Predicate(PredicateNode {
                        field: FieldRef::Named("width".into()),
                        op: OpType::Ge,
                        comparands: vec![Comparand::Int(2560)],
                    }),
                    PNode::Predicate(PredicateNode {
                        field: FieldRef::Named("nsfw".into()),
                        op: OpType::Eq,
                        comparands: vec![Comparand::Bool(false)],
                    }),
                ],
            }),
        ),
        // Category filter using IN
        (
            "animals or art",
            PNode::Predicate(PredicateNode {
                field: FieldRef::Named("category".into()),
                op: OpType::In,
                comparands: vec![
                    Comparand::Text("animals".into()),
                    Comparand::Text("art".into()),
                ],
            }),
        ),
        // Low similarity (possible duplicates or mismatches)
        (
            "low similarity (< 0.5)",
            PNode::Conjugate(ConjugateNode {
                conjugate_type: ConjugateType::And,
                children: vec![
                    PNode::Predicate(PredicateNode {
                        field: FieldRef::Named("similarity".into()),
                        op: OpType::Lt,
                        comparands: vec![Comparand::Float(0.5)],
                    }),
                    PNode::Predicate(PredicateNode {
                        field: FieldRef::Named("nsfw".into()),
                        op: OpType::Eq,
                        comparands: vec![Comparand::Bool(false)],
                    }),
                ],
            }),
        ),
    ]
}

fn main() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let metadata_path = tmp.path().join("metadata.slab");
    let predicates_path = tmp.path().join("predicates.slab");

    // ── Build and store metadata ──────────────────────────────────────

    let metadata = build_metadata();

    {
        let mut writer =
            SlabWriter::new(&metadata_path, WriterConfig::default()).expect("create metadata slab");
        for record in &metadata {
            let bytes = anode::encode(&ANode::MNode(record.clone()));
            writer.add_record(&bytes).expect("write metadata record");
        }
        writer.finish().expect("finalize metadata slab");
    }

    // ── Build and store predicates ────────────────────────────────────

    let predicates = build_predicates();

    {
        let mut writer = SlabWriter::new(&predicates_path, WriterConfig::default())
            .expect("create predicates slab");
        for (_label, pred) in &predicates {
            let bytes = anode::encode(&ANode::PNode(pred.clone()));
            writer.add_record(&bytes).expect("write predicate record");
        }
        writer.finish().expect("finalize predicates slab");
    }

    // ── Read back and generate SQL ────────────────────────────────────

    let meta_reader = SlabReader::open(&metadata_path).expect("open metadata slab");
    let pred_reader = SlabReader::open(&predicates_path).expect("open predicates slab");

    // Decode all metadata records
    let mut decoded_meta: Vec<MNode> = Vec::new();
    for i in 0..metadata.len() {
        let bytes = meta_reader.get(i as i64).expect("read metadata record");
        match anode::decode(&bytes).expect("decode metadata") {
            ANode::MNode(m) => decoded_meta.push(m),
            _ => panic!("expected MNode"),
        }
    }

    // Decode all predicates
    let mut decoded_preds: Vec<PNode> = Vec::new();
    for i in 0..predicates.len() {
        let bytes = pred_reader.get(i as i64).expect("read predicate record");
        match anode::decode(&bytes).expect("decode predicate") {
            ANode::PNode(p) => decoded_preds.push(p),
            _ => panic!("expected PNode"),
        }
    }

    // ── DDL: CREATE TABLE from schema ─────────────────────────────────

    println!("-- =============================================================");
    println!("-- DDL generated from MNode schema (first record as exemplar)");
    println!("-- =============================================================\n");

    let schema_sql = anode_vernacular::render(
        &ANode::MNode(decoded_meta[0].clone()),
        Vernacular::SqlSchema,
    );
    println!("CREATE TABLE image_catalog (");
    println!("    id     SERIAL PRIMARY KEY,");
    println!("    {}", schema_sql);
    println!(");\n");

    // ── DML: INSERT statements ────────────────────────────────────────

    println!("-- =============================================================");
    println!("-- INSERT statements generated from MNode records in slab");
    println!("-- =============================================================\n");

    let columns: Vec<&str> = decoded_meta[0].fields.keys().map(|k| k.as_str()).collect();
    let col_list = columns.join(", ");

    for (i, record) in decoded_meta.iter().enumerate() {
        let values = anode_vernacular::render(&ANode::MNode(record.clone()), Vernacular::Sql);
        println!(
            "INSERT INTO image_catalog ({}) VALUES ({});  -- row {}",
            col_list, values, i
        );
    }

    // ── Queries: SELECT with WHERE from predicates ────────────────────

    println!("\n-- =============================================================");
    println!("-- SELECT queries generated from PNode predicates in slab");
    println!("-- =============================================================\n");

    for (i, pred) in decoded_preds.iter().enumerate() {
        let label = predicates[i].0;
        let where_clause =
            anode_vernacular::render(&ANode::PNode(pred.clone()), Vernacular::Sql);
        println!("-- Filter: {}", label);
        println!(
            "SELECT {} FROM image_catalog WHERE {};",
            col_list, where_clause
        );

        // Evaluate the predicate against all records and show matches
        let matches: Vec<usize> = decoded_meta
            .iter()
            .enumerate()
            .filter(|(_, m)| evaluate(pred, m))
            .map(|(idx, _)| idx)
            .collect();

        if matches.is_empty() {
            println!("-- Matches: (none)");
        } else {
            let match_captions: Vec<&str> = matches
                .iter()
                .map(|&idx| match decoded_meta[idx].fields.get("caption") {
                    Some(MValue::Text(s)) => s.as_str(),
                    _ => "?",
                })
                .collect();
            println!("-- Matches: rows {:?}", matches);
            for caption in &match_captions {
                println!("--   {}", caption);
            }
        }
        println!();
    }

    // ── Summary ───────────────────────────────────────────────────────

    println!("-- =============================================================");
    println!("-- Summary");
    println!("-- =============================================================");
    println!(
        "-- Metadata slab:   {} records from {}",
        decoded_meta.len(),
        metadata_path.display()
    );
    println!(
        "-- Predicate slab:  {} predicates from {}",
        decoded_preds.len(),
        predicates_path.display()
    );
    println!("-- Vernacular:      SQL (via ANode Stage 2 codec)");
}
