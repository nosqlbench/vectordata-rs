// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate PNode predicates from metadata slab statistics.
//!
//! Surveys a metadata slab file to gather per-field value distributions, then
//! generates a configurable number of typed PNode predicates with targeted
//! selectivity. The generated predicates are written to an output slab file.

use std::path::Path;
use std::time::Instant;

use rand::Rng;
use slabtastic::{SlabWriter, WriterConfig};

use crate::formats::pnode::{
    Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode,
};
use crate::pipeline::command::{
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
    render_options_table,
};
use crate::pipeline::rng;

use super::slab::{survey_from_json, survey_slab, FieldStats};

/// Pipeline command: generate predicates from metadata slab field distributions.
pub struct GenPredicatesOp;

/// Create a boxed `GenPredicatesOp` command.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenPredicatesOp)
}

impl CommandOp for GenPredicatesOp {
    fn command_path(&self) -> &str {
        "generate predicates"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate random predicate trees from metadata survey".into(),
            body: format!(r#"# generate predicates

Generate random predicate trees from metadata survey.

## Description

Surveys a metadata slab file to gather per-field value distributions, then
generates a configurable number of typed PNode predicates with targeted
selectivity. The generated predicates are written to an output slab file.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let input_path = resolve_path(input_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        let count: usize = options
            .get("count")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);
        let selectivity: f64 = options
            .get("selectivity")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.1);
        let selectivity_max: Option<f64> = options
            .get("selectivity-max")
            .and_then(|s| s.parse().ok());
        let max_samples: usize = options
            .get("samples")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);
        let max_distinct: usize = options
            .get("max-distinct")
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);
        let strategy = options
            .get("strategy")
            .unwrap_or("eq");
        let seed: u64 = rng::parse_seed(options.get("seed"));

        // Load survey data — from pre-computed JSON if available, otherwise survey the slab
        let survey = if let Some(survey_str) = options.get("survey") {
            let survey_path = resolve_path(survey_str, &ctx.workspace);
            ctx.display.log(&format!("generate predicates: loading survey from {}", survey_path.display()));
            match survey_from_json(&survey_path) {
                Ok(s) => s,
                Err(e) => return error_result(e, start),
            }
        } else {
            match survey_slab(&input_path, max_samples, max_distinct) {
                Ok(s) => s,
                Err(e) => return error_result(e, start),
            }
        };

        // Identify eligible fields
        let eligible: Vec<(&String, &FieldStats)> = survey
            .field_stats
            .iter()
            .filter(|(_, fs)| is_eligible(fs))
            .collect();

        if eligible.is_empty() {
            ctx.display.log("generate predicates: 0 eligible fields, producing 0 predicates");
            // Write empty slab
            let config = WriterConfig::new(512, 4096, u32::MAX, false)
                .map_err(|e| format!("{}", e));
            match config {
                Ok(c) => {
                    let writer = SlabWriter::new(&output_path, c);
                    match writer {
                        Ok(mut w) => { let _ = w.finish(); }
                        Err(e) => return error_result(format!("failed to create output: {}", e), start),
                    }
                }
                Err(e) => return error_result(e, start),
            }
            return CommandResult {
                status: Status::Ok,
                message: "0 predicates generated (no eligible fields)".into(),
                produced: vec![output_path.clone()],
                elapsed: start.elapsed(),
            };
        }

        let mut rng_inst = rng::seeded_rng(seed);

        // Generate predicates
        let mut predicates: Vec<Vec<u8>> = Vec::with_capacity(count);
        for _ in 0..count {
            let target_sel = match selectivity_max {
                Some(max) => rng_inst.random_range(selectivity..=max),
                None => selectivity,
            };

            let pnode = match strategy {
                "eq" => generate_eq_predicate(&eligible, target_sel, &mut rng_inst),
                "compound" => generate_compound_predicate(&eligible, target_sel, &mut rng_inst),
                other => {
                    return error_result(
                        format!("unknown strategy '{}', expected 'eq' or 'compound'", other),
                        start,
                    );
                }
            };
            predicates.push(pnode.to_bytes_named());
        }

        // Write to output slab
        let config = match WriterConfig::new(512, 4096, u32::MAX, false) {
            Ok(c) => c,
            Err(e) => return error_result(format!("writer config error: {}", e), start),
        };
        let mut writer = match SlabWriter::new(&output_path, config) {
            Ok(w) => w,
            Err(e) => return error_result(format!("failed to create output: {}", e), start),
        };
        for rec in &predicates {
            if let Err(e) = writer.add_record(rec) {
                return error_result(format!("write error: {}", e), start);
            }
        }
        if let Err(e) = writer.finish() {
            return error_result(format!("finish error: {}", e), start);
        }

        let message = format!(
            "{} predicates generated from {} eligible fields",
            predicates.len(),
            eligible.len(),
        );
        ctx.display.log(&format!("generate predicates: {}", message));

        CommandResult {
            status: Status::Ok,
            message,
            produced: vec![output_path.clone()],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("input", "Path", true, None, "Metadata slab to survey"),
            opt("output", "Path", true, None, "Output slab for predicates"),
            opt("survey", "Path", false, None, "Pre-computed survey JSON from 'slab survey --output' (skips re-surveying the slab)"),
            opt("count", "int", false, Some("100"), "Number of predicates to generate"),
            opt("selectivity", "float", false, Some("0.1"), "Target selectivity (0.0–1.0)"),
            opt("selectivity-max", "float", false, None, "If set, selectivity is uniform in [selectivity, selectivity-max]"),
            opt("samples", "int", false, Some("1000"), "Survey sample count"),
            opt("max-distinct", "int", false, Some("100"), "Max distinct values tracked per field during survey"),
            opt("strategy", "string", false, Some("eq"), "Predicate strategy: 'eq' (single-field Eq, default) or 'compound' (multi-field AND with mixed ops)"),
            opt("seed", "int", false, Some("42"), "RNG seed"),
        ]
    }
}

/// Supported MNode type tags for predicate generation (lowercase, matching TypeTag::name()).
const ELIGIBLE_TYPES: &[&str] = &[
    "int", "int32", "short", "millis",
    "float", "float32",
    "text", "ascii", "enum_str",
    "bool",
];

/// Check if a field's statistics indicate it is eligible for predicate generation.
fn is_eligible(fs: &FieldStats) -> bool {
    if fs.count == 0 || fs.null_count == fs.count {
        return false;
    }
    // At least one eligible type must be present
    fs.type_counts.keys().any(|t| ELIGIBLE_TYPES.contains(&t.as_str()))
}

/// Determine the dominant (most frequent) type tag for a field.
fn dominant_type(fs: &FieldStats) -> Option<&str> {
    fs.type_counts
        .iter()
        .filter(|(t, _)| ELIGIBLE_TYPES.contains(&t.as_str()))
        .max_by_key(|(_, cnt)| *cnt)
        .map(|(t, _)| t.as_str())
}

/// Generate a simple Eq predicate on a single randomly chosen field.
///
/// Picks one eligible field at random and generates a single `Eq` comparand
/// from the field's observed distinct values. For numeric fields without
/// distinct values, falls back to `Eq` on a random value in the observed range.
fn generate_eq_predicate(
    eligible: &[(&String, &FieldStats)],
    _target_selectivity: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> PNode {
    let idx = rng.random_range(0..eligible.len());
    let (name, fs) = &eligible[idx];
    let field = FieldRef::Named(name.to_string());

    let dtype = dominant_type(fs);

    match dtype {
        Some("text" | "ascii" | "enum_str") => {
            if !fs.distinct.is_empty() {
                let keys: Vec<&String> = fs.distinct.keys().collect();
                let pick = rng.random_range(0..keys.len());
                let raw = keys[pick].as_str();
                let cleaned = if raw.starts_with('\'') && raw.ends_with('\'') && raw.len() >= 2 {
                    &raw[1..raw.len() - 1]
                } else {
                    raw
                };
                PNode::Predicate(PredicateNode {
                    field,
                    op: OpType::Eq,
                    comparands: vec![Comparand::Text(cleaned.to_string())],
                })
            } else {
                // No distinct values available — use Null check as fallback
                PNode::Predicate(PredicateNode {
                    field,
                    op: OpType::Eq,
                    comparands: vec![Comparand::Null],
                })
            }
        }
        Some("bool") => {
            let val = rng.random_bool(0.5);
            PNode::Predicate(PredicateNode {
                field,
                op: OpType::Eq,
                comparands: vec![Comparand::Bool(val)],
            })
        }
        Some("int" | "int32" | "short" | "millis") => {
            if !fs.distinct.is_empty() {
                let keys: Vec<&String> = fs.distinct.keys().collect();
                let pick = rng.random_range(0..keys.len());
                if let Ok(v) = keys[pick].parse::<i64>() {
                    return PNode::Predicate(PredicateNode {
                        field,
                        op: OpType::Eq,
                        comparands: vec![Comparand::Int(v)],
                    });
                }
            }
            // Fallback: random value in observed range
            let min_val = fs.numeric_min as i64;
            let max_val = fs.numeric_max as i64;
            let val = if max_val > min_val {
                rng.random_range(min_val..=max_val)
            } else {
                min_val
            };
            PNode::Predicate(PredicateNode {
                field,
                op: OpType::Eq,
                comparands: vec![Comparand::Int(val)],
            })
        }
        Some("float" | "float32") => {
            let min_val = fs.numeric_min;
            let max_val = fs.numeric_max;
            let val = if max_val > min_val {
                rng.random_range(min_val..=max_val)
            } else {
                min_val
            };
            PNode::Predicate(PredicateNode {
                field,
                op: OpType::Eq,
                comparands: vec![Comparand::Float(val)],
            })
        }
        _ => {
            // Ultimate fallback
            PNode::Predicate(PredicateNode {
                field,
                op: OpType::Eq,
                comparands: vec![Comparand::Null],
            })
        }
    }
}

/// Generate a compound predicate AND-ing sub-predicates for each eligible field.
fn generate_compound_predicate(
    eligible: &[(&String, &FieldStats)],
    target_selectivity: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> PNode {
    let n = eligible.len() as f64;
    let per_field_sel = target_selectivity.powf(1.0 / n);

    let mut children: Vec<PNode> = Vec::new();
    for (name, fs) in eligible {
        if let Some(sub) = generate_field_predicate(name, fs, per_field_sel, rng) {
            children.push(sub);
        }
    }

    match children.len() {
        0 => PNode::Predicate(PredicateNode {
            field: FieldRef::Named(eligible[0].0.clone()),
            op: OpType::Ge,
            comparands: vec![Comparand::Int(i64::MIN)],
        }),
        1 => children.into_iter().next().unwrap(),
        _ => PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children,
        }),
    }
}

/// Generate a sub-predicate for a single field based on its dominant type.
fn generate_field_predicate(
    name: &str,
    fs: &FieldStats,
    per_field_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<PNode> {
    let dtype = dominant_type(fs)?;
    let field = FieldRef::Named(name.to_string());

    match dtype {
        "int" | "int32" | "short" | "millis" => {
            Some(generate_int_predicate(field, fs, per_field_sel, rng))
        }
        "float" | "float32" => {
            Some(generate_float_predicate(field, fs, per_field_sel, rng))
        }
        "text" | "ascii" | "enum_str" => {
            generate_text_predicate(field, fs, per_field_sel, rng)
        }
        "bool" => {
            let val = rng.random_bool(0.5);
            Some(PNode::Predicate(PredicateNode {
                field,
                op: OpType::Eq,
                comparands: vec![Comparand::Bool(val)],
            }))
        }
        _ => None,
    }
}

/// Generate an integer predicate — Eq/In for few distinct values, range for many.
fn generate_int_predicate(
    field: FieldRef,
    fs: &FieldStats,
    per_field_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> PNode {
    let distinct_count = fs.distinct.len();

    if distinct_count > 0 && !fs.distinct_overflow {
        // Few distinct values — use Eq or In
        let num_pick = ((distinct_count as f64 * per_field_sel).ceil() as usize).max(1).min(distinct_count);
        let distinct_vals: Vec<&String> = fs.distinct.keys().collect();
        let mut indices: Vec<usize> = (0..distinct_count).collect();
        // Partial shuffle
        for i in 0..num_pick {
            let j = rng.random_range(i..indices.len());
            indices.swap(i, j);
        }
        let picked: Vec<Comparand> = indices[..num_pick]
            .iter()
            .filter_map(|&i| {
                distinct_vals[i]
                    .parse::<i64>()
                    .ok()
                    .map(Comparand::Int)
            })
            .collect();

        if picked.is_empty() {
            // Fallback to range
            return generate_int_range_predicate(field, fs, per_field_sel, rng);
        }

        if picked.len() == 1 {
            PNode::Predicate(PredicateNode { field, op: OpType::Eq, comparands: picked })
        } else {
            PNode::Predicate(PredicateNode { field, op: OpType::In, comparands: picked })
        }
    } else {
        generate_int_range_predicate(field, fs, per_field_sel, rng)
    }
}

/// Generate an integer range predicate using Ge + Lt AND.
fn generate_int_range_predicate(
    field: FieldRef,
    fs: &FieldStats,
    per_field_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> PNode {
    let min_val = fs.numeric_min as i64;
    let max_val = fs.numeric_max as i64;
    let range = (max_val - min_val).max(1);
    let width = ((range as f64 * per_field_sel).ceil() as i64).max(1).min(range);
    let max_start = (max_val - width).max(min_val);
    let start = if max_start > min_val {
        rng.random_range(min_val..=max_start)
    } else {
        min_val
    };

    if width == 1 {
        PNode::Predicate(PredicateNode {
            field,
            op: OpType::Eq,
            comparands: vec![Comparand::Int(start)],
        })
    } else {
        PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: field.clone(),
                    op: OpType::Ge,
                    comparands: vec![Comparand::Int(start)],
                }),
                PNode::Predicate(PredicateNode {
                    field,
                    op: OpType::Lt,
                    comparands: vec![Comparand::Int(start + width)],
                }),
            ],
        })
    }
}

/// Generate a float range predicate using Ge + Lt AND.
fn generate_float_predicate(
    field: FieldRef,
    fs: &FieldStats,
    per_field_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> PNode {
    let min_val = fs.numeric_min;
    let max_val = fs.numeric_max;
    let range = (max_val - min_val).max(f64::MIN_POSITIVE);
    let width = (range * per_field_sel).max(f64::MIN_POSITIVE);
    let max_start = (max_val - width).max(min_val);
    let start = if max_start > min_val {
        rng.random_range(min_val..=max_start)
    } else {
        min_val
    };

    PNode::Conjugate(ConjugateNode {
        conjugate_type: ConjugateType::And,
        children: vec![
            PNode::Predicate(PredicateNode {
                field: field.clone(),
                op: OpType::Ge,
                comparands: vec![Comparand::Float(start)],
            }),
            PNode::Predicate(PredicateNode {
                field,
                op: OpType::Lt,
                comparands: vec![Comparand::Float(start + width)],
            }),
        ],
    })
}

/// Generate a text predicate — Eq or In from distinct values.
fn generate_text_predicate(
    field: FieldRef,
    fs: &FieldStats,
    per_field_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<PNode> {
    if fs.distinct.is_empty() {
        return None;
    }

    let distinct_count = fs.distinct.len();
    let num_pick = ((distinct_count as f64 * per_field_sel).ceil() as usize).max(1).min(distinct_count);
    let distinct_vals: Vec<&String> = fs.distinct.keys().collect();
    let mut indices: Vec<usize> = (0..distinct_count).collect();
    for i in 0..num_pick {
        let j = rng.random_range(i..indices.len());
        indices.swap(i, j);
    }

    let picked: Vec<Comparand> = indices[..num_pick]
        .iter()
        .map(|&i| {
            // Distinct values from MValue Display have surrounding quotes for text — strip them
            let raw = distinct_vals[i].as_str();
            let cleaned = if raw.starts_with('\'') && raw.ends_with('\'') && raw.len() >= 2 {
                &raw[1..raw.len() - 1]
            } else {
                raw
            };
            Comparand::Text(cleaned.to_string())
        })
        .collect();

    if picked.len() == 1 {
        Some(PNode::Predicate(PredicateNode { field, op: OpType::Eq, comparands: picked }))
    } else {
        Some(PNode::Predicate(PredicateNode { field, op: OpType::In, comparands: picked }))
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> std::path::PathBuf {
    let p = std::path::PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, desc: &str) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::mnode::{MNode, MValue};
    use crate::formats::pnode::PNode as FmtPNode;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;
    use slabtastic::SlabReader;

    fn test_ctx(dir: &std::path::Path) -> StreamContext {
        StreamContext {
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            display: crate::pipeline::display::ProgressDisplay::new(),
        }
    }

    fn create_test_metadata_slab(dir: &std::path::Path, name: &str, records: Vec<MNode>) -> std::path::PathBuf {
        let path = dir.join(name);
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut writer = SlabWriter::new(&path, config).unwrap();
        for node in &records {
            let bytes = node.to_bytes();
            writer.add_record(&bytes).unwrap();
        }
        writer.finish().unwrap();
        path
    }

    fn make_test_records() -> Vec<MNode> {
        let mut records = Vec::new();
        for i in 0..20 {
            let mut node = MNode::new();
            node.insert("user_id".into(), MValue::Int(i));
            node.insert("name".into(), MValue::Text(format!("user_{}", i % 5)));
            node.insert("score".into(), MValue::Float(i as f64 * 1.5));
            node.insert("active".into(), MValue::Bool(i % 2 == 0));
            records.push(node);
        }
        records
    }

    #[test]
    fn test_generate_predicates_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let records = make_test_records();
        let input_path = create_test_metadata_slab(ws, "meta.slab", records);
        let output_path = ws.join("predicates.slab");

        let mut opts = Options::new();
        opts.set("input", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "10".to_string());
        opts.set("seed", "42".to_string());

        let mut op = GenPredicatesOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "gen predicates failed: {}", result.message);
        assert!(result.message.contains("10 predicates generated"), "message: {}", result.message);

        // Verify output slab has decodable PNodes
        let reader = SlabReader::open(&output_path).unwrap();
        for i in 0..10 {
            let data = reader.get(i).unwrap();
            let pnode = FmtPNode::from_bytes_named(&data);
            assert!(pnode.is_ok(), "predicate {} failed to decode: {:?}", i, pnode.err());
            // Check that the pnode references field names from the schema
            let pnode = pnode.unwrap();
            let text = format!("{}", pnode);
            let has_field = text.contains("user_id")
                || text.contains("name")
                || text.contains("score")
                || text.contains("active");
            assert!(has_field, "predicate {} doesn't mention any field: {}", i, text);
        }
    }

    #[test]
    fn test_generate_predicates_selectivity_range() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let records = make_test_records();
        let input_path = create_test_metadata_slab(ws, "meta.slab", records);
        let output_path = ws.join("predicates.slab");

        let mut opts = Options::new();
        opts.set("input", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "20".to_string());
        opts.set("selectivity", "0.05".to_string());
        opts.set("selectivity-max", "0.5".to_string());
        opts.set("seed", "99".to_string());

        let mut op = GenPredicatesOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "gen predicates failed: {}", result.message);
        assert!(result.message.contains("20 predicates generated"), "message: {}", result.message);

        // Verify we can read all 20
        let reader = SlabReader::open(&output_path).unwrap();
        for i in 0..20 {
            let data = reader.get(i).unwrap();
            assert!(FmtPNode::from_bytes_named(&data).is_ok());
        }
    }

    #[test]
    fn test_generate_predicates_empty_slab() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Create an empty slab
        let input_path = ws.join("empty.slab");
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut writer = SlabWriter::new(&input_path, config).unwrap();
        writer.finish().unwrap();

        let output_path = ws.join("predicates.slab");

        let mut opts = Options::new();
        opts.set("input", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "10".to_string());

        let mut op = GenPredicatesOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "gen predicates failed: {}", result.message);
        assert!(result.message.contains("0 predicates"), "message: {}", result.message);
    }
}
