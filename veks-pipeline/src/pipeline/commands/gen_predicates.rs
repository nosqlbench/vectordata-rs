// Copyright (c) nosqlbench contributors
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

use veks_core::formats::pnode::{
    Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode,
};
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
};
use crate::pipeline::rng;

use super::slab::{survey_from_json, survey_slab, FieldStats};

/// Pipeline command: synthesize predicates from metadata slab field distributions.
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

Surveys a metadata slab file (or reads a pre-computed survey JSON) to
understand the value distributions and cardinalities of each field, then
generates a configurable number of PNode predicate trees targeting a
specified selectivity. The generated predicates are written to an output
slab file where each record is a serialized PNode.

## Survey phase

Before generating predicates, the command needs to understand what fields
exist in the metadata and what values they contain. This is done by
reading a sample of records from the metadata slab and collecting per-field
statistics: distinct value counts, value frequencies, numeric min/max
ranges, and type distributions. Only fields with eligible types (int,
float, text, bool, and related variants) are considered for predicate
generation.

If a `survey` option is provided pointing to a pre-computed survey JSON
file, the slab survey is skipped entirely and the cached statistics are
loaded instead. This is useful when the same metadata slab is used to
generate multiple predicate batches.

## Predicate generation strategies

The `strategy` parameter controls the structure of the generated
predicates:

- **eq** (default) -- generates single-field `Eq` predicates. The field
  is chosen by matching its cardinality to the target selectivity (a field
  with N distinct values has per-value selectivity of roughly 1/N). Within
  the chosen field, a value is selected whose observed frequency is closest
  to the target.
- **compound** -- generates multi-field AND predicates. Fields are added
  greedily until the cumulative estimated selectivity reaches the target.
  Each sub-predicate may use Eq, In, or range (Ge/Lt) operators depending
  on the field's type and cardinality.

## Selectivity control

The `selectivity` parameter (0.0 to 1.0) specifies the target fraction of
base records that should satisfy each predicate. If `selectivity-max` is
also set, each predicate's target selectivity is drawn uniformly from
`[selectivity, selectivity-max]`, producing a mix of easy and hard
predicates within a single batch.

## Role in dataset pipelines

This command creates the predicate workload for filtered KNN evaluation.
It is typically followed by `compute predicates`, which evaluates each
generated predicate against the full metadata slab to produce a boolean
answer key (which records match). Together, the predicate slab and answer
key slab provide the ground truth needed to measure filtered-search recall.

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
            ctx.ui.log(&format!("synthesize predicates: loading survey from {}", survey_path.display()));
            match survey_from_json(&survey_path) {
                Ok(s) => s,
                Err(e) => return error_result(e, start),
            }
        } else {
            match survey_slab(&input_path, max_samples, max_distinct, Some(&ctx.ui)) {
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
            ctx.ui.log("synthesize predicates: 0 eligible fields, producing 0 predicates");
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
            // Write verified count for the bound checker
            let var_name = format!("verified_count:{}",
                output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
            let _ = crate::pipeline::variables::set_and_save(
                &ctx.workspace, &var_name, "0");
            ctx.defaults.insert(var_name, "0".to_string());

            return CommandResult {
                status: Status::Ok,
                message: "0 predicates generated (no eligible fields)".into(),
                produced: vec![output_path.clone()],
                elapsed: start.elapsed(),
            };
        }

        let mut rng_inst = rng::seeded_rng(seed);

        // Generate predicates
        let pb = ctx.ui.bar(count as u64, "generating predicates");
        let mut predicates: Vec<Vec<u8>> = Vec::with_capacity(count);
        for pred_i in 0..count {
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
            if (pred_i + 1) % 1_000 == 0 { pb.set_position((pred_i + 1) as u64); }
        }
        pb.finish();

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

        // Write verified count for the bound checker
        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &predicates.len().to_string());
        ctx.defaults.insert(var_name, predicates.len().to_string());

        let message = format!(
            "{} predicates generated from {} eligible fields",
            predicates.len(),
            eligible.len(),
        );
        ctx.ui.log(&format!("synthesize predicates: {}", message));

        CommandResult {
            status: Status::Ok,
            message,
            produced: vec![output_path.clone()],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("input", "Path", true, None, "Metadata slab to survey", OptionRole::Input),
            opt("output", "Path", true, None, "Output slab for predicates", OptionRole::Output),
            opt("survey", "Path", false, None, "Pre-computed survey JSON from 'survey --output' (skips re-surveying the slab)", OptionRole::Input),
            opt("count", "int", false, Some("100"), "Number of predicates to generate", OptionRole::Config),
            opt("selectivity", "float", false, Some("0.1"), "Target selectivity (0.0–1.0)", OptionRole::Config),
            opt("selectivity-max", "float", false, None, "If set, selectivity is uniform in [selectivity, selectivity-max]", OptionRole::Config),
            opt("samples", "int", false, Some("1000"), "Survey sample count", OptionRole::Config),
            opt("max-distinct", "int", false, Some("100"), "Max distinct values tracked per field during survey", OptionRole::Config),
            opt("strategy", "string", false, Some("eq"), "Predicate strategy: 'eq' (single-field Eq, default) or 'compound' (multi-field AND with mixed ops)", OptionRole::Config),
            opt("seed", "int", false, Some("42"), "RNG seed", OptionRole::Config),
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["input", "survey"],
            &["output"],
        )
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
/// from the field's observed distinct values. The target selectivity is used
/// to prefer fields whose cardinality is close to `1/target_selectivity`, and
/// within a field, to prefer lower-frequency values that approximate the
/// target.
fn generate_eq_predicate(
    eligible: &[(&String, &FieldStats)],
    target_selectivity: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> PNode {
    // Prefer fields whose per-value selectivity is close to the target.
    // For a field with N distinct values, per-value sel ≈ 1/N (uniform).
    // Pick the field whose 1/N is closest to target_selectivity.
    let target_distinct = (1.0 / target_selectivity.max(1e-12)).round() as usize;

    // Filter to fields that have enough cardinality (non-overflow, distinct > 1)
    let selective_fields: Vec<usize> = (0..eligible.len())
        .filter(|&i| {
            let fs = eligible[i].1;
            fs.distinct.len() > 1 || fs.distinct_overflow
        })
        .collect();

    let idx = if selective_fields.is_empty() {
        rng.random_range(0..eligible.len())
    } else {
        // Weight by closeness of 1/distinct to target; overflow fields get
        // high cardinality estimate
        let mut best_idx = selective_fields[0];
        let mut best_diff = f64::MAX;
        for &i in &selective_fields {
            let fs = eligible[i].1;
            let est_distinct = if fs.distinct_overflow {
                fs.count.max(1)
            } else {
                fs.distinct.len().max(1)
            };
            let diff = (est_distinct as f64 - target_distinct as f64).abs();
            // Add jitter to avoid always picking the same field
            let jittered = diff + rng.random_range(0.0..1.0);
            if jittered < best_diff {
                best_diff = jittered;
                best_idx = i;
            }
        }
        best_idx
    };

    let (name, fs) = &eligible[idx];
    let field = FieldRef::Named(name.to_string());
    let dtype = dominant_type(fs);

    match dtype {
        Some("text" | "ascii" | "enum_str") => {
            if !fs.distinct.is_empty() {
                // Pick a value weighted toward lower-frequency values
                let value = pick_value_by_selectivity(fs, target_selectivity, rng);
                PNode::Predicate(PredicateNode {
                    field,
                    op: OpType::Eq,
                    comparands: vec![Comparand::Text(value)],
                })
            } else {
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
            if !fs.distinct.is_empty() && !fs.distinct_overflow {
                let value_str = pick_value_by_selectivity(fs, target_selectivity, rng);
                if let Ok(v) = value_str.parse::<i64>() {
                    return PNode::Predicate(PredicateNode {
                        field,
                        op: OpType::Eq,
                        comparands: vec![Comparand::Int(v)],
                    });
                }
            }
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
            PNode::Predicate(PredicateNode {
                field,
                op: OpType::Eq,
                comparands: vec![Comparand::Null],
            })
        }
    }
}

/// Pick a value from a field's distinct values, preferring values whose
/// observed frequency is closest to `target_selectivity`.
fn pick_value_by_selectivity(
    fs: &FieldStats,
    target_selectivity: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> String {
    let total: usize = fs.distinct.values().sum();
    if total == 0 || fs.distinct.is_empty() {
        return String::new();
    }

    // Score each value: how close is its frequency to the target selectivity?
    let target_count = (total as f64 * target_selectivity).max(1.0);
    let scored: Vec<(&String, f64)> = fs.distinct
        .iter()
        .map(|(val, &cnt)| {
            let diff = ((cnt as f64) - target_count).abs();
            // Inverse distance weight — prefer closer matches
            let weight = 1.0 / (diff + 1.0);
            (val, weight)
        })
        .collect();

    // Weighted random selection
    let total_weight: f64 = scored.iter().map(|(_, w)| w).sum();
    let mut pick = rng.random_range(0.0..total_weight);
    for (val, weight) in &scored {
        pick -= weight;
        if pick <= 0.0 {
            let raw = val.as_str();
            let cleaned = if raw.starts_with('\'') && raw.ends_with('\'') && raw.len() >= 2 {
                &raw[1..raw.len() - 1]
            } else {
                raw
            };
            return cleaned.to_string();
        }
    }

    // Fallback: last value
    let (val, _) = scored.last().unwrap();
    let raw = val.as_str();
    if raw.starts_with('\'') && raw.ends_with('\'') && raw.len() >= 2 {
        raw[1..raw.len() - 1].to_string()
    } else {
        raw.to_string()
    }
}

/// Generate a compound predicate AND-ing sub-predicates for a subset of fields.
///
/// Selects enough fields to achieve the target selectivity, rather than using
/// all eligible fields. Fields are shuffled and added until the cumulative
/// estimated selectivity reaches the target.
fn generate_compound_predicate(
    eligible: &[(&String, &FieldStats)],
    target_selectivity: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> PNode {
    // Shuffle field indices
    let mut indices: Vec<usize> = (0..eligible.len()).collect();
    for i in (1..indices.len()).rev() {
        let j = rng.random_range(0..=i);
        indices.swap(i, j);
    }

    // Greedily add fields until cumulative selectivity is low enough.
    // Each field's contribution is bounded: for N distinct values, the
    // best per-field selectivity is roughly 1/N.
    let mut selected: Vec<usize> = Vec::new();
    let mut cumulative_sel = 1.0f64;

    for &idx in &indices {
        if cumulative_sel <= target_selectivity && !selected.is_empty() {
            break;
        }
        let fs = eligible[idx].1;
        let est_distinct = if fs.distinct_overflow {
            fs.count.max(1) as f64
        } else {
            fs.distinct.len().max(1) as f64
        };
        // This field can contribute at most 1/est_distinct selectivity reduction
        cumulative_sel *= 1.0 / est_distinct;
        selected.push(idx);
    }

    // Compute per-field selectivity target: distribute the overall target
    // evenly across selected fields.
    let n = selected.len().max(1) as f64;
    let per_field_sel = target_selectivity.powf(1.0 / n);

    let mut children: Vec<PNode> = Vec::new();
    for &idx in &selected {
        let (name, fs) = &eligible[idx];
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
///
/// Uses observed value frequencies to pick values whose cumulative frequency
/// is closest to `per_field_sel`.
fn generate_int_predicate(
    field: FieldRef,
    fs: &FieldStats,
    per_field_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> PNode {
    let distinct_count = fs.distinct.len();

    if distinct_count > 0 && !fs.distinct_overflow {
        let total: usize = fs.distinct.values().sum();
        if total == 0 {
            return generate_int_range_predicate(field, fs, per_field_sel, rng);
        }

        // Sort by ascending frequency, shuffle within similar bands
        let mut by_freq: Vec<(&String, usize)> = fs.distinct
            .iter()
            .map(|(k, &v)| (k, v))
            .collect();
        by_freq.sort_by_key(|(_, cnt)| *cnt);
        for i in (1..by_freq.len()).rev() {
            let j = rng.random_range(0..=i);
            let f_i = by_freq[i].1.max(1);
            let f_j = by_freq[j].1.max(1);
            if f_i <= f_j * 2 && f_j <= f_i * 2 {
                by_freq.swap(i, j);
            }
        }

        let target_count = (total as f64 * per_field_sel).max(1.0);
        let mut picked: Vec<Comparand> = Vec::new();
        let mut cumulative = 0usize;

        for (val, cnt) in &by_freq {
            if cumulative as f64 >= target_count && !picked.is_empty() {
                break;
            }
            if let Ok(v) = val.parse::<i64>() {
                picked.push(Comparand::Int(v));
                cumulative += cnt;
            }
        }

        if picked.is_empty() {
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
///
/// Uses observed value frequencies to pick values whose cumulative frequency
/// is closest to `per_field_sel`, rather than assuming uniform distribution.
fn generate_text_predicate(
    field: FieldRef,
    fs: &FieldStats,
    per_field_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<PNode> {
    if fs.distinct.is_empty() {
        return None;
    }

    let total: usize = fs.distinct.values().sum();
    if total == 0 {
        return None;
    }

    // Sort values by ascending frequency so we can greedily pick low-frequency
    // values until we reach the target selectivity.
    let mut by_freq: Vec<(&String, usize)> = fs.distinct
        .iter()
        .map(|(k, &v)| (k, v))
        .collect();
    by_freq.sort_by_key(|(_, cnt)| *cnt);

    // Shuffle values with similar frequency to add randomness
    // (partial shuffle within frequency bands)
    let mut shuffled = by_freq.clone();
    for i in (1..shuffled.len()).rev() {
        let j = rng.random_range(0..=i);
        // Only swap if frequencies are within 2x of each other (similar band)
        let f_i = shuffled[i].1.max(1);
        let f_j = shuffled[j].1.max(1);
        if f_i <= f_j * 2 && f_j <= f_i * 2 {
            shuffled.swap(i, j);
        }
    }

    let target_count = (total as f64 * per_field_sel).max(1.0);
    let mut picked: Vec<Comparand> = Vec::new();
    let mut cumulative = 0usize;

    for (val, cnt) in &shuffled {
        if cumulative as f64 >= target_count && !picked.is_empty() {
            break;
        }
        let raw = val.as_str();
        let cleaned = if raw.starts_with('\'') && raw.ends_with('\'') && raw.len() >= 2 {
            &raw[1..raw.len() - 1]
        } else {
            raw
        };
        picked.push(Comparand::Text(cleaned.to_string()));
        cumulative += cnt;
    }

    if picked.is_empty() {
        return None;
    }

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

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, desc: &str, role: OptionRole) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
        role,
}
}

#[cfg(test)]
mod tests {
    use super::*;
    use veks_core::formats::mnode::{MNode, MValue};
    use veks_core::formats::pnode::PNode as FmtPNode;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;
    use slabtastic::SlabReader;

    fn test_ctx(dir: &std::path::Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
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
