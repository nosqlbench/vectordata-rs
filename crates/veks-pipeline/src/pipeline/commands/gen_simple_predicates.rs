// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate trivial integer-equality predicates from
//! caller-configured ranges.
//!
//! Distinct from [`super::gen_predicates`] in that this command does
//! **not** read any metadata or run a survey — it just emits N
//! predicates of the form `field_i == random_int(range_min, range_max)`,
//! sized to the caller's request. Useful for smoke tests, fixtures,
//! and pipeline stages that need *some* predicate slab without caring
//! about its selectivity profile.
//!
//! For real selectivity-targeted predicates against actual metadata,
//! use `generate predicates`, which consumes the rich `SurveyReport`
//! from `analyze survey` to pick comparands at calibrated quantiles.

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

use super::gen_predicates_common::{error_result, opt, resolve_path};

/// Pipeline command: synthesize integer-equality predicates from
/// configured ranges. Survey-free; selectivity is whatever falls
/// out of the configured range.
pub struct GenSimplePredicatesOp;

/// Create a boxed `GenSimplePredicatesOp` command.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenSimplePredicatesOp)
}

impl CommandOp for GenSimplePredicatesOp {
    fn command_path(&self) -> &str {
        "generate simple-predicates"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_GENERATE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_SECONDARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate trivial integer-equality predicates from configured ranges".into(),
            body: format!(
                r#"# generate simple-predicates

Emit `N` predicates of the form `field_i == random_int(range_min,
range_max)` in slab, ivec, or scalar form. No survey is performed —
the predicates are constructed directly from the caller's
configuration. Selectivity against any particular metadata slab is
whatever falls out of the range, which is generally *not* calibrated.

This command exists for smoke tests, fixtures, and stages that need a
predicate slab without caring about its selectivity profile. For
selectivity-targeted predicates, use `generate predicates`, which
consumes the rich `SurveyReport` from `analyze survey`.

## Options

{}"#,
                render_options_table(&options),
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let output_path = resolve_path(output_str, &ctx.workspace);

        let count: usize = options.get("count").and_then(|s| s.parse().ok()).unwrap_or(100);
        let fields: usize = options.get("fields").and_then(|s| s.parse().ok()).unwrap_or(1);
        let range_min: i32 = options.parse_or("range-min", 0i32).unwrap_or(0);
        let range_max: i32 = options.parse_or("range-max", 100i32).unwrap_or(100);
        let seed: u64 = rng::parse_seed(options.get("seed"));
        let format = options.get("format").unwrap_or("slab");

        if range_max <= range_min {
            return error_result(
                format!("range-max ({}) must be > range-min ({})", range_max, range_min),
                start,
            );
        }

        let range = range_max - range_min;
        let mut rng_inst = rng::seeded_rng(seed);

        if let Some(parent) = output_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        ctx.ui.log(&format!(
            "  generate simple-predicates: {} preds, {} fields, range [{}, {}), seed={}",
            count, fields, range_min, range_max, seed,
        ));

        let is_scalar = matches!(format, "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64" | "i64");
        let scalar_elem_size: usize = match format {
            "u8" | "i8" => 1, "u16" | "i16" => 2, "u32" | "i32" => 4, "u64" | "i64" => 8, _ => 0,
        };

        if is_scalar {
            use std::io::Write;
            let mut f = match std::fs::File::create(&output_path) {
                Ok(f) => std::io::BufWriter::new(f),
                Err(e) => return error_result(format!("create {}: {}", output_path.display(), e), start),
            };
            let pb = ctx.ui.bar(count as u64, "generating predicates");
            for i in 0..count {
                for _ in 0..fields {
                    let val = range_min + rng_inst.random_range(0..range);
                    let write_ok = match scalar_elem_size {
                        1 => f.write_all(&[val as u8]).is_ok(),
                        2 => f.write_all(&(val as i16).to_le_bytes()).is_ok(),
                        4 => f.write_all(&val.to_le_bytes()).is_ok(),
                        8 => f.write_all(&(val as i64).to_le_bytes()).is_ok(),
                        _ => false,
                    };
                    if !write_ok {
                        return error_result("write error".into(), start);
                    }
                }
                if (i + 1) % 10_000 == 0 { pb.set_position((i + 1) as u64); }
            }
            pb.finish();
        } else if format == "ivec" {
            use std::io::Write;
            let mut f = match std::fs::File::create(&output_path) {
                Ok(f) => std::io::BufWriter::new(f),
                Err(e) => return error_result(format!("create {}: {}", output_path.display(), e), start),
            };
            let pb = ctx.ui.bar(count as u64, "generating predicates");
            for i in 0..count {
                if f.write_all(&(fields as i32).to_le_bytes()).is_err() {
                    return error_result("write error".into(), start);
                }
                for _ in 0..fields {
                    let val = range_min + rng_inst.random_range(0..range);
                    if f.write_all(&val.to_le_bytes()).is_err() {
                        return error_result("write error".into(), start);
                    }
                }
                if (i + 1) % 10_000 == 0 { pb.set_position((i + 1) as u64); }
            }
            pb.finish();
        } else {
            let config = match WriterConfig::new(512, 4096, u32::MAX, false) {
                Ok(c) => c,
                Err(e) => return error_result(format!("writer config: {}", e), start),
            };
            let mut writer = match SlabWriter::new(&output_path, config) {
                Ok(w) => w,
                Err(e) => return error_result(format!("create slab: {}", e), start),
            };
            let pb = ctx.ui.bar(count as u64, "generating predicates");
            for i in 0..count {
                let pnode = if fields == 1 {
                    let val = range_min + rng_inst.random_range(0..range);
                    PNode::Predicate(PredicateNode {
                        field: FieldRef::Named("field_0".into()),
                        op: OpType::Eq,
                        comparands: vec![Comparand::Int(val as i64)],
                    })
                } else {
                    let children: Vec<PNode> = (0..fields).map(|fi| {
                        let val = range_min + rng_inst.random_range(0..range);
                        PNode::Predicate(PredicateNode {
                            field: FieldRef::Named(format!("field_{}", fi)),
                            op: OpType::Eq,
                            comparands: vec![Comparand::Int(val as i64)],
                        })
                    }).collect();
                    PNode::Conjugate(ConjugateNode {
                        conjugate_type: ConjugateType::And,
                        children,
                    })
                };
                let bytes = pnode.to_bytes_named();
                if let Err(e) = writer.add_record(&bytes) {
                    return error_result(format!("write record {}: {}", i, e), start);
                }
                if (i + 1) % 10_000 == 0 { pb.set_position((i + 1) as u64); }
            }
            pb.finish();
            if let Err(e) = writer.finish() {
                return error_result(format!("finalize: {}", e), start);
            }
        }

        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &count.to_string());
        ctx.defaults.insert(var_name, count.to_string());

        ctx.ui.log(&format!("  wrote {} predicates to {}", count, output_path.display()));

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} simple-int-eq predicates, {} fields, range [{}, {})",
                count, fields, range_min, range_max,
            ),
            produced: vec![output_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("output", "Path", true, None, "Output file for predicates", OptionRole::Output),
            opt("count", "int", false, Some("100"), "Number of predicates to generate", OptionRole::Config),
            opt("fields", "int", false, Some("1"), "Number of integer fields per predicate", OptionRole::Config),
            opt("range-min", "int", false, Some("0"), "Predicate value range minimum (inclusive)", OptionRole::Config),
            opt("range-max", "int", false, Some("100"), "Predicate value range maximum (exclusive)", OptionRole::Config),
            opt("format", "string", false, Some("slab"), "Output format: 'slab', 'ivec', or a scalar type (u8/i8/u16/i16/u32/i32/u64/i64)", OptionRole::Config),
            opt("seed", "int", false, Some("42"), "RNG seed", OptionRole::Config),
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &[],
            &["output"],
        )
    }
}
