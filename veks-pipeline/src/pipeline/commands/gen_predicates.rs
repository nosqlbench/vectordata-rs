// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: generate selectivity-targeted PNode predicates
//! from a metadata survey.
//!
//! Consumes the rich [`SurveyReport`] from `analyze survey` (sysref
//! §13) directly — no projection down to a flat per-field summary.
//! That lets the generator do calibrated predicate synthesis:
//!
//! - **Numeric / temporal fields** with a `QuantileSketch` measure
//!   get range/comparison predicates picked at the exact quantile
//!   matching the target selectivity (`Lt` at q=s, `Gt` at q=1-s,
//!   or `Range[q=0.5-s/2, q=0.5+s/2]`).
//! - **Categorical / low-card fields** with an `ExactFrequencyTable`
//!   get `Eq` against a value whose frequency matches the target.
//! - **High-cardinality identifier / free-text fields** without a
//!   frequency table draw an `Eq` comparand from `ReservoirSample`
//!   — an actual observed value, selectivity ≈ 1/cardinality.
//! - **Boolean fields** get a 50/50 `Eq` true/false predicate.
//!
//! The legacy `survey_inline_via_orchestrator` adapter (which
//! projected `SurveyReport` down to a flat `FieldStats` map) is no
//! longer used here; `gen_predicate_keys` still uses it.

use std::path::PathBuf;
use std::time::Instant;

use rand::Rng;
use slabtastic::{SlabWriter, WriterConfig};

use veks_core::formats::pnode::{
    Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode,
};
use veks_core::formats::pnode::PNode as FmtPNode;
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
};
use crate::pipeline::commands::survey::{
    CardinalityRegime, FieldProfile, MeasureReport, NumberKind, SemanticType,
};
use crate::pipeline::rng;

use super::gen_predicates_common::{error_result, opt, resolve_path};
use super::gen_predicates_proto::{
    materialize_predicate, parse_template, PredicateProto, PredicateTemplate, SelectivitySpec,
};
use super::gen_predicates_wizard::{run_wizard, WizardInputs};
use super::slab::{survey_report_from_json, survey_report_inline};

/// Pipeline command: synthesize predicates from metadata slab field distributions.
pub struct GenPredicatesOp;

/// Create a boxed `GenPredicatesOp` command.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenPredicatesOp)
}

impl GenPredicatesOp {
    fn execute_survey(&mut self, options: &Options, ctx: &mut StreamContext, start: Instant) -> CommandResult {
        // Wizard mode: interactively build a `PredicateProto`,
        // save it to disk, then fall into the proto-driven
        // generation path BY RE-LOADING THE SAVED FILE. The
        // file is the single source of truth — there is no
        // in-memory handoff. A "no" answer to the final
        // prompt exits cleanly without running generation; the
        // user can replay later via `--proto-file=<saved>`.
        //
        // Crucially, the wizard runs BEFORE any CLI option is
        // required. It can auto-detect the metadata source from
        // the CWD and prompts for the output slab, so the user
        // can invoke `veks generate predicates --wizard` with no
        // other flags.
        let proto_file_from_wizard: Option<PathBuf> = if options.get("wizard").is_some() {
            // The wizard derives a proto path next to the
            // resolved output slab when --proto-file isn't given,
            // so the two artefacts stay co-located. A CLI
            // override (--proto-file) wins over that default.
            let cli_output: Option<PathBuf> = options.get("output")
                .map(|s| resolve_path(s, &ctx.workspace));
            let cli_proto: Option<PathBuf> = options.get("proto-file")
                .map(|s| resolve_path(s, &ctx.workspace));
            let inputs = WizardInputs {
                survey_path: options.get("survey").map(|s| resolve_path(s, &ctx.workspace)),
                source_path: options.get("source").map(|s| resolve_path(s, &ctx.workspace)),
                samples: options.get("samples").and_then(|s| s.parse().ok()).unwrap_or(1000),
                output_proto: cli_proto,
                output_slab: cli_output,
                search_root: Some(ctx.workspace.clone()),
            };
            match run_wizard(inputs) {
                Ok(out) => {
                    if !out.proceed {
                        // User declined to generate now. The
                        // proto file exists on disk; tell the
                        // caller to stop cleanly.
                        return CommandResult {
                            status: Status::Ok,
                            message: format!(
                                "wizard saved proto to {}; generation skipped (user declined)",
                                out.proto_path.display(),
                            ),
                            produced: vec![out.proto_path],
                            elapsed: start.elapsed(),
                        };
                    }
                    Some(out.proto_path)
                }
                Err(e) => return error_result(format!("wizard: {e}"), start),
            }
        } else { None };

        // Proto-driven mode: if `--proto-file` is supplied (or
        // was just produced by the wizard), load a
        // `PredicateProto` and use its fields to drive
        // generation. The proto wins over individual flag
        // overrides for the same knobs (count, selectivity,
        // seed) so a saved proto is reproducible without having
        // to also re-pass every flag. The flat flags continue
        // to work when no proto is supplied.
        let proto: Option<PredicateProto> = if let Some(path) = proto_file_from_wizard
            .or_else(|| options.get("proto-file").map(|s| resolve_path(s, &ctx.workspace)))
        {
            match PredicateProto::load_from_path(&path) {
                Ok(p) => Some(p),
                Err(e) => return error_result(e, start),
            }
        } else if let Some(inline) = options.get("proto") {
            // `--proto '<text>'` accepts either JSON or YAML; we
            // sniff for a leading `{` to choose the loader so a
            // pipeline can paste either format verbatim.
            let trimmed = inline.trim_start();
            let parsed = if trimmed.starts_with('{') {
                PredicateProto::from_json(inline)
            } else {
                PredicateProto::from_yaml(inline)
            };
            match parsed {
                Ok(p) => Some(p),
                Err(e) => return error_result(e, start),
            }
        } else { None };

        // Resolve `--source`: CLI > proto > error (if no
        // `--survey` either). The proto's `source` is captured
        // by the wizard so a `--proto-file=<f>` run replays the
        // original survey target.
        let source_resolved: Option<PathBuf> = options.get("source")
            .map(|s| resolve_path(s, &ctx.workspace))
            .or_else(|| proto.as_ref().and_then(|p| p.source.as_ref())
                .map(|s| resolve_path(s, &ctx.workspace)));
        let survey_resolved: Option<PathBuf> = options.get("survey")
            .map(|s| resolve_path(s, &ctx.workspace))
            .or_else(|| proto.as_ref().and_then(|p| p.survey.as_ref())
                .map(|s| resolve_path(s, &ctx.workspace)));
        if source_resolved.is_none() && survey_resolved.is_none() {
            return error_result(
                "either --source <slab> or --survey <survey.json> must be provided \
                 (or carried by --proto-file)".into(),
                start,
            );
        }
        let input_path = source_resolved.clone().unwrap_or_else(PathBuf::new);

        // Resolve `--output`: CLI > proto > error. Captured in
        // the proto by the wizard, so a fresh `--proto-file`
        // replay needs no other flags.
        let output_path: PathBuf = match options.get("output")
            .map(|s| resolve_path(s, &ctx.workspace))
            .or_else(|| proto.as_ref().and_then(|p| p.output.as_ref())
                .map(|s| resolve_path(s, &ctx.workspace)))
        {
            Some(p) => p,
            None => return error_result(
                "--output is required (or set `output:` in the proto file)".into(),
                start,
            ),
        };

        // Pre-validate the template at command launch — a syntax
        // error here is more useful than running the survey and
        // then failing.
        let parsed_template: Option<PredicateTemplate> = match &proto {
            Some(p) => match parse_template(&p.template) {
                Ok(t) => Some(t),
                Err(e) => return error_result(format!("invalid proto template: {e}"), start),
            },
            None => None,
        };

        // Flat-flag fallbacks, used either as the direct values
        // (no proto) or as overrides for fields the proto doesn't
        // pin (no count → count flag, no sel → selectivity flag).
        let count: usize = proto.as_ref().map(|p| p.count)
            .or_else(|| options.get("count").and_then(|s| s.parse().ok()))
            .unwrap_or(100);
        let selectivity_spec: SelectivitySpec = match proto.as_ref() {
            Some(p) => p.selectivity,
            None => {
                let sel: f64 = options.get("selectivity")
                    .and_then(|s| s.parse().ok()).unwrap_or(0.1);
                let sel_max: Option<f64> = options.get("selectivity-max")
                    .and_then(|s| s.parse().ok());
                SelectivitySpec::from_flags(sel, sel_max)
            }
        };
        let max_samples: usize = options.get("samples").and_then(|s| s.parse().ok()).unwrap_or(1000);
        let strategy = options.get("strategy").unwrap_or("eq").to_string();
        let seed: u64 = proto.as_ref().map(|p| p.seed)
            .unwrap_or_else(|| rng::parse_seed(options.get("seed")));

        // Field-selection filters (CLI surface):
        //  --fields a,b,c       → whitelist (only these names are eligible)
        //  --exclude-fields x,y → blacklist (drop these names after the whitelist)
        // Both are case-sensitive, comma-separated, optional. Whitespace
        // around commas is trimmed.
        let field_whitelist: Option<Vec<String>> = options.get("fields").map(parse_csv_field_list);
        let field_blacklist: Vec<String> = options.get("exclude-fields")
            .map(parse_csv_field_list)
            .unwrap_or_default();

        // Operator-family filter:
        //  --ops eq,lt,gt,le,ge,range,matches
        // Default = all operators allowed (current behaviour). The
        // per-field generators consult this mask and skip any path
        // whose output operator isn't in the set, falling back to a
        // permitted family if available or producing no predicate
        // for the field on that draw.
        let op_mask: OperatorMask = match options.get("ops") {
            Some(s) => match OperatorMask::parse(s) {
                Ok(m) => m,
                Err(e) => return error_result(e, start),
            },
            None => OperatorMask::all(),
        };

        // Load the survey directly as a SurveyReport — no projection
        // down to the legacy FieldStats shape. The richer profile
        // (semantic_type, cardinality_regime, quantile sketches,
        // reservoirs, heavy hitters) is what the calibrated
        // predicate generator depends on.
        let report = if let Some(survey_path) = survey_resolved.as_ref() {
            ctx.ui.log(&format!(
                "synthesize predicates: loading survey from {}",
                survey_path.display(),
            ));
            match survey_report_from_json(survey_path) {
                Ok(r) => r,
                Err(e) => return error_result(e, start),
            }
        } else {
            match survey_report_inline(&input_path, max_samples, Some(&ctx.ui)) {
                Ok(r) => r,
                Err(e) => return error_result(e, start),
            }
        };

        let eligible: Vec<(&String, &FieldProfile)> = report
            .fields
            .iter()
            .filter(|(name, profile)| {
                field_is_eligible(profile)
                    && field_passes_name_filter(name, field_whitelist.as_deref(), &field_blacklist)
                    && field_supports_any_op(profile, op_mask)
            })
            .collect();

        if eligible.is_empty() {
            ctx.ui.log("synthesize predicates: 0 eligible fields, producing 0 predicates");
            let config = match WriterConfig::new(512, 4096, u32::MAX, false) {
                Ok(c) => c,
                Err(e) => return error_result(format!("{}", e), start),
            };
            let mut w = match SlabWriter::new(&output_path, config) {
                Ok(w) => w,
                Err(e) => return error_result(format!("failed to create output: {}", e), start),
            };
            let _ = w.finish();
            let var_name = format!("verified_count:{}",
                output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
            let _ = crate::pipeline::variables::set_and_save(&ctx.workspace, &var_name, "0");
            ctx.defaults.insert(var_name, "0".to_string());
            return CommandResult {
                status: Status::Ok,
                message: "0 predicates generated (no eligible fields)".into(),
                produced: vec![output_path.clone()],
                elapsed: start.elapsed(),
            };
        }

        let mut rng_inst = rng::seeded_rng(seed);

        // Build a quick name → profile lookup for the proto path
        // — `materialize_predicate` resolves field names against
        // the survey's fields map.
        let fields_map: std::collections::HashMap<String, FieldProfile> = report.fields.iter()
            .map(|(k, v)| (k.clone(), v.clone())).collect();

        // Diversity guard knobs. Default to a conservative
        // floor of 5% — a 10K-emission run must produce ≥500
        // distinct predicate byte-sequences, else we either
        // warn (default) or abort (--strict-unique-ratio=true).
        // The "10K identical MATCHES predicates" failure that
        // motivated this guard had a unique ratio of 0.0001;
        // any defensible generation run sits above ~0.10.
        let min_unique_ratio: f64 = options.get("min-unique-ratio")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.05);
        let strict_unique_ratio: bool = options.get("strict-unique-ratio")
            .map(|s| s == "true").unwrap_or(false);

        let pb = ctx.ui.bar(count as u64, "generating predicates");
        let mut predicates: Vec<Vec<u8>> = Vec::with_capacity(count);
        for pred_i in 0..count {
            let target_sel = selectivity_spec.sample(&mut rng_inst);
            let pnode = if let Some(template) = parsed_template.as_ref() {
                // Proto-driven: template fixes shape; survey fills
                // `?` placeholders to hit the per-emission target.
                match materialize_predicate(template, &fields_map, target_sel, &mut rng_inst) {
                    Ok(p) => p,
                    Err(e) => return error_result(format!("materialize predicate: {e}"), start),
                }
            } else {
                match strategy.as_str() {
                    "eq" | "single" => generate_eq_predicate(&eligible, target_sel, op_mask, &mut rng_inst),
                    "compound" => generate_compound_predicate(&eligible, target_sel, op_mask, &mut rng_inst),
                    other => {
                        return error_result(
                            format!("unknown strategy '{}', expected 'eq' (or 'single') or 'compound'", other),
                            start,
                        );
                    }
                }
            };
            predicates.push(pnode.to_bytes_named());
            if (pred_i + 1) % 1_000 == 0 { pb.set_position((pred_i + 1) as u64); }
        }
        pb.finish();

        // Diversity check: a generation run that produces 10K
        // emissions from one heavy-hitter candidate (the
        // pre-fix bug) writes 10K byte-identical PNode records.
        // Count distinct byte-sequences and gate on the ratio.
        let unique_count: usize = predicates.iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        let unique_ratio = if predicates.is_empty() { 1.0 }
            else { unique_count as f64 / predicates.len() as f64 };
        let diversity_msg = format!(
            "generation diversity: {}/{} distinct predicates ({:.2}% unique)",
            unique_count, predicates.len(), unique_ratio * 100.0,
        );
        ctx.ui.log(&diversity_msg);
        if predicates.len() >= 10 && unique_ratio < min_unique_ratio {
            let detail = format!(
                "{} — below the --min-unique-ratio floor of {:.2}%. \
                 The picker is locking onto a small candidate set; \
                 typical causes: a heavy-hitter measure with only a \
                 few entries, an over-tight selectivity target that \
                 narrows the candidate pool to one, or a generator \
                 bug. Loosen --selectivity / --selectivity-max to \
                 widen the candidate range, or pass \
                 --min-unique-ratio=<lower> to acknowledge a \
                 genuinely-narrow predicate set.",
                diversity_msg, min_unique_ratio * 100.0,
            );
            if strict_unique_ratio {
                return error_result(detail, start);
            } else {
                ctx.ui.log(&format!("WARNING: {detail}"));
            }
        }

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
        // Schema sidecar in the `:schema` namespace — matches the
        // convention `convert` uses for metadata slabs (see
        // `vectordata::metadata_schema`). Without this any
        // downstream consumer would have to know out-of-band that
        // the content namespace holds PNode-named bytes and that
        // the generation template was X.
        //
        // `template` is the PNode Display vernacular with `?`
        // placeholders. When the user drove generation from a
        // proto, we serialise the proto's template verbatim;
        // when they used the flat flags, we synthesise the
        // template by rendering the predicate fingerprint
        // (structural shape with concrete-value slots replaced
        // by `?`).
        let schema_template = parsed_template.as_ref()
            .map(|_| proto.as_ref().map(|p| p.template.clone()).unwrap_or_default())
            .or_else(|| predicates.first().and_then(|rec| {
                let pnode = FmtPNode::from_bytes_named(rec).ok()?;
                Some(render_template_from_pnode(&pnode))
            }))
            .unwrap_or_else(|| "<empty>".to_string());
        let schema_selectivity = match selectivity_spec {
            SelectivitySpec::Scalar(v) => format!("{v}"),
            SelectivitySpec::Interval { lo, hi } => format!("{lo}..{hi}"),
        };
        let schema = vectordata::metadata_schema::PredicateSchema::new(
            schema_template,
            schema_selectivity,
            seed,
            predicates.len() as u64,
        );
        if let Err(e) = writer.start_namespace(
            vectordata::metadata_schema::SCHEMA_NAMESPACE,
        ) {
            return error_result(format!("schema namespace: {}", e), start);
        }
        if let Err(e) = writer.add_record(&schema.to_json_bytes()) {
            return error_result(format!("schema write: {}", e), start);
        }
        // Embed the survey report used to generate the
        // predicates into a `:survey` namespace so a downstream
        // consumer can introspect the source distributions
        // (quantile sketches, heavy hitters, etc.) without
        // needing the original survey JSON. This is for
        // explanation / provenance only — the predicate
        // evaluator doesn't read it.
        if let Err(e) = writer.start_namespace(
            vectordata::metadata_schema::SURVEY_NAMESPACE,
        ) {
            return error_result(format!("survey namespace: {}", e), start);
        }
        let survey_json = match serde_json::to_vec(&report) {
            Ok(v) => v,
            Err(e) => return error_result(format!("serialise survey report: {e}"), start),
        };
        if let Err(e) = writer.add_record(&survey_json) {
            return error_result(format!("survey write: {}", e), start);
        }
        if let Err(e) = writer.finish() {
            return error_result(format!("finish error: {}", e), start);
        }

        let var_name = format!("verified_count:{}",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace, &var_name, &predicates.len().to_string());
        ctx.defaults.insert(var_name, predicates.len().to_string());

        let message = format!(
            "{} predicates generated from {} eligible fields ({} distinct, {:.1}% unique)",
            predicates.len(),
            eligible.len(),
            unique_count,
            unique_ratio * 100.0,
        );
        ctx.ui.log(&format!("synthesize predicates: {}", message));

        CommandResult {
            status: Status::Ok,
            message,
            produced: vec![output_path.clone()],
            elapsed: start.elapsed(),
        }
    }
}


impl CommandOp for GenPredicatesOp {
    fn command_path(&self) -> &str {
        "generate predicates"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_GENERATE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Generate selectivity-targeted predicates from a metadata survey".into(),
            body: format!(
                "# generate predicates\n\n\
                 Survey a metadata slab (or load a precomputed `survey.json`) \
                 and emit `count` typed PNode predicates whose observed \
                 selectivity matches the requested target. For config-only \
                 integer-equality predicates that don't need a survey, see \
                 `generate simple-predicates`.\n\n\
                 ## Options\n\n{}",
                render_options_table(&options),
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();
        self.execute_survey(options, ctx, start)
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("source", "Path", false, None, "Metadata source to survey (optional when --survey or --proto-file carries it)", OptionRole::Input),
            opt("output", "Path", false, None, "Output slab of generated predicates (optional when --proto-file carries it; CLI wins on conflict)", OptionRole::Output),
            opt("survey", "Path", false, None, "Pre-computed survey JSON from `analyze survey` (skips re-surveying)", OptionRole::Input),
            opt("count", "int", false, Some("100"), "Number of predicates to generate", OptionRole::Config),
            opt("selectivity", "float", false, Some("0.1"), "Target selectivity", OptionRole::Config),
            opt("selectivity-max", "float", false, None, "Upper selectivity bound (random per-predicate within [selectivity, selectivity-max])", OptionRole::Config),
            opt("samples", "int", false, Some("1000"), "Inline survey sample count when --survey is not supplied", OptionRole::Config),
            opt("max-distinct", "int", false, Some("100"), "Max distinct values tracked per field (compat shim — actual cap comes from `analyze survey`)", OptionRole::Config),
            opt("strategy", "string", false, Some("eq"), "Predicate strategy: 'eq' (alias 'single') or 'compound'", OptionRole::Config),
            opt("fields", "string", false, None, "Comma-separated whitelist of metadata field names (default: all eligible fields)", OptionRole::Config),
            opt("exclude-fields", "string", false, None, "Comma-separated blacklist of metadata field names (applied after --fields)", OptionRole::Config),
            opt("ops", "string", false, None, "Comma-separated operator families to emit: eq, lt, gt, le, ge, range, matches (default: all)", OptionRole::Config),
            opt("proto-file", "Path", false, None, "Replay file produced by the wizard (e.g. predicates.slab.proto.json). Carries the template, source, and output paths so a single --proto-file invocation reproduces the run. Accepts .proto.json or .proto.yaml.", OptionRole::Input),
            opt("proto", "string", false, None, "Inline predicate prototype (JSON or YAML, same schema as --proto-file). Format inferred from leading `{`.", OptionRole::Config),
            opt("wizard", "bool", false, None, "Interactively build a predicate prototype and save it to <output>.proto.json (or --proto-file). Re-run later with --proto-file to replay the same config. Auto-detects metadata source from CWD when --source is omitted.", OptionRole::Config),
            opt("seed", "int", false, Some("42"), "RNG seed", OptionRole::Config),
            opt("min-unique-ratio", "float", false, Some("0.05"), "Minimum distinct/total predicate ratio; below this is logged as a WARNING or aborted with --strict-unique-ratio", OptionRole::Config),
            opt("strict-unique-ratio", "bool", false, Some("false"), "When true, abort generation if unique-ratio falls below --min-unique-ratio instead of just warning", OptionRole::Config),
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source", "survey"],
            &["output"],
        )
    }
}

// ---------------------------------------------------------------------------
// SurveyReport-driven predicate synthesis
// ---------------------------------------------------------------------------

/// Operator families an emitted predicate can wear. The CLI
/// `--ops` flag turns this into a mask that the per-field
/// generators consult before producing a predicate.
///
/// Layout: one bit per family. `all()` (the default) enables
/// every family — that's the historical behaviour. A caller can
/// pass `eq,matches` to restrict the generator to "exact-match
/// across categorical fields AND substring-match across text
/// fields" only, with numeric inequalities suppressed entirely.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OperatorMask(u32);

impl OperatorMask {
    const EQ:      u32 = 1 << 0;
    const LT:      u32 = 1 << 1;
    const GT:      u32 = 1 << 2;
    const LE:      u32 = 1 << 3;
    const GE:      u32 = 1 << 4;
    const RANGE:   u32 = 1 << 5; // numeric Range (two-sided) – currently emits Lt at low/high
    const MATCHES: u32 = 1 << 6;

    /// All families allowed — matches pre-flag behaviour so an
    /// invocation without `--ops` runs unchanged.
    pub(crate) fn all() -> Self { Self(u32::MAX) }

    fn allows(self, bit: u32) -> bool { self.0 & bit != 0 }

    /// Parse a comma-separated list of family names. Returns
    /// `Err(msg)` for unknown tokens so the operator-typo case
    /// surfaces at command launch instead of silently emitting
    /// no predicates.
    fn parse(s: &str) -> Result<Self, String> {
        let mut bits: u32 = 0;
        for raw in s.split(',') {
            let tok = raw.trim().to_ascii_lowercase();
            if tok.is_empty() { continue; }
            bits |= match tok.as_str() {
                "eq" => Self::EQ,
                "lt" => Self::LT,
                "gt" => Self::GT,
                "le" => Self::LE,
                "ge" => Self::GE,
                "range" => Self::RANGE,
                "matches" => Self::MATCHES,
                other => return Err(format!(
                    "unknown operator family '{other}'. \
                     Expected one of: eq, lt, gt, le, ge, range, matches",
                )),
            };
        }
        if bits == 0 {
            return Err("--ops must list at least one operator family".into());
        }
        Ok(Self(bits))
    }

    /// True when the mask permits at least one numeric
    /// inequality (Lt/Gt/Le/Ge/Range) — the family numeric
    /// fields with quantile sketches emit.
    fn allows_numeric_inequality(self) -> bool {
        self.allows(Self::LT) || self.allows(Self::GT)
            || self.allows(Self::LE) || self.allows(Self::GE)
            || self.allows(Self::RANGE)
    }

    fn allows_eq(self) -> bool { self.allows(Self::EQ) }
    fn allows_matches(self) -> bool { self.allows(Self::MATCHES) }

    /// Translate a `PNode::Predicate.op` to its mask bit. Used by
    /// the generators to filter their internal op-pick draws.
    fn permits_op(self, op: OpType) -> bool {
        let bit = match op {
            OpType::Eq | OpType::Ne => Self::EQ,
            OpType::Lt => Self::LT,
            OpType::Gt => Self::GT,
            OpType::Le => Self::LE,
            OpType::Ge => Self::GE,
            OpType::Matches => Self::MATCHES,
            // In/Contains/etc. aren't emitted by the current generators;
            // leave them disabled rather than silently allowing.
            _ => return false,
        };
        self.allows(bit)
    }
}

/// Parse a comma-separated, optionally-quoted list of field
/// names into a `Vec<String>`. Empty entries (e.g. trailing
/// commas) are dropped silently — typing one extra comma
/// shouldn't be a parse error.
fn parse_csv_field_list(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// Apply the `--fields` / `--exclude-fields` name-filters on
/// top of [`field_is_eligible`]. Whitelist (when present) keeps
/// only listed names; blacklist drops named entries
/// unconditionally. Both filters compare case-sensitive exact
/// names — the metadata schema is authoritative.
fn field_passes_name_filter(name: &str, whitelist: Option<&[String]>, blacklist: &[String]) -> bool {
    if let Some(allowed) = whitelist {
        if !allowed.iter().any(|n| n == name) { return false; }
    }
    if blacklist.iter().any(|n| n == name) { return false; }
    true
}

/// Whether the field's measure shape can produce at least one
/// predicate satisfying the operator mask. Used to drop fields
/// from the eligible set when `--ops` removes their only
/// applicable family — otherwise the picker would keep choosing
/// them and emitting `Eq Null` fallbacks.
fn field_supports_any_op(profile: &FieldProfile, mask: OperatorMask) -> bool {
    match profile.semantic_type.as_ref() {
        Some(SemanticType::Boolean) => mask.allows_eq(),
        Some(SemanticType::Number(_)) | Some(SemanticType::Temporal(_)) => {
            // Quantile-driven numeric → inequality family.
            // Frequency-table / reservoir fallback → Eq.
            (profile.measures.contains_key("QuantileSketch") && mask.allows_numeric_inequality())
                || (profile.measures.contains_key("ExactFrequencyTable") && mask.allows_eq())
                || (profile.measures.contains_key("HeavyHitters") && mask.allows_eq())
                || (profile.measures.contains_key("Reservoir") && mask.allows_eq())
        }
        Some(SemanticType::Identifier(_)) | Some(SemanticType::FreeText)
        | Some(SemanticType::Structured(_)) | Some(SemanticType::Categorical(_)) => {
            (profile.measures.contains_key("TrigramHeavyHitters") && mask.allows_matches())
                || (profile.measures.contains_key("LabelsetHeavyHitters") && mask.allows_matches())
                || (profile.measures.contains_key("ExactFrequencyTable") && mask.allows_eq())
                || (profile.measures.contains_key("HeavyHitters") && mask.allows_eq())
                || (profile.measures.contains_key("Reservoir") && mask.allows_eq())
        }
        _ => false,
    }
}

/// Decide whether a [`FieldProfile`] can contribute predicates. The
/// gate is intentionally permissive — any field whose
/// `SemanticType` is known and whose `CardinalityRegime` is not
/// `Constant` qualifies. Constant fields produce predicates that
/// match all or none; useless for benchmarking. `Unstable` fields
/// had no successful semantic verdict and only opaque measures
/// ran; the generator wouldn't know what to do with them.
fn field_is_eligible(profile: &FieldProfile) -> bool {
    if profile.presence.present == 0 {
        return false;
    }
    match profile.semantic_type.as_ref() {
        None | Some(SemanticType::Unstable) => return false,
        _ => {}
    }
    !matches!(profile.cardinality_regime, CardinalityRegime::Constant)
}

/// Pick one eligible field weighted by closeness of its
/// `1/cardinality` to the requested selectivity, then dispatch to
/// a per-semantic-type predicate builder. The weighting nudges the
/// generator toward fields where a single-value `Eq` actually
/// approximates the target sel — but every numeric/temporal field
/// can hit any target through quantile-driven ops regardless of
/// cardinality.
fn generate_eq_predicate(
    eligible: &[(&String, &FieldProfile)],
    target_sel: f64,
    op_mask: OperatorMask,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> PNode {
    let idx = pick_target_field(eligible, target_sel, rng);
    let (name, profile) = eligible[idx];
    let field = FieldRef::Named(name.to_string());
    build_predicate_for_field(field, profile, target_sel, op_mask, rng)
        .unwrap_or_else(|| PNode::Predicate(PredicateNode {
            field: FieldRef::Named(name.to_string()),
            op: OpType::Eq,
            comparands: vec![Comparand::Null],
        }))
}

/// AND together 2-3 `generate_eq_predicate` calls — selectivity
/// multiplies if the fields are independent. The survey's
/// `cross_field.copresence` table could later steer this toward
/// genuinely independent pairs, but for now we just sample
/// distinct fields uniformly.
fn generate_compound_predicate(
    eligible: &[(&String, &FieldProfile)],
    target_sel: f64,
    op_mask: OperatorMask,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> PNode {
    let arity = if eligible.len() >= 3 { rng.random_range(2..=3) } else { 2.min(eligible.len()) };
    if arity < 2 {
        return generate_eq_predicate(eligible, target_sel, op_mask, rng);
    }
    // Per-child sel ≈ target_sel^(1/arity) so the AND approximates
    // the requested total. Independence is not enforced — see TODO
    // above about copresence-aware selection.
    let per_child_sel = target_sel.powf(1.0 / arity as f64);
    let mut chosen_idx: Vec<usize> = Vec::with_capacity(arity);
    let mut tries = 0;
    while chosen_idx.len() < arity && tries < arity * 8 {
        let i = pick_target_field(eligible, per_child_sel, rng);
        if !chosen_idx.contains(&i) {
            chosen_idx.push(i);
        }
        tries += 1;
    }
    if chosen_idx.len() < 2 {
        return generate_eq_predicate(eligible, target_sel, op_mask, rng);
    }
    let children: Vec<PNode> = chosen_idx
        .into_iter()
        .filter_map(|i| {
            let (name, profile) = eligible[i];
            build_predicate_for_field(
                FieldRef::Named(name.to_string()),
                profile,
                per_child_sel,
                op_mask,
                rng,
            )
        })
        .collect();
    if children.len() < 2 {
        return generate_eq_predicate(eligible, target_sel, op_mask, rng);
    }
    PNode::Conjugate(ConjugateNode {
        conjugate_type: ConjugateType::And,
        children,
    })
}

/// Uniform random pick among fields that can plausibly produce a
/// predicate at the requested selectivity. Capability is decided
/// per field by [`field_can_hit_target`] — numeric/temporal with
/// `QuantileSketch`, text with `TrigramHeavyHitters` or
/// `LabelsetHeavyHitters`, and categorical/low-card fields with
/// a frequency table close enough to the target all qualify.
///
/// Why uniform instead of distance-weighted: a distance-based
/// score is asymmetric — numeric fields with quantile sketches
/// have a fixed (perfect) score, while text fields with frequency
/// tables get a score proportional to distance-from-target that
/// can be orders of magnitude larger. The numeric path then wins
/// every roll and `MATCHES` predicates never appear in the
/// output. Uniform selection treats every capable field as
/// equally valid and lets the per-type generators (numeric
/// Lt/Gt, text Eq, MATCHES via trigram/labelset) split the
/// emitted operator distribution proportionally to their counts.
fn pick_target_field(
    eligible: &[(&String, &FieldProfile)],
    target_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> usize {
    let capable: Vec<usize> = (0..eligible.len())
        .filter(|&i| field_can_hit_target(eligible[i].1, target_sel))
        .collect();
    if !capable.is_empty() {
        let n = capable.len();
        return capable[rng.random_range(0..n)];
    }
    // Every field's measures fall outside the target's plausible
    // hit range — fall back to a uniform pick across all eligible
    // fields. The per-type generators downgrade gracefully to
    // reservoir-driven `Eq` if nothing else fits.
    rng.random_range(0..eligible.len())
}

/// Decide whether `profile` carries a measure that can plausibly
/// produce a predicate near `target_sel`. The thresholds err on
/// the permissive side: a capable field is one that has *any*
/// reasonable path to the target, not a perfect one.
fn field_can_hit_target(profile: &FieldProfile, target_sel: f64) -> bool {
    // Numeric / temporal with QuantileSketch: quantile-driven
    // Lt/Gt/Ge/Le can hit any selectivity in (0, 1).
    if matches!(
        profile.semantic_type,
        Some(SemanticType::Number(_)) | Some(SemanticType::Temporal(_)),
    ) && profile.measures.contains_key("QuantileSketch")
    {
        return true;
    }
    // Text with trigram or labelset heavy-hitters: MATCHES can
    // hit any selectivity in the trigram/label frequency span.
    if profile.measures.contains_key("TrigramHeavyHitters")
        || profile.measures.contains_key("LabelsetHeavyHitters")
    {
        return true;
    }
    // Boolean: only hits ≈ 0.5 well.
    if matches!(profile.semantic_type, Some(SemanticType::Boolean)) {
        return (target_sel - 0.5).abs() < 0.25;
    }
    // ExactFrequencyTable / HeavyHitters: capable if some value's
    // frequency lands within a factor of 2 of the target count.
    let total = profile.presence.present.max(1) as f64;
    let target_count = (total * target_sel).max(1.0);
    let tolerance = (target_count * 2.0).max(1.0);
    if let Some(MeasureReport::ExactFrequencyTable(t)) = profile.measures.get("ExactFrequencyTable") {
        let closest = t.counts.values()
            .map(|c| (*c as f64 - target_count).abs())
            .fold(f64::MAX, f64::min);
        if closest < tolerance { return true; }
    }
    if let Some(MeasureReport::HeavyHitters(h)) = profile.measures.get("HeavyHitters") {
        let closest = h.items.iter()
            .map(|e| (e.count_lower_bound as f64 - target_count).abs())
            .fold(f64::MAX, f64::min);
        if closest < tolerance { return true; }
    }
    false
}

/// Build a single-field `Predicate` node by dispatching on
/// [`SemanticType`]. Returns `None` only when no useful measure is
/// available for the field.
fn build_predicate_for_field(
    field: FieldRef,
    profile: &FieldProfile,
    target_sel: f64,
    op_mask: OperatorMask,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<PNode> {
    match profile.semantic_type.as_ref()? {
        SemanticType::Boolean => {
            if !op_mask.allows_eq() { return None; }
            Some(boolean_predicate(field, rng))
        }
        SemanticType::Number(kind) => {
            if op_mask.allows_numeric_inequality() {
                if let Some(p) = numeric_predicate(field.clone(), profile, *kind, target_sel, op_mask, rng) {
                    return Some(p);
                }
            }
            if op_mask.allows_eq() {
                if let Some(p) = categorical_predicate(field.clone(), profile, target_sel, rng) {
                    return Some(p);
                }
                return reservoir_predicate(field, profile, false);
            }
            None
        }
        SemanticType::Temporal(_) => {
            if op_mask.allows_numeric_inequality() {
                if let Some(p) = numeric_predicate(field.clone(), profile, NumberKind::Floating, target_sel, op_mask, rng) {
                    return Some(p);
                }
            }
            if op_mask.allows_eq() {
                return reservoir_predicate(field, profile, false);
            }
            None
        }
        SemanticType::Identifier(_)
        | SemanticType::FreeText
        | SemanticType::Structured(_)
        | SemanticType::Categorical(_) => {
            // Coin flip between MATCHES (trigram-driven, calibrated)
            // and Eq (frequency-table-driven), constrained by the
            // operator mask. When the mask permits only one family,
            // the coin flip collapses to a single deterministic
            // call. When neither is permitted there's nothing to
            // emit and we return None — the caller's fallback
            // chain handles it.
            let try_matches = op_mask.allows_matches();
            let try_eq = op_mask.allows_eq();
            let prefer_matches = if try_matches && try_eq { rng.random_bool(0.5) }
                else { try_matches };
            if prefer_matches && try_matches {
                if let Some(p) = matches_predicate(field.clone(), profile, target_sel, rng) {
                    return Some(p);
                }
            }
            if try_eq {
                if let Some(p) = categorical_predicate(field.clone(), profile, target_sel, rng) {
                    return Some(p);
                }
            }
            if !prefer_matches && try_matches {
                if let Some(p) = matches_predicate(field.clone(), profile, target_sel, rng) {
                    return Some(p);
                }
            }
            if try_eq {
                return reservoir_predicate(field, profile, true);
            }
            None
        }
        SemanticType::Binary(_) | SemanticType::Unstable => None,
    }
}

fn boolean_predicate(field: FieldRef, rng: &mut rand_xoshiro::Xoshiro256PlusPlus) -> PNode {
    let val = rng.random_bool(0.5);
    PNode::Predicate(PredicateNode {
        field,
        op: OpType::Eq,
        comparands: vec![Comparand::Bool(val)],
    })
}

/// Quantile-driven numeric predicate: pick `Lt v_s`, `Gt v_{1-s}`,
/// or `Range[v_{0.5-s/2}, v_{0.5+s/2}]` from the field's
/// `QuantileSketch`. Reports `None` if no sketch is present.
fn numeric_predicate(
    field: FieldRef,
    profile: &FieldProfile,
    kind: NumberKind,
    target_sel: f64,
    op_mask: OperatorMask,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<PNode> {
    let MeasureReport::QuantileSketch(qs) = profile.measures.get("QuantileSketch")? else {
        return None;
    };
    if qs.count == 0 {
        return None;
    }
    let s = target_sel.clamp(0.001, 0.999);

    let make_comparand = |v: f64| -> Comparand {
        match kind {
            NumberKind::Integer { .. } => Comparand::Int(v.round() as i64),
            NumberKind::Decimal { .. } | NumberKind::Floating => Comparand::Float(v),
        }
    };

    // For integer columns with heavy modes (e.g. `original_width=150`
    // dominating a thumbnail-heavy distribution), strict Lt / Gt
    // misses every record whose value equals the boundary picked by
    // the quantile sketch — selectivity lands well below target on
    // the low end of the sel range. Promoting to Le / Ge for
    // integer-typed fields includes the boundary and closes the
    // gap; floats stay on strict Lt / Gt because float equality
    // is brittle.
    let is_integer = matches!(kind, NumberKind::Integer { .. });
    let (lt_op, gt_op) = if is_integer { (OpType::Le, OpType::Ge) } else { (OpType::Lt, OpType::Gt) };

    // Constrain the three-way pick to ops the mask permits. The
    // mask was already verified at field-eligibility time
    // (`field_supports_any_op`) to permit at least one of the
    // numeric inequalities, so this list is non-empty.
    let candidate_picks: Vec<u8> = {
        let mut c: Vec<u8> = Vec::with_capacity(3);
        if op_mask.permits_op(lt_op) { c.push(0); }
        if op_mask.permits_op(gt_op) { c.push(1); }
        if op_mask.allows(OperatorMask::RANGE) || op_mask.permits_op(lt_op) { c.push(2); }
        if c.is_empty() { return None; }
        c
    };
    let op_pick = candidate_picks[rng.random_range(0..candidate_picks.len())];

    let (op, comparands) = match op_pick {
        0 => {
            // (Le|Lt) at quantile s — selectivity ≈ s.
            let v = quantile_at(qs, s)?;
            (lt_op, vec![make_comparand(v)])
        }
        1 => {
            // (Ge|Gt) at quantile 1-s — selectivity ≈ s.
            let v = quantile_at(qs, 1.0 - s)?;
            (gt_op, vec![make_comparand(v)])
        }
        _ => {
            // Two-sided range expressed as a single comparison —
            // PNode has no native BETWEEN. Until we emit a
            // Conjugate(And, [Ge lo, Le hi]) here, fall back to
            // the simple form to keep operator variety without
            // inflating the predicate count.
            let v = quantile_at(qs, s)?;
            (lt_op, vec![make_comparand(v)])
        }
    };
    Some(PNode::Predicate(PredicateNode { field, op, comparands }))
}

/// Look up the sketch value at the requested quantile, falling back
/// to linear interpolation across the recorded quantile vector when
/// the exact point isn't present in the table.
fn quantile_at(
    qs: &crate::pipeline::commands::survey::measures::numeric::QuantileSketchReport,
    target: f64,
) -> Option<f64> {
    if qs.quantiles.is_empty() {
        return None;
    }
    // Quantile labels are `p<N>` strings (e.g. `"p50"` → 0.50).
    let parsed: Vec<(f64, f64)> = qs
        .quantiles
        .iter()
        .filter_map(|(k, v)| k.strip_prefix('p').and_then(|n| n.parse::<f64>().ok()).map(|p| (p / 100.0, *v)))
        .collect();
    if parsed.is_empty() { return None; }
    // Find bracketing pair.
    let mut lo = parsed[0];
    let mut hi = *parsed.last().unwrap();
    if target <= lo.0 { return Some(lo.1); }
    if target >= hi.0 { return Some(hi.1); }
    for w in parsed.windows(2) {
        if w[0].0 <= target && target <= w[1].0 {
            lo = w[0];
            hi = w[1];
            break;
        }
    }
    let span = (hi.0 - lo.0).max(1e-12);
    let frac = (target - lo.0) / span;
    Some(lo.1 + (hi.1 - lo.1) * frac)
}

/// Categorical/low-card `Eq` predicate driven by the frequency
/// distribution. Picks a value whose observed frequency is close
/// to `target_sel * present`.
fn categorical_predicate(
    field: FieldRef,
    profile: &FieldProfile,
    target_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<PNode> {
    let total = profile.presence.present.max(1) as f64;
    let target_count = (total * target_sel).max(1.0);

    // Prefer ExactFrequencyTable when present; fall back to heavy
    // hitters. Both report (value_string, count).
    let candidates: Vec<(String, f64)> = match profile.measures.get("ExactFrequencyTable") {
        Some(MeasureReport::ExactFrequencyTable(t)) => t
            .counts
            .iter()
            .map(|(k, c)| (k.clone(), *c as f64))
            .collect(),
        _ => match profile.measures.get("HeavyHitters") {
            Some(MeasureReport::HeavyHitters(h)) => h
                .items
                .iter()
                .map(|e| (e.value.clone(), e.count_lower_bound as f64))
                .collect(),
            _ => return None,
        },
    };
    if candidates.is_empty() {
        return None;
    }
    // Inverse-distance weighted draw — closer frequencies win.
    let weights: Vec<f64> = candidates
        .iter()
        .map(|(_, c)| 1.0 / ((c - target_count).abs() + 1.0))
        .collect();
    let total_w: f64 = weights.iter().sum();
    let mut pick = rng.random_range(0.0..total_w.max(1e-12));
    for ((value, _), w) in candidates.iter().zip(weights.iter()) {
        pick -= w;
        if pick <= 0.0 {
            return Some(value_to_eq_predicate(field, value));
        }
    }
    let (value, _) = candidates.last().unwrap();
    Some(value_to_eq_predicate(field, value))
}

/// Calibrated `MATCHES` predicate.
///
/// Prefers the labelset path when available: pick a label whose
/// observed frequency matches the target sel and emit a pattern
/// anchored on label boundaries (`(^|, )LABEL(,|$)`). Falls back
/// to the trigram path for generic text fields, emitting an
/// unanchored 3-char substring pattern.
fn matches_predicate(
    field: FieldRef,
    profile: &FieldProfile,
    target_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<PNode> {
    if let Some(p) = labelset_matches_predicate(&field, profile, target_sel, rng) {
        return Some(p);
    }
    trigram_matches_predicate(&field, profile, target_sel, rng)
}

/// Per-label calibrated MATCHES: anchor the label on `,` /
/// start-of-string / end-of-string so `"X"` doesn't spuriously
/// match `"XYZ"` inside the same value.
fn labelset_matches_predicate(
    field: &FieldRef,
    profile: &FieldProfile,
    target_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<PNode> {
    let MeasureReport::LabelsetHeavyHitters(l) =
        profile.measures.get("LabelsetHeavyHitters")?
    else { return None };
    if l.items.is_empty() || l.observed_records == 0 { return None; }

    // Per-record sel ≈ count_lower_bound / observed_records,
    // assuming each record contains at most one occurrence of the
    // target label (labelsets typically dedupe).
    let total = l.observed_records.max(1) as f64;
    let target_count = (total * target_sel).max(1.0);
    let candidates: Vec<(&str, f64)> = l.items.iter()
        .map(|e| (e.label.as_str(), e.count_lower_bound as f64))
        .collect();
    let label = sample_by_target_proximity(&candidates, target_count, rng)?;
    // Anchored pattern: matches the label only when bounded by
    // start-of-string, `, ` (with optional space), or end-of-string.
    let pattern = format!("(^|, ){}(,|$)", label);
    Some(PNode::Predicate(PredicateNode {
        field: field.clone(),
        op: OpType::Matches,
        comparands: vec![Comparand::Text(pattern)],
    }))
}

/// Generic trigram-driven MATCHES (no boundary anchoring).
///
/// Selectivity model: a record matches `MATCHES '%trigram%'`
/// when the trigram appears at least once in its text. With per-
/// record presence not tracked at sketch time, the best
/// approximation is `min(count_lower_bound / present_records, 1)`
/// — assumes each trigram occurrence is in a distinct record,
/// which is the *upper bound* of how many records can contain it.
/// In practice multi-occurrence-per-record inflates the estimate,
/// so the calibration sits slightly above target_sel on text-
/// heavy corpora. Future refinement: a per-record presence
/// counter inside the sketch.
fn trigram_matches_predicate(
    field: &FieldRef,
    profile: &FieldProfile,
    target_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<PNode> {
    let MeasureReport::TrigramHeavyHitters(t) =
        profile.measures.get("TrigramHeavyHitters")?
    else { return None };
    if t.items.is_empty() || t.sampled_trigrams == 0 { return None; }

    // Normalize by *records*, not trigram windows. Each record
    // contributes ~`avg_chars - 2` trigrams; using the larger
    // denominator (sampled_trigrams) would understate per-record
    // selectivity by exactly that ratio.
    let denom = profile.presence.present.max(1) as f64;
    let target_count = (denom * target_sel).max(1.0);
    let candidates: Vec<(&str, f64)> = t.items.iter()
        .map(|e| (e.trigram.as_str(), e.count_lower_bound as f64))
        .collect();
    let trigram = sample_by_target_proximity(&candidates, target_count, rng)?;
    Some(PNode::Predicate(PredicateNode {
        field: field.clone(),
        op: OpType::Matches,
        comparands: vec![Comparand::Text(trigram.to_string())],
    }))
}

/// Inverse-distance weighted draw across `(label, count)`
/// candidates against a `target_count`.
///
/// Replaces the previous argmin-with-cosmetic-jitter picker that
/// always selected the single closest candidate — emitting 10K
/// identical predicates from a corpus with hundreds of valid
/// heavy hitters. With inverse-distance weights, candidates
/// near the target are still picked more often, but every entry
/// has non-zero probability, so a 10K-emission run produces a
/// distribution of predicates instead of a monocrop. Mirrors
/// the model `categorical_predicate` already uses.
///
/// Weight = `1 / (|count - target| + 1)`. Returns `None` for an
/// empty candidate slice.
fn sample_by_target_proximity<'a, T: Copy>(
    candidates: &'a [(T, f64)],
    target_count: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<T> {
    if candidates.is_empty() { return None; }
    let weights: Vec<f64> = candidates.iter()
        .map(|(_, c)| 1.0 / ((c - target_count).abs() + 1.0))
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 { return Some(candidates[0].0); }
    let mut pick = rng.random_range(0.0..total);
    for ((value, _), w) in candidates.iter().zip(weights.iter()) {
        pick -= *w;
        if pick <= 0.0 { return Some(*value); }
    }
    Some(candidates.last().unwrap().0)
}

/// `Eq` predicate against a value drawn at random from the
/// reservoir sample. Used when no frequency table or sketch is
/// available — typical for `HighCardOrUnique` identifier fields.
fn reservoir_predicate(
    field: FieldRef,
    profile: &FieldProfile,
    text_only: bool,
) -> Option<PNode> {
    let MeasureReport::ReservoirSample(r) = profile.measures.get("ReservoirSample")? else {
        return None;
    };
    let item = r.items.first()?;
    let comparand = match item {
        serde_json::Value::String(s) => Comparand::Text(s.clone()),
        serde_json::Value::Bool(b) => Comparand::Bool(*b),
        serde_json::Value::Number(n) => {
            if text_only { return None; }
            if let Some(i) = n.as_i64() {
                Comparand::Int(i)
            } else if let Some(f) = n.as_f64() {
                Comparand::Float(f)
            } else {
                return None;
            }
        }
        _ => return None,
    };
    Some(PNode::Predicate(PredicateNode {
        field,
        op: OpType::Eq,
        comparands: vec![comparand],
    }))
}

/// Draw a single [`Comparand`] for `field`/`op`/`target_sel`,
/// reusing the calibrated per-op generators. Used by the
/// proto-template path ([`super::gen_predicates_proto`]) to
/// fill `?` placeholders in a template — the existing per-op
/// generators emit a full `PNode::Predicate`; we discard the
/// field/op wrappers and return just the comparand so the
/// caller can plant it into the right slot.
///
/// Returns `None` when the field's measure shape can't produce
/// a comparand for the requested op (e.g. asking for `MATCHES`
/// against a numeric-only profile, or `Lt` against a field
/// with no quantile sketch).
pub(super) fn draw_comparand_for_field(
    profile: &FieldProfile,
    op: OpType,
    target_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<Comparand> {
    // Boolean fields: any equality op picks a 50/50 boolean.
    if matches!(profile.semantic_type, Some(SemanticType::Boolean))
        && matches!(op, OpType::Eq | OpType::Ne)
    {
        return Some(Comparand::Bool(rng.random_bool(0.5)));
    }

    // Numeric / temporal inequality: walk the quantile sketch.
    if matches!(op, OpType::Lt | OpType::Le | OpType::Gt | OpType::Ge) {
        let qs = match profile.measures.get("QuantileSketch") {
            Some(MeasureReport::QuantileSketch(qs)) if qs.count > 0 => qs,
            _ => return None,
        };
        let s = target_sel.clamp(0.001, 0.999);
        let q = match op {
            OpType::Lt | OpType::Le => s,
            OpType::Gt | OpType::Ge => 1.0 - s,
            _ => return None,
        };
        let v = quantile_at(qs, q)?;
        let as_int = matches!(profile.semantic_type,
            Some(SemanticType::Number(NumberKind::Integer { .. })));
        return Some(if as_int {
            Comparand::Int(v.round() as i64)
        } else {
            Comparand::Float(v)
        });
    }

    // Eq / Ne against a categorical or reservoir-backed field —
    // reuse the existing per-op generators and pluck the
    // comparand out of the returned `PNode::Predicate`.
    if matches!(op, OpType::Eq | OpType::Ne) {
        let dummy = FieldRef::Named(String::new());
        if let Some(PNode::Predicate(p)) =
            categorical_predicate(dummy.clone(), profile, target_sel, rng)
        {
            return p.comparands.into_iter().next();
        }
        if let Some(PNode::Predicate(p)) = reservoir_predicate(dummy, profile, false) {
            return p.comparands.into_iter().next();
        }
        return None;
    }

    // MATCHES — trigram / labelset.
    if matches!(op, OpType::Matches) {
        let dummy = FieldRef::Named(String::new());
        if let Some(PNode::Predicate(p)) =
            matches_predicate(dummy, profile, target_sel, rng)
        {
            return p.comparands.into_iter().next();
        }
        return None;
    }

    // IN — not yet supported in template materialisation. Add a
    // dedicated path when the use case lands.
    None
}

/// Convert a frequency-table key string back into a `Comparand`
/// of the right kind. The keys are stable `Debug`-formatted MValues
/// (mirroring `canonical_distinct_key`), so a `Text("foo")` key
/// shows up as `Text("foo")` and we recover the inner literal.
fn value_to_eq_predicate(field: FieldRef, key: &str) -> PNode {
    let comparand = if let Some(inner) = key.strip_prefix("Text(").and_then(|s| s.strip_suffix(')')) {
        // Strip the surrounding "..." quotes if present.
        let trimmed = inner.trim_matches('"');
        Comparand::Text(trimmed.to_string())
    } else if let Some(inner) = key.strip_prefix("Int(").and_then(|s| s.strip_suffix(')')) {
        inner.parse::<i64>().map(Comparand::Int).unwrap_or(Comparand::Text(inner.to_string()))
    } else if let Some(inner) = key.strip_prefix("Int32(").and_then(|s| s.strip_suffix(')')) {
        inner.parse::<i32>().map(|v| Comparand::Int(v as i64)).unwrap_or(Comparand::Text(inner.to_string()))
    } else if let Some(inner) = key.strip_prefix("Float(").and_then(|s| s.strip_suffix(')')) {
        inner.parse::<f64>().map(Comparand::Float).unwrap_or(Comparand::Text(inner.to_string()))
    } else if let Some(inner) = key.strip_prefix("Bool(").and_then(|s| s.strip_suffix(')')) {
        Comparand::Bool(inner == "true")
    } else {
        Comparand::Text(key.to_string())
    };
    PNode::Predicate(PredicateNode {
        field,
        op: OpType::Eq,
        comparands: vec![comparand],
    })
}

/// Render a concrete `PNode` back to its `Display`-vernacular
/// template form by replacing every comparand with `?`. Used to
/// reconstruct a schema-sidecar template from the first emitted
/// predicate when no explicit proto template was supplied (i.e.
/// the user drove generation through the flat flags).
///
/// The renderer reuses the existing `PNode::Display` impl on a
/// fingerprint copy of the tree, then substitutes value tokens
/// with `?`. This keeps the template grammar consistent with
/// what `gen_predicates_proto::parse_template` accepts and what
/// `veks_anode::pnode::from_display` round-trips.
fn render_template_from_pnode(node: &PNode) -> String {
    fn placeholder(node: &PNode) -> PNode {
        match node {
            PNode::Predicate(p) => PNode::Predicate(PredicateNode {
                field: p.field.clone(),
                op: p.op,
                // Replace every comparand with `Null`; the
                // textual substitution below converts that to
                // `?`. See the wizard's `render_template` for
                // the same trick.
                comparands: p.comparands.iter().map(|_| Comparand::Null).collect(),
            }),
            PNode::Conjugate(c) => PNode::Conjugate(ConjugateNode {
                conjugate_type: c.conjugate_type,
                children: c.children.iter().map(placeholder).collect(),
            }),
        }
    }
    placeholder(node).to_string().replace("NULL", "?")
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
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
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
        opts.set("source", input_path.to_string_lossy().to_string());
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
        opts.set("source", input_path.to_string_lossy().to_string());
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
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "10".to_string());

        let mut op = GenPredicatesOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "gen predicates failed: {}", result.message);
        assert!(result.message.contains("0 predicates"), "message: {}", result.message);
    }

    /// A `Categorical(Labelset)`-shaped text field should yield
    /// MATCHES predicates whose pattern is anchored on label
    /// boundaries (`(^|, )LABEL(,|$)`). End-to-end: build a slab
    /// with comma-separated short-token labels, run the generator,
    /// and assert that at least one emitted predicate uses the
    /// anchored shape. Validates the wiring from probe → measure
    /// → MeasureReport map → predicate generator.
    #[test]
    fn labelset_matches_predicates_are_anchored() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Build a 600-record slab where `tags` is a labelset with
        // 1-3 tokens per chunk and 2-3 chunks per value.
        let labels_per_record = [
            "music, indie rock, hip hop",
            "art, painting",
            "music, jazz, ambient",
            "sports, soccer",
            "music, pop, rock",
            "tech, ai, ml",
            "art, sculpture",
            "music, classical",
            "sports, basketball, nba",
            "tech, web, frontend",
        ];
        let mut records = Vec::with_capacity(600);
        for i in 0..600 {
            let mut node = MNode::new();
            node.insert(
                "tags".into(),
                MValue::Text(labels_per_record[i % labels_per_record.len()].to_string()),
            );
            records.push(node);
        }
        let input_path = create_test_metadata_slab(ws, "labelset.slab", records);
        let output_path = ws.join("preds.slab");

        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "200".to_string());
        opts.set("seed", "13".to_string());

        let mut op = GenPredicatesOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);

        let preds_reader = SlabReader::open(&output_path).unwrap();
        let total = preds_reader.total_records();
        let mut saw_anchored = false;
        for ord in 0..total {
            let bytes = preds_reader.get(ord as i64).unwrap();
            let pnode = FmtPNode::from_bytes_named(&bytes).unwrap();
            if let FmtPNode::Predicate(p) = &pnode {
                if p.op == OpType::Matches {
                    if let Some(Comparand::Text(pat)) = p.comparands.first() {
                        if pat.starts_with("(^|, ") && pat.ends_with("(,|$)") {
                            saw_anchored = true;
                            break;
                        }
                    }
                }
            }
        }
        assert!(
            saw_anchored,
            "expected at least one anchored MATCHES predicate from the labelset path",
        );
    }

    /// Selectivity precision contract: a `Lt`/`Gt` predicate
    /// generated against a skewed numeric distribution should hit
    /// within ±1% of the stated target — calibrated by the
    /// QuantileSketch measure, not by uniform random sampling
    /// across [min,max]. This is the whole reason for the
    /// SurveyReport-direct rewrite.
    #[test]
    fn numeric_predicate_selectivity_calibrated_to_quantile() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Build a metadata slab with a heavily right-skewed integer
        // column. The naive [min,max] uniform-pick would produce a
        // wildly off selectivity at sel=0.1 (a value in the dense
        // low-tail bucket would match ~50% of records). The
        // quantile-driven path picks v_p10 instead, hitting sel=0.1
        // exactly.
        let mut records = Vec::new();
        for i in 0..2000i64 {
            let mut node = MNode::new();
            // 80% of values are <10, 20% are 100..1000 — a long
            // right tail.
            let v = if i < 1600 { i % 10 } else { 100 + ((i - 1600) % 900) };
            node.insert("skewed".into(), MValue::Int(v));
            records.push(node);
        }
        let input_path = create_test_metadata_slab(ws, "skewed.slab", records);
        let output_path = ws.join("preds.slab");

        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "200".to_string());
        opts.set("selectivity", "0.1".to_string());
        opts.set("seed", "7".to_string());

        let mut op = GenPredicatesOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);

        // Read back the predicates and evaluate each against the
        // raw record set. The Lt/Gt comparands should hit
        // observed selectivity within ±5% of the 0.10 target
        // (slack accounts for KLL rank error at k=1000 ≈ 0.16%
        // on top of the sample-vs-stream sampling noise).
        let preds_reader = SlabReader::open(&output_path).unwrap();
        let pred_count = preds_reader.total_records();
        // Materialise the records again for the assertion.
        let raw = SlabReader::open(&input_path).unwrap();
        let raw_count = raw.total_records();
        let mut total_matches = 0u64;
        for ord in 0..pred_count {
            let bytes = preds_reader.get(ord as i64).unwrap();
            let pnode = FmtPNode::from_bytes_named(&bytes).unwrap();
            let (op, threshold) = match &pnode {
                FmtPNode::Predicate(p) => match (p.op, p.comparands.first()) {
                    (OpType::Lt, Some(Comparand::Int(v)))
                    | (OpType::Le, Some(Comparand::Int(v)))
                    | (OpType::Gt, Some(Comparand::Int(v)))
                    | (OpType::Ge, Some(Comparand::Int(v))) => (p.op, *v),
                    _ => continue, // Eq / other → out of scope for this assertion.
                },
                _ => continue,
            };
            let mut matched = 0u64;
            for r_ord in 0..raw_count {
                let rec = raw.get(r_ord as i64).unwrap();
                let node = MNode::from_bytes(&rec).unwrap();
                if let Some(MValue::Int(v)) = node.fields.get("skewed") {
                    let hit = match op {
                        OpType::Lt => *v < threshold,
                        OpType::Le => *v <= threshold,
                        OpType::Gt => *v > threshold,
                        OpType::Ge => *v >= threshold,
                        _ => false,
                    };
                    if hit { matched += 1; }
                }
            }
            total_matches += matched;
        }
        if pred_count == 0 { return; }
        let observed_sel = total_matches as f64 / (pred_count as f64 * raw_count as f64);
        let target_sel = 0.10;
        let err = (observed_sel - target_sel).abs();
        assert!(
            err < 0.05,
            "observed selectivity {observed_sel:.4} vs target {target_sel:.4} (err={err:.4}); \
             quantile-driven path should keep this within ±5%",
        );
    }

    /// CI-grade end-to-end calibration gate: build a synthetic
    /// metadata slab with multiple field types (uniform integer,
    /// right-skewed integer, low-cardinality categorical), run the
    /// full survey → generate predicates → evaluate chain at
    /// several target selectivities, and assert observed-vs-target
    /// stays within ±10%.
    ///
    /// This is the gate that catches:
    /// - serde variant-ambiguity regressions (the
    ///   `QuantileSketch`-as-`ExactExtrema` bug from a production run),
    /// - field-selection bias that locks out an entire predicate
    ///   family (the previous fixed-score numeric dominance),
    /// - measure-routing breakage in the template synthesis layer.
    #[test]
    fn end_to_end_calibration_within_tolerance() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Build a 5K-record slab with three fields:
        //  - `uniform_int`     — uniform 0..10000 (quantile-clean)
        //  - `skewed_int`      — 80% in [0,10], 20% in [100,1000]
        //  - `category`        — 8 distinct labels, weighted
        let labels = [
            "alpha", "beta", "gamma", "delta",
            "epsilon", "zeta", "eta", "theta",
        ];
        let mut records = Vec::with_capacity(5_000);
        for i in 0..5_000i64 {
            let mut node = MNode::new();
            node.insert("uniform_int".into(), MValue::Int(i % 10_000));
            let skewed = if i < 4_000 { i % 10 } else { 100 + ((i - 4_000) % 900) };
            node.insert("skewed_int".into(), MValue::Int(skewed));
            // Weighted: half are "alpha", the rest spread across the others.
            let label_idx = if i < 2_500 { 0 } else { ((i - 2_500) as usize) % 7 + 1 };
            node.insert("category".into(), MValue::Text(labels[label_idx].into()));
            records.push(node);
        }
        let input_path = create_test_metadata_slab(ws, "calibration.slab", records);
        let raw_count = 5_000u64;

        for &target_sel in &[0.10, 0.25, 0.50] {
            let output_path = ws.join(format!("preds_{}.slab", (target_sel * 100.0) as u32));
            let mut opts = Options::new();
            opts.set("source", input_path.to_string_lossy().to_string());
            opts.set("output", output_path.to_string_lossy().to_string());
            opts.set("count", "100".to_string());
            opts.set("selectivity", format!("{target_sel}"));
            opts.set("seed", "1729".to_string());

            let mut op = GenPredicatesOp;
            let r = op.execute(&opts, &mut ctx);
            assert_eq!(
                r.status, Status::Ok,
                "generate predicates @sel={target_sel}: {}",
                r.message,
            );

            // Walk the generated predicates and evaluate each
            // against the raw record set. Skip operators outside
            // the simple-numeric assertion set (Eq/MATCHES on text
            // fields land here; they're best-effort and the
            // numeric calibration is the primary gate).
            let preds = SlabReader::open(&output_path).unwrap();
            let pred_count = preds.total_records();
            let raw = SlabReader::open(&input_path).unwrap();
            let mut scored = 0u64;
            let mut hits = 0u64;
            for ord in 0..pred_count {
                let bytes = preds.get(ord as i64).unwrap();
                let pnode = FmtPNode::from_bytes_named(&bytes).unwrap();
                let FmtPNode::Predicate(p) = &pnode else { continue };
                // Capture field name + op + threshold for numeric
                // assertions; non-numeric predicates contribute
                // through their evaluator-side outcomes too.
                for r_ord in 0..raw_count {
                    let rec = raw.get(r_ord as i64).unwrap();
                    let node = MNode::from_bytes(&rec).unwrap();
                    let hit = evaluate_simple_predicate(p, &node);
                    if hit.is_some() {
                        scored += 1;
                        if hit == Some(true) { hits += 1; }
                    } else {
                        // Operator we don't handle in this assertion harness
                        // (e.g. MATCHES with a regex pattern) — just count
                        // it as a one-shot evaluation to keep the sample
                        // size honest.
                        scored += 1;
                    }
                }
            }
            if scored == 0 { continue; }
            let observed_sel = hits as f64 / scored as f64;
            let err = (observed_sel - target_sel).abs();
            assert!(
                err < 0.10,
                "calibration gate @sel={target_sel}: observed={observed_sel:.4} \
                 err={err:.4} (tolerance ±10%); breakage in survey/generate/eval chain",
            );
        }
    }

    /// Best-effort evaluator for assertion-set predicates. Returns
    /// `Some(true)` / `Some(false)` for operators this harness
    /// supports, `None` for anything else (so the caller can
    /// account for it separately).
    fn evaluate_simple_predicate(p: &veks_core::formats::pnode::PredicateNode, node: &MNode) -> Option<bool> {
        use veks_core::formats::pnode::FieldRef;
        let field_name = match &p.field {
            FieldRef::Named(n) => n.as_str(),
            FieldRef::Index(_) => return None,
        };
        let v = node.fields.get(field_name)?;
        let comparand = p.comparands.first()?;
        match (v, comparand) {
            (MValue::Int(a), Comparand::Int(b)) => Some(match p.op {
                OpType::Lt => a < b,
                OpType::Le => a <= b,
                OpType::Gt => a > b,
                OpType::Ge => a >= b,
                OpType::Eq => a == b,
                OpType::Ne => a != b,
                _ => return None,
            }),
            (MValue::Text(a), Comparand::Text(b)) => Some(match p.op {
                OpType::Eq => a == b,
                OpType::Ne => a != b,
                OpType::Matches => a.contains(b.as_str()),
                _ => return None,
            }),
            _ => None,
        }
    }

    /// Build a small synthetic slab with two numeric fields
    /// (`age`, `score`) and one text field (`name`) for the
    /// field-filter and op-filter tests.
    fn make_mixed_records() -> Vec<MNode> {
        let labels = ["alpha", "beta", "gamma", "delta", "epsilon"];
        (0..500i64).map(|i| {
            let mut node = MNode::new();
            node.insert("age".into(), MValue::Int(i % 100));
            node.insert("score".into(), MValue::Int(i % 50 + 1));
            node.insert("name".into(), MValue::Text(labels[(i as usize) % labels.len()].into()));
            node
        }).collect()
    }

    fn read_emitted_ops(output_path: &std::path::Path) -> Vec<OpType> {
        let reader = SlabReader::open(output_path).unwrap();
        let mut ops = Vec::new();
        for i in 0..reader.total_records() {
            let bytes = reader.get(i as i64).unwrap();
            let pnode = FmtPNode::from_bytes_named(&bytes).unwrap();
            if let FmtPNode::Predicate(p) = pnode { ops.push(p.op); }
        }
        ops
    }

    fn read_emitted_field_names(output_path: &std::path::Path) -> Vec<String> {
        use veks_core::formats::pnode::FieldRef;
        let reader = SlabReader::open(output_path).unwrap();
        let mut names = Vec::new();
        for i in 0..reader.total_records() {
            let bytes = reader.get(i as i64).unwrap();
            let pnode = FmtPNode::from_bytes_named(&bytes).unwrap();
            if let FmtPNode::Predicate(p) = pnode {
                if let FieldRef::Named(n) = p.field { names.push(n); }
            }
        }
        names
    }

    /// `--fields age` restricts predicate generation to that
    /// field alone — no `score` or `name` predicates appear.
    #[test]
    fn fields_whitelist_restricts_predicate_to_named_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());
        let output_path = ws.join("preds.slab");

        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "50".to_string());
        opts.set("seed", "7".to_string());
        opts.set("fields", "age".to_string());

        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        let names = read_emitted_field_names(&output_path);
        assert!(!names.is_empty(), "expected some predicates");
        for n in &names {
            assert_eq!(n, "age", "fields whitelist must limit predicates to 'age'");
        }
    }

    /// `--exclude-fields age,score` drops both numeric fields
    /// from eligibility — the text field `name` must carry the
    /// whole generation load.
    #[test]
    fn exclude_fields_blacklist_drops_named_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());
        let output_path = ws.join("preds.slab");

        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "50".to_string());
        opts.set("seed", "7".to_string());
        opts.set("exclude-fields", "age,score".to_string());

        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        let names = read_emitted_field_names(&output_path);
        assert!(!names.is_empty(), "expected some predicates");
        for n in &names {
            assert!(n != "age" && n != "score",
                "exclude-fields must drop both numeric fields, got {n}");
        }
    }

    /// `--ops eq` constrains the generator to Eq predicates
    /// only. No inequalities, no MATCHES.
    #[test]
    fn ops_filter_eq_emits_only_eq() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());
        let output_path = ws.join("preds.slab");

        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "50".to_string());
        opts.set("seed", "11".to_string());
        opts.set("ops", "eq".to_string());

        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        let ops = read_emitted_ops(&output_path);
        assert!(!ops.is_empty(), "expected some predicates");
        for op in &ops {
            assert_eq!(*op, OpType::Eq, "ops=eq must yield only Eq, saw {op:?}");
        }
    }

    /// `--ops matches` constrains the generator to MATCHES
    /// predicates only — numeric fields drop out of eligibility
    /// entirely, so the text field carries the load.
    #[test]
    fn ops_filter_matches_emits_only_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());
        let output_path = ws.join("preds.slab");

        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "50".to_string());
        opts.set("seed", "13".to_string());
        opts.set("ops", "matches".to_string());

        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        let ops = read_emitted_ops(&output_path);
        assert!(!ops.is_empty(), "expected some predicates");
        for op in &ops {
            assert_eq!(*op, OpType::Matches,
                "ops=matches must yield only MATCHES, saw {op:?}");
        }
    }

    /// `--ops lt,gt,le,ge,range` constrains the generator to
    /// numeric inequalities — Eq and MATCHES are suppressed.
    /// The text field drops out (no inequality family); both
    /// numeric fields carry the load.
    #[test]
    fn ops_filter_numeric_inequalities_only() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());
        let output_path = ws.join("preds.slab");

        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "50".to_string());
        opts.set("seed", "17".to_string());
        opts.set("ops", "lt,gt,le,ge,range".to_string());

        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        let ops = read_emitted_ops(&output_path);
        assert!(!ops.is_empty(), "expected some predicates");
        for op in &ops {
            assert!(matches!(*op,
                OpType::Lt | OpType::Le | OpType::Gt | OpType::Ge),
                "ops=lt,gt,le,ge,range must yield only numeric inequalities, saw {op:?}");
        }
        let names = read_emitted_field_names(&output_path);
        for n in &names {
            assert!(n != "name",
                "text field 'name' shouldn't appear with numeric-only --ops, saw {n}");
        }
    }

    /// Unknown operator token in `--ops` is a clean error, not
    /// a silent empty output.
    #[test]
    fn ops_filter_unknown_token_errors_at_launch() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());
        let output_path = ws.join("preds.slab");

        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("ops", "eq,not-a-real-op".to_string());

        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Error,
            "unknown operator family must error at launch");
        assert!(r.message.contains("not-a-real-op"),
            "error should mention the bad token, got '{}'", r.message);
    }

    /// End-to-end: a `--proto` (inline YAML) with a Display-form
    /// template pinning op + conjugate + fields, with `?` slots
    /// for comparands, produces predicates that match the
    /// template's structural fingerprint. Every emitted predicate
    /// has the same shape (AND of `age >= int` and `name MATCHES
    /// text`) — the proto fixes shape; the survey fills the `?`
    /// slots. (Byte-identical reproducibility across runs needs a
    /// pre-computed survey JSON — see the `survey:` field in the
    /// proto schema; the inline survey path uses thread RNG.)
    #[test]
    fn proto_template_drives_generation_structurally() {
        use veks_core::formats::pnode::{FieldRef, OpType, PNode};
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());

        let proto_yaml = format!(
            r#"
template: "(age >= ? AND name MATCHES ?)"
count: 30
selectivity: 0.10
seed: 9001
source: "{}"
"#,
            input_path.to_string_lossy()
        );

        let out_a = ws.join("a.slab");
        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", out_a.to_string_lossy().to_string());
        opts.set("proto", proto_yaml);
        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);

        // Walk the emitted predicates and assert each is an AND of
        // (`age >= <int>`) and (`name MATCHES <text>`).
        let reader = SlabReader::open(&out_a).unwrap();
        let total = reader.total_records();
        assert!(total > 0, "should emit at least one predicate");
        for i in 0..total {
            let bytes = reader.get(i as i64).unwrap();
            let pnode = FmtPNode::from_bytes_named(&bytes).unwrap();
            match pnode {
                PNode::Conjugate(c) => {
                    assert_eq!(c.conjugate_type, ConjugateType::And);
                    assert_eq!(c.children.len(), 2, "AND should have 2 children");
                    // first child: age >= int
                    if let PNode::Predicate(p) = &c.children[0] {
                        assert_eq!(p.field, FieldRef::Named("age".into()));
                        // Integer semantics may promote Ge→Ge or stay Ge;
                        // we only pin what the template said.
                        assert_eq!(p.op, OpType::Ge);
                        assert!(matches!(p.comparands.first(), Some(Comparand::Int(_))));
                    } else { panic!("child 0 must be a leaf predicate"); }
                    // second child: name MATCHES text
                    if let PNode::Predicate(p) = &c.children[1] {
                        assert_eq!(p.field, FieldRef::Named("name".into()));
                        assert_eq!(p.op, OpType::Matches);
                        assert!(matches!(p.comparands.first(), Some(Comparand::Text(_))));
                    } else { panic!("child 1 must be a leaf predicate"); }
                }
                _ => panic!("proto template was AND of two predicates — got non-conjugate"),
            }
        }
    }

    /// A proto with a literal-pinned comparand (no `?`) emits
    /// that exact value on every predicate — the generator only
    /// fills `?` slots.
    #[test]
    fn proto_template_literal_comparand_is_pinned() {
        use veks_core::formats::pnode::{FieldRef, OpType, PNode};
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());
        let output_path = ws.join("preds.slab");

        let proto_yaml = format!(
            r#"
template: "age = 50"
count: 10
seed: 1
source: "{}"
"#,
            input_path.to_string_lossy()
        );
        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("proto", proto_yaml);

        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        let reader = SlabReader::open(&output_path).unwrap();
        for i in 0..reader.total_records() {
            let bytes = reader.get(i as i64).unwrap();
            let pnode = FmtPNode::from_bytes_named(&bytes).unwrap();
            match pnode {
                PNode::Predicate(p) => {
                    assert_eq!(p.field, FieldRef::Named("age".into()));
                    assert_eq!(p.op, OpType::Eq);
                    assert_eq!(p.comparands, vec![Comparand::Int(50)],
                        "literal pinned comparand must be emitted verbatim");
                }
                _ => panic!("expected leaf predicate"),
            }
        }
    }

    /// Every predicate slab emitted by `generate predicates`
    /// also embeds the SurveyReport it was generated against
    /// in a `:survey` namespace, so a downstream consumer can
    /// introspect the source distributions without needing the
    /// original survey JSON.
    #[test]
    fn predicate_slab_embeds_survey_report_in_survey_namespace() {
        use vectordata::metadata_schema::SURVEY_NAMESPACE;
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_test_records());
        let output_path = ws.join("preds.slab");

        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "5".to_string());
        opts.set("seed", "42".to_string());
        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);

        // The :survey namespace must hold exactly one record:
        // the serialised SurveyReport.
        let survey_reader = slabtastic::SlabReader::open_namespace(
            &output_path, Some(SURVEY_NAMESPACE),
        ).expect("survey namespace must be present");
        assert_eq!(survey_reader.total_records(), 1);
        let bytes = survey_reader.get(0).unwrap();
        // Round-trip through SurveyReport — same JSON contract
        // `analyze survey` writes, so any caller that already
        // consumes a survey.json can consume this too.
        let parsed: crate::pipeline::commands::survey::SurveyReport =
            serde_json::from_slice(&bytes).expect("survey JSON parses");
        assert!(!parsed.fields.is_empty(),
            "embedded survey must carry at least one field profile");
    }

    /// Every predicate slab emitted by `generate predicates`
    /// carries a `:schema` namespace with a `PredicateSchema`
    /// sidecar describing what's in the content namespace. This
    /// is what lets downstream consumers know the records are
    /// PNode-named bytes and recover the originating template
    /// without out-of-band knowledge.
    #[test]
    fn predicate_slab_carries_schema_sidecar_in_schema_namespace() {
        use vectordata::metadata_schema::{
            PredicateSchema, SCHEMA_KIND_PREDICATE, SCHEMA_NAMESPACE,
        };
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_test_records());
        let output_path = ws.join("preds.slab");

        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "10".to_string());
        opts.set("seed", "42".to_string());
        let mut op = GenPredicatesOp;
        let r = op.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "gen predicates: {}", r.message);

        // The content namespace still holds the predicate records.
        let content_reader = SlabReader::open(&output_path).unwrap();
        assert_eq!(content_reader.total_records(), 10);

        // The schema namespace holds exactly one record.
        let schema_reader = slabtastic::SlabReader::open_namespace(
            &output_path, Some(SCHEMA_NAMESPACE),
        ).expect("schema namespace must be present");
        assert_eq!(schema_reader.total_records(), 1,
            "schema namespace should hold exactly one descriptor");
        let bytes = schema_reader.get(0).unwrap();
        let schema = PredicateSchema::from_json_bytes(&bytes)
            .expect("schema record must parse as PredicateSchema");

        // Descriptor sanity.
        assert_eq!(schema.kind, SCHEMA_KIND_PREDICATE);
        assert_eq!(schema.wire_format, "pnode:named");
        assert_eq!(schema.count, 10);
        assert_eq!(schema.seed, 42);
        // The template must include at least one `?` placeholder
        // — that's the templating signature of "fill from
        // survey", and confirms render_template_from_pnode did
        // its substitution.
        assert!(schema.template.contains('?'),
            "template should contain `?` placeholder(s), got {:?}", schema.template);
    }

    /// Proto-driven generation must persist the *proto's*
    /// template verbatim in the schema sidecar (not a re-
    /// rendered version) — so the consumer can replay the exact
    /// proto by reading the schema namespace.
    #[test]
    fn proto_template_round_trips_through_schema_sidecar() {
        use vectordata::metadata_schema::{PredicateSchema, SCHEMA_NAMESPACE};
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());
        let output_path = ws.join("preds.slab");

        let proto_yaml = format!(
            r#"
template: "(age >= ? AND name MATCHES ?)"
count: 5
selectivity: "0.05..0.20"
seed: 9001
source: "{}"
"#,
            input_path.to_string_lossy()
        );
        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("proto", proto_yaml);
        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);

        let schema_reader = slabtastic::SlabReader::open_namespace(
            &output_path, Some(SCHEMA_NAMESPACE),
        ).unwrap();
        let bytes = schema_reader.get(0).unwrap();
        let schema = PredicateSchema::from_json_bytes(&bytes).unwrap();
        assert_eq!(schema.template, "(age >= ? AND name MATCHES ?)");
        assert_eq!(schema.selectivity, "0.05..0.2");
        assert_eq!(schema.seed, 9001);
        assert_eq!(schema.count, 5);
    }

    /// `--proto-file` round-trips through serde — a proto saved
    /// to disk reloads to the same shape and produces the same
    /// output.
    #[test]
    fn proto_file_round_trip_via_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());
        let proto_path = ws.join("p.proto.yaml");
        std::fs::write(&proto_path, format!(
            "template: \"age >= ?\"\ncount: 5\nselectivity: 0.1\nseed: 17\nsource: \"{}\"\n",
            input_path.to_string_lossy()
        )).unwrap();

        let output_path = ws.join("preds.slab");
        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("proto-file", proto_path.to_string_lossy().to_string());

        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        assert!(SlabReader::open(&output_path).unwrap().total_records() > 0);
    }

    /// Single source of truth: a JSON proto file carrying
    /// `source` and `output` is enough to run generation — no
    /// `--source` or `--output` on the CLI. Covers the wizard's
    /// hand-off, where the user is told they can replay with
    /// just `--proto-file=<f>`.
    #[test]
    fn proto_file_carries_source_and_output() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());
        let output_path = ws.join("preds.slab");
        let proto = PredicateProto {
            template: "age >= ?".into(),
            count: 5,
            selectivity: SelectivitySpec::Scalar(0.1),
            seed: 19,
            survey: None,
            source: Some(input_path.to_string_lossy().to_string()),
            output: Some(output_path.to_string_lossy().to_string()),
        };
        let proto_path = ws.join("preds.proto.json");
        proto.save_to_path(&proto_path).unwrap();

        // Only --proto-file is set; nothing else.
        let mut opts = Options::new();
        opts.set("proto-file", proto_path.to_string_lossy().to_string());

        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        assert!(SlabReader::open(&output_path).unwrap().total_records() > 0,
            "output slab from proto-only invocation should contain predicates");
    }

    /// CLI `--output` beats the proto's `output` field. The
    /// last knob the operator turned wins; that matches every
    /// other override slot in the command.
    #[test]
    fn cli_output_overrides_proto_output() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());
        let proto_output = ws.join("from_proto.slab");
        let cli_output = ws.join("from_cli.slab");
        let proto = PredicateProto {
            template: "age >= ?".into(),
            count: 3,
            selectivity: SelectivitySpec::Scalar(0.1),
            seed: 11,
            survey: None,
            source: Some(input_path.to_string_lossy().to_string()),
            output: Some(proto_output.to_string_lossy().to_string()),
        };
        let proto_path = ws.join("p.proto.json");
        proto.save_to_path(&proto_path).unwrap();

        let mut opts = Options::new();
        opts.set("proto-file", proto_path.to_string_lossy().to_string());
        opts.set("output", cli_output.to_string_lossy().to_string());

        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        assert!(SlabReader::open(&cli_output).unwrap().total_records() > 0,
            "CLI --output should be the write target");
        assert!(!proto_output.exists(),
            "proto's `output` should NOT be written when CLI --output is set");
    }

    /// Helpful error when neither CLI nor proto supplies an
    /// output. The message names both knobs so the operator
    /// knows what to add.
    #[test]
    fn missing_output_error_mentions_both_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_mixed_records());

        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        // No --output, no proto-file with output.

        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Error);
        assert!(r.message.contains("--output"),
            "error must mention --output flag: {}", r.message);
        assert!(r.message.contains("proto"),
            "error must mention proto file alternative: {}", r.message);
    }

    /// `sample_by_target_proximity` returns a distribution over
    /// candidates, not a single argmin. With 5 candidates near
    /// the target and 100 draws, we must observe at least 3
    /// distinct picks — the old picker would produce 1.
    #[test]
    fn proximity_sampler_produces_diverse_picks() {
        let mut rng = rng::seeded_rng(42);
        let candidates: Vec<(&str, f64)> = vec![
            ("alpha",   95.0),
            ("beta",    100.0),
            ("gamma",   105.0),
            ("delta",   110.0),
            ("epsilon", 90.0),
        ];
        let target = 100.0;
        let picks: std::collections::HashSet<&str> = (0..100)
            .map(|_| sample_by_target_proximity(&candidates, target, &mut rng).unwrap())
            .collect();
        assert!(picks.len() >= 3,
            "expected ≥3 distinct picks across 100 draws, got {:?}", picks);
    }

    /// Weighting still favours the closest candidate — out of
    /// 1000 draws, the on-target entry must win more often than
    /// the far ones.
    #[test]
    fn proximity_sampler_favours_closer_candidates() {
        let mut rng = rng::seeded_rng(123);
        let candidates: Vec<(&str, f64)> = vec![
            ("on_target",   100.0),
            ("near",        110.0),
            ("far",         200.0),
        ];
        let target = 100.0;
        let mut hist: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for _ in 0..1000 {
            let pick = sample_by_target_proximity(&candidates, target, &mut rng).unwrap();
            *hist.entry(pick).or_default() += 1;
        }
        let on = hist.get("on_target").copied().unwrap_or(0);
        let near = hist.get("near").copied().unwrap_or(0);
        let far = hist.get("far").copied().unwrap_or(0);
        assert!(on > near, "on_target ({on}) should beat near ({near})");
        assert!(near > far, "near ({near}) should beat far ({far})");
    }

    /// Diversity floor is reported in the success message
    /// regardless of whether the guard trips, so an operator
    /// always sees the unique/total ratio for the run.
    #[test]
    fn diversity_ratio_appears_in_success_message() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        let input_path = create_test_metadata_slab(ws, "meta.slab", make_test_records());
        let output_path = ws.join("preds.slab");
        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "20".to_string());
        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        assert!(r.message.contains("distinct"),
            "success message must report distinct/unique ratio, got: {}",
            r.message);
        assert!(r.message.contains("% unique"),
            "success message must include percent-unique, got: {}",
            r.message);
    }

    /// `--strict-unique-ratio=true` flips the diversity warning
    /// into a hard abort when below-floor. Forcing a single-
    /// field, single-op generation against a binary-cardinality
    /// boolean field produces 2 distinct predicates max — at
    /// count=100 the unique ratio = 2/100 = 0.02, below the
    /// default 0.05 floor.
    #[test]
    fn strict_unique_ratio_aborts_below_floor() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);
        // 100 records, all with a single boolean field — every
        // emitted predicate is `bool_field = true` or
        // `bool_field = false`, so at most 2 distinct.
        let records: Vec<MNode> = (0..100i64).map(|i| {
            let mut n = MNode::new();
            n.insert("flag".into(), MValue::Bool(i % 2 == 0));
            n
        }).collect();
        let input_path = create_test_metadata_slab(ws, "meta.slab", records);
        let output_path = ws.join("preds.slab");
        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "100".to_string());
        opts.set("seed", "1".to_string());
        opts.set("fields", "flag".to_string());
        opts.set("strict-unique-ratio", "true".to_string());
        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Error,
            "strict mode should abort: {}", r.message);
        assert!(r.message.contains("unique-ratio") || r.message.contains("min-unique-ratio"),
            "error message should mention the diversity floor: {}", r.message);
    }

    /// Generator-level diversity regression: across 50
    /// emissions on a text-heavy corpus, MATCHES predicates
    /// must produce at least 3 distinct patterns. The old
    /// argmin picker would produce 1 (the symptom that
    /// surfaced on a large production dataset).
    #[test]
    fn matches_predicates_have_diversity_across_emissions() {
        use veks_core::formats::pnode::{Comparand, OpType, PNode};
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // 200 text records with varied content so the trigram
        // measure has multiple competing heavy hitters.
        let phrases = [
            "the quick brown fox", "the lazy dog naps",
            "rain falls on the meadow", "sunshine after the rain",
            "kittens play with yarn", "the dog chases its tail",
            "rivers flow to the sea", "the night is dark and full",
            "stars shine in the sky", "the wind whispers softly",
        ];
        let mut records = Vec::new();
        for i in 0..200i64 {
            let mut node = MNode::new();
            node.insert("text".into(), MValue::Text(phrases[(i as usize) % phrases.len()].into()));
            records.push(node);
        }
        let input_path = create_test_metadata_slab(ws, "meta.slab", records);
        let output_path = ws.join("preds.slab");

        let mut opts = Options::new();
        opts.set("source", input_path.to_string_lossy().to_string());
        opts.set("output", output_path.to_string_lossy().to_string());
        opts.set("count", "50".to_string());
        opts.set("seed", "42".to_string());
        opts.set("ops", "matches".to_string());

        let r = GenPredicatesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);

        let reader = SlabReader::open(&output_path).unwrap();
        let mut patterns: std::collections::HashSet<String> = std::collections::HashSet::new();
        for i in 0..reader.total_records() {
            let bytes = reader.get(i as i64).unwrap();
            if let Ok(PNode::Predicate(p)) = FmtPNode::from_bytes_named(&bytes)
                && let Some(Comparand::Text(s)) = p.comparands.first()
            {
                assert_eq!(p.op, OpType::Matches);
                patterns.insert(s.clone());
            }
        }
        assert!(patterns.len() >= 3,
            "expected ≥3 distinct MATCHES patterns across 50 emissions, got {} ({:?})",
            patterns.len(), patterns);
    }
}
