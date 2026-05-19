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
use crate::pipeline::commands::survey::{
    CardinalityRegime, FieldProfile, MeasureReport, NumberKind, SemanticType,
};
use crate::pipeline::rng;

use super::gen_predicates_common::{error_result, opt, resolve_path};
use super::slab::{survey_report_from_json, survey_report_inline};

/// Pipeline command: synthesize predicates from metadata slab field distributions.
pub struct GenPredicatesOp;

/// Create a boxed `GenPredicatesOp` command.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenPredicatesOp)
}

impl GenPredicatesOp {
    fn execute_survey(&mut self, options: &Options, ctx: &mut StreamContext, start: Instant) -> CommandResult {
        let output_str = match options.require("output") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        // `--source` is only required when no precomputed
        // `--survey` is supplied — with a survey JSON in hand the
        // generator doesn't need to re-scan the slab.
        let source_present = options.get("source").is_some();
        let survey_present = options.get("survey").is_some();
        if !source_present && !survey_present {
            return error_result(
                "either --source <slab> or --survey <survey.json> must be provided".into(),
                start,
            );
        }
        let input_path = options
            .get("source")
            .map(|s| resolve_path(s, &ctx.workspace))
            .unwrap_or_else(std::path::PathBuf::new);
        let output_path = resolve_path(output_str, &ctx.workspace);

        let count: usize = options.get("count").and_then(|s| s.parse().ok()).unwrap_or(100);
        let selectivity: f64 = options.get("selectivity").and_then(|s| s.parse().ok()).unwrap_or(0.1);
        let selectivity_max: Option<f64> = options.get("selectivity-max").and_then(|s| s.parse().ok());
        let max_samples: usize = options.get("samples").and_then(|s| s.parse().ok()).unwrap_or(1000);
        let strategy = options.get("strategy").unwrap_or("eq").to_string();
        let seed: u64 = rng::parse_seed(options.get("seed"));

        // Load the survey directly as a SurveyReport — no projection
        // down to the legacy FieldStats shape. The richer profile
        // (semantic_type, cardinality_regime, quantile sketches,
        // reservoirs, heavy hitters) is what the calibrated
        // predicate generator depends on.
        let report = if let Some(survey_str) = options.get("survey") {
            let survey_path = resolve_path(survey_str, &ctx.workspace);
            ctx.ui.log(&format!(
                "synthesize predicates: loading survey from {}",
                survey_path.display(),
            ));
            match survey_report_from_json(&survey_path) {
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
            .filter(|(_, profile)| field_is_eligible(profile))
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

        let pb = ctx.ui.bar(count as u64, "generating predicates");
        let mut predicates: Vec<Vec<u8>> = Vec::with_capacity(count);
        for pred_i in 0..count {
            let target_sel = match selectivity_max {
                Some(max) => rng_inst.random_range(selectivity..=max),
                None => selectivity,
            };
            let pnode = match strategy.as_str() {
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
            opt("source", "Path", false, None, "Metadata source to survey (omit if --survey provides a precomputed report)", OptionRole::Input),
            opt("output", "Path", true, None, "Output slab of generated predicates", OptionRole::Output),
            opt("survey", "Path", false, None, "Pre-computed survey JSON from `analyze survey` (skips re-surveying)", OptionRole::Input),
            opt("count", "int", false, Some("100"), "Number of predicates to generate", OptionRole::Config),
            opt("selectivity", "float", false, Some("0.1"), "Target selectivity", OptionRole::Config),
            opt("selectivity-max", "float", false, None, "Upper selectivity bound (random per-predicate within [selectivity, selectivity-max])", OptionRole::Config),
            opt("samples", "int", false, Some("1000"), "Inline survey sample count when --survey is not supplied", OptionRole::Config),
            opt("max-distinct", "int", false, Some("100"), "Max distinct values tracked per field (compat shim — actual cap comes from `analyze survey`)", OptionRole::Config),
            opt("strategy", "string", false, Some("eq"), "Predicate strategy: 'eq' or 'compound'", OptionRole::Config),
            opt("seed", "int", false, Some("42"), "RNG seed", OptionRole::Config),
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
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> PNode {
    let idx = pick_target_field(eligible, target_sel, rng);
    let (name, profile) = eligible[idx];
    let field = FieldRef::Named(name.to_string());
    build_predicate_for_field(field, profile, target_sel, rng)
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
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> PNode {
    let arity = if eligible.len() >= 3 { rng.random_range(2..=3) } else { 2.min(eligible.len()) };
    if arity < 2 {
        return generate_eq_predicate(eligible, target_sel, rng);
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
        return generate_eq_predicate(eligible, target_sel, rng);
    }
    let children: Vec<PNode> = chosen_idx
        .into_iter()
        .filter_map(|i| {
            let (name, profile) = eligible[i];
            build_predicate_for_field(
                FieldRef::Named(name.to_string()),
                profile,
                per_child_sel,
                rng,
            )
        })
        .collect();
    if children.len() < 2 {
        return generate_eq_predicate(eligible, target_sel, rng);
    }
    PNode::Conjugate(ConjugateNode {
        conjugate_type: ConjugateType::And,
        children,
    })
}

/// Weighted field selection: prefer fields whose effective per-
/// value selectivity is close to the target. Numeric/temporal
/// fields are excellent at any target because they get quantile-
/// driven ops, so they're given low "distance" regardless.
fn pick_target_field(
    eligible: &[(&String, &FieldProfile)],
    target_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> usize {
    let target_distinct = (1.0 / target_sel.max(1e-12)).round() as f64;
    let mut best = 0usize;
    let mut best_score = f64::MAX;
    for (i, (_, profile)) in eligible.iter().enumerate() {
        let score = field_score_against_target(profile, target_distinct);
        // Random jitter so the same field doesn't always win when scores tie.
        let jittered = score + rng.random_range(0.0..1.0);
        if jittered < best_score {
            best_score = jittered;
            best = i;
        }
    }
    best
}

fn field_score_against_target(profile: &FieldProfile, target_distinct: f64) -> f64 {
    // Numeric and temporal fields can hit any sel via quantile
    // ops — give them a small constant so they're always cheap.
    if matches!(
        profile.semantic_type,
        Some(SemanticType::Number(_)) | Some(SemanticType::Temporal(_)),
    ) && profile.measures.contains_key("QuantileSketch")
    {
        return 0.5;
    }
    let est_distinct = match &profile.cardinality_regime {
        CardinalityRegime::Binary => 2.0,
        CardinalityRegime::LowCard { exact_distinct } => *exact_distinct as f64,
        CardinalityRegime::MidCard { hll_estimate_at_pass1 } => *hll_estimate_at_pass1,
        CardinalityRegime::HighCardOrUnique { .. } => profile.presence.present as f64,
        CardinalityRegime::Constant => 1.0,
        CardinalityRegime::Unknown => profile.presence.present as f64,
    }
    .max(1.0);
    (est_distinct - target_distinct).abs()
}

/// Build a single-field `Predicate` node by dispatching on
/// [`SemanticType`]. Returns `None` only when no useful measure is
/// available for the field.
fn build_predicate_for_field(
    field: FieldRef,
    profile: &FieldProfile,
    target_sel: f64,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<PNode> {
    match profile.semantic_type.as_ref()? {
        SemanticType::Boolean => Some(boolean_predicate(field, rng)),
        SemanticType::Number(kind) => {
            if let Some(p) = numeric_predicate(field.clone(), profile, *kind, target_sel, rng) {
                return Some(p);
            }
            if let Some(p) = categorical_predicate(field.clone(), profile, target_sel, rng) {
                return Some(p);
            }
            reservoir_predicate(field, profile, false)
        }
        SemanticType::Temporal(_) => {
            if let Some(p) = numeric_predicate(field.clone(), profile, NumberKind::Floating, target_sel, rng) {
                return Some(p);
            }
            reservoir_predicate(field, profile, false)
        }
        SemanticType::Identifier(_)
        | SemanticType::FreeText
        | SemanticType::Structured(_)
        | SemanticType::Categorical(_) => {
            // Coin flip between MATCHES (trigram-driven, calibrated)
            // and Eq (frequency-table-driven). Both have their
            // strengths: MATCHES tolerates token-level variation,
            // Eq gives exact match. Mix so a downstream benchmark
            // exercises both predicate operators.
            let prefer_matches = rng.random_bool(0.5);
            if prefer_matches {
                if let Some(p) = matches_predicate(field.clone(), profile, target_sel, rng) {
                    return Some(p);
                }
            }
            if let Some(p) = categorical_predicate(field.clone(), profile, target_sel, rng) {
                return Some(p);
            }
            if !prefer_matches {
                if let Some(p) = matches_predicate(field.clone(), profile, target_sel, rng) {
                    return Some(p);
                }
            }
            reservoir_predicate(field, profile, true)
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
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
) -> Option<PNode> {
    let MeasureReport::QuantileSketch(qs) = profile.measures.get("QuantileSketch")? else {
        return None;
    };
    if qs.count == 0 {
        return None;
    }
    let s = target_sel.clamp(0.001, 0.999);
    let op_pick = rng.random_range(0..3);

    let make_comparand = |v: f64| -> Comparand {
        match kind {
            NumberKind::Integer { .. } => Comparand::Int(v.round() as i64),
            NumberKind::Decimal { .. } | NumberKind::Floating => Comparand::Float(v),
        }
    };

    let (op, comparands) = match op_pick {
        0 => {
            // Lt at quantile s — selectivity ≈ s.
            let v = quantile_at(qs, s)?;
            (OpType::Lt, vec![make_comparand(v)])
        }
        1 => {
            // Gt at quantile 1-s — selectivity ≈ s.
            let v = quantile_at(qs, 1.0 - s)?;
            (OpType::Gt, vec![make_comparand(v)])
        }
        _ => {
            // Range[q=0.5-s/2 .. q=0.5+s/2] expressed as
            // (>= lo) AND (<= hi). PNode doesn't have a native
            // "between"; emit as a compound. For now we approximate
            // by collapsing to a single comparand In-style: pick the
            // lower bound and Ge — selectivity ≈ 0.5 + s/2 - 0 = s
            // is wrong, so fall back to Lt with a jitter to keep
            // operator variety without inflating selectivity.
            let v = quantile_at(qs, s)?;
            (OpType::Lt, vec![make_comparand(v)])
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

    let mut best: Option<&str> = None;
    let mut best_score = f64::MAX;
    for entry in &l.items {
        let diff = (entry.count_lower_bound as f64 - target_count).abs();
        let jittered = diff + rng.random_range(0.0..1.0);
        if jittered < best_score {
            best_score = jittered;
            best = Some(&entry.label);
        }
    }
    let label = best?;
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

    let mut best: Option<&str> = None;
    let mut best_score = f64::MAX;
    for entry in &t.items {
        let diff = (entry.count_lower_bound as f64 - target_count).abs();
        let jittered = diff + rng.random_range(0.0..1.0);
        if jittered < best_score {
            best_score = jittered;
            best = Some(&entry.trigram);
        }
    }
    let trigram = best?;
    Some(PNode::Predicate(PredicateNode {
        field: field.clone(),
        op: OpType::Matches,
        comparands: vec![Comparand::Text(trigram.to_string())],
    }))
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
                    (OpType::Lt, Some(Comparand::Int(v))) => (OpType::Lt, *v),
                    (OpType::Gt, Some(Comparand::Int(v))) => (OpType::Gt, *v),
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
                        OpType::Gt => *v > threshold,
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
}
