// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Interactive wizard for predicate-generation proto construction.
//!
//! Standalone flow that walks the user through:
//! 1. loading a survey (from JSON or via inline scan of a slab),
//! 2. surveying the available metadata fields with their semantic
//!    type, cardinality, and capable operator families,
//! 3. picking which fields participate,
//! 4. picking an operator per field,
//! 5. picking a conjugate form (single / AND / OR),
//! 6. picking target selectivity, count, and seed,
//! 7. rendering the resulting predicate template in `PNode::Display`
//!    vernacular, and
//! 8. saving a [`PredicateProto`] YAML the user (or a downstream
//!    pipeline step) can replay with `--proto-file`.
//!
//! Designed to be invoked two ways:
//!   - **Standalone**: `veks generate predicates --wizard …`. The
//!     wizard runs interactively, writes the proto, and (if the
//!     user chooses) immediately drives the generator.
//!   - **From the bootstrap wizard**: when the user selects R or F
//!     facets and no proto exists, the bootstrap can call
//!     [`run_wizard`] directly to build one.

use std::collections::HashSet;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use veks_core::formats::pnode::{
    Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode,
};

use crate::pipeline::commands::survey::{
    CardinalityRegime, FieldProfile, NumberKind, SemanticType, SurveyReport,
};

use super::gen_predicates_proto::{PredicateProto, SelectivitySpec};

/// Inputs to the wizard. All paths are optional; the wizard
/// auto-detects metadata sources in the CWD when `survey_path`
/// and `source_path` are both `None`, and prompts the user for
/// the output slab when `output_slab` is `None`. The only
/// guaranteed parameter is `output_proto`: every wizard run
/// produces exactly one proto file at that path.
#[derive(Debug, Clone)]
pub struct WizardInputs {
    pub survey_path: Option<PathBuf>,
    pub source_path: Option<PathBuf>,
    /// Number of inline samples when scanning a slab. Ignored
    /// when `survey_path` is set.
    pub samples: usize,
    /// Where to save the generated proto file. When `None`, the
    /// wizard derives a default of `<output_slab>.proto.json`
    /// so the proto and the output slab stay co-located.
    pub output_proto: Option<PathBuf>,
    /// Destination slab for generated predicates. When `None`,
    /// the wizard prompts (defaulting to `./predicates.slab`
    /// or — if the wizard auto-detected a source — a sibling
    /// of the source).
    pub output_slab: Option<PathBuf>,
    /// Directory to search when auto-detecting metadata sources.
    /// Defaults to the process CWD when unset.
    pub search_root: Option<PathBuf>,
}

/// Result of a wizard run.
///
/// The wizard ALWAYS saves the proto file (so the user can
/// re-run with `--proto-file=<path>` later, regardless of
/// whether they want to generate now). The `proceed` flag
/// records the user's answer to the final "generate
/// predicates with this config?" prompt — when false, the
/// caller should print a re-run hint and exit cleanly without
/// running generation.
#[derive(Debug, Clone)]
pub struct WizardOutput {
    /// Path the proto was written to (matches `inputs.output_proto`).
    pub proto_path: PathBuf,
    /// Destination slab the wizard resolved (either the user's
    /// explicit `--output` or the value the wizard prompted
    /// for). Captured here so the "skipped" branch can echo it
    /// back to the user, and so the caller doesn't need to
    /// re-read the proto file just to learn the output path.
    pub output_slab: PathBuf,
    /// True when the user answered Y to the final prompt and
    /// generation should proceed by re-loading the saved file.
    /// False means the user wants to stop after the save.
    pub proceed: bool,
}

/// Run the interactive wizard.
///
/// Reads from `stdin`, writes to `stdout`. The caller is
/// responsible for ensuring stdin is a TTY when invoking this
/// path — for non-interactive contexts, construct a
/// [`PredicateProto`] directly via its serde API.
pub fn run_wizard(mut inputs: WizardInputs) -> Result<WizardOutput, String> {
    // Step 0 — if no survey or source was given, auto-detect.
    // Mirrors the main bootstrap wizard's pattern: scan a
    // directory for slabs whose `:schema` namespace carries a
    // `kind: "metadata"` descriptor, then prompt the user to
    // pick when there's more than one match.
    if inputs.survey_path.is_none() && inputs.source_path.is_none() {
        let root = inputs.search_root.clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        match autodetect_metadata_source(&root)? {
            None => {
                return Err(format!(
                    "no metadata slab found under {} — pass --source <slab> or --survey <survey.json>",
                    root.display(),
                ));
            }
            Some(picked) => {
                println!("Auto-detected metadata source: {}", picked.display());
                inputs.source_path = Some(picked);
            }
        }
    }

    let report = load_survey(&inputs)?;
    let eligible = list_eligible_fields(&report);
    if eligible.is_empty() {
        return Err("survey has no eligible fields (every field is Constant/Unstable/no measures)".into());
    }

    println!();
    println!("=== predicate generation wizard ===");
    println!();
    print_field_table(&eligible);
    println!();

    let chosen_indices = prompt_field_selection(&eligible)?;
    if chosen_indices.is_empty() {
        return Err("no fields selected".into());
    }

    let mut leaves: Vec<(String, OpType)> = Vec::with_capacity(chosen_indices.len());
    for idx in &chosen_indices {
        let (name, profile) = &eligible[*idx];
        let capable = capable_op_families(profile);
        let op = prompt_op_for_field(name, profile, &capable)?;
        leaves.push((name.clone(), op));
    }

    let conjugate = prompt_conjugate(&leaves)?;
    let selectivity = prompt_selectivity()?;
    let count = prompt_count(report.source.total_records)?;
    let seed = prompt_seed()?;

    let pnode_template = build_template(&leaves, conjugate);
    let rendered = render_template(&pnode_template);

    println!();
    println!("Resulting template:");
    println!("  {}", rendered);
    println!();

    // Output slab — either accept the value the caller passed
    // (CLI `--output` is authoritative when given) or prompt.
    // The default suggestion sits next to the source slab so
    // the artefacts stay co-located.
    let output_slab = match inputs.output_slab.clone() {
        Some(p) => p,
        None => prompt_output_slab(inputs.source_path.as_deref())?,
    };

    // Derive the proto path AFTER the output slab is resolved,
    // so the proto lives next to its target slab instead of in
    // whatever directory the user happened to run from. An
    // explicit `--proto-file` from the CLI wins; otherwise we
    // append `.proto.json` to the output slab path.
    let proto_path = inputs.output_proto.clone().unwrap_or_else(|| {
        let mut p = output_slab.clone();
        let name = p.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "predicates".into());
        p.set_file_name(format!("{name}.proto.json"));
        p
    });

    let proto = PredicateProto {
        template: rendered.clone(),
        count,
        selectivity,
        seed,
        survey: inputs.survey_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string()),
        source: inputs.source_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string()),
        output: Some(output_slab.to_string_lossy().to_string()),
    };

    proto.save_to_path(&proto_path)?;
    println!("Saved proto: {}", proto_path.display());
    // Print the replay hint up front, regardless of the
    // answer to the next prompt. Operators routinely re-run
    // the same config later (different machine, different
    // run, CI replay), so the hint belongs on every wizard
    // exit, not just the "skipped" branch.
    println!("Replay this run any time with:");
    println!("  veks generate predicates --proto-file {}", proto_path.display());

    println!();
    let proceed = prompt_yes_no(
        "Generate predicates with this config now?", true,
    )?;
    if !proceed {
        println!();
        println!("Wizard complete — proto saved without running generation.");
    }

    Ok(WizardOutput {
        proto_path,
        output_slab,
        proceed,
    })
}

fn prompt_output_slab(source: Option<&std::path::Path>) -> Result<PathBuf, String> {
    let default = match source {
        Some(s) => {
            let parent = s.parent().unwrap_or_else(|| std::path::Path::new("."));
            parent.join("predicates.slab")
        }
        None => PathBuf::from("predicates.slab"),
    };
    let raw = prompt("Output predicates slab", &default.to_string_lossy())?;
    Ok(PathBuf::from(raw))
}

/// Walk `root` recursively (bounded depth) looking for `.slab`
/// files whose `:schema` namespace contains a metadata
/// descriptor. Returns the chosen path: unique match → that
/// path; multiple → user picks; none → `Ok(None)` so the caller
/// can produce a useful error with the search root in it.
///
/// The bounded recursion handles canonical dataset layouts where
/// the metadata slab lives under `profiles/<name>/metadata_content.slab`
/// — a shallow walk catches that without enumerating an entire
/// home directory if the user is in the wrong place.
fn autodetect_metadata_source(root: &std::path::Path) -> Result<Option<PathBuf>, String> {
    /// Cap on directory depth (root = depth 0). Canonical dataset
    /// layouts put metadata at depth 2 (`profiles/<name>/x.slab`),
    /// so depth 4 is comfortably more than enough while keeping
    /// the walk fast on misdirected invocations.
    const MAX_DEPTH: usize = 4;
    let mut candidates: Vec<PathBuf> = Vec::new();
    walk_for_metadata_slabs(root, 0, MAX_DEPTH, &mut candidates)?;
    candidates.sort();
    match candidates.len() {
        0 => Ok(None),
        1 => Ok(Some(candidates.pop().unwrap())),
        _ => {
            println!();
            println!("Multiple metadata slabs found under {}:", root.display());
            for (i, p) in candidates.iter().enumerate() {
                let shown = p.strip_prefix(root).unwrap_or(p);
                println!("  [{}] {}", i + 1, shown.display());
            }
            loop {
                let raw = prompt("Pick metadata source", "1")?;
                if let Ok(n) = raw.trim().parse::<usize>()
                    && n >= 1 && n <= candidates.len() {
                        return Ok(Some(candidates.remove(n - 1)));
                    }
                println!("  pick a number in 1..={}", candidates.len());
            }
        }
    }
}

/// Depth-bounded recursive walk that collects `.slab` files
/// whose schema namespace says they're metadata. Skips hidden
/// directories and known noise (target/, node_modules/, .git/,
/// caches) so an accidental run in a wrong place doesn't take
/// forever.
fn walk_for_metadata_slabs(
    dir: &std::path::Path,
    depth: usize,
    max_depth: usize,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()), // unreadable subdir → silently skip
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let ft = match entry.file_type() {
            Ok(f) => f,
            Err(_) => continue,
        };
        if ft.is_dir() {
            if depth >= max_depth { continue; }
            if name_str.starts_with('.') { continue; }
            if matches!(name_str.as_ref(),
                "target" | "node_modules" | ".git" | ".cache"
                | "__pycache__" | "build" | "dist") { continue; }
            walk_for_metadata_slabs(&path, depth + 1, max_depth, out)?;
        } else if ft.is_file() {
            if path.extension().and_then(|e| e.to_str()) != Some("slab") { continue; }
            if is_metadata_slab(&path) {
                out.push(path);
            }
        }
    }
    Ok(())
}

/// Classify `path` as a metadata source, predicate source, or
/// neither. We try two layers in order:
///
/// 1. **Schema sidecar** — open the `:schema` namespace and
///    inspect the descriptor's `kind`. This is the canonical
///    signal for slabs produced by recent pipeline runs.
/// 2. **Filename convention** — older slabs (pre-schema-sidecar)
///    don't carry a descriptor; we fall back to the same name
///    heuristic the bootstrap wizard's `detect_roles` uses
///    (token "metadata" present, "predicates"/"indices"/"results"/
///    "predkeys" absent).
///
/// Returning `false` here means the auto-detector silently skips
/// the file; that's correct for predicate slabs, vector slabs,
/// and anything else that isn't a metadata source.
fn is_metadata_slab(path: &std::path::Path) -> bool {
    matches!(classify_slab(path), SlabKind::Metadata)
}

enum SlabKind { Metadata, Predicate, Other }

fn classify_slab(path: &std::path::Path) -> SlabKind {
    use vectordata::metadata_schema::{
        SCHEMA_KIND_METADATA, SCHEMA_KIND_PREDICATE, SCHEMA_NAMESPACE,
    };
    // Layer 1: schema sidecar (authoritative when present).
    if let Ok(r) = slabtastic::SlabReader::open_namespace(path, Some(SCHEMA_NAMESPACE))
        && r.total_records() > 0
            && let Ok(bytes) = r.get(0)
                && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    // Default to metadata when the descriptor is
                    // present but pre-dates the `kind` discriminator.
                    let kind = v.get("kind").and_then(|k| k.as_str())
                        .unwrap_or(SCHEMA_KIND_METADATA);
                    return match kind {
                        SCHEMA_KIND_METADATA => SlabKind::Metadata,
                        SCHEMA_KIND_PREDICATE => SlabKind::Predicate,
                        _ => SlabKind::Other,
                    };
                }
    // Layer 2: filename convention. Mirrors the bootstrap wizard's
    // `detect_roles` for the MetadataContent / MetadataPredicates /
    // MetadataResults split.
    classify_by_filename(path)
}

fn classify_by_filename(path: &std::path::Path) -> SlabKind {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    let tokens: Vec<&str> = stem.split(['_', '-', '.'])
        .filter(|s| !s.is_empty())
        .collect();
    let has_token_or_substring = |k: &str| stem.contains(k);
    let has_predicates = has_token_or_substring("predicate") || has_token_or_substring("predkey");
    let has_indices = tokens.contains(&"indices")
        || has_token_or_substring("neighbors")
        || has_token_or_substring("groundtruth");
    let has_results = has_token_or_substring("result");
    let has_metadata = has_token_or_substring("metadata") || tokens.contains(&"content");
    if has_predicates { return SlabKind::Predicate; }
    if has_results || has_indices { return SlabKind::Other; }
    if has_metadata { return SlabKind::Metadata; }
    SlabKind::Other
}

fn prompt_yes_no(label: &str, default_yes: bool) -> Result<bool, String> {
    let default_str = if default_yes { "Y/n" } else { "y/N" };
    loop {
        let raw = prompt(label, default_str)?;
        let t = raw.trim().to_ascii_lowercase();
        if t.is_empty() || t == default_str.to_ascii_lowercase() {
            return Ok(default_yes);
        }
        match t.as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => println!("  please answer y or n"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Survey loading
// ─────────────────────────────────────────────────────────────────────────────

fn load_survey(inputs: &WizardInputs) -> Result<SurveyReport, String> {
    if let Some(path) = &inputs.survey_path {
        return super::slab::survey_report_from_json(path);
    }
    let source = inputs.source_path.as_ref()
        .ok_or_else(|| "wizard needs either --survey or --source".to_string())?;
    println!("Surveying {} (samples = {}) …", source.display(), inputs.samples);
    super::slab::survey_report_inline(source, inputs.samples, None)
}

// ─────────────────────────────────────────────────────────────────────────────
// Field eligibility + display
// ─────────────────────────────────────────────────────────────────────────────

fn list_eligible_fields(report: &SurveyReport) -> Vec<(String, &FieldProfile)> {
    let mut out = Vec::new();
    for (name, profile) in &report.fields {
        if profile.presence.present == 0 { continue; }
        match profile.semantic_type {
            None | Some(SemanticType::Unstable) => continue,
            _ => {}
        }
        if matches!(profile.cardinality_regime, CardinalityRegime::Constant) { continue; }
        out.push((name.clone(), profile));
    }
    out
}

/// The op families a given field can plausibly produce. Mirrors
/// `gen_predicates::field_supports_any_op` but returns the
/// explicit allowed-op list so the wizard can show it to the user.
fn capable_op_families(profile: &FieldProfile) -> Vec<OpType> {
    let mut ops = Vec::new();
    let has = |k: &str| profile.measures.contains_key(k);
    match profile.semantic_type.as_ref() {
        Some(SemanticType::Boolean) => {
            ops.push(OpType::Eq);
            ops.push(OpType::Ne);
        }
        Some(SemanticType::Number(_)) | Some(SemanticType::Temporal(_)) => {
            if has("QuantileSketch") {
                ops.extend([OpType::Lt, OpType::Le, OpType::Gt, OpType::Ge]);
            }
            if has("ExactFrequencyTable") || has("HeavyHitters") || has("ReservoirSample") {
                ops.push(OpType::Eq);
                ops.push(OpType::Ne);
            }
        }
        Some(SemanticType::Identifier(_)) | Some(SemanticType::FreeText)
        | Some(SemanticType::Structured(_)) | Some(SemanticType::Categorical(_)) => {
            if has("TrigramHeavyHitters") || has("LabelsetHeavyHitters") {
                ops.push(OpType::Matches);
            }
            if has("ExactFrequencyTable") || has("HeavyHitters") || has("ReservoirSample") {
                ops.push(OpType::Eq);
                ops.push(OpType::Ne);
            }
        }
        _ => {}
    }
    ops
}

fn print_field_table(eligible: &[(String, &FieldProfile)]) {
    let semantic = |p: &FieldProfile| -> String {
        match p.semantic_type.as_ref() {
            Some(SemanticType::Boolean) => "Boolean".into(),
            Some(SemanticType::Number(NumberKind::Integer { .. })) => "Integer".into(),
            Some(SemanticType::Number(_)) => "Number".into(),
            Some(SemanticType::Temporal(_)) => "Temporal".into(),
            Some(SemanticType::Identifier(_)) => "Identifier".into(),
            Some(SemanticType::FreeText) => "FreeText".into(),
            Some(SemanticType::Categorical(_)) => "Categorical".into(),
            Some(SemanticType::Structured(_)) => "Structured".into(),
            Some(SemanticType::Binary(_)) => "Binary".into(),
            Some(SemanticType::Unstable) | None => "?".into(),
        }
    };
    let cardinality = |p: &FieldProfile| -> String {
        match p.cardinality_regime {
            CardinalityRegime::Constant => "constant".into(),
            CardinalityRegime::Binary => "binary".into(),
            CardinalityRegime::LowCard { .. } => "low".into(),
            CardinalityRegime::MidCard { .. } => "mid".into(),
            CardinalityRegime::HighCardOrUnique { .. } => "high".into(),
            CardinalityRegime::Unknown => "?".into(),
        }
    };
    println!("  Available fields:");
    println!("    {:>3}  {:<28}  {:<12}  {:<10}  capable ops",
        "#", "field", "type", "cardinality");
    println!("    {:>3}  {:<28}  {:<12}  {:<10}  {}",
        "—", "—".repeat(28), "—".repeat(12), "—".repeat(10), "—".repeat(20));
    for (i, (name, profile)) in eligible.iter().enumerate() {
        let ops = capable_op_families(profile)
            .iter()
            .map(op_token)
            .collect::<Vec<_>>()
            .join(", ");
        println!("    {:>3}  {:<28}  {:<12}  {:<10}  {}",
            i + 1,
            truncate(name, 28),
            semantic(profile),
            cardinality(profile),
            ops);
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max { s.to_string() } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
    }
}

fn op_token(op: &OpType) -> String {
    match op {
        OpType::Eq => "eq",
        OpType::Ne => "ne",
        OpType::Lt => "lt",
        OpType::Le => "le",
        OpType::Gt => "gt",
        OpType::Ge => "ge",
        OpType::In => "in",
        OpType::Matches => "matches",
    }.into()
}

fn op_from_token(tok: &str) -> Option<OpType> {
    match tok.trim().to_ascii_lowercase().as_str() {
        "eq" => Some(OpType::Eq),
        "ne" => Some(OpType::Ne),
        "lt" => Some(OpType::Lt),
        "le" => Some(OpType::Le),
        "gt" => Some(OpType::Gt),
        "ge" => Some(OpType::Ge),
        "matches" => Some(OpType::Matches),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prompts
// ─────────────────────────────────────────────────────────────────────────────

fn prompt_field_selection(eligible: &[(String, &FieldProfile)]) -> Result<Vec<usize>, String> {
    loop {
        let raw = prompt("Select fields (numbers, comma-separated; or 'all')", "all")?;
        let raw = raw.trim();
        if raw.eq_ignore_ascii_case("all") {
            return Ok((0..eligible.len()).collect());
        }
        let mut chosen: Vec<usize> = Vec::new();
        let mut bad: Vec<String> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        for tok in raw.split(',') {
            let tok = tok.trim();
            if tok.is_empty() { continue; }
            // Allow numbers (1-based) OR field names.
            match tok.parse::<usize>() {
                Ok(n) if n >= 1 && n <= eligible.len() => {
                    if seen.insert(n - 1) { chosen.push(n - 1); }
                }
                _ => {
                    if let Some((i, _)) = eligible.iter().enumerate()
                        .find(|(_, (name, _))| name == tok)
                    {
                        if seen.insert(i) { chosen.push(i); }
                    } else {
                        bad.push(tok.to_string());
                    }
                }
            }
        }
        if !bad.is_empty() {
            println!("  Unknown field selectors: {}", bad.join(", "));
            continue;
        }
        if chosen.is_empty() {
            println!("  Nothing selected — pick at least one field or type 'all'.");
            continue;
        }
        println!("  → {}", chosen.iter()
            .map(|i| eligible[*i].0.as_str())
            .collect::<Vec<_>>().join(", "));
        return Ok(chosen);
    }
}

fn prompt_op_for_field(
    name: &str,
    profile: &FieldProfile,
    capable: &[OpType],
) -> Result<OpType, String> {
    if capable.is_empty() {
        return Err(format!("field '{name}' has no capable operators in the survey"));
    }
    let tokens: Vec<String> = capable.iter().map(op_token).collect();
    let default = &tokens[0];
    let semantic = profile.semantic_type.as_ref()
        .map(|s| format!("{s:?}"))
        .unwrap_or_else(|| "?".into());
    loop {
        let prompt_line = format!("  {name} ({semantic}, capable: {})", tokens.join(", "));
        println!("{prompt_line}");
        let raw = prompt("    op", default)?;
        if let Some(op) = op_from_token(&raw) {
            if capable.contains(&op) {
                return Ok(op);
            }
            println!("    '{}' isn't a capable op for this field. Pick one of: {}",
                raw.trim(), tokens.join(", "));
        } else {
            println!("    unknown op '{}'. Pick one of: {}", raw.trim(), tokens.join(", "));
        }
    }
}

fn prompt_conjugate(leaves: &[(String, OpType)]) -> Result<Option<ConjugateType>, String> {
    if leaves.len() < 2 {
        return Ok(None); // single predicate
    }
    println!();
    println!("Conjugate form?");
    println!("  [a] AND — every predicate must match");
    println!("  [o] OR  — any predicate matches");
    loop {
        let raw = prompt("  conjugate", "a")?;
        match raw.trim().to_ascii_lowercase().as_str() {
            "a" | "and" => return Ok(Some(ConjugateType::And)),
            "o" | "or" => return Ok(Some(ConjugateType::Or)),
            other => println!("  unknown '{other}'. Pick a or o."),
        }
    }
}

fn prompt_selectivity() -> Result<SelectivitySpec, String> {
    loop {
        let raw = prompt("Target selectivity (scalar like 0.1, or interval like 0.05..0.20)", "0.10")?;
        let raw = raw.trim();
        if let Some((lo, hi)) = raw.split_once("..") {
            let lo: f64 = lo.trim().parse().map_err(|_| "selectivity lo isn't a number".to_string())?;
            let hi: f64 = hi.trim().parse().map_err(|_| "selectivity hi isn't a number".to_string())?;
            if lo > hi {
                println!("  inverted interval ({lo}..{hi}). Try again.");
                continue;
            }
            return Ok(SelectivitySpec::Interval { lo, hi });
        }
        if let Ok(v) = raw.parse::<f64>() {
            return Ok(SelectivitySpec::Scalar(v));
        }
        println!("  unrecognised value. Try 0.1 or 0.05..0.20.");
    }
}

fn prompt_count(total_records: u64) -> Result<usize, String> {
    let default = recommended_count(total_records);
    println!("  Predicate count: how many predicates to emit. A useful");
    println!("  bench corpus is ~1% of metadata records, clamped to");
    println!("  [100, 10000] — large enough for statistical signal, small");
    println!("  enough to keep eval bounded.");
    println!("  Metadata records in survey: {total_records}");
    let default_str = default.to_string();
    loop {
        let raw = prompt("Predicate count", &default_str)?;
        if let Ok(n) = raw.trim().parse::<usize>() {
            if n > 0 { return Ok(n); }
            println!("  count must be > 0.");
        } else {
            println!("  unrecognised count.");
        }
    }
}

/// Compute the default predicate count from the surveyed
/// metadata size. Heuristic: ~1% of records (so a 10K-record
/// slab gets 100 predicates, 1M gets 10000), clamped to
/// [100, 10000]. Empty surveys (total = 0) fall through to the
/// floor.
pub(crate) fn recommended_count(total_records: u64) -> usize {
    let raw = total_records / 100;
    raw.clamp(100, 10000) as usize
}

fn prompt_seed() -> Result<u64, String> {
    loop {
        let raw = prompt("RNG seed", "42")?;
        if let Ok(n) = raw.trim().parse::<u64>() {
            return Ok(n);
        }
        println!("  unrecognised seed.");
    }
}

fn prompt(label: &str, default: &str) -> Result<String, String> {
    print!("{label} [{default}]: ");
    io::stdout().flush().map_err(|e| format!("stdout flush: {e}"))?;
    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line).map_err(|e| format!("stdin read: {e}"))?;
    let trimmed = line.trim_end_matches(['\n', '\r']).to_string();
    Ok(if trimmed.is_empty() { default.to_string() } else { trimmed })
}

// ─────────────────────────────────────────────────────────────────────────────
// Template assembly
// ─────────────────────────────────────────────────────────────────────────────

fn build_template(
    leaves: &[(String, OpType)],
    conjugate: Option<ConjugateType>,
) -> PNode {
    let leaf_nodes: Vec<PNode> = leaves.iter().map(|(name, op)| {
        PNode::Predicate(PredicateNode {
            field: FieldRef::Named(name.clone()),
            op: *op,
            // `?` placeholder rendered via the templating layer's
            // `parse_template`. The vernacular renderer emits the
            // Comparand verbatim, so we plant a sentinel value here
            // that round-trips through `PNode::Display` as `?` —
            // see `render_template` below.
            comparands: vec![Comparand::Null],
        })
    }).collect();
    match conjugate {
        None => leaf_nodes.into_iter().next().expect("at least one leaf"),
        Some(kind) => PNode::Conjugate(ConjugateNode {
            conjugate_type: kind,
            children: leaf_nodes,
        }),
    }
}

/// Render the constructed template, substituting `?` for the
/// placeholder Null comparands so the result is a valid
/// `gen_predicates_proto::parse_template` input.
///
/// Why we don't add a `?` variant to `Comparand`: the wire format
/// is downstream of veks-anode's codec layer, which is the
/// authoritative shape for predicates. Introducing a placeholder
/// variant would leak the templating concern into the wire
/// format. Instead, we use `Null` as the template-only sentinel
/// in the in-memory tree and emit `?` at the textual layer.
fn render_template(node: &PNode) -> String {
    let s = node.to_string();
    s.replace("NULL", "?")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::gen_predicates_proto::parse_template;

    fn pred(name: &str, op: OpType) -> (String, OpType) {
        (name.to_string(), op)
    }

    /// The rendered template parses back via the proto template
    /// parser, with `?` slots in every leaf — the wizard's
    /// textual output is a valid proto template input.
    #[test]
    fn wizard_render_round_trips_to_template_with_placeholders() {
        let leaves = vec![
            pred("age", OpType::Ge),
            pred("name", OpType::Matches),
        ];
        let node = build_template(&leaves, Some(ConjugateType::And));
        let rendered = render_template(&node);
        assert_eq!(rendered, "(age >= ? AND name MATCHES ?)");
        let parsed = parse_template(&rendered).unwrap();
        // Walking the parsed template must find two `?` slots
        // (one per leaf).
        let count = count_placeholders(&parsed);
        assert_eq!(count, 2);
    }

    fn count_placeholders(t: &super::super::gen_predicates_proto::PredicateTemplate) -> usize {
        use super::super::gen_predicates_proto::PredicateTemplate as T;
        match t {
            T::Predicate { comparands, .. } => comparands.iter().filter(|c| c.is_none()).count(),
            T::Conjugate { children, .. } => children.iter().map(count_placeholders).sum(),
        }
    }

    /// `autodetect_metadata_source` finds metadata slabs nested
    /// under canonical dataset layout (`profiles/<name>/*.slab`),
    /// not just at the top level. Regression guard for the
    /// "dataset directory has empty top level → wizard errors"
    /// bug.
    #[test]
    fn autodetect_finds_slab_under_profiles_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let nested = root.join("profiles").join("base");
        std::fs::create_dir_all(&nested).unwrap();
        write_metadata_slab(&nested.join("metadata_content.slab"));

        let found = autodetect_metadata_source(root).unwrap();
        assert!(found.is_some(), "wizard must find metadata slab nested two dirs down");
        assert_eq!(found.unwrap().file_name().unwrap(), "metadata_content.slab");
    }

    /// Slabs without a `:schema` namespace AND a non-metadata
    /// filename are skipped — keeps the auto-detect tightly
    /// scoped to actual metadata sources.
    #[test]
    fn autodetect_skips_non_metadata_slabs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Realistic non-metadata names: base/query vectors, ground-truth
        // indices, and a predicate slab. None should be picked up.
        write_plain_slab(&root.join("base.slab"));
        write_plain_slab(&root.join("query.slab"));
        write_plain_slab(&root.join("groundtruth.slab"));
        write_plain_slab(&root.join("predicates.slab"));
        let found = autodetect_metadata_source(root).unwrap();
        assert!(found.is_none(),
            "wizard should not pick up slabs whose name and schema both lack a metadata signal");
    }

    /// Pre-schema-sidecar slabs: when there's no `:schema`
    /// namespace, the wizard falls back to filename conventions
    /// — the same signal `detect_roles` uses. Regression guard
    /// for older datasets (real-world `metadata_content.slab`
    /// files in `profiles/<name>/` were written without the
    /// schema descriptor).
    #[test]
    fn autodetect_finds_legacy_slab_by_filename_when_schema_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let nested = root.join("profiles").join("base");
        std::fs::create_dir_all(&nested).unwrap();
        // No :schema namespace — only the data payload, like
        // older slabs.
        write_plain_slab(&nested.join("metadata_content.slab"));

        let found = autodetect_metadata_source(root).unwrap();
        assert!(found.is_some(),
            "filename fallback should pick up schema-less metadata_content.slab");
        assert_eq!(found.unwrap().file_name().unwrap(), "metadata_content.slab");
    }

    /// Filename-based classifier correctly distinguishes the
    /// canonical sibling slabs in a dataset directory: only
    /// `metadata_content.slab` is classified as a metadata
    /// source; `metadata_predicates.slab`, the R-facet index
    /// (`metadata_results.slab` / legacy `metadata_indices.slab`),
    /// and `predicates.slab` are all skipped.
    #[test]
    fn autodetect_picks_only_metadata_content_when_siblings_present() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let nested = root.join("profiles").join("base");
        std::fs::create_dir_all(&nested).unwrap();
        write_plain_slab(&nested.join("metadata_content.slab"));
        write_plain_slab(&nested.join("metadata_predicates.slab"));
        write_plain_slab(&nested.join("metadata_results.slab"));
        write_plain_slab(&nested.join("metadata_indices.slab"));
        write_plain_slab(&root.join("predicates.slab"));

        let found = autodetect_metadata_source(root).unwrap();
        assert!(found.is_some(), "metadata_content.slab should win");
        assert_eq!(found.unwrap().file_name().unwrap(), "metadata_content.slab");
    }

    /// Bounded depth — the wizard doesn't traverse arbitrarily
    /// deep, so misdirected invocations fail fast rather than
    /// scanning a whole filesystem.
    #[test]
    fn autodetect_respects_depth_bound() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // 5 levels deep — past the MAX_DEPTH=4 cap.
        let deep = root.join("a").join("b").join("c").join("d").join("e");
        std::fs::create_dir_all(&deep).unwrap();
        write_metadata_slab(&deep.join("metadata_content.slab"));
        let found = autodetect_metadata_source(root).unwrap();
        assert!(found.is_none(),
            "wizard should not descend past the depth cap (regression guard)");
    }

    /// Build a slab carrying a `:schema` namespace with
    /// kind=metadata so the auto-detector's classifier accepts
    /// it. Minimal record count — we only care about the
    /// descriptor, not the payload.
    fn write_metadata_slab(path: &std::path::Path) {
        use slabtastic::{SlabWriter, WriterConfig};
        use vectordata::metadata_schema::{MetadataSchema, SCHEMA_NAMESPACE};
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut w = SlabWriter::new(path, config).unwrap();
        // Payload page (the namespaces command lists this as the default ns).
        w.add_record(b"placeholder").unwrap();
        // Schema sidecar.
        w.start_namespace(SCHEMA_NAMESPACE).unwrap();
        let schema = MetadataSchema::new("test-fixture", vec![]);
        w.add_record(&schema.to_json_bytes()).unwrap();
        w.finish().unwrap();
    }

    /// A slab without a schema namespace — the negative-case
    /// fixture for `autodetect_skips_non_metadata_slabs`.
    fn write_plain_slab(path: &std::path::Path) {
        use slabtastic::{SlabWriter, WriterConfig};
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut w = SlabWriter::new(path, config).unwrap();
        w.add_record(b"placeholder").unwrap();
        w.finish().unwrap();
    }

    #[test]
    fn wizard_single_field_renders_without_parens() {
        let leaves = vec![pred("age", OpType::Eq)];
        let node = build_template(&leaves, None);
        let rendered = render_template(&node);
        assert_eq!(rendered, "age = ?");
        let parsed = parse_template(&rendered).unwrap();
        assert_eq!(count_placeholders(&parsed), 1);
    }

    /// The recommended predicate count scales with the
    /// surveyed metadata size, clamped to `[100, 10000]` so
    /// tiny corpora don't get 1-predicate runs and huge ones
    /// don't generate millions.
    #[test]
    fn recommended_count_clamps_to_useful_range() {
        // Empty / tiny — floor at 100.
        assert_eq!(recommended_count(0), 100);
        assert_eq!(recommended_count(50), 100);
        assert_eq!(recommended_count(9_999), 100);
        // 1% mid-range.
        assert_eq!(recommended_count(10_000), 100);
        assert_eq!(recommended_count(100_000), 1_000);
        assert_eq!(recommended_count(500_000), 5_000);
        // Cap at 10K for very large corpora.
        assert_eq!(recommended_count(1_000_000), 10_000);
        assert_eq!(recommended_count(100_000_000), 10_000);
    }

    /// `?` placeholder in the rendered template is the
    /// templating-layer sentinel — confirms the wizard's
    /// `Null → ?` substitution doesn't accidentally swallow
    /// other NULL occurrences. (NULL inside a `'…'` text
    /// literal isn't substituted because we use `replace` over
    /// the whole string only when the renderer never emits
    /// `NULL` inside string literals — `PNode::Display` always
    /// quotes Text comparands, so a literal `NULL` only appears
    /// at comparand positions.)
    #[test]
    fn wizard_render_replaces_only_unquoted_null() {
        let leaves = vec![pred("status", OpType::Eq)];
        let node = build_template(&leaves, None);
        let rendered = render_template(&node);
        assert_eq!(rendered, "status = ?");
    }
}
