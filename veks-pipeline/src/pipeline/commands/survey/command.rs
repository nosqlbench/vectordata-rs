// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `analyze survey` — pipeline command-op.
//!
//! Parses CLI/YAML options into a [`SurveyConfig`], invokes the
//! two-pass [`super::orchestrator::survey`] driver, and writes the
//! report to the configured output path.
//!
//! This is the operator-facing surface. The driver itself is a
//! library function so tests and downstream tooling can invoke it
//! directly without going through the pipeline runner.

use std::path::PathBuf;
use std::time::Instant;

use crate::pipeline::command::{
    render_options_table, ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc,
    OptionRole, Options, Status, StreamContext,
};

use super::findings::{render_findings, FindingsConfig, Severity};
use super::orchestrator::{survey, SurveyConfig};

/// `analyze survey` — incremental metadata survey.
pub struct SurveyOp;

/// Factory used by the pipeline command registry.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(SurveyOp)
}

impl CommandOp for SurveyOp {
    fn command_path(&self) -> &str {
        "analyze survey"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_ANALYZE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Two-pass type-driven metadata survey".into(),
            body: format!(
                r#"# analyze survey

Two-pass incremental metadata survey. Pass 1 explores each field's
encoding and cardinality regime; Pass 2 runs a measure suite
tailored to that verdict. See `docs/sysref/13-metadata-survey.md`
for the full design.

Out of scope: this command surveys ANode → MNode slab files only.
Other formats (ivec metadata, Parquet, NPY) are not supported.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source = match options.require("source") {
            Ok(s) => s,
            Err(e) => return err(e, start),
        };
        let input_path = resolve(source, &ctx.workspace);

        let output_path = match options.get("output") {
            Some(s) => resolve(s, &ctx.workspace),
            None => ctx.workspace.join("survey.json"),
        };

        let cfg = SurveyConfig {
            samples: parse_opt(options, "samples", 100_000),
            distinct_cap: parse_opt::<u32>(options, "distinct-cap", 4_096),
            low_card_threshold: parse_opt::<u32>(options, "low-card-threshold", 64),
            mid_card_threshold: parse_opt::<u32>(options, "mid-card-threshold", 4_096),
            reservoir_size: parse_opt(options, "reservoir-size", 1_024),
            reservoir_seed: parse_opt(options, "reservoir-seed", 0xC011EC70u64),
            hll_precision: parse_opt::<u8>(options, "hll-precision", 12),
            top_k: parse_opt(options, "top-k", 64),
            quantile_k: parse_opt(options, "quantile-k", 1000),
            max_pair_analyses: parse_opt(options, "max-pair-analyses", 1_024),
            semantic_confidence: parse_opt(options, "semantic-confidence", 0.95),
        };

        let report = match survey(&input_path, &cfg, Some(&ctx.ui)) {
            Ok(r) => r,
            Err(e) => return err(e, start),
        };

        // Emit JSON.
        let json = match serde_json::to_string_pretty(&report) {
            Ok(s) => s,
            Err(e) => return err(format!("JSON serialization failed: {}", e), start),
        };
        if let Some(parent) = output_path.parent()
            && !parent.exists()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            return err(format!("failed to create directory {}: {}", parent.display(), e), start);
        }
        if let Err(e) = std::fs::write(&output_path, json) {
            return err(format!("failed to write {}: {}", output_path.display(), e), start);
        }
        ctx.ui.log(&format!("Survey JSON written to {}", output_path.display()));

        let mut produced = vec![output_path.clone()];

        // Findings — Markdown and JSON. Both are produced from a
        // single curated tree so they cannot drift; either or both
        // can be disabled by passing an empty string.
        let findings_md_path = match options.get("findings-markdown") {
            Some("") => None,
            Some(s) => Some(resolve(s, &ctx.workspace)),
            None => output_path.parent().map(|p| p.join("survey.findings.md"))
                .or_else(|| Some(ctx.workspace.join("survey.findings.md"))),
        };
        let findings_json_path = match options.get("findings-json") {
            Some("") => None,
            Some(s) => Some(resolve(s, &ctx.workspace)),
            None => output_path.parent().map(|p| p.join("survey.findings.json"))
                .or_else(|| Some(ctx.workspace.join("survey.findings.json"))),
        };
        if findings_md_path.is_some() || findings_json_path.is_some() {
            let mut findings_cfg = FindingsConfig::default();
            if let Some(severity_str) = options.get("findings-severity") {
                findings_cfg.min_severity = parse_severity(severity_str);
            }
            if let Some(name) = output_path.file_name().and_then(|n| n.to_str()) {
                findings_cfg.survey_json_filename = name.to_string();
            }
            let (md_text, findings_json) = render_findings(&report, &findings_cfg);
            if let Some(md_path) = findings_md_path {
                if let Err(e) = std::fs::write(&md_path, &md_text) {
                    return err(format!("failed to write {}: {}", md_path.display(), e), start);
                }
                ctx.ui.log(&format!("Findings Markdown written to {}", md_path.display()));
                produced.push(md_path);
            }
            if let Some(json_path) = findings_json_path {
                match serde_json::to_string_pretty(&findings_json) {
                    Ok(s) => {
                        if let Err(e) = std::fs::write(&json_path, s) {
                            return err(format!("failed to write {}: {}", json_path.display(), e), start);
                        }
                        ctx.ui.log(&format!("Findings JSON written to {}", json_path.display()));
                        produced.push(json_path);
                    }
                    Err(e) => return err(format!("findings JSON serialization failed: {}", e), start),
                }
            }
        }

        let message = format!(
            "{} records sampled, {} fields ({} unstable)",
            report.source.sampled_records,
            report.fields.len(),
            report
                .fields
                .values()
                .filter(|f| matches!(
                    f.semantic_type,
                    Some(super::types::SemanticType::Unstable)
                ))
                .count(),
        );

        CommandResult {
            status: Status::Ok,
            message,
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt("source", "Path", true, None, "Slab file to survey", OptionRole::Input),
            opt("output", "Path", false, Some("survey.json"), "Structured JSON report path", OptionRole::Output),
            opt("samples", "int", false, Some("100000"), "Per-pass sample cap", OptionRole::Config),
            opt("distinct-cap", "int", false, Some("4096"), "Bounded distinct-tracker capacity during Pass 1", OptionRole::Config),
            opt("low-card-threshold", "int", false, Some("64"), "Cardinality <= this is LowCard with exact frequency table", OptionRole::Config),
            opt("mid-card-threshold", "int", false, Some("4096"), "Cardinality between low/mid is MidCard with HLL+HeavyHitters", OptionRole::Config),
            opt("reservoir-size", "int", false, Some("1024"), "Per-field reservoir sample size", OptionRole::Config),
            opt("reservoir-seed", "int", false, Some("3221234288"), "Reservoir RNG seed (for reproducibility)", OptionRole::Config),
            opt("hll-precision", "int", false, Some("12"), "HyperLogLog precision (register count = 2^p)", OptionRole::Config),
            opt("top-k", "int", false, Some("64"), "HeavyHitters top-K capacity", OptionRole::Config)
                .with_extended_description(
                    "Maximum number of high-frequency values a Misra-Gries sketch \
                     keeps per mid-cardinality field. Higher values catch more \
                     of the tail at the cost of memory and analysis time.\n\n\
                     Typical choices:\n  \
                     - 16-32  : fast, only the dominant modes\n  \
                     - 64     : default; good balance for most metadata\n  \
                     - 256+   : preserves long-tailed mode distributions\n\n\
                     Only fields classified MidCard (cardinality between \
                     --low-card-threshold and --mid-card-threshold) use this \
                     measure. LowCard fields use an exact frequency table; \
                     HighCard fields rely on HyperLogLog and a reservoir sample.",
                ),
            opt("quantile-k", "int", false, Some("1000"), "KLL quantile sketch parameter k (floor: 1000; lower values are silently clamped up)", OptionRole::Config),
            opt("max-pair-analyses", "int", false, Some("1024"), "Maximum cross-field pair analyzers to schedule", OptionRole::Config),
            opt("semantic-confidence", "float", false, Some("0.95"), "Minimum semantic-probe match rate to commit a SemanticType verdict (sysref §13.3.3)", OptionRole::Config),
            opt("findings-markdown", "Path", false, None, "Markdown findings output (default: survey.findings.md alongside --output; empty string disables)", OptionRole::Output),
            opt("findings-json", "Path", false, None, "JSON findings output (default: survey.findings.json alongside --output; empty string disables)", OptionRole::Output),
            opt("findings-severity", "string", false, Some("info"), "Minimum severity to include in findings: info|notable|warning|error", OptionRole::Config),
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["source"],
            &["output"],
        )
    }
}

fn resolve(path_str: &str, workspace: &std::path::Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn err(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn opt(
    name: &str,
    type_name: &str,
    required: bool,
    default: Option<&str>,
    desc: &str,
    role: OptionRole,
) -> OptionDesc {
    OptionDesc {
        name: name.into(),
        type_name: type_name.into(),
        required,
        default: default.map(str::to_string),
        description: desc.into(), extended_description: None,
        role,
    }
}

fn parse_opt<T: std::str::FromStr>(options: &Options, key: &str, fallback: T) -> T {
    options.get(key).and_then(|s| s.parse().ok()).unwrap_or(fallback)
}

fn parse_severity(s: &str) -> Severity {
    match s.to_ascii_lowercase().as_str() {
        "notable" => Severity::Notable,
        "warning" => Severity::Warning,
        "error" => Severity::Error,
        _ => Severity::Info,
    }
}
