// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `verify predicate-strata`: check a stratified predicate set's claims
//! against the answer keys it produced (TS-167).
//!
//! The stratified generator records, per query ordinal, the family,
//! the cell (`family:1e-d`), the exact match count it expects over the
//! census population, and — when the queries' own metadata rows were
//! given — whether the query's own passage satisfies its predicate.
//! This step reads those claims back and holds them against what
//! `compute evaluate-predicates` wrote at every profile:
//!
//! - one record per query, and both annotation namespaces the same
//!   length (TS-156, TS-111);
//! - at the profile whose base is the census population, the realised
//!   match count equals the recorded one exactly (TS-43) — except for
//!   the control family, whose count is by construction and is held to
//!   its band there like everywhere else (TS-115);
//! - at every other profile above the reliability threshold, the
//!   realised selectivity lies in the half-decade band of the record's
//!   cell, and no record is empty (TS-42, TS-43); below the threshold
//!   only the control family is held to a non-empty result;
//! - every `query_in_filter` label agrees with evaluating the predicate
//!   against the query's own row (TS-161, TS-166).
//!
//! Realised counts are read from the record lengths of the results
//! slab, so the pass costs one sequential read of each profile's R
//! facet and decodes nothing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use slabtastic::SlabReader;
use vectordata::dataset::DatasetConfig;
use vectordata::metadata_schema::{FAMILIES_NAMESPACE, GENERATION_NAMESPACE, SURVEY_NAMESPACE};
use veks_core::formats::anode::{self, ANode};
use veks_core::formats::mnode::{MNode, MValue};
use veks_core::formats::pnode::PNode;
use veks_core::formats::pnode::eval::evaluate;
use vectordata::io::XvecReader;

use crate::pipeline::command::{
    render_options_table, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext,
};
use crate::pipeline::commands::survey::SurveyReport;

use super::gen_predicates_common::{error_result, resolve_path};

pub struct VerifyPredicateStrataOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(VerifyPredicateStrataOp)
}

/// One record's claims, as the generator wrote them.
struct Claim {
    family: String,
    cell: String,
    decade: i32,
    expected: u64,
    query_in_filter: Option<bool>,
}

/// What one profile showed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReport {
    pub profile: String,
    pub base_count: u64,
    /// Whether the profile's base is the census population, where
    /// counts must match exactly.
    pub census_profile: bool,
    pub above_threshold: bool,
    pub records: usize,
    pub exact_mismatches: usize,
    pub band_violations: usize,
    pub zero_matches: usize,
    /// Zero matches in the control family below the threshold (TS-42).
    pub control_zero_below_threshold: usize,
    pub per_family: BTreeMap<String, FamilyReport>,
    pub first_violations: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FamilyReport {
    pub records: usize,
    pub mean_claimed_selectivity: f64,
    pub mean_realised_selectivity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrataReport {
    pub schema_version: u32,
    pub predicates: usize,
    pub query_count: Option<usize>,
    pub census_population: u64,
    pub reliability_threshold: u64,
    pub label_checks: Option<usize>,
    pub label_disagreements: Option<usize>,
    pub profiles: Vec<ProfileReport>,
    pub violations: usize,
    pub seconds: f64,
}

impl CommandOp for VerifyPredicateStrataOp {
    fn command_path(&self) -> &str {
        "verify predicate-strata"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_VERIFY
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn command_doc(&self) -> CommandDoc {
        CommandDoc {
            summary: "Hold a stratified predicate set's claims against its answer keys at every profile".into(),
            body: format!(
                r#"# verify predicate-strata

Reads the `families` and `generation` namespaces of a stratified
predicate facet (`predicates`) and, for every profile with a `results`
facet, compares each record's recorded match count and selectivity band
with the realised match count in the results slab. Counts must match
exactly at the profile whose base is the census population and lie in
the record's half-decade band elsewhere above the
`reliability-threshold`; below it only the control family must be
non-empty. With `queries` it checks that there is one record per query;
with the queries' own metadata rows (`query-metadata`) it re-derives
every `query_in_filter` label. Writes a JSON report to `output` and
fails on any violation.

## Options

{}"#,
                render_options_table(&self.describe_options())
            ),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        let opt = |name: &str, ty: &str, required: bool, default: Option<&str>, desc: &str, role: OptionRole| OptionDesc {
            name: name.into(),
            type_name: ty.into(),
            required,
            default: default.map(str::to_string),
            description: desc.into(),
            extended_description: None,
            role,
        };
        vec![
            opt("predicates", "Path", true, None, "Stratified predicate facet (with families and generation namespaces)", OptionRole::Input),
            opt("results", "string", false, Some("metadata_results.slab"), "Results facet file name under each profile directory", OptionRole::Config),
            opt("queries", "Path", false, None, "Query vectors, to check that there is one predicate per query", OptionRole::Input),
            opt("query-metadata", "Path", false, None, "The queries' own metadata rows, to re-derive every query_in_filter label", OptionRole::Input),
            opt("reliability-threshold", "int", false, Some("10000000"), "Base count from which every family must hold its band and be non-empty", OptionRole::Config),
            opt("output", "Path", true, None, "JSON report", OptionRole::Output),
        ]
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();
        let predicates_path = match options.require("predicates") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let output_path = match options.require("output") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let results_name = options.get("results").unwrap_or("metadata_results.slab").to_string();
        let threshold = match options.parse_or::<u64>("reliability-threshold", 10_000_000) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };

        // ── The claims ──────────────────────────────────────────────
        let predicates = match read_pnodes(&predicates_path) {
            Ok(p) => p,
            Err(e) => return error_result(e, start),
        };
        let families = match read_mnodes(&predicates_path, Some(FAMILIES_NAMESPACE)) {
            Ok(f) => f,
            Err(e) => return error_result(e, start),
        };
        let generation = match read_mnodes(&predicates_path, Some(GENERATION_NAMESPACE)) {
            Ok(g) => g,
            Err(e) => return error_result(e, start),
        };
        let mut violations: Vec<String> = Vec::new();
        if families.len() != predicates.len() || generation.len() != predicates.len() {
            violations.push(format!(
                "{} predicates but {} families and {} generation records",
                predicates.len(),
                families.len(),
                generation.len()
            ));
        }
        let population = match census_population(&predicates_path) {
            Ok(n) => n,
            Err(e) => return error_result(e, start),
        };
        let query_count = match options.get("queries") {
            Some(q) => match XvecReader::<f32>::open_path(&resolve_path(q, &ctx.workspace)) {
                Ok(r) => Some(r.count()),
                Err(e) => return error_result(format!("failed to open queries: {}", e), start),
            },
            None => None,
        };
        if let Some(q) = query_count
            && q != predicates.len()
        {
            violations.push(format!(
                "{} predicates for {} queries; record i must be query i's predicate",
                predicates.len(),
                q
            ));
        }
        let claims: Vec<Claim> = families
            .iter()
            .zip(generation.iter())
            .map(|(f, g)| {
                let cell = text(g, "cell").unwrap_or_default();
                let decade = cell
                    .split(":1e")
                    .nth(1)
                    .and_then(|d| d.parse::<i32>().ok())
                    .unwrap_or(0);
                Claim {
                    family: text(f, "family").unwrap_or_default(),
                    cell,
                    decade,
                    expected: int(g, "expected_count").unwrap_or(0).max(0) as u64,
                    query_in_filter: match f.fields.get("query_in_filter") {
                        Some(MValue::Bool(b)) => Some(*b),
                        _ => None,
                    },
                }
            })
            .collect();

        // ── Labels against the queries' own rows (TS-166) ───────────
        let (mut label_checks, mut label_disagreements) = (None, None);
        if let Some(s) = options.get("query-metadata") {
            let path = resolve_path(s, &ctx.workspace);
            let rows = match read_mnodes(&path, None) {
                Ok(r) => r,
                Err(e) => return error_result(e, start),
            };
            if rows.len() != predicates.len() {
                violations.push(format!(
                    "query metadata holds {} rows for {} predicates",
                    rows.len(),
                    predicates.len()
                ));
            } else {
                let (mut checks, mut bad) = (0usize, 0usize);
                for (i, claim) in claims.iter().enumerate() {
                    let Some(recorded) = claim.query_in_filter else { continue };
                    checks += 1;
                    let actual = evaluate(&predicates[i], &rows[i]);
                    if actual != recorded {
                        bad += 1;
                        if bad <= 5 {
                            violations.push(format!(
                                "query {}: recorded query_in_filter={} but `{}` evaluates to {} on the query's own row",
                                i, recorded, predicates[i], actual
                            ));
                        }
                    }
                }
                label_checks = Some(checks);
                label_disagreements = Some(bad);
                if bad > 5 {
                    violations.push(format!("... {} label disagreements in total", bad));
                }
            }
        }

        // ── Every profile with a results facet ──────────────────────
        let profiles = discover_profiles(&ctx.workspace, &results_name);
        if profiles.is_empty() {
            return error_result(
                format!("no profile holds a {} results facet yet", results_name),
                start,
            );
        }
        let mut reports: Vec<ProfileReport> = Vec::new();
        let profile_pb = ctx.ui.bar_with_unit(profiles.len() as u64, "profiles checked", "profiles");
        for (name, base_count, results_path) in profiles {
            let base_count = base_count.unwrap_or(population);
            let realised = match read_record_lengths(&results_path) {
                Ok(v) => v,
                Err(e) => return error_result(e, start),
            };
            let mut report = ProfileReport {
                profile: name.clone(),
                base_count,
                census_profile: base_count == population,
                above_threshold: base_count >= threshold,
                records: realised.len(),
                exact_mismatches: 0,
                band_violations: 0,
                zero_matches: 0,
                control_zero_below_threshold: 0,
                per_family: BTreeMap::new(),
                first_violations: Vec::new(),
            };
            if realised.len() != claims.len() {
                report.first_violations.push(format!(
                    "{} results records for {} predicates",
                    realised.len(),
                    claims.len()
                ));
            }
            let n = base_count as f64;
            let mut sums: BTreeMap<String, (usize, f64, f64)> = BTreeMap::new();
            for (i, (claim, &count)) in claims.iter().zip(realised.iter()).enumerate() {
                let sel = count as f64 / n;
                let claimed = claim.expected as f64 / population as f64;
                let e = sums.entry(claim.family.clone()).or_insert((0, 0.0, 0.0));
                e.0 += 1;
                e.1 += claimed;
                e.2 += sel;
                let note = |msg: String, report: &mut ProfileReport| {
                    if report.first_violations.len() < 8 {
                        report.first_violations.push(msg);
                    }
                };
                // A control predicate's count is by construction, not by
                // census (TS-115), so it is held to its band everywhere.
                if report.census_profile && claim.family != "control" {
                    if count != claim.expected {
                        report.exact_mismatches += 1;
                        note(format!("query {} ({}): expected {} matches, results hold {}", i, claim.cell, claim.expected, count), &mut report);
                    }
                } else if report.above_threshold {
                    let lo = 10f64.powi(claim.decade) / 10f64.sqrt();
                    let hi = 10f64.powi(claim.decade) * 10f64.sqrt();
                    if count == 0 {
                        report.zero_matches += 1;
                        note(format!("query {} ({}): no matches at base {}", i, claim.cell, base_count), &mut report);
                    } else if sel < lo || sel >= hi {
                        report.band_violations += 1;
                        note(format!("query {} ({}): selectivity {:.3e} outside [{:.3e}, {:.3e}) at base {}", i, claim.cell, sel, lo, hi, base_count), &mut report);
                    }
                } else if claim.family == "control" && count == 0 {
                    report.control_zero_below_threshold += 1;
                    note(format!("query {} ({}): the control family is empty at base {}", i, claim.cell, base_count), &mut report);
                }
            }
            for (family, (records, claimed, realised)) in sums {
                report.per_family.insert(
                    family,
                    FamilyReport {
                        records,
                        mean_claimed_selectivity: claimed / records as f64,
                        mean_realised_selectivity: realised / records as f64,
                    },
                );
            }
            let bad = report.exact_mismatches + report.band_violations + report.zero_matches + report.control_zero_below_threshold
                + usize::from(realised.len() != claims.len());
            if bad > 0 {
                violations.push(format!(
                    "profile {} (base {}): {} exact mismatch(es), {} band violation(s), {} empty result(s), {} empty control result(s) below the threshold",
                    name, base_count, report.exact_mismatches, report.band_violations, report.zero_matches, report.control_zero_below_threshold
                ));
            }
            ctx.ui.log(&format!(
                "predicate-strata: {} (base {}): {} records, {} exact mismatch(es), {} band violation(s), {} empty",
                name, base_count, report.records, report.exact_mismatches, report.band_violations, report.zero_matches
            ));
            reports.push(report);
            profile_pb.inc(1);
        }
        profile_pb.finish();

        let report = StrataReport {
            schema_version: 1,
            predicates: predicates.len(),
            query_count,
            census_population: population,
            reliability_threshold: threshold,
            label_checks,
            label_disagreements,
            profiles: reports,
            violations: violations.len(),
            seconds: start.elapsed().as_secs_f64(),
        };
        if let Some(parent) = output_path.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            return error_result(format!("failed to create {}: {}", parent.display(), e), start);
        }
        match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&output_path, json) {
                    return error_result(format!("failed to write {}: {}", output_path.display(), e), start);
                }
            }
            Err(e) => return error_result(format!("report serialisation failed: {}", e), start),
        }
        let profiles_checked = report.profiles.len();
        if violations.is_empty() {
            CommandResult {
                status: Status::Ok,
                message: format!(
                    "{} predicates hold their claims at {} profile(s){}",
                    report.predicates,
                    profiles_checked,
                    match label_checks {
                        Some(c) => format!("; {} query_in_filter label(s) re-derived", c),
                        None => String::new(),
                    }
                ),
                produced: vec![output_path],
                elapsed: start.elapsed(),
            }
        } else {
            CommandResult {
                status: Status::Error,
                message: format!(
                    "{} violation(s) across {} profile(s): {}",
                    violations.len(),
                    profiles_checked,
                    violations.iter().take(4).cloned().collect::<Vec<_>>().join("; ")
                ),
                produced: vec![output_path],
                elapsed: start.elapsed(),
            }
        }
    }
}

fn text(m: &MNode, key: &str) -> Option<String> {
    match m.fields.get(key) {
        Some(MValue::Text(t)) => Some(t.clone()),
        _ => None,
    }
}

fn int(m: &MNode, key: &str) -> Option<i64> {
    match m.fields.get(key) {
        Some(MValue::Int(v)) => Some(*v),
        Some(MValue::Int32(v)) => Some(*v as i64),
        Some(MValue::Short(v)) => Some(*v as i64),
        _ => None,
    }
}

/// Every record of a namespace, decoded as MNodes.
fn read_mnodes(path: &Path, namespace: Option<&str>) -> Result<Vec<MNode>, String> {
    let reader = match namespace {
        Some(ns) => SlabReader::open_namespace(path, Some(ns)),
        None => SlabReader::open(path),
    }
    .map_err(|e| format!("failed to open {} ({}): {}", path.display(), namespace.unwrap_or("content"), e))?;
    let mut out = Vec::with_capacity(reader.total_records() as usize);
    for entry in reader.page_entries() {
        let page = reader
            .read_data_page(&entry)
            .map_err(|e| format!("{}: {}", path.display(), e))?;
        for i in 0..page.record_count() {
            let bytes = page
                .get_record(i)
                .ok_or_else(|| format!("{}: record {} is missing", path.display(), out.len()))?;
            match anode::decode(bytes) {
                Ok(ANode::MNode(m)) => out.push(m),
                Ok(_) => return Err(format!("{}: record {} is not an MNode", path.display(), out.len())),
                Err(e) => return Err(format!("{}: record {}: {}", path.display(), out.len(), e)),
            }
        }
    }
    Ok(out)
}

/// Every predicate of the content namespace.
fn read_pnodes(path: &Path) -> Result<Vec<PNode>, String> {
    let reader = SlabReader::open(path)
        .map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let mut out = Vec::with_capacity(reader.total_records() as usize);
    for entry in reader.page_entries() {
        let page = reader
            .read_data_page(&entry)
            .map_err(|e| format!("{}: {}", path.display(), e))?;
        for i in 0..page.record_count() {
            let bytes = page
                .get_record(i)
                .ok_or_else(|| format!("{}: record {} is missing", path.display(), out.len()))?;
            out.push(
                PNode::from_bytes_named(bytes)
                    .map_err(|e| format!("{}: record {}: {}", path.display(), out.len(), e))?,
            );
        }
    }
    Ok(out)
}

/// The census population the generator drew against, from the survey
/// the predicate facet carries.
fn census_population(predicates: &Path) -> Result<u64, String> {
    let reader = SlabReader::open_namespace(predicates, Some(SURVEY_NAMESPACE))
        .map_err(|e| format!("{} carries no survey namespace: {}", predicates.display(), e))?;
    for entry in reader.page_entries() {
        let page = reader
            .read_data_page(&entry)
            .map_err(|e| format!("{}: {}", predicates.display(), e))?;
        if page.record_count() > 0 {
            let bytes = page.get_record(0).ok_or("survey record missing")?;
            let survey: SurveyReport = serde_json::from_slice(bytes)
                .map_err(|e| format!("survey namespace of {} does not parse: {}", predicates.display(), e))?;
            return Ok(survey.source.total_records);
        }
    }
    Err(format!("{} carries an empty survey namespace", predicates.display()))
}

/// Realised match counts: each results record is packed i32 ordinals,
/// so its length in bytes over four is its count.
fn read_record_lengths(path: &Path) -> Result<Vec<u64>, String> {
    let reader = SlabReader::open(path)
        .map_err(|e| format!("failed to open results {}: {}", path.display(), e))?;
    let mut out = Vec::with_capacity(reader.total_records() as usize);
    for entry in reader.page_entries() {
        let page = reader
            .read_data_page(&entry)
            .map_err(|e| format!("{}: {}", path.display(), e))?;
        for i in 0..page.record_count() {
            let bytes = page
                .get_record(i)
                .ok_or_else(|| format!("{}: record {} is missing", path.display(), out.len()))?;
            out.push((bytes.len() / 4) as u64);
        }
    }
    Ok(out)
}

/// Profiles holding a results facet: from `dataset.yaml` when it loads
/// (partition profiles excluded), else from the `profiles/` directory,
/// with a sized profile's base count parsed from its name. Sorted by
/// base count, the census profile (no declared count) last.
fn discover_profiles(workspace: &Path, results_name: &str) -> Vec<(String, Option<u64>, PathBuf)> {
    let mut found: IndexMap<String, (Option<u64>, PathBuf)> = IndexMap::new();
    if let Ok(config) = DatasetConfig::load_and_resolve(&workspace.join("dataset.yaml")) {
        for (name, profile) in &config.profiles.profiles {
            if profile.partition {
                continue;
            }
            let path = workspace.join(format!("profiles/{}/{}", name, results_name));
            if path.exists() {
                found.insert(name.clone(), (profile.base_count, path));
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir(workspace.join("profiles")) {
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if found.contains_key(&name) {
                continue;
            }
            let path = entry.path().join(results_name);
            if path.exists() {
                let bc = vectordata::dataset::source::parse_number_with_suffix(&name).ok();
                found.insert(name, (bc, path));
            }
        }
    }
    let mut out: Vec<(String, Option<u64>, PathBuf)> = found
        .into_iter()
        .map(|(n, (bc, p))| (n, bc, p))
        .collect();
    out.sort_by_key(|(_, bc, _)| bc.unwrap_or(u64::MAX));
    out
}
