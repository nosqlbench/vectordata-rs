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
//!   its binomial draw there like everywhere else (TS-115);
//! - at every other profile, the realised count is **credible** under
//!   the record's sampling model (TS-173): a censused predicate's
//!   matches in the shuffled prefix of `N` rows are a hypergeometric
//!   draw of its census count from the population, and a control
//!   predicate's are a binomial draw at its constructed selectivity.
//!   Sampling noise around the half-decade band and empties where the
//!   expected count is small are what the model predicts, and are
//!   reported, not failed;
//! - above the reliability threshold, a record that clears the
//!   profile's floor `M + 3√M` (TS-11, TS-51) is non-empty (TS-42);
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
    /// Matches a record needs at this profile to apply to it (TS-51):
    /// `M + 3√M`.
    pub floor: f64,
    pub records: usize,
    /// Censused records whose count differs at the census profile.
    pub exact_mismatches: usize,
    /// Records whose count is outside the two-sided
    /// [`CREDIBILITY_ALPHA`] region of their sampling model (TS-173).
    pub incredible_counts: usize,
    /// Records that clear the floor at this profile, above the
    /// threshold (TS-51).
    pub applicable: usize,
    /// Applicable records with no matches (TS-42).
    pub applicable_empty: usize,
    /// Records with no matches, every family.
    pub empties: usize,
    /// How many empties the sampling models predict: the sum of each
    /// record's probability of an empty draw.
    pub empties_expected: f64,
    /// Records whose realised selectivity at this profile lies outside
    /// their cell's half-decade band — sampling noise at a sized
    /// profile, reported for the reader (TS-43).
    pub out_of_band: usize,
    pub per_family: BTreeMap<String, FamilyReport>,
    pub first_violations: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FamilyReport {
    pub records: usize,
    pub mean_claimed_selectivity: f64,
    pub mean_realised_selectivity: f64,
    pub out_of_band: usize,
    pub empties: usize,
    /// Records of the family whose count is not credible (TS-173).
    pub incredible: usize,
}

/// Running sums for one family at one profile.
#[derive(Default)]
struct FamilySums {
    records: usize,
    claimed: f64,
    realised: f64,
    out_of_band: usize,
    empties: usize,
    incredible: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrataReport {
    pub schema_version: u32,
    pub predicates: usize,
    pub query_count: Option<usize>,
    pub census_population: u64,
    pub reliability_threshold: u64,
    /// `M` in the floor `M + 3√M` (TS-11).
    pub min_matches: u64,
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
facet, compares each record's recorded match count with the realised
match count in the results slab. A censused record's count must match
exactly at the profile whose base is the census population; at every
other profile each count must be credible under the record's sampling
model — a hypergeometric draw of the census count through the shuffled
prefix for a censused predicate, a binomial draw at the constructed
selectivity for a control predicate — so sampling noise around the
half-decade band and empties where few matches are expected are
reported, not failed. Above the `reliability-threshold` a record that
clears the floor `M + 3√M` (`min-matches`) must be non-empty. With
`queries` it checks that there is one record per query;
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
            opt("reliability-threshold", "int", false, Some("10000000"), "Base count from which a record that clears the floor must be non-empty (TS-46)", OptionRole::Config),
            opt("min-matches", "int", false, Some("100"), "M in the floor s·N ≥ M + 3√M that decides which records apply to a profile (TS-11, TS-51)", OptionRole::Config),
            opt("output", "Path", true, None, "JSON report", OptionRole::Output),
        ]
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    /// The facets named by option are the manifest's inputs; the
    /// per-profile results facets are discovered, not named, and are
    /// other steps' outputs. The report is the one output.
    fn project_artifacts(&self, step_id: &str, options: &Options) -> crate::pipeline::command::ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id,
            self.command_path(),
            options,
            &["predicates", "queries", "query-metadata"],
            &["output"],
        )
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
        let min_matches = match options.parse_or::<u64>("min-matches", 100) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let floor = min_matches as f64 + 3.0 * (min_matches as f64).sqrt();

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
                floor,
                records: realised.len(),
                exact_mismatches: 0,
                incredible_counts: 0,
                applicable: 0,
                applicable_empty: 0,
                empties: 0,
                empties_expected: 0.0,
                out_of_band: 0,
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
            let mut sums: BTreeMap<String, FamilySums> = BTreeMap::new();
            for (i, (claim, &count)) in claims.iter().zip(realised.iter()).enumerate() {
                let sel = count as f64 / n;
                let claimed = claim.expected as f64 / population as f64;
                // A control predicate's count is by construction (TS-115):
                // a binomial draw at its selectivity. A censused one is
                // the shuffle's hypergeometric draw of its census count.
                let model = if claim.family == "control" {
                    CountModel::Binomial { n: base_count, p: claimed }
                } else {
                    CountModel::Hypergeometric { pop: population, k: claim.expected, n: base_count }
                };
                let lo = 10f64.powi(claim.decade) / 10f64.sqrt();
                let hi = 10f64.powi(claim.decade) * 10f64.sqrt();
                let in_band = sel >= lo && sel < hi;
                let applies = report.above_threshold && model.mean() >= floor;
                let f = sums.entry(claim.family.clone()).or_default();
                f.records += 1;
                f.claimed += claimed;
                f.realised += sel;
                if !in_band {
                    f.out_of_band += 1;
                    report.out_of_band += 1;
                }
                if count == 0 {
                    f.empties += 1;
                    report.empties += 1;
                }
                report.empties_expected += model.p_empty();
                if applies {
                    report.applicable += 1;
                }
                let note = |msg: String, report: &mut ProfileReport| {
                    if report.first_violations.len() < 8 {
                        report.first_violations.push(msg);
                    }
                };
                if report.census_profile && claim.family != "control" {
                    if count != claim.expected {
                        report.exact_mismatches += 1;
                        note(format!("query {} ({}): expected {} matches, results hold {}", i, claim.cell, claim.expected, count), &mut report);
                    }
                } else if applies && count == 0 {
                    report.applicable_empty += 1;
                    note(format!("query {} ({}): no matches at base {} where {:.1} are expected, above the floor of {:.1}", i, claim.cell, base_count, model.mean(), floor), &mut report);
                } else if !model.credible(count) {
                    report.incredible_counts += 1;
                    sums.get_mut(&claim.family).expect("family sums exist").incredible += 1;
                    let sigma = model.sigma();
                    let z = if sigma > 0.0 { (count as f64 - model.mean()) / sigma } else { f64::INFINITY };
                    note(
                        format!(
                            "query {} ({}): {} matches at base {} is not credible for {:.2} expected (σ {:.2}, {:+.1}σ)",
                            i, claim.cell, count, base_count, model.mean(), sigma, z
                        ),
                        &mut report,
                    );
                }
            }
            for (family, s) in sums {
                report.per_family.insert(
                    family,
                    FamilyReport {
                        records: s.records,
                        mean_claimed_selectivity: s.claimed / s.records as f64,
                        mean_realised_selectivity: s.realised / s.records as f64,
                        out_of_band: s.out_of_band,
                        empties: s.empties,
                        incredible: s.incredible,
                    },
                );
            }
            let bad = report.exact_mismatches + report.incredible_counts + report.applicable_empty
                + usize::from(realised.len() != claims.len());
            if bad > 0 {
                violations.push(format!(
                    "profile {} (base {}): {} exact mismatch(es), {} incredible count(s), {} empty among {} applicable",
                    name, base_count, report.exact_mismatches, report.incredible_counts, report.applicable_empty, report.applicable
                ));
            }
            ctx.ui.log(&format!(
                "predicate-strata: {} (base {}): {} records, {} exact mismatch(es), {} incredible, {} empty among {} applicable, {} empty overall ({:.1} expected), {} outside their band",
                name, base_count, report.records, report.exact_mismatches, report.incredible_counts, report.applicable_empty, report.applicable, report.empties, report.empties_expected, report.out_of_band
            ));
            reports.push(report);
            profile_pb.inc(1);
        }
        profile_pb.finish();

        let report = StrataReport {
            schema_version: 2,
            predicates: predicates.len(),
            query_count,
            census_population: population,
            reliability_threshold: threshold,
            min_matches,
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

// ── Sampling models (TS-173) ────────────────────────────────────────

/// Two-sided tail probability below which a realised count is not
/// credible under its record's sampling model. At `1e-9` a set of ten
/// thousand records checked at fifty profiles produces a false failure
/// once in two thousand builds; a count this far out is an evaluation
/// error, not noise.
pub const CREDIBILITY_ALPHA: f64 = 1e-9;

/// How a record's match count at a profile is distributed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CountModel {
    /// A hash predicate of selectivity `p` over `n` independent rows
    /// (the control family, TS-115).
    Binomial { n: u64, p: f64 },
    /// A censused predicate with `k` matches in a population of `pop`,
    /// seen through the shuffled prefix of `n` rows.
    Hypergeometric { pop: u64, k: u64, n: u64 },
}

impl CountModel {
    /// Expected matches.
    pub fn mean(&self) -> f64 {
        match *self {
            CountModel::Binomial { n, p } => n as f64 * p,
            CountModel::Hypergeometric { pop, k, n } => {
                if pop == 0 { 0.0 } else { n as f64 * k as f64 / pop as f64 }
            }
        }
    }

    /// Standard deviation of the count.
    pub fn sigma(&self) -> f64 {
        match *self {
            CountModel::Binomial { n, p } => (n as f64 * p * (1.0 - p)).max(0.0).sqrt(),
            CountModel::Hypergeometric { pop, k, n } => {
                if pop < 2 {
                    return 0.0;
                }
                let (pop_f, k_f, n_f) = (pop as f64, k as f64, n as f64);
                let p = k_f / pop_f;
                (n_f * p * (1.0 - p) * (pop_f - n_f) / (pop_f - 1.0)).max(0.0).sqrt()
            }
        }
    }

    /// The counts the model can produce at all.
    fn support(&self) -> (u64, u64) {
        match *self {
            CountModel::Binomial { n, .. } => (0, n),
            CountModel::Hypergeometric { pop, k, n } => ((n + k).saturating_sub(pop), k.min(n)),
        }
    }

    /// Natural log of the probability of exactly `x` matches.
    fn log_pmf(&self, x: u64) -> f64 {
        let (lo, hi) = self.support();
        if x < lo || x > hi {
            return f64::NEG_INFINITY;
        }
        match *self {
            CountModel::Binomial { n, p } => {
                if p <= 0.0 {
                    return if x == 0 { 0.0 } else { f64::NEG_INFINITY };
                }
                if p >= 1.0 {
                    return if x == n { 0.0 } else { f64::NEG_INFINITY };
                }
                ln_choose(n, x) + x as f64 * p.ln() + (n - x) as f64 * (-p).ln_1p()
            }
            CountModel::Hypergeometric { pop, k, n } => {
                ln_choose(k, x) + ln_choose(pop - k, n - x) - ln_choose(pop, n)
            }
        }
    }

    /// Probability of no matches at all.
    pub fn p_empty(&self) -> f64 {
        self.log_pmf(0).exp()
    }

    /// Whether `count` lies inside the model's two-sided
    /// `1 − CREDIBILITY_ALPHA` region: the tail beyond it, away from
    /// the mean, holds at least half the alpha. The tail is summed from
    /// `count` outward and stops as soon as it is large enough, so a
    /// count near the mean costs one term.
    pub fn credible(&self, count: u64) -> bool {
        let half = CREDIBILITY_ALPHA / 2.0;
        let (lo, hi) = self.support();
        if count < lo || count > hi {
            return false;
        }
        let downward = (count as f64) <= self.mean();
        let mut acc = 0.0f64;
        let mut i = count;
        loop {
            let term = self.log_pmf(i).exp();
            acc += term;
            if acc >= half {
                return true;
            }
            // Terms fall monotonically away from the mean; once one is
            // negligible against the alpha the rest cannot add it up.
            if term < half * 1e-6 {
                return false;
            }
            if downward {
                if i == lo {
                    return false;
                }
                i -= 1;
            } else {
                if i == hi {
                    return false;
                }
                i += 1;
            }
        }
    }
}

/// `ln C(a, b)`; `-∞` when `b > a`.
fn ln_choose(a: u64, b: u64) -> f64 {
    if b > a {
        return f64::NEG_INFINITY;
    }
    ln_gamma(a as f64 + 1.0) - ln_gamma(b as f64 + 1.0) - ln_gamma((a - b) as f64 + 1.0)
}

/// `ln Γ(x)` for `x > 0` by the Lanczos approximation, accurate to
/// about `2e-10` absolute.
fn ln_gamma(x: f64) -> f64 {
    const COF: [f64; 6] = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut y = x;
    let mut ser = 1.000000000190015;
    for c in COF {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_factorials_and_large_arguments() {
        for (n, f) in [(1u64, 1.0f64), (2, 1.0), (5, 24.0), (10, 362880.0)] {
            assert!((ln_gamma(n as f64).exp() - f).abs() / f < 1e-9, "Γ({})", n);
        }
        // Stirling at 5e8: ln Γ(x) = (x−½) ln x − x + ½ ln 2π + 1/(12x).
        let x = 5e8f64;
        let stirling = (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * x);
        assert!((ln_gamma(x) - stirling).abs() < 1e-4, "{} vs {}", ln_gamma(x), stirling);
    }

    #[test]
    fn pmfs_sum_to_one_and_have_the_right_mean() {
        for model in [
            CountModel::Binomial { n: 300, p: 0.01 },
            CountModel::Binomial { n: 100_000, p: 1e-5 },
            CountModel::Hypergeometric { pop: 6000, k: 60, n: 300 },
            CountModel::Hypergeometric { pop: 6000, k: 5900, n: 300 },
        ] {
            let (lo, hi) = model.support();
            let (mut total, mut mean) = (0.0, 0.0);
            for x in lo..=hi {
                let p = model.log_pmf(x).exp();
                total += p;
                mean += x as f64 * p;
            }
            assert!((total - 1.0).abs() < 1e-9, "{:?} sums to {}", model, total);
            assert!((mean - model.mean()).abs() < 1e-6, "{:?} mean {} vs {}", model, mean, model.mean());
        }
    }

    #[test]
    fn credibility_admits_noise_and_empties_where_expected_and_refuses_the_rest() {
        // One expected match: empty is the likeliest outcome.
        let one = CountModel::Binomial { n: 100_000, p: 1e-5 };
        assert!(one.credible(0) && one.credible(1) && one.credible(6));
        assert!(!one.credible(20));
        // (1 − 1e-5)^100000 against e^-1: the binomial, not its Poisson limit.
        assert!((one.p_empty() - (-1.0f64).exp()).abs() < 1e-5);
        // Twenty expected: empty is still credible; twenty-five: not.
        assert!(CountModel::Binomial { n: 2_000_000, p: 1e-5 }.credible(0));
        assert!(!CountModel::Binomial { n: 2_500_000, p: 1e-5 }.credible(0));
        // Three hundred expected: the half-decade band top (316) is
        // one sigma away and credible; a count twice the mean is not.
        let three_hundred = CountModel::Hypergeometric { pop: 495_930_736, k: 14_877, n: 10_000_000 };
        assert!((three_hundred.mean() - 300.0).abs() < 0.1);
        assert!(three_hundred.credible(359) && three_hundred.credible(240));
        assert!(!three_hundred.credible(600) && !three_hundred.credible(120));
        // A million expected: six sigma either way is the edge.
        let big = CountModel::Binomial { n: 10_000_000, p: 0.1 };
        let sigma = (1e7 * 0.1 * 0.9f64).sqrt();
        assert!(big.credible((1e6 + 5.5 * sigma) as u64) && big.credible((1e6 - 5.5 * sigma) as u64));
        assert!(!big.credible((1e6 + 7.0 * sigma) as u64) && !big.credible((1e6 - 7.0 * sigma) as u64));
        // At the census profile a censused count is exact and the
        // model says so: only the census count itself is possible.
        let exact = CountModel::Hypergeometric { pop: 6000, k: 60, n: 6000 };
        assert!(exact.credible(60) && !exact.credible(59) && !exact.credible(61));
        assert_eq!(exact.sigma(), 0.0);
        assert!((big.sigma() - sigma).abs() < 1e-6);
        assert!((three_hundred.sigma() - (300.0f64 * (1.0 - 3e-5) * (1.0 - 10_000_000.0 / 495_930_736.0)).sqrt()).abs() < 0.01);
        assert!(exact.p_empty() == 0.0);
        assert!(CountModel::Hypergeometric { pop: 6000, k: 0, n: 6000 }.p_empty() == 1.0);
    }
}
