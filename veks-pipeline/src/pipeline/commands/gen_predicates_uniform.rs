// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `generate predicates --strategy uniform` — a predicate set of **one
//! form at one level** (PS-23, PL-2).
//!
//! A system under test declares its indexes against forms, and a set
//! whose every predicate takes the same form — `topic_l3 = ? AND
//! citation_percentile >= ?`, two parts — measures that one index
//! declaration at one selectivity. The form is given as parts, each a
//! field and the access it takes, joined by `+` (a conjunction) or `|`
//! (a disjunction); the level is the selectivity the set is planned
//! for, and every predicate's selectivity is planned into the
//! half-decade band around it. A part's literals come from the survey's
//! census — a value's count, a threshold's prefix sum — and a two-part
//! conjunction whose pair the census tabulated takes its exact count;
//! any other combination is estimated from the parts' marginals under
//! independence, and the record says so. A part may be a no-op that
//! holds the form constant (PS-23); the record says that too.
//!
//! The published record is a PNode. The `families` namespace names the
//! form and the planned selectivity per predicate; the `generation`
//! namespace says where each came from. The set's profile is tagged
//! `family: uniform`, `predicates: uniform-<n>`, `form`, `form_shape`
//! and `selectivity` (PS-11, PS-12).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use indexmap::IndexMap;
use rand::seq::{IndexedRandom, SliceRandom};
use serde::{Deserialize, Serialize};
use slabtastic::{SlabReader, SlabWriter, WriterConfig};

use vectordata::metadata_schema::{
    FAMILIES_NAMESPACE, GENERATION_NAMESPACE, PredicateSchema, SCHEMA_NAMESPACE, SURVEY_NAMESPACE,
};
use veks_core::formats::anode::{self, ANode};
use veks_core::formats::mnode::{MNode, MValue};
use veks_core::formats::pnode::{
    Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode,
};

use crate::pipeline::command::{CommandResult, OptionDesc, OptionRole, Options, Status, StreamContext};
use crate::pipeline::commands::survey::{HierarchyNode, MeasureReport, SurveyReport};
use crate::pipeline::rng;

use super::gen_predicates::comparand_from_key;
use super::gen_predicates_common::{error_result, opt, resolve_path};
use super::slab::survey_report_from_json;

/// Half-decade band factor: a level `s` admits `[s/√10, s·√10)`, the
/// same tiling the stratified strategy uses.
const BAND: f64 = 3.162_277_660_168_379_5;

// ---------------------------------------------------------------------------
// Forms
// ---------------------------------------------------------------------------

/// The access a part takes: what an index serves. A form holds one
/// operator per part, so a range is a lower bound (`range`, `>=`) or
/// an upper one (`le`, `<=`), never a mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Access {
    Eq,
    Ge,
    Le,
}

impl Access {
    pub fn label(self) -> &'static str {
        match self {
            Access::Eq => "eq",
            Access::Ge => "range",
            Access::Le => "le",
        }
    }

    fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "eq" | "equality" => Ok(Access::Eq),
            "range" | "ge" => Ok(Access::Ge),
            "le" => Ok(Access::Le),
            other => Err(format!("form: unknown access `{other}`; a part is `field.eq`, `field.range` (a lower bound) or `field.le`")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Junction {
    And,
    Or,
}

/// One part of a form: a field and the access it takes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormPart {
    pub field: String,
    pub access: Access,
}

/// A form: parts under one junction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Form {
    pub parts: Vec<FormPart>,
    pub junction: Junction,
}

impl Form {
    /// `field.access` parts joined by `+` for a conjunction or `|` for
    /// a disjunction; one part needs no junction.
    pub fn parse(spec: &str) -> Result<Form, String> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Err("form: empty".to_string());
        }
        let (junction, sep) = match (spec.contains('+'), spec.contains('|')) {
            (true, true) => return Err("form: one junction per form, `+` or `|`".to_string()),
            (false, true) => (Junction::Or, '|'),
            _ => (Junction::And, '+'),
        };
        let mut parts = Vec::new();
        for raw in spec.split(sep) {
            let raw = raw.trim();
            let Some((field, access)) = raw.rsplit_once('.') else {
                return Err(format!("form: part `{raw}` is not `field.access`"));
            };
            if field.is_empty() {
                return Err(format!("form: part `{raw}` names no field"));
            }
            parts.push(FormPart { field: field.to_string(), access: Access::parse(access)? });
        }
        Ok(Form { parts, junction })
    }

    /// The derived id a tag carries (PS-11): the parts in canonical
    /// order, `field.access` joined by `_`.
    pub fn id(&self) -> String {
        let mut parts: Vec<String> = self.parts.iter().map(|p| format!("{}.{}", p.field, p.access.label())).collect();
        parts.sort();
        parts.join("_")
    }

    /// The parts of the junction a census counts (PS-23).
    pub fn arity(&self) -> usize {
        self.parts.len()
    }
}

// ---------------------------------------------------------------------------
// Leaves: what a part may take, with its census count
// ---------------------------------------------------------------------------

/// One literal a part may take, with its exact count over the census.
#[derive(Debug, Clone)]
pub struct Leaf {
    pub pnode: PNode,
    pub key: String,
    pub count: u64,
    pub selectivity: f64,
    /// Matches everything: holds the form constant without filtering.
    pub noop: bool,
}

fn eq(field: &str, c: Comparand) -> PNode {
    PNode::Predicate(PredicateNode {
        field: FieldRef::Named(field.to_string()),
        op: OpType::Eq,
        comparands: vec![c],
    })
}

fn cmp(field: &str, op: OpType, v: i64) -> PNode {
    PNode::Predicate(PredicateNode {
        field: FieldRef::Named(field.to_string()),
        op,
        comparands: vec![Comparand::Int(v)],
    })
}

fn junction_node(junction: Junction, children: Vec<PNode>) -> PNode {
    if children.len() == 1 {
        return children.into_iter().next().unwrap();
    }
    PNode::Conjugate(ConjugateNode {
        conjugate_type: match junction {
            Junction::And => ConjugateType::And,
            Junction::Or => ConjugateType::Or,
        },
        children,
    })
}

fn hierarchy_leaves(nodes: &[HierarchyNode], fields: &[String], depth: usize, field: &str, n: f64, out: &mut Vec<Leaf>) {
    for node in nodes {
        if fields.get(depth).map(String::as_str) == Some(field) && node.count > 0 {
            out.push(Leaf {
                pnode: eq(field, comparand_from_key(&node.value)),
                key: node.value.clone(),
                count: node.count,
                selectivity: node.count as f64 / n,
                noop: false,
            });
        }
        if depth + 1 < fields.len() {
            hierarchy_leaves(&node.children, fields, depth + 1, field, n, out);
        }
    }
}

/// The leaves a part may take, from the census.
fn part_leaves(survey: &SurveyReport, part: &FormPart, n: f64) -> Result<Vec<Leaf>, String> {
    let mut out: Vec<Leaf> = Vec::new();
    match part.access {
        Access::Eq => {
            if let Some(profile) = survey.fields.get(&part.field)
                && let Some(MeasureReport::ExactValueCensus(c)) = profile.measures.get("ExactValueCensus")
            {
                for (key, count) in &c.counts {
                    if *count > 0 {
                        out.push(Leaf {
                            pnode: eq(&part.field, comparand_from_key(key)),
                            key: key.clone(),
                            count: *count,
                            selectivity: *count as f64 / n,
                            noop: false,
                        });
                    }
                }
            }
            if out.is_empty() {
                for h in &survey.hierarchies {
                    if h.fields.iter().any(|f| f == &part.field) {
                        hierarchy_leaves(&h.nodes, &h.fields, 0, &part.field, n, &mut out);
                    }
                }
            }
            if out.is_empty() {
                return Err(format!(
                    "the survey holds no value census for `{}`; list it under `census` (or a hierarchy) on the survey step",
                    part.field
                ));
            }
        }
        Access::Ge | Access::Le => {
            let Some(profile) = survey.fields.get(&part.field) else {
                return Err(format!("the survey has no field `{}`", part.field));
            };
            let Some(MeasureReport::ExactIntegerHistogram(h)) = profile.measures.get("ExactIntegerHistogram") else {
                return Err(format!(
                    "the survey holds no integer histogram for `{}`; a range part needs a censused integer field",
                    part.field
                ));
            };
            let mut prefix: Vec<u64> = Vec::with_capacity(h.counts.len() + 1);
            prefix.push(0);
            for c in &h.counts {
                prefix.push(prefix.last().unwrap() + c);
            }
            let total = *prefix.last().unwrap();
            for (i, c) in h.counts.iter().enumerate() {
                if *c == 0 {
                    continue;
                }
                let v = h.min + i as i64;
                let (op, count) = match part.access {
                    Access::Ge => (OpType::Ge, total - prefix[i]),
                    _ => (OpType::Le, prefix[i + 1]),
                };
                if count == 0 {
                    continue;
                }
                out.push(Leaf {
                    pnode: cmp(&part.field, op, v),
                    key: format!("{}{v}", if op == OpType::Ge { ">=" } else { "<=" }),
                    count,
                    selectivity: count as f64 / n,
                    noop: count == total,
                });
            }
            if out.is_empty() {
                return Err(format!("the histogram of `{}` is empty", part.field));
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

/// One drawn predicate: a leaf per part, in form order.
#[derive(Debug, Clone)]
pub struct Draw {
    pub leaves: Vec<usize>,
    pub selectivity: f64,
    pub count: u64,
    pub exact: bool,
    pub noops: usize,
}

fn combined(junction: Junction, sels: &[f64]) -> f64 {
    match junction {
        Junction::And => sels.iter().product(),
        Junction::Or => 1.0 - sels.iter().map(|s| 1.0 - s).product::<f64>(),
    }
}

/// Draw `count` distinct predicates of the form whose estimated
/// selectivity lies in `[lo, hi)`, under independence of the parts:
/// in a conjunction no part may be more selective than the band's
/// floor and the last part is chosen to land in it; in a disjunction
/// the complements play that role. A no-op part is taken only when no
/// other literal lands. Returns the draws and how many attempts were
/// made; fewer than `count` is a shortfall the caller reports.
pub fn draw_independent(
    parts: &[Vec<Leaf>],
    junction: Junction,
    lo: f64,
    hi: f64,
    count: usize,
    seed: u64,
) -> (Vec<Draw>, usize) {
    let mut rng = rng::seeded_rng(seed);
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let mut draws: Vec<Draw> = Vec::new();
    let attempts_max = count * 64 + 256;
    let mut attempts = 0usize;
    let n = parts.len();
    while draws.len() < count && attempts < attempts_max {
        attempts += 1;
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);
        let mut chosen: Vec<Option<usize>> = vec![None; n];
        let mut acc = 1.0f64; // product of s (And) or of (1-s) (Or)
        let mut ok = true;
        for (k, &pi) in order.iter().enumerate() {
            let last = k + 1 == n;
            let leaves = &parts[pi];
            let admits = |s: f64| -> bool {
                match junction {
                    Junction::And => {
                        if last { s * acc >= lo && s * acc < hi } else { s >= lo && s * acc >= lo }
                    }
                    Junction::Or => {
                        let c = 1.0 - s;
                        let total = 1.0 - c * acc;
                        if last { total >= lo && total < hi } else { total < hi }
                    }
                }
            };
            let real: Vec<usize> = (0..leaves.len()).filter(|&i| !leaves[i].noop && admits(leaves[i].selectivity)).collect();
            let pool: Vec<usize> = if real.is_empty() {
                (0..leaves.len()).filter(|&i| leaves[i].noop && admits(leaves[i].selectivity)).collect()
            } else {
                real
            };
            let Some(&i) = pool.choose(&mut rng) else {
                ok = false;
                break;
            };
            chosen[pi] = Some(i);
            acc *= match junction {
                Junction::And => leaves[i].selectivity,
                Junction::Or => 1.0 - leaves[i].selectivity,
            };
        }
        if !ok {
            continue;
        }
        let leaves: Vec<usize> = chosen.into_iter().map(|c| c.unwrap()).collect();
        if !seen.insert(leaves.clone()) {
            continue;
        }
        let sels: Vec<f64> = leaves.iter().enumerate().map(|(pi, &i)| parts[pi][i].selectivity).collect();
        let s = combined(junction, &sels);
        let noops = leaves.iter().enumerate().filter(|(pi, i)| parts[*pi][**i].noop).count();
        draws.push(Draw { leaves, selectivity: s, count: 0, exact: false, noops });
    }
    (draws, attempts)
}

/// Exact two-part conjunctions from a pair census: for every `a` value
/// the `b` thresholds (a range part) or values (an equality part) with
/// their tabulated counts, admitted when they land in the band.
fn pair_draws(
    survey: &SurveyReport,
    form: &Form,
    n: f64,
    lo: f64,
    hi: f64,
) -> Option<(Vec<Vec<Leaf>>, Vec<Draw>)> {
    if form.junction != Junction::And || form.parts.len() != 2 {
        return None;
    }
    // The census tabulates (a, b) with `a` the equality field.
    let (ai, bi) = match (form.parts[0].access, form.parts[1].access) {
        (Access::Eq, _) => (0usize, 1usize),
        (_, Access::Eq) => (1, 0),
        _ => return None,
    };
    let (a, b) = (&form.parts[ai], &form.parts[bi]);
    let p = survey.pair_census.iter().find(|p| p.a == a.field && p.b == b.field)?;
    let b_ints: Option<Vec<i64>> = p.b_values.iter().map(|k| match comparand_from_key(k) { Comparand::Int(v) => Some(v), _ => None }).collect();
    if b.access == Access::Le || (b.access == Access::Ge && b_ints.is_none()) {
        return None;
    }
    let mut a_leaves: Vec<Leaf> = Vec::new();
    let mut b_leaves: Vec<Leaf> = Vec::new();
    let mut b_index: HashMap<String, usize> = HashMap::new();
    let mut draws: Vec<Draw> = Vec::new();
    let order: Vec<usize> = match &b_ints {
        Some(ints) => {
            let mut idx: Vec<usize> = (0..ints.len()).collect();
            idx.sort_by_key(|&j| ints[j]);
            idx
        }
        None => (0..p.b_values.len()).collect(),
    };
    for (i, key) in p.a_values.iter().enumerate() {
        let row = &p.counts[i];
        let row_total: u64 = row.iter().sum();
        if row_total == 0 {
            continue;
        }
        let a_index = a_leaves.len();
        a_leaves.push(Leaf {
            pnode: eq(&a.field, comparand_from_key(key)),
            key: key.clone(),
            count: row_total,
            selectivity: row_total as f64 / n,
            noop: false,
        });
        let mut pairs: Vec<(String, PNode, u64)> = Vec::new();
        match (&b_ints, b.access) {
            (Some(ints), Access::Ge) => {
                let mut suffix = 0u64;
                for &j in order.iter().rev() {
                    suffix += row[j];
                    if suffix > 0 {
                        pairs.push((format!(">={}", ints[j]), cmp(&b.field, OpType::Ge, ints[j]), suffix));
                    }
                }
            }
            _ => {
                for (j, bkey) in p.b_values.iter().enumerate() {
                    if row[j] > 0 {
                        pairs.push((bkey.clone(), eq(&b.field, comparand_from_key(bkey)), row[j]));
                    }
                }
            }
        }
        for (bkey, pnode, count) in pairs {
            let s = count as f64 / n;
            if s < lo || s >= hi {
                continue;
            }
            let b_i = *b_index.entry(bkey.clone()).or_insert_with(|| {
                b_leaves.push(Leaf { pnode: pnode.clone(), key: bkey.clone(), count, selectivity: s, noop: false });
                b_leaves.len() - 1
            });
            let mut leaves = vec![0usize; 2];
            leaves[ai] = a_index;
            leaves[bi] = b_i;
            draws.push(Draw { leaves, selectivity: s, count, exact: true, noops: 0 });
        }
    }
    let mut parts: Vec<Vec<Leaf>> = vec![Vec::new(), Vec::new()];
    parts[ai] = a_leaves;
    parts[bi] = b_leaves;
    Some((parts, draws))
}

// ---------------------------------------------------------------------------
// The command
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartReport {
    pub field: String,
    pub access: Access,
    pub candidates: usize,
    pub noops: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniformReport {
    pub schema_version: u32,
    pub seed: u64,
    pub form: String,
    pub form_id: String,
    pub form_shape: String,
    pub junction: Junction,
    pub parts: Vec<PartReport>,
    /// The level as planned, spelled as given.
    pub level: String,
    pub level_value: f64,
    pub band: (f64, f64),
    /// Records written, one per query ordinal.
    pub predicates: usize,
    pub distinct: usize,
    /// Counts read from a pair census rather than estimated.
    pub exact: bool,
    pub shortfall: usize,
    pub attempts: usize,
    /// Records with at least one no-op part.
    pub with_noop: usize,
    pub census_population: u64,
}

pub fn describe_options() -> Vec<OptionDesc> {
    vec![
        opt(
            "form",
            "string",
            false,
            None,
            "uniform: the one form every predicate takes — `field.access` parts joined by `+` (conjunction) or `|` (disjunction), e.g. `topic_l3.eq+citation_percentile.range`",
            OptionRole::Config,
        ),
        opt(
            "selectivity",
            "string",
            false,
            None,
            "uniform: the level the set is planned for, e.g. `1e-2`; every predicate lands in the half-decade band around it",
            OptionRole::Config,
        ),
        opt(
            "band",
            "float",
            false,
            Some("3.1623"),
            "uniform: the band factor — a level s admits [s/band, s·band)",
            OptionRole::Config,
        ),
    ]
}

/// Whether a written uniform facet is complete: every record it was
/// asked for is present.
pub fn check_artifact(output: &Path, options: &Options) -> Option<bool> {
    let want: u64 = options.get("count")?.parse().ok()?;
    let reader = SlabReader::open(output).ok()?;
    Some(reader.total_records() == want)
}

pub(super) fn run(
    options: &Options,
    ctx: &mut StreamContext,
    start: Instant,
    output_path: &Path,
    survey_path: Option<&Path>,
    seed: u64,
) -> CommandResult {
    let Some(survey_path) = survey_path else {
        return error_result("strategy uniform needs --survey: the census tables are the literal pools".into(), start);
    };
    let survey: SurveyReport = match survey_report_from_json(survey_path) {
        Ok(r) => r,
        Err(e) => return error_result(e, start),
    };
    let population = survey.source.total_records;
    if population == 0 {
        return error_result("the survey covers zero records".into(), start);
    }
    let n = population as f64;
    let form_spec = match options.get("form") {
        Some(f) => f.to_string(),
        None => return error_result("strategy uniform needs --form".into(), start),
    };
    let form = match Form::parse(&form_spec) {
        Ok(f) => f,
        Err(e) => return error_result(e, start),
    };
    let level_text = match options.get("selectivity") {
        Some(s) => s.trim().to_string(),
        None => return error_result("strategy uniform needs --selectivity, the level the set is planned for".into(), start),
    };
    let level: f64 = match level_text.parse::<f64>() {
        Ok(v) if v > 0.0 && v < 1.0 => v,
        _ => return error_result(format!("selectivity `{level_text}` is not a fraction in (0, 1)"), start),
    };
    let band = match options.parse_or::<f64>("band", BAND) {
        Ok(b) if b > 1.0 => b,
        Ok(_) => return error_result("band must exceed 1".into(), start),
        Err(e) => return error_result(e, start),
    };
    let (lo, hi) = (level / band, level * band);
    let count = match options.parse_opt::<usize>("count") {
        Ok(Some(c)) if c > 0 => c,
        Ok(_) => return error_result("uniform writes one predicate per query ordinal: give `count`".into(), start),
        Err(e) => return error_result(e, start),
    };
    let report_path = match options.get("report") {
        Some(s) => resolve_path(s, &ctx.workspace),
        None => output_path.with_extension("json"),
    };

    // Exact pairs where the census tabulated them; independence otherwise.
    let (parts, mut draws, exact, attempts) = match pair_draws(&survey, &form, n, lo, hi) {
        Some((parts, pool)) if !pool.is_empty() => {
            let mut rng = rng::seeded_rng(seed);
            let mut pool = pool;
            pool.shuffle(&mut rng);
            pool.truncate(count);
            (parts, pool, true, 0usize)
        }
        _ => {
            let mut parts: Vec<Vec<Leaf>> = Vec::new();
            for part in &form.parts {
                match part_leaves(&survey, part, n) {
                    Ok(l) => parts.push(l),
                    Err(e) => return error_result(e, start),
                }
            }
            let (draws, attempts) = draw_independent(&parts, form.junction, lo, hi, count, seed);
            (parts, draws, false, attempts)
        }
    };
    for d in draws.iter_mut() {
        if !exact {
            d.count = (d.selectivity * n).round() as u64;
        }
    }
    let distinct = draws.len();
    if distinct == 0 {
        return error_result(
            format!(
                "no predicate of form `{form_spec}` lands in [{lo:.3e}, {hi:.3e}); the census offers {} candidate(s) per part",
                parts.iter().map(|p| p.len().to_string()).collect::<Vec<_>>().join("/")
            ),
            start,
        );
    }
    let shortfall = count.saturating_sub(distinct);
    if shortfall > 0 {
        ctx.ui.log(&format!(
            "uniform: {distinct} distinct predicate(s) land in the band; {shortfall} of {count} slot(s) repeat one"
        ));
    }
    // Record i is query i's predicate; the distinct draws repeat only
    // when the band holds fewer than the slots.
    let records: Vec<(usize, &Draw)> = (0..count).map(|i| (i % distinct, &draws[i % distinct])).collect();
    let pnode_of = |d: &Draw| -> PNode {
        junction_node(form.junction, d.leaves.iter().enumerate().map(|(pi, &i)| parts[pi][i].pnode.clone()).collect())
    };
    let form_shape = super::analyze_predicate_forms::form_of(&pnode_of(&draws[0]));
    let form_id = form.id();
    let with_noop = records.iter().filter(|(_, d)| d.noops > 0).count();
    ctx.ui.log(&format!(
        "uniform: {count} predicates of form {} at {level_text} ({}), {distinct} distinct{}; population {population}",
        form_shape,
        if exact { "exact pair counts" } else { "independence estimate" },
        if with_noop > 0 { format!(", {with_noop} with a no-op part") } else { String::new() },
    ));

    if let Some(parent) = output_path.parent()
        && !parent.exists()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return error_result(format!("failed to create {}: {}", parent.display(), e), start);
    }
    let config = match WriterConfig::new(512, 4096, u32::MAX, false) {
        Ok(c) => c,
        Err(e) => return error_result(format!("writer config error: {}", e), start),
    };
    let mut writer = match SlabWriter::new(output_path, config) {
        Ok(w) => w,
        Err(e) => return error_result(format!("failed to create output: {}", e), start),
    };
    for (_, d) in &records {
        if let Err(e) = writer.add_record(&pnode_of(d).to_bytes_named()) {
            return error_result(format!("write error: {}", e), start);
        }
    }
    let schema = PredicateSchema::new("<uniform>", level_text.clone(), seed, count as u64);
    let families: Vec<Vec<u8>> = records
        .iter()
        .map(|(i, d)| {
            let mut fields = IndexMap::new();
            fields.insert("family".to_string(), MValue::Text("uniform".to_string()));
            fields.insert("form".to_string(), MValue::Text(form_id.clone()));
            fields.insert("selectivity".to_string(), MValue::Float(d.selectivity));
            fields.insert("estimated".to_string(), MValue::Bool(!d.exact));
            fields.insert("noop_parts".to_string(), MValue::Int(d.noops as i64));
            fields.insert("predicate".to_string(), MValue::Int(*i as i64));
            anode::encode(&ANode::MNode(MNode { fields }))
        })
        .collect();
    let generation: Vec<Vec<u8>> = records
        .iter()
        .map(|(_, d)| {
            let mut fields = IndexMap::new();
            fields.insert("form".to_string(), MValue::Text(form_id.clone()));
            fields.insert("level".to_string(), MValue::Text(level_text.clone()));
            fields.insert("expected_count".to_string(), MValue::Int(d.count as i64));
            fields.insert(
                "source".to_string(),
                MValue::Text(if d.exact { "census:pair" } else { "census:independence" }.to_string()),
            );
            fields.insert("vernacular".to_string(), MValue::Text(format!("{}", pnode_of(d))));
            anode::encode(&ANode::MNode(MNode { fields }))
        })
        .collect();
    let survey_bytes = match serde_json::to_vec(&survey) {
        Ok(v) => v,
        Err(e) => return error_result(format!("serialise survey report: {e}"), start),
    };
    let sections: [(&str, Vec<Vec<u8>>); 4] = [
        (SCHEMA_NAMESPACE, vec![schema.to_json_bytes()]),
        (SURVEY_NAMESPACE, vec![survey_bytes]),
        (FAMILIES_NAMESPACE, families),
        (GENERATION_NAMESPACE, generation),
    ];
    for (name, recs) in sections {
        if let Err(e) = writer.start_namespace(name) {
            return error_result(format!("{} namespace: {}", name, e), start);
        }
        for r in recs {
            if let Err(e) = writer.add_record(&r) {
                return error_result(format!("{} write: {}", name, e), start);
            }
        }
    }
    if let Err(e) = writer.finish() {
        return error_result(format!("finish error: {}", e), start);
    }

    // The class of what was written, held to the census the check
    // will run (PS-23), and the tags of the set it fills (PS-12).
    let classes = match super::analyze_predicate_forms::facet_form_classes(output_path) {
        Ok(c) => c,
        Err(e) => return error_result(format!("census of the written facet: {e}"), start),
    };
    if classes.forms != 1 || classes.parts != Some(form.arity()) {
        return error_result(
            format!(
                "the written facet holds {} form(s) of {:?} part(s), not one form of {} (PS-23)",
                classes.forms,
                classes.parts,
                form.arity()
            ),
            start,
        );
    }
    for profile in crate::pipeline::dataset_lookup::profiles_declaring_facet(&ctx.workspace, "metadata_predicates", output_path) {
        for (key, value) in [
            ("family", serde_yaml::Value::from("uniform")),
            ("predicates", serde_yaml::Value::from(format!("uniform-{}", form.arity()))),
            ("form", serde_yaml::Value::from(form_id.clone())),
            ("form_shape", serde_yaml::Value::from(form_shape.clone())),
            ("selectivity", serde_yaml::Value::from(level_text.clone())),
            ("forms", serde_yaml::Value::from(1u64)),
        ] {
            ctx.attributes.push(crate::pipeline::command::AttributeWrite { profile: profile.clone(), key: key.to_string(), value });
        }
    }
    let var_name = format!("verified_count:{}", output_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"));
    let _ = crate::pipeline::variables::set_and_save(&ctx.workspace, &var_name, &count.to_string());
    ctx.defaults.insert(var_name, count.to_string());

    let report = UniformReport {
        schema_version: 1,
        seed,
        form: form_spec.clone(),
        form_id: form_id.clone(),
        form_shape: form_shape.clone(),
        junction: form.junction,
        parts: form
            .parts
            .iter()
            .zip(parts.iter())
            .map(|(p, leaves)| PartReport {
                field: p.field.clone(),
                access: p.access,
                candidates: leaves.len(),
                noops: leaves.iter().filter(|l| l.noop).count(),
            })
            .collect(),
        level: level_text.clone(),
        level_value: level,
        band: (lo, hi),
        predicates: count,
        distinct,
        exact,
        shortfall,
        attempts,
        with_noop,
        census_population: population,
    };
    if let Err(e) = serde_json::to_string_pretty(&report)
        .map_err(|e| e.to_string())
        .and_then(|s| std::fs::write(&report_path, s).map_err(|e| e.to_string()))
    {
        return error_result(format!("write report {}: {e}", report_path.display()), start);
    }
    CommandResult {
        status: Status::Ok,
        message: format!(
            "{count} predicates of {form_shape} at {level_text}: {distinct} distinct, {}",
            if exact { "exact" } else { "estimated" }
        ),
        produced: vec![output_path.to_path_buf(), report_path],
        elapsed: start.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A form is parts under one junction, and its id is canonical**
    /// (PS-11): order-free, joined by `_`.
    #[test]
    fn a_form_parses_and_derives_its_id() {
        let f = Form::parse("topic_l3.eq+citation_percentile.range").unwrap();
        assert_eq!(f.junction, Junction::And);
        assert_eq!(f.arity(), 2);
        assert_eq!(f.id(), "citation_percentile.range_topic_l3.eq");
        assert_eq!(Form::parse("citation_percentile.range + topic_l3.eq").unwrap().id(), f.id());
        let o = Form::parse("year.range|isopenaccess.eq").unwrap();
        assert_eq!(o.junction, Junction::Or);
        assert_eq!(Form::parse("year.range").unwrap().arity(), 1);
        assert!(Form::parse("a.eq+b.eq|c.eq").is_err());
        assert!(Form::parse("topic").is_err());
        assert!(Form::parse("topic.like").is_err());
    }

    fn leaves(sels: &[f64], noop: Option<f64>) -> Vec<Leaf> {
        let mut out: Vec<Leaf> = sels
            .iter()
            .enumerate()
            .map(|(i, s)| Leaf { pnode: cmp("f", OpType::Ge, i as i64), key: i.to_string(), count: 0, selectivity: *s, noop: false })
            .collect();
        if let Some(s) = noop {
            out.push(Leaf { pnode: cmp("f", OpType::Ge, -1), key: "noop".into(), count: 0, selectivity: s, noop: true });
        }
        out
    }

    /// **Every draw lands in the band, draws are distinct, and a no-op
    /// part is taken only when nothing else lands** (PS-23).
    #[test]
    fn draws_land_in_the_band_and_are_distinct() {
        let a = leaves(&[0.5, 0.2, 0.1, 0.05, 0.02], None);
        let b = leaves(&[0.6, 0.3, 0.15, 0.08, 0.04], Some(1.0));
        let (lo, hi) = (0.01 / BAND, 0.01 * BAND);
        let (draws, attempts) = draw_independent(&[a.clone(), b.clone()], Junction::And, lo, hi, 6, 7);
        assert!(!draws.is_empty() && attempts > 0);
        let mut seen = HashSet::new();
        for d in &draws {
            assert!(d.selectivity >= lo && d.selectivity < hi, "{d:?}");
            assert!(seen.insert(d.leaves.clone()));
            assert_eq!(d.noops, 0, "real literals land, so no no-op is taken: {d:?}");
        }
        // A level only a no-op can reach: the second part must be the no-op.
        let (draws, _) = draw_independent(&[leaves(&[0.5, 0.2], None), leaves(&[0.001], Some(1.0))], Junction::And, 0.2 / BAND, 0.2 * BAND, 2, 3);
        assert!(!draws.is_empty());
        assert!(draws.iter().all(|d| d.noops == 1), "{draws:?}");
        // A disjunction grows: parts each below the band combine into it.
        let (draws, _) = draw_independent(&[leaves(&[0.05, 0.02], None), leaves(&[0.06, 0.03], None)], Junction::Or, 0.1 / BAND, 0.1 * BAND, 3, 11);
        assert!(!draws.is_empty());
        for d in &draws {
            assert!(d.selectivity >= 0.1 / BAND && d.selectivity < 0.1 * BAND, "{d:?}");
        }
        // Determinism: the same seed draws the same set.
        let (x, _) = draw_independent(&[a.clone(), b.clone()], Junction::And, lo, hi, 4, 99);
        let (y, _) = draw_independent(&[a, b], Junction::And, lo, hi, 4, 99);
        assert_eq!(x.iter().map(|d| d.leaves.clone()).collect::<Vec<_>>(), y.iter().map(|d| d.leaves.clone()).collect::<Vec<_>>());
    }
}
