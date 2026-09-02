// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `generate predicates --strategy stratified` — a predicate set with
//! known selectivity in every decade, drawn from the census.
//!
//! The existing strategies target a selectivity and hope. This one
//! never estimates: every candidate's selectivity is read from the
//! survey's census tables — a topic's node in the hierarchy tree, a
//! value's count, a range's prefix-sum over a dense histogram, a
//! conjunction's cell in a joint table — and a candidate is admitted to
//! whichever half-decade band its measured selectivity lands in. The
//! sampler then draws, seeded, a configured number of predicates from
//! every `(family, decade)` cell and records what it drew and what it
//! could not.
//!
//! Four families, spanning how strongly a filter correlates with the
//! query: **topical** (a topic at any level, or a topic conjoined with
//! one bibliographic qualifier to reach a lower decade), **structural**
//! (passage-level fields, free of paper blocking), **bibliographic**
//! (paper-level fields, the realistic majority) and **control** (a
//! seeded hash range, the null hypothesis — the only family that can
//! fill any cell on demand, and the only one that reads the hash).
//!
//! The published predicate record is an ordinary PNode and nothing
//! else. What a predicate *is* goes into the `families` namespace of
//! the same slab, one record per predicate ordinal; why it was
//! *selected* goes into the `generation` namespace. Absence of either
//! namespace means one unlabelled family, as the `forms` precedent
//! establishes.
//!
//! See the topic-stratified predicate SRD, §3, §4.4–4.5, §6.4, §10.4
//! and §11.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use indexmap::IndexMap;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use slabtastic::{SlabReader, SlabWriter, WriterConfig};

use vectordata::io::{VectorReader, XvecReader};
use vectordata::metadata_schema::{
    FAMILIES_NAMESPACE, GENERATION_NAMESPACE, PredicateSchema, SCHEMA_NAMESPACE, SURVEY_NAMESPACE,
};
use veks_core::formats::anode::{self, ANode};
use veks_core::formats::mnode::{MNode, MValue};
use veks_core::formats::pnode::{
    Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode,
};

use crate::pipeline::command::{
    CommandResult, OptionDesc, OptionRole, Options, Status, StreamContext,
};
use crate::pipeline::commands::survey::{HierarchyNode, MeasureReport, SurveyReport};
use crate::pipeline::rng;

use super::compute_knn_stdarch::select_dot_fn;
use super::compute_topic_labels::read_labels;
use super::compute_topics::{LevelModel, TopicModel, TopicModelReport};
use super::gen_predicates::comparand_from_key;
use super::gen_predicates_common::{error_result, opt, resolve_path};
use super::slab::survey_report_from_json;

/// Range widths tried for integer fields, in field units. Only widths
/// narrower than the field's span are used.
const RANGE_WIDTHS: [i64; 5] = [2, 5, 10, 20, 50];

/// Default per-decade counts, decade 10⁻¹ first: the taper of TS-54.
const TAPERED: [usize; 3] = [10, 20, 50];

/// Default modulus of the control field.
const DEFAULT_BUCKETS: u64 = 16_777_216;

// ---------------------------------------------------------------------------
// Families, decades, configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Family {
    Topical,
    Structural,
    Bibliographic,
    Control,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Topical => "topical",
            Family::Structural => "structural",
            Family::Bibliographic => "bibliographic",
            Family::Control => "control",
        }
    }

    fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "topical" => Ok(Family::Topical),
            "structural" => Ok(Family::Structural),
            "bibliographic" => Ok(Family::Bibliographic),
            "control" => Ok(Family::Control),
            other => Err(format!("families: unknown family `{}`", other)),
        }
    }
}

/// The decade a selectivity belongs to: `⌊log10 s + ½⌋`, so the band
/// `[d/√10, d·√10)` tiles the axis without gaps or overlap and its
/// upper edge belongs to the next decade up.
pub fn decade_of(selectivity: f64) -> Option<i32> {
    if selectivity <= 0.0 || !selectivity.is_finite() {
        return None;
    }
    Some((selectivity.log10() + 0.5).floor() as i32)
}

/// Parse `decades`: `1e-1..1e-7`, or a comma list `1e-1,1e-2`. Returns
/// exponents, descending (coarsest first).
pub fn parse_decades(spec: &str) -> Result<Vec<i32>, String> {
    let exp = |s: &str| -> Result<i32, String> {
        let v: f64 = s
            .trim()
            .parse()
            .map_err(|_| format!("decades: `{}` is not a number", s.trim()))?;
        decade_of(v).ok_or_else(|| format!("decades: `{}` is not a positive selectivity", s.trim()))
    };
    let mut out: Vec<i32> = if let Some((a, b)) = spec.split_once("..") {
        let (hi, lo) = (exp(a)?, exp(b)?);
        let (hi, lo) = (hi.max(lo), hi.min(lo));
        (lo..=hi).rev().collect()
    } else {
        spec.split(',')
            .filter(|s| !s.trim().is_empty())
            .map(exp)
            .collect::<Result<_, _>>()?
    };
    out.sort_unstable_by(|a, b| b.cmp(a));
    out.dedup();
    if out.is_empty() {
        return Err("decades: at least one decade is required".into());
    }
    Ok(out)
}

/// Parse `per-cell`: `tapered`, one count, or one count per decade
/// (coarsest first; the last repeats).
pub fn parse_per_cell(spec: &str, decades: usize) -> Result<Vec<usize>, String> {
    let taper = |list: &[usize]| -> Vec<usize> {
        (0..decades)
            .map(|i| *list.get(i).or(list.last()).unwrap_or(&1))
            .collect()
    };
    if spec.trim().eq_ignore_ascii_case("tapered") {
        return Ok(taper(&TAPERED));
    }
    let list: Vec<usize> = spec
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| format!("per-cell: `{}` is not a count", s.trim()))
        })
        .collect::<Result<_, _>>()?;
    if list.is_empty() || list.contains(&0) {
        return Err("per-cell: counts must be positive".into());
    }
    Ok(taper(&list))
}

fn parse_fields(spec: &str) -> Vec<String> {
    spec.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// How topical cells mix query placement (TS-19).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Placement {
    /// Half in-topic, half out-of-topic, each backfilling the other.
    Mixed,
    InTopic,
    OutOfTopic,
    /// Draw without regard to placement; still labelled.
    Any,
}

impl Placement {
    fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mixed" => Ok(Placement::Mixed),
            "in-topic" => Ok(Placement::InTopic),
            "out-of-topic" => Ok(Placement::OutOfTopic),
            "any" => Ok(Placement::Any),
            other => Err(format!(
                "query-placement: expected mixed, in-topic, out-of-topic or any, got `{}`",
                other
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Candidates
// ---------------------------------------------------------------------------

/// A predicate the sampler may draw, with its exact selectivity.
#[derive(Debug, Clone)]
struct Candidate {
    pnode: PNode,
    family: Family,
    /// Exact matches over the census population.
    count: u64,
    selectivity: f64,
    /// Which census table it came from.
    source: &'static str,
    /// Topical only: the level (1-based) and label of its topic.
    topic: Option<(usize, String)>,
    conjunct: bool,
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

fn and(a: PNode, b: PNode) -> PNode {
    PNode::Conjugate(ConjugateNode {
        conjugate_type: ConjugateType::And,
        children: vec![a, b],
    })
}

fn between(field: &str, lo: i64, hi: i64) -> PNode {
    and(cmp(field, OpType::Ge, lo), cmp(field, OpType::Le, hi))
}

/// The integer a canonical census key denotes, if it is an integer.
fn key_int(key: &str) -> Option<i64> {
    match comparand_from_key(key) {
        Comparand::Int(v) => Some(v),
        _ => None,
    }
}

/// Candidates from one hierarchy tree: every node at every level.
fn topical_candidates(
    nodes: &[HierarchyNode],
    fields: &[String],
    depth: usize,
    n: f64,
    out: &mut Vec<Candidate>,
) {
    for node in nodes {
        let field = &fields[depth];
        let c = comparand_from_key(&node.value);
        let label = match &c {
            Comparand::Text(t) => t.clone(),
            other => format!("{:?}", other),
        };
        out.push(Candidate {
            pnode: eq(field, c),
            family: Family::Topical,
            count: node.count,
            selectivity: node.count as f64 / n,
            source: "census:hierarchy",
            topic: Some((depth + 1, label)),
            conjunct: false,
        });
        if !node.children.is_empty() && depth + 1 < fields.len() {
            topical_candidates(&node.children, fields, depth + 1, n, out);
        }
    }
}

/// Candidates from a dense integer histogram: every threshold in both
/// directions and aligned ranges of a few widths.
fn histogram_candidates(
    field: &str,
    family: Family,
    min: i64,
    counts: &[u64],
    n: f64,
    out: &mut Vec<Candidate>,
) {
    let width = counts.len() as i64;
    if width == 0 {
        return;
    }
    let mut prefix: Vec<u64> = Vec::with_capacity(counts.len() + 1);
    prefix.push(0);
    for c in counts {
        prefix.push(prefix.last().unwrap() + c);
    }
    let total = *prefix.last().unwrap();
    let sum = |lo: i64, hi: i64| -> u64 {
        // inclusive [lo, hi] in field units
        let a = (lo - min).clamp(0, width) as usize;
        let b = (hi - min + 1).clamp(0, width) as usize;
        if b > a { prefix[b] - prefix[a] } else { 0 }
    };
    let push = |out: &mut Vec<Candidate>, pnode: PNode, count: u64| {
        if count > 0 && count < total {
            out.push(Candidate {
                pnode,
                family,
                count,
                selectivity: count as f64 / n,
                source: "census:histogram",
                topic: None,
                conjunct: false,
            });
        }
    };
    for (i, c) in counts.iter().enumerate() {
        if *c == 0 {
            continue;
        }
        let v = min + i as i64;
        push(out, cmp(field, OpType::Ge, v), total - prefix[i]);
        push(out, cmp(field, OpType::Le, v), prefix[i + 1]);
    }
    for w in RANGE_WIDTHS {
        if w >= width {
            continue;
        }
        let mut lo = min;
        while lo + w - 1 < min + width {
            let hi = lo + w - 1;
            push(out, between(field, lo, hi), sum(lo, hi));
            lo += w;
        }
    }
}

/// Candidates from an exact value table: equality with every value.
fn value_candidates(
    field: &str,
    family: Family,
    counts: &IndexMap<String, u64>,
    n: f64,
    out: &mut Vec<Candidate>,
) {
    for (key, count) in counts {
        if *count == 0 {
            continue;
        }
        out.push(Candidate {
            pnode: eq(field, comparand_from_key(key)),
            family,
            count: *count,
            selectivity: *count as f64 / n,
            source: "census:values",
            topic: None,
            conjunct: false,
        });
    }
}

/// Conjunction candidates from one joint table of a topic field
/// against a bibliographic field: `topic = x AND b >= t` for integer
/// `b`, `topic = x AND b = v` otherwise.
fn pair_candidates(
    a: &str,
    b: &str,
    a_values: &[String],
    b_values: &[String],
    counts: &[Vec<u64>],
    topic_level: usize,
    n: f64,
    out: &mut Vec<Candidate>,
) {
    let b_ints: Option<Vec<i64>> = b_values.iter().map(|k| key_int(k)).collect();
    // Column order by numeric value, for suffix sums.
    let order: Vec<usize> = match &b_ints {
        Some(ints) => {
            let mut idx: Vec<usize> = (0..ints.len()).collect();
            idx.sort_by_key(|&j| ints[j]);
            idx
        }
        None => (0..b_values.len()).collect(),
    };
    for (i, key) in a_values.iter().enumerate() {
        let topic_c = comparand_from_key(key);
        let label = match &topic_c {
            Comparand::Text(t) => t.clone(),
            other => format!("{:?}", other),
        };
        let row = &counts[i];
        let row_total: u64 = row.iter().sum();
        if row_total == 0 {
            continue;
        }
        match &b_ints {
            Some(ints) => {
                // Suffix sums over the numeric order.
                let mut suffix = 0u64;
                let mut thresholds: Vec<(i64, u64)> = Vec::with_capacity(order.len());
                for &j in order.iter().rev() {
                    suffix += row[j];
                    thresholds.push((ints[j], suffix));
                }
                for (t, count) in thresholds {
                    if count == 0 || count == row_total {
                        continue;
                    }
                    out.push(Candidate {
                        pnode: and(eq(a, topic_c.clone()), cmp(b, OpType::Ge, t)),
                        family: Family::Topical,
                        count,
                        selectivity: count as f64 / n,
                        source: "census:pair",
                        topic: Some((topic_level, label.clone())),
                        conjunct: true,
                    });
                }
            }
            None => {
                for (j, bkey) in b_values.iter().enumerate() {
                    let count = row[j];
                    if count == 0 || count == row_total {
                        continue;
                    }
                    out.push(Candidate {
                        pnode: and(eq(a, topic_c.clone()), eq(b, comparand_from_key(bkey))),
                        family: Family::Topical,
                        count,
                        selectivity: count as f64 / n,
                        source: "census:pair",
                        topic: Some((topic_level, label.clone())),
                        conjunct: true,
                    });
                }
            }
        }
    }
}

/// Control predicates for one cell: `c` disjoint ranges of the hash of
/// width `round(d · K)`. Selectivity is exact by construction.
fn control_candidates(field: &str, decade: i32, c: usize, buckets: u64, n: f64) -> Vec<Candidate> {
    let d = 10f64.powi(decade);
    let w = ((d * buckets as f64).round() as u64).max(1);
    let mut out = Vec::new();
    for i in 0..c as u64 {
        let lo = i * w;
        let hi = lo + w - 1;
        if hi >= buckets {
            break;
        }
        let count = (w as f64 / buckets as f64 * n).round() as u64;
        out.push(Candidate {
            pnode: between(field, lo as i64, hi as i64),
            family: Family::Control,
            count,
            selectivity: w as f64 / buckets as f64,
            source: "control",
            topic: None,
            conjunct: false,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Query placement
// ---------------------------------------------------------------------------

/// The `(level, label)` topics the query set falls in.
fn query_topics(
    queries: &Path,
    centroids: &Path,
    model: &Path,
    labels: &Path,
) -> Result<(HashSet<(usize, String)>, usize), String> {
    let report: TopicModelReport = serde_json::from_str(
        &std::fs::read_to_string(model)
            .map_err(|e| format!("failed to read {}: {}", model.display(), e))?,
    )
    .map_err(|e| format!("model report {} does not parse: {}", model.display(), e))?;
    let cents = XvecReader::<f32>::open_path(centroids)
        .map_err(|e| format!("failed to open centroids {}: {}", centroids.display(), e))?;
    let dim = cents.dim();
    let mut levels = Vec::with_capacity(report.levels.len());
    let mut offset = 0usize;
    let mut clusters = 1usize;
    for &k in &report.levels {
        clusters *= k;
        let mut data = Vec::with_capacity(clusters * dim);
        for c in 0..clusters {
            data.extend_from_slice(
                &cents
                    .get(offset + c)
                    .map_err(|e| format!("centroid {}: {}", offset + c, e))?,
            );
        }
        offset += clusters;
        levels.push(LevelModel {
            branching: k,
            clusters,
            centroids: data,
            empty: vec![false; clusters],
            runs: vec![],
        });
    }
    if offset != cents.count() {
        return Err(format!(
            "centroid file holds {} records but the model report implies {}",
            cents.count(),
            offset
        ));
    }
    let topic_model = TopicModel { dim, levels };
    let mut label_table: Vec<Vec<String>> = vec![Vec::new(); report.levels.len()];
    for (level, code, label) in read_labels(labels)? {
        if level == 0 || level > label_table.len() {
            continue;
        }
        let v = &mut label_table[level - 1];
        if v.len() <= code {
            v.resize(code + 1, String::new());
        }
        v[code] = label;
    }
    let q = XvecReader::<f32>::open_path(queries)
        .map_err(|e| format!("failed to open queries {}: {}", queries.display(), e))?;
    if q.dim() != dim {
        return Err(format!(
            "queries have dim {} but centroids have dim {}",
            q.dim(),
            dim
        ));
    }
    let (dot, _) = select_dot_fn();
    let mut codes = vec![0u16; report.levels.len()];
    let mut topics = HashSet::new();
    for i in 0..q.count() {
        let mut v = q.get(i).map_err(|e| format!("query {}: {}", i, e))?;
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        topic_model.descend(&v, dot, &mut codes);
        for (l, &code) in codes.iter().enumerate() {
            let label = label_table[l]
                .get(code as usize)
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("l{}-{:05}", l + 1, code));
            topics.insert((l + 1, label));
        }
    }
    Ok((topics, q.count()))
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

/// What one cell did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellReport {
    pub family: Family,
    pub decade: i32,
    pub target: usize,
    pub candidates: usize,
    pub drawn: usize,
    pub shortfall: usize,
    pub conjunctions: usize,
    pub in_topic: usize,
    pub out_of_topic: usize,
}

/// A drawn predicate with everything both namespaces record.
struct Drawn {
    candidate: Candidate,
    cell: String,
    pool: usize,
    placement: Option<&'static str>,
}

fn cell_seed(seed: u64, family: Family, decade: i32) -> u64 {
    let mut z = seed
        .wrapping_add((family as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((decade as i64 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Draw one cell. Single-field candidates first, conjunctions only to
/// make up a shortfall (TS-116); topical cells honour the placement
/// mix when query topics are known.
fn draw_cell(
    family: Family,
    decade: i32,
    target: usize,
    pool: &[Candidate],
    seed: u64,
    placement: Placement,
    query_topics: Option<&HashSet<(usize, String)>>,
) -> (Vec<Drawn>, CellReport) {
    let mut rng = rng::seeded_rng(cell_seed(seed, family, decade));
    let cell = format!("{}:1e{}", family.as_str(), decade);
    let in_topic = |c: &Candidate| -> Option<bool> {
        let qt = query_topics?;
        c.topic.as_ref().map(|t| qt.contains(t))
    };
    let mut singles: Vec<&Candidate> = pool.iter().filter(|c| !c.conjunct).collect();
    let mut conjuncts: Vec<&Candidate> = pool.iter().filter(|c| c.conjunct).collect();
    singles.shuffle(&mut rng);
    conjuncts.shuffle(&mut rng);
    let mut take: Vec<&Candidate> = Vec::with_capacity(target);
    let use_placement =
        family == Family::Topical && query_topics.is_some() && placement != Placement::Any;
    if use_placement {
        let mut ordered: Vec<&Candidate> = singles
            .iter()
            .copied()
            .chain(conjuncts.iter().copied())
            .collect();
        // Stable partition by placement, singles before conjunctions
        // within each side.
        let ins: Vec<&Candidate> = ordered
            .iter()
            .copied()
            .filter(|c| in_topic(c) == Some(true))
            .collect();
        let outs: Vec<&Candidate> = ordered
            .iter()
            .copied()
            .filter(|c| in_topic(c) == Some(false))
            .collect();
        ordered.clear();
        let (want_in, want_out) = match placement {
            Placement::InTopic => (target, 0),
            Placement::OutOfTopic => (0, target),
            _ => (target.div_ceil(2), target / 2),
        };
        let got_in = want_in.min(ins.len());
        let got_out = want_out.min(outs.len());
        take.extend(ins.iter().take(got_in));
        take.extend(outs.iter().take(got_out));
        // Backfill from whichever side has more, up to the target.
        let mut extra_in = ins.iter().skip(got_in);
        let mut extra_out = outs.iter().skip(got_out);
        while take.len() < target {
            match (extra_in.next(), extra_out.next()) {
                (Some(c), _) if placement != Placement::OutOfTopic => take.push(c),
                (_, Some(c)) if placement != Placement::InTopic => take.push(c),
                (Some(c), None) | (None, Some(c)) => take.push(c),
                (None, None) => break,
                _ => break,
            }
        }
    } else {
        take.extend(singles.iter().take(target));
        if take.len() < target {
            let need = target - take.len();
            take.extend(conjuncts.iter().take(need));
        }
    }
    let drawn: Vec<Drawn> = take
        .iter()
        .map(|c| Drawn {
            candidate: (*c).clone(),
            cell: cell.clone(),
            pool: pool.len(),
            placement: match in_topic(c) {
                Some(true) => Some("in-topic"),
                Some(false) => Some("out-of-topic"),
                None => None,
            },
        })
        .collect();
    let report = CellReport {
        family,
        decade,
        target,
        candidates: pool.len(),
        drawn: drawn.len(),
        shortfall: target.saturating_sub(drawn.len()),
        conjunctions: drawn.iter().filter(|d| d.candidate.conjunct).count(),
        in_topic: drawn
            .iter()
            .filter(|d| d.placement == Some("in-topic"))
            .count(),
        out_of_topic: drawn
            .iter()
            .filter(|d| d.placement == Some("out-of-topic"))
            .count(),
    };
    (drawn, report)
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloorReport {
    pub decade: i32,
    /// Smallest base count at which `s · N ≥ M + 3√M` for this decade.
    pub min_base_for_floor: u64,
    pub reliable_at_base_count: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationReport {
    pub schema_version: u32,
    pub seed: u64,
    pub base_count: u64,
    pub census_population: u64,
    pub families: Vec<Family>,
    pub decades: Vec<i32>,
    pub per_cell_targets: Vec<usize>,
    pub min_matches: u64,
    pub reliability_threshold: u64,
    pub predicates: usize,
    pub candidates: HashMap<String, usize>,
    pub cells: Vec<CellReport>,
    pub floors: Vec<FloorReport>,
    pub query_count: Option<usize>,
    pub placement: Placement,
    pub seconds: f64,
}

// ---------------------------------------------------------------------------
// The strategy
// ---------------------------------------------------------------------------

/// Options the stratified strategy adds to `generate predicates`.
pub fn describe_options() -> Vec<OptionDesc> {
    vec![
        opt(
            "base-count",
            "int",
            false,
            None,
            "stratified: N of the full base, for the reliability floors in the report (default: the census population)",
            OptionRole::Config,
        ),
        opt(
            "families",
            "string",
            false,
            Some("topical,structural,bibliographic,control"),
            "stratified: families to draw, and their order in the output",
            OptionRole::Config,
        ),
        opt(
            "topic-fields",
            "string",
            false,
            None,
            "stratified: topic fields outermost first (default: the survey's first declared hierarchy)",
            OptionRole::Config,
        ),
        opt(
            "bibliographic-fields",
            "string",
            false,
            Some("citation_percentile,year,isopenaccess"),
            "stratified: censused paper-level fields",
            OptionRole::Config,
        ),
        opt(
            "structural-fields",
            "string",
            false,
            Some("section_class,passage_position,word_count"),
            "stratified: censused passage-level fields",
            OptionRole::Config,
        ),
        opt(
            "control-field",
            "string",
            false,
            Some("sample_bucket"),
            "stratified: the hash field of the control family",
            OptionRole::Config,
        ),
        opt(
            "buckets",
            "int",
            false,
            Some("16777216"),
            "stratified: modulus of the control field",
            OptionRole::Config,
        ),
        opt(
            "decades",
            "string",
            false,
            Some("1e-1..1e-7"),
            "stratified: target decades, as a range or a comma list",
            OptionRole::Config,
        ),
        opt(
            "per-cell",
            "string",
            false,
            Some("tapered"),
            "stratified: predicates per (family, decade) cell — `tapered` (10, 20, then 50), one count, or one per decade coarsest first",
            OptionRole::Config,
        ),
        opt(
            "min-matches",
            "int",
            false,
            Some("100"),
            "stratified: M in the floor s·N ≥ M + 3√M",
            OptionRole::Config,
        ),
        opt(
            "reliability-threshold",
            "int",
            false,
            Some("10000000"),
            "stratified: base count above which the floor is promised",
            OptionRole::Config,
        ),
        opt(
            "query-placement",
            "string",
            false,
            Some("mixed"),
            "stratified: mixed, in-topic, out-of-topic or any; needs `queries`",
            OptionRole::Config,
        ),
        opt(
            "queries",
            "Path",
            false,
            None,
            "stratified: query vectors, for in-topic / out-of-topic placement",
            OptionRole::Input,
        ),
        opt(
            "centroids",
            "Path",
            false,
            None,
            "stratified: topic centroids, required with `queries`",
            OptionRole::Input,
        ),
        opt(
            "model",
            "Path",
            false,
            None,
            "stratified: topic model report, required with `queries`",
            OptionRole::Input,
        ),
        opt(
            "labels",
            "Path",
            false,
            None,
            "stratified: topic label slab, required with `queries`",
            OptionRole::Input,
        ),
        opt(
            "report",
            "Path",
            false,
            None,
            "stratified: generation report JSON (default: beside `output` with a .json extension)",
            OptionRole::Output,
        ),
    ]
}

/// The families namespace record of one predicate (TS-62).
fn family_record(d: &Drawn) -> Vec<u8> {
    let mut fields = IndexMap::new();
    fields.insert(
        "family".to_string(),
        MValue::Text(d.candidate.family.as_str().to_string()),
    );
    fields.insert(
        "selectivity".to_string(),
        MValue::Float(d.candidate.selectivity),
    );
    if let Some((level, _)) = &d.candidate.topic {
        fields.insert("topic_level".to_string(), MValue::Int(*level as i64));
        fields.insert("conjunct".to_string(), MValue::Bool(d.candidate.conjunct));
    }
    if let Some(p) = d.placement {
        fields.insert("query_placement".to_string(), MValue::Text(p.to_string()));
    }
    anode::encode(&ANode::MNode(MNode { fields }))
}

/// The generation namespace record of one predicate (TS-82).
fn generation_record(d: &Drawn) -> Vec<u8> {
    let mut fields = IndexMap::new();
    fields.insert("cell".to_string(), MValue::Text(d.cell.clone()));
    fields.insert("pool".to_string(), MValue::Int(d.pool as i64));
    fields.insert(
        "source".to_string(),
        MValue::Text(d.candidate.source.to_string()),
    );
    fields.insert(
        "expected_count".to_string(),
        MValue::Int(d.candidate.count as i64),
    );
    fields.insert(
        "vernacular".to_string(),
        MValue::Text(format!("{}", d.candidate.pnode)),
    );
    anode::encode(&ANode::MNode(MNode { fields }))
}

/// Run the stratified strategy. Called by `generate predicates` once
/// `--strategy stratified` is seen; `output` and `seed` are already
/// resolved the way the other strategies resolve them.
pub(super) fn run(
    options: &Options,
    ctx: &mut StreamContext,
    start: Instant,
    output_path: &Path,
    survey_path: Option<&Path>,
    seed: u64,
) -> CommandResult {
    let Some(survey_path) = survey_path else {
        return error_result(
            "strategy stratified needs --survey: the census tables are the candidate pools".into(),
            start,
        );
    };
    let survey: SurveyReport = match survey_report_from_json(survey_path) {
        Ok(r) => r,
        Err(e) => return error_result(e, start),
    };
    let population = survey.source.total_records;
    if population == 0 {
        return error_result("the survey covers zero records".into(), start);
    }
    if survey.source.census.is_none() {
        return error_result(
            "the survey has no census: run `analyze survey` with the census pass (its default) over the M facet".into(),
            start,
        );
    }
    let n = population as f64;
    let base_count = match options.parse_opt::<u64>("base-count") {
        Ok(v) => v.unwrap_or(population),
        Err(e) => return error_result(e, start),
    };
    let families: Vec<Family> = match options
        .get("families")
        .unwrap_or("topical,structural,bibliographic,control")
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(Family::parse)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(f) => f,
        Err(e) => return error_result(e, start),
    };
    let decades = match parse_decades(options.get("decades").unwrap_or("1e-1..1e-7")) {
        Ok(d) => d,
        Err(e) => return error_result(e, start),
    };
    let per_cell = match parse_per_cell(options.get("per-cell").unwrap_or("tapered"), decades.len())
    {
        Ok(p) => p,
        Err(e) => return error_result(e, start),
    };
    let min_matches = match options.parse_or::<u64>("min-matches", 100) {
        Ok(v) => v,
        Err(e) => return error_result(e, start),
    };
    let reliability_threshold = match options.parse_or::<u64>("reliability-threshold", 10_000_000) {
        Ok(v) => v,
        Err(e) => return error_result(e, start),
    };
    let buckets = match options.parse_or::<u64>("buckets", DEFAULT_BUCKETS) {
        Ok(v) if v > 0 => v,
        Ok(_) => return error_result("buckets must be positive".into(), start),
        Err(e) => return error_result(e, start),
    };
    let placement = match Placement::parse(options.get("query-placement").unwrap_or("mixed")) {
        Ok(p) => p,
        Err(e) => return error_result(e, start),
    };
    let topic_fields: Vec<String> = match options.get("topic-fields") {
        Some(s) => parse_fields(s),
        None => survey
            .hierarchies
            .first()
            .map(|h| h.fields.clone())
            .unwrap_or_else(|| vec!["topic_l1".into(), "topic_l2".into(), "topic_l3".into()]),
    };
    let bibliographic_fields = parse_fields(
        options
            .get("bibliographic-fields")
            .unwrap_or("citation_percentile,year,isopenaccess"),
    );
    let structural_fields = parse_fields(
        options
            .get("structural-fields")
            .unwrap_or("section_class,passage_position,word_count"),
    );
    let control_field = options
        .get("control-field")
        .unwrap_or("sample_bucket")
        .to_string();
    let report_path = match options.get("report") {
        Some(s) => resolve_path(s, &ctx.workspace),
        None => output_path.with_extension("json"),
    };

    // Query placement, when every input for it is present.
    let placement_inputs: Vec<Option<PathBuf>> = ["queries", "centroids", "model", "labels"]
        .iter()
        .map(|k| options.get(k).map(|s| resolve_path(s, &ctx.workspace)))
        .collect();
    let (query_topics, query_count) = match placement_inputs.as_slice() {
        [Some(q), Some(c), Some(m), Some(l)] => match query_topics(q, c, m, l) {
            Ok((t, n)) => (Some(t), Some(n)),
            Err(e) => return error_result(e, start),
        },
        [None, None, None, None] => (None, None),
        _ => {
            return error_result(
                "query placement needs all of --queries, --centroids, --model and --labels; none of them, or all".into(),
                start,
            )
        }
    };

    // ── Candidate pools ─────────────────────────────────────────────
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut sources: HashMap<String, usize> = HashMap::new();
    let note = |sources: &mut HashMap<String, usize>, key: String, n: usize| {
        *sources.entry(key).or_insert(0) += n;
    };
    if families.contains(&Family::Topical) {
        let hierarchy = survey.hierarchies.iter().find(|h| h.fields == topic_fields);
        match hierarchy {
            Some(h) => {
                let before = candidates.len();
                topical_candidates(&h.nodes, &h.fields, 0, n, &mut candidates);
                note(
                    &mut sources,
                    format!("hierarchy {}", h.fields.join(">")),
                    candidates.len() - before,
                );
            }
            None => {
                return error_result(
                    format!(
                        "the survey has no hierarchy census for {}; declare it with `hierarchy:` on the survey step",
                        topic_fields.join(">")
                    ),
                    start,
                );
            }
        }
        for p in &survey.pair_census {
            let Some(level) = topic_fields.iter().position(|f| f == &p.a) else {
                continue;
            };
            if !bibliographic_fields.contains(&p.b) {
                continue;
            }
            let before = candidates.len();
            pair_candidates(
                &p.a,
                &p.b,
                &p.a_values,
                &p.b_values,
                &p.counts,
                level + 1,
                n,
                &mut candidates,
            );
            note(
                &mut sources,
                format!("pair {}:{}", p.a, p.b),
                candidates.len() - before,
            );
        }
    }
    let field_candidates = |family: Family,
                            fields: &[String],
                            candidates: &mut Vec<Candidate>,
                            sources: &mut HashMap<String, usize>|
     -> Result<(), String> {
        for field in fields {
            let Some(profile) = survey.fields.get(field) else {
                return Err(format!(
                    "the survey has no field `{}` for the {} family",
                    field,
                    family.as_str()
                ));
            };
            if !profile.censused {
                return Err(format!(
                    "field `{}` is not censused; the {} family needs exact counts — list it under `census` on the survey step",
                    field,
                    family.as_str()
                ));
            }
            let before = candidates.len();
            if let Some(MeasureReport::ExactIntegerHistogram(h)) =
                profile.measures.get("ExactIntegerHistogram")
            {
                histogram_candidates(field, family, h.min, &h.counts, n, candidates);
            }
            if let Some(MeasureReport::ExactValueCensus(c)) =
                profile.measures.get("ExactValueCensus")
            {
                value_candidates(field, family, &c.counts, n, candidates);
            }
            *sources.entry(format!("field {}", field)).or_insert(0) += candidates.len() - before;
        }
        Ok(())
    };
    if families.contains(&Family::Bibliographic)
        && let Err(e) = field_candidates(
            Family::Bibliographic,
            &bibliographic_fields,
            &mut candidates,
            &mut sources,
        )
    {
        return error_result(e, start);
    }
    if families.contains(&Family::Structural)
        && let Err(e) = field_candidates(
            Family::Structural,
            &structural_fields,
            &mut candidates,
            &mut sources,
        )
    {
        return error_result(e, start);
    }
    // Bin every candidate by (family, decade); drop those outside the
    // configured decades.
    let mut pools: BTreeMap<(Family, i32), Vec<Candidate>> = BTreeMap::new();
    for c in candidates {
        if let Some(d) = decade_of(c.selectivity)
            && decades.contains(&d)
        {
            pools.entry((c.family, d)).or_default().push(c);
        }
    }
    ctx.ui.log(&format!(
        "stratified: {} candidates in {} cells from {} census sources; population {}, base {}",
        pools.values().map(Vec::len).sum::<usize>(),
        pools.len(),
        sources.len(),
        population,
        base_count,
    ));

    // ── Draw ───────────────────────────────────────────────────────
    let mut drawn: Vec<Drawn> = Vec::new();
    let mut cells: Vec<CellReport> = Vec::new();
    for &family in &families {
        for (di, &decade) in decades.iter().enumerate() {
            let target = per_cell[di];
            let (d, report) = if family == Family::Control {
                let pool = control_candidates(&control_field, decade, target, buckets, n);
                draw_cell(family, decade, target, &pool, seed, placement, None)
            } else {
                let pool = pools
                    .get(&(family, decade))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                draw_cell(
                    family,
                    decade,
                    target,
                    pool,
                    seed,
                    placement,
                    query_topics.as_ref(),
                )
            };
            drawn.extend(d);
            cells.push(report);
        }
    }
    let shortfalls: usize = cells.iter().map(|c| c.shortfall).sum();
    for c in cells.iter().filter(|c| c.shortfall > 0) {
        ctx.ui.log(&format!(
            "stratified: cell {}:1e{} short by {} ({} candidates for {})",
            c.family.as_str(),
            c.decade,
            c.shortfall,
            c.candidates,
            c.target,
        ));
    }

    // ── Write the slab: content, schema, survey, families, generation.
    if let Some(parent) = output_path.parent()
        && !parent.exists()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return error_result(
            format!("failed to create {}: {}", parent.display(), e),
            start,
        );
    }
    let config = match WriterConfig::new(512, 4096, u32::MAX, false) {
        Ok(c) => c,
        Err(e) => return error_result(format!("writer config error: {}", e), start),
    };
    let mut writer = match SlabWriter::new(output_path, config) {
        Ok(w) => w,
        Err(e) => return error_result(format!("failed to create output: {}", e), start),
    };
    for d in &drawn {
        if let Err(e) = writer.add_record(&d.candidate.pnode.to_bytes_named()) {
            return error_result(format!("write error: {}", e), start);
        }
    }
    let schema = PredicateSchema::new(
        "<stratified>",
        format!(
            "1e{}..1e{}",
            decades.first().unwrap(),
            decades.last().unwrap()
        ),
        seed,
        drawn.len() as u64,
    );
    let sections: [(&str, Vec<Vec<u8>>); 4] = [
        (SCHEMA_NAMESPACE, vec![schema.to_json_bytes()]),
        (
            SURVEY_NAMESPACE,
            vec![match serde_json::to_vec(&survey) {
                Ok(v) => v,
                Err(e) => return error_result(format!("serialise survey report: {e}"), start),
            }],
        ),
        (
            FAMILIES_NAMESPACE,
            drawn.iter().map(family_record).collect(),
        ),
        (
            GENERATION_NAMESPACE,
            drawn.iter().map(generation_record).collect(),
        ),
    ];
    for (name, records) in sections {
        if let Err(e) = writer.start_namespace(name) {
            return error_result(format!("{} namespace: {}", name, e), start);
        }
        for r in records {
            if let Err(e) = writer.add_record(&r) {
                return error_result(format!("{} write: {}", name, e), start);
            }
        }
    }
    if let Err(e) = writer.finish() {
        return error_result(format!("finish error: {}", e), start);
    }
    let var_name = format!(
        "verified_count:{}",
        output_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("output")
    );
    let _ = crate::pipeline::variables::set_and_save(
        &ctx.workspace,
        &var_name,
        &drawn.len().to_string(),
    );
    ctx.defaults.insert(var_name, drawn.len().to_string());

    // ── Report ─────────────────────────────────────────────────────
    let floors: Vec<FloorReport> = decades
        .iter()
        .map(|&d| {
            let s = 10f64.powi(d);
            let need = (min_matches as f64 + 3.0 * (min_matches as f64).sqrt()) / s;
            FloorReport {
                decade: d,
                min_base_for_floor: need.ceil() as u64,
                reliable_at_base_count: base_count >= reliability_threshold
                    && base_count as f64 >= need,
            }
        })
        .collect();
    let report = GenerationReport {
        schema_version: 1,
        seed,
        base_count,
        census_population: population,
        families: families.clone(),
        decades: decades.clone(),
        per_cell_targets: per_cell.clone(),
        min_matches,
        reliability_threshold,
        predicates: drawn.len(),
        candidates: sources,
        cells,
        floors,
        query_count,
        placement,
        seconds: start.elapsed().as_secs_f64(),
    };
    let mut produced = vec![output_path.to_path_buf()];
    match serde_json::to_string_pretty(&report) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&report_path, json) {
                return error_result(
                    format!("failed to write {}: {}", report_path.display(), e),
                    start,
                );
            }
            produced.push(report_path);
        }
        Err(e) => return error_result(format!("report serialisation failed: {}", e), start),
    }

    CommandResult {
        status: Status::Ok,
        message: format!(
            "{} predicates drawn over {} families × {} decades ({} cells short by {} in total){}",
            drawn.len(),
            families.len(),
            decades.len(),
            report.cells.iter().filter(|c| c.shortfall > 0).count(),
            shortfalls,
            match query_count {
                Some(q) => format!("; placement from {} queries", q),
                None => String::new(),
            },
        ),
        produced,
        elapsed: start.elapsed(),
    }
}

/// Complete when the content namespace holds at least one predicate
/// and the `families` namespace holds exactly as many records
/// (TS-111). Absence of the namespace is one unlabelled family, which
/// is what every earlier predicate set is; this check applies to a
/// stratified step only.
pub(super) fn check_artifact(output: &Path) -> Option<bool> {
    let content = SlabReader::open(output).ok()?.total_records();
    let families = SlabReader::open_namespace(output, Some(FAMILIES_NAMESPACE))
        .ok()?
        .total_records();
    let generation = SlabReader::open_namespace(output, Some(GENERATION_NAMESPACE))
        .ok()?
        .total_records();
    Some(content > 0 && families == content && generation == content)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decades_tile_the_axis() {
        assert_eq!(decade_of(1e-4), Some(-4));
        assert_eq!(
            decade_of(3.17e-4),
            Some(-3),
            "just above d·√10 belongs to the next decade up"
        );
        assert_eq!(decade_of(3.15e-4), Some(-4));
        assert_eq!(
            decade_of(10f64.powf(-3.5)),
            Some(-3),
            "the upper edge is excluded from the lower band"
        );
        assert_eq!(decade_of(0.0), None);
        assert_eq!(parse_decades("1e-1..1e-3").unwrap(), vec![-1, -2, -3]);
        assert_eq!(parse_decades("1e-3..1e-1").unwrap(), vec![-1, -2, -3]);
        assert_eq!(parse_decades("1e-2, 1e-4,1e-2").unwrap(), vec![-2, -4]);
        assert!(parse_decades("").is_err());
        assert_eq!(
            parse_per_cell("tapered", 5).unwrap(),
            vec![10, 20, 50, 50, 50]
        );
        assert_eq!(parse_per_cell("7", 3).unwrap(), vec![7, 7, 7]);
        assert_eq!(parse_per_cell("1,2", 4).unwrap(), vec![1, 2, 2, 2]);
        assert!(parse_per_cell("0", 2).is_err());
    }

    #[test]
    fn histogram_candidates_are_exact_prefix_sums() {
        let counts = [10u64, 20, 30, 40]; // values 5..8
        let mut out = Vec::new();
        histogram_candidates("f", Family::Structural, 5, &counts, 1000.0, &mut out);
        let find = |s: &str| {
            out.iter()
                .find(|c| format!("{}", c.pnode) == s)
                .unwrap_or_else(|| {
                    panic!(
                        "no candidate `{}` in {:?}",
                        s,
                        out.iter()
                            .map(|c| format!("{}", c.pnode))
                            .collect::<Vec<_>>()
                    )
                })
        };
        assert_eq!(find("f >= 6").count, 90);
        assert_eq!(find("f <= 6").count, 30);
        assert_eq!(find("f >= 8").count, 40);
        assert!(
            out.iter().all(|c| c.count > 0 && c.count < 100),
            "no all-or-nothing candidates"
        );
        assert!(
            out.iter()
                .any(|c| c.pnode.to_string().contains("f >= 5 AND f <= 6")),
            "width-2 ranges: {:?}",
            out.iter().map(|c| c.pnode.to_string()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn control_ranges_are_disjoint_and_exact() {
        let out = control_candidates("h", -2, 5, 1000, 50_000.0);
        assert_eq!(out.len(), 5);
        assert!(
            out.iter()
                .all(|c| (c.selectivity - 0.01).abs() < 1e-12 && c.count == 500)
        );
        assert_eq!(out[0].pnode.to_string(), "(h >= 0 AND h <= 9)");
        assert_eq!(out[4].pnode.to_string(), "(h >= 40 AND h <= 49)");
        // A cell that would run past the modulus stops short.
        assert_eq!(control_candidates("h", -1, 20, 1000, 1.0).len(), 10);
    }

    #[test]
    fn drawing_is_seeded_and_backfills_with_conjunctions() {
        let single = |i: i64| Candidate {
            pnode: cmp("x", OpType::Eq, i),
            family: Family::Topical,
            count: 1,
            selectivity: 1e-3,
            source: "t",
            topic: Some((1, format!("t{}", i))),
            conjunct: false,
        };
        let conj = |i: i64| Candidate {
            conjunct: true,
            ..single(100 + i)
        };
        let pool: Vec<Candidate> = (0..3).map(single).chain((0..5).map(conj)).collect();
        let (a, ra) = draw_cell(Family::Topical, -3, 6, &pool, 7, Placement::Any, None);
        let (b, _) = draw_cell(Family::Topical, -3, 6, &pool, 7, Placement::Any, None);
        assert_eq!(ra.drawn, 6);
        assert_eq!(
            ra.conjunctions, 3,
            "singles first, conjunctions make up the rest"
        );
        assert_eq!(ra.shortfall, 0);
        let names = |v: &[Drawn]| {
            v.iter()
                .map(|d| d.candidate.pnode.to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&a), names(&b), "same seed, same draw");
        let (c, _) = draw_cell(Family::Topical, -3, 6, &pool, 8, Placement::Any, None);
        assert_ne!(names(&a), names(&c), "different seed, different order");
        let (_, short) = draw_cell(Family::Topical, -3, 20, &pool, 7, Placement::Any, None);
        assert_eq!(short.shortfall, 12);
    }

    #[test]
    fn placement_mixes_in_and_out_of_topic() {
        let mk = |i: usize, label: &str| Candidate {
            pnode: cmp("x", OpType::Eq, i as i64),
            family: Family::Topical,
            count: 1,
            selectivity: 1e-2,
            source: "t",
            topic: Some((1, label.to_string())),
            conjunct: false,
        };
        let pool: Vec<Candidate> = (0..4)
            .map(|i| mk(i, "in"))
            .chain((4..10).map(|i| mk(i, "out")))
            .collect();
        let qt: HashSet<(usize, String)> = [(1usize, "in".to_string())].into_iter().collect();
        let (d, r) = draw_cell(
            Family::Topical,
            -2,
            6,
            &pool,
            1,
            Placement::Mixed,
            Some(&qt),
        );
        assert_eq!((r.in_topic, r.out_of_topic), (3, 3));
        assert!(d.iter().all(|x| x.placement.is_some()));
        let (_, r) = draw_cell(
            Family::Topical,
            -2,
            8,
            &pool,
            1,
            Placement::Mixed,
            Some(&qt),
        );
        assert_eq!(
            (r.in_topic, r.out_of_topic),
            (4, 4),
            "backfilled from the larger side"
        );
        let (_, r) = draw_cell(
            Family::Topical,
            -2,
            3,
            &pool,
            1,
            Placement::OutOfTopic,
            Some(&qt),
        );
        assert_eq!((r.in_topic, r.out_of_topic), (0, 3));
    }
}
