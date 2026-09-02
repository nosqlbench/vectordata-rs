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

use std::collections::{BTreeMap, HashMap};
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

/// Default slots per decade, decade 10⁻¹ first: the taper of TS-54 as
/// absolute counts for the three coarsest decades, every decade below
/// them sharing the rest of the query slots (TS-159).
const TAPERED: [Slots; 3] = [Slots::Absolute(10), Slots::Absolute(20), Slots::Absolute(50)];

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
/// How many of a family's query slots one decade gets (TS-159).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slots {
    /// This many, when any decade says `rest`; otherwise a weight.
    Absolute(usize),
    /// An equal share of what the absolute decades leave.
    Rest,
}

impl std::fmt::Display for Slots {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Slots::Absolute(n) => write!(f, "{}", n),
            Slots::Rest => write!(f, "rest"),
        }
    }
}

/// Parse `per-cell`: `tapered`, one count, or one entry per decade
/// (coarsest first), each a count or `rest`. Numbers alone are weights;
/// with a `rest` anywhere they are absolute and the `rest` decades
/// share what remains.
pub fn parse_per_cell(spec: &str, decades: usize) -> Result<Vec<Slots>, String> {
    if decades == 0 {
        return Err("per-cell: no decades to fill".into());
    }
    let spec = spec.trim();
    if spec.eq_ignore_ascii_case("tapered") {
        return Ok((0..decades)
            .map(|i| TAPERED.get(i).copied().unwrap_or(Slots::Rest))
            .collect());
    }
    let parts: Vec<&str> = spec.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    let parse_one = |s: &str| -> Result<Slots, String> {
        if s.eq_ignore_ascii_case("rest") {
            return Ok(Slots::Rest);
        }
        let n: usize = s
            .parse()
            .map_err(|_| format!("per-cell: `{}` is not a count or `rest`", s))?;
        Ok(Slots::Absolute(n))
    };
    let out: Vec<Slots> = match parts.len() {
        0 => return Err("per-cell: empty".into()),
        1 => vec![parse_one(parts[0])?; decades],
        n if n == decades => parts.iter().map(|s| parse_one(s)).collect::<Result<_, _>>()?,
        n => {
            return Err(format!(
                "per-cell: {} entries for {} decades; give one, or one per decade",
                n, decades
            ))
        }
    };
    if out.iter().all(|s| *s == Slots::Absolute(0)) {
        return Err("per-cell: counts must be positive".into());
    }
    Ok(out)
}

/// A family's slots per decade from its share `count` and the
/// per-cell spec (TS-159): weights are apportioned; absolute counts
/// are taken as they are, capped at the share, and the `rest` decades
/// split what is left evenly. Absolute counts that exceed the share
/// are scaled down as weights.
pub fn slots_per_decade(count: usize, spec: &[Slots]) -> Vec<usize> {
    let has_rest = spec.contains(&Slots::Rest);
    let weights: Vec<usize> = spec
        .iter()
        .map(|s| match s {
            Slots::Absolute(n) => *n,
            Slots::Rest => 0,
        })
        .collect();
    if !has_rest {
        return apportion(count, &weights);
    }
    let fixed: usize = weights.iter().sum();
    if fixed >= count {
        return apportion(count, &weights);
    }
    let rest_weights: Vec<usize> = spec.iter().map(|s| usize::from(*s == Slots::Rest)).collect();
    let shared = apportion(count - fixed, &rest_weights);
    weights.iter().zip(shared).map(|(w, s)| w + s).collect()
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

/// Where every query lies in the topic hierarchy: for query *i*, its
/// `(level, label)` at each level, by the same descent that assigned
/// the base (TS-137). Placement is decided per (query, predicate) pair
/// from this (TS-19).
pub struct QueryTopics {
    /// `per_query[i][l]` is query *i*'s topic at level *l* + 1.
    pub per_query: Vec<Vec<(usize, String)>>,
}

impl QueryTopics {
    pub fn count(&self) -> usize {
        self.per_query.len()
    }

    /// Whether query `q` lies in `topic`.
    fn query_in(&self, q: usize, topic: &(usize, String)) -> bool {
        self.per_query[q].iter().any(|t| t == topic)
    }

    /// Query `q`'s own label at `level`.
    fn label_at(&self, q: usize, level: usize) -> Option<&str> {
        self.per_query[q]
            .iter()
            .find(|(l, _)| *l == level)
            .map(|(_, s)| s.as_str())
    }
}

/// Descend every query through the fitted model and name its topics.
fn query_topics(
    queries: &Path,
    centroids: &Path,
    model: &Path,
    labels: &Path,
) -> Result<QueryTopics, String> {
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
    let mut per_query = Vec::with_capacity(q.count());
    for i in 0..q.count() {
        let mut v = q.get(i).map_err(|e| format!("query {}: {}", i, e))?;
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        topic_model.descend(&v, dot, &mut codes);
        let mut topics = Vec::with_capacity(codes.len());
        for (l, &code) in codes.iter().enumerate() {
            let label = label_table[l]
                .get(code as usize)
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("l{}-{:05}", l + 1, code));
            topics.push((l + 1, label));
        }
        per_query.push(topics);
    }
    Ok(QueryTopics { per_query })
}

// ---------------------------------------------------------------------------
// Drawing and pairing
// ---------------------------------------------------------------------------

/// What one cell did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellReport {
    pub family: Family,
    pub decade: i32,
    /// Query slots apportioned to the cell (TS-156).
    pub target: usize,
    pub candidates: usize,
    /// Distinct predicates drawn.
    pub drawn: usize,
    /// Slots the cell filled; `target − filled` went to backfill.
    pub filled: usize,
    /// Slots the cell could not fill.
    pub shortfall: usize,
    pub conjunctions: usize,
    pub in_topic: usize,
    pub out_of_topic: usize,
}

/// One query's predicate with everything both namespaces record.
struct Drawn {
    candidate: Candidate,
    cell: String,
    pool: usize,
    /// In-topic or out-of-topic, for a topical pair whose query's
    /// topics are known.
    placement: Option<&'static str>,
    /// The query's own label at the predicate's level, when known.
    query_topic: Option<String>,
    /// Index of the distinct predicate among all distinct predicates.
    predicate: usize,
    /// Drawn from the control family to fill a slot no cell could.
    backfill: bool,
}

fn cell_seed(seed: u64, family: Family, decade: i32) -> u64 {
    let mut z = seed
        .wrapping_add((family as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((decade as i64 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Apportion `total` slots over `weights` by largest remainder, so the
/// shares sum to `total` exactly and each is within one of its ideal
/// (TS-156). All-zero weights yield all-zero shares.
pub fn apportion(total: usize, weights: &[usize]) -> Vec<usize> {
    let sum: usize = weights.iter().sum();
    if sum == 0 || weights.is_empty() {
        return vec![0; weights.len()];
    }
    let mut shares: Vec<usize> = weights.iter().map(|w| total * w / sum).collect();
    let mut given: usize = shares.iter().sum();
    let mut by_remainder: Vec<(usize, usize)> = weights
        .iter()
        .enumerate()
        .filter(|(_, w)| **w > 0)
        .map(|(i, w)| ((total * w) % sum, i))
        .collect();
    by_remainder.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut i = 0;
    while given < total {
        shares[by_remainder[i % by_remainder.len()].1] += 1;
        given += 1;
        i += 1;
    }
    shares
}

/// Draw the distinct predicates of one cell: single-field candidates
/// first, conjunctions only to make up a shortfall (TS-116), in seeded
/// order, at most `target`.
fn draw_distinct(pool: &[Candidate], target: usize, seed: u64) -> Vec<&Candidate> {
    let mut rng = rng::seeded_rng(seed);
    let mut singles: Vec<&Candidate> = pool.iter().filter(|c| !c.conjunct).collect();
    let mut conjuncts: Vec<&Candidate> = pool.iter().filter(|c| c.conjunct).collect();
    singles.shuffle(&mut rng);
    conjuncts.shuffle(&mut rng);
    let mut take: Vec<&Candidate> = Vec::with_capacity(target);
    take.extend(singles.iter().take(target));
    if take.len() < target {
        let need = target - take.len();
        take.extend(conjuncts.iter().take(need));
    }
    take
}

/// The unassigned queries, in a seeded order, with an index by topic
/// so an in-topic pair is found without a scan.
struct QueryPool {
    /// Queries not yet given a predicate, in draw order.
    free: Vec<usize>,
    taken: Vec<bool>,
    /// Topic → free queries in it, in draw order.
    by_topic: HashMap<(usize, String), Vec<usize>>,
}

impl QueryPool {
    fn new(count: usize, topics: Option<&QueryTopics>, seed: u64) -> Self {
        let mut rng = rng::seeded_rng(seed ^ 0x51_6C_6F_74_73);
        let mut free: Vec<usize> = (0..count).collect();
        free.shuffle(&mut rng);
        let mut by_topic: HashMap<(usize, String), Vec<usize>> = HashMap::new();
        if let Some(t) = topics {
            for &q in &free {
                for topic in &t.per_query[q] {
                    by_topic.entry(topic.clone()).or_default().push(q);
                }
            }
        }
        QueryPool {
            free,
            taken: vec![false; count],
            by_topic,
        }
    }

    fn remaining(&self) -> usize {
        self.free.len()
    }

    /// Take the next free query satisfying `accept`.
    fn take_where(&mut self, accept: impl Fn(usize) -> bool) -> Option<usize> {
        let pos = self.free.iter().position(|&q| accept(q))?;
        let q = self.free.remove(pos);
        self.taken[q] = true;
        Some(q)
    }

    /// Take a free query lying in `topic`.
    fn take_in(&mut self, topic: &(usize, String)) -> Option<usize> {
        let list = self.by_topic.get_mut(topic)?;
        while let Some(q) = list.pop() {
            if !self.taken[q] {
                self.taken[q] = true;
                if let Some(pos) = self.free.iter().position(|&f| f == q) {
                    self.free.remove(pos);
                }
                return Some(q);
            }
        }
        None
    }
}

/// Fill one cell's slots: draw its distinct predicates and pair each
/// with a query. A topical cell whose queries' topics are known honours
/// the placement mix per pair (TS-19, TS-157): an in-topic slot pairs a
/// predicate with a query inside its topic, an out-of-topic slot with
/// one outside it. Any other cell pairs with queries in draw order,
/// which is the zero correlation those families measure. Distinct
/// predicates repeat only when the pool is smaller than the slots.
#[allow(clippy::too_many_arguments)]
fn fill_cell(
    family: Family,
    decade: i32,
    slots: usize,
    pool: &[Candidate],
    seed: u64,
    placement: Placement,
    topics: Option<&QueryTopics>,
    queries: &mut QueryPool,
    next_predicate: &mut usize,
    out: &mut Vec<(usize, Drawn)>,
) -> CellReport {
    let cell = format!("{}:1e{}", family.as_str(), decade);
    let distinct = draw_distinct(pool, slots, cell_seed(seed, family, decade));
    let mut report = CellReport {
        family,
        decade,
        target: slots,
        candidates: pool.len(),
        drawn: distinct.len(),
        filled: 0,
        shortfall: 0,
        conjunctions: 0,
        in_topic: 0,
        out_of_topic: 0,
    };
    if distinct.is_empty() || slots == 0 {
        report.shortfall = slots;
        return report;
    }
    // Distinct ids are assigned in draw order the first time a
    // predicate is paired.
    let mut ids: Vec<Option<usize>> = vec![None; distinct.len()];
    let mut push = |di: usize, q: usize, placement: Option<&'static str>, report: &mut CellReport| {
        let c = distinct[di];
        let id = *ids[di].get_or_insert_with(|| {
            let id = *next_predicate;
            *next_predicate += 1;
            id
        });
        let query_topic = match (&c.topic, topics) {
            (Some((level, _)), Some(t)) => t.label_at(q, *level).map(str::to_string),
            _ => None,
        };
        out.push((
            q,
            Drawn {
                candidate: c.clone(),
                cell: cell.clone(),
                pool: pool.len(),
                placement,
                query_topic,
                predicate: id,
                backfill: false,
            },
        ));
        report.filled += 1;
        if c.conjunct {
            report.conjunctions += 1;
        }
        match placement {
            Some("in-topic") => report.in_topic += 1,
            Some("out-of-topic") => report.out_of_topic += 1,
            _ => {}
        }
    };
    let placed = family == Family::Topical && topics.is_some() && placement != Placement::Any;
    if placed {
        let t = topics.unwrap();
        let (want_in, want_out) = match placement {
            Placement::InTopic => (slots, 0),
            Placement::OutOfTopic => (0, slots),
            _ => (slots.div_ceil(2), slots / 2),
        };
        // In-topic pairs: cycle the distinct predicates, each taking a
        // free query inside its topic, until the share is met or no
        // predicate has one left.
        let mut got_in = 0;
        let mut progress = true;
        while got_in < want_in && progress {
            progress = false;
            for di in 0..distinct.len() {
                if got_in >= want_in {
                    break;
                }
                let Some(topic) = &distinct[di].topic else { continue };
                if let Some(q) = queries.take_in(topic) {
                    push(di, q, Some("in-topic"), &mut report);
                    got_in += 1;
                    progress = true;
                }
            }
        }
        // Out-of-topic pairs, then whatever the in-topic side could not
        // fill, from any free query outside the predicate's topic.
        let mut got_out = 0;
        let want_out = want_out + (want_in - got_in);
        progress = true;
        while got_out < want_out && progress {
            progress = false;
            for di in 0..distinct.len() {
                if got_out >= want_out {
                    break;
                }
                let Some(topic) = &distinct[di].topic else { continue };
                if let Some(q) = queries.take_where(|q| !t.query_in(q, topic)) {
                    push(di, q, Some("out-of-topic"), &mut report);
                    got_out += 1;
                    progress = true;
                }
            }
        }
    } else {
        let mut i = 0;
        while report.filled < slots {
            let di = i % distinct.len();
            let Some(q) = queries.take_where(|_| true) else { break };
            let placement = match (&distinct[di].topic, topics) {
                (Some(topic), Some(t)) => Some(if t.query_in(q, topic) { "in-topic" } else { "out-of-topic" }),
                _ => None,
            };
            push(di, q, placement, &mut report);
            i += 1;
        }
    }
    report.shortfall = slots - report.filled;
    report
}

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
    /// The `per-cell` spec, one entry per decade (TS-159).
    pub per_cell: Vec<String>,
    pub min_matches: u64,
    pub reliability_threshold: u64,
    /// Records written: one per query ordinal (TS-156).
    pub predicates: usize,
    /// Distinct predicates among them.
    pub distinct_predicates: usize,
    /// Slots apportioned per cell, in `families` × `decades` order.
    pub slots_per_cell: Vec<usize>,
    /// Records filled from the control family because no cell could.
    pub backfilled: usize,
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
            "count",
            "int",
            false,
            None,
            "stratified: records to write, one per query ordinal (default: the number of `queries`; required without them)",
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
            "stratified: a family's query slots per decade, coarsest first — `tapered` (10, 20, 50, the rest shared below), one weight, or one entry per decade; numbers alone are weights, with `rest` they are counts",
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
            "stratified: mix of topical pairs whose query lies inside its predicate's topic — mixed, in-topic, out-of-topic or any; needs `queries`",
            OptionRole::Config,
        ),
        opt(
            "queries",
            "Path",
            false,
            None,
            "stratified: the query vectors; record i is query i's predicate, and placement is decided per pair",
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
    if let Some((level, label)) = &d.candidate.topic {
        fields.insert("topic_level".to_string(), MValue::Int(*level as i64));
        fields.insert("topic".to_string(), MValue::Text(label.clone()));
        fields.insert("conjunct".to_string(), MValue::Bool(d.candidate.conjunct));
    }
    if let Some(p) = d.placement {
        fields.insert("query_placement".to_string(), MValue::Text(p.to_string()));
    }
    if let Some(qt) = &d.query_topic {
        fields.insert("query_topic".to_string(), MValue::Text(qt.clone()));
    }
    fields.insert("predicate".to_string(), MValue::Int(d.predicate as i64));
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
    fields.insert("backfill".to_string(), MValue::Bool(d.backfill));
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
    let topics: Option<QueryTopics> = match placement_inputs.as_slice() {
        [Some(q), Some(c), Some(m), Some(l)] => match query_topics(q, c, m, l) {
            Ok(t) => Some(t),
            Err(e) => return error_result(e, start),
        },
        [None, None, None, None] => None,
        _ => {
            return error_result(
                "query placement needs all of --queries, --centroids, --model and --labels; none of them, or all".into(),
                start,
            )
        }
    };
    let query_count = topics.as_ref().map(QueryTopics::count);
    // One record per query ordinal (TS-156): the count is the number of
    // queries, or `count` when the queries are not given.
    let count = match (options.parse_opt::<usize>("count"), query_count) {
        (Err(e), _) => return error_result(e, start),
        (Ok(Some(c)), Some(q)) if c != q => {
            return error_result(
                format!("count {} does not equal the {} queries; record i is query i's predicate", c, q),
                start,
            )
        }
        (Ok(Some(c)), _) | (Ok(None), Some(c)) => c,
        (Ok(None), None) => {
            return error_result(
                "stratified writes one predicate per query ordinal: give `queries` (with centroids, model and labels) or `count`".into(),
                start,
            )
        }
    };
    if count == 0 {
        return error_result("count must be positive".into(), start);
    }

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
        "stratified: {} candidates in {} cells from {} census sources; population {}, base {}; {} query slots",
        pools.values().map(Vec::len).sum::<usize>(),
        pools.len(),
        sources.len(),
        population,
        base_count,
        count,
    ));

    // ── Apportion the query slots and fill the cells ───────────────
    // Families share the slots equally; within a family the per-cell
    // spec says how the decades split them (TS-156, TS-159). Topical
    // cells fill first, finest decade first: they need particular
    // queries (TS-157). The other families take any query, which is
    // the point of them.
    let family_shares = apportion(count, &vec![1; families.len()]);
    let mut slots: Vec<usize> = Vec::with_capacity(families.len() * decades.len());
    for share in &family_shares {
        slots.extend(slots_per_decade(*share, &per_cell));
    }
    let mut order: Vec<(usize, Family, i32)> = Vec::new();
    for (fi, &family) in families.iter().enumerate() {
        for (di, &decade) in decades.iter().enumerate() {
            order.push((fi * decades.len() + di, family, decade));
        }
    }
    order.sort_by_key(|&(_, family, decade)| (family != Family::Topical, decade));
    let mut queries = QueryPool::new(count, topics.as_ref(), seed);
    let mut paired: Vec<(usize, Drawn)> = Vec::with_capacity(count);
    let mut cells_by_index: BTreeMap<usize, CellReport> = BTreeMap::new();
    let mut next_predicate = 0usize;
    for &(ci, family, decade) in &order {
        let target = slots[ci];
        let report = if family == Family::Control {
            let pool = control_candidates(&control_field, decade, target, buckets, n);
            fill_cell(family, decade, target, &pool, seed, placement, None, &mut queries, &mut next_predicate, &mut paired)
        } else {
            let pool = pools
                .get(&(family, decade))
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            fill_cell(family, decade, target, pool, seed, placement, topics.as_ref(), &mut queries, &mut next_predicate, &mut paired)
        };
        cells_by_index.insert(ci, report);
    }
    let cells: Vec<CellReport> = cells_by_index.into_values().collect();
    let shortfalls: usize = cells.iter().map(|c| c.shortfall).sum();
    for c in cells.iter().filter(|c| c.shortfall > 0) {
        ctx.ui.log(&format!(
            "stratified: cell {}:1e{} short by {} of {} slots ({} candidates)",
            c.family.as_str(),
            c.decade,
            c.shortfall,
            c.target,
            c.candidates,
        ));
    }
    // Backfill: every query left over takes a control predicate, at
    // the decades in turn, so no query is without one (TS-156).
    let mut backfilled = 0usize;
    if queries.remaining() > 0 {
        let need = queries.remaining();
        let per_decade = need.div_ceil(decades.len());
        let mut pools_by_decade: Vec<(i32, Vec<Candidate>, usize)> = decades
            .iter()
            .map(|&d| (d, control_candidates(&control_field, d, per_decade, buckets, n), 0usize))
            .collect();
        let mut di = 0usize;
        while let Some(q) = queries.take_where(|_| true) {
            let (decade, pool, used) = &mut pools_by_decade[di % decades.len()];
            let c = &pool[*used % pool.len()];
            *used += 1;
            paired.push((
                q,
                Drawn {
                    candidate: c.clone(),
                    cell: format!("control:1e{}", decade),
                    pool: pool.len(),
                    placement: None,
                    query_topic: None,
                    predicate: next_predicate,
                    backfill: true,
                },
            ));
            next_predicate += 1;
            backfilled += 1;
            di += 1;
        }
        ctx.ui.log(&format!(
            "stratified: {} query slot(s) no cell could fill took a control predicate",
            backfilled
        ));
    }
    paired.sort_by_key(|(q, _)| *q);
    debug_assert!(paired.iter().enumerate().all(|(i, (q, _))| i == *q));
    let drawn: Vec<Drawn> = paired.into_iter().map(|(_, d)| d).collect();
    let distinct_predicates = next_predicate;

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
        per_cell: per_cell.iter().map(|s| s.to_string()).collect(),
        min_matches,
        reliability_threshold,
        predicates: drawn.len(),
        distinct_predicates,
        slots_per_cell: slots,
        backfilled,
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
            "{} predicates, one per query, {} distinct, over {} families × {} decades ({} cells short by {} slots in total, {} backfilled){}",
            drawn.len(),
            distinct_predicates,
            families.len(),
            decades.len(),
            report.cells.iter().filter(|c| c.shortfall > 0).count(),
            shortfalls,
            backfilled,
            match query_count {
                Some(q) => format!("; placement per pair from {} queries", q),
                None => String::new(),
            },
        ),
        produced,
        elapsed: start.elapsed(),
    }
}

/// Complete when the content namespace holds exactly the expected
/// number of predicates — one per query, from `count` or the `queries`
/// facet (TS-156) — **and** the `families` and `generation` namespaces
/// hold as many records each (TS-111). An unequal count means the
/// pairing with queries or the annotation is off, which TS-64 exists
/// to prevent. Absence of the namespace is one unlabelled family,
/// which is what every earlier predicate set is; this check applies
/// to a stratified step only.
pub(super) fn check_artifact(output: &Path, options: &Options) -> Option<bool> {
    let workspace = super::compute_topics::workspace_of(output, options.get("output"));
    let expected: Option<u64> = match options.get("count") {
        Some(c) => c.trim().parse::<u64>().ok(),
        None => options.get("queries").and_then(|q| {
            XvecReader::<f32>::open_path(&resolve_path(q, &workspace))
                .ok()
                .map(|r| r.count() as u64)
        }),
    };
    let content = SlabReader::open(output).ok()?.total_records();
    let families = SlabReader::open_namespace(output, Some(FAMILIES_NAMESPACE))
        .ok()?
        .total_records();
    let generation = SlabReader::open_namespace(output, Some(GENERATION_NAMESPACE))
        .ok()?
        .total_records();
    let count_ok = match expected {
        Some(e) => content == e,
        None => content > 0,
    };
    Some(count_ok && families == content && generation == content)
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
            vec![Slots::Absolute(10), Slots::Absolute(20), Slots::Absolute(50), Slots::Rest, Slots::Rest]
        );
        assert_eq!(parse_per_cell("7", 3).unwrap(), vec![Slots::Absolute(7); 3]);
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
    fn apportionment_is_exact_and_proportional() {
        assert_eq!(apportion(10_000, &[10, 20, 50, 50, 50, 50, 50]).iter().sum::<usize>(), 10_000);
        assert_eq!(apportion(100, &[10, 20, 50, 50, 50, 50, 50]), vec![3, 7, 18, 18, 18, 18, 18], "largest remainders first, ties by position");
        assert_eq!(apportion(7, &[0, 0]), vec![0, 0]);
        assert_eq!(apportion(3, &[1, 1]), vec![2, 1]);
        assert_eq!(apportion(5, &[0, 1]), vec![0, 5], "a zero weight never gets a remainder slot");
    }

    /// The taper keeps the coarse decades at their absolute counts and
    /// hands everything else to the decades below; plain numbers are
    /// weights.
    #[test]
    fn slots_follow_the_per_cell_spec() {
        let tapered = parse_per_cell("tapered", 7).unwrap();
        assert_eq!(tapered[..3], [Slots::Absolute(10), Slots::Absolute(20), Slots::Absolute(50)]);
        assert!(tapered[3..].iter().all(|s| *s == Slots::Rest));
        let s = slots_per_decade(2_500, &tapered);
        assert_eq!(s, vec![10, 20, 50, 605, 605, 605, 605]);
        assert_eq!(slots_per_decade(40, &tapered), vec![5, 10, 25, 0, 0, 0, 0], "a share below the fixed counts scales them as weights");
        assert_eq!(slots_per_decade(75, &parse_per_cell("4,6,8", 3).unwrap()), vec![17, 25, 33]);
        assert_eq!(slots_per_decade(9, &parse_per_cell("3", 3).unwrap()), vec![3, 3, 3]);
        assert_eq!(slots_per_decade(10, &parse_per_cell("2,rest,rest", 3).unwrap()), vec![2, 4, 4]);
        assert!(parse_per_cell("1,2", 3).is_err());
        assert!(parse_per_cell("0,0", 2).is_err());
        assert!(parse_per_cell("1,x", 2).is_err());
    }

    fn topical(i: i64, label: &str, conjunct: bool) -> Candidate {
        Candidate {
            pnode: cmp("x", OpType::Eq, i),
            family: Family::Topical,
            count: 1,
            selectivity: 1e-3,
            source: "t",
            topic: Some((1, label.to_string())),
            conjunct,
        }
    }

    #[test]
    fn drawing_is_seeded_and_backfills_with_conjunctions() {
        let pool: Vec<Candidate> = (0..3)
            .map(|i| topical(i, &format!("t{i}"), false))
            .chain((0..5).map(|i| topical(100 + i, &format!("c{i}"), true)))
            .collect();
        let names = |v: &[&Candidate]| v.iter().map(|c| c.pnode.to_string()).collect::<Vec<_>>();
        let a = draw_distinct(&pool, 6, 7);
        let b = draw_distinct(&pool, 6, 7);
        assert_eq!(a.len(), 6);
        assert_eq!(a.iter().filter(|c| c.conjunct).count(), 3, "singles first, conjunctions make up the rest");
        assert_eq!(names(&a), names(&b), "same seed, same draw");
        let c = draw_distinct(&pool, 6, 8);
        assert_ne!(names(&a), names(&c), "different seed, different order");
        assert_eq!(draw_distinct(&pool, 20, 7).len(), 8, "no more than the pool");
    }

    /// Twelve queries, six in topic `in`, six elsewhere; a cell of four
    /// `in` predicates and six `out` ones, ten slots, mixed placement:
    /// every in-topic pair's query really is in the topic, every
    /// out-of-topic pair's is not, no query is used twice, and the draw
    /// is reproducible.
    #[test]
    fn placement_is_decided_per_pair() {
        let pool: Vec<Candidate> = (0..4)
            .map(|i| topical(i, "in", false))
            .chain((4..10).map(|i| topical(i, "out", false)))
            .collect();
        let per_query: Vec<Vec<(usize, String)>> = (0..12)
            .map(|q| vec![(1usize, if q % 2 == 0 { "in".to_string() } else { "elsewhere".to_string() })])
            .collect();
        let topics = QueryTopics { per_query };
        let run = |seed: u64, placement: Placement| {
            let mut queries = QueryPool::new(12, Some(&topics), seed);
            let mut out = Vec::new();
            let mut next = 0;
            let r = fill_cell(Family::Topical, -3, 10, &pool, seed, placement, Some(&topics), &mut queries, &mut next, &mut out);
            (r, out, queries.remaining())
        };
        let (r, out, left) = run(1, Placement::Mixed);
        assert_eq!((r.filled, r.in_topic, r.out_of_topic, r.shortfall), (10, 5, 5, 0));
        assert_eq!(left, 2);
        let mut seen = std::collections::HashSet::new();
        for (q, d) in &out {
            assert!(seen.insert(*q), "query {q} paired twice");
            let in_topic = q % 2 == 0 && d.candidate.topic.as_ref().unwrap().1 == "in";
            assert_eq!(d.placement, Some(if in_topic { "in-topic" } else { "out-of-topic" }), "query {q} with {}", d.candidate.pnode);
            assert_eq!(d.query_topic.as_deref(), Some(if q % 2 == 0 { "in" } else { "elsewhere" }));
        }
        assert!(out.iter().map(|(_, d)| d.predicate).max().unwrap() < r.drawn);
        let (_, again, _) = run(1, Placement::Mixed);
        assert_eq!(
            out.iter().map(|(q, d)| (*q, d.candidate.pnode.to_string())).collect::<Vec<_>>(),
            again.iter().map(|(q, d)| (*q, d.candidate.pnode.to_string())).collect::<Vec<_>>(),
        );
        // In-topic only: the six `in` queries pair in-topic, the rest
        // of the slots fall back to out-of-topic pairs.
        let (r, _, _) = run(3, Placement::InTopic);
        assert_eq!((r.in_topic, r.out_of_topic, r.filled), (6, 4, 10));
    }

    /// Without query topics a cell pairs its distinct predicates with
    /// queries in draw order, repeating only when the pool is smaller
    /// than the slots.
    #[test]
    fn a_cell_without_topics_repeats_only_under_pool_exhaustion() {
        let pool: Vec<Candidate> = (0..3).map(|i| topical(i, "t", false)).collect();
        let mut queries = QueryPool::new(8, None, 5);
        let mut out = Vec::new();
        let mut next = 0;
        let r = fill_cell(Family::Structural, -2, 8, &pool, 5, Placement::Mixed, None, &mut queries, &mut next, &mut out);
        assert_eq!((r.drawn, r.filled, r.shortfall), (3, 8, 0));
        assert_eq!(out.iter().map(|(_, d)| d.predicate).collect::<std::collections::HashSet<_>>().len(), 3);
        assert!(out.iter().all(|(_, d)| d.placement.is_none()));
        let r = fill_cell(Family::Structural, -2, 4, &[], 5, Placement::Mixed, None, &mut queries, &mut next, &mut out);
        assert_eq!((r.filled, r.shortfall), (0, 4), "an empty pool fills nothing and reports it");
    }
}
