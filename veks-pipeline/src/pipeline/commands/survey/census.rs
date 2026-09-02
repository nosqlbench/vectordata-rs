// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pass 3 — the census.
//!
//! An **exhaustive, exact** counting pass over every record of the
//! slab, for a declared set of fields, hierarchies and field pairs.
//! Passes 1 and 2 are sampled and capped by design; they discover
//! schema and distributions. The census exists because a consumer
//! that stratifies predicates by selectivity needs the *count*, not an
//! estimate, and a 100,000-row sample cannot see a value whose
//! selectivity is 10⁻⁵ at all.
//!
//! What is counted is declared (`census`, `hierarchy`, `census-pair`)
//! and bounded by declaration (`census-cap`, `pair-cells-cap`), never
//! by the sampled regime verdict — a misclassification in Pass 1 must
//! not be able to silence the census.
//!
//! Three accumulators:
//!
//! - **field census** — exact value → count for one field, as an
//!   [`ExactValueCensusReport`]; integer-encoded fields additionally
//!   get a dense [`ExactIntegerHistogramReport`] so a range's
//!   selectivity is a prefix-sum difference.
//! - **hierarchy census** — exact path tuples over an ordered list of
//!   fields, folded into a tree with a count at every node, with the
//!   nesting invariant (every value at level *k*+1 has exactly one
//!   parent) verified as it goes.
//! - **pair census** — the exact joint table of two fields.
//!
//! ## How the pass is shaped
//!
//! Reading a record is the cost, and it is independent per page;
//! counting is cheap and must be sequential so that interned ids,
//! first-seen parents and report order never depend on scheduling.
//! Worker threads therefore walk each page's records with the
//! zero-allocation MNode scanner and extract **only the declared
//! fields** into a per-page array of [`Slot`]s plus one text arena —
//! two allocations per page, none per record. One consumer applies
//! pages strictly in page order, interning values on first sight, so
//! the per-record cost is one hash probe per declared use.
//!
//! See `docs/sysref/13-metadata-survey.md` §13.4 (Pass 3) and the
//! topic-stratified predicate SRD, TS-139 … TS-148.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use slabtastic::{PageEntry, SlabReader};
use veks_core::formats::anode::{self, ANode};
use veks_core::formats::mnode::MValue;
use veks_core::formats::mnode::scan::{self, ScanError};

use super::measures::cardinality::canonical_distinct_key;
use super::progress::{ProgressDriver, SurveyPass};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default cap on distinct values (or histogram width) per censused
/// field.
pub const DEFAULT_CENSUS_CAP: usize = 65_536;

/// Default cap on the cells of one pair's joint table (`|a| × |b|`).
pub const DEFAULT_PAIR_CELLS_CAP: usize = 4_194_304;

/// What the census pass counts. Mirrors the `census`, `census-cap`,
/// `hierarchy`, `census-pair` and `pair-cells-cap` options.
#[derive(Debug, Clone, PartialEq)]
pub struct CensusConfig {
    /// Census every field whose Pass 1 regime already shows it to be
    /// enumerable (`Constant`, `Binary`, `LowCard`, `MidCard`).
    pub auto: bool,
    /// Fields censused regardless of regime.
    pub listed: Vec<String>,
    /// Distinct values (or histogram width) per field before the
    /// field is over cap.
    pub cap: usize,
    /// Ordered field lists, outermost first.
    pub hierarchies: Vec<Vec<String>>,
    /// Field pairs whose joint table is counted.
    pub pairs: Vec<(String, String)>,
    /// Cells per pair table before the pair is over cap.
    pub pair_cells_cap: usize,
    /// Threads extracting pages during the pass; `0` means available
    /// parallelism. Set by the command from the governor's `threads`
    /// resource rather than by an operator option. Counting itself is
    /// sequential whatever this is, so the report does not depend on
    /// it.
    pub threads: usize,
}

impl Default for CensusConfig {
    fn default() -> Self {
        CensusConfig {
            auto: true,
            listed: Vec::new(),
            cap: DEFAULT_CENSUS_CAP,
            hierarchies: Vec::new(),
            pairs: Vec::new(),
            pair_cells_cap: DEFAULT_PAIR_CELLS_CAP,
            threads: 0,
        }
    }
}

impl CensusConfig {
    /// A configuration under which the pass has nothing to count and
    /// is skipped entirely — `census: none` with no hierarchy or pair.
    pub fn is_noop(&self) -> bool {
        !self.auto && self.listed.is_empty() && self.hierarchies.is_empty() && self.pairs.is_empty()
    }

    /// Parse the `census` option: `auto` (default), `none`, or a
    /// comma-separated field list; `auto` may appear in the list to
    /// combine the regime-selected fields with named ones.
    pub fn parse_fields(spec: &str) -> Result<(bool, Vec<String>), String> {
        let mut auto = false;
        let mut listed = Vec::new();
        let mut saw_none = false;
        for raw in spec.split(',') {
            let token = raw.trim();
            if token.is_empty() {
                continue;
            }
            match token {
                "auto" => auto = true,
                "none" => saw_none = true,
                name => {
                    if !listed.iter().any(|l| l == name) {
                        listed.push(name.to_string());
                    }
                }
            }
        }
        if saw_none && (auto || !listed.is_empty()) {
            return Err("census: `none` cannot be combined with `auto` or field names".into());
        }
        if !saw_none && !auto && listed.is_empty() {
            return Err("census: expected `auto`, `none`, or a comma-separated field list".into());
        }
        Ok((auto, listed))
    }

    /// Parse the `hierarchy` option: comma-separated declarations,
    /// each `outer>inner>innermost` with at least two levels.
    pub fn parse_hierarchies(spec: &str) -> Result<Vec<Vec<String>>, String> {
        let mut out = Vec::new();
        for raw in spec.split(',') {
            let decl = raw.trim();
            if decl.is_empty() {
                continue;
            }
            let levels: Vec<String> = decl.split('>').map(|s| s.trim().to_string()).collect();
            if levels.len() < 2 || levels.iter().any(String::is_empty) {
                return Err(format!(
                    "hierarchy: `{}` must name at least two non-empty levels as `a>b>c`",
                    decl
                ));
            }
            let mut seen = std::collections::HashSet::new();
            for level in &levels {
                if !seen.insert(level.as_str()) {
                    return Err(format!("hierarchy: `{}` names `{}` twice", decl, level));
                }
            }
            out.push(levels);
        }
        Ok(out)
    }

    /// Parse the `census-pair` option: comma-separated `a:b`
    /// declarations.
    pub fn parse_pairs(spec: &str) -> Result<Vec<(String, String)>, String> {
        let mut out = Vec::new();
        for raw in spec.split(',') {
            let decl = raw.trim();
            if decl.is_empty() {
                continue;
            }
            let (a, b) = decl
                .split_once(':')
                .ok_or_else(|| format!("census-pair: `{}` must be `a:b`", decl))?;
            let (a, b) = (a.trim(), b.trim());
            if a.is_empty() || b.is_empty() {
                return Err(format!(
                    "census-pair: `{}` must name two non-empty fields",
                    decl
                ));
            }
            if a == b {
                return Err(format!("census-pair: `{}` pairs a field with itself", decl));
            }
            out.push((a.to_string(), b.to_string()));
        }
        Ok(out)
    }

    /// Upper bound on the memory the pass can hold, from the caps —
    /// what the command declares to the governor before the pass
    /// begins. `field_count` is the number of fields that will be
    /// censused (known only after Pass 1 under `auto`); callers
    /// without that knowledge pass the number of listed fields plus
    /// an allowance for `auto` ones.
    pub fn estimated_memory_bytes(&self, field_count: usize) -> u64 {
        // Interned key + count + hash slot per distinct value.
        const PER_VALUE: u64 = 96;
        // Dense histogram slot.
        const PER_BIN: u64 = 8;
        let fields = field_count as u64 * self.cap as u64 * (PER_VALUE + PER_BIN);
        let hierarchies = self
            .hierarchies
            .iter()
            .map(|h| h.len() as u64 * self.cap as u64 * PER_VALUE)
            .sum::<u64>();
        let pairs = self.pairs.len() as u64 * self.pair_cells_cap as u64 * 8;
        fields + hierarchies + pairs
    }
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

/// Exact value → count over the full population of one field.
///
/// Keys are the survey's canonical value key — the same rendering
/// `ExactFrequencyTable` uses — so a consumer that decodes one decodes
/// the other. Ordered by count, descending.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactValueCensusReport {
    /// Non-null observations counted.
    pub population: u64,
    /// Distinct values.
    pub distinct: u32,
    /// Records where the field was null or absent. `population +
    /// missing` equals the slab's record count.
    pub missing: u64,
    /// Exact frequency per value.
    pub counts: IndexMap<String, u64>,
}

/// Dense, ordered exact histogram of an integer field: one count per
/// integer from `min` to `max` inclusive, so a range's selectivity is
/// a prefix-sum difference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactIntegerHistogramReport {
    /// Non-null integer observations counted.
    pub population: u64,
    /// Records where the field was null or absent.
    pub missing: u64,
    /// Smallest observed value; `counts[0]` is its frequency.
    pub min: i64,
    /// Largest observed value; `counts[counts.len() - 1]` is its
    /// frequency.
    pub max: i64,
    /// One count per integer in `min..=max`.
    pub counts: Vec<u64>,
}

/// One node of a hierarchy census tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchyNode {
    /// The value at this level, as its canonical key.
    pub value: String,
    /// Records under this node — for an inner node, the sum of its
    /// children.
    pub count: u64,
    /// Children, ordered by count descending. Empty at the innermost
    /// level.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<HierarchyNode>,
}

/// The verified, counted tree of one declared hierarchy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchyCensusReport {
    /// Levels, outermost first — the declaration as given.
    pub fields: Vec<String>,
    /// Records with every level present and non-null.
    pub population: u64,
    /// Records missing at least one level; not placed in the tree.
    pub incomplete: u64,
    /// Distinct values per level, outermost first.
    pub level_sizes: Vec<u32>,
    /// Top-level nodes, ordered by count descending.
    pub nodes: Vec<HierarchyNode>,
}

/// The exact joint table of one declared pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairCensusReport {
    /// Row field.
    pub a: String,
    /// Column field.
    pub b: String,
    /// Records with both fields present and non-null.
    pub population: u64,
    /// Row labels (canonical keys), in `counts` order.
    pub a_values: Vec<String>,
    /// Column labels (canonical keys), in `counts[i]` order.
    pub b_values: Vec<String>,
    /// `counts[i][j]` is the number of records with `a = a_values[i]`
    /// and `b = b_values[j]`.
    pub counts: Vec<Vec<u64>>,
}

/// A field the census could not keep, and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DroppedField {
    pub field: String,
    pub reason: String,
}

/// Summary of the pass, recorded under `source.census`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CensusInfo {
    /// Records scanned — every record of the slab.
    pub records: u64,
    /// Whether regime-selected fields were included.
    pub auto: bool,
    /// Fields whose exact counts are in the report.
    pub fields: Vec<String>,
    /// `auto` fields that exceeded the cap and left the census.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped: Vec<DroppedField>,
}

// ---------------------------------------------------------------------------
// Slots: what a worker leaves for the consumer
// ---------------------------------------------------------------------------

/// A declared field's value in one record, as the worker extracted
/// it. Text lives in the page's arena, so a slot is `Copy` and a
/// record costs no allocation.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Slot {
    /// The record has no such field.
    Absent,
    /// The field is present and null.
    Null,
    /// An integer-payload wire type (`Int`, `Int32`, `Short`,
    /// `EnumOrd`, `Millis`).
    Int {
        tag: u8,
        v: i64,
    },
    Bool(bool),
    /// A text-payload wire type (`Text`, `Ascii`, `EnumStr`, `Date`,
    /// `Time`, `DateTime`); the bytes are `arena[start..start + len]`.
    Str {
        tag: u8,
        start: u32,
        len: u32,
    },
    /// Any other wire type, already rendered as its canonical key in
    /// the arena.
    Key {
        start: u32,
        len: u32,
    },
}

/// The `MValue` variant name `anode::decode` produces for a wire tag,
/// so census keys read exactly as `ExactFrequencyTable`'s.
fn variant_name(tag: u8) -> &'static str {
    match tag {
        0 | 10 => "Text",
        1 => "Int",
        3 => "Bool",
        6 => "EnumStr",
        7 => "EnumOrd",
        11 => "Ascii",
        12 => "Int32",
        13 => "Short",
        18 => "Millis",
        20 => "Date",
        21 => "Time",
        22 => "DateTime",
        _ => "Value",
    }
}

/// Integer tags that get a dense histogram. `Millis` is an integer on
/// the wire but a timestamp in meaning; its range would never fit.
fn histogram_tag(tag: u8) -> bool {
    matches!(tag, 1 | 7 | 12 | 13)
}

/// The declared fields, in one index space shared by workers (who
/// fill slots by position) and the consumer (who reads them).
struct Layout {
    names: Vec<String>,
    by_name: HashMap<Vec<u8>, usize>,
}

impl Layout {
    fn build(plan: &CensusPlan) -> Layout {
        let mut layout = Layout {
            names: Vec::new(),
            by_name: HashMap::new(),
        };
        let mut add = |name: &str| {
            if !layout.by_name.contains_key(name.as_bytes()) {
                layout
                    .by_name
                    .insert(name.as_bytes().to_vec(), layout.names.len());
                layout.names.push(name.to_string());
            }
        };
        for f in &plan.fields {
            add(&f.name);
        }
        for h in &plan.hierarchies {
            for level in h {
                add(level);
            }
        }
        for (a, b) in &plan.pairs {
            add(a);
            add(b);
        }
        layout
    }

    fn width(&self) -> usize {
        self.names.len()
    }

    fn index_of(&self, name: &str) -> usize {
        self.by_name[name.as_bytes()]
    }
}

enum Extract {
    /// Not an MNode record (a PNode, say); skipped without comment.
    NotMNode,
    /// Malformed; counted as a decode error.
    Malformed,
}

/// Walk one record with the zero-allocation scanner and append its
/// declared fields' slots. Text is copied into `arena` once; nothing
/// else is allocated.
fn extract_record(
    bytes: &[u8],
    layout: &Layout,
    slots: &mut Vec<Slot>,
    arena: &mut String,
) -> Result<(), Extract> {
    let fields = scan::fields(bytes).map_err(|e| match e {
        ScanError::InvalidDialect(_) => Extract::NotMNode,
        _ => Extract::Malformed,
    })?;
    let base = slots.len();
    slots.resize(base + layout.width(), Slot::Absent);
    let mut exotic = false;
    for field in fields {
        let field = match field {
            Ok(f) => f,
            Err(_) => {
                slots.truncate(base);
                return Err(Extract::Malformed);
            }
        };
        let Some(&i) = layout.by_name.get(field.name) else {
            continue;
        };
        let slot = if field.is_null() {
            Slot::Null
        } else if let Some(v) = field.as_i64() {
            Slot::Int { tag: field.tag, v }
        } else if let Some(b) = field.as_bool() {
            Slot::Bool(b)
        } else if let Some(s) = field.as_str() {
            let start = arena.len() as u32;
            arena.push_str(s);
            Slot::Str {
                tag: field.tag,
                start,
                len: s.len() as u32,
            }
        } else {
            exotic = true;
            Slot::Absent
        };
        slots[base + i] = slot;
    }
    if exotic {
        // A declared field of a wire type the scanner does not read
        // directly — float, bytes, uuid, collection. Rare enough to
        // materialise the record once and key those slots exactly as
        // `ExactFrequencyTable` would.
        let mnode = match anode::decode(bytes) {
            Ok(ANode::MNode(m)) => m,
            _ => {
                slots.truncate(base);
                return Err(Extract::Malformed);
            }
        };
        for (i, name) in layout.names.iter().enumerate() {
            if slots[base + i] == Slot::Absent
                && let Some(v) = mnode.fields.get(name.as_str())
                && !matches!(v, MValue::Null)
            {
                let key = canonical_distinct_key(v);
                let start = arena.len() as u32;
                arena.push_str(&key);
                slots[base + i] = Slot::Key {
                    start,
                    len: key.len() as u32,
                };
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Interning
// ---------------------------------------------------------------------------

/// Per-field value interner. Text is looked up by the borrowed arena
/// slice and integers by value, so the hot path allocates nothing;
/// the canonical key is built once, on first sight, and kept per id
/// for the report.
#[derive(Default)]
struct ValueInterner {
    text: HashMap<String, u32>,
    ints: HashMap<i64, u32>,
    /// `[false, true]` slots.
    bools: [Option<u32>; 2],
    other: HashMap<String, u32>,
    keys: Vec<String>,
}

impl ValueInterner {
    fn len(&self) -> usize {
        self.keys.len()
    }

    /// Id for a present, non-null slot, assigning one if unseen.
    /// `None` when the value is unseen and the interner already holds
    /// `cap` values.
    fn intern(&mut self, slot: Slot, arena: &str, cap: usize) -> Option<u32> {
        match slot {
            Slot::Str { tag, start, len } => {
                let s = &arena[start as usize..(start + len) as usize];
                if let Some(id) = self.text.get(s) {
                    return Some(*id);
                }
                if self.keys.len() >= cap {
                    return None;
                }
                let id = self.keys.len() as u32;
                self.keys.push(format!("{}({:?})", variant_name(tag), s));
                self.text.insert(s.to_string(), id);
                Some(id)
            }
            Slot::Int { tag, v } => {
                if let Some(id) = self.ints.get(&v) {
                    return Some(*id);
                }
                if self.keys.len() >= cap {
                    return None;
                }
                let id = self.keys.len() as u32;
                self.keys.push(format!("{}({})", variant_name(tag), v));
                self.ints.insert(v, id);
                Some(id)
            }
            Slot::Bool(b) => {
                let slot = usize::from(b);
                if let Some(id) = self.bools[slot] {
                    return Some(id);
                }
                if self.keys.len() >= cap {
                    return None;
                }
                let id = self.keys.len() as u32;
                self.keys.push(format!("Bool({})", b));
                self.bools[slot] = Some(id);
                Some(id)
            }
            Slot::Key { start, len } => {
                let key = &arena[start as usize..(start + len) as usize];
                if let Some(id) = self.other.get(key) {
                    return Some(*id);
                }
                if self.keys.len() >= cap {
                    return None;
                }
                let id = self.keys.len() as u32;
                self.keys.push(key.to_string());
                self.other.insert(key.to_string(), id);
                Some(id)
            }
            Slot::Absent | Slot::Null => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Field census
// ---------------------------------------------------------------------------

/// Dense integer histogram that grows to cover the observed range and
/// stops growing — flagging overflow — once the range would exceed
/// its cap.
struct IntHistogram {
    min: i64,
    max: i64,
    counts: Vec<u64>,
    cap: usize,
    overflow: bool,
    populated: bool,
    population: u64,
}

impl IntHistogram {
    fn new(cap: usize) -> Self {
        IntHistogram {
            min: 0,
            max: 0,
            counts: Vec::new(),
            cap,
            overflow: false,
            populated: false,
            population: 0,
        }
    }

    fn observe(&mut self, v: i64) {
        if !self.populated {
            self.populated = true;
            self.min = v;
            self.max = v;
            self.counts = vec![1];
            self.population = 1;
            return;
        }
        if v >= self.min && v <= self.max {
            self.counts[(v - self.min) as usize] += 1;
            self.population += 1;
            return;
        }
        if self.overflow {
            return;
        }
        let new_min = self.min.min(v);
        let new_max = self.max.max(v);
        let width = (new_max as i128 - new_min as i128 + 1) as u128;
        if width > self.cap as u128 {
            self.overflow = true;
            return;
        }
        if v < self.min {
            let grow = (self.min - v) as usize;
            let mut fresh = vec![0u64; grow];
            fresh.append(&mut self.counts);
            self.counts = fresh;
            self.min = v;
        } else {
            let grow = (v - self.max) as usize;
            self.counts.resize(self.counts.len() + grow, 0);
            self.max = v;
        }
        self.counts[(v - self.min) as usize] += 1;
        self.population += 1;
    }
}

/// Accumulator for one censused field.
struct FieldCensus {
    name: String,
    slot: usize,
    listed: bool,
    values: ValueInterner,
    counts: Vec<u64>,
    value_overflow: bool,
    histogram: Option<IntHistogram>,
    present: u64,
    nulls: u64,
    absent: u64,
    cap: usize,
}

impl FieldCensus {
    fn new(name: &str, slot: usize, listed: bool, integer: bool, cap: usize) -> Self {
        FieldCensus {
            name: name.to_string(),
            slot,
            listed,
            values: ValueInterner::default(),
            counts: Vec::new(),
            value_overflow: false,
            histogram: if integer {
                Some(IntHistogram::new(cap))
            } else {
                None
            },
            present: 0,
            nulls: 0,
            absent: 0,
            cap,
        }
    }

    fn observe(&mut self, slot: Slot, arena: &str) {
        match slot {
            Slot::Absent => {
                self.absent += 1;
                return;
            }
            Slot::Null => {
                self.nulls += 1;
                return;
            }
            _ => {}
        }
        self.present += 1;
        if !self.value_overflow {
            match self.values.intern(slot, arena, self.cap) {
                Some(id) => {
                    let id = id as usize;
                    if id >= self.counts.len() {
                        self.counts.resize(id + 1, 0);
                    }
                    self.counts[id] += 1;
                }
                None => self.value_overflow = true,
            }
        }
        if let Some(h) = self.histogram.as_mut()
            && let Slot::Int { tag, v } = slot
            && histogram_tag(tag)
        {
            h.observe(v);
        }
    }

    fn finalize(self, total_records: u64) -> FieldCensusResult {
        let missing = total_records.saturating_sub(self.present);
        let mut dropped = Vec::new();
        let value = if self.value_overflow {
            dropped.push(format!(
                "more than {} distinct values; not enumerable at this cap",
                self.cap
            ));
            None
        } else {
            let mut pairs: Vec<(String, u64)> = self
                .values
                .keys
                .iter()
                .cloned()
                .zip(self.counts.iter().copied())
                .collect();
            // Stable on ties so the report is reproducible.
            pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let distinct = pairs.len() as u32;
            Some(ExactValueCensusReport {
                population: self.present,
                distinct,
                missing,
                counts: pairs.into_iter().collect(),
            })
        };
        let histogram = match self.histogram {
            Some(h) if h.overflow => {
                dropped.push(format!(
                    "integer range wider than {}; no dense histogram",
                    self.cap
                ));
                None
            }
            Some(h) if h.populated => Some(ExactIntegerHistogramReport {
                population: h.population,
                missing,
                min: h.min,
                max: h.max,
                counts: h.counts,
            }),
            _ => None,
        };
        FieldCensusResult {
            name: self.name,
            listed: self.listed,
            present: self.present,
            nulls: self.nulls,
            absent: self.absent,
            value,
            histogram,
            dropped,
        }
    }
}

/// What the pass produced for one field.
#[derive(Debug)]
pub struct FieldCensusResult {
    pub name: String,
    pub listed: bool,
    /// Non-null observations.
    pub present: u64,
    /// Explicit nulls.
    pub nulls: u64,
    /// Records without the field.
    pub absent: u64,
    /// The value table, unless the field exceeded the cap.
    pub value: Option<ExactValueCensusReport>,
    /// The dense histogram, for integer fields within the cap.
    pub histogram: Option<ExactIntegerHistogramReport>,
    /// Reasons a table was not produced.
    pub dropped: Vec<String>,
}

// ---------------------------------------------------------------------------
// Hierarchy census
// ---------------------------------------------------------------------------

/// Bound on the nesting violations retained for the error message.
const MAX_REPORTED_VIOLATIONS: usize = 8;

struct HierarchyCensus {
    fields: Vec<String>,
    slots: Vec<usize>,
    interners: Vec<ValueInterner>,
    /// `parent[k][id]` for `k >= 1`: the id at level `k - 1` this
    /// value was first seen under.
    parent: Vec<Vec<Option<u32>>>,
    leaf_counts: Vec<u64>,
    population: u64,
    incomplete: u64,
    violations: Vec<String>,
    violation_count: u64,
    over_cap_level: Option<usize>,
    cap: usize,
}

impl HierarchyCensus {
    fn new(fields: &[String], slots: Vec<usize>, cap: usize) -> Self {
        HierarchyCensus {
            fields: fields.to_vec(),
            slots,
            interners: fields.iter().map(|_| ValueInterner::default()).collect(),
            parent: fields.iter().map(|_| Vec::new()).collect(),
            leaf_counts: Vec::new(),
            population: 0,
            incomplete: 0,
            violations: Vec::new(),
            violation_count: 0,
            over_cap_level: None,
            cap,
        }
    }

    fn observe(&mut self, row: &[Slot], arena: &str) {
        if self.over_cap_level.is_some() {
            return;
        }
        // Every level must be present and non-null for the record to
        // have a place in the tree.
        if self
            .slots
            .iter()
            .any(|&s| matches!(row[s], Slot::Absent | Slot::Null))
        {
            self.incomplete += 1;
            return;
        }
        let mut prev: Option<u32> = None;
        let mut leaf: u32 = 0;
        for k in 0..self.fields.len() {
            let id = match self.interners[k].intern(row[self.slots[k]], arena, self.cap) {
                Some(id) => id,
                None => {
                    self.over_cap_level = Some(k);
                    return;
                }
            };
            if k > 0 {
                let parents = &mut self.parent[k];
                if id as usize >= parents.len() {
                    parents.resize(id as usize + 1, None);
                }
                match parents[id as usize] {
                    None => parents[id as usize] = prev,
                    Some(p) if Some(p) == prev => {}
                    Some(p) => {
                        self.violation_count += 1;
                        if self.violations.len() < MAX_REPORTED_VIOLATIONS {
                            let first = &self.interners[k - 1].keys[p as usize];
                            let second = &self.interners[k - 1].keys[prev.unwrap() as usize];
                            self.violations.push(format!(
                                "{} = {} seen under {} = {} and {} = {}",
                                self.fields[k],
                                self.interners[k].keys[id as usize],
                                self.fields[k - 1],
                                first,
                                self.fields[k - 1],
                                second
                            ));
                        }
                    }
                }
            }
            prev = Some(id);
            leaf = id;
        }
        if leaf as usize >= self.leaf_counts.len() {
            self.leaf_counts.resize(leaf as usize + 1, 0);
        }
        self.leaf_counts[leaf as usize] += 1;
        self.population += 1;
    }

    fn finalize(self) -> Result<HierarchyCensusReport, String> {
        let decl = self.fields.join(">");
        if let Some(k) = self.over_cap_level {
            return Err(format!(
                "hierarchy {}: level `{}` has more than {} distinct values; raise census-cap or drop the declaration",
                decl, self.fields[k], self.cap
            ));
        }
        if self.violation_count > 0 {
            return Err(format!(
                "hierarchy {}: nesting violated {} time(s) — a value has more than one parent; first cases: {}",
                decl,
                self.violation_count,
                self.violations.join("; ")
            ));
        }
        let depth = self.fields.len();
        // children[k][parent_id] -> ids at level k+1
        let mut children: Vec<Vec<Vec<u32>>> = (0..depth)
            .map(|k| vec![Vec::new(); self.interners[k].len()])
            .collect();
        for k in 1..depth {
            for (id, parent) in self.parent[k].iter().enumerate() {
                if let Some(p) = parent {
                    children[k - 1][*p as usize].push(id as u32);
                }
            }
        }
        fn build(
            level: usize,
            id: u32,
            depth: usize,
            interners: &[ValueInterner],
            children: &[Vec<Vec<u32>>],
            leaf_counts: &[u64],
        ) -> HierarchyNode {
            let value = interners[level].keys[id as usize].clone();
            if level + 1 == depth {
                let count = leaf_counts.get(id as usize).copied().unwrap_or(0);
                return HierarchyNode {
                    value,
                    count,
                    children: Vec::new(),
                };
            }
            let mut kids: Vec<HierarchyNode> = children[level][id as usize]
                .iter()
                .map(|c| build(level + 1, *c, depth, interners, children, leaf_counts))
                .collect();
            kids.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));
            let count = kids.iter().map(|k| k.count).sum();
            HierarchyNode {
                value,
                count,
                children: kids,
            }
        }
        let mut nodes: Vec<HierarchyNode> = (0..self.interners[0].len() as u32)
            .map(|id| build(0, id, depth, &self.interners, &children, &self.leaf_counts))
            .collect();
        nodes.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));
        Ok(HierarchyCensusReport {
            fields: self.fields,
            population: self.population,
            incomplete: self.incomplete,
            level_sizes: self.interners.iter().map(|i| i.len() as u32).collect(),
            nodes,
        })
    }
}

// ---------------------------------------------------------------------------
// Pair census
// ---------------------------------------------------------------------------

struct PairCensus {
    a: String,
    b: String,
    slot_a: usize,
    slot_b: usize,
    ia: ValueInterner,
    ib: ValueInterner,
    cells: HashMap<(u32, u32), u64>,
    population: u64,
    cells_cap: usize,
    value_cap: usize,
    over_cap: bool,
}

impl PairCensus {
    fn new(
        a: &str,
        b: &str,
        slot_a: usize,
        slot_b: usize,
        cells_cap: usize,
        value_cap: usize,
    ) -> Self {
        PairCensus {
            a: a.to_string(),
            b: b.to_string(),
            slot_a,
            slot_b,
            ia: ValueInterner::default(),
            ib: ValueInterner::default(),
            cells: HashMap::new(),
            population: 0,
            cells_cap,
            value_cap,
            over_cap: false,
        }
    }

    fn observe(&mut self, row: &[Slot], arena: &str) {
        if self.over_cap {
            return;
        }
        let (av, bv) = (row[self.slot_a], row[self.slot_b]);
        if matches!(av, Slot::Absent | Slot::Null) || matches!(bv, Slot::Absent | Slot::Null) {
            return;
        }
        let (Some(ai), Some(bi)) = (
            self.ia.intern(av, arena, self.value_cap),
            self.ib.intern(bv, arena, self.value_cap),
        ) else {
            self.over_cap = true;
            return;
        };
        if self.ia.len() * self.ib.len() > self.cells_cap {
            self.over_cap = true;
            return;
        }
        *self.cells.entry((ai, bi)).or_insert(0) += 1;
        self.population += 1;
    }

    fn finalize(self) -> Result<PairCensusReport, String> {
        if self.over_cap {
            return Err(format!(
                "census-pair {}:{}: joint table would exceed {} cells (or a side exceeds {} values); raise pair-cells-cap or drop the declaration",
                self.a, self.b, self.cells_cap, self.value_cap
            ));
        }
        let rows = self.ia.len();
        let cols = self.ib.len();
        let mut counts = vec![vec![0u64; cols]; rows];
        for ((ai, bi), n) in self.cells {
            counts[ai as usize][bi as usize] = n;
        }
        Ok(PairCensusReport {
            a: self.a,
            b: self.b,
            population: self.population,
            a_values: self.ia.keys,
            b_values: self.ib.keys,
            counts,
        })
    }
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

/// One field the pass will census.
#[derive(Debug, Clone, PartialEq)]
pub struct CensusFieldPlan {
    pub name: String,
    /// Named by the operator rather than selected by regime; over cap
    /// is an error rather than a drop.
    pub listed: bool,
    /// Integer-encoded, so a dense histogram is produced as well.
    pub integer: bool,
}

/// What the pass counts, resolved against the survey's fields.
#[derive(Debug, Clone, PartialEq)]
pub struct CensusPlan {
    pub fields: Vec<CensusFieldPlan>,
    pub hierarchies: Vec<Vec<String>>,
    pub pairs: Vec<(String, String)>,
    pub cap: usize,
    pub pair_cells_cap: usize,
}

/// Everything the pass produced.
#[derive(Debug)]
pub struct CensusOutcome {
    /// Records scanned.
    pub records: u64,
    pub fields: Vec<FieldCensusResult>,
    pub hierarchies: Vec<HierarchyCensusReport>,
    pub pairs: Vec<PairCensusReport>,
    /// Records that failed to decode and were skipped.
    pub decode_errors: u64,
}

/// One page, extracted off the consumer thread.
struct ExtractedPage {
    index: usize,
    /// `records × width` slots, row-major.
    slots: Vec<Slot>,
    arena: String,
    /// MNode records extracted.
    records: u64,
    /// Records in the page, decodable or not — what progress counts.
    page_records: u64,
    decode_errors: u64,
    /// A page that could not be read; the pass fails on it, in order.
    error: Option<String>,
}

/// Read one page and extract the declared fields of every record.
/// This is the cost of the pass, and what the worker threads do.
fn extract_page(
    reader: &SlabReader,
    index: usize,
    entry: &PageEntry,
    layout: &Layout,
) -> ExtractedPage {
    let page = match reader.read_data_page(entry) {
        Ok(p) => p,
        Err(e) => {
            return ExtractedPage {
                index,
                slots: Vec::new(),
                arena: String::new(),
                records: 0,
                page_records: 0,
                decode_errors: 0,
                error: Some(format!(
                    "census: failed to read page at offset {}: {}",
                    entry.file_offset, e
                )),
            };
        }
    };
    let n = page.record_count();
    let mut slots = Vec::with_capacity(n * layout.width());
    let mut arena = String::new();
    let mut records = 0u64;
    let mut decode_errors = 0u64;
    for i in 0..n {
        let Some(bytes) = page.get_record(i) else {
            continue;
        };
        match extract_record(bytes, layout, &mut slots, &mut arena) {
            Ok(()) => records += 1,
            Err(Extract::NotMNode) => {}
            Err(Extract::Malformed) => decode_errors += 1,
        }
    }
    ExtractedPage {
        index,
        slots,
        arena,
        records,
        page_records: n as u64,
        decode_errors,
        error: None,
    }
}

/// The counting state. Applied to pages strictly in page order, so
/// interned ids, first-seen parents and report order never depend on
/// how extraction was scheduled.
struct Accumulators {
    width: usize,
    fields: Vec<FieldCensus>,
    hierarchies: Vec<HierarchyCensus>,
    pairs: Vec<PairCensus>,
    records: u64,
    decode_errors: u64,
}

impl Accumulators {
    fn new(plan: &CensusPlan, layout: &Layout) -> Self {
        Accumulators {
            width: layout.width(),
            fields: plan
                .fields
                .iter()
                .map(|f| {
                    FieldCensus::new(
                        &f.name,
                        layout.index_of(&f.name),
                        f.listed,
                        f.integer,
                        plan.cap,
                    )
                })
                .collect(),
            hierarchies: plan
                .hierarchies
                .iter()
                .map(|h| {
                    let slots = h.iter().map(|level| layout.index_of(level)).collect();
                    HierarchyCensus::new(h, slots, plan.cap)
                })
                .collect(),
            pairs: plan
                .pairs
                .iter()
                .map(|(a, b)| {
                    PairCensus::new(
                        a,
                        b,
                        layout.index_of(a),
                        layout.index_of(b),
                        plan.pair_cells_cap,
                        plan.cap,
                    )
                })
                .collect(),
            records: 0,
            decode_errors: 0,
        }
    }

    fn apply(&mut self, page: ExtractedPage, progress: &mut ProgressDriver) -> Result<(), String> {
        if let Some(e) = page.error {
            return Err(e);
        }
        self.decode_errors += page.decode_errors;
        self.records += page.records;
        let arena = page.arena.as_str();
        if self.width > 0 {
            for row in page.slots.chunks_exact(self.width) {
                for f in self.fields.iter_mut() {
                    f.observe(row[f.slot], arena);
                }
                for h in self.hierarchies.iter_mut() {
                    h.observe(row, arena);
                }
                for p in self.pairs.iter_mut() {
                    p.observe(row, arena);
                }
            }
        }
        progress.tick(page.page_records);
        Ok(())
    }
}

/// Run the census pass over every page of `reader`.
///
/// `threads` extract pages in parallel (`0` = available parallelism);
/// one consumer applies them in page order, so the outcome is
/// identical for any thread count.
///
/// Errors — a listed field over cap, a hierarchy that is not a tree,
/// a pair over its cells cap, an unreadable page — abort the survey
/// rather than produce a report that presents a truncated count as
/// exact.
pub fn run_census_pass(
    reader: &SlabReader,
    page_entries: &[PageEntry],
    total_records: u64,
    plan: &CensusPlan,
    threads: usize,
    progress: &mut ProgressDriver,
) -> Result<CensusOutcome, String> {
    let layout = Layout::build(plan);
    let layout = &layout;
    let mut acc = Accumulators::new(plan, layout);
    let threads = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    };
    reader.advise_sequential();
    progress.begin_pass(SurveyPass::Census, total_records);

    if threads <= 1 || page_entries.len() <= 1 {
        for (index, entry) in page_entries.iter().enumerate() {
            acc.apply(extract_page(reader, index, entry, layout), progress)?;
        }
    } else {
        // Workers pull page indices from a shared counter and send
        // extracted pages over a bounded channel; the consumer applies
        // them strictly in index order, holding early arrivals aside.
        // The bound keeps the held-aside set to a few pages per
        // worker. Dropping the receiver — on an error return — makes
        // every pending send fail, which is how the workers stop.
        let next = AtomicUsize::new(0);
        let (tx, rx) = mpsc::sync_channel::<ExtractedPage>(threads * 2);
        let outcome: Result<(), String> = std::thread::scope(|scope| {
            for _ in 0..threads {
                let tx = tx.clone();
                let next = &next;
                scope.spawn(move || {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(entry) = page_entries.get(index) else {
                            break;
                        };
                        if tx.send(extract_page(reader, index, entry, layout)).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(tx);
            let mut held: BTreeMap<usize, ExtractedPage> = BTreeMap::new();
            let mut next_to_apply = 0usize;
            for page in rx {
                held.insert(page.index, page);
                while let Some(page) = held.remove(&next_to_apply) {
                    acc.apply(page, progress)?;
                    next_to_apply += 1;
                }
            }
            Ok(())
        });
        outcome?;
    }
    progress.end_pass();

    let Accumulators {
        fields,
        hierarchies,
        pairs,
        records,
        decode_errors,
        ..
    } = acc;
    let mut field_results = Vec::with_capacity(fields.len());
    for f in fields {
        let listed = f.listed;
        // `missing` is measured against the slab's record count so that
        // `population + missing == source.total_records` is the
        // artifact's own integrity check (SRD TS-129).
        let r = f.finalize(total_records);
        if listed && r.value.is_none() {
            return Err(format!(
                "census: listed field `{}` has more than {} distinct values; it is not enumerable at this cap — raise census-cap or remove it from `census`",
                r.name, plan.cap
            ));
        }
        field_results.push(r);
    }
    let hierarchies = hierarchies
        .into_iter()
        .map(HierarchyCensus::finalize)
        .collect::<Result<Vec<_>, _>>()?;
    let pairs = pairs
        .into_iter()
        .map(PairCensus::finalize)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(CensusOutcome {
        records,
        fields: field_results,
        hierarchies,
        pairs,
        decode_errors,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use veks_core::formats::mnode::MNode;

    /// A layout over the given names, in order.
    fn layout(names: &[&str]) -> Layout {
        let plan = CensusPlan {
            fields: names
                .iter()
                .map(|n| CensusFieldPlan {
                    name: n.to_string(),
                    listed: false,
                    integer: false,
                })
                .collect(),
            hierarchies: vec![],
            pairs: vec![],
            cap: 100,
            pair_cells_cap: 100,
        };
        Layout::build(&plan)
    }

    /// Encode an MNode and extract it against `layout`, the way a
    /// worker would.
    fn row(layout: &Layout, fields: &[(&str, MValue)]) -> (Vec<Slot>, String) {
        let mut map = IndexMap::new();
        for (k, v) in fields {
            map.insert((*k).to_string(), v.clone());
        }
        let bytes = anode::encode(&ANode::MNode(MNode { fields: map }));
        let mut slots = Vec::new();
        let mut arena = String::new();
        assert!(
            extract_record(&bytes, layout, &mut slots, &mut arena).is_ok(),
            "an MNode extracts"
        );
        (slots, arena)
    }

    #[test]
    fn parse_fields_forms() {
        assert_eq!(CensusConfig::parse_fields("auto").unwrap(), (true, vec![]));
        assert_eq!(CensusConfig::parse_fields("none").unwrap(), (false, vec![]));
        assert_eq!(
            CensusConfig::parse_fields("auto, topic_l3").unwrap(),
            (true, vec!["topic_l3".to_string()])
        );
        assert_eq!(
            CensusConfig::parse_fields("a,b,a").unwrap(),
            (false, vec!["a".to_string(), "b".to_string()])
        );
        assert!(CensusConfig::parse_fields("none,a").is_err());
        assert!(CensusConfig::parse_fields("").is_err());
    }

    #[test]
    fn parse_hierarchies_and_pairs() {
        assert_eq!(
            CensusConfig::parse_hierarchies("l1>l2>l3, x>y").unwrap(),
            vec![
                vec!["l1".to_string(), "l2".into(), "l3".into()],
                vec!["x".into(), "y".into()]
            ]
        );
        assert!(CensusConfig::parse_hierarchies("solo").is_err());
        assert!(CensusConfig::parse_hierarchies("a>a").is_err());
        assert_eq!(
            CensusConfig::parse_pairs("a:b, c:d").unwrap(),
            vec![("a".to_string(), "b".to_string()), ("c".into(), "d".into())]
        );
        assert!(CensusConfig::parse_pairs("a").is_err());
        assert!(CensusConfig::parse_pairs("a:a").is_err());
    }

    /// Every wire type the scanner reads directly, and two it does
    /// not, intern to exactly the key `ExactFrequencyTable` would
    /// have produced for the same value.
    #[test]
    fn interned_keys_match_the_canonical_key_for_every_wire_type() {
        let values = vec![
            MValue::Text("plain".into()),
            MValue::Ascii("ascii".into()),
            MValue::EnumStr("enum".into()),
            MValue::Text("quote\"inside".into()),
            MValue::Int(-42),
            MValue::Int32(7),
            MValue::Short(-3),
            MValue::EnumOrd(9),
            MValue::Millis(1_700_000_000_000),
            MValue::Bool(true),
            MValue::Bool(false),
            MValue::Date("2024-01-02".into()),
            MValue::Float(2.5),
            MValue::Bytes(vec![1, 2, 3]),
        ];
        let lay = layout(&["v"]);
        for value in values {
            let (slots, arena) = row(&lay, &[("v", value.clone())]);
            let mut interner = ValueInterner::default();
            let id = interner.intern(slots[0], &arena, 10).expect("interns");
            assert_eq!(
                interner.keys[id as usize],
                canonical_distinct_key(&value),
                "key for {:?}",
                value
            );
        }
    }

    #[test]
    fn extraction_marks_absent_null_and_skips_undeclared_fields() {
        let lay = layout(&["a", "b", "c"]);
        let (slots, arena) = row(
            &lay,
            &[
                ("b", MValue::Null),
                ("z", MValue::Int(1)),
                ("a", MValue::Text("x".into())),
            ],
        );
        assert_eq!(slots.len(), 3);
        assert!(matches!(slots[0], Slot::Str { tag: 0, .. }));
        assert_eq!(slots[1], Slot::Null);
        assert_eq!(slots[2], Slot::Absent);
        assert_eq!(arena, "x");
    }

    #[test]
    fn extraction_rejects_non_mnode_bytes() {
        let lay = layout(&["a"]);
        let mut slots = Vec::new();
        let mut arena = String::new();
        assert!(matches!(
            extract_record(&[0xff, 0, 0], &lay, &mut slots, &mut arena),
            Err(Extract::NotMNode)
        ));
        assert!(matches!(
            extract_record(&[], &lay, &mut slots, &mut arena),
            Err(Extract::Malformed)
        ));
        assert!(slots.is_empty(), "nothing appended for a rejected record");
    }

    #[test]
    fn interner_shares_ids_and_caps() {
        let lay = layout(&["v"]);
        let mut i = ValueInterner::default();
        let (s1, a1) = row(&lay, &[("v", MValue::Text("x".into()))]);
        let a = i.intern(s1[0], &a1, 2).unwrap();
        assert_eq!(i.intern(s1[0], &a1, 2).unwrap(), a);
        let (s2, a2) = row(&lay, &[("v", MValue::Int(7))]);
        let b = i.intern(s2[0], &a2, 2).unwrap();
        assert_ne!(a, b);
        let (s3, a3) = row(&lay, &[("v", MValue::Int32(7))]);
        assert_eq!(i.intern(s3[0], &a3, 2).unwrap(), b, "same integer, same id");
        let (s4, a4) = row(&lay, &[("v", MValue::Text("y".into()))]);
        assert!(
            i.intern(s4[0], &a4, 2).is_none(),
            "third distinct is over cap"
        );
        assert_eq!(i.keys[a as usize], "Text(\"x\")");

        let mut b = ValueInterner::default();
        let t = b.intern(Slot::Bool(true), "", 4).unwrap();
        let f = b.intern(Slot::Bool(false), "", 4).unwrap();
        assert_ne!(t, f);
        assert_eq!(b.intern(Slot::Bool(true), "", 4).unwrap(), t);
        assert_eq!(b.keys[t as usize], "Bool(true)");
        assert!(
            b.other.is_empty() && b.text.is_empty(),
            "booleans take the slot path"
        );
    }

    #[test]
    fn int_histogram_grows_both_ways_and_caps() {
        let mut h = IntHistogram::new(10);
        h.observe(5);
        h.observe(3);
        h.observe(8);
        assert_eq!((h.min, h.max), (3, 8));
        assert_eq!(h.counts, vec![1, 0, 1, 0, 0, 1]);
        h.observe(100);
        assert!(h.overflow);
        assert_eq!(h.population, 3, "the over-cap value is not counted");
        h.observe(4);
        assert_eq!(h.counts[1], 1, "in-range values still count after overflow");
    }

    #[test]
    fn field_census_counts_presence_and_values() {
        let lay = layout(&["k"]);
        let mut f = FieldCensus::new("k", 0, false, true, 16);
        for value in [
            Some(MValue::Int(2)),
            Some(MValue::Int(2)),
            Some(MValue::Int(9)),
            Some(MValue::Null),
            None,
        ] {
            let (slots, arena) = match value {
                Some(v) => row(&lay, &[("k", v)]),
                None => row(&lay, &[]),
            };
            f.observe(slots[0], &arena);
        }
        let r = f.finalize(5);
        assert_eq!((r.present, r.nulls, r.absent), (3, 1, 1));
        let v = r.value.unwrap();
        assert_eq!(v.population, 3);
        assert_eq!(v.missing, 2);
        assert_eq!(v.distinct, 2);
        assert_eq!(v.counts.get_index(0).unwrap(), (&"Int(2)".to_string(), &2));
        let h = r.histogram.unwrap();
        assert_eq!((h.min, h.max), (2, 9));
        assert_eq!(h.counts.iter().sum::<u64>(), 3);
        assert!(r.dropped.is_empty());
    }

    #[test]
    fn field_census_over_cap_drops_value_table_with_reason() {
        let lay = layout(&["k"]);
        let mut f = FieldCensus::new("k", 0, false, false, 2);
        for i in 0..5 {
            let (slots, arena) = row(&lay, &[("k", MValue::Text(format!("v{}", i)))]);
            f.observe(slots[0], &arena);
        }
        let r = f.finalize(5);
        assert!(r.value.is_none());
        assert_eq!(r.present, 5, "presence is still exact");
        assert_eq!(r.dropped.len(), 1);
    }

    #[test]
    fn millis_is_censused_but_never_histogrammed() {
        let lay = layout(&["t"]);
        let mut f = FieldCensus::new("t", 0, false, true, 16);
        let (slots, arena) = row(&lay, &[("t", MValue::Millis(5))]);
        f.observe(slots[0], &arena);
        let r = f.finalize(1);
        assert_eq!(r.value.unwrap().counts.get_index(0).unwrap().0, "Millis(5)");
        assert!(r.histogram.is_none());
    }

    #[test]
    fn hierarchy_builds_tree_and_detects_violation() {
        let lay = layout(&["l1", "l2"]);
        let fields = vec!["l1".to_string(), "l2".to_string()];
        let mut h = HierarchyCensus::new(&fields, vec![0, 1], 100);
        for _ in 0..3 {
            let (s, a) = row(
                &lay,
                &[
                    ("l1", MValue::Text("a".into())),
                    ("l2", MValue::Text("a1".into())),
                ],
            );
            h.observe(&s, &a);
        }
        let (s, a) = row(
            &lay,
            &[
                ("l1", MValue::Text("a".into())),
                ("l2", MValue::Text("a2".into())),
            ],
        );
        h.observe(&s, &a);
        let (s, a) = row(
            &lay,
            &[
                ("l1", MValue::Text("b".into())),
                ("l2", MValue::Text("b1".into())),
            ],
        );
        h.observe(&s, &a);
        let (s, a) = row(&lay, &[("l1", MValue::Text("b".into()))]);
        h.observe(&s, &a);
        let r = h.finalize().unwrap();
        assert_eq!(r.population, 5);
        assert_eq!(r.incomplete, 1);
        assert_eq!(r.level_sizes, vec![2, 3]);
        assert_eq!(r.nodes[0].value, "Text(\"a\")");
        assert_eq!(r.nodes[0].count, 4);
        assert_eq!(r.nodes[0].children[0].value, "Text(\"a1\")");
        assert_eq!(r.nodes[0].children[0].count, 3);
        assert_eq!(r.nodes[1].count, 1);

        let mut bad = HierarchyCensus::new(&fields, vec![0, 1], 100);
        let (s, a) = row(
            &lay,
            &[
                ("l1", MValue::Text("a".into())),
                ("l2", MValue::Text("shared".into())),
            ],
        );
        bad.observe(&s, &a);
        let (s, a) = row(
            &lay,
            &[
                ("l1", MValue::Text("b".into())),
                ("l2", MValue::Text("shared".into())),
            ],
        );
        bad.observe(&s, &a);
        let err = bad.finalize().unwrap_err();
        assert!(err.contains("nesting violated"), "{}", err);
        assert!(err.contains("shared"), "{}", err);
    }

    #[test]
    fn pair_builds_dense_table_and_caps() {
        let lay = layout(&["a", "b"]);
        let mut p = PairCensus::new("a", "b", 0, 1, 100, 100);
        for (a, b) in [("x", Some(1)), ("x", Some(2)), ("y", Some(1)), ("y", None)] {
            let mut fields = vec![("a", MValue::Text(a.into()))];
            if let Some(b) = b {
                fields.push(("b", MValue::Int(b)));
            }
            let (s, ar) = row(&lay, &fields);
            p.observe(&s, &ar);
        }
        let r = p.finalize().unwrap();
        assert_eq!(r.population, 3);
        assert_eq!(r.a_values, vec!["Text(\"x\")", "Text(\"y\")"]);
        assert_eq!(r.b_values, vec!["Int(1)", "Int(2)"]);
        assert_eq!(r.counts, vec![vec![1, 1], vec![1, 0]]);

        let mut tight = PairCensus::new("a", "b", 0, 1, 2, 100);
        for (a, b) in [(1, 1), (1, 2), (2, 1)] {
            let (s, ar) = row(&lay, &[("a", MValue::Int(a)), ("b", MValue::Int(b))]);
            tight.observe(&s, &ar);
        }
        assert!(tight.finalize().is_err());
    }

    #[test]
    fn layout_dedups_names_across_declarations() {
        let plan = CensusPlan {
            fields: vec![CensusFieldPlan {
                name: "l2".into(),
                listed: false,
                integer: false,
            }],
            hierarchies: vec![vec!["l1".into(), "l2".into()]],
            pairs: vec![("l2".into(), "year".into())],
            cap: 10,
            pair_cells_cap: 10,
        };
        let lay = Layout::build(&plan);
        assert_eq!(lay.names, vec!["l2", "l1", "year"]);
        assert_eq!(lay.index_of("year"), 2);
    }

    #[test]
    fn memory_estimate_scales_with_declarations() {
        let base = CensusConfig::default();
        let with_pair = CensusConfig {
            pairs: vec![("a".into(), "b".into())],
            ..CensusConfig::default()
        };
        assert!(with_pair.estimated_memory_bytes(3) > base.estimated_memory_bytes(3));
        assert_eq!(
            with_pair.estimated_memory_bytes(3) - base.estimated_memory_bytes(3),
            DEFAULT_PAIR_CELLS_CAP as u64 * 8
        );
    }
}
