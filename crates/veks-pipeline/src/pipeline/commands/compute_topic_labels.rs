// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `compute topic-labels` — names every topic cluster from the most
//! distinctive terms of its member passages.
//!
//! Labels are cosmetic to the measurement and carry the whole
//! credibility argument: a predicate reads `topic_l2 =
//! 'photovoltaic-grid-integration'`, and a person has to be able to
//! say what it asks for without reference to any document. So labels
//! are generated from the text, never load-bearing for correctness,
//! and unique within a level because they are stored as comparands.
//!
//! The method is class-based TF-IDF: term frequencies are aggregated
//! per cluster from a sample of its member passages and weighted
//! against the term's frequency across every cluster at the same
//! level, so a term that is everywhere scores nothing and a term that
//! is one cluster's own scores high. Unigrams and bigrams both
//! compete; the top terms are joined with hyphens into a slug.
//!
//! The sample is a **seeded subset of row groups** of the passage
//! table, not a sample of rows. The passage table is hundreds of row
//! groups of about a million rows each, so two thousand uniformly
//! sampled members of a typical leaf cluster would touch every group
//! and read the whole file. Reading a fixed number of groups bounds
//! the read; within each group rows are visited in a seeded order and
//! accepted per level until the cluster's cap. A cluster that meets
//! fewer than `min-sample` rows is given a positional label rather
//! than one fitted to noise.
//!
//! See the topic-stratified predicate SRD, §6.7, §9.3 and §10.5.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Instant;

use indexmap::IndexMap;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use slabtastic::{SlabReader, SlabWriter, WriterConfig};

use vectordata::io::{VectorReader, XvecReader};
use veks_core::formats::anode::{self, ANode};
use veks_core::formats::mnode::{MNode, MValue};
use veks_core::formats::passage_table::{parquet_row_groups, read_text_column_row_group};

use crate::pipeline::command::{
    ArtifactManifest, ArtifactState, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole,
    Options, ResourceDesc, Status, StreamContext, render_options_table,
};

use super::compute_topics::TopicModelReport;
use super::source_window::resolve_path;

/// `compute topic-labels`.
pub struct ComputeTopicLabelsOp;

/// Factory used by the pipeline command registry.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeTopicLabelsOp)
}

/// Ranked terms kept per cluster in the `terms` field.
const TERMS_KEPT: usize = 10;

/// Shortest and longest token kept.
const MIN_TOKEN: usize = 3;
const MAX_TOKEN: usize = 30;

/// Function words and academic boilerplate that no label should be
/// made of. Deliberately short: class-based TF-IDF already discounts
/// whatever is common across clusters, and a longer list would start
/// removing terms that are topical in some field.
const STOPWORDS: &[&str] = &[
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "any",
    "can",
    "had",
    "her",
    "was",
    "one",
    "our",
    "out",
    "has",
    "have",
    "been",
    "were",
    "with",
    "this",
    "that",
    "from",
    "they",
    "will",
    "would",
    "there",
    "their",
    "what",
    "about",
    "which",
    "when",
    "make",
    "like",
    "time",
    "just",
    "know",
    "take",
    "into",
    "year",
    "your",
    "some",
    "could",
    "them",
    "than",
    "then",
    "these",
    "those",
    "also",
    "other",
    "such",
    "only",
    "over",
    "more",
    "most",
    "very",
    "each",
    "both",
    "same",
    "where",
    "while",
    "after",
    "before",
    "between",
    "under",
    "through",
    "here",
    "how",
    "its",
    "may",
    "might",
    "should",
    "shall",
    "using",
    "used",
    "use",
    "uses",
    "well",
    "via",
    "per",
    "within",
    "without",
    "upon",
    "among",
    "however",
    "thus",
    "therefore",
    "hence",
    "respectively",
    "furthermore",
    "moreover",
    "although",
    "though",
    "whereas",
    "because",
    "since",
    "whether",
    "either",
    "neither",
    "nor",
    "yet",
    "still",
    "even",
    "much",
    "many",
    "less",
    "few",
    "several",
    "various",
    "given",
    "based",
    "due",
    "according",
    "along",
    "etc",
    "fig",
    "figure",
    "figures",
    "table",
    "tables",
    "section",
    "sections",
    "shown",
    "show",
    "shows",
    "showed",
    "see",
    "seen",
    "found",
    "obtained",
    "presented",
    "present",
    "paper",
    "study",
    "studies",
    "work",
    "works",
    "results",
    "result",
    "two",
    "three",
    "four",
    "five",
    "first",
    "second",
    "third",
    "new",
    "non",
    "does",
    "did",
    "done",
    "being",
    "has",
    "having",
    "against",
    "toward",
    "towards",
    "across",
    "above",
    "below",
    "further",
    "another",
    "own",
    "off",
    "again",
    "once",
    "during",
    "until",
    "whom",
    "whose",
    "who",
    "why",
    "him",
    "his",
    "she",
    "hers",
    "him",
    "himself",
    "herself",
    "itself",
    "themselves",
    "ours",
    "yours",
    "let",
    "get",
    "got",
    "put",
    "set",
    "way",
    "case",
    "cases",
    "number",
    "total",
    "order",
    "type",
    "types",
    "form",
    "forms",
    "part",
    "parts",
    "term",
    "terms",
    "value",
    "values",
    "point",
    "points",
    "level",
    "levels",
    "high",
    "low",
    "large",
    "small",
    "different",
    "similar",
    "important",
    "possible",
    "general",
    "specific",
    "significant",
    "significantly",
    "observed",
    "compared",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "al",
    "et",
];

// ---------------------------------------------------------------------------
// Tokenising
// ---------------------------------------------------------------------------

/// A term interner shared by every level: `id ↔ term`.
#[derive(Default)]
struct Terms {
    ids: HashMap<String, u32>,
    names: Vec<String>,
}

impl Terms {
    fn intern(&mut self, term: &str) -> u32 {
        if let Some(id) = self.ids.get(term) {
            return *id;
        }
        let id = self.names.len() as u32;
        self.names.push(term.to_string());
        self.ids.insert(term.to_string(), id);
        id
    }
}

static STOPWORD_SET: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| STOPWORDS.iter().copied().collect());

fn is_stopword(t: &str) -> bool {
    STOPWORD_SET.contains(t)
}

/// Lower-cased alphabetic tokens of `text`, with stopwords and very
/// short or long tokens removed, plus the bigrams of tokens that were
/// adjacent in the text. Returns `(term, count)` pairs.
fn tokenize(text: &str) -> Vec<(String, u32)> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    let mut prev: Option<String> = None;
    let mut token = String::new();
    let flush =
        |token: &mut String, prev: &mut Option<String>, counts: &mut HashMap<String, u32>| {
            if token.is_empty() {
                return;
            }
            let t = std::mem::take(token);
            let n = t.chars().count();
            if !(MIN_TOKEN..=MAX_TOKEN).contains(&n) || is_stopword(&t) {
                // A dropped token breaks adjacency: no bigram spans it.
                *prev = None;
                return;
            }
            *counts.entry(t.clone()).or_insert(0) += 1;
            if let Some(p) = prev.as_ref() {
                *counts.entry(format!("{} {}", p, t)).or_insert(0) += 1;
            }
            *prev = Some(t);
        };
    for ch in text.chars() {
        if ch.is_alphabetic() {
            for lower in ch.to_lowercase() {
                token.push(lower);
            }
        } else {
            let adjacency_break = !(ch == '-' || ch == '\'');
            flush(&mut token, &mut prev, &mut counts);
            if adjacency_break && !ch.is_whitespace() {
                // Punctuation ends a phrase; whitespace does not.
                prev = None;
            }
        }
    }
    flush(&mut token, &mut prev, &mut counts);
    let mut out: Vec<(String, u32)> = counts.into_iter().collect();
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// Per-level accumulation and scoring
// ---------------------------------------------------------------------------

/// Term counts for every cluster of one level. `samples` counts the
/// passages reserved for a cluster — reservation happens as rows are
/// visited, so the cap moves as it fills — and the tokens follow once
/// the reserved passages have been tokenised.
struct LevelCounts {
    clusters: usize,
    cap: usize,
    counts: Vec<HashMap<u32, u32>>,
    tokens: Vec<u64>,
    samples: Vec<u32>,
}

impl LevelCounts {
    fn new(clusters: usize, cap: usize) -> Self {
        LevelCounts {
            clusters,
            cap,
            counts: (0..clusters).map(|_| HashMap::new()).collect(),
            tokens: vec![0; clusters],
            samples: vec![0; clusters],
        }
    }

    fn accepts(&self, cluster: usize) -> bool {
        cluster < self.clusters && (self.samples[cluster] as usize) < self.cap
    }

    /// Reserve one sample slot; the caller adds its tokens later.
    fn reserve(&mut self, cluster: usize) {
        self.samples[cluster] += 1;
    }

    fn add_tokens(&mut self, cluster: usize, doc: &[(u32, u32)]) {
        let map = &mut self.counts[cluster];
        for (id, n) in doc {
            *map.entry(*id).or_insert(0) += n;
            self.tokens[cluster] += *n as u64;
        }
    }
}

/// One cluster's ranked terms by class-based TF-IDF.
fn rank_terms(
    level: &LevelCounts,
    cluster: usize,
    global: &HashMap<u32, u64>,
    mean_tokens: f64,
) -> Vec<(u32, f64)> {
    let total = level.tokens[cluster];
    if total == 0 {
        return Vec::new();
    }
    let mut scored: Vec<(u32, f64)> = level.counts[cluster]
        .iter()
        .map(|(id, n)| {
            let tf = *n as f64 / total as f64;
            let f = global.get(id).copied().unwrap_or(1) as f64;
            let idf = (1.0 + mean_tokens / f).ln();
            (*id, tf * idf)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(TERMS_KEPT);
    scored
}

/// Characters two words must share from the start to count as the
/// same stem when neither is a prefix of the other: `robotic` and
/// `robots`.
const STEM_PREFIX: usize = 5;

/// Shortest word that can be a stem of a longer one: `cell` / `cells`,
/// `gene` / `genes`.
const MIN_STEM: usize = 4;

/// Whether `w` is a morphological variant of a word already in the
/// slug: the same word, one a prefix of the other (both at least
/// [`MIN_STEM`] long), or the same first [`STEM_PREFIX`] characters.
fn shares_stem(used: &[String], w: &str) -> bool {
    let wl = w.chars().count();
    used.iter().any(|u| {
        if u == w {
            return true;
        }
        let ul = u.chars().count();
        if ul.min(wl) >= MIN_STEM && (u.starts_with(w) || w.starts_with(u.as_str())) {
            return true;
        }
        ul >= STEM_PREFIX
            && wl >= STEM_PREFIX
            && u.chars().take(STEM_PREFIX).eq(w.chars().take(STEM_PREFIX))
    })
}

/// Join ranked terms into a slug, `top` terms deep, without repeating
/// a word — or a stem — already present, so `robot`, `robots` and
/// `robotic` do not make a label between them.
fn slug(terms: &[String], top: usize) -> String {
    let mut words: Vec<String> = Vec::new();
    let mut taken = 0;
    for term in terms {
        if taken >= top {
            break;
        }
        let fresh: Vec<String> = term
            .split(' ')
            .filter(|w| !shares_stem(&words, w))
            .map(str::to_string)
            .collect();
        if fresh.is_empty() {
            continue;
        }
        words.extend(fresh);
        taken += 1;
    }
    words.join("-")
}

fn positional(level: usize, code: usize) -> String {
    format!("l{}-{:05}", level + 1, code)
}

/// A finished label.
#[derive(Debug, Clone, PartialEq)]
struct Label {
    level: usize,
    code: usize,
    label: String,
    terms: Vec<String>,
    sample_size: u32,
    positional: bool,
}

/// Label every cluster of a level: slugs from the ranked terms, unique
/// within the level, positional where the sample is too thin.
fn label_level(
    level_ix: usize,
    level: &LevelCounts,
    terms: &Terms,
    top: usize,
    min_sample: usize,
) -> (Vec<Label>, u32) {
    let mut global: HashMap<u32, u64> = HashMap::new();
    for map in &level.counts {
        for (id, n) in map {
            *global.entry(*id).or_insert(0) += *n as u64;
        }
    }
    let populated = level.tokens.iter().filter(|t| **t > 0).count().max(1);
    let mean_tokens = level.tokens.iter().sum::<u64>() as f64 / populated as f64;

    let ranked: Vec<Vec<String>> = (0..level.clusters)
        .into_par_iter()
        .map(|c| {
            rank_terms(level, c, &global, mean_tokens)
                .into_iter()
                .map(|(id, _)| terms.names[id as usize].clone())
                .collect()
        })
        .collect();

    let mut used: HashSet<String> = HashSet::new();
    let mut labels = Vec::with_capacity(level.clusters);
    let mut collisions = 0u32;
    for (c, ranked_terms) in ranked.iter().enumerate() {
        let thin = (level.samples[c] as usize) < min_sample || ranked_terms.is_empty();
        let (mut label, is_positional) = if thin {
            (positional(level_ix, c), true)
        } else {
            (slug(ranked_terms, top), false)
        };
        if !is_positional && used.contains(&label) {
            // Extend with the next distinguishing terms, then fall
            // back to the code.
            collisions += 1;
            let mut depth = top + 1;
            let mut resolved = false;
            while depth <= ranked_terms.len() {
                let candidate = slug(ranked_terms, depth);
                if !used.contains(&candidate) && candidate != label {
                    label = candidate;
                    resolved = true;
                    break;
                }
                depth += 1;
            }
            if !resolved {
                label = format!("{}-{}", label, c);
            }
        }
        used.insert(label.clone());
        labels.push(Label {
            level: level_ix,
            code: c,
            label,
            terms: ranked_terms.clone(),
            sample_size: level.samples[c],
            positional: is_positional,
        });
    }
    (labels, collisions)
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LevelLabelReport {
    pub clusters: usize,
    pub labelled: usize,
    pub positional: usize,
    pub collisions: u32,
    pub sample_min: u32,
    pub sample_median: u32,
    pub sample_max: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopicLabelsReport {
    pub schema_version: u32,
    pub passages: String,
    pub levels: Vec<usize>,
    pub row_groups_available: usize,
    pub row_groups_read: Vec<usize>,
    pub rows_visited: u64,
    pub docs_tokenized: u64,
    pub distinct_terms: usize,
    pub seconds: f64,
    pub per_level: Vec<LevelLabelReport>,
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

fn error_result(message: String, start: Instant) -> CommandResult {
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
        description: desc.into(),
        extended_description: None,
        role,
    }
}

fn report_path(options: &Options, output: &Path, workspace: &Path) -> PathBuf {
    match options.get("report") {
        Some(s) => resolve_path(s, workspace),
        None => output.with_extension("json"),
    }
}

/// Clusters per level from the model report's branchings.
fn clusters_per_level(levels: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(levels.len());
    let mut n = 1;
    for k in levels {
        n *= k;
        out.push(n);
    }
    out
}

fn read_model(path: &Path) -> Result<TopicModelReport, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read model report {}: {}", path.display(), e))?;
    serde_json::from_str(&text)
        .map_err(|e| format!("model report {} does not parse: {}", path.display(), e))
}

fn label_record(l: &Label) -> Vec<u8> {
    let mut fields = IndexMap::new();
    fields.insert("level".to_string(), MValue::Int(l.level as i64 + 1));
    fields.insert("code".to_string(), MValue::Int(l.code as i64));
    fields.insert("label".to_string(), MValue::Text(l.label.clone()));
    fields.insert("terms".to_string(), MValue::Text(l.terms.join(", ")));
    fields.insert("sample_size".to_string(), MValue::Int(l.sample_size as i64));
    fields.insert("positional".to_string(), MValue::Bool(l.positional));
    anode::encode(&ANode::MNode(MNode { fields }))
}

/// Read every label record back: `(level, code, label)` in slab order.
pub fn read_labels(path: &Path) -> Result<Vec<(usize, usize, String)>, String> {
    let reader =
        SlabReader::open(path).map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let mut out = Vec::with_capacity(reader.total_records() as usize);
    for entry in reader.page_entries() {
        let page = reader
            .read_data_page(&entry)
            .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
        for i in 0..page.record_count() {
            let Some(bytes) = page.get_record(i) else {
                continue;
            };
            let Ok(ANode::MNode(m)) = anode::decode(bytes) else {
                return Err(format!(
                    "{} holds a record that is not an MNode",
                    path.display()
                ));
            };
            let level = match m.fields.get("level") {
                Some(MValue::Int(v)) => *v as usize,
                _ => return Err(format!("{} record {} lacks `level`", path.display(), i)),
            };
            let code = match m.fields.get("code") {
                Some(MValue::Int(v)) => *v as usize,
                _ => return Err(format!("{} record {} lacks `code`", path.display(), i)),
            };
            let label = match m.fields.get("label") {
                Some(MValue::Text(t)) => t.clone(),
                _ => return Err(format!("{} record {} lacks `label`", path.display(), i)),
            };
            out.push((level, code, label));
        }
    }
    Ok(out)
}

impl CommandOp for ComputeTopicLabelsOp {
    fn command_path(&self) -> &str {
        "compute topic-labels"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_COMPUTE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Name every topic cluster from its members' most distinctive terms".into(),
            body: format!(
                r#"# compute topic-labels

Reads a seeded subset of the passage table's row groups, visits each
group's rows in a seeded order, and accepts passages for their
cluster at every level until each cluster holds `sample-per-cluster`
of them. Term frequencies (unigrams and bigrams) are aggregated per
cluster and weighted by class-based TF-IDF against the level, and the
top terms are joined into a hyphenated slug. Labels are unique within
a level; a cluster met by fewer than `min-sample` passages gets a
positional label such as `l3-04187`.

Output is a slab of one MNode per cluster, in level then code order:
`level`, `code`, `label`, `terms` (ranked), `sample_size`,
`positional`. It is the code → label table enrichment applies.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt(
                "passages",
                "Path",
                true,
                None,
                "Passage table (parquet), row-aligned with the assignments",
                OptionRole::Input,
            ),
            opt(
                "assignments",
                "Path",
                true,
                None,
                "Topic assignments from `compute topics` (u16vecs, one code per level)",
                OptionRole::Input,
            ),
            opt(
                "model",
                "Path",
                true,
                None,
                "Model report from `compute topics` (JSON), for the branching per level",
                OptionRole::Input,
            ),
            opt(
                "text-column",
                "string",
                false,
                Some("text"),
                "Column of the passage table holding the text",
                OptionRole::Config,
            ),
            opt(
                "row-groups",
                "int",
                false,
                Some("64"),
                "Row groups read, chosen by seed",
                OptionRole::Config,
            ),
            opt(
                "sample-per-cluster",
                "int",
                false,
                Some("2000"),
                "Cap on passages sampled per cluster at each level",
                OptionRole::Config,
            ),
            opt(
                "min-sample",
                "int",
                false,
                Some("20"),
                "Below this many passages a cluster gets a positional label",
                OptionRole::Config,
            ),
            opt(
                "top-terms",
                "int",
                false,
                Some("3"),
                "Terms joined into the slug",
                OptionRole::Config,
            ),
            opt(
                "seed",
                "int",
                false,
                Some("42"),
                "Row-group and row-order selection",
                OptionRole::Config,
            ),
            opt(
                "output",
                "Path",
                true,
                None,
                "Label slab, one MNode per cluster in level then code order",
                OptionRole::Output,
            ),
            opt(
                "report",
                "Path",
                false,
                None,
                "Labelling report JSON (default: beside `output` with a .json extension)",
                OptionRole::Output,
            ),
        ]
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc {
                name: "mem".into(),
                description: "One row group's text plus per-cluster term tables".into(),
                adjustable: false,
            },
            ResourceDesc {
                name: "threads".into(),
                description: "Parallel tokenising and scoring".into(),
                adjustable: true,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();
        let passages_str = match options.require("passages") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };
        let assignments_str = match options.require("assignments") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };
        let model_str = match options.require("model") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s.to_string(),
            Err(e) => return error_result(e, start),
        };
        let text_column = options.get("text-column").unwrap_or("text").to_string();
        let row_groups_wanted = match options.parse_or::<usize>("row-groups", 64) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let cap = match options.parse_or::<usize>("sample-per-cluster", 2000) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let min_sample = match options.parse_or::<usize>("min-sample", 20) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let top = match options.parse_or::<usize>("top-terms", 3) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        let seed = match options.parse_or::<u64>("seed", 42) {
            Ok(v) => v,
            Err(e) => return error_result(e, start),
        };
        if row_groups_wanted == 0 || cap == 0 || top == 0 {
            return error_result(
                "row-groups, sample-per-cluster and top-terms must be positive".into(),
                start,
            );
        }

        let passages_path = resolve_path(&passages_str, &ctx.workspace);
        let assignments_path = resolve_path(&assignments_str, &ctx.workspace);
        let model_path = resolve_path(&model_str, &ctx.workspace);
        let output_path = resolve_path(&output_str, &ctx.workspace);
        let report_path = report_path(options, &output_path, &ctx.workspace);

        let model = match read_model(&model_path) {
            Ok(m) => m,
            Err(e) => return error_result(e, start),
        };
        let clusters = clusters_per_level(&model.levels);
        let depth = model.levels.len();
        let assignments = match XvecReader::<u16>::open_path(&assignments_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!(
                        "failed to open assignments {}: {}",
                        assignments_path.display(),
                        e
                    ),
                    start,
                );
            }
        };
        if assignments.dim() != depth {
            return error_result(
                format!(
                    "assignments carry {} codes per record but the model has {} levels",
                    assignments.dim(),
                    depth
                ),
                start,
            );
        }
        let groups = match parquet_row_groups(&passages_path) {
            Ok(g) => g,
            Err(e) => return error_result(e, start),
        };
        let total_rows: u64 = groups.iter().map(|(_, n)| *n).sum();
        if total_rows as usize != assignments.count() {
            return error_result(
                format!(
                    "passages hold {} rows but the assignments hold {} records; they must be row-aligned",
                    total_rows,
                    assignments.count()
                ),
                start,
            );
        }

        let threads = {
            let t = ctx.governor.current_or("threads", ctx.threads as u64) as usize;
            if t == 0 {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            } else {
                t
            }
        };
        let pool = match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
            Ok(p) => p,
            Err(e) => return error_result(format!("failed to build thread pool: {}", e), start),
        };

        // Seeded choice of row groups, visited in the chosen order so
        // the early caps fill from scattered parts of the corpus.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut chosen: Vec<usize> = (0..groups.len()).collect();
        chosen.shuffle(&mut rng);
        chosen.truncate(row_groups_wanted.min(groups.len()));
        ctx.ui.log(&format!(
            "topic-labels: {} levels {:?}; reading {} of {} row groups of {} ({} rows), cap {} per cluster, {} threads",
            depth, model.levels, chosen.len(), groups.len(),
            passages_path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
            total_rows, cap, threads,
        ));

        let mut terms = Terms::default();
        let mut levels: Vec<LevelCounts> =
            clusters.iter().map(|c| LevelCounts::new(*c, cap)).collect();
        let mut rows_visited = 0u64;
        let mut docs_tokenized = 0u64;
        let pb = ctx
            .ui
            .bar_with_unit(chosen.len() as u64, "labelling topics", "groups");
        for (gi, &g) in chosen.iter().enumerate() {
            let (first_row, texts) =
                match read_text_column_row_group(&passages_path, &text_column, g) {
                    Ok(v) => v,
                    Err(e) => return error_result(e, start),
                };
            // Seeded visiting order within the group.
            let mut order: Vec<usize> = (0..texts.len()).collect();
            order.shuffle(&mut rng);
            // Decide acceptance sequentially — it depends on the caps —
            // then tokenise the accepted texts in parallel.
            // Which passages to tokenise, and for which levels: decided
            // sequentially because it depends on the caps filling.
            let mut accepted: Vec<(usize, Vec<u16>, Vec<bool>)> = Vec::new();
            for &i in &order {
                rows_visited += 1;
                let codes = match assignments.get(first_row as usize + i) {
                    Ok(c) => c,
                    Err(e) => {
                        return error_result(
                            format!(
                                "failed to read assignment {}: {}",
                                first_row as usize + i,
                                e
                            ),
                            start,
                        );
                    }
                };
                let mask: Vec<bool> = codes
                    .iter()
                    .enumerate()
                    .map(|(l, &c)| levels[l].accepts(c as usize))
                    .collect();
                if mask.iter().any(|m| *m) {
                    for (l, (&c, m)) in codes.iter().zip(&mask).enumerate() {
                        if *m {
                            levels[l].reserve(c as usize);
                        }
                    }
                    accepted.push((i, codes, mask));
                }
            }
            let tokenized: Vec<Vec<(String, u32)>> = pool.install(|| {
                accepted
                    .par_iter()
                    .map(|(i, _, _)| tokenize(&texts[*i]))
                    .collect()
            });
            docs_tokenized += tokenized.len() as u64;
            for ((_, codes, mask), doc) in accepted.iter().zip(&tokenized) {
                let interned: Vec<(u32, u32)> =
                    doc.iter().map(|(t, n)| (terms.intern(t), *n)).collect();
                for (l, (&c, m)) in codes.iter().zip(mask).enumerate() {
                    if *m {
                        levels[l].add_tokens(c as usize, &interned);
                    }
                }
            }
            pb.set_position(gi as u64 + 1);
        }
        pb.finish();

        // Score and label each level.
        let mut all_labels: Vec<Label> = Vec::new();
        let mut per_level = Vec::with_capacity(depth);
        for (l, level) in levels.iter().enumerate() {
            let (labels, collisions) =
                pool.install(|| label_level(l, level, &terms, top, min_sample));
            let mut samples: Vec<u32> = level.samples.clone();
            samples.sort_unstable();
            let positional = labels.iter().filter(|x| x.positional).count();
            per_level.push(LevelLabelReport {
                clusters: level.clusters,
                labelled: level.clusters - positional,
                positional,
                collisions,
                sample_min: samples.first().copied().unwrap_or(0),
                sample_median: samples.get(samples.len() / 2).copied().unwrap_or(0),
                sample_max: samples.last().copied().unwrap_or(0),
            });
            ctx.ui.log(&format!(
                "topic-labels: level {} — {} clusters, {} positional, {} collisions resolved, samples {}..{} (median {})",
                l + 1, level.clusters, positional, collisions,
                per_level[l].sample_min, per_level[l].sample_max, per_level[l].sample_median,
            ));
            all_labels.extend(labels);
        }

        // Write the slab.
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
            Err(e) => return error_result(format!("slab config: {}", e), start),
        };
        let mut writer = match SlabWriter::new(&output_path, config) {
            Ok(w) => w,
            Err(e) => {
                return error_result(
                    format!("failed to create {}: {}", output_path.display(), e),
                    start,
                );
            }
        };
        for l in &all_labels {
            if let Err(e) = writer.add_record(&label_record(l)) {
                return error_result(
                    format!("failed to write {}: {}", output_path.display(), e),
                    start,
                );
            }
        }
        if let Err(e) = writer.finish() {
            return error_result(
                format!("failed to finish {}: {}", output_path.display(), e),
                start,
            );
        }
        let mut produced = vec![output_path.clone()];

        let report = TopicLabelsReport {
            schema_version: 1,
            passages: passages_str,
            levels: model.levels.clone(),
            row_groups_available: groups.len(),
            row_groups_read: chosen.clone(),
            rows_visited,
            docs_tokenized,
            distinct_terms: terms.names.len(),
            seconds: start.elapsed().as_secs_f64(),
            per_level,
        };
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

        let positional_total: usize = report.per_level.iter().map(|l| l.positional).sum();
        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} clusters labelled over {} levels from {} passages in {} row groups ({} positional)",
                all_labels.len(),
                depth,
                docs_tokenized,
                chosen.len(),
                positional_total,
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    /// Complete when the slab holds exactly Σ levels records — one per
    /// cluster, in level then code order, matching the model — and
    /// every label is unique within its level.
    fn check_artifact(&self, output: &Path, options: &Options) -> ArtifactState {
        if !output.exists() {
            return ArtifactState::Absent;
        }
        let workspace = super::compute_topics::workspace_of(output, options.get("output"));
        let Some(model_str) = options.get("model") else {
            return ArtifactState::Partial;
        };
        let Ok(model) = read_model(&resolve_path(model_str, &workspace)) else {
            return ArtifactState::Partial;
        };
        let clusters = clusters_per_level(&model.levels);
        let Ok(labels) = read_labels(output) else {
            return ArtifactState::Partial;
        };
        if labels.len() != clusters.iter().sum::<usize>() {
            return ArtifactState::Partial;
        }
        let mut expected = clusters
            .iter()
            .enumerate()
            .flat_map(|(l, n)| (0..*n).map(move |c| (l + 1, c)));
        let mut seen: HashSet<(usize, String)> = HashSet::new();
        for (level, code, label) in &labels {
            if expected.next() != Some((*level, *code)) {
                return ArtifactState::Partial;
            }
            if !seen.insert((*level, label.clone())) {
                return ArtifactState::Partial;
            }
        }
        ArtifactState::Complete
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        let mut manifest = crate::pipeline::command::manifest_from_keys(
            step_id,
            self.command_path(),
            options,
            &["passages", "assignments", "model"],
            &["output", "report"],
        );
        if options.get("report").is_none()
            && let Some(o) = options.get("output")
        {
            manifest.outputs.push(
                PathBuf::from(o)
                    .with_extension("json")
                    .to_string_lossy()
                    .to_string(),
            );
        }
        manifest
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_drops_stopwords_and_forms_adjacent_bigrams() {
        let t = tokenize("Grid integration of the renewables; grid-integration! 42 ok");
        let map: HashMap<String, u32> = t.into_iter().collect();
        assert_eq!(map.get("grid"), Some(&2));
        assert_eq!(map.get("integration"), Some(&2));
        assert_eq!(map.get("renewables"), Some(&1));
        assert_eq!(
            map.get("grid integration"),
            Some(&2),
            "hyphen keeps adjacency"
        );
        assert!(
            !map.contains_key("integration renewables"),
            "a dropped stopword breaks adjacency"
        );
        assert!(!map.contains_key("the"));
        assert!(!map.contains_key("of"));
        assert!(!map.contains_key("ok"), "two letters is too short");
        assert!(
            !map.contains_key("renewables grid"),
            "punctuation ends a phrase"
        );
    }

    #[test]
    fn class_tfidf_prefers_the_distinctive_term() {
        let mut terms = Terms::default();
        let mut level = LevelCounts::new(2, 100);
        let shared = terms.intern("model");
        let a = terms.intern("photovoltaic");
        let b = terms.intern("neural");
        for _ in 0..10 {
            level.samples[0] += 1;
            level.add_tokens(0, &[(shared, 5), (a, 3)]);
            level.samples[1] += 1;
            level.add_tokens(1, &[(shared, 5), (b, 3)]);
        }
        let (labels, collisions) = label_level(1, &level, &terms, 1, 1);
        assert_eq!(collisions, 0);
        assert_eq!(labels[0].label, "photovoltaic");
        assert_eq!(labels[1].label, "neural");
        assert_eq!(labels[0].terms[0], "photovoltaic");
        assert!(labels[0].terms.contains(&"model".to_string()));
    }

    #[test]
    fn collisions_extend_then_suffix_and_thin_clusters_are_positional() {
        let mut terms = Terms::default();
        let mut level = LevelCounts::new(4, 100);
        let x = terms.intern("optics");
        let y = terms.intern("lasers");
        let z = terms.intern("fibres");
        // Clusters 0 and 1 share their best term; 1 has a second term
        // to extend with, 2 has exactly the same terms as 0, 3 is thin.
        level.samples[0] += 50;
        level.add_tokens(0, &[(x, 10)]);
        // Enough `optics` that it still outranks the rarer `lasers`
        // under class TF-IDF, so the label collides and must extend.
        level.samples[1] += 50;
        level.add_tokens(1, &[(x, 100), (y, 2)]);
        level.samples[2] += 50;
        level.add_tokens(2, &[(x, 10)]);
        level.samples[3] += 3;
        level.add_tokens(3, &[(z, 1)]);
        let (labels, collisions) = label_level(2, &level, &terms, 1, 20);
        assert_eq!(labels[0].label, "optics");
        assert_eq!(
            labels[1].label, "optics-lasers",
            "extended with its next term"
        );
        assert_eq!(labels[2].label, "optics-2", "no further term: code suffix");
        assert_eq!(labels[3].label, "l3-00003");
        assert!(labels[3].positional);
        assert_eq!(collisions, 2);
        let set: HashSet<&String> = labels.iter().map(|l| &l.label).collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn slug_dedups_words_and_stems_across_terms() {
        let terms = vec![
            "grid integration".to_string(),
            "grid".to_string(),
            "renewables".to_string(),
        ];
        assert_eq!(slug(&terms, 3), "grid-integration-renewables");
        assert_eq!(slug(&terms, 1), "grid-integration");
        let variants = vec![
            "robot".to_string(),
            "robots".to_string(),
            "robotic".to_string(),
            "teleoperation".to_string(),
            "manipulator arm".to_string(),
        ];
        assert_eq!(slug(&variants, 3), "robot-teleoperation-manipulator-arm");
        let short = vec!["cell".to_string(), "cells".to_string(), "gene".to_string()];
        assert_eq!(
            slug(&short, 3),
            "cell-gene",
            "same word, and a four-letter stem is compared whole"
        );
    }

    #[test]
    fn clusters_per_level_is_the_running_product() {
        assert_eq!(clusters_per_level(&[10, 30, 33]), vec![10, 300, 9_900]);
    }
}
