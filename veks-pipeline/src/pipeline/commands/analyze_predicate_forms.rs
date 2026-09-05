// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `analyze predicate-forms` — enumerate the shapes a predicate facet holds.
//!
//! A system under test declares indexes against predicate *forms*, not
//! predicates: `topic_l3 = ? AND citation_percentile >= ?` needs the same
//! indexes whatever the literals are. This command decodes every
//! predicate, abstracts its literals, canonicalises conjunct order, and
//! reports the distinct forms with their counts, the fields each form
//! touches and how (equality, range, set, pattern), and the grammar
//! space actually used: depth, arity, conjunction and operator kinds.
//! With the `families` namespace present, forms are attributed to the
//! families that produced them; with evaluation indices, each form
//! carries its realised selectivity range.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use slabtastic::SlabReader;
use veks_core::formats::pnode::{ConjugateType, FieldRef, OpType, PNode};

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    Status, StreamContext, render_options_table,
};

use super::compute_prefiltered_knn::PredicateIndices;
use super::verify_predicate_strata::{read_mnodes, text};

pub struct AnalyzePredicateFormsOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(AnalyzePredicateFormsOp)
}

impl CommandOp for AnalyzePredicateFormsOp {
    fn command_path(&self) -> &str {
        "analyze predicate-forms"
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
            summary: "Enumerate the predicate forms a facet holds, with the indexes each needs".into(),
            body: format!(
                r#"# analyze predicate-forms

Decode every predicate in a facet, abstract its literals, and report
the distinct forms present: `topic_l3 = ? AND citation_percentile >= ?`
is one form however many predicates instantiate it. Conjunct order is
canonicalised, so `A AND B` and `B AND A` are the same form.

For each form: how many predicates take it, which fields it touches and
by what access — equality, inequality, range, set membership, pattern —
and, when `--metadata-indices` is given, the realised selectivity range
of its predicates. The field × access table that follows is the index
declaration a system under test needs to serve the whole facet.

When the facet carries a `families` namespace (a stratified facet), each
form is attributed to the families that produced it.

## Options

{}
"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let predicates_path = match options.require("predicates") {
            Ok(s) => resolve(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let indices_path = options.get("metadata-indices").map(|s| resolve(s, &ctx.workspace));
        let metadata_path = options.get("metadata").map(|s| resolve(s, &ctx.workspace));

        let pred_reader = match SlabReader::open(&predicates_path) {
            Ok(r) => r,
            Err(e) => {
                return error_result(
                    format!("open predicates {}: {}", predicates_path.display(), e),
                    start,
                )
            }
        };
        let n_preds = pred_reader.total_records() as usize;

        let indices = match &indices_path {
            Some(p) => match PredicateIndices::open(p) {
                Ok(r) => Some(r),
                Err(e) => {
                    return error_result(
                        format!("open metadata-indices {}: {}", p.display(), e),
                        start,
                    )
                }
            },
            None => None,
        };
        let metadata_total: u64 = metadata_path
            .as_ref()
            .and_then(|p| SlabReader::open(p).ok().map(|r| r.total_records()))
            .unwrap_or(0);

        // Families are optional: a plain predicate facet has none, a
        // stratified one has one record per predicate.
        let families: Option<Vec<String>> = read_mnodes(
            &predicates_path,
            Some(vectordata::metadata_schema::FAMILIES_NAMESPACE),
        )
        .ok()
        .filter(|f| f.len() == n_preds)
        .map(|f| f.iter().map(|m| text(m, "family").unwrap_or_default()).collect());

        let mut census = FormCensus::default();
        for i in 0..n_preds {
            let pnode = match pred_reader.get(i as i64) {
                Ok(bytes) => match PNode::from_bytes_named(&bytes) {
                    Ok(p) => p,
                    Err(e) => {
                        return error_result(format!("decode predicate at ordinal {i}: {e}"), start)
                    }
                },
                Err(e) => return error_result(format!("read predicate at ordinal {i}: {e}"), start),
            };
            let selectivity = match &indices {
                Some(idx) => match idx.get_ordinals(i) {
                    Ok(ords) if metadata_total > 0 => Some(ords.len() as f64 / metadata_total as f64),
                    Ok(_) => None,
                    Err(e) => {
                        return error_result(format!("read indices for predicate {i}: {e}"), start)
                    }
                },
                None => None,
            };
            census.record(&pnode, families.as_ref().map(|f| f[i].as_str()), selectivity);
        }

        render(ctx, &census, n_preds, families.is_some(), metadata_total > 0);

        CommandResult {
            status: Status::Ok,
            message: format!(
                "{} predicates take {} distinct forms over {} fields",
                n_preds,
                census.forms.len(),
                census.fields().len()
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            opt(
                "predicates",
                "Path",
                true,
                None,
                "Predicate slab (PNode records; a `families` namespace, if present, attributes forms)",
                OptionRole::Input,
            ),
            opt(
                "metadata-indices",
                "Path",
                false,
                None,
                "Per-predicate matching ordinals, as produced by `compute evaluate-predicates`; adds each form's selectivity range",
                OptionRole::Input,
            ),
            opt(
                "metadata",
                "Path",
                false,
                None,
                "Metadata slab — the record count that turns match counts into selectivities",
                OptionRole::Input,
            ),
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id,
            self.command_path(),
            options,
            &["predicates", "metadata-indices", "metadata"],
            &[],
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Forms
// ─────────────────────────────────────────────────────────────────────────────

/// How a predicate leaf reaches into a field, which is what an index
/// has to support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Access {
    Equality,
    Inequality,
    Range,
    Set,
    Pattern,
}

impl Access {
    pub(crate) fn of(op: OpType) -> Self {
        match op {
            OpType::Eq => Access::Equality,
            OpType::Ne => Access::Inequality,
            OpType::Gt | OpType::Lt | OpType::Ge | OpType::Le => Access::Range,
            OpType::In => Access::Set,
            OpType::Matches => Access::Pattern,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Access::Equality => "equality",
            Access::Inequality => "inequality",
            Access::Range => "range",
            Access::Set => "set",
            Access::Pattern => "pattern",
        }
    }
}

fn field_name(f: &FieldRef) -> String {
    match f {
        FieldRef::Named(n) => n.clone(),
        FieldRef::Index(i) => format!("field[{i}]"),
    }
}

fn op_symbol(op: OpType) -> &'static str {
    match op {
        OpType::Eq => "=",
        OpType::Ne => "!=",
        OpType::Lt => "<",
        OpType::Le => "<=",
        OpType::Gt => ">",
        OpType::Ge => ">=",
        OpType::In => "in",
        OpType::Matches => "matches",
    }
}

/// The form of a predicate: its tree with every literal replaced by `?`
/// and the conjuncts of each junction sorted, so two predicates that
/// differ only in literals or in conjunct order render identically.
pub(crate) fn form_of(p: &PNode) -> String {
    match p {
        PNode::Predicate(leaf) => {
            let arity = leaf.comparands.len();
            let slot = match (leaf.op, arity) {
                (OpType::In, n) if n != 1 => format!("(?×{n})"),
                _ => "?".to_string(),
            };
            format!("{} {} {}", field_name(&leaf.field), op_symbol(leaf.op), slot)
        }
        PNode::Conjugate(c) => {
            let mut parts: Vec<String> = c.children.iter().map(form_of).collect();
            parts.sort();
            match (c.conjugate_type, parts.len()) {
                (_, 1) => parts.remove(0),
                (ConjugateType::Or, _) => format!("({})", parts.join(" OR ")),
                (ConjugateType::And | ConjugateType::Pred, _) => {
                    format!("({})", parts.join(" AND "))
                }
            }
        }
    }
}

/// Every (field, access) pair a predicate needs served.
pub(crate) fn accesses_of(p: &PNode, out: &mut BTreeSet<(String, Access)>) {
    match p {
        PNode::Predicate(leaf) => {
            out.insert((field_name(&leaf.field), Access::of(leaf.op)));
        }
        PNode::Conjugate(c) => {
            for child in &c.children {
                accesses_of(child, out);
            }
        }
    }
}

fn depth_of(p: &PNode) -> usize {
    match p {
        PNode::Predicate(_) => 1,
        PNode::Conjugate(c) => 1 + c.children.iter().map(depth_of).max().unwrap_or(0),
    }
}

fn leaves_of(p: &PNode) -> usize {
    match p {
        PNode::Predicate(_) => 1,
        PNode::Conjugate(c) => c.children.iter().map(leaves_of).sum(),
    }
}

fn junctions_of(p: &PNode, out: &mut BTreeMap<&'static str, usize>) {
    if let PNode::Conjugate(c) = p {
        if c.children.len() > 1 {
            let kind = match c.conjugate_type {
                ConjugateType::And | ConjugateType::Pred => "AND",
                ConjugateType::Or => "OR",
            };
            *out.entry(kind).or_default() += 1;
        }
        for child in &c.children {
            junctions_of(child, out);
        }
    }
}

#[derive(Default)]
struct FormStats {
    count: usize,
    accesses: BTreeSet<(String, Access)>,
    families: BTreeMap<String, usize>,
    selectivities: Vec<f64>,
}

#[derive(Default)]
pub(crate) struct FormCensus {
    forms: BTreeMap<String, FormStats>,
    max_depth: usize,
    max_leaves: usize,
    junctions: BTreeMap<&'static str, usize>,
    ops: BTreeMap<&'static str, usize>,
}

impl FormCensus {
    fn record(&mut self, p: &PNode, family: Option<&str>, selectivity: Option<f64>) {
        let form = form_of(p);
        let entry = self.forms.entry(form).or_default();
        entry.count += 1;
        if entry.accesses.is_empty() {
            accesses_of(p, &mut entry.accesses);
        }
        if let Some(f) = family {
            *entry.families.entry(f.to_string()).or_default() += 1;
        }
        if let Some(s) = selectivity {
            entry.selectivities.push(s);
        }
        self.max_depth = self.max_depth.max(depth_of(p));
        self.max_leaves = self.max_leaves.max(leaves_of(p));
        junctions_of(p, &mut self.junctions);
        count_ops(p, &mut self.ops);
    }

    /// Fields touched by any form.
    fn fields(&self) -> BTreeSet<String> {
        self.forms
            .values()
            .flat_map(|s| s.accesses.iter().map(|(f, _)| f.clone()))
            .collect()
    }

    /// Predicates per (field, access): the index declaration.
    fn index_needs(&self) -> BTreeMap<(String, Access), usize> {
        let mut out: BTreeMap<(String, Access), usize> = BTreeMap::new();
        for stats in self.forms.values() {
            for key in &stats.accesses {
                *out.entry(key.clone()).or_default() += stats.count;
            }
        }
        out
    }
}

fn count_ops(p: &PNode, out: &mut BTreeMap<&'static str, usize>) {
    match p {
        PNode::Predicate(leaf) => *out.entry(op_symbol(leaf.op)).or_default() += 1,
        PNode::Conjugate(c) => {
            for child in &c.children {
                count_ops(child, out);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────────────────────

fn render(ctx: &mut StreamContext, census: &FormCensus, n_preds: usize, with_families: bool, with_sel: bool) {
    let say = |ctx: &mut StreamContext, line: String| ctx.ui.log(&line);

    say(ctx, format!("predicate forms — {n_preds} predicates, {} distinct forms", census.forms.len()));
    say(ctx, String::new());
    say(ctx, "grammar space used".into());
    say(ctx, format!("  fields:      {}", census.fields().len()));
    say(ctx, format!("  max depth:   {}   max leaves per predicate: {}", census.max_depth, census.max_leaves));
    let junctions: Vec<String> = census.junctions.iter().map(|(k, n)| format!("{k}×{n}")).collect();
    say(ctx, format!("  junctions:   {}", if junctions.is_empty() { "none (every predicate is a single leaf)".to_string() } else { junctions.join("  ") }));
    let ops: Vec<String> = census.ops.iter().map(|(k, n)| format!("{k}×{n}")).collect();
    say(ctx, format!("  operators:   {}", ops.join("  ")));
    say(ctx, String::new());

    let mut forms: Vec<(&String, &FormStats)> = census.forms.iter().collect();
    forms.sort_by(|a, b| b.1.count.cmp(&a.1.count).then(a.0.cmp(b.0)));
    let width = forms.iter().map(|(f, _)| f.len()).max().unwrap_or(4).max(4);
    let sel_head = if with_sel { "   selectivity min / median / max" } else { "" };
    say(ctx, format!("{:<width$}   {:>7}  {:>6}{sel_head}", "form", "count", "share", width = width));
    for (form, stats) in &forms {
        let share = 100.0 * stats.count as f64 / n_preds.max(1) as f64;
        let mut line = format!("{:<width$}   {:>7}  {:>5.1}%", form, stats.count, share, width = width);
        if with_sel && !stats.selectivities.is_empty() {
            let mut s = stats.selectivities.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            line.push_str(&format!(
                "   {:.2e} / {:.2e} / {:.2e}",
                s[0],
                s[s.len() / 2],
                s[s.len() - 1]
            ));
        }
        say(ctx, line);
        if with_families && !stats.families.is_empty() {
            let fams: Vec<String> = stats.families.iter().map(|(f, n)| format!("{f}×{n}")).collect();
            say(ctx, format!("{:<width$}   families: {}", "", fams.join(", "), width = width));
        }
    }
    say(ctx, String::new());

    say(ctx, "index needs — predicates touching each field by access".into());
    let needs = census.index_needs();
    let fw = needs.keys().map(|(f, _)| f.len()).max().unwrap_or(5).max(5);
    say(ctx, format!("  {:<fw$}  {:<10}  {:>7}", "field", "access", "preds", fw = fw));
    for ((field, access), n) in &needs {
        say(ctx, format!("  {:<fw$}  {:<10}  {:>7}", field, access.label(), n, fw = fw));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn resolve(s: &str, workspace: &std::path::Path) -> PathBuf {
    let p = PathBuf::from(s);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn opt(name: &str, type_name: &str, required: bool, default: Option<&str>, description: &str, role: OptionRole) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: description.to_string(),
        extended_description: None,
        role,
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veks_core::formats::pnode::{Comparand, ConjugateNode, PredicateNode};

    fn leaf(field: &str, op: OpType, c: Comparand) -> PNode {
        PNode::Predicate(PredicateNode {
            field: FieldRef::Named(field.into()),
            op,
            comparands: vec![c],
        })
    }

    fn and(children: Vec<PNode>) -> PNode {
        PNode::Conjugate(ConjugateNode { conjugate_type: ConjugateType::And, children })
    }

    /// Literals do not distinguish forms, and neither does conjunct order.
    #[test]
    fn a_form_abstracts_literals_and_conjunct_order() {
        let a = and(vec![
            leaf("topic_l3", OpType::Eq, Comparand::Text("tax".into())),
            leaf("citation_percentile", OpType::Ge, Comparand::Int(99)),
        ]);
        let b = and(vec![
            leaf("citation_percentile", OpType::Ge, Comparand::Int(50)),
            leaf("topic_l3", OpType::Eq, Comparand::Text("ml".into())),
        ]);
        assert_eq!(form_of(&a), form_of(&b));
        assert_eq!(form_of(&a), "(citation_percentile >= ? AND topic_l3 = ?)");
        assert_eq!(form_of(&leaf("year", OpType::Le, Comparand::Int(2000))), "year <= ?");
        // A one-child junction is its child.
        assert_eq!(form_of(&and(vec![leaf("year", OpType::Eq, Comparand::Int(1))])), "year = ?");
    }

    /// The access a leaf needs follows its operator, and a range on one
    /// field spelled as two leaves is one (field, range) need.
    #[test]
    fn index_needs_follow_operators_and_fold_a_range_pair() {
        let range = and(vec![
            leaf("year", OpType::Ge, Comparand::Int(1990)),
            leaf("year", OpType::Le, Comparand::Int(2000)),
        ]);
        let mut acc = BTreeSet::new();
        accesses_of(&range, &mut acc);
        assert_eq!(acc.len(), 1);
        assert!(acc.contains(&("year".to_string(), Access::Range)));
        assert_eq!(Access::of(OpType::Eq), Access::Equality);
        assert_eq!(Access::of(OpType::In), Access::Set);
        assert_eq!(Access::of(OpType::Matches), Access::Pattern);
        assert_eq!(Access::of(OpType::Ne), Access::Inequality);
    }

    /// The census counts predicates per form, attributes families, keeps
    /// selectivities, and sums index needs across forms.
    #[test]
    fn the_census_counts_forms_families_and_needs() {
        let mut census = FormCensus::default();
        let topical = |t: &str| leaf("topic_l3", OpType::Eq, Comparand::Text(t.into()));
        census.record(&topical("a"), Some("topical"), Some(0.01));
        census.record(&topical("b"), Some("topical"), Some(0.001));
        census.record(
            &and(vec![topical("c"), leaf("citation_percentile", OpType::Ge, Comparand::Int(90))]),
            Some("conjunction"),
            Some(0.0001),
        );
        assert_eq!(census.forms.len(), 2);
        let single = &census.forms["topic_l3 = ?"];
        assert_eq!(single.count, 2);
        assert_eq!(single.families["topical"], 2);
        assert_eq!(single.selectivities.len(), 2);
        assert_eq!(census.max_depth, 2);
        assert_eq!(census.max_leaves, 2);
        assert_eq!(census.junctions["AND"], 1);
        let needs = census.index_needs();
        assert_eq!(needs[&("topic_l3".to_string(), Access::Equality)], 3);
        assert_eq!(needs[&("citation_percentile".to_string(), Access::Range)], 1);
    }
}
