// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Predicate prototype — a serialisable template for predicate
//! generation that uses the PNode `Display` vernacular as its
//! shape language.
//!
//! A prototype pins everything *except* the comparand values:
//! field names, operators, conjugate form, and selectivity
//! target. The comparand values are filled in at generation
//! time by drawing from the survey to land on the requested
//! selectivity. Wherever the user writes `?` in the template,
//! the generator substitutes a survey-derived value.
//!
//! ```yaml
//! # example.proto.yaml
//! template: "(age >= ? AND name MATCHES ?)"
//! selectivity: 0.05..0.20
//! count: 200
//! seed: 42
//! ```
//!
//! The template grammar is the PNode Display form (parsed by
//! [`veks_core::formats::pnode::from_display`]) with one
//! extension: `?` is accepted in the comparand position to
//! denote "fill at generation time".
//!
//! Codec stack reused from veks-anode:
//! ```text
//! template.text  <->  [parse_template]  <->  PredicateTemplate
//!                                                  ^
//!                                                  | materialize_predicate(survey, target_sel)
//!                                                  v
//!                                                PNode (concrete)  <->  PNode::Display
//! ```

use std::collections::HashMap;
use std::path::Path;

use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use veks_core::formats::pnode::{
    Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode,
};

use crate::pipeline::commands::survey::FieldProfile;

// ─────────────────────────────────────────────────────────────────────────────
// YAML schema
// ─────────────────────────────────────────────────────────────────────────────

/// Serialisable predicate prototype. The `template` field carries
/// a PNode Display string with `?` placeholders for comparands;
/// every other knob is pinned so the same `.proto.yaml` produces
/// the same predicates on re-run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateProto {
    /// PNode-Display predicate template. See module docs for the
    /// grammar. Required.
    pub template: String,
    /// Number of predicates to emit. Required.
    #[serde(default = "default_count")]
    pub count: usize,
    /// Target selectivity. Accepts either a scalar (`0.1`) or a
    /// closed interval (`0.05..0.20`) — the generator draws a
    /// uniform random target per predicate within the interval.
    #[serde(default = "default_selectivity")]
    pub selectivity: SelectivitySpec,
    /// RNG seed for deterministic replay. Required for
    /// reproducibility; defaults to a fixed value so the same
    /// file always produces the same output unless edited.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Optional reference to a survey JSON file. When absent,
    /// the generator runs an inline survey on `source`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub survey: Option<String>,
    /// Optional metadata source (slab). Required when `survey`
    /// is absent; the generator surveys this path inline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Optional output slab path. When set, the generator writes
    /// to this path unless the CLI passes `--output` (CLI wins).
    /// Captured here so a single `--proto-file=<f>` invocation
    /// is enough to replay the wizard's choices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

fn default_count() -> usize { 100 }
fn default_seed() -> u64 { 42 }
fn default_selectivity() -> SelectivitySpec { SelectivitySpec::Scalar(0.1) }

/// Selectivity spec — a single value or an interval. Serialises
/// as either a YAML number or a string in `lo..hi` form so the
/// proto file stays human-readable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectivitySpec {
    Scalar(f64),
    Interval { lo: f64, hi: f64 },
}

impl SelectivitySpec {
    /// Sample a target sel per emission. For `Scalar` returns
    /// the pinned value; for `Interval` draws uniformly in
    /// `[lo, hi]`.
    pub fn sample(self, rng: &mut Xoshiro256PlusPlus) -> f64 {
        match self {
            Self::Scalar(v) => v,
            Self::Interval { lo, hi } => {
                if (hi - lo).abs() < f64::EPSILON { lo } else { rng.random_range(lo..=hi) }
            }
        }
    }

    /// Default value used by `clap`-driven CLI paths so the proto
    /// loader and the flag-driven loader produce the same shape.
    pub fn from_flags(sel: f64, sel_max: Option<f64>) -> Self {
        match sel_max {
            Some(max) => Self::Interval { lo: sel, hi: max },
            None => Self::Scalar(sel),
        }
    }
}

impl Serialize for SelectivitySpec {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        match self {
            // Scalars round-trip as plain YAML numbers, intervals
            // as `lo..hi` strings — the latter is more
            // human-readable than a two-element list.
            Self::Scalar(v) => ser.serialize_f64(*v),
            Self::Interval { lo, hi } => ser.serialize_str(&format!("{lo}..{hi}")),
        }
    }
}

impl<'de> Deserialize<'de> for SelectivitySpec {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        // Accept either a YAML number or a `lo..hi` string.
        let v: serde_yaml::Value = serde::Deserialize::deserialize(de)?;
        match v {
            serde_yaml::Value::Number(n) => {
                let f = n.as_f64().ok_or_else(|| serde::de::Error::custom(
                    "selectivity number must fit in f64"))?;
                Ok(Self::Scalar(f))
            }
            serde_yaml::Value::String(s) => {
                if let Some((lo, hi)) = s.split_once("..") {
                    let lo: f64 = lo.trim().parse().map_err(|_| serde::de::Error::custom(
                        format!("selectivity 'lo' is not a number: {lo:?}")))?;
                    let hi: f64 = hi.trim().parse().map_err(|_| serde::de::Error::custom(
                        format!("selectivity 'hi' is not a number: {hi:?}")))?;
                    if lo > hi {
                        return Err(serde::de::Error::custom(
                            format!("selectivity interval is inverted: {lo}..{hi}")));
                    }
                    Ok(Self::Interval { lo, hi })
                } else if let Ok(f) = s.parse::<f64>() {
                    Ok(Self::Scalar(f))
                } else {
                    Err(serde::de::Error::custom(format!(
                        "selectivity must be a number or 'lo..hi' string, got {s:?}")))
                }
            }
            other => Err(serde::de::Error::custom(format!(
                "selectivity must be a number or 'lo..hi' string, got {other:?}"))),
        }
    }
}

/// Cheap heuristic — declares a byte buffer "binary" if any of
/// its first 256 bytes is a NUL or a non-printable control
/// outside the usual whitespace set. JSON and YAML never
/// contain raw NULs, and slab files virtually always start
/// with binary header bytes, so this catches the
/// `--proto-file=<binary>` mistake without false positives on
/// legitimate UTF-8 text.
fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(256).any(|&b| {
        b == 0 || (b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r')
    })
}

impl PredicateProto {
    /// Load a proto from a file, dispatching on extension.
    /// `.json` → JSON; anything else → YAML. The two formats
    /// are otherwise interchangeable — both produce the same
    /// `PredicateProto` struct, so a `.proto.json` saved by
    /// the wizard and a hand-edited `.proto.yaml` both feed
    /// the same generator code path.
    ///
    /// Refuses obvious mismatches early with a clear message:
    /// `.slab` files (binary slab format), or any file whose
    /// first bytes don't look like JSON/YAML, fail with a
    /// hint instead of a cryptic UTF-8 error from the parser.
    pub fn load_from_path(path: &Path) -> Result<Self, String> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext.eq_ignore_ascii_case("slab") {
            return Err(format!(
                "{} looks like a slab file, not a proto. \
                 Use --output for the slab; --proto-file expects a .json or .yaml proto.",
                path.display(),
            ));
        }
        let bytes = std::fs::read(path)
            .map_err(|e| format!("read {}: {}", path.display(), e))?;
        if looks_binary(&bytes) {
            return Err(format!(
                "{} appears to be a binary file, not a JSON/YAML proto. \
                 --proto-file expects a .proto.json or .proto.yaml.",
                path.display(),
            ));
        }
        let text = std::str::from_utf8(&bytes).map_err(|e| format!(
            "read {}: file is not valid UTF-8 ({}). \
             --proto-file expects a .proto.json or .proto.yaml.",
            path.display(), e,
        ))?;
        if ext.eq_ignore_ascii_case("json") {
            Self::from_json(text)
        } else {
            Self::from_yaml(text)
        }
    }

    /// Parse a proto from a YAML string.
    pub fn from_yaml(text: &str) -> Result<Self, String> {
        serde_yaml::from_str(text)
            .map_err(|e| format!("parse proto YAML: {}", e))
    }

    /// Parse a proto from a JSON string.
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text)
            .map_err(|e| format!("parse proto JSON: {}", e))
    }

    /// Serialise to YAML — used by hand-written fixtures and
    /// the legacy file format.
    pub fn to_yaml(&self) -> Result<String, String> {
        serde_yaml::to_string(self).map_err(|e| format!("serialise proto YAML: {}", e))
    }

    /// Serialise to pretty-printed JSON — the wizard's default
    /// output format for freshly-constructed protos.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("serialise proto JSON: {}", e))
    }

    /// Save the proto to disk, dispatching format on the file
    /// extension. `.json` → JSON; anything else → YAML.
    /// Creates parent directories as needed.
    pub fn save_to_path(&self, path: &Path) -> Result<(), String> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let body = if ext.eq_ignore_ascii_case("json") {
            self.to_json()?
        } else {
            self.to_yaml()?
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
        std::fs::write(path, body)
            .map_err(|e| format!("write {}: {e}", path.display()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Template tree
// ─────────────────────────────────────────────────────────────────────────────

/// Parsed template — same shape as a `PNode` but each leaf
/// records an `Option<Comparand>` per comparand slot (None = `?`
/// placeholder, Some = literal pin).
#[derive(Debug, Clone, PartialEq)]
pub enum PredicateTemplate {
    Predicate {
        field: FieldRef,
        op: OpType,
        /// One slot per comparand position. `None` = the user
        /// wrote `?` and the generator must fill from the
        /// survey. `Some(c)` = literal pin.
        comparands: Vec<Option<Comparand>>,
    },
    Conjugate {
        conjugate_type: ConjugateType,
        children: Vec<PredicateTemplate>,
    },
}

/// Parse a template string in the PNode Display vernacular,
/// accepting `?` as a comparand placeholder. See module docs
/// for the grammar.
pub fn parse_template(text: &str) -> Result<PredicateTemplate, String> {
    let mut p = Parser { input: text, cursor: 0 };
    let node = p.parse_node()?;
    p.skip_ws();
    if p.cursor < p.input.len() {
        return Err(format!("trailing input after template at byte {}", p.cursor));
    }
    Ok(node)
}

// ─────────────────────────────────────────────────────────────────────────────
// Materialisation — fill `?` slots from a survey to hit target sel
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a concrete [`PNode`] from a template by drawing
/// comparand values from the survey. Each `?` slot is filled
/// independently; literal pins pass through unchanged.
///
/// `target_sel` applies to the WHOLE tree. For conjugates with N
/// `?` leaves, each leaf is targeted at `target_sel^(1/N)` so
/// the AND/OR of independent fills approximates the requested
/// total (matches the existing `generate_compound_predicate`
/// math).
pub fn materialize_predicate(
    template: &PredicateTemplate,
    fields: &HashMap<String, FieldProfile>,
    target_sel: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<PNode, String> {
    let leaf_count = count_placeholder_leaves(template);
    let per_leaf_sel = if leaf_count > 1 {
        target_sel.powf(1.0 / leaf_count as f64)
    } else {
        target_sel
    };
    materialize_inner(template, fields, per_leaf_sel, rng)
}

fn materialize_inner(
    template: &PredicateTemplate,
    fields: &HashMap<String, FieldProfile>,
    target_sel: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<PNode, String> {
    match template {
        PredicateTemplate::Predicate { field, op, comparands } => {
            let mut filled = Vec::with_capacity(comparands.len());
            for slot in comparands {
                match slot {
                    Some(c) => filled.push(c.clone()),
                    None => {
                        let field_name = match field {
                            FieldRef::Named(n) => n.clone(),
                            FieldRef::Index(i) => return Err(format!(
                                "template references field[{i}] but materialisation needs a field NAME \
                                 (survey is keyed by name)")),
                        };
                        let profile = fields.get(&field_name).ok_or_else(|| format!(
                            "template references field '{field_name}' but the survey has no profile for it"))?;
                        filled.push(super::gen_predicates::draw_comparand_for_field(
                            profile, *op, target_sel, rng,
                        ).ok_or_else(|| format!(
                            "no measure on field '{field_name}' can produce a {:?} comparand at sel {:.4}",
                            op, target_sel))?);
                    }
                }
            }
            Ok(PNode::Predicate(PredicateNode {
                field: field.clone(),
                op: *op,
                comparands: filled,
            }))
        }
        PredicateTemplate::Conjugate { conjugate_type, children } => {
            let materialised: Result<Vec<PNode>, _> = children.iter()
                .map(|c| materialize_inner(c, fields, target_sel, rng))
                .collect();
            Ok(PNode::Conjugate(ConjugateNode {
                conjugate_type: *conjugate_type,
                children: materialised?,
            }))
        }
    }
}

fn count_placeholder_leaves(t: &PredicateTemplate) -> usize {
    match t {
        PredicateTemplate::Predicate { comparands, .. } => {
            if comparands.iter().any(|c| c.is_none()) { 1 } else { 0 }
        }
        PredicateTemplate::Conjugate { children, .. } => {
            children.iter().map(count_placeholder_leaves).sum()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Template parser — Display grammar with `?` as comparand
// ─────────────────────────────────────────────────────────────────────────────
//
// Grammar mirrors `veks_core::formats::pnode::from_display`,
// extended at one position: `comparand := ? | <display-comparand>`.
// Keeping the parser local (rather than reusing the veks-anode
// parser with surgery) lets `?` semantics stay a generation-
// time concern and keeps the codec layer pure.

struct Parser<'a> {
    input: &'a str,
    cursor: usize,
}

impl<'a> Parser<'a> {
    fn rest(&self) -> &'a str { &self.input[self.cursor..] }
    fn peek_char(&self) -> Option<char> { self.rest().chars().next() }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() { self.cursor += c.len_utf8(); } else { break; }
        }
    }

    fn consume_char(&mut self, expected: char) -> Result<(), String> {
        match self.peek_char() {
            Some(c) if c == expected => {
                self.cursor += c.len_utf8();
                Ok(())
            }
            Some(c) => Err(format!("expected '{expected}' at byte {}, got '{c}'", self.cursor)),
            None => Err(format!("expected '{expected}' at byte {}, got EOF", self.cursor)),
        }
    }

    fn try_consume_literal(&mut self, lit: &str) -> bool {
        if self.rest().starts_with(lit) {
            self.cursor += lit.len();
            true
        } else { false }
    }

    fn try_consume_keyword(&mut self, kw: &str) -> bool {
        if !self.rest().starts_with(kw) { return false; }
        let after = &self.rest()[kw.len()..];
        let next = after.chars().next();
        let ok = matches!(next, None | Some(' ') | Some('\t') | Some('\n') | Some('('));
        if ok { self.cursor += kw.len(); }
        ok
    }

    fn parse_node(&mut self) -> Result<PredicateTemplate, String> {
        self.skip_ws();
        if self.peek_char() == Some('(') {
            self.parse_conjugate_or_paren()
        } else {
            self.parse_predicate()
        }
    }

    fn parse_conjugate_or_paren(&mut self) -> Result<PredicateTemplate, String> {
        self.consume_char('(')?;
        let first = self.parse_node()?;
        self.skip_ws();
        let kind = if self.try_consume_keyword("AND") {
            Some(ConjugateType::And)
        } else if self.try_consume_keyword("OR") {
            Some(ConjugateType::Or)
        } else { None };
        match kind {
            None => {
                self.skip_ws();
                self.consume_char(')')?;
                Ok(first)
            }
            Some(k) => {
                let mut children = vec![first];
                self.skip_ws();
                children.push(self.parse_node()?);
                let kw = match k {
                    ConjugateType::And => "AND",
                    ConjugateType::Or => "OR",
                    ConjugateType::Pred => unreachable!(),
                };
                loop {
                    self.skip_ws();
                    if !self.try_consume_keyword(kw) { break; }
                    self.skip_ws();
                    children.push(self.parse_node()?);
                }
                self.skip_ws();
                self.consume_char(')')?;
                Ok(PredicateTemplate::Conjugate { conjugate_type: k, children })
            }
        }
    }

    fn parse_predicate(&mut self) -> Result<PredicateTemplate, String> {
        let field = self.parse_field_ref()?;
        self.skip_ws();
        let op = self.parse_op()?;
        self.skip_ws();
        let comparands = self.parse_comparand_list()?;
        if comparands.is_empty() {
            return Err(format!("predicate has no comparands at byte {}", self.cursor));
        }
        Ok(PredicateTemplate::Predicate { field, op, comparands })
    }

    fn parse_field_ref(&mut self) -> Result<FieldRef, String> {
        if self.rest().starts_with("field[") {
            self.cursor += "field[".len();
            let start = self.cursor;
            while let Some(c) = self.peek_char() {
                if c.is_ascii_digit() { self.cursor += 1; } else { break; }
            }
            let digits = &self.input[start..self.cursor];
            if digits.is_empty() {
                return Err(format!("expected digits after 'field[' at byte {}", start));
            }
            let n: u8 = digits.parse()
                .map_err(|_| format!("field index does not fit in u8 at byte {}", start))?;
            self.consume_char(']')?;
            return Ok(FieldRef::Index(n));
        }
        let start = self.cursor;
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() || c == '(' || c == ')' { break; }
            self.cursor += c.len_utf8();
        }
        if self.cursor == start {
            return Err(format!("expected field name at byte {}", start));
        }
        Ok(FieldRef::Named(self.input[start..self.cursor].to_string()))
    }

    fn parse_op(&mut self) -> Result<OpType, String> {
        if self.try_consume_literal("!=") { return Ok(OpType::Ne); }
        if self.try_consume_literal(">=") { return Ok(OpType::Ge); }
        if self.try_consume_literal("<=") { return Ok(OpType::Le); }
        if self.try_consume_literal(">") { return Ok(OpType::Gt); }
        if self.try_consume_literal("<") { return Ok(OpType::Lt); }
        if self.try_consume_literal("=") { return Ok(OpType::Eq); }
        if self.try_consume_keyword("IN") { return Ok(OpType::In); }
        if self.try_consume_keyword("MATCHES") { return Ok(OpType::Matches); }
        Err(format!("expected operator at byte {}", self.cursor))
    }

    fn parse_comparand_list(&mut self) -> Result<Vec<Option<Comparand>>, String> {
        self.skip_ws();
        if self.peek_char() == Some('(') {
            self.consume_char('(')?;
            let mut out = Vec::new();
            self.skip_ws();
            if self.peek_char() == Some(')') {
                self.consume_char(')')?;
                return Ok(out);
            }
            out.push(self.parse_comparand_slot()?);
            loop {
                self.skip_ws();
                if !self.try_consume_literal(",") { break; }
                self.skip_ws();
                out.push(self.parse_comparand_slot()?);
            }
            self.skip_ws();
            self.consume_char(')')?;
            Ok(out)
        } else {
            Ok(vec![self.parse_comparand_slot()?])
        }
    }

    fn parse_comparand_slot(&mut self) -> Result<Option<Comparand>, String> {
        self.skip_ws();
        if self.peek_char() == Some('?') {
            self.cursor += 1;
            return Ok(None);
        }
        // Delegate to the strict veks-anode parser for the literal
        // forms. We do this by carving off a minimal slice that
        // ends at the next ',' or ')' boundary (respecting string
        // literals) and reusing the same character-class predicates.
        let lit_start = self.cursor;
        let mut in_string = false;
        let mut depth: i32 = 0;
        while let Some(c) = self.peek_char() {
            match c {
                '\'' => { in_string = !in_string; self.cursor += 1; }
                '(' if !in_string => { depth += 1; self.cursor += 1; }
                ')' if !in_string => {
                    if depth == 0 { break; }
                    depth -= 1; self.cursor += 1;
                }
                ',' if !in_string && depth == 0 => break,
                c if c.is_whitespace() && !in_string && depth == 0 => break,
                _ => { self.cursor += c.len_utf8(); }
            }
        }
        let lit = &self.input[lit_start..self.cursor];
        if lit.is_empty() {
            return Err(format!("expected comparand or '?' at byte {}", lit_start));
        }
        let c = parse_literal_comparand(lit)
            .map_err(|e| format!("at byte {lit_start}: {e}"))?;
        Ok(Some(c))
    }
}

/// Parse a single Display-format comparand literal (no `?`). We
/// re-derive the Comparand parser here rather than reaching into
/// veks-anode private helpers — the grammar is small enough.
fn parse_literal_comparand(s: &str) -> Result<Comparand, String> {
    let s = s.trim();
    if s == "NULL" { return Ok(Comparand::Null); }
    if s == "true" { return Ok(Comparand::Bool(true)); }
    if s == "false" { return Ok(Comparand::Bool(false)); }
    if s.starts_with("X'") && s.ends_with('\'') {
        let hex = &s[2..s.len()-1];
        if !hex.len().is_multiple_of(2) {
            return Err("odd-length hex literal".into());
        }
        let mut out = Vec::with_capacity(hex.len()/2);
        for chunk in hex.as_bytes().chunks(2) {
            let hi = decode_hex(chunk[0]).ok_or("non-hex character")?;
            let lo = decode_hex(chunk[1]).ok_or("non-hex character")?;
            out.push((hi << 4) | lo);
        }
        return Ok(Comparand::Bytes(out));
    }
    if s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2 {
        return Ok(Comparand::Text(s[1..s.len()-1].to_string()));
    }
    // Numeric
    if s.contains('.') || s.contains('e') || s.contains('E') {
        let v: f64 = s.parse().map_err(|_| format!("invalid float: {s:?}"))?;
        return Ok(Comparand::Float(v));
    }
    let v: i64 = s.parse().map_err(|_| format!("invalid integer: {s:?}"))?;
    Ok(Comparand::Int(v))
}

fn decode_hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_concrete_template_round_trips_via_display() {
        let template_str = "age >= 18";
        let t = parse_template(template_str).unwrap();
        match t {
            PredicateTemplate::Predicate { field, op, comparands } => {
                assert_eq!(field, FieldRef::Named("age".into()));
                assert_eq!(op, OpType::Ge);
                assert_eq!(comparands, vec![Some(Comparand::Int(18))]);
            }
            _ => panic!("expected predicate"),
        }
    }

    #[test]
    fn parse_placeholder() {
        let t = parse_template("age >= ?").unwrap();
        match t {
            PredicateTemplate::Predicate { comparands, .. } => {
                assert_eq!(comparands, vec![None]);
            }
            _ => panic!("expected predicate"),
        }
    }

    #[test]
    fn parse_conjugate_mixed() {
        let t = parse_template("(age >= ? AND name MATCHES ?)").unwrap();
        match t {
            PredicateTemplate::Conjugate { conjugate_type, children } => {
                assert_eq!(conjugate_type, ConjugateType::And);
                assert_eq!(children.len(), 2);
                assert!(matches!(children[0], PredicateTemplate::Predicate { .. }));
                assert!(matches!(children[1], PredicateTemplate::Predicate { .. }));
            }
            _ => panic!("expected AND conjugate"),
        }
    }

    #[test]
    fn parse_pinned_literal_text() {
        let t = parse_template("name = 'alice'").unwrap();
        match t {
            PredicateTemplate::Predicate { comparands, .. } => {
                assert_eq!(comparands, vec![Some(Comparand::Text("alice".into()))]);
            }
            _ => panic!("expected predicate"),
        }
    }

    #[test]
    fn parse_proto_yaml_round_trip() {
        let yaml = r#"
template: "(age >= ? AND name MATCHES ?)"
count: 50
selectivity: 0.1
seed: 7
"#;
        let proto = PredicateProto::from_yaml(yaml).unwrap();
        assert_eq!(proto.count, 50);
        assert_eq!(proto.seed, 7);
        assert!(matches!(proto.selectivity, SelectivitySpec::Scalar(v) if (v - 0.1).abs() < 1e-9));
        let _ = parse_template(&proto.template).unwrap();
    }

    #[test]
    fn parse_proto_yaml_with_interval_selectivity() {
        let yaml = r#"
template: "age >= ?"
selectivity: "0.05..0.20"
"#;
        let proto = PredicateProto::from_yaml(yaml).unwrap();
        match proto.selectivity {
            SelectivitySpec::Interval { lo, hi } => {
                assert!((lo - 0.05).abs() < 1e-9);
                assert!((hi - 0.20).abs() < 1e-9);
            }
            _ => panic!("expected interval"),
        }
    }

    #[test]
    fn rejects_inverted_interval() {
        let yaml = r#"
template: "age >= ?"
selectivity: "0.5..0.1"
"#;
        let r = PredicateProto::from_yaml(yaml);
        assert!(r.is_err(), "inverted interval must error");
    }

    /// JSON round-trip through `to_json` / `from_json`. The
    /// wizard's default save format — must produce a string
    /// that `load_from_path` reads back into the same struct.
    #[test]
    fn proto_json_serialise_round_trip() {
        let proto = PredicateProto {
            template: "(age >= ? AND name MATCHES ?)".into(),
            count: 50,
            selectivity: SelectivitySpec::Interval { lo: 0.05, hi: 0.20 },
            seed: 7,
            survey: None,
            source: Some("meta.slab".into()),
            output: None,
        };
        let json = proto.to_json().unwrap();
        // Sanity: JSON contains the template verbatim.
        assert!(json.contains("(age >= ? AND name MATCHES ?)"),
            "template should appear verbatim in JSON: {json}");
        let parsed = PredicateProto::from_json(&json).unwrap();
        assert_eq!(parsed.template, proto.template);
        assert_eq!(parsed.count, proto.count);
        assert_eq!(parsed.seed, proto.seed);
    }

    /// A `.slab` file passed to `--proto-file` fails fast with
    /// a message that points the user at the right flag, instead
    /// of the cryptic `stream did not contain valid UTF-8` we
    /// used to surface from the YAML parser.
    #[test]
    fn load_from_path_rejects_slab_extension_with_helpful_message() {
        let tmp = tempfile::tempdir().unwrap();
        let slab_path = tmp.path().join("predicates.slab");
        std::fs::write(&slab_path, b"\x00\x01\x02SLAB").unwrap();
        let err = PredicateProto::load_from_path(&slab_path).unwrap_err();
        assert!(err.contains(".slab") || err.contains("slab file"),
            "error should call out the .slab mismatch, got: {err}");
        assert!(err.contains("--proto-file") || err.contains("--output"),
            "error should tell user which flag they probably wanted: {err}");
    }

    /// A binary file with a non-slab extension (or no extension)
    /// also fails fast — the heuristic looks at the file's
    /// first bytes, not just its name.
    #[test]
    fn load_from_path_rejects_binary_content() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("noisy.json");
        // JSON extension but binary bytes: classic
        // copy/paste-the-wrong-file mistake.
        std::fs::write(&path, b"\x00\x00\x00\x00binary garbage").unwrap();
        let err = PredicateProto::load_from_path(&path).unwrap_err();
        assert!(err.contains("binary") || err.contains("JSON/YAML"),
            "binary content should be reported as such: {err}");
    }

    /// `load_from_path` dispatches on file extension — `.json`
    /// → JSON loader; everything else → YAML loader. Confirms
    /// the wizard's `.proto.json` and a hand-written
    /// `.proto.yaml` both feed the same generator.
    #[test]
    fn load_from_path_dispatches_on_extension() {
        let tmp = tempfile::tempdir().unwrap();
        let proto = PredicateProto {
            template: "age = ?".into(),
            count: 1,
            selectivity: SelectivitySpec::Scalar(0.1),
            seed: 1,
            survey: None,
            source: None,
            output: None,
        };
        let json_path = tmp.path().join("p.proto.json");
        let yaml_path = tmp.path().join("p.proto.yaml");
        proto.save_to_path(&json_path).unwrap();
        proto.save_to_path(&yaml_path).unwrap();
        let loaded_json = PredicateProto::load_from_path(&json_path).unwrap();
        let loaded_yaml = PredicateProto::load_from_path(&yaml_path).unwrap();
        assert_eq!(loaded_json.template, proto.template);
        assert_eq!(loaded_yaml.template, proto.template);
        // The on-disk bytes differ (JSON vs YAML), but the
        // parsed structs are identical.
        let json_text = std::fs::read_to_string(&json_path).unwrap();
        let yaml_text = std::fs::read_to_string(&yaml_path).unwrap();
        assert!(json_text.starts_with('{'), "JSON should start with '{{', got: {json_text}");
        assert!(!yaml_text.starts_with('{'), "YAML shouldn't start with '{{', got: {yaml_text}");
    }

    #[test]
    fn proto_yaml_serialise_round_trip() {
        let proto = PredicateProto {
            template: "(age >= ? AND name MATCHES ?)".into(),
            count: 50,
            selectivity: SelectivitySpec::Interval { lo: 0.05, hi: 0.20 },
            seed: 7,
            survey: None,
            source: Some("meta.slab".into()),
            output: None,
        };
        let yaml = proto.to_yaml().unwrap();
        let round_tripped = PredicateProto::from_yaml(&yaml).unwrap();
        assert_eq!(round_tripped.template, proto.template);
        assert_eq!(round_tripped.count, proto.count);
        assert_eq!(round_tripped.seed, proto.seed);
        assert_eq!(round_tripped.source, proto.source);
        match (round_tripped.selectivity, proto.selectivity) {
            (SelectivitySpec::Interval { lo: a, hi: b },
             SelectivitySpec::Interval { lo: c, hi: d }) => {
                assert!((a - c).abs() < 1e-9);
                assert!((b - d).abs() < 1e-9);
            }
            _ => panic!("selectivity shape mismatch"),
        }
    }
}
