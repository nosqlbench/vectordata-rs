// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Profile selectors: the language after the colon in a dataset spec.
//!
//! `dataset:10m` names one profile, as it always has. `dataset:size=10m,
//! predicates=uniform*` names the *set* of profiles whose attributes
//! match, and `or(10m,20m)`, `not(family=uniform)`, `selectivity=1e-3..1e-2`
//! compose atoms over a profile's automatic `profile` tag, its
//! structural fields, and its declared `attributes:`.
//!
//! The grammar, value readings, matching rules, and the way a spec is
//! split into a dataset head and a selector are the ones
//! `docs/design/srd-profile-selectors.md` states (PS-1 to PS-8). This
//! module is the one parser every surface calls (PS-14); the surfaces
//! decide what a set means to them (PS-9).

use std::fmt;

use serde_yaml::Value as Yaml;

use crate::dataset::profile::DSProfileGroup;
use crate::model::ProfileConfig;

/// A parsed selector expression.
#[derive(Debug, Clone)]
pub struct Selector {
    expr: Expr,
    text: String,
}

#[derive(Debug, Clone)]
enum Expr {
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Not(Vec<Expr>),
    Atom(Atom),
}

#[derive(Debug, Clone)]
struct Atom {
    key: String,
    op: Op,
    value: Value,
}

/// A comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// How a value was read from its spelling (PS-5). Every reading folds
/// case: patterns and literals are matched case-insensitively, and
/// count suffixes are read whichever way they are written.
#[derive(Debug, Clone)]
enum Value {
    /// `^…` or `…$`: a regular expression over the whole canonical text.
    Regex(regex::Regex),
    /// Contains `*`, `?`, or `[`: a glob over the whole canonical text.
    Glob(String),
    /// `lo..hi`: a half-open numeric interval.
    Interval(f64, f64),
    /// A number, possibly count-suffixed (`10m`, `128mi`, `1e-3`).
    Number(f64),
    Bool(bool),
    /// Anything else, or anything quoted.
    Literal(String),
}

/// A selector that could not be parsed, with where it went wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectorError {
    /// Byte offset into the selector text.
    pub position: usize,
    pub message: String,
}

impl fmt::Display for SelectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "selector error at byte {}: {}", self.position, self.message)
    }
}

impl std::error::Error for SelectorError {}

/// What a selector is evaluated against: one profile's automatic
/// `profile` tag, its structural fields, and its declared attributes.
/// Built by the loader from a profile **as loaded**, after inheritance,
/// so an inherited `base_count` is selectable (PS-7); attributes are
/// never inherited (PS-19), so the map is the profile's own.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProfileFacts {
    pub name: String,
    pub base_count: Option<u64>,
    pub maxk: Option<u32>,
    pub partition: bool,
    pub inherits: Option<String>,
    pub attributes: Vec<(String, Yaml)>,
}

/// The keys a selector reads before the attribute map (PS-7).
pub const STRUCTURAL_KEYS: &[&str] = &["profile", "base_count", "maxk", "partition", "inherits"];

impl ProfileFacts {
    /// The facts of a profile **as loaded** (PS-7): structure after
    /// inheritance, attributes its own (PS-19). Attribute order is by
    /// key, because the loaded map has none and a message that lists
    /// them must be stable.
    pub fn of_profile(name: &str, profile: &ProfileConfig) -> Self {
        let mut attributes: Vec<(String, Yaml)> = profile
            .attributes
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        attributes.sort_by(|a, b| a.0.cmp(&b.0));
        Self {
            name: name.to_string(),
            base_count: profile.base_count,
            maxk: profile.maxk,
            partition: profile.partition,
            inherits: profile.inherits.clone(),
            attributes,
        }
    }

    /// The facts of a profile **as declared** in a writer-side group,
    /// with `base_count` and `maxk` read through the `inherits` chain
    /// the way the reader resolves them (PS-7), so a catalog entry
    /// selects the same profiles a loaded dataset does. Attributes are
    /// the profile's own, in declaration order (PS-19).
    pub fn of_declared(name: &str, group: &DSProfileGroup) -> Option<Self> {
        let profile = group.profiles.get(name)?;
        let (base_count, maxk) = declared_structure(group, name);
        Some(Self {
            name: name.to_string(),
            base_count,
            maxk,
            partition: profile.partition,
            inherits: profile.inherits.clone(),
            attributes: profile
                .attributes
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        })
    }

    /// One line naming the profile and everything a selector can read
    /// of it, for a message that shows what was on offer.
    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(n) = self.base_count {
            parts.push(format!("base_count={n}"));
        }
        if let Some(k) = self.maxk {
            parts.push(format!("maxk={k}"));
        }
        if self.partition {
            parts.push("partition=true".into());
        }
        if let Some(p) = &self.inherits {
            parts.push(format!("inherits={p}"));
        }
        for (k, v) in &self.attributes {
            parts.push(format!("{k}={}", yaml_summary(v)));
        }
        if parts.is_empty() {
            self.name.clone()
        } else {
            format!("{} ({})", self.name, parts.join(", "))
        }
    }
}

/// `base_count` and `maxk` of a declared profile as the reader would
/// see them: its own, else its parent's. `maxk` crosses any parent;
/// `base_count` crosses only a named parent, because inheriting from
/// `default` is a size step and the child's count is its own (P-2). A
/// partition profile is self-contained and inherits nothing.
fn declared_structure(group: &DSProfileGroup, name: &str) -> (Option<u64>, Option<u32>) {
    let Some(own) = group.profiles.get(name) else {
        return (None, None);
    };
    let mut base_count = own.base_count;
    let mut maxk = own.maxk;
    let mut current = name.to_string();
    // A chain is at most one hop per profile; a cycle stops here.
    for _ in 0..group.profiles.len() {
        let Some(p) = group.profiles.get(&current) else { break };
        if p.partition || current == "default" {
            break;
        }
        let parent = match p.inherits.as_deref() {
            Some(i) if i != current && group.profiles.contains_key(i) => i.to_string(),
            _ => "default".to_string(),
        };
        let Some(pp) = group.profiles.get(&parent) else { break };
        maxk = maxk.or(pp.maxk);
        if parent != "default" {
            base_count = base_count.or(pp.base_count);
        }
        current = parent;
    }
    (base_count, maxk)
}

fn yaml_summary(v: &Yaml) -> String {
    match v {
        Yaml::String(s) => s.clone(),
        other => crate::dataset::yaml_edit::render_scalar(other)
            .unwrap_or_else(|_| serde_yaml::to_string(other).unwrap_or_default().trim().replace('\n', " ")),
    }
}

/// Why a selector produced no usable selection (PS-9, PS-10).
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionError {
    /// The selector text is malformed.
    Syntax(SelectorError),
    /// No profile matches. Carries what was on offer, with everything a
    /// selector can read of each, so the message says why.
    NoMatch {
        selector: String,
        profiles: Vec<ProfileFacts>,
    },
    /// A single-profile surface was given a selector matching more than
    /// one profile.
    Ambiguous {
        selector: String,
        matches: Vec<String>,
    },
    /// No selector was given and the dataset has no `default` profile.
    NoDefault { profiles: Vec<String> },
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectionError::Syntax(e) => {
                write!(f, "selector error at {}: {}", e.position, e.message)
            }
            SelectionError::NoMatch { selector, profiles } => {
                writeln!(f, "selector `{selector}` matches no profile; the profiles are:")?;
                for p in profiles {
                    writeln!(f, "  {}", p.summary())?;
                }
                Ok(())
            }
            SelectionError::Ambiguous { selector, matches } => write!(
                f,
                "selector `{selector}` matches {} profiles ({}) where one is required; \
                 narrow it, or use a surface that takes a set",
                matches.len(),
                matches.join(", ")
            ),
            SelectionError::NoDefault { profiles } => write!(
                f,
                "no selector given and the dataset has no `default` profile; name one of: {}",
                profiles.join(", ")
            ),
        }
    }
}

impl std::error::Error for SelectionError {}

impl From<SelectionError> for crate::Error {
    fn from(e: SelectionError) -> Self {
        crate::Error::Other(e.to_string())
    }
}

/// The profiles a spec's selector names, in the order `facts` lists
/// them. No selector means `default` (PS-10); `profile=*` means all; a
/// selector matching nothing is an error naming what was on offer.
pub fn resolve(
    selector: Option<&str>,
    facts: &[ProfileFacts],
) -> Result<Vec<String>, SelectionError> {
    let Some(text) = selector else {
        return if facts.iter().any(|f| f.name == "default") {
            Ok(vec!["default".to_string()])
        } else {
            Err(SelectionError::NoDefault {
                profiles: facts.iter().map(|f| f.name.clone()).collect(),
            })
        };
    };
    let selector = Selector::parse(text).map_err(SelectionError::Syntax)?;
    let names = selector.select(facts);
    if names.is_empty() {
        return Err(SelectionError::NoMatch {
            selector: selector.text().to_string(),
            profiles: facts.to_vec(),
        });
    }
    Ok(names)
}

/// The one profile a spec's selector names, for a surface that takes a
/// single profile (PS-10). More than one match is an error, never a
/// silent first.
pub fn resolve_one(
    selector: Option<&str>,
    facts: &[ProfileFacts],
) -> Result<String, SelectionError> {
    let mut names = resolve(selector, facts)?;
    if names.len() == 1 {
        return Ok(names.remove(0));
    }
    Err(SelectionError::Ambiguous {
        selector: selector.unwrap_or_default().trim().to_string(),
        matches: names,
    })
}

impl Selector {
    /// Parse a selector: a bare profile name or an expression (PS-3).
    pub fn parse(text: &str) -> Result<Self, SelectorError> {
        let mut p = Parser { s: text, pos: 0 };
        p.skip_ws();
        if p.at_end() {
            return Err(p.err("empty selector"));
        }
        let expr = p.parse_expr(0)?;
        p.skip_ws();
        if !p.at_end() {
            return Err(p.err(&format!("unexpected '{}'", p.rest_char())));
        }
        Ok(Selector {
            expr,
            text: text.trim().to_string(),
        })
    }

    /// The literal profile name this selector is, if it is one bare
    /// name and nothing else (PS-4, PS-10). Surfaces that print a
    /// profile name use it; every other selector is a set.
    pub fn bare_name(&self) -> Option<&str> {
        match &self.expr {
            Expr::Atom(Atom {
                key,
                op: Op::Eq,
                value: Value::Literal(v),
            }) if key == "profile" && self.text.eq_ignore_ascii_case(v) => Some(self.text.as_str()),
            _ => None,
        }
    }

    /// Whether this selector matches the profile described by `facts`.
    pub fn matches(&self, facts: &ProfileFacts) -> bool {
        eval(&self.expr, facts)
    }

    /// The names of the profiles in `facts` this selector matches, in
    /// the order given (PS-9).
    pub fn select<'a, I>(&self, facts: I) -> Vec<String>
    where
        I: IntoIterator<Item = &'a ProfileFacts>,
    {
        facts
            .into_iter()
            .filter(|f| self.matches(f))
            .map(|f| f.name.clone())
            .collect()
    }

    /// The selector as written.
    pub fn text(&self) -> &str {
        &self.text
    }
}

impl fmt::Display for Selector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsing
// ─────────────────────────────────────────────────────────────────────────────

struct Parser<'a> {
    s: &'a str,
    pos: usize,
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-'
}

impl<'a> Parser<'a> {
    fn at_end(&self) -> bool {
        self.pos >= self.s.len()
    }

    fn rest(&self) -> &'a str {
        &self.s[self.pos..]
    }

    fn rest_char(&self) -> char {
        self.rest().chars().next().unwrap_or('\0')
    }

    fn err(&self, message: &str) -> SelectorError {
        SelectorError {
            position: self.pos,
            message: message.to_string(),
        }
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.rest().chars().next() {
            if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.rest().starts_with(lit) {
            self.pos += lit.len();
            true
        } else {
            false
        }
    }

    fn ident(&mut self) -> Option<&'a str> {
        let start = self.pos;
        while let Some(c) = self.rest().chars().next() {
            if is_ident_char(c) {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
        (self.pos > start).then(|| &self.s[start..self.pos])
    }

    /// `expr := term { "," term }` — a comma is AND (PS-8).
    fn parse_expr(&mut self, depth: usize) -> Result<Expr, SelectorError> {
        let mut terms = vec![self.parse_term(depth)?];
        loop {
            self.skip_ws();
            if self.eat(",") {
                self.skip_ws();
                terms.push(self.parse_term(depth)?);
            } else {
                break;
            }
        }
        Ok(if terms.len() == 1 {
            terms.remove(0)
        } else {
            Expr::And(terms)
        })
    }

    /// `term := and(...) | or(...) | not(...) | atom | name`
    fn parse_term(&mut self, depth: usize) -> Result<Expr, SelectorError> {
        self.skip_ws();
        let start = self.pos;
        let Some(word) = self.ident() else {
            return Err(self.err("expected a profile name or a `key op value` atom"));
        };
        self.skip_ws();
        // A junction is a keyword immediately followed by `(`.
        let lower = word.to_ascii_lowercase();
        if self.rest().starts_with('(')
            && matches!(lower.as_str(), "and" | "or" | "not")
        {
            self.pos += 1;
            self.skip_ws();
            let inner = self.parse_expr(depth + 1)?;
            let terms = match inner {
                Expr::And(v) => v,
                other => vec![other],
            };
            self.skip_ws();
            if !self.eat(")") {
                return Err(self.err(&format!("expected ')' to close `{lower}(`")));
            }
            return Ok(match lower.as_str() {
                "and" => Expr::And(terms),
                "or" => Expr::Or(terms),
                _ => Expr::Not(terms),
            });
        }
        // An atom carries an operator; a bare name is `profile=<name>`.
        let op = if self.eat("!=") {
            Some(Op::Ne)
        } else if self.eat("<=") {
            Some(Op::Le)
        } else if self.eat(">=") {
            Some(Op::Ge)
        } else if self.eat("=") {
            Some(Op::Eq)
        } else if self.eat("<") {
            Some(Op::Lt)
        } else if self.eat(">") {
            Some(Op::Gt)
        } else {
            None
        };
        match op {
            None => {
                self.skip_ws();
                if !(self.at_end() || self.rest().starts_with(',') || self.rest().starts_with(')')) {
                    self.pos = start + word.len();
                    return Err(self.err(&format!(
                        "expected an operator after '{word}' (one of = != < <= > >=)"
                    )));
                }
                Ok(Expr::Atom(Atom {
                    key: "profile".to_string(),
                    op: Op::Eq,
                    value: Value::Literal(word.to_ascii_lowercase()),
                }))
            }
            Some(op) => {
                self.skip_ws();
                let value = self.parse_value(op)?;
                Ok(Expr::Atom(Atom {
                    key: word.to_ascii_lowercase(),
                    op,
                    value,
                }))
            }
        }
    }

    /// `value := quoted | bare` — bare runs to the next `,` or `)`.
    fn parse_value(&mut self, op: Op) -> Result<Value, SelectorError> {
        let start = self.pos;
        if let Some(q) = self.rest().chars().next().filter(|c| *c == '\'' || *c == '"') {
            self.pos += 1;
            let body_start = self.pos;
            match self.rest().find(q) {
                Some(i) => {
                    let body = &self.s[body_start..body_start + i];
                    self.pos = body_start + i + 1;
                    return Ok(Value::Literal(body.to_lowercase()));
                }
                None => return Err(self.err("unterminated quoted value")),
            }
        }
        let mut end = self.pos;
        for (i, c) in self.rest().char_indices() {
            if c == ',' || c == ')' {
                break;
            }
            end = self.pos + i + c.len_utf8();
        }
        let raw = self.s[start..end].trim();
        if raw.is_empty() {
            return Err(self.err("expected a value"));
        }
        // `size==10m` is a slip, not a value of `=10m`; a value that
        // means an operator character literally is quoted.
        if raw.starts_with(['=', '<', '>', '!']) {
            return Err(self.err(&format!(
                "a value cannot begin with '{}'; quote it to mean it literally",
                raw.chars().next().unwrap_or_default()
            )));
        }
        self.pos = end;
        read_bare_value(raw, op).map_err(|m| SelectorError {
            position: start,
            message: m,
        })
    }
}

/// Read a bare value by its spelling (PS-5), in the order the SRD fixes.
fn read_bare_value(raw: &str, op: Op) -> Result<Value, String> {
    if raw.starts_with('^') || raw.ends_with('$') {
        let re = regex::RegexBuilder::new(raw)
            .case_insensitive(true)
            .build()
            .map_err(|e| format!("bad regular expression '{raw}': {e}"))?;
        return Ok(Value::Regex(re));
    }
    if raw.contains('*') || raw.contains('?') || raw.contains('[') {
        return Ok(Value::Glob(raw.to_lowercase()));
    }
    if let Some((lo, hi)) = raw.split_once("..")
        && let (Some(lo), Some(hi)) = (parse_number(lo), parse_number(hi))
    {
        if matches!(op, Op::Lt | Op::Le | Op::Gt | Op::Ge) {
            return Err(format!("an interval '{raw}' takes = or !=, not a comparison"));
        }
        return Ok(Value::Interval(lo, hi));
    }
    if let Some(n) = parse_number(raw) {
        return Ok(Value::Number(n));
    }
    if matches!(op, Op::Lt | Op::Le | Op::Gt | Op::Ge) {
        return Err(format!("'{raw}' is not a number, and {} compares numbers", op_text(op)));
    }
    match raw.to_ascii_lowercase().as_str() {
        "true" => Ok(Value::Bool(true)),
        "false" => Ok(Value::Bool(false)),
        other => Ok(Value::Literal(other.to_string())),
    }
}

fn op_text(op: Op) -> &'static str {
    match op {
        Op::Eq => "=",
        Op::Ne => "!=",
        Op::Lt => "<",
        Op::Le => "<=",
        Op::Gt => ">",
        Op::Ge => ">=",
    }
}

/// A number as a selector spells one: a float, or a count with a
/// suffix under the window grammar (`10m`, `128mi`, `100k`), whichever
/// case the suffix is written in.
pub fn parse_number(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(f) = s.parse::<f64>() {
        return Some(f);
    }
    let candidates = [
        s.to_string(),
        s.to_ascii_lowercase(),
        s.to_ascii_uppercase(),
        {
            // `128mi` → `128Mi`, `10m` → `10M`: the grammar's mixed cases.
            let digits_end = s
                .char_indices()
                .find(|(_, c)| !(c.is_ascii_digit() || *c == '.' || *c == '_'))
                .map(|(i, _)| i)
                .unwrap_or(s.len());
            let (num, unit) = s.split_at(digits_end);
            let mut u = unit.chars();
            match u.next() {
                Some(first) => format!("{num}{}{}", first.to_ascii_uppercase(), u.as_str().to_ascii_lowercase()),
                None => s.to_string(),
            }
        },
    ];
    candidates
        .iter()
        .find_map(|c| super::source::parse_number_with_suffix(c).ok())
        .map(|n| n as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation
// ─────────────────────────────────────────────────────────────────────────────

fn eval(expr: &Expr, facts: &ProfileFacts) -> bool {
    match expr {
        Expr::And(terms) => terms.iter().all(|t| eval(t, facts)),
        Expr::Or(terms) => terms.iter().any(|t| eval(t, facts)),
        Expr::Not(terms) => !terms.iter().all(|t| eval(t, facts)),
        Expr::Atom(atom) => eval_atom(atom, facts),
    }
}

/// Look a key up: structural keys first (PS-7), then the attribute map
/// with a dotted path into map-valued attributes, keys folding case.
fn lookup(facts: &ProfileFacts, key: &str) -> Option<Yaml> {
    match key {
        "profile" => return Some(Yaml::String(facts.name.clone())),
        "base_count" => return facts.base_count.map(|n| Yaml::Number(n.into())),
        "maxk" => return facts.maxk.map(|n| Yaml::Number(n.into())),
        "partition" => return Some(Yaml::Bool(facts.partition)),
        "inherits" => return facts.inherits.clone().map(Yaml::String),
        _ => {}
    }
    let mut parts = key.split('.');
    let head = parts.next()?;
    let mut current = facts
        .attributes
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(head))
        .map(|(_, v)| v.clone())?;
    for part in parts {
        let Yaml::Mapping(m) = current else {
            return None;
        };
        let found = m
            .iter()
            .find(|(k, _)| k.as_str().is_some_and(|k| k.eq_ignore_ascii_case(part)))
            .map(|(_, v)| v.clone())?;
        current = found;
    }
    Some(current)
}

fn eval_atom(atom: &Atom, facts: &ProfileFacts) -> bool {
    // Absent matches nothing under any operator (PS-6).
    let Some(value) = lookup(facts, &atom.key) else {
        return false;
    };
    match value {
        // A list matches when any element does; each atom on its own.
        Yaml::Sequence(items) => items.iter().any(|v| scalar_matches(atom, v)),
        // A map as a whole matches nothing; reach in with a dotted key.
        Yaml::Mapping(_) => false,
        scalar => scalar_matches(atom, &scalar),
    }
}

/// The canonical text of a scalar: strings as written, numbers as YAML
/// would serialise them, booleans as `true`/`false`.
fn canonical_text(v: &Yaml) -> Option<String> {
    match v {
        Yaml::String(s) => Some(s.clone()),
        Yaml::Number(n) => Some(n.to_string()),
        Yaml::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// A scalar's numeric value: a YAML number, or a string that reads as a
/// number under the count rule (`size: 10m`).
fn numeric_value(v: &Yaml) -> Option<f64> {
    match v {
        Yaml::Number(n) => n.as_f64(),
        Yaml::String(s) => parse_number(s),
        _ => None,
    }
}

fn scalar_matches(atom: &Atom, v: &Yaml) -> bool {
    match (&atom.value, atom.op) {
        (Value::Number(n), Op::Eq | Op::Ne) => {
            let hit = numeric_value(v).is_some_and(|x| x == *n);
            if atom.op == Op::Eq { hit } else { !hit }
        }
        (Value::Interval(lo, hi), Op::Eq | Op::Ne) => {
            let hit = numeric_value(v).is_some_and(|x| x >= *lo && x < *hi);
            if atom.op == Op::Eq { hit } else { !hit }
        }
        (Value::Number(n), op) => {
            let Some(x) = numeric_value(v) else {
                return false;
            };
            match op {
                Op::Lt => x < *n,
                Op::Le => x <= *n,
                Op::Gt => x > *n,
                Op::Ge => x >= *n,
                Op::Eq | Op::Ne => unreachable!(),
            }
        }
        (Value::Interval(..), _) => false,
        (Value::Bool(b), Op::Eq | Op::Ne) => {
            let hit = match v {
                Yaml::Bool(x) => x == b,
                other => canonical_text(other).is_some_and(|t| t.eq_ignore_ascii_case(&b.to_string())),
            };
            if atom.op == Op::Eq { hit } else { !hit }
        }
        (Value::Regex(re), Op::Eq | Op::Ne) => {
            let hit = canonical_text(v).is_some_and(|t| re.is_match(&t));
            if atom.op == Op::Eq { hit } else { !hit }
        }
        (Value::Glob(g), Op::Eq | Op::Ne) => {
            let hit = canonical_text(v).is_some_and(|t| glob_match(g, &t.to_lowercase()));
            if atom.op == Op::Eq { hit } else { !hit }
        }
        (Value::Literal(l), Op::Eq | Op::Ne) => {
            let hit = canonical_text(v).is_some_and(|t| t.eq_ignore_ascii_case(l));
            if atom.op == Op::Eq { hit } else { !hit }
        }
        // A comparison against a pattern, boolean, or literal is a parse
        // error already (read_bare_value); nothing reaches here.
        (Value::Bool(_) | Value::Regex(_) | Value::Glob(_) | Value::Literal(_), _) => false,
    }
}

/// Glob over the whole text: `*` any run, `?` one character, `[...]` a
/// class with ranges and a leading `!` or `^` for negation.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    fn go(p: &[char], t: &[char]) -> bool {
        match p.first() {
            None => t.is_empty(),
            Some('*') => (0..=t.len()).any(|i| go(&p[1..], &t[i..])),
            Some('?') => !t.is_empty() && go(&p[1..], &t[1..]),
            Some('[') => {
                let Some(close) = p.iter().position(|c| *c == ']') else {
                    return !t.is_empty() && t[0] == '[' && go(&p[1..], &t[1..]);
                };
                let Some(&c) = t.first() else {
                    return false;
                };
                let class = &p[1..close];
                let (negate, class) = match class.first() {
                    Some('!') | Some('^') => (true, &class[1..]),
                    _ => (false, class),
                };
                let mut hit = false;
                let mut i = 0;
                while i < class.len() {
                    if i + 2 < class.len() && class[i + 1] == '-' {
                        if class[i] <= c && c <= class[i + 2] {
                            hit = true;
                        }
                        i += 3;
                    } else {
                        if class[i] == c {
                            hit = true;
                        }
                        i += 1;
                    }
                }
                hit != negate && go(&p[close + 1..], &t[1..])
            }
            Some(&c) => !t.is_empty() && t[0] == c && go(&p[1..], &t[1..]),
        }
    }
    go(&p, &t)
}

// ─────────────────────────────────────────────────────────────────────────────
// The spec: dataset head and selector
// ─────────────────────────────────────────────────────────────────────────────

/// A dataset spec split into its head and its selector (PS-1, PS-2).
#[derive(Debug, Clone)]
pub struct DatasetSpec {
    /// A catalog name, a path, or a URL, exactly as written.
    pub head: String,
    /// `None` when the spec names no selector: `default` on every
    /// surface (PS-1).
    pub selector: Option<Selector>,
}

impl DatasetSpec {
    /// Split a spec by the head's shape without parsing the selector
    /// (PS-2): what completion needs while a selector is still being
    /// typed. The tail is everything after the head's colon, possibly
    /// empty.
    pub fn split_head(spec: &str) -> (&str, Option<&str>) {
        match head_len(spec) {
            Some(i) => (&spec[..i], Some(&spec[i + 1..])),
            None => (spec, None),
        }
    }

    /// Split by the head's shape and parse what follows. A malformed
    /// selector is a selector error, never a dataset lookup (PS-2).
    pub fn parse(spec: &str) -> Result<Self, SelectorError> {
        let spec = spec.trim();
        let split = head_len(spec);
        let (head, tail) = match split {
            Some(i) => (&spec[..i], &spec[i + 1..]),
            None => (spec, ""),
        };
        let selector = if tail.trim().is_empty() {
            None
        } else {
            Some(Selector::parse(tail).map_err(|e| SelectorError {
                position: e.position + head.len() + 1,
                message: e.message,
            })?)
        };
        Ok(DatasetSpec {
            head: head.to_string(),
            selector,
        })
    }
}

/// Where the head of a spec ends: the byte index of the colon that
/// starts the selector, or `None` when there is no selector.
///
/// A URL's `://` and port belong to the head, as does a Windows drive
/// letter; a path is any head with a separator; a catalog name holds
/// no colon at all.
fn head_len(spec: &str) -> Option<usize> {
    let bytes = spec.as_bytes();
    // URL: scheme "://" authority [/path]
    if let Some(scheme_end) = spec.find("://")
        && spec[..scheme_end]
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
        && spec[..scheme_end].starts_with(|c: char| c.is_ascii_alphabetic())
    {
        let after = scheme_end + 3;
        let path_start = spec[after..].find('/').map(|i| after + i).unwrap_or(spec.len());
        // A colon inside the authority is a port; the selector can only
        // start in the path.
        return spec[path_start..].find(':').map(|i| path_start + i);
    }
    // Windows drive letter: "X:\" or "X:/"
    if bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
    {
        return spec[3..].find(':').map(|i| 3 + i);
    }
    spec.find(':')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(name: &str, attrs: &[(&str, Yaml)]) -> ProfileFacts {
        ProfileFacts {
            name: name.into(),
            base_count: None,
            maxk: None,
            partition: false,
            inherits: None,
            attributes: attrs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect(),
        }
    }
    fn s(v: &str) -> Yaml {
        Yaml::String(v.into())
    }
    fn n(v: f64) -> Yaml {
        Yaml::from(v)
    }
    fn sel(text: &str) -> Selector {
        Selector::parse(text).unwrap_or_else(|e| panic!("{text}: {e}"))
    }
    fn names(selector: &str, all: &[ProfileFacts]) -> Vec<String> {
        sel(selector).select(all.iter())
    }

    fn tessera() -> Vec<ProfileFacts> {
        let mut v = vec![
            ProfileFacts {
                base_count: Some(495_930_736),
                ..facts("default", &[("size", s("495m")), ("family", s("stratified")), ("predicates", s("mixed"))])
            },
            ProfileFacts {
                base_count: Some(10_000_000),
                inherits: Some("default".into()),
                ..facts("10m", &[("size", s("10m")), ("family", s("stratified")), ("predicates", s("mixed")),
                    ("selectivity_ladder", Yaml::Sequence(vec![n(1e-1), n(1e-2), n(1e-3)]))])
            },
            ProfileFacts {
                base_count: Some(10_000_000),
                inherits: Some("10m-unfiltered".into()),
                ..facts("10m-uniform-2-1e-2", &[("size", s("10m")), ("family", s("uniform")), ("predicates", s("uniform-2")), ("selectivity", n(1e-2))])
            },
            ProfileFacts {
                base_count: Some(10_000_000),
                ..facts("10m-uniform-2-1e-3", &[("size", s("10m")), ("family", s("uniform")), ("predicates", s("uniform-2")), ("selectivity", n(1e-3))])
            },
            ProfileFacts {
                base_count: Some(100_000_000),
                ..facts("100m", &[("size", s("100m")), ("family", s("stratified")), ("predicates", s("mixed"))])
            },
        ];
        v.push(ProfileFacts {
            partition: true,
            ..facts("label-3", &[("nested", Yaml::Mapping({
                let mut m = serde_yaml::Mapping::new();
                m.insert(s("kind"), s("oracle"));
                m
            }))])
        });
        v
    }

    /// Case 1 and PS-4: a bare name is the literal profile, never a set.
    #[test]
    fn a_bare_name_is_one_profile() {
        let all = tessera();
        assert_eq!(names("10m", &all), ["10m"]);
        assert_eq!(names("10M", &all), ["10m"], "case folds");
        assert!(names("10", &all).is_empty(), "a literal is not a prefix");
        assert_eq!(sel("10m").bare_name(), Some("10m"));
        assert_eq!(sel("profile=10m").bare_name(), None);
    }

    /// Case 4: each spelling has one reading.
    #[test]
    fn values_are_read_by_their_spelling() {
        let all = tessera();
        assert_eq!(names("profile=10m*", &all), ["10m", "10m-uniform-2-1e-2", "10m-uniform-2-1e-3"]);
        assert_eq!(names("profile=^10m-.*$", &all), ["10m-uniform-2-1e-2", "10m-uniform-2-1e-3"]);
        assert_eq!(names("profile=1?m", &all), ["10m"]);
        assert_eq!(names("profile=[1]0m", &all), ["10m"]);
        assert_eq!(names("selectivity=1e-3..1e-2", &all), ["10m-uniform-2-1e-3"]);
        assert_eq!(names("partition=true", &all), ["label-3"]);
        assert_eq!(names("family=uniform", &all), ["10m-uniform-2-1e-2", "10m-uniform-2-1e-3"]);
        // Quoted: a literal that would otherwise be a glob.
        let quoted = facts("odd", &[("form", s("a*b"))]);
        assert!(sel("form='a*b'").matches(&quoted));
        assert!(!sel("form='a?b'").matches(&quoted));
    }

    /// Case 5 and 5a: numbers compare numerically under the count rule,
    /// and case folds on keys, values, and suffixes.
    #[test]
    fn numbers_compare_under_the_count_rule() {
        let all = tessera();
        assert_eq!(names("selectivity=0.001", &all), ["10m-uniform-2-1e-3"]);
        assert_eq!(names("size>=100m", &all), ["default", "100m"]);
        assert_eq!(names("SIZE>=100M", &all), ["default", "100m"]);
        assert_eq!(names("size=10M", &all), ["10m", "10m-uniform-2-1e-2", "10m-uniform-2-1e-3"]);
        assert_eq!(names("base_count<20m", &all), ["10m", "10m-uniform-2-1e-2", "10m-uniform-2-1e-3"]);
        assert_eq!(names("Family=UNIFORM", &all), ["10m-uniform-2-1e-2", "10m-uniform-2-1e-3"]);
        assert_eq!(parse_number("128mi"), Some(134_217_728.0));
        assert_eq!(parse_number("128Mi"), Some(134_217_728.0));
        assert_eq!(parse_number("100k"), Some(100_000.0));
        assert_eq!(parse_number("1e-3"), Some(0.001));
    }

    /// Case 5b: the RE2 subset is normative; a lookaround is refused.
    #[test]
    fn a_lookaround_regex_is_refused() {
        let err = Selector::parse("profile=^(?=1)0m$").unwrap_err();
        assert!(err.message.contains("regular expression"), "{err}");
    }

    /// Case 6 and 7: a comparison against a string is false, not an
    /// error; an absent attribute matches nothing, even under `!=`.
    #[test]
    fn strings_and_absence_are_false_not_errors() {
        let all = tessera();
        assert!(names("family<1", &all).is_empty());
        assert!(names("selectivity!=1", &all).len() == 2, "only profiles that have the key");
        assert_eq!(
            names("not(selectivity=1e-2)", &all).len(),
            all.len() - 1,
            "not() is how to say anything but, undescribed included"
        );
    }

    /// Case 8: a list matches on any element, each atom on its own.
    #[test]
    fn a_list_matches_on_any_element() {
        let all = tessera();
        assert_eq!(names("selectivity_ladder=1e-2", &all), ["10m"]);
        assert_eq!(names("selectivity_ladder>=1e-3,selectivity_ladder<1e-2", &all), ["10m"]);
        assert_eq!(names("selectivity_ladder=1e-3..1e-2", &all), ["10m"]);
        // A map as a whole matches nothing; a dotted key reaches in.
        assert!(names("nested=oracle", &all).is_empty());
        assert_eq!(names("nested.kind=oracle", &all), ["label-3"]);
    }

    /// Case 9, 9a, 23: junctions, repeated keys, bare names as terms.
    #[test]
    fn junctions_compose_and_keys_repeat() {
        let all = tessera();
        assert_eq!(names("size=10m,or(family=uniform,predicates=mixed)", &all), ["10m", "10m-uniform-2-1e-2", "10m-uniform-2-1e-3"]);
        assert_eq!(names("not(size=10m,family=uniform)", &all), ["default", "10m", "100m", "label-3"]);
        assert_eq!(names("and(size=10m,family=uniform)", &all), ["10m-uniform-2-1e-2", "10m-uniform-2-1e-3"]);
        assert_eq!(names("selectivity>=1e-3,selectivity<1e-2", &all), ["10m-uniform-2-1e-3"]);
        assert_eq!(names("or(size=10m,size=100m),family=stratified", &all), ["10m", "100m"]);
        assert!(names("size=10m,size=100m", &all).is_empty(), "two equalities on a scalar under AND");
        assert_eq!(names("or(10m,100m)", &all), ["10m", "100m"]);
        assert_eq!(names("or(profile=10m*,profile=^100.*$)", &all).len(), 4);
        assert_eq!(names("size=10m, family = uniform", &all).len(), 2, "whitespace is ignored");
    }

    /// Case 10: structural keys select, and shadow attributes.
    #[test]
    fn structural_keys_select_before_attributes() {
        let all = tessera();
        assert_eq!(names("inherits=default", &all), ["10m"]);
        assert_eq!(names("base_count=10000000", &all).len(), 3);
        let shadowed = ProfileFacts {
            base_count: Some(5),
            ..facts("x", &[("base_count", n(99.0))])
        };
        assert!(sel("base_count=5").matches(&shadowed));
        assert!(!sel("base_count=99").matches(&shadowed));
    }

    /// Case 3 and PS-2: the head is found by shape; a bad selector is a
    /// selector error, never a lookup.
    #[test]
    fn the_head_is_found_by_shape() {
        let spec = DatasetSpec::parse("https://host:8080/ds").unwrap();
        assert_eq!(spec.head, "https://host:8080/ds");
        assert!(spec.selector.is_none());
        let spec = DatasetSpec::parse("https://host/ds:10m").unwrap();
        assert_eq!((spec.head.as_str(), spec.selector.unwrap().bare_name()), ("https://host/ds", Some("10m")));
        let spec = DatasetSpec::parse("tessera:profile=^a:b$").unwrap();
        assert_eq!(spec.head, "tessera");
        assert!(spec.selector.unwrap().matches(&facts("a:b", &[])));
        let spec = DatasetSpec::parse(r"C:\data\ds:10m").unwrap();
        assert_eq!(spec.head, r"C:\data\ds");
        let spec = DatasetSpec::parse("./ds").unwrap();
        assert!(spec.selector.is_none());
        let spec = DatasetSpec::parse("tessera:").unwrap();
        assert!(spec.selector.is_none(), "an empty selector is no selector");
        let err = DatasetSpec::parse("tessera:size=10m,").unwrap_err();
        assert!(err.position >= "tessera:".len(), "{err}");
        let err = DatasetSpec::parse("tessera:size>abc").unwrap_err();
        assert!(err.message.contains("compares numbers"), "{err}");
        let err = DatasetSpec::parse("tessera:size 10m").unwrap_err();
        assert!(err.message.contains("operator"), "{err}");
    }

    #[test]
    fn globs_match_whole_text() {
        assert!(glob_match("10m*", "10m-uniform"));
        assert!(!glob_match("10m", "10m-uniform"));
        assert!(glob_match("*form*", "10m-uniform"));
        assert!(glob_match("1?m", "10m"));
        assert!(glob_match("[a-c]x", "bx"));
        assert!(!glob_match("[!a-c]x", "bx"));
    }

    /// **No selector means `default`, `profile=*` means all, and a
    /// single surface refuses a set** (PS-10).
    #[test]
    fn resolution_defaults_and_refuses_ambiguity() {
        let offer = vec![
            facts("default", &[]),
            facts("10m", &[("size", Yaml::from("10m"))]),
            facts("20m", &[("size", Yaml::from("20m"))]),
        ];
        assert_eq!(resolve(None, &offer).unwrap(), vec!["default"]);
        assert_eq!(resolve_one(None, &offer).unwrap(), "default");
        assert_eq!(resolve(Some("profile=*"), &offer).unwrap(), vec!["default", "10m", "20m"]);
        assert_eq!(resolve_one(Some("10M"), &offer).unwrap(), "10m");
        match resolve_one(Some("or(10m, 20m)"), &offer) {
            Err(SelectionError::Ambiguous { matches, .. }) => {
                assert_eq!(matches, vec!["10m", "20m"]);
            }
            other => panic!("a set on a single surface must be ambiguous: {other:?}"),
        }
        let err = resolve(Some("size=99m"), &offer).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("matches no profile"), "{text}");
        assert!(text.contains("10m (size=10m)"), "the offer lists attributes: {text}");
        assert!(matches!(resolve(Some("size=="), &offer), Err(SelectionError::Syntax(_))));

        let no_default = vec![facts("10m", &[])];
        assert!(matches!(resolve(None, &no_default), Err(SelectionError::NoDefault { .. })));
    }

    /// **Attributes never inherit** (PS-19): a child of a named parent
    /// selects on its own attributes and on the structure it inherits.
    #[test]
    fn attributes_never_inherit_but_structure_does() {
        let yaml = r#"
format_version: 3
profiles:
  default:
    base_vectors: base.fvec
    maxk: 100
  1m:
    inherits: default
    base_count: 1000000
    attributes:
      family: sized
  1m-sel:
    inherits: 1m
    query_vectors: q.fvec
    attributes:
      selectivity: 0.01
"#;
        let config: crate::model::DatasetConfig = serde_yaml::from_str(yaml).unwrap();
        let child = ProfileFacts::of_profile("1m-sel", &config.profiles["1m-sel"]);
        assert_eq!(child.base_count, Some(1_000_000), "structure is read after inheritance (PS-7)");
        assert_eq!(child.maxk, Some(100));
        assert_eq!(
            child.attributes,
            vec![("selectivity".to_string(), Yaml::from(0.01))],
            "a parent's attributes are not the child's (PS-19)"
        );
        assert!(Selector::parse("family=sized").unwrap().matches(
            &ProfileFacts::of_profile("1m", &config.profiles["1m"])
        ));
        assert!(!Selector::parse("family=sized").unwrap().matches(&child));
        assert!(Selector::parse("base_count=1m,selectivity=1e-2").unwrap().matches(&child));

        // The declared form reads the same structure through the chain.
        let group: DSProfileGroup = serde_yaml::from_str(
            &yaml.splitn(3, '\n').nth(2).unwrap().replacen("profiles:\n", "", 1)
                .lines().map(|l| l.strip_prefix("  ").unwrap_or(l)).collect::<Vec<_>>().join("\n"),
        )
        .unwrap();
        let declared = ProfileFacts::of_declared("1m-sel", &group).unwrap();
        assert_eq!(declared.base_count, Some(1_000_000));
        assert_eq!(declared.maxk, Some(100));
        assert_eq!(declared.attributes, child.attributes);
        let sized = ProfileFacts::of_declared("1m", &group).unwrap();
        assert_eq!(sized.base_count, Some(1_000_000));
        assert_eq!(sized.maxk, Some(100), "maxk crosses the size step");
    }
}
