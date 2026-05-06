// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Built-in [`SubtreeProvider`]s that downstream embedders can attach
//! to a [`Node`](crate::Node) to enable common context-sensitive
//! completion scenarios without implementing a grammar from scratch.
//!
//! Each provider here is a concrete demonstration of what
//! [`SubtreeProvider`] + [`PartialParse`] are for: a self-contained
//! grammar-aware completer that turns "the cursor sits inside this
//! subtree" into useful, situation-specific candidates.
//!
//! # Currently provided
//!
//! - [`metricsql_provider`] — MetricsQL / PromQL elaboration.
//!   Recognises mixed syntax (label matchers, function calls, range
//!   selectors, operators, aggregation modifiers, time units) and
//!   dispatches to a caller-supplied [`MetricsqlCatalog`] for the
//!   site-specific parts (metric names, label keys, label values).
//!
//! # Conventions for adding more providers
//!
//! - Take a single trait-object catalog (or closure) so the provider
//!   stays parametric over site data without a hard dependency on
//!   any particular metrics / database / grammar implementation.
//! - Use [`PartialParse`] grammar helpers (`bracket_state`,
//!   `trigger_char`, `ident_before_cursor`, `before_cursor`) rather
//!   than re-tokenising the raw line. Those helpers respect quotes
//!   and escapes the same way the rest of the engine does.
//! - Return an empty [`Vec`] when the cursor isn't in a position
//!   the provider knows about, so the caller can fall through to
//!   the engine's default completion (TODO: today the deepest
//!   matching subtree provider takes over completely; future work
//!   could let a provider opt back into the default by returning a
//!   sentinel).

use std::sync::Arc;

use crate::{BracketState, PartialParse, SubtreeProvider};

// =====================================================================
// Free-standing grammar helpers — the inner versions of PartialParse
// methods that we need to call against a SLICE (e.g. the inner
// expression after stripping a shell-wrapper quote), not against the
// raw_line + cursor stored on a real PartialParse.
// =====================================================================

/// Detect the open shell-wrapper quote — the outermost still-open
/// `'…` or `"…` whose opener is preceded by whitespace or
/// start-of-string. Returns `(byte_after_opener, opener_char)`.
///
/// Distinguishes the SHELL wrapper (bash's quoting around the whole
/// expression for shell-safety) from MetricsQL's own string quotes
/// (which appear after `=`, `!=`, `=~`, `!~` inside `{…}`).
fn shell_wrapper_quote(raw_line: &str, cursor: usize) -> Option<(usize, char)> {
    let before = &raw_line[..cursor.min(raw_line.len())];
    let bytes = before.as_bytes();
    let mut wrapper_open: Option<u8> = None;
    let mut wrapper_open_at: Option<usize> = None;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match wrapper_open {
            Some(q) => {
                if c == b'\\' && i + 1 < bytes.len() {
                    i += 2;
                    continue;
                }
                if c == q {
                    wrapper_open = None;
                    wrapper_open_at = None;
                }
            }
            None => {
                if c == b'\'' || c == b'"' {
                    let prev_is_ws_or_start =
                        i == 0 || (bytes[i - 1] as char).is_whitespace();
                    if prev_is_ws_or_start {
                        wrapper_open = Some(c);
                        wrapper_open_at = Some(i + 1);
                    }
                }
            }
        }
        i += 1;
    }
    wrapper_open_at.map(|s| (s, wrapper_open.expect("opener tracked alongside start") as char))
}

fn bracket_state_of(s: &str) -> BracketState {
    let mut state = BracketState::default();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if let Some(q) = state.inside_quote {
            if c == '\\' { chars.next(); continue; }
            if c == q { state.inside_quote = None; }
            continue;
        }
        match c {
            '(' => state.paren += 1,
            ')' => state.paren -= 1,
            '{' => state.brace += 1,
            '}' => state.brace -= 1,
            '[' => state.bracket += 1,
            ']' => state.bracket -= 1,
            '"' | '\'' => state.inside_quote = Some(c),
            _ => {}
        }
    }
    state
}

fn ident_before_cursor_of(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut i = bytes.len();
    while i > 0 {
        let c = bytes[i - 1] as char;
        if is_grammar_ident_char(c) { i -= 1; } else { break; }
    }
    &s[i..]
}

fn trigger_char_of(s: &str) -> Option<char> {
    let mut chars = s.chars().rev();
    while let Some(c) = chars.clone().next() {
        if is_grammar_ident_char(c) { chars.next(); } else { break; }
    }
    chars.next()
}

#[inline]
fn is_grammar_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == ':'
}

/// Label-match operator variants offered after a label key.
/// Order matters for display: `=` first (most common), then
/// `!=`, then regex variants. Tab-completion presents these as
/// suggestion variants per label key so users discover the full
/// operator surface without typing into a backslash-escape thicket.
const LABEL_MATCH_OPERATORS: &[&str] = &["=", "!=", "=~", "!~"];

// =====================================================================
// Multi-candidate / anti-auto-close helpers
//
// See docs/sysref/11a-completion-spec.md §A.2 (readline auto-close
// problem) and §F.0–F.2.bis (the layered defense). Summary:
//
//   1. Grammar-valid expansion (preferred): for each unique grammar
//      match, emit its natural continuation set (e.g. metric →
//      metric, metric{, metric[, metric), metric)' …).
//   2. Ghost-prefix fallback: if step 1 still yields one candidate
//      and the wrapper quote is open, append the user's typed
//      prefix as a no-op alternative.
//   3. Sentinel last resort: if the ghost would dedupe (typed
//      prefix == sole candidate), append a deliberately invalid
//      "<typed>#options" sentinel so bash sees ≥2 entries and
//      doesn't auto-close.
// =====================================================================

/// Expand a single metric-name match into the natural continuation
/// set for tab completion. See §11a F.1.
///
/// Progressive disclosure (the "names view first, then continuations"
/// rule):
///   - Always emits: `metric` (APPEND — bare).
///   - Continuations (`{`, `[`, `)`, wrapper-close TERMINAL) are
///     gated behind one of:
///       a. `ident == metric` — the user has fully typed the
///          metric name; their next decision IS the continuation.
///       b. `is_unique_match` — only one metric matched the
///          typed prefix; the user has effectively pinned it
///          down even mid-typing. Emitting continuations also
///          gives bash a longest-common-prefix that is the full
///          metric name, so single-tab advancement still works.
///       c. `pp.tap_count >= 2` — rapid follow-up tap, "show me
///          everything".
///   - Otherwise (multiple matches with a partial prefix), keep
///     the names-only view so the candidate list isn't
///     N_metrics × 5 entries while the user is still narrowing.
fn expand_metric_continuations(
    pp: &PartialParse,
    target_start: usize,
    metric: &str,
    paren_depth: i32,
    wrapper_quote: Option<char>,
    ident: &str,
    is_unique_match: bool,
) -> Vec<String> {
    let mut variants: Vec<String> = vec![metric.to_string()];
    let show_continuations = ident == metric || is_unique_match || pp.tap_count >= 2;
    if show_continuations {
        variants.push(format!("{metric}{{"));
        variants.push(format!("{metric}["));
        if paren_depth > 0 {
            variants.push(format!("{metric})"));
        }
        if let Some(q) = wrapper_quote {
            if paren_depth > 0 {
                variants.push(format!("{metric}){q}"));
            } else {
                variants.push(format!("{metric}{q}"));
            }
        }
    }
    variants.into_iter()
        .map(|v| pp.splice_candidate(target_start, &v))
        .collect()
}

/// Append the standard "close enclosing scopes / TERMINAL" set
/// of candidates to `out`. Applies uniformly at every "after a
/// just-closed inner expression" position (after `)`, `}`, `]`).
///
/// Emits, when applicable:
///   - `)` — close the enclosing function call (when `bs.paren > 0`)
///   - `,` — next varargs/argument (when `bs.paren > 0`)
///   - `<wrapper>` — TERMINAL when all outer scopes closed and
///                   wrapper open
///   - `)<wrapper>` — close enclosing call + TERMINAL (when
///                    `bs.paren == 1` and wrapper open)
///
/// Only fires when `ident` is empty (the user is at a clean
/// post-expression boundary, not mid-typing some identifier).
///
/// (See §11a §C.14 — post-expression closure suggestions —
/// and the case-3 user report: `'avg(…{…}<TAB>` should offer
/// `)` and `,` since it's inside a varargs call.)
fn append_close_continuations(
    pp: &PartialParse,
    out: &mut Vec<String>,
    ident: &str,
    bs: &BracketState,
    wrapper_quote: Option<char>,
) {
    if !ident.is_empty() {
        return;
    }
    let target_start = pp.cursor_offset;
    if bs.paren > 0 {
        out.push(pp.splice_candidate(target_start, ")"));
        out.push(pp.splice_candidate(target_start, ","));
    }
    if let Some(q) = wrapper_quote {
        if bs.paren == 0 && bs.brace == 0 && bs.bracket == 0 {
            out.push(pp.splice_candidate(target_start, &q.to_string()));
        } else if bs.paren == 1 && bs.brace == 0 && bs.bracket == 0 {
            out.push(pp.splice_candidate(target_start, &format!("){q}")));
        }
    }
}

/// Function-name emission with LCP-advancement guarantee.
/// For each matching function, emit BOTH `func(` (open the
/// call) and `func()` (nullary call form, valid metricsql for
/// some functions). This means a UNIQUE function match yields
/// two candidates whose longest-common-prefix is `func(` — so
/// bash advances the line ALL the way to the function-call
/// entry point on a single tab, not just up to (real - 1) like
/// a typed-prefix or one-char-back ghost would. The user is
/// then ready to type args inside the parens.
///
/// For multiple matching functions, the pair-emission is
/// harmless: bash multi-matches and shows the menu either way.
///
/// See docs/sysref/11a-completion-spec.md §F.1.bis.
fn emit_function_candidates(
    pp: &PartialParse,
    target_start: usize,
    ident: &str,
) -> Vec<String> {
    let mut out = Vec::new();
    for f in METRICSQL_FUNCTIONS.iter().filter(|f| f.starts_with(ident)) {
        out.push(pp.splice_candidate(target_start, &format!("{f}(")));
        out.push(pp.splice_candidate(target_start, &format!("{f}()")));
    }
    out
}

/// Final-pass guarantee: when a shell wrapper quote is open and
/// `out` has fewer than two candidates, append a ghost so bash
/// readline can't fire its single-match auto-close behavior.
/// See §11a F.2 / F.2.bis.
///
/// Three-tier ghost (each tier picked so bash's longest-common-
/// prefix algorithm still advances the line on a single tab):
///
///   1. **Smart ghost** — when the sole real candidate strictly
///      extends the typed text, emit `real[..real.len()-1]`.
///      LCP = real-minus-1, so single tab advances most of the
///      way to the completion. Critical for unique-match cases
///      like `'avg_o` → `'avg_over_time(` where a typed-text
///      ghost would freeze advancement at `'avg_o`.
///   2. **Typed-prefix ghost** (§F.2) — when (1) doesn't apply
///      (no candidate, or candidate doesn't extend typed), emit
///      the typed prefix itself. Suppresses auto-close but
///      doesn't advance.
///   3. **Sentinel** (§F.2.bis) — when both above dedupe, emit
///      `<typed>#options` as a deliberately ugly placeholder.
///
/// No-op when the wrapper quote is closed or `out.len() >= 2`.
fn enforce_multi_candidate(
    pp: &PartialParse,
    out: &mut Vec<String>,
    wrapper_quote: Option<char>,
) {
    if wrapper_quote.is_none() || out.len() >= 2 {
        return;
    }
    let typed = pp.shell_current_word();
    if typed.is_empty() {
        return;
    }
    // Tier 1: smart ghost from the real candidate.
    if let Some(real) = out.first() {
        if real.starts_with(typed) && real.len() > typed.len() {
            // Largest split < real.len() that's > typed.len() and
            // sits on a char boundary.
            let mut split = real.len() - 1;
            while split > typed.len() && !real.is_char_boundary(split) {
                split -= 1;
            }
            if split > typed.len() {
                let ghost = real[..split].to_string();
                if !out.iter().any(|c| c == &ghost) {
                    out.push(ghost);
                    return;
                }
            }
        }
    }
    // Tier 2: typed-prefix ghost.
    if !out.iter().any(|c| c == typed) {
        out.push(typed.to_string());
        return;
    }
    // Tier 3: sentinel.
    let sentinel = format!("{typed}#options");
    if !out.iter().any(|c| c == &sentinel) {
        out.push(sentinel);
    }
}

// =====================================================================
// MetricsQL / PromQL provider
// =====================================================================

/// Site-specific data needed by [`metricsql_provider`].
///
/// Implementations supply the parts of MetricsQL that depend on what's
/// actually in the user's metrics store: which metric names exist,
/// which label keys a metric has, which values a label takes. The
/// built-in parts (functions, operators, time units, aggregation
/// modifiers, comparison operators) are baked in.
///
/// All methods receive a `prefix` and should return only candidates
/// that start with it. Returning the full set and letting the caller
/// filter would also work — the engine's tab cycle calls the
/// provider on every keystroke, so cheaper-when-prefix-known is
/// usually preferable.
pub trait MetricsqlCatalog: Send + Sync + 'static {
    /// Metric names matching the prefix.
    fn metric_names(&self, prefix: &str) -> Vec<String>;

    /// Label keys for the given metric matching the prefix. The
    /// caller passes `metric=""` when the metric name isn't known
    /// (e.g., the user typed `{foo=` standalone, no metric); a
    /// reasonable implementation returns the union across all
    /// metrics or a curated common-keys list.
    fn label_keys(&self, metric: &str, prefix: &str) -> Vec<String>;

    /// Label values for the given (metric, label) matching the
    /// prefix. Same `metric=""` convention as [`label_keys`].
    fn label_values(&self, metric: &str, label: &str, prefix: &str) -> Vec<String>;
}

/// MetricsQL function names the provider recognises out of the box.
/// This list intentionally mirrors VictoriaMetrics' own
/// `metricsql/parser/expr.go` `IsAggrFunc` / `IsRollupFunc` /
/// `IsTransformFunc` etc. tables but is not exhaustive — extend
/// downstream by wrapping your own provider on top.
pub const METRICSQL_FUNCTIONS: &[&str] = &[
    // Rate / range
    "rate", "irate", "increase", "increase_pure", "delta", "idelta",
    "deriv", "deriv_fast", "predict_linear", "holt_winters",
    "changes", "changes_prometheus", "resets",
    "avg_over_time", "min_over_time", "max_over_time", "sum_over_time",
    "count_over_time", "stddev_over_time", "stdvar_over_time",
    "first_over_time", "last_over_time", "quantile_over_time",
    "median_over_time", "mode_over_time", "absent_over_time",
    "present_over_time", "distinct_over_time", "histogram_over_time",
    "rollup", "rollup_rate", "rollup_increase", "rollup_delta",
    "rollup_deriv", "rollup_scrape_interval", "rollup_candlestick",
    "lag", "lifetime", "scrape_interval",
    // Aggregation
    "sum", "min", "max", "avg", "stddev", "stdvar", "count",
    "count_values", "bottomk", "topk", "quantile", "median", "group",
    "limitk", "any", "geomean", "histogram", "outliersk", "mode",
    "zscore", "share", "absent",
    // Transform
    "abs", "absent", "ceil", "clamp_max", "clamp_min", "clamp",
    "exp", "floor", "ln", "log2", "log10", "round", "scalar", "sgn",
    "sort", "sort_desc", "sort_by_label", "sort_by_label_desc",
    "sqrt", "time", "timestamp", "vector", "year", "month", "day_of_month",
    "day_of_week", "days_in_month", "minute", "hour",
    "label_replace", "label_join", "label_set", "label_del",
    "label_keep", "label_lowercase", "label_uppercase", "label_value",
    "label_match", "label_mismatch", "label_copy", "label_move",
    "label_transform", "labels_equal",
    "histogram_quantile", "histogram_share", "histogram_avg",
    "histogram_stddev", "histogram_stdvar", "buckets_limit",
    "prometheus_buckets",
    // Math
    "pi", "rand", "rand_normal", "rand_exponential", "now", "step",
    "start", "end",
];

/// Aggregation modifiers (clauses that follow an aggregation
/// function or precede the bracketed argument list).
pub const METRICSQL_AGGR_MODIFIERS: &[&str] = &[
    "by", "without",
];

/// Vector-matching modifiers that can follow a binary operator
/// between two instant vectors.
pub const METRICSQL_BIN_MATCHING_MODIFIERS: &[&str] = &[
    "on", "ignoring", "group_left", "group_right",
];

/// Logical / set operators between vectors.
pub const METRICSQL_LOGICAL_OPS: &[&str] = &[
    "and", "or", "unless",
];

/// Comparison operators (suffixable with the `bool` modifier).
pub const METRICSQL_COMPARISON_OPS: &[&str] = &[
    "==", "!=", ">", "<", ">=", "<=",
];

/// Arithmetic operators between numeric scalars / vectors.
pub const METRICSQL_ARITH_OPS: &[&str] = &[
    "+", "-", "*", "/", "%", "^", "atan2",
];

/// Boolean / set / arithmetic operators between two instant vectors.
/// Aggregate of [`METRICSQL_LOGICAL_OPS`], [`METRICSQL_COMPARISON_OPS`],
/// [`METRICSQL_ARITH_OPS`], and [`METRICSQL_BIN_MATCHING_MODIFIERS`]
/// for callers that want the full set in one slice.
pub const METRICSQL_BIN_OPS: &[&str] = &[
    "and", "or", "unless", "ignoring", "on", "group_left", "group_right",
    "atan2", "+", "-", "*", "/", "%", "^",
    "==", "!=", ">", "<", ">=", "<=",
];

/// Time-unit suffixes valid inside `[…]` range selectors and
/// `offset` / `@` modifiers.
pub const METRICSQL_TIME_UNITS: &[&str] = &[
    "ms", "s", "m", "h", "d", "w", "y", "i",
];

/// Comparison-operation `bool` modifier, offset / `@` keywords, and
/// MetricsQL extensions like `WITH` template macros and
/// `keep_metric_names`.
pub const METRICSQL_KEYWORDS: &[&str] = &[
    "bool", "offset", "@", "start()", "end()", "default",
    "WITH", "keep_metric_names", "limit",
];

/// Functions whose first argument is a scalar (used for
/// position-dependent argument-typing inside function calls).
const METRICSQL_SCALAR_FIRST_ARG: &[&str] = &[
    "histogram_quantile", "quantile", "quantile_over_time",
    "predict_linear", "holt_winters", "topk", "bottomk",
    "limitk", "outliersk", "round", "clamp", "clamp_min", "clamp_max",
];

/// Build a [`SubtreeProvider`] that completes MetricsQL / PromQL
/// expressions inside the subtree it's attached to.
///
/// Attach to any `Node` whose subtree should be interpreted as a
/// query expression:
///
/// ```
/// use std::sync::Arc;
/// use veks_completion::{CommandTree, Node};
/// use veks_completion::providers::{metricsql_provider, MetricsqlCatalog};
///
/// struct EmptyCatalog;
/// impl MetricsqlCatalog for EmptyCatalog {
///     fn metric_names(&self, _: &str) -> Vec<String> { vec![] }
///     fn label_keys(&self, _: &str, _: &str) -> Vec<String> { vec![] }
///     fn label_values(&self, _: &str, _: &str, _: &str) -> Vec<String> { vec![] }
/// }
///
/// let tree = CommandTree::new("nbrs")
///     .command("query",
///         Node::leaf(&[]).with_subtree_provider(
///             metricsql_provider(Arc::new(EmptyCatalog))
///         ));
/// ```
///
/// What it understands:
///
/// | Cursor context                              | Suggestions                              |
/// |---------------------------------------------|------------------------------------------|
/// | Top of expression / after a binary operator | metric names + function names            |
/// | Inside `{` (label-matcher block)            | label keys for the preceding metric      |
/// | After `key=` or `key=~` inside `{…}`        | label values (in quoted form)            |
/// | Inside `"…"` after `key=`                   | label values (bare; quote is open)       |
/// | Inside `[` (range selector)                 | time units (`5m`, `1h`, etc.)            |
/// | After `sum`/`avg`/`count`… (aggregation)   | `by` / `without`                         |
/// | After `sum by`/`without`                    | label keys (any)                         |
/// | After `offset`                              | time-unit suggestions                    |
///
/// Doesn't understand (yet, contributions welcome):
///
/// - subqueries `[5m:1m]` (treated as range selector — close enough)
/// - `@` modifier value completion (needs timestamp parsing)
/// - regex inside `=~`/`!~` (returns plain values; user fills `.+` etc.)
pub fn metricsql_provider(catalog: Arc<dyn MetricsqlCatalog>) -> SubtreeProvider {
    Arc::new(move |pp: &PartialParse| {
        complete_metricsql(&*catalog, pp)
    })
}

/// MetricsQL-specific diagnostic flags. These are the
/// provider-side counterpart to [`crate::DIAGNOSTIC_FLAGS`] —
/// downstream embedders that attach the [`metricsql_provider`]
/// also call [`metricsql_diagnostic_args`] to expose grammar-level
/// introspection.
pub const METRICSQL_DIAGNOSTIC_FLAGS: &[&str] = &[
    "---metricsql-vocab",     // print built-in MetricsQL vocabulary
    "---metricsql-context",   // <line> <point>: print parser state
];

/// Provider-side diagnostic dispatcher for MetricsQL. Mirrors the
/// engine's [`crate::handle_diagnostic_args`] but covers the flags
/// the metricsql provider knows about.
///
/// Returns `true` if a recognised flag was handled; `false` if no
/// `---metricsql-*` flag appeared on the command line. Embedders
/// who attach the provider should call this in addition to the
/// engine's diagnostic dispatcher:
///
/// ```ignore
/// if veks_completion::handle_complete_env("myapp", &tree) { return; }
/// if veks_completion::handle_diagnostic_args("myapp", &tree) { return; }
/// if veks_completion::providers::metricsql_diagnostic_args("myapp") { return; }
/// // … normal CLI parsing …
/// ```
pub fn metricsql_diagnostic_args(app_name: &str) -> bool {
    let argv: Vec<String> = std::env::args().collect();
    let flag_idx = argv.iter().position(|a| a.starts_with("---metricsql"));
    let Some(idx) = flag_idx else { return false; };
    let flag = argv[idx].as_str();
    let rest: Vec<&str> = argv.iter().skip(idx + 1).map(|s| s.as_str()).collect();
    match flag {
        "---metricsql-vocab" => {
            println!("# functions");
            for f in METRICSQL_FUNCTIONS { println!("{f}"); }
            println!();
            println!("# aggregation modifiers");
            for m in METRICSQL_AGGR_MODIFIERS { println!("{m}"); }
            println!();
            println!("# time units");
            for u in METRICSQL_TIME_UNITS { println!("{u}"); }
            println!();
            println!("# binary operators");
            for op in METRICSQL_BIN_OPS { println!("{op}"); }
            println!();
            println!("# keywords");
            for k in METRICSQL_KEYWORDS { println!("{k}"); }
        }
        "---metricsql-context" => {
            // Same line-prefix convention as the engine's trace
            // diagnostics: user supplies the post-binary line, we
            // prepend the app name so split_line works.
            let user_line = rest.first().copied().unwrap_or("");
            let user_point: usize = rest.get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(user_line.len());
            let line_with_app = format!("{} {}", app_name, user_line);
            let cursor = user_point + app_name.len() + 1;
            let pp = PartialParse {
                completed: Vec::new(),
                partial: "",
                tree_path: Vec::new(),
                raw_line: &line_with_app,
                cursor_offset: cursor,
                tap_count: 1,
            };
            crate::print_partial_parse_for_diagnostics(&pp);
            println!();
            println!("# metricsql-derived");
            let bs = pp.bracket_state();
            let category = if bs.inside_quote.is_some() {
                "label-value (inside open quote)"
            } else if bs.bracket > 0 {
                "time-unit (inside `[`)"
            } else if bs.brace > 0 {
                match pp.trigger_char() {
                    Some('=') | Some('~') => "label-value (after `=`)",
                    _ => "label-key (inside `{`)",
                }
            } else {
                "top-of-expression (metric / function name)"
            };
            println!("category: {category}");
        }
        _ => return false,
    }
    true
}

fn complete_metricsql(
    catalog: &dyn MetricsqlCatalog,
    pp: &PartialParse,
) -> Vec<String> {
    // Capture the wrapper-quote state once so every branch can pass
    // it through to the metric-name expansion + multi-candidate
    // enforcement. Then dispatch to the inner per-context router
    // and finalize the candidate list with the anti-auto-close
    // guarantee. See docs/sysref/11a-completion-spec.md §F.
    let wrapper = shell_wrapper_quote(pp.raw_line, pp.cursor_offset);
    let wrapper_quote: Option<char> = wrapper.map(|(_, q)| q);
    let mut out = complete_metricsql_inner(catalog, pp, wrapper);
    enforce_multi_candidate(pp, &mut out, wrapper_quote);
    out
}

fn complete_metricsql_inner(
    catalog: &dyn MetricsqlCatalog,
    pp: &PartialParse,
    wrapper: Option<(usize, char)>,
) -> Vec<String> {
    // Cursor sits past a closed wrapper-quote pair? The user's
    // metricsql expression is finished — emit nothing rather
    // than splice metric names after the closed wrapper. (See
    // §11a §H, case-5 destructive append.)
    if wrapper.is_none() && cursor_past_closed_wrapper(pp.raw_line, pp.cursor_offset) {
        return Vec::new();
    }

    // Shell-wrapper handling. When the user wraps the expression in
    // single (or double) quotes for shell safety —
    //   metricsql query '{job="prom"…
    // — the leading quote belongs to the SHELL, not to MetricsQL.
    // The default `bracket_state` would see `inside_quote=Some(\')`
    // and treat the entire expression as one big string literal.
    // Detect the wrapper, slice past it, and analyse only the
    // MetricsQL portion. All grammar offsets we then compute are
    // converted back to raw_line coordinates by adding `inner_start`
    // before passing to `splice_candidate`.
    let inner_start = wrapper.map(|(s, _)| s).unwrap_or(0);
    let wrapper_quote: Option<char> = wrapper.map(|(_, q)| q);
    let before = &pp.raw_line[inner_start..pp.cursor_offset.min(pp.raw_line.len())];

    let bs = bracket_state_of(before);
    let ident = ident_before_cursor_of(before);
    let inner_cursor = pp.cursor_offset - inner_start;
    // `to_raw` converts an offset that's relative to the inner
    // expression slice into a raw_line offset, suitable for handing
    // to `splice_candidate`. When there's no shell wrapper, this is
    // the identity.
    let to_raw = |inner: usize| inner_start + inner;

    // (1) Inside an unclosed quote → label-value completion.
    // Each value is offered as the bare string (cursor stays
    // inside the open `"…`) plus a `value"` variant (CLOSE the
    // string, leave cursor still inside `{…}`). When inside the
    // wrapper, also emit `value"}` (CLOSE string + matcher) and
    // — if the expression would then be syntactically complete —
    // `value"}<wrapper>` as TERMINAL. See §11a C.6.
    if bs.inside_quote.is_some() {
        if let Some(LabelValueContext { metric, label }) = label_value_context(before) {
            let prefix = current_quoted_prefix(before);
            let target_start = pp.cursor_offset.saturating_sub(prefix.len());
            let mut out = Vec::new();
            for v in catalog.label_values(metric, label, prefix) {
                out.push(pp.splice_candidate(target_start, &v));
                out.push(pp.splice_candidate(target_start, &format!("{v}\"")));
                out.push(pp.splice_candidate(target_start, &format!("{v}\"}}")));
                if let Some(q) = wrapper_quote {
                    // CLOSE inner string + matcher + TERMINAL wrapper.
                    out.push(pp.splice_candidate(target_start, &format!("{v}\"}}{q}")));
                }
            }
            return out;
        }
        return Vec::new();
    }

    // (2) Inside `[…]` → time-unit suggestions, with subquery
    // awareness. A bare range selector `[5m]` and a subquery
    // `[5m:1m]` look the same up to the `:` — the first segment is
    // a range, the second is a step. Both are durations, so the
    // unit candidates are the same set; we only need to recognise
    // the `:` so the partial-ident scan doesn't include the colon.
    if bs.bracket > 0 {
        // The cursor is inside `[…]` — either a range selector
        // (`[5m]`) or a subquery (`[5m:1m]`). The `ident` here may
        // contain digits + a unit + a `:` (because `:` is also a
        // valid identifier char in metric names). For unit-prefix
        // filtering, strip everything up to the most recent `:`
        // (subquery step is independent of the range), then strip
        // leading digits of the duration the user is mid-typing.
        let after_last_open_bracket = bracket_inner_segment(before);
        let is_subquery_step = after_last_open_bracket.contains(':');
        let after_colon = match ident.rsplit_once(':') {
            Some((_, suffix)) => suffix,
            None => ident,
        };
        let unit_prefix: &str = after_colon.trim_start_matches(|c: char| c.is_ascii_digit());
        let target_start = pp.cursor_offset;
        let mut out: Vec<String> = METRICSQL_TIME_UNITS.iter()
            .filter(|u| u.starts_with(unit_prefix))
            .map(|u| pp.splice_candidate(target_start, u))
            .collect();
        // After the range duration, suggest `:` to start a
        // subquery step (only if no `:` exists yet in this bracket).
        if !is_subquery_step && after_last_open_bracket.chars().any(|c| c.is_alphabetic()) {
            out.push(pp.splice_candidate(pp.cursor_offset, ":"));
        }
        return out;
    }

    // (3) Inside `{…}` → label-key+operator OR label-value start
    //     OR between-matchers (after a closed `"value"` pair).
    if bs.brace > 0 {
        // Discriminate between-matchers from fresh label-key
        // position FIRST. Without this guard, a closed value
        // pair like `{job="prometheus"<cursor>` falls into the
        // default arm below and computes a target_start at the
        // start of the brace — splicing a candidate there would
        // delete the existing pair. (See §11a §H, this is the
        // bracket-state-discrimination class of bug.)
        if brace_between_matchers(before) {
            let target_start = pp.cursor_offset;
            let mut out = vec![
                pp.splice_candidate(target_start, ","),
                pp.splice_candidate(target_start, "}"),
            ];
            // Offer multi-step "close all the way out" candidates
            // when nested inside open structures, so a single tab
            // gets the user to the end of their expression.
            // `})` — close matcher + enclosing func call.
            // `}{closer}` — TERMINAL when the brace is the only
            //               outer scope still open.
            // `}){closer}` — close matcher + func + TERMINAL when
            //                the func is the only other open scope.
            if bs.paren > 0 {
                out.push(pp.splice_candidate(target_start, "})"));
            }
            if let Some(q) = wrapper_quote {
                if bs.paren == 0 && bs.bracket == 0 {
                    out.push(pp.splice_candidate(target_start, &format!("}}{q}")));
                } else if bs.paren == 1 && bs.bracket == 0 {
                    out.push(pp.splice_candidate(target_start, &format!("}}){q}")));
                }
            }
            return out;
        }
        let trig = trigger_char_of(before);
        match trig {
            Some('=') | Some('~') => {
                // `key=`, `key!=`, `key=~`, `key!~` — value position.
                // All four operators end with `=` or `~`, so this
                // arm covers them. `=` is a COMP_WORDBREAKS char so
                // target_start = cursor.
                if let Some(LabelValueContext { metric, label }) = label_value_context(before) {
                    let target_start = pp.cursor_offset;
                    return catalog.label_values(metric, label, "")
                        .into_iter()
                        .map(|v| pp.splice_candidate(target_start, &format!("\"{}\"", v)))
                        .collect();
                }
                return Vec::new();
            }
            _ => {
                // Label-key position. The key starts right after
                // the most recent `{` or `,` within the brace
                // group. Each candidate gets every match-operator
                // appended so the user lands at the value-entry
                // position with one tab+enter and discovers the
                // full operator surface (`=`, `!=`, `=~`, `!~`)
                // in the candidate list.
                let metric = metric_name_before_brace(before).unwrap_or("");
                let target_start_inner = label_key_start_in_brace(before, inner_cursor);
                let target_start = to_raw(target_start_inner);
                let mut out = Vec::new();
                for k in catalog.label_keys(metric, ident) {
                    for op in LABEL_MATCH_OPERATORS {
                        out.push(pp.splice_candidate(
                            target_start,
                            &format!("{}{}", k, op),
                        ));
                    }
                }
                return out;
            }
        }
    }

    // (4) Keyword-driven positions: by / without / offset / @ /
    //     bool / on / ignoring / group_left / group_right.
    //
    // Context discrimination (same kind as branch (5)): the
    // `by`/`without`/`on`/etc. branch fires for the modifier
    // *list* — i.e., when the cursor is INSIDE the modifier's
    // open paren. `sum by <cursor>` (no paren yet) means the
    // user is between the keyword and the `(`; offering label
    // keys here would splice them at the wrong position. Require
    // `bs.paren > 0` so we only emit when the user is actually
    // inside the list.
    if let Some(prev_kw) = preceding_keyword(before) {
        if (prev_kw == "by" || prev_kw == "without"
            || prev_kw == "on" || prev_kw == "ignoring"
            || prev_kw == "group_left" || prev_kw == "group_right")
            && bs.paren > 0
        {
            // All clauses take a `(label, label, …)` list of
            // label keys. Same target-start logic.
            let target_start_inner = label_grouping_target_start(before, inner_cursor);
            let target_start = to_raw(target_start_inner);
            return catalog.label_keys("", ident)
                .into_iter()
                .map(|k| pp.splice_candidate(target_start, &k))
                .collect();
        }
        if prev_kw == "offset" {
            let target_start = pp.cursor_offset.saturating_sub(ident.len());
            return METRICSQL_TIME_UNITS.iter()
                .filter(|u| u.starts_with(ident))
                .map(|u| pp.splice_candidate(target_start, u))
                .collect();
        }
        if prev_kw == "@" {
            // `@` modifier value position. MetricsQL accepts a
            // unix timestamp (number) or `start()` / `end()`.
            let target_start = pp.cursor_offset.saturating_sub(ident.len());
            return ["start()", "end()"].iter()
                .filter(|s| s.starts_with(ident))
                .map(|s| pp.splice_candidate(target_start, s))
                .collect();
        }
        if METRICSQL_COMPARISON_OPS.contains(&prev_kw) {
            // After a comparison operator: suggest `bool` modifier
            // (followed by an expression) plus expression starters
            // (so the user can also type a metric / function).
            // Metric names get the §F.1 grammar-valid expansion.
            let target_start = pp.cursor_offset.saturating_sub(ident.len());
            let mut out: Vec<String> = vec![];
            if "bool".starts_with(ident) {
                out.push(pp.splice_candidate(target_start, "bool"));
            }
            out.extend(emit_function_candidates(pp, target_start, ident));
            let metric_matches = catalog.metric_names(ident);
            let is_unique = metric_matches.len() == 1;
            for m in metric_matches {
                out.extend(expand_metric_continuations(
                    pp, target_start, &m, bs.paren, wrapper_quote, ident, is_unique,
                ));
            }
            out.sort();
            out.dedup();
            return out;
        }
    }

    // (5) Various ident-driven positions:
    //     - aggregation function followed by `by`/`without`
    //     - alternate aggregation placement: `sum (expr) by (...)`
    //     - vector-matching modifier after binop: `expr1 + on(label) expr2`
    //     - WITH templates
    //
    // Context discrimination: the SAME preceding ident has TWO
    // meanings depending on whether we're inside that ident's open
    // paren:
    //
    //   `sum <space>` → alternate-placement: offer `by`/`without`
    //   `sum(<cursor>` → arg position: top-of-expression
    //
    // Distinguish via `enclosing_function_call`. When the
    // innermost open call is the agg function itself, fall through
    // to (5e)/(6) so the user gets metric/function suggestions for
    // the agg's first argument.
    if let Some(prev_ident) = preceding_ident(before) {
        if is_aggregation_function(prev_ident) {
            let inside_own_call = bs.paren > 0 && enclosing_function_call(before)
                .map(|(fn_name, _)| fn_name == prev_ident)
                .unwrap_or(false);
            if !inside_own_call {
                let target_start = pp.cursor_offset.saturating_sub(ident.len());
                return METRICSQL_AGGR_MODIFIERS.iter()
                    .filter(|m| m.starts_with(ident))
                    .map(|m| pp.splice_candidate(target_start, m))
                    .collect();
            }
        }
        // `keep_metric_names` modifier after function-call position.
        if prev_ident == "keep_metric_names" {
            // Already typed; offer expression continuations
            // (binop position, see (6) fallback).
        }
    }

    // (5b) After a `)` token: alternate-placement aggregation
    // modifiers + binary operators are valid. `metric_or_call)`
    // → suggest `by`, `without`, binops, `keep_metric_names`.
    if matches!(last_significant_char(before), Some(')')) {
        let target_start = pp.cursor_offset.saturating_sub(ident.len());
        let mut out: Vec<String> = vec![];
        // Aggregation modifiers (alternate placement).
        for m in METRICSQL_AGGR_MODIFIERS.iter().filter(|m| m.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, m));
        }
        // Vector-matching modifiers.
        for m in METRICSQL_BIN_MATCHING_MODIFIERS.iter().filter(|m| m.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, m));
        }
        // Logical / comparison / arithmetic ops.
        for op in METRICSQL_LOGICAL_OPS.iter().filter(|o| o.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, op));
        }
        for op in METRICSQL_COMPARISON_OPS.iter().filter(|o| o.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, op));
        }
        for op in METRICSQL_ARITH_OPS.iter().filter(|o| o.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, op));
        }
        // `keep_metric_names` and `offset` and `@` — common
        // post-expression modifiers.
        for kw in ["keep_metric_names", "offset", "@"].iter()
            .filter(|k| k.starts_with(ident))
        {
            out.push(pp.splice_candidate(target_start, kw));
        }
        append_close_continuations(pp, &mut out, ident, &bs, wrapper_quote);
        out.sort();
        out.dedup();
        return out;
    }

    // (5c) After `}` (closing label-matcher block): same pattern
    // as `)` — binops, modifiers, post-expression keywords.
    if matches!(last_significant_char(before), Some('}')) {
        let target_start = pp.cursor_offset.saturating_sub(ident.len());
        let mut out: Vec<String> = vec![];
        for op in METRICSQL_LOGICAL_OPS.iter().filter(|o| o.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, op));
        }
        for op in METRICSQL_COMPARISON_OPS.iter().filter(|o| o.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, op));
        }
        for op in METRICSQL_ARITH_OPS.iter().filter(|o| o.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, op));
        }
        for kw in ["[", "offset", "@"].iter().filter(|k| k.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, kw));
        }
        append_close_continuations(pp, &mut out, ident, &bs, wrapper_quote);
        out.sort();
        out.dedup();
        return out;
    }

    // (5d) After `]` (closing range-selector): same pattern.
    if matches!(last_significant_char(before), Some(']')) {
        let target_start = pp.cursor_offset.saturating_sub(ident.len());
        let mut out: Vec<String> = vec![];
        for op in METRICSQL_LOGICAL_OPS.iter().filter(|o| o.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, op));
        }
        for op in METRICSQL_COMPARISON_OPS.iter().filter(|o| o.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, op));
        }
        for op in METRICSQL_ARITH_OPS.iter().filter(|o| o.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, op));
        }
        for kw in ["offset", "@"].iter().filter(|k| k.starts_with(ident)) {
            out.push(pp.splice_candidate(target_start, kw));
        }
        append_close_continuations(pp, &mut out, ident, &bs, wrapper_quote);
        out.sort();
        out.dedup();
        return out;
    }

    // (5e) Function-argument-position typing. Inside `(`, look at
    // the enclosing function name and the argument index. For
    // scalar-first-arg functions (histogram_quantile, quantile,
    // topk, …), arg 0 is a number (no completion offered);
    // subsequent args are vector expressions (top-of-expression).
    if bs.paren > 0 {
        if let Some((fn_name, arg_idx)) = enclosing_function_call(before) {
            if METRICSQL_SCALAR_FIRST_ARG.contains(&fn_name) && arg_idx == 0 {
                // Inside the scalar arg — no useful suggestions
                // for an arbitrary number, but don't fall through
                // to top-of-expression either since that'd offer
                // metric names where the user expects a literal.
                return Vec::new();
            }
            // For all other arg positions (and non-typed
            // functions), fall through to top-of-expression.
        }
    }

    // (6) Top-of-expression. Layered:
    //
    //   - When the user has typed a prefix, show ALL matches
    //     (metric names + functions) — they've already filtered by
    //     letter, so showing both surfaces is just helpful.
    //   - When the user has typed nothing yet (empty prefix):
    //       tap 1 → metric names only (the most common starting
    //               point — "what data do I have?")
    //       tap 2+ → metric names + functions (the user wants to
    //               build out an expression starting with a
    //               function — discover the vocabulary).
    //
    // Function suggestions land at `func(` so the cursor sits
    // ready for argument entry — no extra typing needed to enter
    // the call. Metric suggestions go through
    // `expand_metric_continuations` so each unique match is offered
    // alongside its grammar-valid next steps (`{`, `[`, `)`,
    // wrapper-close TERMINAL) — see §11a F.1, the anti-auto-close
    // rule.
    let target_start = pp.cursor_offset.saturating_sub(ident.len());
    let mut metrics: Vec<String> = Vec::new();
    let metric_matches = catalog.metric_names(ident);
    let is_unique = metric_matches.len() == 1;
    for m in metric_matches {
        metrics.extend(expand_metric_continuations(
            pp, target_start, &m, bs.paren, wrapper_quote, ident, is_unique,
        ));
    }
    let functions = emit_function_candidates(pp, target_start, ident);
    let show_functions = !ident.is_empty() || pp.tap_count >= 2;
    let mut out = metrics;
    if show_functions {
        out.extend(functions);
    }
    out.sort();
    out.dedup();
    out
}

// ---- internal grammar helpers ---------------------------------------

struct LabelValueContext<'a> {
    metric: &'a str,
    label: &'a str,
}

/// Walk back from the cursor to find the `metric{label<op>` context,
/// where `<op>` is any of the four match operators `=`, `!=`,
/// `=~`, `!~`. Returns the metric name and the label key being
/// assigned to.
fn label_value_context(before: &str) -> Option<LabelValueContext<'_>> {
    let bytes = before.as_bytes();
    // Find the last operator-end position (byte after `=` or `~`)
    // outside of quoted regions.
    let mut in_quote: Option<u8> = None;
    let mut last_op_end: Option<usize> = None;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match in_quote {
            Some(q) => {
                if c == b'\\' { i += 2; continue; }
                if c == q { in_quote = None; }
            }
            None => match c {
                b'"' | b'\'' => in_quote = Some(c),
                b'=' | b'~' => last_op_end = Some(i + 1),
                _ => {}
            }
        }
        i += 1;
    }
    let op_end = last_op_end?;
    // Walk back over operator characters (`=`, `~`, `!`) to find
    // where the label-key identifier ends. This correctly handles
    // `=`, `!=`, `=~`, `!~`.
    let mut key_end = op_end;
    while key_end > 0 {
        let c = bytes[key_end - 1];
        if c == b'=' || c == b'~' || c == b'!' { key_end -= 1; } else { break; }
    }
    // Walk back over label-key chars.
    let mut key_start = key_end;
    while key_start > 0 && is_label_char(bytes[key_start - 1] as char) {
        key_start -= 1;
    }
    let label = &before[key_start..key_end];
    if label.is_empty() { return None; }
    // Walk back from key_start to find the most recent `{` (start
    // of the matcher block).
    let mut j = key_start;
    while j > 0 {
        let c = bytes[j - 1];
        if c == b'{' { j -= 1; break; }
        j -= 1;
    }
    if j == 0 || bytes[j] != b'{' { return None; }
    // Metric name: identifier ending at the brace position.
    let metric_end = j;
    let mut metric_start = metric_end;
    while metric_start > 0 && is_label_char(bytes[metric_start - 1] as char) {
        metric_start -= 1;
    }
    let metric = &before[metric_start..metric_end];
    Some(LabelValueContext { metric, label })
}

/// Byte offset where the current label key starts, given that
/// the cursor is somewhere inside a `{ … }` matcher block.
/// Walks back from the cursor to the most recent `{` or `,`
/// outside of quotes, then advances past any whitespace.
/// True iff the cursor sits in "past-the-key" position inside
/// a `{…}` block — i.e., the segment from the most recent `{`
/// or `,` up to the cursor contains a match operator (`=`,
/// `!=`, `=~`, `!~`). In that state, emitting label-KEY
/// candidates (whose splice target_start is right after the
/// `{`/`,`) would REPLACE the existing key + operator + value
/// content, destroying user-typed text. Safe candidates here
/// are append-only: `,` (next matcher) and `}` (close).
///
/// Permissive on purpose: even when the segment looks
/// syntactically broken (e.g. `job="prometheus")` with a stray
/// `)`), we still route here because the real failure mode is
/// the destructive splice — emitting append-only candidates
/// can't make a broken line worse, while emitting label-key
/// candidates demonstrably destroys content.
///
/// (See docs/sysref/11a-completion-spec.md §H — context
/// discrimination class of bug.)
fn brace_between_matchers(before: &str) -> bool {
    let bytes = before.as_bytes();
    // Walk back to the most recent `{` or `,` not inside a string.
    let mut i = bytes.len();
    let mut in_str = false;
    while i > 0 {
        let c = bytes[i - 1];
        if in_str {
            if c == b'"' { in_str = false; }
            i -= 1;
            continue;
        }
        match c {
            b'"' => { in_str = true; i -= 1; }
            b',' | b'{' => break,
            _ => i -= 1,
        }
    }
    let segment = before[i..].trim();
    let segment = segment.trim_start_matches(|c| c == '{' || c == ',').trim();
    if segment.is_empty() {
        return false;
    }
    // Require non-empty content AFTER the last operator char
    // (`=` or `~`). This distinguishes:
    //   `job=`        → key + op, value-start position (empty
    //                   after op) — falls through to trigger-
    //                   arm `Some('=')` for value emission
    //   `job="x"`     → complete value, between-matchers
    //   `job="x")`    → broken (stray `)`) but still past the
    //                   key+op — between-matchers (safe append)
    //   `job!=`       → op-end, value-start (empty after op)
    let last_op = segment.rfind(|c: char| c == '=' || c == '~');
    match last_op {
        Some(pos) => !segment[pos + 1..].trim().is_empty(),
        None => false,
    }
}

/// True iff the cursor is positioned AFTER a closed shell-
/// wrapper quote pair — the user's expression is done. The
/// engine should emit nothing rather than offer candidates
/// that would splice gibberish past the closing wrapper. (See
/// the case-5 bug in the §C spec: cursor at end of
/// `'expr')'` was emitting metric-name candidates that landed
/// AFTER the closing `'`, producing strings like
/// `'expr')'http_requests_total`.)
///
/// Detection: walk the line looking for any closed wrapper-
/// quote pair (whose opener was preceded by whitespace or
/// start-of-string) whose closer sits at-or-before the cursor.
fn cursor_past_closed_wrapper(raw_line: &str, cursor: usize) -> bool {
    let before = &raw_line[..cursor.min(raw_line.len())];
    let bytes = before.as_bytes();
    let mut wrapper_open: Option<u8> = None;
    let mut had_close = false;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match wrapper_open {
            Some(q) => {
                if c == b'\\' && i + 1 < bytes.len() { i += 2; continue; }
                if c == q { wrapper_open = None; had_close = true; }
            }
            None => {
                if c == b'\'' || c == b'"' {
                    let prev_is_ws_or_start =
                        i == 0 || (bytes[i - 1] as char).is_whitespace();
                    if prev_is_ws_or_start {
                        wrapper_open = Some(c);
                    }
                }
            }
        }
        i += 1;
    }
    // Past-closed: a wrapper closed at or before the cursor AND
    // there's no still-open wrapper.
    had_close && wrapper_open.is_none()
}

fn label_key_start_in_brace(before: &str, cursor: usize) -> usize {
    let bytes = before.as_bytes();
    let mut in_quote: Option<u8> = None;
    let mut last_split: Option<usize> = None;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match in_quote {
            Some(q) => {
                if c == b'\\' { i += 2; continue; }
                if c == q { in_quote = None; }
            }
            None => match c {
                b'"' | b'\'' => in_quote = Some(c),
                b'{' | b',' => last_split = Some(i),
                _ => {}
            }
        }
        i += 1;
    }
    let mut p = last_split.map(|x| x + 1).unwrap_or(0);
    // Skip whitespace after the split.
    while p < bytes.len() && (bytes[p] as char).is_whitespace() {
        p += 1;
    }
    // Clamp to cursor (defensive — shouldn't exceed it).
    p.min(cursor)
}

/// Byte offset where the current label key starts inside a
/// `by ( … )` / `without ( … )` clause. Walks back to the most
/// recent `(` or `,`, then past whitespace.
fn label_grouping_target_start(before: &str, cursor: usize) -> usize {
    let bytes = before.as_bytes();
    let mut last_split: Option<usize> = None;
    for (i, &c) in bytes.iter().enumerate() {
        if c == b'(' || c == b',' { last_split = Some(i); }
    }
    let mut p = last_split.map(|x| x + 1).unwrap_or(0);
    while p < bytes.len() && (bytes[p] as char).is_whitespace() {
        p += 1;
    }
    p.min(cursor)
}

/// Slice of `before` after the most recent unclosed `[` — i.e.
/// everything inside the current range-selector bracket. Used to
/// detect subquery position (`[5m:` here).
fn bracket_inner_segment(before: &str) -> &str {
    let bytes = before.as_bytes();
    let mut depth: i32 = 0;
    let mut last_open: Option<usize> = None;
    for (i, &c) in bytes.iter().enumerate() {
        match c {
            b'[' => { depth += 1; if depth == 1 { last_open = Some(i); } }
            b']' => { depth -= 1; }
            _ => {}
        }
    }
    match last_open {
        Some(p) if depth > 0 => &before[p + 1..],
        _ => "",
    }
}

/// Last non-whitespace, non-identifier character in `before`.
/// Used for "what closes the previous token?" decisions: was it
/// `)`, `}`, `]`, an operator char, etc.
fn last_significant_char(before: &str) -> Option<char> {
    let bytes = before.as_bytes();
    let mut i = bytes.len();
    while i > 0 {
        let c = bytes[i - 1] as char;
        if c.is_whitespace() || is_grammar_ident_char(c) {
            i -= 1;
        } else {
            return Some(c);
        }
    }
    None
}

/// If the cursor sits inside a function call `func(arg0, arg1, …)`,
/// return the function name and the zero-indexed argument position.
/// Walks back from cursor to find the enclosing `(` and counts
/// commas at the same paren depth.
fn enclosing_function_call(before: &str) -> Option<(&str, usize)> {
    let bytes = before.as_bytes();
    let mut depth: i32 = 0;
    let mut commas: usize = 0;
    let mut paren_pos: Option<usize> = None;
    let mut in_quote: Option<u8> = None;
    // Walk forward keeping the deepest unclosed `(` and count
    // commas inside it.
    for (i, &c) in bytes.iter().enumerate() {
        match in_quote {
            Some(q) => {
                if c == b'\\' { /* skip */ }
                if c == q { in_quote = None; }
            }
            None => match c {
                b'"' | b'\'' => in_quote = Some(c),
                b'(' => {
                    depth += 1;
                    if depth == 1 {
                        paren_pos = Some(i);
                        commas = 0;
                    }
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 { paren_pos = None; commas = 0; }
                }
                b',' if depth == 1 => commas += 1,
                _ => {}
            }
        }
    }
    let p = paren_pos?;
    if depth == 0 { return None; }
    // Function name = identifier ending at p.
    let mut start = p;
    while start > 0 && is_grammar_ident_char(bytes[start - 1] as char) {
        start -= 1;
    }
    if start == p { return None; }
    Some((&before[start..p], commas))
}

/// Identifier immediately before the `{` matcher block. Returns
/// None if the input doesn't end with `metric{...` (cursor inside).
fn metric_name_before_brace(before: &str) -> Option<&str> {
    let bytes = before.as_bytes();
    let brace = bytes.iter().rposition(|&b| b == b'{')?;
    let metric_end = brace;
    let mut metric_start = metric_end;
    while metric_start > 0 && is_label_char(bytes[metric_start - 1] as char) {
        metric_start -= 1;
    }
    if metric_end == metric_start { return None; }
    Some(&before[metric_start..metric_end])
}

/// Current value being typed inside the open quote. Walks back to
/// the most recent unescaped `"` or `'` (which is the opening
/// quote) and returns everything after it.
fn current_quoted_prefix(before: &str) -> &str {
    let bytes = before.as_bytes();
    let mut i = bytes.len();
    while i > 0 {
        let c = bytes[i - 1];
        if c == b'"' || c == b'\'' {
            // Check escape: if the byte before is `\\`, treat as
            // escaped and continue.
            if i >= 2 && bytes[i - 2] == b'\\' { i -= 1; continue; }
            return &before[i..];
        }
        i -= 1;
    }
    before
}

/// Preceding keyword (e.g. `by`, `without`, `offset`) before the
/// current identifier-or-cursor, separated by whitespace OR by
/// punctuation that doesn't carry semantic weight here (`(`, `)`,
/// `,`). For input `sum by (`, returns `"by"` even though `(`
/// sits between the keyword and the cursor.
fn preceding_keyword(before: &str) -> Option<&str> {
    // Strip the partial identifier under the cursor.
    let trimmed = before.trim_end_matches(|c: char| is_label_char(c));
    // Then strip trailing whitespace + non-semantic punctuation
    // (`(`, `)`, `,`) so we land on the actual previous word.
    let trimmed = trimmed.trim_end_matches(|c: char| {
        c.is_whitespace() || c == '(' || c == ')' || c == ','
    });
    let last_word = trimmed
        .rsplit(|c: char| {
            c.is_whitespace() || c == '(' || c == ')' || c == ','
        })
        .next()?;
    if last_word.is_empty() { return None; }
    Some(last_word)
}

/// Preceding identifier (any name) before the current
/// identifier-or-cursor, separated by whitespace.
fn preceding_ident(before: &str) -> Option<&str> {
    preceding_keyword(before)
}

fn is_aggregation_function(name: &str) -> bool {
    matches!(
        name,
        "sum" | "min" | "max" | "avg" | "stddev" | "stdvar" | "count"
        | "count_values" | "bottomk" | "topk" | "quantile" | "median"
        | "group" | "limitk" | "any" | "geomean" | "histogram"
        | "outliersk" | "mode" | "zscore" | "share"
    )
}

#[inline]
fn is_label_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

// =====================================================================
// Tests — actual MetricsQL examples
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub catalog with a small set of canned metrics, label keys,
    /// and label values — enough to verify the provider's branching.
    struct StubCatalog;

    impl MetricsqlCatalog for StubCatalog {
        fn metric_names(&self, prefix: &str) -> Vec<String> {
            ["up", "node_cpu_seconds_total", "http_requests_total",
             "process_cpu_seconds", "node_memory_MemAvailable_bytes"]
                .iter()
                .filter(|n| n.starts_with(prefix))
                .map(|s| s.to_string())
                .collect()
        }
        fn label_keys(&self, _metric: &str, prefix: &str) -> Vec<String> {
            ["job", "instance", "mode", "code", "method", "le"]
                .iter()
                .filter(|n| n.starts_with(prefix))
                .map(|s| s.to_string())
                .collect()
        }
        fn label_values(&self, _metric: &str, label: &str, prefix: &str) -> Vec<String> {
            let pool: &[&str] = match label {
                "job" => &["prometheus", "node_exporter", "api_gateway"],
                "instance" => &["node-1:9100", "node-2:9100", "node-3:9100"],
                "mode" => &["idle", "user", "system", "iowait", "irq", "softirq"],
                "code" => &["200", "201", "301", "404", "500", "503"],
                "method" => &["GET", "POST", "PUT", "DELETE", "PATCH"],
                "le" => &["0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "+Inf"],
                _ => &[],
            };
            pool.iter()
                .filter(|v| v.starts_with(prefix))
                .map(|s| s.to_string())
                .collect()
        }
    }

    /// Helper: build a PartialParse with the cursor at end of `line`,
    /// tap_count = 1 (single tap, the default UX).
    fn pp_at_end<'a>(line: &'a str) -> PartialParse<'a> {
        pp_at_end_with_tap(line, 1)
    }

    /// Helper: same as [`pp_at_end`] but lets the caller specify
    /// the tap count — used to verify layered behavior.
    fn pp_at_end_with_tap<'a>(line: &'a str, tap_count: u32) -> PartialParse<'a> {
        PartialParse {
            completed: vec![],
            partial: "",
            tree_path: vec![],
            raw_line: line,
            cursor_offset: line.len(),
            tap_count,
        }
    }

    fn run(line: &str) -> Vec<String> {
        let cat = StubCatalog;
        complete_metricsql(&cat, &pp_at_end(line))
    }

    // ---- top-of-expression: metric names + functions ----------------

    #[test]
    fn top_level_with_empty_prefix_tap1_offers_metrics_only() {
        // Tap 1 with no typed prefix: metric names only
        // (the most common starting point).
        let out = run("");
        assert!(out.iter().any(|s| s == "up"));
        // Functions are NOT shown at tap 1 with empty prefix.
        assert!(!out.iter().any(|s| s == "rate("),
            "tap 1 with empty prefix should hide functions: {:?}",
            out.iter().take(5).collect::<Vec<_>>());
    }

    #[test]
    fn top_level_with_empty_prefix_tap2_adds_functions() {
        let line = "";
        let pp = pp_at_end_with_tap(line, 2);
        let cat = StubCatalog;
        let out = complete_metricsql(&cat, &pp);
        assert!(out.iter().any(|s| s == "up"),
            "metrics still present at tap 2");
        assert!(out.iter().any(|s| s == "rate("),
            "functions land with `(` at tap 2: {:?}",
            out.iter().filter(|s| s.starts_with("rate")).collect::<Vec<_>>());
        assert!(out.iter().any(|s| s == "histogram_quantile("));
    }

    #[test]
    fn top_level_with_prefix_shows_both_immediately() {
        // With a typed prefix, both metric and function matches
        // appear on tap 1 — the user has filtered by letter.
        let out = run("hist");
        assert!(out.iter().any(|s| s == "histogram_quantile("));
        assert!(out.iter().any(|s| s == "histogram_over_time("));
        assert!(!out.iter().any(|s| s == "rate("),
            "non-matching candidate should be filtered: {:?}", out);
    }

    #[test]
    fn metric_name_prefix_returns_only_matches() {
        let out = run("node_");
        // `node_cpu_seconds_total` and `node_memory_MemAvailable_bytes`
        // both match.
        assert!(out.iter().any(|s| s == "node_cpu_seconds_total"));
        assert!(out.iter().any(|s| s == "node_memory_MemAvailable_bytes"));
        assert!(!out.iter().any(|s| s == "up"));
    }

    // ---- inside { … }: label-key completion -------------------------
    //
    // The candidates include the bash-preserved prefix so bash's
    // substitution preserves the `up{` (or whatever) the user
    // already typed. `{` is NOT a default COMP_WORDBREAKS char, so
    // bash sees `up{` as the current word and replaces the whole
    // thing with our candidate.

    #[test]
    fn inside_brace_offers_label_keys_with_each_operator() {
        // Label-key candidates now include each match operator
        // (=, !=, =~, !~) appended so the user lands at the value
        // position with one tab and discovers the operator surface.
        let out = run("up{");
        for op in ["=", "!=", "=~", "!~"] {
            for k in ["job", "instance"] {
                let want = format!("up{{{k}{op}");
                assert!(out.iter().any(|s| s == &want),
                    "missing `{want}` in candidates: {:?}", out);
            }
        }
    }

    #[test]
    fn inside_brace_with_partial_filters_keys_to_one() {
        // Prefix `ins` filters down to just `instance`; we still
        // get all four operators per match.
        let out = run("up{ins");
        let want_eq = "up{instance=".to_string();
        let want_ne = "up{instance!=".to_string();
        let want_re = "up{instance=~".to_string();
        let want_nre = "up{instance!~".to_string();
        assert!(out.contains(&want_eq), "missing `=` variant: {:?}", out);
        assert!(out.contains(&want_ne), "missing `!=` variant: {:?}", out);
        assert!(out.contains(&want_re), "missing `=~` variant: {:?}", out);
        assert!(out.contains(&want_nre), "missing `!~` variant: {:?}", out);
        // Other label keys must NOT appear.
        assert!(!out.iter().any(|s| s.contains("job")), "{:?}", out);
    }

    #[test]
    fn inside_brace_after_first_matcher_still_offers_keys() {
        // `up{job="prometheus", ` — cursor right after `, ` should
        // still be in label-key position. Operator-appended now.
        let out = run("up{job=\"prometheus\", ");
        assert!(out.iter().any(|s| s == "instance="));
        assert!(out.iter().any(|s| s == "mode="));
        assert!(out.iter().any(|s| s == "instance=~"));
    }

    // ---- inside { … }: label-value completion -----------------------

    #[test]
    fn after_eq_offers_quoted_label_values() {
        let out = run("up{job=");
        // Engine produces splice-ready candidates that include the
        // bash "current word" prefix (the engine's COMP_WORDBREAKS
        // strips `=` `"` `'` `(` `:` so grammar tokens stay
        // unsplit). Values are quoted because no string is open yet.
        assert!(out.iter().any(|s| s == "up{job=\"prometheus\""));
        assert!(out.iter().any(|s| s == "up{job=\"node_exporter\""));
    }

    #[test]
    fn inside_open_quote_offers_bare_label_values() {
        let out = run("up{job=\"prom");
        // Cursor is inside an open `"…` — return bare values (no
        // surrounding quote, since the open quote is already there).
        // Splice-ready: includes the prefix up to the value position.
        assert!(out.iter().any(|s| s == "up{job=\"prometheus"));
        // Non-matching value is filtered.
        assert!(!out.iter().any(|s| s.ends_with("node_exporter")));
    }

    #[test]
    fn label_value_for_specific_label_only() {
        let out = run("up{mode=");
        // Should return values from the `mode` pool, not `job`.
        assert!(out.iter().any(|s| s == "up{mode=\"idle\""));
        assert!(!out.iter().any(|s| s.ends_with("\"prometheus\"")));
    }

    #[test]
    fn http_request_with_code_label() {
        // `http_requests_total{code=` — values from the `code`
        // pool (HTTP status codes).
        let out = run("http_requests_total{code=");
        assert!(out.iter().any(|s| s == "http_requests_total{code=\"200\""));
        assert!(out.iter().any(|s| s == "http_requests_total{code=\"500\""));
    }

    #[test]
    fn inside_quote_for_method() {
        let out = run("http_requests_total{method=\"P");
        // Splice-ready candidates: full prefix + matching value.
        assert!(out.iter().any(|s| s == "http_requests_total{method=\"POST"));
        assert!(out.iter().any(|s| s == "http_requests_total{method=\"PUT"));
        assert!(out.iter().any(|s| s == "http_requests_total{method=\"PATCH"));
        assert!(!out.iter().any(|s| s.ends_with("GET")));
    }

    // ---- range selector [ … ]: time units ---------------------------

    #[test]
    fn inside_bracket_offers_time_units_with_bash_prefix() {
        // With the engine's "raw mode" COMP_WORDBREAKS (only ` \t\n<>;|&`),
        // `rate(http_requests_total[` is one shell word — candidates
        // splice the entire prefix.
        let out = run("rate(http_requests_total[");
        assert!(out.iter().any(|s| s == "rate(http_requests_total[s"),
            "candidate must include bash-preserved prefix: {:?}", out);
        assert!(out.iter().any(|s| s == "rate(http_requests_total[m"));
        assert!(out.iter().any(|s| s == "rate(http_requests_total[h"));
    }

    #[test]
    fn time_unit_prefix_filter_keeps_digits() {
        // The digit `5` survives in the bash prefix, so the
        // candidate produces e.g. `rate(http_requests_total[5m`.
        let out = run("rate(http_requests_total[5");
        assert!(out.iter().any(|s| s == "rate(http_requests_total[5m"));
        assert!(out.iter().any(|s| s == "rate(http_requests_total[5h"));
    }

    // ---- aggregation modifiers: by / without ------------------------

    #[test]
    fn after_aggregation_offers_modifiers() {
        let out = run("sum ");
        assert!(out.iter().any(|s| s == "by"));
        assert!(out.iter().any(|s| s == "without"));
    }

    #[test]
    fn after_by_offers_label_keys() {
        // Splice-ready: candidate includes the `(` since the only
        // wordbreak in `sum by (` under raw-mode WORDBREAKS is the
        // space after `by`.
        let out = run("sum by (");
        assert!(out.iter().any(|s| s == "(job"));
        assert!(out.iter().any(|s| s == "(instance"));
    }

    // ---- offset modifier: time units --------------------------------

    #[test]
    fn after_offset_offers_time_units() {
        let out = run("rate(http_requests_total[5m]) offset ");
        assert!(out.iter().any(|s| s == "m"));
        assert!(out.iter().any(|s| s == "h"));
    }

    // ---- complex realistic queries ----------------------------------

    #[test]
    fn realistic_complex_query_inside_label_value() {
        // Mid-typing a complex aggregation. Splice prefix runs
        // from the last space-delimited word boundary up to the
        // value position, so the candidate is a long string.
        let line = "sum by (instance) (rate(node_cpu_seconds_total{mode=\"i";
        let out = run(line);
        let prefix = "(rate(node_cpu_seconds_total{mode=\"";
        assert!(out.iter().any(|s| s == &format!("{prefix}idle")));
        assert!(out.iter().any(|s| s == &format!("{prefix}iowait")));
        assert!(out.iter().any(|s| s == &format!("{prefix}irq")));
        assert!(!out.iter().any(|s| s.ends_with("user")),
            "non-matching value should be filtered: {:?}", out);
    }

    #[test]
    fn realistic_complex_query_label_keys_after_comma() {
        // `up{job="prometheus", instance="node-1:9100", ` — cursor
        // ready for the next matcher key. Now operator-appended.
        let out = run("up{job=\"prometheus\", instance=\"node-1:9100\", ");
        assert!(out.iter().any(|s| s == "mode="));
    }

    #[test]
    fn histogram_quantile_outer_function_lands_at_paren() {
        // `histogram_quantile(0.95, …)` — typing the function
        // name and tabbing must put the cursor at the open
        // paren ready for arg entry. The pair-emission
        // (§F.1.bis) gives bash an LCP of `histogram_quantile(`
        // so single-tab advances all the way to the entry point.
        let out = run("histogram_q");
        assert_eq!(out, vec![
            "histogram_quantile(".to_string(),
            "histogram_quantile()".to_string(),
        ]);
    }

    #[test]
    fn label_value_inside_regex_match() {
        // `=~` regex match — provider currently treats it as `=`,
        // returning plain values. The user adds `.+` if they want
        // a regex. (Listed as a known gap in the rustdoc.)
        let out = run("up{job=~\"prom");
        assert!(out.iter().any(|s| s == "up{job=~\"prometheus"));
    }

    #[test]
    fn nested_function_with_metric_name() {
        // `irate(node_…)` — under raw-mode WORDBREAKS the whole
        // `irate(node_` is one shell word, so the candidate splices
        // the function-call prefix in front of the metric name.
        let out = run("irate(node_");
        assert!(out.iter().any(|s| s == "irate(node_cpu_seconds_total"));
    }

    // =================================================================
    // Deeper layered scenarios — exercise the bash-prefix machinery
    // across multi-matcher queries, nested functions, mixed grammar,
    // escaped strings, and end-to-end through the engine entry point.
    // =================================================================

    /// Helper: drive the FULL engine path (not just the metricsql
    /// function) — same path bash invokes via handle_complete_env.
    /// Builds a tree with the metricsql provider attached at the
    /// `query` subcommand and calls `complete_at_tap_with_raw`.
    fn engine_run(line: &str) -> Vec<String> {
        use std::sync::Arc;
        use crate::{CommandTree, Node, complete_at_tap_with_raw};

        let tree = CommandTree::new("metricsql")
            .command("query", Node::leaf(&[]))
            .with_metricsql_at(&["query"], Arc::new(StubCatalog));

        // Synthesise the words split from the raw line — same way
        // the demo does it. The provider sees raw_line + cursor,
        // so the tokenisation only matters for tree-walking (to
        // resolve we're inside the `query` subtree).
        let cursor = line.len();
        let words: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
        let words_ref: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        complete_at_tap_with_raw(&tree, &words_ref, 1, line, cursor)
    }

    /// Substitute a candidate into the bash word for verification.
    /// Returns what the FINAL command line would look like after
    /// the user accepts that candidate. Used to verify candidates
    /// produce the expected user-visible result.
    fn apply_substitution(line: &str, candidate: &str) -> String {
        let cursor = line.len();
        let pp = pp_at_end(line);
        let bws = pp.shell_word_start();
        let mut out = String::with_capacity(line.len() + candidate.len());
        out.push_str(&line[..bws]);
        out.push_str(candidate);
        out.push_str(&line[cursor..]);
        out
    }

    // ---- 2-deep: substitution actually produces correct lines ------

    #[test]
    fn substitution_inside_brace_keeps_metric_prefix_and_adds_operator() {
        let line = "up{";
        let cands = run(line);
        // The `=` variant of `job` should be present and produce
        // `up{job=` after the shell substitutes.
        let job_eq = cands.iter().find(|s| s.as_str() == "up{job=")
            .expect("expected `up{job=` candidate");
        assert_eq!(apply_substitution(line, job_eq), "up{job=",
            "selecting `{job_eq}` from `{line}` should yield `up{{job=`");
    }

    #[test]
    fn substitution_inside_bracket_keeps_metric_prefix() {
        let line = "rate(http_requests_total[";
        let cands = run(line);
        let m_cand = cands.iter().find(|s| s.ends_with("[m")).expect("expected a `[m` candidate");
        assert_eq!(
            apply_substitution(line, m_cand),
            "rate(http_requests_total[m",
            "substitution should preserve the function call + metric + `[`",
        );
    }

    #[test]
    fn substitution_with_partial_unit_digit_keeps_5() {
        let line = "rate(http_requests_total[5";
        let cands = run(line);
        let m_cand = cands.iter().find(|s| s.ends_with("[5m")).expect("expected a `[5m` candidate");
        assert_eq!(
            apply_substitution(line, m_cand),
            "rate(http_requests_total[5m",
            "substitution should keep the `5` digit prefix",
        );
    }

    #[test]
    fn substitution_after_eq_inserts_quoted_value() {
        let line = "up{job=";
        let cands = run(line);
        let prom = cands.iter().find(|s| *s == "up{job=\"prometheus\"")
            .expect("expected splice-ready quoted value");
        assert_eq!(apply_substitution(line, prom), "up{job=\"prometheus\"");
    }

    #[test]
    fn substitution_inside_open_quote_preserves_partial_value() {
        let line = "up{job=\"prom";
        let cands = run(line);
        let prom = cands.iter().find(|s| *s == "up{job=\"prometheus")
            .expect("expected splice-ready bare value");
        assert_eq!(apply_substitution(line, prom), "up{job=\"prometheus");
    }

    // ---- 3-deep: multi-matcher with multiple completions in one ----

    #[test]
    fn multi_matcher_label_value_then_label_key() {
        // First completion: cursor after first `=`.
        let line1 = "up{job=";
        let cands1 = run(line1);
        let pick1 = cands1.iter().find(|s| *s == "up{job=\"prometheus\"").unwrap();
        let after1 = apply_substitution(line1, pick1);
        assert_eq!(after1, "up{job=\"prometheus\"");

        // User types `,` and a space, then tabs for the next key.
        // Under raw-mode WORDBREAKS the shell word starts after
        // the trailing space, so the candidate is bare `instance=`.
        let line2 = format!("{after1}, ");
        let cands2 = run(&line2);
        assert!(cands2.iter().any(|s| s == "instance="),
            "after `, ` should offer operator-appended label keys: {:?}", cands2);

        // Pick `instance=`. Resulting line ready for value.
        let after2 = apply_substitution(&line2, "instance=");
        assert_eq!(after2, "up{job=\"prometheus\", instance=");

        // Tab again — splice-ready value candidates for `instance`.
        // The shell word starts after the space following `,`, so
        // the candidate carries `instance=` + quoted value.
        let cands3 = run(&after2);
        assert!(cands3.iter().any(|s| *s == "instance=\"node-1:9100\""),
            "after `instance=` should offer splice-ready instance values: {:?}", cands3);
    }

    // ---- 3-deep: nested function with completion at every layer ----

    #[test]
    fn nested_function_completions_compose() {
        // Outermost function position. With `histogram_q` typed,
        // the function name lands at `(` so the user is ready for
        // arg entry.
        let line1 = "histogram_q";
        let cands1 = run(line1);
        assert_eq!(cands1, vec![
            "histogram_quantile(".to_string(),
            "histogram_quantile()".to_string(),
        ]);
        assert_eq!(
            apply_substitution(line1, "histogram_quantile("),
            "histogram_quantile(",
        );

        // Inside histogram_quantile: open paren, type metric name.
        // Under raw-mode WORDBREAKS the shell word starts after
        // the space following `0.95,`, so the candidate is the
        // splice-ready `rate(http_requests_total` (function-call
        // prefix preserved up to the metric position).
        let line2 = "histogram_quantile(0.95, rate(http_";
        let cands2 = run(line2);
        let want = "rate(http_requests_total";
        assert!(cands2.iter().any(|s| s == want),
            "expected metric: {:?}", cands2);
        let pick = cands2.iter().find(|s| *s == want).unwrap();
        assert_eq!(
            apply_substitution(line2, pick),
            "histogram_quantile(0.95, rate(http_requests_total",
        );

        // Now inside `[…]` for the range selector.
        let line3 = "histogram_quantile(0.95, rate(http_requests_total[";
        let cands3 = run(line3);
        let pick = cands3.iter()
            .find(|s| s.ends_with("[m"))
            .expect("expected a `[m` candidate");
        assert!(apply_substitution(line3, pick).ends_with("[m"));
    }

    // ---- 3-deep: aggregation + by + nested rate() ------------------

    #[test]
    fn aggregation_with_by_and_inner_rate() {
        // After `sum`, expect `by`/`without`.
        let cands = run("sum ");
        assert!(cands.iter().any(|s| s == "by"));

        // After `sum by (`, expect splice-ready label keys.
        let line = "sum by (";
        let cands = run(line);
        assert!(cands.iter().any(|s| s == "(instance"));
        assert_eq!(apply_substitution(line, "(instance"), "sum by (instance");

        // Build out the full aggregation, completing inside the
        // rate's brace. Under raw-mode WORDBREAKS the candidate
        // wraps the entire word from the space after `(` up to the
        // value position.
        let line = "sum by (instance) (rate(node_cpu_seconds_total{";
        let cands = run(line);
        let pick = cands.iter().find(|s| s.ends_with("{mode=")).expect("expected mode= variant");
        assert_eq!(
            apply_substitution(line, pick),
            "sum by (instance) (rate(node_cpu_seconds_total{mode=",
        );
    }

    // ---- 3-deep: regex matcher inside open quote -------------------

    #[test]
    fn regex_match_inside_open_quote_offers_values() {
        // `=~"prom` — provider currently treats `=~` like `=` and
        // returns splice-ready values inside the open quote.
        let line = "up{job=~\"prom";
        let cands = run(line);
        let want = "up{job=~\"prometheus";
        assert!(cands.iter().any(|s| s == want));
        let pick = cands.iter().find(|s| *s == want).unwrap();
        assert_eq!(apply_substitution(line, pick), "up{job=~\"prometheus");
    }

    // ---- 3-deep: escaped quote inside string -----------------------

    #[test]
    fn escaped_quote_inside_string_doesnt_close_value() {
        // `up{job="abc\"d` — the `\"` is escaped, so we're still
        // inside the open string. Cursor at end. The bracket_state
        // helper honors `\` escapes inside quoted strings.
        let line = "up{job=\"abc\\\"d";
        let pp = pp_at_end(line);
        let bs = pp.bracket_state();
        assert_eq!(bs.inside_quote, Some('"'),
            "escaped \\\" should not close the string: {:?}", bs);
    }

    // ---- 4-deep: end-to-end via the engine entry point -------------

    #[test]
    fn engine_path_returns_shell_correct_candidates_for_label_keys() {
        // Drive the same code path bash invokes. Label-key
        // candidates now arrive with each match operator appended
        // — pick the `=` variant.
        let cands = engine_run("metricsql query up{");
        assert!(cands.iter().any(|s| s == "up{job="),
            "engine path must produce shell-correct candidates: {:?}", cands);
    }

    #[test]
    fn engine_path_substitution_round_trip_inside_brace() {
        let line = "metricsql query up{ins";
        let cands = engine_run(line);
        let pick = cands.iter().find(|s| s.as_str() == "up{instance=")
            .expect("expected up{instance=");
        let pp = pp_at_end(line);
        let sws = pp.shell_word_start();
        let final_line = format!("{}{}", &line[..sws], pick);
        assert_eq!(final_line, "metricsql query up{instance=");
    }

    #[test]
    fn engine_path_round_trip_inside_bracket() {
        let line = "metricsql query rate(http_requests_total[5";
        let cands = engine_run(line);
        let pick = cands.iter().find(|s| s.ends_with("[5m")).expect("expected ...[5m");
        let pp = pp_at_end(line);
        let bws = pp.shell_word_start();
        let final_line = format!("{}{}", &line[..bws], pick);
        assert_eq!(
            final_line,
            "metricsql query rate(http_requests_total[5m",
        );
    }

    // ---- 4-deep: bracket-state robustness with escapes -------------

    #[test]
    fn bracket_state_balances_through_escaped_special_chars() {
        // `up{job="value with } brace inside"` — the `}` is inside
        // a string, so brace depth at end is 1 (only the opening
        // `{`), not 0.
        let line = "up{job=\"value with } brace inside\"";
        let pp = pp_at_end(line);
        let bs = pp.bracket_state();
        assert_eq!(bs.brace, 1, "open brace count should be 1: {:?}", bs);
        assert_eq!(bs.inside_quote, None, "string is closed");
    }

    // =================================================================
    // 5-deep: grammar features enumerated in the rustdoc — proof
    // that the building blocks support everything we claim.
    // =================================================================

    // ---- shell-wrapper quotes (single + double) --------------------

    #[test]
    fn shell_wrapped_single_quote_brace_offers_label_keys() {
        // `metricsql query '{` — bash sees `'` as quoting; the
        // engine recognises `'` as a SHELL wrapper (preceded by
        // whitespace) and analyses the inner expression `{` as
        // MetricsQL — label-key position.
        let cands = engine_run("metricsql query '{");
        assert!(cands.iter().any(|s| s.ends_with("{job=")),
            "shell-wrapped `'{{` should offer label keys: {:?}", cands);
    }

    #[test]
    fn shell_wrapped_double_quote_brace_offers_label_keys() {
        let cands = engine_run("metricsql query \"{");
        assert!(cands.iter().any(|s| s.ends_with("{job=")),
            "shell-wrapped `\"{{` should offer label keys: {:?}", cands);
    }

    #[test]
    fn shell_wrapped_with_metric_offers_label_keys_for_that_metric() {
        let cands = engine_run("metricsql query 'up{");
        assert!(cands.iter().any(|s| s.ends_with("{job=")));
        assert!(cands.iter().any(|s| s.ends_with("{instance=")));
    }

    #[test]
    fn shell_wrapped_top_of_expression_offers_metrics() {
        // `metricsql query '` — empty inside the wrapper. Tap 1
        // top-of-expression: metric names only. Under raw-mode
        // WORDBREAKS the wrapper `'` is no longer a wordbreak, so
        // the splice-ready candidate carries the leading `'`.
        let cands = engine_run("metricsql query '");
        assert!(cands.iter().any(|s| s == "'up"),
            "should offer splice-ready metric names: {:?}",
            cands.iter().take(10).collect::<Vec<_>>());
        // Tap 1 doesn't include functions yet.
        assert!(!cands.iter().any(|s| s.ends_with("rate(")),
            "tap 1 with empty prefix shouldn't show functions: {:?}", cands);
    }

    // ---- all four label-match operators ----------------------------

    #[test]
    fn label_keys_offer_all_four_match_operators() {
        let cands = run("up{");
        for k in ["job", "instance", "mode"] {
            for op in ["=", "!=", "=~", "!~"] {
                let want = format!("up{{{k}{op}");
                assert!(cands.contains(&want), "missing `{want}`: {:?}", cands);
            }
        }
    }

    #[test]
    fn after_neq_offers_label_values() {
        // `up{job!=` — value position via `!=` operator. Under
        // raw-mode WORDBREAKS `=` is no longer a wordbreak, so the
        // splice-ready candidate carries the full prefix.
        let cands = run("up{job!=");
        assert!(cands.iter().any(|s| s == "up{job!=\"prometheus\""),
            "expected splice-ready prefixed candidate: {:?}", cands);
    }

    #[test]
    fn after_regex_neq_offers_label_values() {
        // `up{job!~` — value position via `!~` regex non-match.
        // The candidate is splice-ready: full prefix + quoted value.
        let cands = run("up{job!~");
        assert!(cands.iter().any(|s| s == "up{job!~\"prometheus\""),
            "expected prefixed candidate: {:?}", cands);
    }

    // ---- subquery syntax `[5m:1m]` ---------------------------------

    #[test]
    fn subquery_inside_bracket_after_colon_offers_time_units() {
        // Under raw-mode WORDBREAKS `:` and `(` are no longer
        // wordbreaks, so the splice-ready candidate carries the
        // entire `rate(up[5m:` prefix.
        let cands = run("rate(up[5m:");
        assert!(cands.iter().any(|s| s == "rate(up[5m:m"),
            "subquery step position should offer splice-ready time units: {:?}", cands);
        assert!(cands.iter().any(|s| s == "rate(up[5m:h"));
    }

    #[test]
    fn after_range_unit_offers_subquery_colon() {
        // After the user types `[5m`, suggest `:` as a way to
        // start the subquery step.
        let cands = run("rate(up[5m");
        assert!(cands.iter().any(|s| s.ends_with(":")),
            "should offer `:` to start subquery step: {:?}", cands);
    }

    // ---- @ modifier (start() / end()) ------------------------------

    #[test]
    fn after_at_modifier_offers_start_end() {
        let cands = run("rate(up[5m]) @ ");
        assert!(cands.iter().any(|s| s == "start()"));
        assert!(cands.iter().any(|s| s == "end()"));
    }

    #[test]
    fn after_at_with_partial_filters() {
        let cands = run("rate(up[5m]) @ s");
        assert!(cands.iter().any(|s| s == "start()"));
        assert!(!cands.iter().any(|s| s == "end()"));
    }

    // ---- bool modifier on comparison ops --------------------------

    #[test]
    fn after_comparison_operator_offers_bool_modifier() {
        // `up > ` — comparison operator. Suggest `bool` plus
        // expression starters.
        let cands = run("up > ");
        assert!(cands.iter().any(|s| s == "bool"),
            "after `>` should offer `bool` modifier: {:?}",
            cands.iter().take(10).collect::<Vec<_>>());
    }

    // ---- vector-matching modifiers (on/ignoring/group_*) ----------

    #[test]
    fn after_on_offers_label_keys() {
        // Splice-ready: the candidate carries the `on(` prefix
        // since `(` is no longer a wordbreak under raw-mode.
        let cands = run("a + on(");
        assert!(cands.iter().any(|s| s == "on(job"),
            "after `on(` should offer label keys: {:?}", cands);
    }

    #[test]
    fn after_ignoring_offers_label_keys() {
        let cands = run("a + ignoring(");
        assert!(cands.iter().any(|s| s == "ignoring(job"));
    }

    #[test]
    fn after_group_left_offers_label_keys() {
        let cands = run("a + on(job) group_left(");
        assert!(cands.iter().any(|s| s == "group_left(instance"));
    }

    // ---- alternate aggregation placement: `sum (expr) by (...)` ---

    #[test]
    fn after_close_paren_offers_aggregation_modifiers_and_binops() {
        // `sum(rate(up[5m])) ` — cursor right after the closing
        // paren. Should offer `by`, `without`, plus binary ops,
        // plus `keep_metric_names`, `offset`, `@`.
        let cands = run("sum(rate(up[5m])) ");
        assert!(cands.iter().any(|s| s == "by"),
            "after `)` should offer `by`: {:?}",
            cands.iter().take(15).collect::<Vec<_>>());
        assert!(cands.iter().any(|s| s == "without"));
        assert!(cands.iter().any(|s| s == "+"));
        assert!(cands.iter().any(|s| s == "and"));
        assert!(cands.iter().any(|s| s == "keep_metric_names"));
    }

    // ---- post-`}` and post-`]` binary operator suggestions --------

    #[test]
    fn after_close_brace_offers_binops_and_modifiers() {
        let cands = run("up{job=\"prometheus\"} ");
        assert!(cands.iter().any(|s| s == "+"));
        assert!(cands.iter().any(|s| s == "and"));
        assert!(cands.iter().any(|s| s == "offset"));
        assert!(cands.iter().any(|s| s == "["));
    }

    #[test]
    fn after_close_range_bracket_offers_binops_and_modifiers() {
        let cands = run("rate(up[5m]) ");
        // The cursor lands after the outer `)`, not `]` directly.
        // Test the `]` case by constructing a line that ends with
        // a closing `]` outside any function.
        // For range vectors used directly: `up[5m] `.
        let cands = run("up[5m] ");
        assert!(cands.iter().any(|s| s == "offset"));
        assert!(cands.iter().any(|s| s == "@"));
        let _ = cands;
    }

    // ---- function-arg position typing ------------------------------

    #[test]
    fn histogram_quantile_first_arg_returns_no_metric_names() {
        // First arg of histogram_quantile is a scalar. Don't
        // pollute the suggestion list with metric names that
        // would be syntactically wrong here.
        let cands = run("histogram_quantile(");
        assert!(!cands.iter().any(|s| s.contains("up")),
            "scalar-first-arg position must not offer metric names: {:?}",
            cands.iter().take(10).collect::<Vec<_>>());
    }

    #[test]
    fn histogram_quantile_second_arg_offers_full_expression_set() {
        // Second arg of histogram_quantile is a vector expression.
        // Top-of-expression set: tap 1 = metrics; tap 2 = + functions.
        let line = "histogram_quantile(0.95, ";
        let cat = StubCatalog;
        let pp1 = pp_at_end_with_tap(line, 1);
        let pp2 = pp_at_end_with_tap(line, 2);
        let cands1 = complete_metricsql(&cat, &pp1);
        let cands2 = complete_metricsql(&cat, &pp2);
        assert!(cands1.iter().any(|s| s == "http_requests_total"),
            "tap 1 vector arg should offer metric names: {:?}", cands1);
        assert!(!cands1.iter().any(|s| s == "rate("),
            "tap 1 should not yet offer functions");
        assert!(cands2.iter().any(|s| s == "rate("),
            "tap 2 vector arg should add functions (with `(`): {:?}", cands2);
    }

    #[test]
    fn topk_first_arg_is_scalar() {
        let cands = run("topk(");
        assert!(!cands.iter().any(|s| s.contains("up")),
            "topk first arg should be silent: {:?}", cands);
    }

    // ---- bracket-state robustness w/ shell wrapper ---------------

    #[test]
    fn shell_wrapped_inside_label_value_offers_values() {
        // `metricsql query 'up{job="prom` — shell wrapper is `'`.
        // Inside it, the inner expression has an open `"` for the
        // label value. The provider does NOT confuse the outer
        // wrapper with the inner string quote. Under raw-mode
        // WORDBREAKS the splice-ready candidate carries the entire
        // word starting after the `query ` space.
        let cands = engine_run("metricsql query 'up{job=\"prom");
        assert!(cands.iter().any(|s| s == "'up{job=\"prometheus"),
            "shell-wrapped expression should still complete label values: {:?}", cands);
    }

    // =================================================================
    // 6-deep: scenarios driven by the user's UX walkthrough.
    // Function-name → `(`, two-tier inside func calls, bare-brace
    // label keys with no metric name.
    // =================================================================

    #[test]
    fn user_walkthrough_function_partial_lands_at_open_paren() {
        // `metricsql query 'avg_` cursor at end. Tap should yield
        // `…'avg_over_time(`, not `…'avg_over_time` with trailing
        // space added by the shell.
        let cands = engine_run("metricsql query 'avg_");
        let pick = cands.iter().find(|s| s.ends_with("avg_over_time("))
            .unwrap_or_else(|| panic!("expected avg_over_time(: {:?}", cands));
        let line = "metricsql query 'avg_";
        let pp = pp_at_end(line);
        let sws = pp.shell_word_start();
        let final_line = format!("{}{}", &line[..sws], pick);
        assert_eq!(final_line, "metricsql query 'avg_over_time(",
            "function suggestion must land at `(`, not at the bare name");
    }

    #[test]
    fn user_walkthrough_inside_func_call_tap1_metrics_only() {
        // `metricsql query 'avg_over_time(` cursor inside the
        // function call. Tap 1 = metric family names only.
        let cands = engine_run("metricsql query 'avg_over_time(");
        // At least one metric family name.
        assert!(cands.iter().any(|s| s.ends_with("up")),
            "tap 1 should offer metric names: {:?}",
            cands.iter().take(10).collect::<Vec<_>>());
        // Functions NOT yet shown.
        assert!(!cands.iter().any(|s| s.ends_with("rate(")),
            "tap 1 should not yet offer functions: {:?}",
            cands.iter().filter(|s| s.contains("rate")).collect::<Vec<_>>());
    }

    #[test]
    fn user_walkthrough_inside_func_call_tap2_adds_inner_functions() {
        // Same line, tap 2 — should now include functions for
        // stacking (rate, irate, increase, ...).
        use std::sync::Arc;
        use crate::{CommandTree, Node, complete_at_tap_with_raw};
        let tree = CommandTree::new("metricsql")
            .command("query", Node::leaf(&[]))
            .with_metricsql_at(&["query"], Arc::new(StubCatalog));
        let line = "metricsql query 'avg_over_time(";
        let cursor = line.len();
        let words: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
        let words_ref: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        let cands_t1 = complete_at_tap_with_raw(&tree, &words_ref, 1, line, cursor);
        let cands_t2 = complete_at_tap_with_raw(&tree, &words_ref, 2, line, cursor);
        // Tap 1: metrics only (no functions).
        assert!(!cands_t1.iter().any(|s| s.ends_with("rate(")),
            "tap 1 inside func call should be metrics only");
        // Tap 2: now includes functions (with `(` suffix).
        assert!(cands_t2.iter().any(|s| s.ends_with("rate(")),
            "tap 2 should add inner functions for stacking: {:?}",
            cands_t2.iter().filter(|s| s.contains("rate")).take(5).collect::<Vec<_>>());
        assert!(cands_t2.iter().any(|s| s.ends_with("irate(")),
            "tap 2 should include `irate(`");
        assert!(cands_t2.iter().any(|s| s.ends_with("increase(")),
            "tap 2 should include `increase(`");
    }

    #[test]
    fn user_walkthrough_bare_brace_inside_func_call_offers_label_keys() {
        // `metricsql query 'avg_over_time({` — bare-brace
        // (no metric name preceding the `{`). Should still offer
        // every label key from the catalog. Each key arrives with
        // an operator appended for one-tab value entry.
        let cands = engine_run("metricsql query 'avg_over_time({");
        assert!(cands.iter().any(|s| s.ends_with("{job=")),
            "bare-brace should offer label keys with operators: {:?}",
            cands.iter().filter(|s| s.contains("job")).collect::<Vec<_>>());
        assert!(cands.iter().any(|s| s.ends_with("{instance=")));
        assert!(cands.iter().any(|s| s.ends_with("{job!=")),
            "should also offer the `!=` variant");
    }

    #[test]
    fn user_walkthrough_function_substitution_round_trip() {
        // Verify the function-name-with-paren candidate, when the
        // user accepts it, produces the expected line.
        let line = "metricsql query 'avg_";
        let cands = engine_run(line);
        let pick = cands.iter().find(|s| s.ends_with("avg_over_time(")).unwrap();
        let pp = pp_at_end(line);
        let sws = pp.shell_word_start();
        assert_eq!(
            format!("{}{}", &line[..sws], pick),
            "metricsql query 'avg_over_time(",
        );
    }

    // =================================================================
    // Anti-auto-close (multi-candidate guarantee). See
    // docs/sysref/11a-completion-spec.md §A.2 + §F.
    // =================================================================

    #[test]
    fn full_metric_match_emits_continuations_on_tap1() {
        // Progressive disclosure exception: when the typed ident
        // EXACTLY equals a metric name, the user has finished
        // typing it — emit continuations immediately on tap 1, no
        // rapid follow-up needed. (See §11a F.1 progressive
        // disclosure rule.)
        let cands = engine_run("metricsql query 'abs(http_requests_total");
        assert!(cands.len() >= 2,
            "full match must emit continuations as multi-candidate: {cands:?}");
        let want_bare        = "'abs(http_requests_total";
        let want_open_labels = "'abs(http_requests_total{";
        let want_open_range  = "'abs(http_requests_total[";
        let want_close_func  = "'abs(http_requests_total)";
        let want_terminal    = "'abs(http_requests_total)'";
        for want in [want_bare, want_open_labels, want_open_range,
                     want_close_func, want_terminal] {
            assert!(cands.iter().any(|s| s == want),
                "missing continuation `{want}`: {cands:?}");
        }
    }

    #[test]
    fn partial_metric_match_with_multiple_matches_is_just_names() {
        // Progressive disclosure: tap 1 with a partial prefix
        // matching MULTIPLE metrics shows just the names — keeps
        // the candidate list clean while the user is still
        // narrowing down. (User's request, §11a F.1
        // progressive disclosure rule.) StubCatalog has
        // `node_cpu_seconds_total` and `node_memory_MemAvailable_bytes`,
        // both matching `node_`.
        let cands = engine_run("metricsql query 'node_");
        // Both matching metrics present.
        assert!(cands.iter().any(|s| s == "'node_cpu_seconds_total"));
        assert!(cands.iter().any(|s| s == "'node_memory_MemAvailable_bytes"));
        // No continuation suffixes on tap 1 — the user hasn't
        // narrowed down which metric they want.
        assert!(!cands.iter().any(|s| s == "'node_cpu_seconds_total{"),
            "multi-match tap 1 must NOT yet show `{{` continuation: {cands:?}");
    }

    #[test]
    fn partial_metric_unique_match_emits_continuations_on_tap1() {
        // Refinement of the disclosure rule: when the typed
        // prefix uniquely matches a single metric (even if not
        // fully typed), the user has effectively pinned it down.
        // Emit continuations so bash's longest-common-prefix
        // advances to the full name AND the multi-candidate set
        // suppresses readline auto-close. Without this, the
        // ghost-prefix fallback creates two candidates whose
        // common prefix == the typed text, so bash gets stuck.
        let cands = engine_run("metricsql query 'http_requests_to");
        assert!(cands.iter().any(|s| s == "'http_requests_total"));
        assert!(cands.iter().any(|s| s == "'http_requests_total{"),
            "unique match must emit `{{` so common-prefix advances: {cands:?}");
        assert!(cands.iter().any(|s| s == "'http_requests_total["));
        assert!(cands.iter().any(|s| s == "'http_requests_total'"));
    }

    #[test]
    fn partial_metric_multi_match_tap2_adds_continuations() {
        // Multi-match path: tap 2 should still add continuations
        // for "show me everything" exploration.
        use std::sync::Arc;
        use crate::{CommandTree, Node, complete_at_tap_with_raw};
        let tree = CommandTree::new("metricsql")
            .command("query", Node::leaf(&[]))
            .with_metricsql_at(&["query"], Arc::new(StubCatalog));
        let line = "metricsql query 'node_";
        let cursor = line.len();
        let words: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
        let words_ref: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        let cands_t2 = complete_at_tap_with_raw(&tree, &words_ref, 2, line, cursor);
        assert!(cands_t2.iter().any(|s| s == "'node_cpu_seconds_total{"),
            "tap 2 multi-match must add `{{` continuation: {cands_t2:?}");
        assert!(cands_t2.iter().any(|s| s == "'node_memory_MemAvailable_bytes{"));
    }

    #[test]
    fn unique_function_match_emits_lcp_advancing_pair() {
        // `'delt` matches only `delta(`. The categorical
        // function-emission pair (§11a F.1.bis) emits BOTH
        // `delta(` and `delta()` so bash's longest-common-prefix
        // is `delta(` — single tab advances all the way to the
        // function-call entry point. Without this, even a "smart"
        // ghost only gets the user one char short and forces
        // them to type `(` themselves.
        let line = "metricsql query 'delt";
        let cands = engine_run(line);
        assert!(cands.iter().any(|s| s == "'delta("));
        assert!(cands.iter().any(|s| s == "'delta()"));
        // LCP must include the entry-point `(`.
        let lcp = longest_common_prefix(&cands);
        assert!(lcp.ends_with('('),
            "LCP must reach the function-call entry point: {lcp:?}");
        assert!(lcp.contains("delta("),
            "LCP must include `delta(`: {lcp:?}");
    }

    #[test]
    fn unique_function_match_in_func_arg_also_advances() {
        // Categorical: pair-emission fires for ANY function-emission
        // site. Function inside an enclosing function call follows
        // the same rule, no special-case wiring.
        let line = "metricsql query 'avg(rat";
        let cands = engine_run(line);
        assert!(cands.iter().any(|s| s == "'avg(rate("));
        assert!(cands.iter().any(|s| s == "'avg(rate()"));
        let lcp = longest_common_prefix(&cands);
        assert!(lcp.ends_with("rate("),
            "LCP must reach `rate(` entry point: {lcp:?}");
    }

    fn longest_common_prefix(strs: &[String]) -> String {
        if strs.is_empty() { return String::new(); }
        let first = &strs[0];
        let mut end = first.len();
        for s in &strs[1..] {
            let cap = end.min(s.len());
            let mut i = 0;
            while i < cap && first.as_bytes()[i] == s.as_bytes()[i] { i += 1; }
            end = i;
        }
        while end > 0 && !first.is_char_boundary(end) { end -= 1; }
        first[..end].to_string()
    }

    #[test]
    fn enforce_multi_candidate_skips_when_no_wrapper_open() {
        // Without a shell wrapper, bash readline can't auto-close
        // anything — the §F.2 multi-candidate guarantee is a no-op.
        // Function emission still produces the LCP-advancing pair
        // (`delta(` + `delta()`) — that's a §F.1.bis property
        // independent of the wrapper, so it applies uniformly.
        let cands = run("delt");
        assert_eq!(cands, vec!["delta(".to_string(), "delta()".to_string()],
            "function emission pair fires regardless of wrapper: {cands:?}");
    }

    #[test]
    fn open_string_label_value_offers_close_quote_continuations() {
        // `'up{job="prom` — single matching value `prometheus`.
        // Without the §11a F.4 fix, bash sees single match and
        // auto-closes the wrapper. Now we emit bare value, value+`"`,
        // value+`"}`, value+`"}<wrapper>`.
        let cands = engine_run("metricsql query 'up{job=\"prom");
        assert!(cands.len() >= 2,
            "open-string single match must suppress auto-close: {cands:?}");
        for want in [
            "'up{job=\"prometheus",
            "'up{job=\"prometheus\"",
            "'up{job=\"prometheus\"}",
            "'up{job=\"prometheus\"}'",
        ] {
            assert!(cands.iter().any(|s| s == want),
                "missing close-string continuation `{want}`: {cands:?}");
        }
    }

    #[test]
    fn post_close_paren_offers_wrapper_terminal() {
        // `'sum(rate(up[5m])) ` — expression syntactically
        // complete; wrapper still open. Engine must offer the
        // wrapper-close TERMINAL alongside binops/modifiers
        // (§11a C.14).
        let cands = engine_run("metricsql query 'sum(rate(up[5m])) ");
        assert!(cands.iter().any(|s| s == "'"),
            "post-`)` complete-expression should offer wrapper-close TERMINAL: {cands:?}");
    }

    // ---- context discrimination (§11a §I) -------------------------

    #[test]
    fn aggregation_function_inside_own_call_offers_top_of_expression() {
        // Bug report: `'avg(<TAB>` returned only `by`/`without`
        // (alternate-placement modifiers) instead of metric names.
        // Same lexical content (`avg` as preceding ident) has TWO
        // semantics: alternate-placement when followed by space,
        // function-arg-position when inside its own open paren.
        let cands = engine_run("metricsql query 'avg(");
        assert!(cands.iter().any(|s| s == "'avg(up"),
            "inside agg call must offer metric names, not by/without: {cands:?}");
        assert!(cands.iter().any(|s| s == "'avg(http_requests_total"));
        assert!(!cands.iter().any(|s| s == "'avg(by"),
            "by/without is alternate-placement; not valid inside the call: {cands:?}");
    }

    #[test]
    fn aggregation_function_alternate_placement_still_offers_modifiers() {
        // Counter-test: `'sum <TAB>` (with space, no paren) IS
        // alternate-placement position — `by`/`without` are
        // correct here.
        let cands = engine_run("metricsql query 'sum ");
        assert!(cands.iter().any(|s| s == "by"));
        assert!(cands.iter().any(|s| s == "without"));
    }

    #[test]
    fn modifier_keyword_outside_paren_does_not_emit_label_keys() {
        // Same class of bug: `'sum by <TAB>` (no paren yet) was
        // returning bare label keys, which spliced at the wrong
        // position. Require `bs.paren > 0` to gate the modifier-
        // list label-key branch.
        let cands = engine_run("metricsql query 'sum by ");
        assert!(!cands.iter().any(|s| s == "job"),
            "label keys must not be offered outside the modifier paren: {cands:?}");
        assert!(!cands.iter().any(|s| s == "instance"),
            "label keys must not be offered outside the modifier paren: {cands:?}");
    }

    #[test]
    fn closed_value_pair_inside_brace_does_not_destroy_line() {
        // User report: `'avg(http_requests{job="prometheus"<TAB>`
        // returned label-key candidates whose splice prefix
        // replaced the existing `job="prometheus"` pair —
        // bash's common-prefix completion truncated the line to
        // `'avg(http_requests{`. Discriminator
        // `brace_between_matchers` now recognises the closed
        // pair and emits only safe append-only continuations:
        // `,` (next matcher), `}` (close), `})` (close all the
        // way to the func call), `})<wrapper>` (TERMINAL).
        let cands = engine_run("metricsql query 'avg(http_requests{job=\"prometheus\"");
        // Must NOT contain any candidate that would replace the
        // existing pair (i.e., no candidate that ends with `=` /
        // `!=` / `=~` / `!~` and is shorter than the line).
        for cand in &cands {
            assert!(!cand.ends_with("{job="),
                "destructive splice candidate (truncates value): {cand:?}");
            assert!(!cand.contains("{job=\"") || cand.contains("\"prometheus\""),
                "candidate must preserve the existing closed value: {cand:?}");
        }
        // Must offer the safe append-only continuations.
        let want_comma = "'avg(http_requests{job=\"prometheus\",";
        let want_close = "'avg(http_requests{job=\"prometheus\"}";
        let want_close_all = "'avg(http_requests{job=\"prometheus\"})";
        let want_terminal = "'avg(http_requests{job=\"prometheus\"})'";
        for want in [want_comma, want_close, want_close_all, want_terminal] {
            assert!(cands.iter().any(|s| s == want),
                "missing safe continuation `{want}`: {cands:?}");
        }
    }

    #[test]
    fn syntax_error_inside_brace_does_not_destroy_line() {
        // User report case 4: `'avg(...{job="prometheus")` —
        // the stray `)` before `}` is a syntax error, but the
        // engine must NOT splice candidates that delete the
        // already-typed `job="prometheus")`. The permissive
        // `brace_between_matchers` (any non-empty content after
        // last `=`/`~`) routes this to safe append-only `,`/`}`.
        let cands = engine_run("metricsql query 'avg(http_requests_total{job=\"prometheus\")");
        // No candidate may shorten the existing line.
        let line_word_after_sws = "'avg(http_requests_total{job=\"prometheus\")";
        for cand in &cands {
            assert!(cand.starts_with(line_word_after_sws),
                "candidate must extend, not truncate: {cand:?}");
        }
        // Append-only continuations present.
        assert!(cands.iter().any(|s| s.ends_with(",")),
            "must offer `,` continuation: {cands:?}");
        assert!(cands.iter().any(|s| s.ends_with("}")),
            "must offer `}}` continuation: {cands:?}");
    }

    #[test]
    fn cursor_past_closed_wrapper_emits_nothing() {
        // User report case 5: `'expr'<TAB>` — the wrapper is
        // closed; the user's metricsql expression is done. The
        // engine must emit nothing rather than splice metric
        // names AFTER the closing wrapper. Without this guard
        // we'd get gibberish like `'expr'http_requests_total`.
        let cands = engine_run("metricsql query 'avg(http_requests_total{job=\"prometheus\")'");
        assert!(cands.is_empty(),
            "past-closed-wrapper must yield no candidates: {cands:?}");
    }

    #[test]
    fn complete_inner_inside_func_call_offers_close_paren_and_comma() {
        // User report case 3: `'avg(...{job="prometheus"}<TAB>`
        // — inside the open varargs `avg(...)` call. After the
        // closed `}` the user wants `)` (close func) and `,`
        // (next varg) alongside the binops. Categorical via
        // `append_close_continuations` shared across all
        // post-`)`/`}`/`]` branches.
        let cands = engine_run("metricsql query 'avg(http_requests_total{job=\"prometheus\"}");
        assert!(cands.iter().any(|s| s.ends_with(")")),
            "must offer `)` close-func continuation: {cands:?}");
        assert!(cands.iter().any(|s| s.ends_with(",")),
            "must offer `,` next-varg continuation: {cands:?}");
        // TERMINAL also present (close func + wrapper).
        assert!(cands.iter().any(|s| s.ends_with(")'")),
            "must offer `)<wrapper>` TERMINAL: {cands:?}");
    }

    #[test]
    fn after_comma_inside_brace_still_offers_label_keys() {
        // Counter-test: `'up{job="prometheus", <TAB>` is
        // genuine fresh label-key position (after `,`), not
        // between-matchers. Must offer label keys.
        let cands = engine_run("metricsql query 'up{job=\"prometheus\", ");
        assert!(cands.iter().any(|s| s == "instance="),
            "after `, ` should offer fresh label keys: {cands:?}");
        assert!(cands.iter().any(|s| s == "mode="));
    }

    #[test]
    fn modifier_keyword_inside_paren_offers_label_keys() {
        // Counter-test: `'sum by(<TAB>` is INSIDE the modifier
        // paren — label keys are correct here. Splice prefix
        // is `by(` — bash's current word starts after the space
        // separating `sum` and `by`, and the `'` before `sum` is
        // preserved on the line outside the splice.
        let cands = engine_run("metricsql query 'sum by(");
        assert!(cands.iter().any(|s| s == "by(job"),
            "expected splice-ready label key: {cands:?}");
        assert!(cands.iter().any(|s| s == "by(instance"));
    }

    #[test]
    fn metric_top_of_expression_offers_terminal_when_wrapped() {
        // `'up<TAB>` — `up` is a complete metric expression. With
        // the wrapper open, the user should be able to pick the
        // TERMINAL form `'up'`.
        let cands = engine_run("metricsql query 'up");
        assert!(cands.iter().any(|s| s == "'up'"),
            "expected TERMINAL `'up'` continuation: {cands:?}");
    }

    #[test]
    fn bracket_state_paren_inside_string_doesnt_count() {
        let line = "rate(http_requests_total{label=\"value with (paren\"})";
        let pp = pp_at_end(line);
        let bs = pp.bracket_state();
        // Outer `(` opened at byte 4, closed at end → 0.
        // Inner `(` is inside the string, doesn't count.
        assert_eq!(bs.paren, 0, "paren depth should be 0: {:?}", bs);
        assert_eq!(bs.brace, 0, "brace depth should be 0: {:?}", bs);
    }
}
