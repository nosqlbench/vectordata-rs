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

use crate::{PartialParse, SubtreeProvider};

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

/// Boolean / set / arithmetic operators between two instant vectors.
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

/// Comparison-operation `bool` modifier and offset / `@` keywords.
pub const METRICSQL_KEYWORDS: &[&str] = &[
    "bool", "offset", "@", "start()", "end()", "default",
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

fn complete_metricsql(
    catalog: &dyn MetricsqlCatalog,
    pp: &PartialParse,
) -> Vec<String> {
    let bs = pp.bracket_state();
    let before = pp.before_cursor();
    let ident = pp.ident_before_cursor();

    // (1) Inside an unclosed quote → label-value completion (bare,
    // since the engine's COMP_WORDBREAKS handling will append).
    if bs.inside_quote.is_some() {
        // Parse back to find `key=` or `key=~` … the metric name
        // sits before the `{`.
        if let Some(LabelValueContext { metric, label }) = label_value_context(before) {
            let prefix = current_quoted_prefix(before);
            return catalog.label_values(metric, label, prefix);
        }
        return Vec::new();
    }

    // (2) Inside `[…]` → time-unit suggestions. The user is
    // typically mid-typing a duration like `5m` — the partial
    // ident may be `"5"` (digit prefix), so strip leading digits
    // before filtering.
    if bs.bracket > 0 {
        let unit_prefix: &str = ident.trim_start_matches(|c: char| c.is_ascii_digit());
        return METRICSQL_TIME_UNITS.iter()
            .filter(|u| u.starts_with(unit_prefix))
            .map(|u| u.to_string())
            .collect();
    }

    // (3) Inside `{…}` → label-key completion (when not in a value
    // position) OR label-value start (when right after `=`/`=~`).
    if bs.brace > 0 {
        let trig = pp.trigger_char();
        match trig {
            Some('=') | Some('~') => {
                // `key=` or `key=~` — about to type a value. Offer
                // quoted form (start of value).
                if let Some(LabelValueContext { metric, label }) = label_value_context(before) {
                    return catalog.label_values(metric, label, "")
                        .into_iter()
                        .map(|v| format!("\"{}\"", v))
                        .collect();
                }
                return Vec::new();
            }
            _ => {
                // Label-key position. Resolve the metric (the
                // identifier before the `{`).
                let metric = metric_name_before_brace(before).unwrap_or("");
                return catalog.label_keys(metric, ident);
            }
        }
    }

    // (4) After `by` / `without` keyword → label-key list.
    if let Some(prev_kw) = preceding_keyword(before) {
        if prev_kw == "by" || prev_kw == "without" {
            return catalog.label_keys("", ident);
        }
        if prev_kw == "offset" {
            return METRICSQL_TIME_UNITS.iter()
                .filter(|u| u.starts_with(ident))
                .map(|u| u.to_string())
                .collect();
        }
    }

    // (5) After an aggregation function name → suggest `by`/`without`.
    if let Some(prev_ident) = preceding_ident(before) {
        if is_aggregation_function(prev_ident) {
            // The user just typed the function name, possibly with
            // whitespace; suggest the modifier keywords.
            return METRICSQL_AGGR_MODIFIERS.iter()
                .filter(|m| m.starts_with(ident))
                .map(|m| m.to_string())
                .collect();
        }
    }

    // (6) Top-of-expression position → metric names + functions.
    let mut out: Vec<String> = METRICSQL_FUNCTIONS.iter()
        .filter(|f| f.starts_with(ident))
        .map(|f| f.to_string())
        .collect();
    out.extend(catalog.metric_names(ident));
    // Stable order: functions first (built-in vocabulary), then
    // catalog-supplied metric names, both alphabetical within group.
    out.sort();
    out.dedup();
    out
}

// ---- internal grammar helpers ---------------------------------------

struct LabelValueContext<'a> {
    metric: &'a str,
    label: &'a str,
}

/// Walk back from the cursor to find the `metric{label=` context.
/// Returns the metric name and the label key being assigned to.
fn label_value_context(before: &str) -> Option<LabelValueContext<'_>> {
    // Find the last `=` (possibly followed by `~` for regex match)
    // that's the value separator. Then walk back to find the label
    // key, then walk further back past `{` and any other matchers
    // to find the metric name.
    let bytes = before.as_bytes();
    // Find the last `=` outside of quotes by re-scanning.
    let mut in_quote: Option<u8> = None;
    let mut last_eq: Option<usize> = None;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match in_quote {
            Some(q) => {
                if c == b'\\' { i += 2; continue; }
                if c == q { in_quote = None; }
            }
            None => {
                match c {
                    b'"' | b'\'' => { in_quote = Some(c); }
                    b'=' => { last_eq = Some(i); }
                    _ => {}
                }
            }
        }
        i += 1;
    }
    let eq = last_eq?;
    // Label key: the identifier immediately before `=`.
    let key_end = eq;
    let mut key_start = key_end;
    while key_start > 0 && is_label_char(bytes[key_start - 1] as char) {
        key_start -= 1;
    }
    let label = &before[key_start..key_end];
    if label.is_empty() { return None; }
    // Metric name: walk back from key_start past `{` and possibly
    // earlier matchers separated by `,`.
    let mut j = key_start;
    while j > 0 {
        let c = bytes[j - 1];
        if c == b'{' { j -= 1; break; }
        j -= 1;
    }
    if j == 0 || bytes[j] != b'{' { return None; }
    // Identifier ending at j.
    let metric_end = j;
    let mut metric_start = metric_end;
    while metric_start > 0 && is_label_char(bytes[metric_start - 1] as char) {
        metric_start -= 1;
    }
    let metric = &before[metric_start..metric_end];
    Some(LabelValueContext { metric, label })
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

    /// Helper: build a PartialParse with the cursor at end of `line`.
    fn pp_at_end<'a>(line: &'a str) -> PartialParse<'a> {
        PartialParse {
            completed: vec![],
            partial: "",
            tree_path: vec![],
            raw_line: line,
            cursor_offset: line.len(),
        }
    }

    fn run(line: &str) -> Vec<String> {
        let cat = StubCatalog;
        complete_metricsql(&cat, &pp_at_end(line))
    }

    // ---- top-of-expression: metric names + functions ----------------

    #[test]
    fn top_level_offers_metrics_and_functions() {
        let out = run("");
        assert!(out.iter().any(|s| s == "up"));
        assert!(out.iter().any(|s| s == "rate"));
        assert!(out.iter().any(|s| s == "histogram_quantile"));
    }

    #[test]
    fn top_level_filters_by_prefix() {
        let out = run("hist");
        assert!(out.iter().any(|s| s == "histogram_quantile"));
        assert!(out.iter().any(|s| s == "histogram_over_time"));
        assert!(!out.iter().any(|s| s == "rate"),
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

    #[test]
    fn inside_brace_offers_label_keys() {
        let out = run("up{");
        assert!(out.iter().any(|s| s == "job"));
        assert!(out.iter().any(|s| s == "instance"));
    }

    #[test]
    fn inside_brace_with_partial_filters_keys() {
        let out = run("up{ins");
        assert_eq!(out, vec!["instance".to_string()]);
    }

    #[test]
    fn inside_brace_after_first_matcher_still_offers_keys() {
        // `up{job="prometheus", ` — cursor right after `, ` should
        // still be in label-key position.
        let out = run("up{job=\"prometheus\", ");
        assert!(out.iter().any(|s| s == "instance"));
        assert!(out.iter().any(|s| s == "mode"));
    }

    // ---- inside { … }: label-value completion -----------------------

    #[test]
    fn after_eq_offers_quoted_label_values() {
        let out = run("up{job=");
        // Values returned with surrounding quotes since the user
        // hasn't opened a quote yet.
        assert!(out.iter().any(|s| s == "\"prometheus\""));
        assert!(out.iter().any(|s| s == "\"node_exporter\""));
    }

    #[test]
    fn inside_open_quote_offers_bare_label_values() {
        let out = run("up{job=\"prom");
        // Cursor is inside an open `"…` — return bare values (no
        // surrounding quote, since the open quote is already there).
        assert!(out.iter().any(|s| s == "prometheus"));
        // The prefix filter should exclude non-matching values.
        assert!(!out.iter().any(|s| s == "node_exporter"));
    }

    #[test]
    fn label_value_for_specific_label_only() {
        let out = run("up{mode=");
        // Should return values from the `mode` pool, not `job`.
        assert!(out.iter().any(|s| s == "\"idle\""));
        assert!(!out.iter().any(|s| s == "\"prometheus\""));
    }

    #[test]
    fn http_request_with_code_label() {
        // `http_requests_total{code=` — values from the `code`
        // pool (HTTP status codes).
        let out = run("http_requests_total{code=");
        assert!(out.iter().any(|s| s == "\"200\""));
        assert!(out.iter().any(|s| s == "\"500\""));
    }

    #[test]
    fn inside_quote_for_method() {
        let out = run("http_requests_total{method=\"P");
        // Bare values with `P` prefix.
        assert!(out.iter().any(|s| s == "POST"));
        assert!(out.iter().any(|s| s == "PUT"));
        assert!(out.iter().any(|s| s == "PATCH"));
        assert!(!out.iter().any(|s| s == "GET"));
    }

    // ---- range selector [ … ]: time units ---------------------------

    #[test]
    fn inside_bracket_offers_time_units() {
        let out = run("rate(http_requests_total[");
        assert!(out.iter().any(|s| s == "s"));
        assert!(out.iter().any(|s| s == "m"));
        assert!(out.iter().any(|s| s == "h"));
    }

    #[test]
    fn time_unit_prefix_filter() {
        let out = run("rate(http_requests_total[5");
        // `5` is the digit; the partial ident before cursor is `""`
        // since digits aren't part of label_char. So all units are
        // shown — that's the right behavior, the user filters next.
        assert!(out.iter().any(|s| s == "m"));
        assert!(out.iter().any(|s| s == "h"));
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
        let out = run("sum by (");
        assert!(out.iter().any(|s| s == "job"));
        assert!(out.iter().any(|s| s == "instance"));
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
        // Mid-typing a complex aggregation.
        let out = run("sum by (instance) (rate(node_cpu_seconds_total{mode=\"i");
        // Inside the open quote after `mode=`, completing values
        // for the `mode` label that start with `i`.
        assert!(out.iter().any(|s| s == "idle"));
        assert!(out.iter().any(|s| s == "iowait"));
        assert!(out.iter().any(|s| s == "irq"));
        assert!(!out.iter().any(|s| s == "user"),
            "non-matching value should be filtered: {:?}", out);
    }

    #[test]
    fn realistic_complex_query_label_keys_after_comma() {
        // `up{job="prometheus", instance="node-1:9100", ` — cursor
        // ready for the next matcher key.
        let out = run("up{job=\"prometheus\", instance=\"node-1:9100\", ");
        assert!(out.iter().any(|s| s == "mode"));
    }

    #[test]
    fn histogram_quantile_outer_function() {
        // `histogram_quantile(0.95, rate(latency_bucket[5m]))` —
        // typing this from scratch, top of expression.
        let out = run("histogram_q");
        assert_eq!(out, vec!["histogram_quantile".to_string()]);
    }

    #[test]
    fn label_value_inside_regex_match() {
        // `=~` regex match — provider currently treats it as `=`,
        // returning plain values. The user adds `.+` if they want
        // a regex. (Listed as a known gap in the rustdoc.)
        let out = run("up{job=~\"prom");
        assert!(out.iter().any(|s| s == "prometheus"));
    }

    #[test]
    fn nested_function_with_metric_name() {
        // `irate(node_cpu_seconds_total{mode!="idle"}` — top of an
        // expression that's been typed; cursor would be after the
        // closing brace, but here we test the metric-name completion
        // immediately after the function paren.
        let out = run("irate(node_");
        assert!(out.iter().any(|s| s == "node_cpu_seconds_total"));
    }
}
