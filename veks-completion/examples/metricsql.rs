// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end example: a CLI with grammar-aware MetricsQL / PromQL
//! tab completion, automatic `--help`, and structured argv parsing
//! — all driven by a single `CommandTree` declaration.
//!
//! This is the integration scenario the built-in
//! `providers::metricsql_provider` was designed for. Read this top
//! to bottom; the rustdoc comments narrate each step.
//!
//! # Try it interactively
//!
//! ```bash
//! cargo build --example metricsql -p veks-completion
//! eval "$(./target/debug/examples/metricsql completions)"
//!
//! # Now type and tab around:
//! ./target/debug/examples/metricsql query 'up{<TAB>'
//! ./target/debug/examples/metricsql query 'rate(http_requests_total[<TAB>'
//! ./target/debug/examples/metricsql query 'sum by (<TAB>'
//! ./target/debug/examples/metricsql --help
//! ./target/debug/examples/metricsql query --help
//! ```
//!
//! # Or run the programmatic demo
//!
//! ```bash
//! ./target/debug/examples/metricsql demo
//! ```
//!
//! Walks through 10 cursor positions inside MetricsQL queries and
//! prints what the completer would suggest at each.

use std::sync::Arc;

use veks_completion::{
    CommandTree, Node,
    handle_complete_env, print_bash_script,
    parse_argv, render_usage,
    PartialParse, complete_at_tap_with_raw,
};
use veks_completion::providers::MetricsqlCatalog;

// ---------------------------------------------------------------------
// Step 1: implement the site-specific data the completer needs.
//
// The built-in MetricsQL grammar is baked into the provider —
// functions, time units, operators, aggregation modifiers. What the
// provider can't know without you is what's actually in your metrics
// store: which metric names exist, which label keys each metric
// carries, which values those labels take.
//
// You implement the `MetricsqlCatalog` trait against whatever data
// source makes sense (an HTTP call to your TSDB, a cached snapshot,
// a static config file). For this example we use a hand-rolled stub
// so it runs without external dependencies.
// ---------------------------------------------------------------------

struct InMemoryCatalog {
    metrics: Vec<&'static str>,
    label_keys: Vec<&'static str>,
    label_values: std::collections::HashMap<&'static str, Vec<&'static str>>,
}

impl InMemoryCatalog {
    fn new() -> Self {
        let mut label_values = std::collections::HashMap::new();
        label_values.insert("job", vec!["prometheus", "node_exporter", "api_gateway"]);
        label_values.insert("instance", vec!["host-1:9100", "host-2:9100", "host-3:9100"]);
        label_values.insert("mode", vec!["idle", "user", "system", "iowait"]);
        label_values.insert("code", vec!["200", "201", "301", "404", "500", "503"]);
        label_values.insert("method", vec!["GET", "POST", "PUT", "DELETE"]);
        InMemoryCatalog {
            metrics: vec![
                "up",
                "node_cpu_seconds_total",
                "node_memory_MemAvailable_bytes",
                "http_requests_total",
                "http_request_duration_seconds_bucket",
            ],
            label_keys: vec!["job", "instance", "mode", "code", "method", "le"],
            label_values,
        }
    }
}

impl MetricsqlCatalog for InMemoryCatalog {
    fn metric_names(&self, prefix: &str) -> Vec<String> {
        self.metrics.iter()
            .filter(|m| m.starts_with(prefix))
            .map(|m| m.to_string())
            .collect()
    }

    fn label_keys(&self, _metric: &str, prefix: &str) -> Vec<String> {
        // A real implementation might filter by metric (the metric
        // name is passed in for that purpose). Here we return the
        // union for simplicity.
        self.label_keys.iter()
            .filter(|k| k.starts_with(prefix))
            .map(|k| k.to_string())
            .collect()
    }

    fn label_values(&self, _metric: &str, label: &str, prefix: &str) -> Vec<String> {
        self.label_values.get(label)
            .map(|vs| vs.iter()
                .filter(|v| v.starts_with(prefix))
                .map(|v| v.to_string())
                .collect())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------
// Step 2: build the command tree.
//
// The CLI exposes:
//   - `query <expr>`  — execute a MetricsQL query
//   - `validate <expr>` — parse without executing
//   - `completions`   — print the bash activation snippet
//
// The two opt-in `CommandTree` builders we use:
//   - `with_auto_help()`         attaches `--help` to every node
//   - `with_metricsql_at(path, catalog)` attaches the MetricsQL
//                                provider at the named subcommand
// ---------------------------------------------------------------------

fn build_tree() -> CommandTree {
    let catalog: Arc<dyn MetricsqlCatalog> = Arc::new(InMemoryCatalog::new());

    CommandTree::new("metricsql")
        // `query` is leaf-shaped at the tree level — the user types
        // a freeform expression as the next word. The MetricsQL
        // provider attached below interprets that expression
        // grammar-aware-ly when the cursor is inside it.
        .command("query",
            Node::leaf(&["--from", "--to", "--step"])
                .with_help("Execute a MetricsQL query against the configured backend.")
                .with_flag_help("--from", "Start of query window (RFC3339 or relative)")
                .with_flag_help("--to",   "End of query window")
                .with_flag_help("--step", "Resolution / step (e.g. '15s', '1m')"))
        .command("validate",
            Node::leaf(&[])
                .with_help("Parse a MetricsQL expression without executing."))
        // Built-in option (1): auto-help. Every node now has
        // `--help` and is recognised by parse_argv.
        .with_auto_help()
        // Built-in option (2): MetricsQL grammar-aware completion at
        // the `query` subtree. Same can be done for `validate` if
        // you want completion there too — left out for brevity.
        .with_metricsql_at(&["query"], catalog)
}

// ---------------------------------------------------------------------
// Step 3: wire the entry point.
//
// Three phases in main():
//   1. `handle_complete_env` — bash sets `_METRICSQL_COMPLETE=bash`
//      and re-invokes the binary on every tab. This call returns
//      true when we're being invoked for completion and we should
//      exit immediately (the engine already printed candidates).
//   2. `completions` subcommand — prints the bash activation snippet
//      so users can `eval "$(metricsql completions)"`.
//   3. Real argv parsing — `parse_argv` walks the same tree and
//      hands back a structured `ParsedCommand`. Check `--help`,
//      branch on the resolved path, dispatch to the right handler.
// ---------------------------------------------------------------------

fn main() {
    let tree = build_tree();

    // (1) Completion callback. When this returns true, bash
    // captured the candidate output; we exit silently.
    if handle_complete_env("metricsql", &tree) {
        return;
    }

    let args: Vec<String> = std::env::args().collect();

    // (2) Activation snippet.
    if args.get(1).map(|s| s.as_str()) == Some("completions") {
        print_bash_script("metricsql");
        return;
    }

    // (Demo mode — for the example. Not part of the real CLI.)
    if args.get(1).map(|s| s.as_str()) == Some("demo") {
        run_demo(&tree);
        return;
    }

    // (3) Structured argv parse. The tree the parser walks is the
    // same tree the completer walks — so a flag added to the tree
    // is automatically a known flag for the parser.
    let argv: Vec<&str> = args.iter().skip(1).map(|s| s.as_str()).collect();
    let parsed = match parse_argv(&tree, &argv) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(2);
        }
    };

    // --help is uniformly available because we called
    // `with_auto_help()`. Resolve the help target by walking the
    // path we parsed and rendering its usage block.
    if parsed.flags.contains_key("--help") {
        let mut node = &tree.root;
        for segment in &parsed.path {
            if let Some(child) = node.child(segment) {
                node = child;
            }
        }
        let mut full_path = vec!["metricsql"];
        full_path.extend(parsed.path.iter().copied());
        println!("{}", render_usage(node, &full_path));
        return;
    }

    // Dispatch on the resolved subcommand path. In a real CLI you'd
    // use `Extras` (see veks-completion docs) to attach a handler
    // to each leaf and dispatch by downcast — keeps the routing
    // decision next to the node it concerns.
    match parsed.path.as_slice() {
        ["query"] => {
            let expr = parsed.positionals.first().unwrap_or(&"");
            let from = parsed.flags.get("--from").and_then(|v| v.first()).map(|s| s.as_str()).unwrap_or("now-1h");
            let to   = parsed.flags.get("--to").and_then(|v| v.first()).map(|s| s.as_str()).unwrap_or("now");
            println!("(would execute) query: '{}' window=[{}..{}]", expr, from, to);
        }
        ["validate"] => {
            let expr = parsed.positionals.first().unwrap_or(&"");
            println!("(would validate) expression: '{}'", expr);
        }
        _ => {
            eprintln!("usage: metricsql <query|validate> [args]");
            eprintln!("Run with --help for details, or `metricsql demo` to see");
            eprintln!("what the MetricsQL completer suggests at sample cursor positions.");
        }
    }
}

// ---------------------------------------------------------------------
// Demo: walk through realistic MetricsQL cursor positions and show
// what the completer suggests at each. This is what a downstream
// integration test could look like — drive `complete_at_tap_with_raw`
// directly with raw line + cursor offset.
// ---------------------------------------------------------------------

fn run_demo(tree: &CommandTree) {
    println!("=== MetricsQL completion demo ===\n");

    // Each scenario: (description, full COMP_LINE, cursor offset).
    // The completer sees what bash would send: the raw line + cursor
    // byte position. The MetricsQL subtree provider attached at
    // `query` interprets the expression token grammar-aware-ly.
    let scenarios: Vec<(&str, &str, usize)> = vec![
        // Top-of-expression: function or metric names.
        ("Top of expression",
            "metricsql query rate(",                  21),
        // Inside `{`: label keys for the preceding metric.
        ("Label key inside {…}",
            "metricsql query up{",                    19),
        // After `key=`: quoted label values.
        ("Quoted values after key=",
            "metricsql query up{job=",                23),
        // Inside open quote: bare label values.
        ("Bare values inside open quote",
            "metricsql query up{job=\"prom",          28),
        // Inside `[…]`: time units.
        ("Time units in range selector",
            "metricsql query rate(http_requests_total[",  41),
        // Mid-typed time unit (digit prefix → strip).
        ("Time units after partial digit",
            "metricsql query rate(http_requests_total[5",  42),
        // After aggregation function: by/without modifiers.
        ("Aggregation modifier",
            "metricsql query sum ",                   20),
        // Inside `by (`: label keys.
        ("Label keys after `by (`",
            "metricsql query sum by (",               24),
        // After `offset`: time units.
        ("Time units after `offset`",
            "metricsql query rate(http_requests_total[5m]) offset ",  53),
        // Realistic complex query, cursor mid-string.
        ("Complex aggregation, mid-value string",
            "metricsql query sum by (instance) (rate(node_cpu_seconds_total{mode=\"i", 70),
    ];

    for (desc, line, cursor) in scenarios {
        // The engine wants the words split (program name + …). For
        // demo purposes we synthesise a minimal split — the
        // subtree provider will see the raw line + cursor anyway.
        let words = split_for_demo(line, cursor);
        let words_ref: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        let cands = complete_at_tap_with_raw(tree, &words_ref, 1, line, cursor);
        println!("─── {desc} ─────────────────────────────");
        println!("input:  {:?}", &line[..cursor]);
        println!("       (cursor here ↑ at byte {cursor})");
        let preview = if cands.len() > 8 {
            let mut head = cands[..8].join(", ");
            head.push_str(&format!(", … (+{} more)", cands.len() - 8));
            head
        } else {
            cands.join(", ")
        };
        println!("cands:  [{preview}]");
        println!();
    }

    // Also demonstrate parse_argv on a complete invocation.
    println!("─── Argv parse (--help) ───────────────────────────");
    let parsed = parse_argv(tree, &["query", "--help"]).unwrap();
    println!("path:        {:?}", parsed.path);
    println!("flags:       {:?}", parsed.flags);
    println!("positionals: {:?}", parsed.positionals);
    println!();

    println!("─── Help block for `query` ────────────────────────");
    let query_node = tree.root.child("query").unwrap();
    println!("{}", render_usage(query_node, &["metricsql", "query"]));

    // And drive PartialParse helpers directly to show what a
    // grammar-aware provider sees at the cursor.
    println!("─── PartialParse internals at one cursor position ─");
    let raw = "metricsql query up{job=\"prom";
    let cursor = raw.len();
    let pp = PartialParse {
        completed: vec!["metricsql", "query"],
        partial: "up{job=\"prom",
        tree_path: vec!["query"],
        raw_line: raw,
        cursor_offset: cursor,
    };
    println!("raw_line:           {:?}", pp.raw_line);
    println!("cursor_offset:      {}", pp.cursor_offset);
    println!("before_cursor():    {:?}", pp.before_cursor());
    println!("ident_before_cursor(): {:?}", pp.ident_before_cursor());
    println!("trigger_char():     {:?}", pp.trigger_char());
    let bs = pp.bracket_state();
    println!("bracket_state(): paren={} brace={} bracket={} inside_quote={:?}",
             bs.paren, bs.brace, bs.bracket, bs.inside_quote);
}

/// Synthesise a `words` split from a raw line + cursor for the
/// demo. The actual `handle_complete_env` does this via `split_line`
/// which honours quotes and escapes; here we use a naive whitespace
/// split since the demo only tests subtree-provider behaviour.
fn split_for_demo(line: &str, cursor: usize) -> Vec<String> {
    let head = &line[..cursor.min(line.len())];
    let mut tokens: Vec<String> = head.split_whitespace().map(|s| s.to_string()).collect();
    // The last token may be a partial — leave the trailing partial
    // empty if the head ends with whitespace.
    if head.ends_with(char::is_whitespace) {
        tokens.push(String::new());
    }
    tokens
}
