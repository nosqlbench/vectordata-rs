# 11. Shell Completions

Dynamic tab-completion (plus argv parsing and help rendering) driven
by a single `CommandTree` declaration. The same tree drives every
CLI surface — completions, `--help`, argv parse — so they can't drift
out of sync.

> **Companion doc**: [§11a Completion Functional Spec](./11a-completion-spec.md)
> covers the per-context behavioral contract — what candidates the
> engine must emit for every cursor position, and how those
> candidates must be shaped so bash readline doesn't auto-close
> wrapper quotes or mis-splice grammar tokens. Read that one when
> implementing or auditing context-specific completion behavior.

---

## 11.1 User Setup

```bash
eval "$(veks completions)"                       # activate now
echo 'eval "$(veks completions)"' >> ~/.bashrc   # persist
```

The activation snippet calls back into the `veks` binary on each tab
press. The binary walks its own `CommandTree` and returns candidates;
no static completion file to keep in sync. The snippet is emitted
using `argv[0]` verbatim (whatever the user invoked), so pasting it
into `~/.bashrc` re-invokes the binary the same way the user just
did.

---

## 11.2 Architecture

```
User presses Tab
  → bash calls: _VEKS_COMPLETE=bash veks "$COMP_LINE" "$COMP_POINT"
    → veks builds CommandTree from clap definitions (dyncomp.rs)
    → veks_completion::complete_rotating_with_raw(tree, words, tap_count,
                                                  raw_line, cursor)
  → bash displays candidates (with `complete -o nosort` so the layer
    ordering the engine returns is preserved)
```

### Two-crate design

**`veks-completion`** — generic library, zero non-`std` dependencies:

- `Node`, `CommandTree` — the tree model. `Node` is a single struct
  (not an enum); a node carries `children` and `flags` orthogonally,
  so a node with both is hybrid.
- `complete_rotating_with_raw`, `complete_at_tap_with_raw`,
  `complete` — the completion entry points.
- `parse_argv`, `parse_argv_lenient` — argv parser companion.
- `render_usage` — `--help` block formatter.
- `ClosedValues`, `Directive`, `apply_directives` — declarative
  flag-set builders that drive completion + validation + help from
  one source.
- `Extras` — free-form payload slot for handlers / dispatch state.
- `SubtreeProvider`, `PartialParse` — context-sensitive completion
  hooks.
- `providers` module — built-in subtree providers (currently
  `metricsql_provider`).
- `next_tap_state`, `TAP_ADVANCE_MS` — pure tap-cadence rule + state
  type; lets embedders script timing scenarios.

**`veks/src/cli/dyncomp.rs`** — veks-specific wiring:

- `build_tree()` walks the clap `Command` tree → `CommandTree`.
- Hidden subcommands (clap `hide(true)`) become `level=2` so they
  surface only on rapid double-tap.
- Registers global value providers for `--dataset`, `--profile`, etc.

The completion crate has no dependency on clap, veks, or any
pipeline code. The clap-to-tree conversion lives in the `veks`
binary crate.

---

## 11.3 Command Tree

```rust
pub struct Node {
    // Identity / display
    category:  Option<String>,
    level:     Option<u32>,
    help:      Option<String>,

    // Subcommand children — empty ⇒ leaf-shaped.
    children: BTreeMap<String, Node>,

    // Flags accepted on this node.
    flags:           Vec<String>,
    boolean_flags:   HashSet<String>,
    flag_help:       BTreeMap<String, String>,
    value_providers: BTreeMap<String, ValueProvider>,

    // Discovery extras
    dynamic_options:  Option<DynamicOptionsProvider>,
    promoted_globals: Vec<(String, ValueProvider)>,
    subtree_provider: Option<SubtreeProvider>,
    extras:           Option<Extras>,
}
```

A node with no children is **leaf-shaped** (`is_leaf() ⇒ true`); a
node with children is **group-shaped**; a node with both is
**hybrid** (e.g., `report` accepts `--workload` *and* has a `base`
subcommand). Walkers branch on `children.is_empty()` only when the
distinction actually matters; otherwise the same code paths apply
uniformly to all three shapes.

---

## 11.4 Multi-Tap Rotating Tiers

Tab tabs through layers cumulatively, with a 200 ms window:

| Tap | When | Result |
|-----|------|--------|
| 1   | cold or after pause       | layer 1 only |
| 2   | within 200 ms             | layers 1 + 2 (cumulative superset) |
| 3   | within 200 ms             | layers 1 + 2 + 3 (cumulative, max) |
|     |                           | persistent state resets at max |
| 4   | within 200 ms             | layer 1 (fresh — state was just reset) |

Within each tap's result, candidates are sorted in **layer order**
(layer 1 candidates first, then layer 2, …) with `--`-flags last and
alphabetical otherwise within each layer. The sort survives the
shell only because the bash script uses `complete -o nosort`.

The cadence rule is exposed as a pure function for embedders that
want their own clock / state:

```rust
let (tap_count, next_state) = next_tap_state(
    Some((prev_state, "current_input_key")),
    now_ms,
    "current_input_key",
    max_level,
);
```

`TAP_ADVANCE_MS` (200 ms) and `TapState` are public.

---

## 11.5 Value Providers

When the user tabs after a `--option`, the engine calls the
registered value provider:

```rust
pub type ValueProvider =
    Arc<dyn Fn(partial: &str, context: &[&str]) -> Vec<String> + Send + Sync>;
```

### Closed value sets — `ClosedValues`

A static or runtime-owned set that produces both completion and
validation from one declaration:

```rust
let metrics = ClosedValues::Static(&["L2", "IP", "COSINE"]);
assert_eq!(metrics.complete("CO"), vec!["COSINE"]);
assert!(metrics.validate("L2"));
node.with_value_provider("--metric", metrics.into_provider());
```

### Per-flag aliases

`Node::with_value_provider_aliases(&["--tofile", "--to-file"], provider)`
registers one provider against every alias in one call.

### Global providers

`CommandTree::global_value_provider(token, provider)` registers
across the whole tree. `Node::with_promoted_global(token, provider)`
stages a global on a specific node and `CommandTree::lift_promoted_globals()`
lifts all staged entries into the tree-level map at finalisation —
lets specs declare globals as part of node construction.

### Dynamic options

`Node::with_dynamic_options(provider)` discovers additional
`key=value` options from context (e.g., reading a referenced file
to extract parameter names that become completable).

---

## 11.6 Argv Parser Companion

`parse_argv(tree, &argv) → ParsedCommand` walks the same tree the
completer uses:

```rust
let parsed = parse_argv(&tree, &[
    "compute", "knn", "--metric", "L2", "--verbose", "data.fvec",
])?;
// → ParsedCommand {
//     path: ["compute", "knn"],
//     flags: { "--metric": ["L2"], "--verbose": [""] },
//     positionals: ["data.fvec"],
//   }
```

Strict mode rejects unknown flags with `ParseError::UnknownFlag`;
`parse_argv_lenient` treats them as positionals. Handles `--key
value`, `--key=value`, boolean flags, repeats, and multi-level
subcommand traversal.

---

## 11.7 Help Rendering

`render_usage(node, path) → String` formats a `--help` block from
the tree:

```rust
let leaf = Node::leaf_with_flags(&["--metric"], &["--verbose"])
    .with_help("Compute KNN over base vectors")
    .with_flag_help("--metric",  "Distance metric: L2 / IP / COSINE")
    .with_flag_help("--verbose", "Print per-step progress");

println!("{}", render_usage(&leaf, &["myapp", "compute", "knn"]));
```

Output:

```text
USAGE: myapp compute knn

Compute KNN over base vectors

FLAGS:
  --metric   Distance metric: L2 / IP / COSINE
  --verbose  Print per-step progress
```

Subcommands are listed in a `SUBCOMMANDS:` section when the node
has children.

---

## 11.8 End-to-End Coding Scenario (Worked Example)

`veks-completion/examples/metricsql.rs` is a complete, runnable
adoption walkthrough — the same pattern any downstream user will
follow when wiring veks-completion into a tool that needs
grammar-aware completion. Read it as the reference implementation.

The four steps the example demonstrates:

### Step 1 — Implement the site-specific catalog

```rust
struct InMemoryCatalog { /* metrics, labels, values from your store */ }

impl MetricsqlCatalog for InMemoryCatalog {
    fn metric_names(&self, prefix: &str) -> Vec<String> { … }
    fn label_keys(&self, metric: &str, prefix: &str) -> Vec<String> { … }
    fn label_values(&self, metric: &str, label: &str, prefix: &str) -> Vec<String> { … }
}
```

The built-in MetricsQL grammar (functions, operators, time units,
modifiers) is baked into the provider; you supply the parts that
depend on what's actually in your metrics store.

### Step 2 — Build the tree with built-in options

```rust
fn build_tree() -> CommandTree {
    let catalog: Arc<dyn MetricsqlCatalog> = Arc::new(InMemoryCatalog::new());
    CommandTree::new("metricsql")
        .command("query",
            Node::leaf(&["--from", "--to", "--step"])
                .with_help("Execute a MetricsQL query against the configured backend.")
                .with_flag_help("--from", "Start of query window (RFC3339 or relative)")
                .with_flag_help("--to",   "End of query window")
                .with_flag_help("--step", "Resolution / step (e.g. '15s', '1m')"))
        .command("validate",
            Node::leaf(&[])
                .with_help("Parse a MetricsQL expression without executing."))
        .with_auto_help()                          // built-in: --help everywhere
        .with_metricsql_at(&["query"], catalog)    // built-in: grammar-aware completion
}
```

### Step 3 — Wire the entry point

```rust
fn main() {
    let tree = build_tree();

    // (1) Tab-completion callback — bash sets _METRICSQL_COMPLETE=bash
    if handle_complete_env("metricsql", &tree) { return; }

    let args: Vec<String> = std::env::args().collect();

    // (2) Print the activation snippet
    if args.get(1).map(|s| s.as_str()) == Some("completions") {
        print_bash_script("metricsql");
        return;
    }

    // (3) Structured argv parse against the same tree
    let argv: Vec<&str> = args.iter().skip(1).map(|s| s.as_str()).collect();
    let parsed = parse_argv(&tree, &argv).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(2);
    });

    // --help is uniformly available because of with_auto_help()
    if parsed.flags.contains_key("--help") {
        let mut node = &tree.root;
        for segment in &parsed.path {
            if let Some(child) = node.child(segment) { node = child; }
        }
        let mut full_path = vec!["metricsql"];
        full_path.extend(parsed.path.iter().copied());
        println!("{}", render_usage(node, &full_path));
        return;
    }

    // Dispatch on the resolved subcommand path
    match parsed.path.as_slice() {
        ["query"]    => { /* execute */ }
        ["validate"] => { /* parse only */ }
        _            => { /* usage */ }
    }
}
```

### Step 4 — Drive the demo / integration tests

The example's `demo` mode walks ten realistic MetricsQL cursor
positions and prints what the completer offers at each. Same idea
applies to integration tests — drive `complete_at_tap_with_raw`
directly with raw line + cursor offset:

```rust
let cands = complete_at_tap_with_raw(
    &tree,
    &["metricsql", "query", "up{job="],
    /*tap_count=*/ 1,
    /*raw_line=*/ "metricsql query up{job=",
    /*cursor=*/ 22,
);
assert!(cands.contains(&"\"prometheus\"".to_string()));
```

### Run it

```bash
cargo build --example metricsql -p veks-completion
eval "$(./target/debug/examples/metricsql completions)"

# Tab around interactively
./target/debug/examples/metricsql query 'up{<TAB>'
./target/debug/examples/metricsql query 'rate(http_requests_total[<TAB>'
./target/debug/examples/metricsql query 'sum by (<TAB>'
./target/debug/examples/metricsql --help
./target/debug/examples/metricsql query --help

# Or watch the demo print its scenario results
cargo run --example metricsql -p veks-completion -- demo
```

---

## 11.9 Built-in Options for Downstream Adopters

Two opt-in builders on `CommandTree` enable common capabilities
without writing the boilerplate.

### `with_auto_help`

Adds `--help` (boolean) to every node in the tree. Idempotent —
nodes that already declare `--help` are left alone. Once attached,
`--help` shows in tab completion at every level and is recognised by
`parse_argv` as a known flag. Pair with `render_usage` in your
handler:

```rust
let tree = CommandTree::new("myapp")
    .command("compute", Node::group(vec![
        ("knn", Node::leaf(&["--metric"])),
    ]))
    .with_auto_help();

let parsed = parse_argv(&tree, &argv)?;
if parsed.flags.contains_key("--help") {
    let node = walk_to(&tree.root, &parsed.path);
    println!("{}", render_usage(node, &parsed.path));
    return Ok(());
}
```

### `with_metricsql_at`

Attaches the built-in `providers::metricsql_provider` `SubtreeProvider`
at a specified subcommand path. Equivalent to manually descending to
the node and calling `with_subtree_provider(metricsql_provider(
catalog))`, but surfaces the intent at tree-construction time:

```rust
let catalog: Arc<dyn MetricsqlCatalog> = Arc::new(MyCatalog::new());
let tree = CommandTree::new("nbrs")
    .command("query", Node::leaf(&[]))
    .with_metricsql_at(&["query"], catalog);
```

Inside the `query` subtree, the provider performs grammar-aware
completion: top-of-expression returns metric/function names; inside
`{` returns label keys; after `key=` returns quoted values; inside
`[` returns time units; after aggregation functions returns
`by`/`without`; after `by`/`without` returns label keys; after
`offset` returns time units. See `providers::METRICSQL_FUNCTIONS`
for the baked-in vocabulary.

The `MetricsqlCatalog` trait supplies the site-specific parts
(metric names, label keys, label values for a given metric).

---

## 11.10 Subtree Providers (Context-Sensitive Completion)

For grammar-aware completion in any subtree (not just MetricsQL):

```rust
let provider: SubtreeProvider = Arc::new(|pp: &PartialParse| {
    // pp.completed       — tokenised completed words
    // pp.partial         — partial word under cursor
    // pp.tree_path       — resolved tree path
    // pp.raw_line        — full COMP_LINE (for grammar-aware providers)
    // pp.cursor_offset   — cursor byte offset within raw_line
    //
    // pp.bracket_state() — depth of (), {}, [], and quote state
    // pp.before_cursor() — slice of raw_line before cursor
    // pp.after_cursor()  — slice of raw_line after cursor
    // pp.trigger_char()  — last symbol-char before cursor
    // pp.ident_before_cursor() — partial identifier
    parse_my_dsl_and_complete(pp)
});

node = node.with_subtree_provider(provider);
```

The deepest matching subtree provider on the cursor's path takes
over completion. For DSLs with mixed syntax (label matchers, function
calls, range selectors, operators) — the MetricsQL provider in
`providers.rs` is the working example.

---

## 11.11 Directive Sets (Vocab-Driven CLIs)

For repeated flag patterns, declare `Directive`s once and apply onto
any node:

```rust
const DIRS: &[Directive] = &[
    Directive::closed("--metric", &["L2", "IP", "COSINE"])
        .with_help("Distance metric"),
    Directive::value("--name").with_help("Run name"),
    Directive::boolean("--verbose").with_help("Verbose output"),
];

let leaf = apply_directives(Node::leaf(&[]), DIRS);
```

Each directive becomes:
- a flag entry (boolean or value-taking),
- a `flag_help` entry (when `with_help` is set),
- a value provider (when `Directive::closed` is used — also feeds
  validation via `ClosedValues::validate`).

---

## 11.12 Bash Script Generation

`print_bash_script("veks")` generates a deliberately minimal hook
that hands raw `$COMP_LINE` + `$COMP_POINT` to the binary and
forces bash into "raw mode" so its own quoting heuristics don't
fight the engine's grammar-aware splicing:

```bash
_veks_complete() {
    local IFS=$'\n'
    local COMP_WORDBREAKS=$' \t\n<>;|&'
    COMPREPLY=($(_VEKS_COMPLETE=bash "veks" "$COMP_LINE" "$COMP_POINT"))
}
complete -o nosort -o nospace -F _veks_complete veks
```

Each line earns its place — see §11.13 for the rationale behind
`COMP_WORDBREAKS` and `-o nospace`.

- `local IFS=$'\n'` — candidates may contain spaces (quoted label
  values, multi-token completions); newline-only IFS preserves them
  through the `($(…))` array expansion.
- `local COMP_WORDBREAKS=$' \t\n<>;|&'` — strips bash's default
  `' " = ( :` from the wordbreak set so grammar tokens stay unsplit.
  See §11.13 "Raw Mode".
- `-o nosort` — preserves the engine's intentional layer ordering
  (rapid-tap tier ordering). Without this, readline alphabetises
  and the layer-stratified display is destroyed.
- `-o nospace` — most engine candidates are mid-context inserts
  (`delta(`, `up{`, `[5m`); a trailing space would push the cursor
  outside the context the user is building. The user types their
  own space when done.
- The shell's PID is read by the binary via `getppid()` (no env
  var plumbing needed).
- The completer name in the snippet is `argv[0]` verbatim — whatever
  the user invoked when they ran `veks completions`. No
  `current_exe()` resolution; no `cwd.join()` absolutising. So the
  snippet pasted to `~/.bashrc` re-invokes the binary the same way
  the user just did.

---

## 11.13 Completion Semantics: Intent + Raw Mode

Two layered design rules govern how the engine produces candidates.
Both come from a single core insight: **a completion candidate is
not just a string; it is a transition between context states.**
The engine and the bash hook cooperate so those transitions land
correctly without the shell's default heuristics second-guessing.

### Layer 1 — Completion Intent

Every candidate fits one of four intent categories. The category
governs whether the candidate opens a context, closes one, extends
within one, or wraps the whole expression up.

| Intent     | Example                  | Cursor lands  | Trailing space? | Outer wrappers      |
|------------|--------------------------|---------------|-----------------|---------------------|
| `OPEN`     | `delta(`, `{`, `[`       | inside        | NO              | must stay open      |
| `APPEND`   | `job=`, `http_requests`  | at end        | maybe           | never closed        |
| `CLOSE`    | `)`, `]`, `}`            | after         | maybe           | unaffected          |
| `TERMINAL` | finished label value `"`-closed | after  | YES             | may close           |

Concrete consequences:

- An `OPEN` candidate (`delta(`) inside a shell-wrapper-quoted
  expression must NOT cause readline to auto-close the wrapper —
  the user is mid-construction.
- An `APPEND` candidate (a label key) extends current context;
  trailing space depends on whether the user is composing a list
  (no space) or finishing a token (space).
- The engine currently expresses intent implicitly via the shape
  of the candidate (presence of `(`, trailing operator, etc.) and
  via the `-o nospace` shell directive (which uniformly suppresses
  trailing space — the user provides it). Promoting intent to a
  first-class enum is a natural future refactor; today the model
  lives in this doc and in the choice of candidate string.

### Layer 2 — Raw Mode (Engine ↔ Bash Hook Cooperation)

The engine alone cannot stop readline from "helpfully":

- Auto-closing an unmatched `'` or `"` when the candidate ends
  inside an open context (e.g. `delta(` inside `'…`).
- Splitting a candidate at `=`, `(`, `:` — bash's default
  `COMP_WORDBREAKS` includes these, so it would hand the engine
  fragmented "current word" boundaries that don't match grammar.
- Adding a trailing space after every completion.

The hook puts bash into a deliberately minimal "raw mode" so the
engine owns the splice contract end-to-end:

1. **`local COMP_WORDBREAKS=$' \t\n<>;|&'`** — keep only real shell
   metacharacters as wordbreaks. Strip:
   - `' "` (shell wrapper quotes) — so `'metricsql expr` is one
     word; readline can't auto-close.
   - `=` — so `up{job=` is one word; `key=value` candidates splice
     cleanly.
   - `(` — so `delta(rate(` is one word; nested function-call
     completions work.
   - `:` — so `[5m:1m]` subquery step is one word.
   This must match `PartialParse::DEFAULT_BASH_WORDBREAKS` exactly
   so the engine's `shell_word_start()` reasons about the same
   boundaries bash will splice against.

2. **`-o nospace`** — engine candidates are typically mid-context
   inserts; a trailing space would land the cursor outside the
   context being built. The user types their own space.

3. **`-o nosort`** — preserves engine's tier ordering.

4. **`local IFS=$'\n'`** — candidates may contain spaces.

The cost of raw mode is that **every candidate must be
splice-ready**: it must include the entire shell-perceived
"current word" prefix, not just the new content. The engine helper
`PartialParse::splice_candidate(target_start, suggestion)` does
this automatically — it returns `raw_line[shell_word_start..target_start]
+ suggestion`, so providers think in their own grammar terms
("the value is `prometheus`") and the helper produces the
shell-correct candidate (`up{job="prometheus"`).

### Why Both Layers Are Needed

Layer 2 alone (raw mode) makes mechanical splicing correct, but
without Layer 1 (intent) the engine still produces candidates that
land the cursor in the wrong place — e.g., a `delta(` candidate
followed by a trailing space would close the function-call entry
context the user is trying to enter. Layer 1 alone (intent
classification) cannot prevent readline from auto-closing the
wrapper quote — only the hook configuration can. The two layers
are independent forces; both must be on for grammar-aware
completion to "just work" inside complex expressions.

### Verifying the Contract

The provider tests in `veks-completion/src/providers.rs` exercise
both layers:

- Candidate-shape assertions verify Layer 1 (intent): e.g.,
  function-name completions end with `(` (OPEN), label keys end
  with `=`/`!=`/`=~`/`!~` (APPEND with operator), label values
  inside an open quote come back as a bare string (APPEND inside
  open `"`).
- `apply_substitution` round-trip tests verify Layer 2: take a
  candidate, splice it into the line at `shell_word_start`, and
  check the resulting line is exactly what the user expected to
  see — proving the engine's view of the splice point matches the
  hook's `COMP_WORDBREAKS`-driven view.
