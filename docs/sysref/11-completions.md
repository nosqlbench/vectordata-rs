# 11. Shell Completions

Dynamic tab-completion (plus argv parsing and help rendering) driven
by a single `CommandTree` declaration. The same tree drives every
CLI surface — completions, `--help`, argv parse — so they can't drift
out of sync.

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

## 11.8 Built-in Options for Downstream Adopters

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

## 11.9 Subtree Providers (Context-Sensitive Completion)

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

## 11.10 Directive Sets (Vocab-Driven CLIs)

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

## 11.11 Bash Script Generation

`print_bash_script("veks")` generates:

```bash
_veks_complete() {
    local IFS=$'\n'
    COMPREPLY=($(_VEKS_COMPLETE=bash _COMP_SHELL_PID=$$ "veks" \
        "$COMP_LINE" "$COMP_POINT" 2>/dev/null))
    if [[ ${#COMPREPLY[@]} -ge 1 ]] \
        && [[ "${COMPREPLY[0]}" == *= || "${COMPREPLY[0]}" == */ ]]; then
        compopt -o nospace 2>/dev/null
    fi
}
complete -o nosort -F _veks_complete veks
```

Key details:

- `-o nosort` — preserves the engine's intentional layer ordering.
  Without this, readline alphabetises and the layer-stratified
  display is destroyed.
- `_COMP_SHELL_PID=$$` — passes the shell's PID so the engine can
  scope tap-cadence state per shell.
- The completer name in the snippet is `argv[0]` verbatim — whatever
  the user invoked when they ran `veks completions`. No
  `current_exe()` resolution; no `cwd.join()` absolutising. So the
  snippet pasted to `~/.bashrc` re-invokes the binary the same way
  the user just did.
