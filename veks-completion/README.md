# veks-completion

Dynamic shell completion engine for Rust CLI tools, plus a companion
argv parser, help renderer, and tap-cadence model.

Zero non-`std` dependencies. One `CommandTree` declaration drives tab
completion, argv parsing, and `--help` rendering — single source of
truth for every CLI surface.

## What's in here

- **Tab completion** — context-aware candidates, value providers,
  per-flag aliases, dynamic option discovery.
- **Multi-tap rotating tiers** — single tap shows the primary
  commands; rapid follow-up taps reveal cumulative supersets in
  layer-ordered display.
- **Argv parser** — `parse_argv(tree, &argv) → ParsedCommand` walks
  the same tree the completer uses to produce a structured parse
  (path, flags, positionals).
- **Help rendering** — `render_usage(node, path)` formats a `--help`
  block from the tree.
- **Closed value sets** — `ClosedValues` produces both completion
  and validation from one declaration.
- **Subtree-aware completion** — context-sensitive providers can
  take over completion inside any subtree with a structured
  `PartialParse`.
- **Free-form attachment** — `Extras` slot lets embedders carry
  handler payloads, parser state, dispatch rules without forcing the
  crate to grow generics.
- **Directive sets** — bundle every surface a flag appears in
  (CLI form, help, value set, repeatability, optional YAML mirror)
  into one declaration; expand a slice of them onto a node in one
  call.

## Why not clap_complete?

Clap provides `clap_complete` for generating static shell completion
scripts. It works well for simple CLIs with a fixed command tree, but
breaks down for tools with dynamic commands, multi-level subgroups, or
context-sensitive value completion. This crate was built for the veks
CLI but is fully generic and reusable. The specific limitations it
addresses:

### 1. Dynamic pipeline commands

Veks has a `pipeline` command group whose subcommands are registered
at runtime from a `CommandRegistry`, not from clap's derive macros.
Clap's completion generator only sees the derive-defined tree.

### 2. Shorthand dispatch

`veks run` is shorthand for `veks prepare run`; `veks config` for
`veks datasets config`. These are hidden top-level subcommand aliases.
Clap either hides them entirely (no completions) or shows them
alongside the originals (duplicates). veks-completion uses tap-tier
levels: hidden shortcuts get `level=2` and surface only on rapid
double-tap.

### 3. One-level-at-a-time completion with rotating tiers

Clap eagerly chains subcommands. veks-completion completes one level
at a time. Within a level, rapid double-tap reveals the next tier as
a cumulative superset, layered in display order so primary commands
always appear first.

### 4. Context-sensitive value completion

Per-option `ValueProvider` functions run arbitrary logic. `ClosedValues`
turns a static value set into a provider with one declaration that's
also queryable by parsers for validation.

### 5. Mixed derive + dynamic tree

The completion engine walks the augmented `clap::Command` tree at
completion time, so it always reflects the actual command structure
regardless of how commands were registered.

## How it works

`Node` is a single struct that carries everything a CLI-tree node *can*
have: subcommand children, flags (value-taking + boolean), value
providers, discovery metadata (category + level), help text, plus
hooks (subtree provider, free-form extras). A node with no children is
"leaf-shaped"; with children it's "group-shaped"; with both it's
hybrid (e.g. a group that itself accepts cross-cutting flags).

A `CommandTree` is a root node + globals. At startup an embedder builds
the tree once. At completion time the engine walks the tree following
the user's typed words and returns candidates.

The bash script the engine emits is minimal — it just calls back into
the binary. Completions never get out of sync with the installed
binary because the binary itself produces them.

## Usage — tab completion

```rust
use veks_completion::{CommandTree, Node, handle_complete_env};

let tree = CommandTree::new("myapp")
    .command("run", Node::leaf_with_flags(
        &["--input", "--output", "--threads"],
        &["--verbose", "--dry-run"],
    ))
    .group("compute", Node::group(vec![
        ("knn",   Node::leaf(&["--base", "--query", "--metric"])),
        ("stats", Node::leaf(&["--input"])),
    ]));

// In main(), before argument parsing:
if handle_complete_env("myapp", &tree) {
    std::process::exit(0);
}

// And register a `completions` subcommand that prints the bash
// activation snippet for `eval "$(myapp completions)"`.
```

## Usage — argv parsing

The same `CommandTree` produces structured parses:

```rust
use veks_completion::{parse_argv, ParsedCommand};

let parsed = parse_argv(&tree, &[
    "compute", "knn", "--metric", "L2", "--verbose", "data.fvec",
])?;
assert_eq!(parsed.path,        vec!["compute", "knn"]);
assert_eq!(parsed.flags["--metric"],  vec!["L2".to_string()]);
assert_eq!(parsed.flags["--verbose"], vec!["".to_string()]); // boolean
assert_eq!(parsed.positionals, vec!["data.fvec"]);
```

`parse_argv_lenient` treats unknown flags as positionals — useful for
pass-through CLIs.

## Usage — help rendering

```rust
use veks_completion::render_usage;

let leaf = Node::leaf_with_flags(&["--metric"], &["--verbose"])
    .with_help("Compute KNN over base vectors")
    .with_flag_help("--metric",  "Distance metric: L2 / IP / COSINE")
    .with_flag_help("--verbose", "Print per-step progress");

println!("{}", render_usage(&leaf, &["myapp", "compute", "knn"]));
// → USAGE: myapp compute knn
//
//   Compute KNN over base vectors
//
//   FLAGS:
//     --metric   Distance metric: L2 / IP / COSINE
//     --verbose  Print per-step progress
```

## Usage — closed value sets (completion + validation)

```rust
use veks_completion::{ClosedValues, Node};

let metrics = ClosedValues::Static(&["L2", "IP", "COSINE"]);
assert!(metrics.validate("L2"));
assert!(!metrics.validate("bogus"));
assert_eq!(metrics.complete("CO"), vec!["COSINE"]);

let leaf = Node::leaf(&["--metric"])
    .with_value_provider("--metric", metrics.clone().into_provider());
```

## Usage — directive sets

When you have many flags that share a vocabulary shape, declare them
once as a `&[Directive]` and apply onto any node:

```rust
use veks_completion::{Directive, apply_directives, Node};

const DIRS: &[Directive] = &[
    Directive::closed("--metric", &["L2", "IP", "COSINE"])
        .with_help("Distance metric"),
    Directive::value("--name").with_help("Run name"),
    Directive::boolean("--verbose").with_help("Verbose output"),
];

let leaf = apply_directives(Node::leaf(&[]), DIRS);
// → flags + flag-help + value-providers + validation, all wired.
```

## Usage — multi-tap rotating tiers

Tag commands with `with_level(n)` to control which tap-tier reveals
them:

```rust
let tree = CommandTree::new("nbrs")
    .command("run",          Node::leaf(&[]).with_level(1))
    .command("--inspector",  Node::leaf(&[]).with_level(2))
    .command("--summary",    Node::leaf(&[]).with_level(2))
    .command("describe",     Node::leaf(&[]).with_level(3))
    .command("bench",        Node::leaf(&[]).with_level(3));
```

Behavior:

- **Tap 1 (cold or after pause):** layer 1 only — `run`
- **Tap 2 (within 200 ms):** cumulative layers 1+2 — `run`,
  `--inspector`, `--summary`
- **Tap 3 (within 200 ms):** cumulative layers 1+2+3 — all five.
  At max, the persistent state resets, so a fourth rapid tap
  cycles back to layer 1.
- **Pause > 200 ms:** any tap starts fresh at layer 1.
- **Within each tap's result:** sorted in *layer order* (layer 1
  first, then layer 2, …); within a layer, `--`-flags last,
  alphabetical otherwise.

The cadence rule is also exposed as a pure function for embedders
that want their own clock/state:

```rust
use veks_completion::{TapState, next_tap_state, TAP_ADVANCE_MS};

let (tap_count, next) = next_tap_state(
    Some((prev_state, "current_input_key")),
    now_ms,
    "current_input_key",
    max_level,
);
```

`TAP_ADVANCE_MS` is the public window constant (200 ms).

## Built-in options for downstream adopters

When you build a `CommandTree`, two opt-in builders enable common
capabilities without writing the boilerplate yourself. Both are
fully additive — they walk the tree once at finalisation time:

### `with_auto_help`

```rust
let tree = CommandTree::new("myapp")
    .command("compute", Node::group(vec![
        ("knn", Node::leaf(&["--metric"])),
    ]))
    .command("run", Node::leaf(&["--input"]))
    .with_auto_help();
```

Walks every node and adds `--help` (boolean) plus a default help line
("Show usage information for this command."). Idempotent — nodes that
already declare `--help` are left alone. Once attached, `--help`:

- shows up in tab completion at every level,
- is recognised by `parse_argv` as a known boolean flag (so users
  can type `--help` anywhere without "unknown flag" errors), and
- can be paired with `render_usage(node, path)` in your handler to
  print the help block:

  ```rust
  let parsed = parse_argv(&tree, &argv)?;
  if parsed.flags.contains_key("--help") {
      let node = walk_to(&tree.root, &parsed.path);
      println!("{}", render_usage(node, &parsed.path));
      return Ok(());
  }
  ```

### `with_metricsql_at`

```rust
use std::sync::Arc;
use veks_completion::providers::{MetricsqlCatalog, metricsql_provider};

let catalog: Arc<dyn MetricsqlCatalog> = Arc::new(MyCatalog::new());
let tree = CommandTree::new("nbrs")
    .command("query", Node::leaf(&[]))
    .with_metricsql_at(&["query"], catalog);
```

Attaches a built-in `metricsql_provider` `SubtreeProvider` at the
specified node path. Equivalent to manually `walk_path_mut(...)`-ing
to the node and calling `with_subtree_provider(metricsql_provider(
catalog))`, but surfaces the intent at tree-construction time.

The MetricsQL provider understands:

| Cursor context                       | Suggestions                              |
|--------------------------------------|------------------------------------------|
| Top of expression / after a binop    | metric names + function names            |
| Inside `{` (label-matcher block)     | label keys for the preceding metric      |
| After `key=` or `key=~` inside `{…}` | label values (in quoted form)            |
| Inside `"…"` after `key=`            | label values (bare; quote already open)  |
| Inside `[` (range selector)          | time units (`5m`, `1h`, etc.)            |
| After `sum`/`avg`/`count`…           | `by` / `without`                         |
| After `sum by`/`without`             | label keys (any)                         |
| After `offset`                       | time-unit suggestions                    |

Site-specific data (metric names, label keys, label values) comes
from your `MetricsqlCatalog` impl. The built-in vocabulary
(functions, time units, operators, aggregation modifiers) is baked
in — see `providers::METRICSQL_FUNCTIONS`,
`providers::METRICSQL_TIME_UNITS`, etc.

## Usage — context-sensitive subtree completion

When tab completion needs grammar-aware behavior in a subtree (e.g.
inside a query DSL), attach a `SubtreeProvider`:

```rust
use std::sync::Arc;
use veks_completion::{Node, PartialParse, SubtreeProvider};

let metrics_provider: SubtreeProvider = Arc::new(|pp: &PartialParse| {
    // pp.completed = full chain of words to the left of the cursor
    // pp.partial   = the partial word under the cursor
    // pp.tree_path = the resolved tree-node path
    // Return whatever candidates make sense given that state.
    parse_my_dsl_and_complete(pp.partial)
});

let tree = CommandTree::new("app")
    .command("metrics",
        Node::group(vec![("match", Node::leaf(&[]))])
            .with_subtree_provider(metrics_provider));
```

The deepest matching subtree provider on the path takes over
completion. This is the recommended replacement for the pre-walker
hook pattern.

## Usage — handler attachment

Embedders that want to dispatch to handlers from the same tree can
attach arbitrary payloads via `Extras`:

```rust
use veks_completion::{Extras, Node};

struct MyHandler { … }

let leaf = Node::leaf(&["--input"])
    .with_extras(Extras::new(MyHandler { … }));

// Later, after parse_argv resolves `path`:
if let Some(extras) = node.extras() {
    if let Some(handler) = extras.downcast::<MyHandler>() {
        handler.run(parsed);
    }
}
```

`Extras` wraps `Arc<dyn Any + Send + Sync>` — no generic on `Node`,
no hard dependency on tokio or any handler framework.

## Architecture

```
veks-completion/        # This crate — generic completion + parse + help
  src/lib.rs            # Node, CommandTree, complete(), parse_argv(),
                        # render_usage(), ClosedValues, Directive, …

veks/src/cli/
  dyncomp.rs            # Walks clap::Command -> CommandTree;
                        # registers global ValueProviders.
  mod.rs                # Wires `eval "$(veks completions)"` snippet.
```

The completion crate has no dependency on clap, veks, or any pipeline
code. The clap-to-tree conversion lives in `dyncomp.rs` inside the
`veks` binary crate.

## Examples

See `examples/basic.rs` for a complete working demo:

```bash
cargo run --example basic -p veks-completion
```
