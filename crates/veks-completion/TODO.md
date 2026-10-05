# veks-completion — improvement notes

Source: gaps surfaced while building `nbrs/src/cli_spec/`, a runtime CLI spec
that drives both argv parsing and shell completion from a single declaration.
The crate currently owns *completion* but stops short of a few capabilities
that any "single source of truth" CLI model needs. Each item below names a
concrete gap and proposes a direction; none is urgent in isolation, but
together they would let a downstream crate skip building a parallel
`Command`/`Flag` model.

**Status**: every item landed, including the full Leaf/Group unification
that was previously deferred. `Node` is now a single struct, not an enum.
A node carries `children` *and* `flags` orthogonally; "leaf" is just
`children.is_empty()`. The internal pattern-match boilerplate that
`Node::Leaf { … } | Node::Group { … }` forced is gone.

**Beyond the original 10**: a `providers` module landed with a built-in
`metricsql_provider` for grammar-aware MetricsQL/PromQL completion
(label matchers, function calls, range selectors, operators,
aggregation modifiers, time units). `PartialParse` grew `raw_line` +
`cursor_offset` + grammar helpers (`bracket_state`, `trigger_char`,
`ident_before_cursor`, `before_cursor`/`after_cursor`) so subtree
providers can do real grammar work. Two opt-in `CommandTree` builders
(`with_auto_help`, `with_metricsql_at`) let downstream embedders
enable common capabilities at tree-construction time. 17 MetricsQL
scenario tests pin the provider against realistic queries.

## 1. Closed-value-set providers without per-set glue functions ✅ DONE

**Today (was):** every closed-set provider required a dedicated `fn(&str,
&[&str]) -> Vec<String>` that closed over a `&'static` constant.

**Now:** `ClosedValues::Static(&'static [&'static str])` and
`ClosedValues::Owned(Vec<String>)` produce both completion (`complete()`)
and validation (`validate()`) from one declaration. Convert to a
`ValueProvider` via `into_provider()` or the `From` impl. Spec authors stop
writing wrapper fns.

## 2. Tree-global value providers in the spec ✅ DONE

**Now:** `Node::with_promoted_global(token, provider)` stages a global on
the node it conceptually belongs to. `CommandTree::lift_promoted_globals()`
walks the tree at finalization and lifts every staged entry into the
tree-level globals map. The post-build patching pattern is no longer
needed.

## 3. Subcommand-as-leaf vs. group with handler ✅ DONE — full unification

**Now:** `Node` is a single struct. Children and flags are orthogonal
fields. A node with no children is leaf-shaped (`is_leaf() ⇒ true`); a
node with children is group-shaped; a node with both is hybrid (e.g.
`report --workload x base`, where `report` accepts `--workload` *and* has
a `base` subcommand).

The completion engine handles all three shapes uniformly: subcommands
precede flags in the candidate list when the cursor is at a fresh-word
position; value-completion semantics for `--flag <TAB>` and `--key=<TAB>`
forms work the same regardless of whether the parent node also has
children. `with_child` and `with_flags` / `with_boolean_flags` work on
any node — the variant-shape decision is no longer made at construction
time, it's just whether you ever called `with_child`.

API additions: `Node::new()`, `Node::is_leaf()`, `Node::is_group()`,
`Node::with_flags(&[…])`, `Node::with_boolean_flags(&[…])`,
`Node::children()`, `Node::flags()`. The old per-shape constructors
(`Node::leaf`, `Node::leaf_with_flags`, `Node::group`, `Node::empty_group`)
remain as ergonomic shortcuts that just preset which fields are
non-empty. The `with_group_flags` / `with_group_value_provider` /
`group_flags` shortcuts from the prior additive pass are gone — use the
unified `with_flags` / `with_value_provider` / `flags` accessors.

Pinned by `hybrid_node_completes_children_and_flags_together`.

## 4. Per-flag value providers via aliases ✅ DONE

**Now:** `Node::with_value_provider_aliases(&["--tofile", "--to-file",
"--out"], provider)` registers one provider against every alias in the
slice in one call. No more per-alias `with_value_provider` boilerplate.

## 5. Validation surface co-located with completion ✅ DONE

**Now:** `ClosedValues::validate(&str) -> bool` shares the same set the
completer uses. Parsers can call it on each parsed value to reject
out-of-set inputs at parse time rather than runtime. The directive-set
adapter (item 10) wires this through automatically.

## 6. Help/usage rendering shares the same model ✅ DONE

**Now:** `Node::with_help(text)` and `Node::with_flag_help(flag, text)`
attach help on every node and flag. `render_usage(node, path) -> String`
walks the tree to format a `--help` block (USAGE / description / FLAGS
section / SUBCOMMANDS section). The same model that drives tab drives
help.

## 7. Subtree-defined dynamic interpretation ✅ DONE

**Now:** `Node::with_subtree_provider(provider)` attaches a
`SubtreeProvider` (`Arc<dyn Fn(&PartialParse) -> Vec<String>>`) to any
node. The completion engine fires the deepest matching provider when the
cursor is inside the subtree, threading through a structured
`PartialParse { completed, partial, tree_path }`. Pre-walker hook patterns
move *into* the tree.

## 8. Async/handler integration ✅ DONE — via `Extras` payload

**Now:** `Node::with_extras(Extras::new(my_handler))` attaches an arbitrary
`Send + Sync + 'static` payload to any node. Recover via
`node.extras().and_then(|e| e.downcast::<Handler>())`.

**Design choice:** the TODO proposed `extras: T` as a generic on `Node`.
That would have rippled into every existing test, every downstream consumer,
and every internal walker. The `Box<dyn Any>`-via-`Arc` form here gets the
same downstream functionality (handler/parser-state/dispatch attachment)
without forcing every existing call site to thread a type parameter.
Compromise on aesthetic; identical user-facing capability.

## 9. Argv parser integration ✅ DONE

**Now:** `parse_argv(tree, &argv) -> Result<ParsedCommand, ParseError>`
walks the same `CommandTree` the completer uses to produce a structured
parse:
  - `path: Vec<&str>` — subcommand chain reached
  - `flags: BTreeMap<String, Vec<String>>` — collected flag values
    (booleans map to a single empty entry; repeats append)
  - `positionals: Vec<&str>` — anything not consumed as a flag/subcommand

Strict mode rejects unknown flags; `parse_argv_lenient` treats them as
positionals (for pass-through CLIs). Handles `--key value`, `--key=value`,
boolean flags, and multi-level subcommand traversal. Single source of
truth: a flag added to the tree picks up both completion and parsing.

## 10. Vocab-style structured directive registries ✅ DONE

**Now:** `Directive { cli_flag, help, values, boolean, repeatable,
yaml_directive }` bundles every surface a flag appears in. Const
constructors (`Directive::closed`, `::value`, `::boolean`) plus
`with_help`, `with_yaml`, `repeatable` builders. `apply_directives(node,
&[Directive]) -> Node` expands a slice into per-flag completion + flag-help
+ value-provider registrations in one call.

Vocab-shaped CLIs become a one-liner:

```rust
const REPORT_DIRECTIVES: &[Directive] = &[
    Directive::closed("--metric", &["L2", "IP", "COSINE"])
        .with_help("Distance metric"),
    Directive::value("--workload").with_help("Workload spec path"),
    Directive::boolean("--dry-run").with_help("Print plan and exit"),
];

let leaf = apply_directives(Node::leaf(&[]), REPORT_DIRECTIVES);
```

---

## Adapter that lives downstream until veks absorbs the shape

The bridge from veks → "spec is also the parser + help" was built once
already in `/mnt/datamir/home/jshook/projects/nb-rs/nbrs/src/cli_spec/`.
With items 1–10 above landed, that adapter shrinks substantially:

- `cli_spec/mod.rs` — the parallel `Command`/`Flag`/`ValueProvider` types
  can now be replaced by direct `Node` + `Directive` use.
- `cli_spec/walker.rs` — argv → `ParsedCommand` is now `parse_argv`.
- `cli_spec/completion.rs` — most of the manual `Command` →
  `CommandTree` translation is no longer needed; `apply_directives`
  registers flags + completion + help in one call.
- `cli_spec/help.rs` — `render_usage` replaces hand-maintained help blocks.

Reading that module pre-update is still the fastest way to see what the
new APIs are designed to support; updating it to use them is the natural
next downstream change.
