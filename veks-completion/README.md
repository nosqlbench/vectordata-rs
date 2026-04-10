# veks-completion

Dynamic shell completion engine for Rust CLI tools.

Zero dependencies. Define your command tree, get context-aware tab completion
with option filtering, value providers, and dynamic discovery — no static
code generation.

## Why not clap_complete?

Clap provides `clap_complete` for generating static shell completion scripts.
It works well for simple CLIs with a fixed command tree, but it breaks down
for tools with dynamic commands, multi-level subgroups, or context-sensitive
value completion. This crate was built for the veks CLI but is fully generic
and reusable. Here are the specific limitations it addresses:

### 1. Dynamic pipeline commands

Veks has a `pipeline` command group whose subcommands are registered at
runtime from a `CommandRegistry`, not from clap's derive macros. Commands
like `compute knn`, `analyze stats`, `transform extract` etc. are
two-level paths (`group subcommand`) that the registry resolves
dynamically. Clap's completion generator only sees the derive-defined
tree and misses the entire pipeline command catalog.

### 2. Shorthand dispatch

Veks supports shorthand commands: `veks run` is equivalent to
`veks prepare run`, and `veks config` resolves to `veks datasets config`.
These shorthands are implemented as hidden root-level subcommands
cloned from the full tree. Clap's completion engine either hides them
(no completions) or shows them alongside the originals (duplicates).

### 3. One-level-at-a-time completion

Clap's completion eagerly chains subcommands: typing `veks com<TAB>`
might expand to `veks compute knn` in one step if there's only one
match. This is disorienting for a deep command tree. The veks completion
engine completes one level at a time — `veks com<TAB>` expands to
`veks compute`, and a second tab shows the compute subcommands.

### 4. Context-sensitive value completion

Pipeline commands accept `--option value` pairs where the valid values
depend on context: `--dataset` should complete to known dataset names,
`--profile` to profile names within the selected dataset, `--source` to
files matching vector formats. Clap's value hints are limited to
predefined categories (file paths, directory paths, etc.). This crate
supports per-option `ValueProvider` functions that run arbitrary logic
at completion time.

### 5. Mixed derive + dynamic tree

The veks CLI combines clap derive-based subcommands (`datasets`,
`prepare`, `interact`) with dynamically-registered pipeline commands
and runtime-generated help trees. The completion engine walks the
full augmented `clap::Command` tree (built by `build_augmented_cli()`)
at completion time, so it always reflects the actual command structure
regardless of how commands were registered.

## How it works

The crate provides a generic `CommandTree` with `Node` variants (Leaf
and Group). At startup, `dyncomp::build_tree()` walks the clap
`Command` tree recursively and builds the `CommandTree`. When the
shell invokes completion (via the `_VEKS_COMPLETE=bash` env var), the
engine:

1. Parses the current input words
2. Walks the `CommandTree` to find the deepest matching node
3. Returns candidates: child commands (for groups) or `--option` names
   (for leaves)
4. For `--option <TAB>`, invokes the registered `ValueProvider` if one
   exists for that option

The completion script itself is minimal — it sets the env var and
re-invokes the veks binary, which does the actual tree walk and prints
candidates. This means completions are always current with the
installed binary.

## Usage

```rust
use veks_completion::{CommandTree, Node, handle_complete_env, print_bash_script};

let tree = CommandTree::new("myapp")
    .command("run", Node::leaf_with_flags(
        &["--input", "--output", "--threads"],
        &["--verbose", "--dry-run"],
    ))
    .group("compute", Node::group(vec![
        ("knn", Node::leaf(&["--base", "--query", "--metric"])),
        ("stats", Node::leaf(&["--input"])),
    ]))
    .hidden_command("_debug", Node::leaf(&["--trace"]));

// In main(), before argument parsing:
if handle_complete_env("myapp", &tree) {
    std::process::exit(0);
}

// Add a "completions" subcommand that prints the bash script:
// eval "$(myapp completions)"
```

Users enable completion with:

```bash
# Add to .bashrc:
eval "$(myapp completions)"
```

## Examples

See `examples/basic.rs` for a complete working demo:

```bash
cargo run --example basic -p veks-completion
```

## Architecture

```
veks-completion/        # This crate — generic completion engine
  src/lib.rs            # CommandTree, Node, complete(), print_bash_script()

veks/src/cli/
  dyncomp.rs            # Walks clap::Command -> CommandTree
                        # Registers global ValueProviders (datasets, profiles)
  mod.rs                # Wires eval "$(veks completions)" output
```

The completion crate has no dependency on clap, veks, or any pipeline
code. It is a pure tree-walker that takes a `CommandTree` and returns
string candidates. The clap-to-tree conversion lives in `dyncomp.rs`
inside the `veks` binary crate.
