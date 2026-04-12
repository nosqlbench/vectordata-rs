# 11. Shell Completions

Dynamic tab-completion for bash (and other shells) that stays in sync
with the CLI definition automatically.

---

## 11.1 User Setup

```bash
eval "$(veks completions)"                  # activate for current session
echo 'eval "$(veks completions)"' >> ~/.bashrc  # make permanent
```

This generates a bash completion script that calls back into the
veks binary on each tab press. The binary walks its own clap command
tree and returns candidates — no static completion file to maintain.

---

## 11.2 Architecture

```
User presses Tab
  → bash calls: _VEKS_COMPLETE=bash veks -- word1 word2 partial
    → veks builds CommandTree from clap definitions
    → veks_completion::complete(tree, words) returns candidates
  → bash displays candidates
```

The completion tree is built fresh on every invocation from
`clap::Command`. This means:
- Every subcommand, option, and flag is always current
- Hidden commands are excluded from suggestions
- No manual sync between CLI definition and completions

### Two-crate design

**`veks-completion`** — generic library, no clap dependency:
- `CommandTree`, `Node`, `ValueProvider` types
- `complete()` function that walks the tree
- `print_bash_script()` for script generation

**`veks/src/cli/dyncomp.rs`** — veks-specific wiring:
- `build_tree()` walks clap commands → `CommandTree`
- Registers global value providers for `--dataset`, `--at`, etc.

---

## 11.3 Command Tree

```rust
pub enum Node {
    Group { children: BTreeMap<String, Node> },
    Leaf {
        options: Vec<String>,           // --flag and --option names
        flags: HashSet<String>,         // boolean flags (no value)
        value_providers: BTreeMap<String, ValueProvider>,
        dynamic_options: Option<DynamicOptionsProvider>,
    },
}
```

**Group** nodes represent subcommand levels (`veks datasets` →
`list`, `probe`, `cache`, ...). Completing at a group shows its
children.

**Leaf** nodes represent terminal commands with their options.
Completing at a leaf shows available options, filtering out those
already on the command line.

---

## 11.4 Value Providers

When the user tabs after a `--option`, the completion engine calls
a registered value provider to suggest values:

```rust
pub type ValueProvider = fn(partial: &str, context: &[&str]) -> Vec<String>;
```

### Global providers

Registered on the `CommandTree`, apply to any command:

| Option | Provider | Values |
|--------|----------|--------|
| `--dataset` | `complete_dataset_names` | Dataset names from configured catalogs |
| `--profile` | `complete_profile_names` | Profile names (filtered by `--dataset`) |
| `--metric` | `complete_metrics` | L2, COSINE, DOT_PRODUCT, L1 |
| `--at` | `complete_catalog_urls` | Catalog indexes with URL descriptions |
| `--shell` | `complete_shells` | bash, zsh, fish, elvish, powershell |

### Catalog index completion

`--at` shows numbered indexes with descriptions via stderr:

```
$ veks datasets list --at <tab>

  1 = https://example.com/datasets/production/
  2 = https://example.com/datasets/staging/
1  2
```

Descriptions print to `/dev/tty` (visible to user), candidates go
to stdout (captured by bash). When only one match remains, it
auto-completes silently.

### Per-command providers

Registered on individual `Node::Leaf` instances for command-specific
option values.

### Dynamic options

The `DynamicOptionsProvider` discovers additional `key=value` options
from context — for example, reading a referenced file to extract
parameter names that become completable.

---

## 11.5 Completion Algorithm

1. Split input into completed words + partial word
2. Walk the tree following completed words to find current node
3. Check if previous word is a `--option` needing a value:
   - If value provider exists → call it with partial
   - If flag (boolean) → skip, treat partial as next option
4. At a **Group** → return matching child names
5. At a **Leaf** → return matching options minus already-consumed ones
6. Sort: bare params first, then `--flags` alphabetically

### Consumed option filtering

Options already present on the command line are filtered out.
Boolean flags and `--option value` pairs are both tracked.
`key=value` bare params are recognized and filtered by the `key=`
prefix.

---

## 11.6 Bash Script Generation

`print_bash_script("veks")` generates:

```bash
_veks_complete() {
    COMP_WORDBREAKS="${COMP_WORDBREAKS//:}"
    # Parse COMP_LINE into words (handles quotes)
    local IFS=$'\n'
    COMPREPLY=($(_VEKS_COMPLETE=bash veks -- "${words[@]}" 2>/dev/tty))
}
complete -o default -o bashdefault -o nosort -F _veks_complete veks
```

Key details:
- `2>/dev/tty` — stderr goes to terminal (for catalog descriptions),
  not captured by COMPREPLY
- `-o nosort` — preserves the engine's intentional ordering
  (bare params before flags)
- Quote-aware word splitting handles paths with spaces
- `COMP_WORDBREAKS` strips `:` to avoid breaking URLs

---

## 11.7 Adding Completions for New Commands

New clap subcommands automatically get completion support — no
registration needed. To add value suggestions for a specific option:

```rust
// In dyncomp.rs build_tree():
tree = tree.global_value_provider("--my-option", my_provider);

fn my_provider(partial: &str, _context: &[&str]) -> Vec<String> {
    vec!["value1", "value2", "value3"]
        .into_iter()
        .filter(|v| v.starts_with(partial))
        .map(|v| v.to_string())
        .collect()
}
```
