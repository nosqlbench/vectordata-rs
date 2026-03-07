<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 07 — Command Documentation

## 7.1 Overview

Every pipeline command — and every CLI command generally — MUST provide
built-in markdown documentation that can be surfaced when users explore
and use the system. This documentation is the authoritative reference for
what each command does, how it works, what its options mean, and what
resources it consumes.

## 7.2 Requirements

### REQ-DOC-01: Every command has built-in documentation

Every `CommandOp` implementation MUST provide structured markdown
documentation via a new trait method:

```rust
pub trait CommandOp: Send {
    // ... existing methods ...

    /// Return markdown documentation for this command.
    ///
    /// The documentation is structured with a summary line (first paragraph),
    /// followed by detailed description, option semantics, examples, and notes.
    fn command_doc(&self) -> CommandDoc;
}
```

Where `CommandDoc` is:

```rust
pub struct CommandDoc {
    /// One-line summary (plain text, no markdown).
    /// Used in completion suggestions and command listings.
    pub summary: String,

    /// Full markdown documentation body.
    /// Rendered when the user requests detailed help.
    pub body: String,
}
```

### REQ-DOC-02: Documentation structure

The `body` field MUST follow a consistent markdown structure:

```markdown
# command-path

One-line summary (same as `summary` field).

## Description

Multi-paragraph explanation of what the command does, when to use it,
and how it fits into a typical pipeline.

## Options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `source` | Path | yes | — | Input file path |
| `output` | Path | yes | — | Output file path |
| `threads` | int | no | `num_cpus` | Worker thread count |

## Examples

### Basic usage

```yaml
- run: command path
  source: input.fvec
  output: output.fvec
```

### With format conversion

```yaml
- run: command path
  source: input.npy
  from: npy
  output: output.fvec
```

## Resources

This command declares: `mem`, `threads`, `readahead`

| Resource | Usage |
|----------|-------|
| `mem` | Buffers up to N vectors in memory per thread |
| `threads` | Parallel distance computation |
| `readahead` | Sequential prefetch window for mmap'd vectors |

## Notes

- Any caveats, edge cases, or important behavioral details.
- References to related commands.
```

The `Options` section MUST be consistent with `describe_options()` — in
fact, it SHOULD be generated from the same source data to prevent drift.

The `Resources` section MUST be consistent with `describe_resources()`.

### REQ-DOC-03: Completion summaries

The dynamic completion system MUST surface the `summary` field when
listing commands. Summaries are prefixed with `#` to make them
innocuous in shell completion (comment syntax):

```
$ veks pipeline compute <TAB>
filtered-knn    # Brute-force filtered KNN with predicate-key pre-filtering
knn             # Brute-force exact K-nearest-neighbor computation
sort            # Sort vectors by ordinal mapping
```

```
$ veks pipeline generate <TAB>
derive              # Derive new facets from existing data
fvec-extract        # Extract a subset of vectors from an fvec file
hvec-extract        # Extract a subset of vectors from an hvec file
ivec-extract        # Extract a subset of vectors from an ivec file
predicate-keys      # Evaluate predicates against metadata to produce match ordinals
predicates          # Generate random predicate trees from metadata survey
vectors             # Generate synthetic random vectors
```

The `#` prefix ensures that if the completion is accidentally accepted,
it parses as a shell comment and does nothing. The completion engine
formats these as:

```
command-name    # summary text
```

Where the spacing aligns summaries for readability.

### REQ-DOC-04: Option-level help in completions

When completing option names, the completion system SHOULD show the
option's description from `describe_options()`:

```
$ veks pipeline compute knn --<TAB>
--base          # Path to base vector file (fvec/hvec)
--query         # Path to query vector file (fvec/hvec)
--indices       # Output path for neighbor indices (ivec)
--distances     # Output path for neighbor distances (fvec)
--neighbors     # Number of neighbors to compute (default: 100)
--metric        # Distance metric: L2, Cosine, DotProduct, L1
--resources     # Resource configuration (mem, threads, readahead)
```

### REQ-DOC-04a: Option value hints in completions

When the user has selected an option and the cursor is at the position
where a value argument should be provided, the completion system MUST
include a `# description` entry in the suggestions to clarify what kind
of value is expected. This hint uses the option's `description` and
`type_name` from `describe_options()`, and if a `default` is present it
is shown as well.

The hint entry is formatted as a shell comment so that accidentally
accepting it has no effect:

```
$ veks pipeline compute knn --metric <TAB>
# Distance metric: L2, Cosine, DotProduct, L1 (type: enum)
L2
Cosine
DotProduct
L1
```

```
$ veks pipeline compute knn --neighbors <TAB>
# Number of neighbors to compute (type: int, default: 100)
```

```
$ veks pipeline compute knn --base <TAB>
# Path to base vector file (fvec/hvec) (type: Path)
<file completions follow>
```

The hint format is:

```
# <description> (type: <type_name>[, default: <default>])
```

**Rules:**

1. The `# description` hint MUST always appear as the first suggestion
   in the completion list.
2. If the option has a known set of valid values (e.g., an `enum` type
   with variants listed in the description), those values SHOULD follow
   the hint as concrete completions.
3. If the option's `type_name` is `"Path"`, normal file/directory
   completions SHOULD follow the hint (via shell `ValueHint`).
4. If no concrete values can be suggested (e.g., a freeform `int` or
   `String`), the hint alone is sufficient — the user sees what is
   expected and types a value manually.
5. The hint is advisory only — it MUST NOT prevent the user from typing
   any value they choose.

### REQ-DOC-05: Universal `help` subcommand and `--help` flag

Every command and subcommand at every level of the CLI tree MUST support
both a `help` subcommand and a `--help` flag. Both MUST produce identical
output for the same command.

This applies universally:

- Root-level commands: `veks import help` and `veks import --help`
- Pipeline groups: `veks pipeline compute help` and `veks pipeline compute --help`
- Pipeline commands: `veks pipeline compute knn help` and `veks pipeline compute knn --help`
- Top-level: `veks help` and `veks --help`

The `help` subcommand exists so that users who think in terms of
`<command> help` get the same result as users who think in terms of
`<command> --help`. Neither form should be privileged over the other.

Documentation is surfaced through these additional paths:

1. **`veks help <command-path>`** — Renders the full markdown `body` to
   the terminal. Uses a lightweight markdown renderer for syntax
   highlighting and formatting.

2. **`veks help --list`** — Lists all commands with their summaries,
   grouped by category.

3. **`veks help --markdown <command-path>`** — Emits raw markdown to
   stdout for piping to documentation generators.

### REQ-DOC-06: Documentation is code

The `command_doc()` method is part of the trait — it is compiled and
shipped with the binary. There are no external documentation files to
lose, no markdown files to forget to update. When a command's options
change, the documentation is updated in the same commit.

To enforce consistency:

1. **Test: summary is non-empty** — Every registered command MUST have a
   non-empty `summary`.

2. **Test: body mentions all options** — The `body` markdown MUST contain
   the name of every option returned by `describe_options()`. This is
   verified by a test that iterates all registered commands.

3. **Test: body mentions all resources** — If `describe_resources()` is
   non-empty, the `body` MUST contain a `## Resources` section mentioning
   each declared resource.

### REQ-DOC-07: Group-level documentation

Command groups (e.g., `analyze`, `compute`, `generate`, `import`, `slab`)
SHOULD also have group-level documentation that describes the group's
purpose and lists its commands with summaries. This is surfaced via:

```
$ veks help compute
# compute — Vector computation commands

Commands for computing distance-based results over vector datasets.

| Command | Summary |
|---------|---------|
| `knn` | Brute-force exact K-nearest-neighbor computation |
| `filtered-knn` | Brute-force filtered KNN with predicate-key pre-filtering |
| `sort` | Sort vectors by ordinal mapping |
```

Group documentation is registered separately from individual commands,
via a `GroupDoc` registry or a convention-based method.

## 7.3 Integration with Existing Infrastructure

### `build_pipeline_command()` in cli.rs

The existing function already builds the clap `Command` tree from
`describe_options()`. It MUST be extended to:

1. Set `.about(doc.summary)` on each subcommand (replacing the current
   `format!("{} {}", group, subname)` placeholder).
2. Set `.long_about(doc.body)` for `--help` rendering.
3. Populate completion descriptions from `doc.summary`.
4. For each option `Arg`, set `.help()` to include the description and
   type information so that value-position completions can surface the
   `# description (type: ..., default: ...)` hint per REQ-DOC-04a.

### `describe_options()` consistency

The `OptionDesc` struct already has a `description` field. The
`command_doc()` body's Options table SHOULD be generated from the same
`describe_options()` data to prevent drift. A helper function can render
the options table from the `Vec<OptionDesc>`:

```rust
fn render_options_table(options: &[OptionDesc]) -> String {
    // Generates the markdown table from OptionDesc entries
}
```

Commands can call this helper in their `command_doc()` implementation to
ensure the body always matches the declared options.

### `describe_resources()` consistency

Similarly, if `describe_resources()` returns entries, a helper renders
the Resources section:

```rust
fn render_resources_table(resources: &[ResourceDesc]) -> String {
    // Generates the markdown table from ResourceDesc entries
}
```

## 7.4 Example Implementation

```rust
impl CommandOp for ComputeKnn {
    fn command_path(&self) -> &str { "compute knn" }

    fn command_doc(&self) -> CommandDoc {
        let options_table = render_options_table(&self.describe_options());
        let resources_table = render_resources_table(&self.describe_resources());

        CommandDoc {
            summary: "Brute-force exact K-nearest-neighbor computation".into(),
            body: format!(r#"# compute knn

Brute-force exact K-nearest-neighbor computation.

## Description

Computes exact K-nearest neighbors for every query vector against the
full base vector corpus using brute-force distance computation. This is
the ground-truth reference implementation — results are guaranteed exact
(no approximation).

The computation is partitioned across base vector segments to bound peak
memory. Each partition's results are cached to `.cache/` so that
interrupted runs resume from the last completed partition.

SIMD-accelerated distance functions (AVX-512, AVX2, NEON) are used when
available. The metric must match the metric used by the ANN index being
benchmarked.

## Options

{options_table}

## Examples

### In a dataset.yaml pipeline

```yaml
- id: compute-knn
  run: compute knn
  after: [import-base, import-query]
  base: base_vectors.fvec
  query: query_vectors.fvec
  indices: neighbor_indices.ivec
  distances: neighbor_distances.fvec
  neighbors: 100
  metric: L2
```

### Direct CLI invocation

```sh
veks pipeline compute knn \
  --base base_vectors.fvec \
  --query query_vectors.fvec \
  --indices neighbor_indices.ivec \
  --distances neighbor_distances.fvec \
  --neighbors 100 \
  --metric L2 \
  --resources 'mem:25%-50%,threads:8'
```

## Resources

{resources_table}

## Notes

- For large datasets (>100M vectors), expect hours to days of computation.
  Use `--resources 'mem:...'` to prevent system saturation.
- Partition results are cached in `.cache/compute-knn.part_*.cache`.
  Do not delete these unless you want to recompute from scratch.
- The output ivec file uses 0-based ordinals into the base vector file.
"#),
        }
    }

    // ... other trait methods ...
}
```

## 7.5 Test Requirements

```rust
#[test]
fn test_all_commands_have_documentation() {
    let registry = CommandRegistry::with_builtins();
    for path in registry.command_paths() {
        let factory = registry.get(&path).unwrap();
        let cmd = factory();
        let doc = cmd.command_doc();

        assert!(
            !doc.summary.is_empty(),
            "Command '{}' has empty summary",
            path,
        );
        assert!(
            !doc.body.is_empty(),
            "Command '{}' has empty body",
            path,
        );

        // Verify all options are mentioned in the body
        for opt in cmd.describe_options() {
            assert!(
                doc.body.contains(&opt.name),
                "Command '{}' doc body does not mention option '{}'",
                path, opt.name,
            );
        }

        // Verify all resources are mentioned if declared
        let resources = cmd.describe_resources();
        if !resources.is_empty() {
            assert!(
                doc.body.contains("## Resources"),
                "Command '{}' declares resources but doc has no Resources section",
                path,
            );
            for res in &resources {
                assert!(
                    doc.body.contains(&res.name),
                    "Command '{}' doc body does not mention resource '{}'",
                    path, res.name,
                );
            }
        }
    }
}
```

## 7.6 Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-DOC-01 | **Done** | `command_doc()` trait method with `CommandDoc` struct |
| REQ-DOC-02 | **Done** | All commands follow the markdown structure convention |
| REQ-DOC-03 | **Done** | Completion summaries via clap `.about(doc.summary)` |
| REQ-DOC-04 | **Done** | Option help set from `describe_options()` descriptions |
| REQ-DOC-04a | **Done** | Type/default info appended to `Arg::help()` — shells render as description annotation, never inserted into command line. `--resources` and `--governor` use `ArgValueCandidates` for concrete value completions. |
| REQ-DOC-05 | **Done** | `--help` works everywhere via clap. `veks help` is a custom command that shows root-level commands and pipeline commands/groups. `veks help <root-cmd>` renders clap help for root commands; `veks help <pipeline-group>` shows group summary; `veks help <pipeline-cmd>` renders full markdown docs. Sub-commands with subcommands (datasets, analyze, pipeline) support clap's built-in `help` subcommand. |
| REQ-DOC-06 | **Done** | `test_all_commands_have_documentation` enforces consistency |
| REQ-DOC-07 | **Done** | `veks help <group>` shows group purpose + command table |
