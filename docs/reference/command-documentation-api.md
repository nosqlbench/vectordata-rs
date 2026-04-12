<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Command Documentation API Reference

Every pipeline command provides built-in markdown documentation through
the `CommandOp` trait. This reference describes the structs, trait
methods, helpers, and conventions that make up the documentation API.

## `CommandDoc` struct

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

The `summary` field is a single line of plain text -- no markdown
formatting, no trailing period. It appears in shell completion
suggestions and command listings.

The `body` field is a full markdown document following the structure
convention described below.

## `command_doc()` trait method

```rust
pub trait CommandOp: Send {
    /// Return markdown documentation for this command.
    fn command_doc(&self) -> CommandDoc;
}
```

The `command_doc()` method is defined on the `CommandOp` trait. The
default implementation generates documentation from the command's
`command_path()` and `describe_options()`, producing a basic body with
an Options table. Commands should override this default to add
descriptions, examples, resource tables, and notes.

Example override:

```rust
impl CommandOp for ComputeKnn {
    fn command_doc(&self) -> CommandDoc {
        let options_table = render_options_table(&self.describe_options());
        let resources_table = render_resources_table(&self.describe_resources());

        CommandDoc {
            summary: "Brute-force exact K-nearest-neighbor computation".into(),
            body: format!(
                "# compute knn\n\n\
                 Brute-force exact K-nearest-neighbor computation.\n\n\
                 ## Description\n\n\
                 Computes exact K-nearest neighbors for every query vector...\n\n\
                 ## Options\n\n{options_table}\n\n\
                 ## Examples\n\n...\n\n\
                 ## Resources\n\n{resources_table}\n\n\
                 ## Notes\n\n- ..."
            ),
        }
    }
}
```

## `render_options_table()` helper

```rust
fn render_options_table(options: &[OptionDesc]) -> String
```

Generates a markdown table from a `Vec<OptionDesc>`. The output follows
this format:

```markdown
| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `source` | Path | yes | -- | Input file path |
| `output` | Path | yes | -- | Output file path |
| `threads` | int | no | `num_cpus` | Worker thread count |
```

Use this helper in `command_doc()` implementations to keep the Options
section consistent with `describe_options()`. When options change, the
table updates automatically because it is generated from the same source
data.

## `describe_resources()` trait method

```rust
pub trait CommandOp: Send {
    /// Declare which resource types this command consumes.
    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]  // default: no resource declarations
    }
}
```

Returns the list of resources the command interacts with. Commands that
process arbitrarily large data must declare at minimum `mem`. The
governor uses this information for validation, completion filtering, and
enforcement.

## `ResourceDesc` struct

```rust
pub struct ResourceDesc {
    /// Resource name (e.g., "mem", "threads", "segmentsize").
    pub name: String,

    /// Human-readable description of how this command uses the resource.
    pub description: String,

    /// Whether the command can dynamically adjust this resource mid-execution.
    pub adjustable: bool,
}
```

The `name` field must match one of the recognized resource types (`mem`,
`threads`, `segments`, `segmentsize`, `iothreads`, `cache`,
`readahead`).

The `description` field explains how the command uses the resource, not
the resource itself. For example: "Buffers up to N vectors in memory per
thread" rather than "Memory budget".

The `adjustable` field indicates whether the command can respond to
mid-execution changes to this resource. If `true`, the governor may
adjust the effective value between checkpoint calls.

## Body structure convention

The `body` field follows a consistent section structure:

```markdown
# command-path

One-line summary (same as the `summary` field).

## Description

Multi-paragraph explanation of what the command does, when to use it,
and how it fits into a typical pipeline.

## Options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| ... | ... | ... | ... | ... |

## Examples

### Basic usage

```yaml
- run: command path
  source: input.fvec
  output: output.fvec
```

### With resource limits

```sh
veks pipeline command path --source input.fvec --resources 'mem:32GiB'
```

## Resources

This command declares: `mem`, `threads`, `readahead`

| Resource | Usage |
|----------|-------|
| `mem` | Buffers up to N vectors in memory per thread |
| `threads` | Parallel distance computation |
| `readahead` | Sequential prefetch window for mmap'd vectors |

## Notes

- Caveats, edge cases, or important behavioral details.
- References to related commands.
```

The `## Resources` section is only present when `describe_resources()`
returns a non-empty list. A helper function `render_resources_table()`
generates the table from `Vec<ResourceDesc>`:

```rust
fn render_resources_table(resources: &[ResourceDesc]) -> String
```

## CLI integration

The `summary` and `body` fields are wired into clap's help system
through `build_pipeline_command()`:

| `CommandDoc` field | clap method | Where it appears |
|--------------------|-------------|------------------|
| `summary` | `.about()` | `--help` short output, command listings |
| `body` | `.long_about()` | `--help` detailed output, `veks help <command>` |

Previously, `build_pipeline_command()` used placeholder strings like
`format!("{} {}", group, subname)` for the about text. With the
documentation API, it sets `.about(doc.summary)` and
`.long_about(doc.body)` from the command's `command_doc()`.

## Completion integration

The dynamic completion system surfaces summaries when listing available
commands. Summaries are prefixed with `#` to make them safe in shell
contexts -- if accidentally accepted as input, they parse as a comment:

```
$ veks pipeline compute <TAB>
filtered-knn    # Brute-force filtered KNN with predicate pre-filtering
knn             # Brute-force exact K-nearest-neighbor computation
sort            # Sort vectors by ordinal mapping
```

Option-level completions also show descriptions from
`describe_options()`:

```
$ veks pipeline compute knn --<TAB>
--base          # Path to base vector file (fvec/mvec)
--query         # Path to query vector file (fvec/mvec)
--neighbors     # Number of neighbors to compute (default: 100)
--resources     # Resource configuration (mem, threads, readahead)
```

## Test enforcement

The test `test_all_commands_have_documentation` verifies documentation
completeness for every registered command:

```rust
#[test]
fn test_all_commands_have_documentation() {
    let registry = CommandRegistry::with_builtins();
    for path in registry.command_paths() {
        let factory = registry.get(&path).unwrap();
        let cmd = factory();
        let doc = cmd.command_doc();

        // Summary must be non-empty
        assert!(!doc.summary.is_empty(),
            "Command '{}' has empty summary", path);

        // Body must be non-empty
        assert!(!doc.body.is_empty(),
            "Command '{}' has empty body", path);

        // Body must mention every option from describe_options()
        for opt in cmd.describe_options() {
            assert!(doc.body.contains(&opt.name),
                "Command '{}' doc body does not mention option '{}'",
                path, opt.name);
        }

        // Body must mention every resource from describe_resources()
        let resources = cmd.describe_resources();
        if !resources.is_empty() {
            assert!(doc.body.contains("## Resources"),
                "Command '{}' declares resources but doc has no Resources section",
                path);
            for res in &resources {
                assert!(doc.body.contains(&res.name),
                    "Command '{}' doc body does not mention resource '{}'",
                    path, res.name);
            }
        }
    }
}
```

This test enforces three invariants:

1. Every command has a non-empty `summary` and `body`.
2. The `body` mentions every option returned by `describe_options()`,
   preventing documentation drift when options are added or renamed.
3. If `describe_resources()` is non-empty, the `body` contains a
   `## Resources` section that mentions each declared resource.
