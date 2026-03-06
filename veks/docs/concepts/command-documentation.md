<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Command Documentation

Every pipeline command carries its own markdown documentation, compiled
directly into the binary. There are no external doc files to lose or fall out
of sync.

## The CommandDoc struct

Each command implements the `command_doc()` trait method, which returns a
`CommandDoc` with two fields:

| Field     | Purpose                                              |
|-----------|------------------------------------------------------|
| `summary` | One-line description, used for shell completions      |
| `body`    | Full markdown, rendered by `--help`                   |

Completion summaries are prefixed with `#` so that shells treat them as
comments rather than executable text.

## Structured body format

The `body` field follows a consistent structure across all commands:

1. **Description** — What the command does and when to use it.
2. **Options** — A markdown table of all accepted options.
3. **Examples** — One or more pipeline YAML snippets.
4. **Resources** — What system resources the command consumes.
5. **Notes** — Edge cases, caveats, or related commands.

## Generated sections

Two sections are generated from trait methods rather than hand-written:

- The **Options** table is produced by `render_options_table()` from the
  command's `describe_options()` output. This ensures the documented options
  always match the code.
- The **Resources** section is produced from `describe_resources()`, keeping
  resource declarations and documentation in lockstep.

The `render_options_table()` helper generates consistent markdown tables with
columns for name, type, default, and description.

## Test enforcement

Tests enforce documentation completeness:

- Every command must return a non-empty `summary`.
- The `body` must mention every option declared by `describe_options()`.

This prevents undocumented options from shipping and catches documentation
regressions at compile time.
