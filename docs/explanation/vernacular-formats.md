<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Vernacular Formats

A *vernacular* is a human-readable text representation of an MNode or PNode
record. The term reflects that each format speaks the "native language" of a
particular domain: SQL for relational databases, CQL for Cassandra, CDDL for
schema definitions, and so on.

## Available vernaculars

| Vernacular     | Direction     | MNode rendering                        | PNode rendering              |
|----------------|---------------|----------------------------------------|------------------------------|
| `cddl`         | render + parse | CDDL type group `{ name : tstr, ... }`| CDDL structure               |
| `cddl-value`   | render only   | CDDL with literal values               | Same as cddl                 |
| `sql`           | render + parse | SQL VALUES tuple `('val', 42, TRUE)`   | SQL WHERE clause             |
| `sql-schema`    | render only   | SQL column definitions                 | SQL WHERE clause             |
| `sqlite`        | render + parse | Delegates to sql                       | Delegates to sql             |
| `sqlite-schema` | render only  | Delegates to sql-schema                | Delegates to sql             |
| `cql`           | render + parse | CQL VALUES tuple `('val', 42, true)`   | CQL WHERE clause             |
| `cql-schema`    | render only   | CQL column definitions                 | CQL WHERE clause             |
| `json`          | render + parse | Pretty-printed JSON object             | JSON predicate tree          |
| `jsonl`         | render + parse | Compact single-line JSON               | Compact JSON predicate tree  |
| `yaml`          | render + parse | YAML key-value document                | YAML tree structure          |
| `readout`       | render + parse | Tab-indented, colon-aligned display    | Indented infix notation      |
| `display`       | render only   | Rust `Display` trait output            | Rust `Display` trait output  |

## Render vs parse

Most vernaculars support both directions:

- **Render**: ANode → text (always supported for all vernaculars)
- **Parse**: text → ANode (supported for json, jsonl, sql, sqlite, cql, cddl,
  yaml, readout)

Schema-only formats (`sql-schema`, `sqlite-schema`, `cql-schema`) and the
`display` format do not support parsing. Attempting to parse these returns an
error message indicating the format is not yet supported.

## MNode vs PNode handling

Each vernacular may render MNode and PNode records differently:

- **SQL**: MNode renders as a VALUES tuple; PNode renders as a WHERE clause
- **JSON**: MNode renders as a flat object; PNode renders as a nested tree with
  `type`, `field`, `op`, and `value`/`values` keys
- **CDDL**: MNode renders as a type group; PNode renders as a structural
  definition with operator names

When a vernacular has no PNode-specific rendering, it falls back to the closest
equivalent. For example, `sql-schema` applied to a PNode produces a SQL WHERE
clause (same as `sql`).

## Parse type inference

When parsing, the codec infers MValue types from the text representation:

| Input                     | SQL/CQL inference | JSON inference | YAML inference |
|---------------------------|-------------------|----------------|----------------|
| `'text'`                  | Text              | —              | —              |
| `"text"`                  | —                 | Text           | Text           |
| `42`                      | Int               | Int            | Int            |
| `3.14`                    | Float             | Float          | Float          |
| `TRUE` / `true`           | Bool              | Bool           | Bool           |
| `NULL` / `null`           | Null              | Null           | Null           |
| `[1, 2]`                  | —                 | List           | —              |
| `{"k": "v"}`              | —                 | Map            | —              |

SQL VALUES parsing assigns auto-generated field names (`col_0`, `col_1`, ...)
since the VALUES tuple does not carry column names.
