<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ANode Vernacular API Reference

Module: `vectordata::formats::anode_vernacular`

Source: `vectordata/src/formats/anode_vernacular.rs`

## Types

### `Vernacular`

```rust
pub enum Vernacular {
    Cddl,
    CddlValue,
    Sql,
    SqlSchema,
    Sqlite,
    SqliteSchema,
    Cql,
    CqlSchema,
    Json,
    Jsonl,
    Yaml,
    Readout,
    Display,
}
```

Enumerates all supported human-readable output formats.

#### `Vernacular::from_str`

```rust
pub fn from_str(s: &str) -> Option<Self>
```

Parse a format name (case-insensitive). Accepts hyphenated and concatenated
forms:

| Input strings              | Variant        |
|----------------------------|----------------|
| `"cddl"`                  | `Cddl`         |
| `"cddl-value"`, `"cddlvalue"` | `CddlValue` |
| `"sql"`                   | `Sql`           |
| `"sql-schema"`, `"sqlschema"` | `SqlSchema` |
| `"sqlite"`                | `Sqlite`        |
| `"sqlite-schema"`, `"sqliteschema"` | `SqliteSchema` |
| `"cql"`                   | `Cql`           |
| `"cql-schema"`, `"cqlschema"` | `CqlSchema` |
| `"json"`                  | `Json`          |
| `"jsonl"`                 | `Jsonl`         |
| `"yaml"`                  | `Yaml`          |
| `"readout"`               | `Readout`       |
| `"display"`               | `Display`       |

Returns `None` for unrecognized names.

## Functions

### `render`

```rust
pub fn render(node: &ANode, vernacular: Vernacular) -> String
```

Render an ANode to text in the given vernacular format. Always succeeds —
every ANode variant has a rendering for every vernacular.

**Dispatch for MNode:**

| Vernacular      | Delegates to                            |
|-----------------|-----------------------------------------|
| Cddl            | `mnode::vernacular::to_cddl`            |
| CddlValue       | Internal CDDL value renderer            |
| Sql, Sqlite      | `mnode::vernacular::to_sql`             |
| SqlSchema, SqliteSchema | `mnode::vernacular::to_sql_schema` |
| Cql              | `mnode::vernacular::to_cql`             |
| CqlSchema        | `mnode::vernacular::to_cql_schema`      |
| Json             | Internal JSON renderer (pretty)         |
| Jsonl            | Internal JSON renderer (compact)        |
| Yaml             | Internal YAML renderer                  |
| Readout          | Internal readout renderer               |
| Display          | `format!("{}", mnode)`                  |

**Dispatch for PNode:**

| Vernacular      | Delegates to                            |
|-----------------|-----------------------------------------|
| Cddl, CddlValue | `pnode::vernacular::to_cddl`           |
| Sql, Sqlite, SqlSchema, SqliteSchema | `pnode::vernacular::to_sql` |
| Cql, CqlSchema   | `pnode::vernacular::to_cql`            |
| Json             | Internal JSON renderer (pretty)         |
| Jsonl            | Internal JSON renderer (compact)        |
| Yaml             | Internal YAML renderer                  |
| Readout          | Internal readout renderer               |
| Display          | `format!("{}", pnode)`                  |

### `parse`

```rust
pub fn parse(text: &str, vernacular: Vernacular) -> Result<ANode, String>
```

Parse text in the given vernacular into an ANode.

**Supported vernaculars for parsing:**

| Vernacular | Parser behavior                                          |
|------------|----------------------------------------------------------|
| Json, Jsonl | Parses JSON object → MNode. Requires top-level object.  |
| Sql, Sqlite | Parses `(val, val, ...)` VALUES → MNode with `col_N` names. |
| Cql        | Parses `(val, val, ...)` VALUES → MNode with `col_N` names. |
| Cddl       | Parses `{ key : type, ... }` → MNode with typed Null placeholders. |
| Yaml       | Parses `key: value` lines → MNode.                      |
| Readout    | Parses `key : value` lines → MNode.                     |

**Unsupported:** CddlValue, SqlSchema, SqliteSchema, CqlSchema, Display.
Returns `Err("... not yet supported by ANode encoding")`.

**Errors:** Returns `Err` with a descriptive message for parse failures or
unsupported formats.
