// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Vernacular codec — bi-directional conversion between ANode and
//! human-readable text formats.
//!
//! This is Stage 2 of the two-stage codec pipeline:
//! ```text
//! bytes <-> [Stage 1: ANode binary codec] <-> ANode <-> [Stage 2: Vernacular codec] <-> text
//! ```

use super::anode::ANode;
use super::mnode::{self, MNode, MValue};
use super::pnode;

/// Human-readable vernacular format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vernacular {
    /// CDDL schema types.
    Cddl,
    /// CDDL with actual values (MNode only).
    CddlValue,
    /// SQL INSERT VALUES fragment.
    Sql,
    /// SQL CREATE TABLE columns.
    SqlSchema,
    /// SQLite dialect (delegates to SQL).
    Sqlite,
    /// SQLite CREATE TABLE columns (delegates to SQL).
    SqliteSchema,
    /// CQL VALUES fragment.
    Cql,
    /// CQL column definitions.
    CqlSchema,
    /// Pretty-printed JSON.
    Json,
    /// Single-line JSON.
    Jsonl,
    /// YAML.
    Yaml,
    /// Tab-indented, colon-aligned field display.
    Readout,
    /// Rust Display trait output.
    Display,
}

impl Vernacular {
    /// Parse a vernacular name from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cddl" => Some(Self::Cddl),
            "cddl-value" | "cddlvalue" => Some(Self::CddlValue),
            "sql" => Some(Self::Sql),
            "sql-schema" | "sqlschema" => Some(Self::SqlSchema),
            "sqlite" => Some(Self::Sqlite),
            "sqlite-schema" | "sqliteschema" => Some(Self::SqliteSchema),
            "cql" => Some(Self::Cql),
            "cql-schema" | "cqlschema" => Some(Self::CqlSchema),
            "json" => Some(Self::Json),
            "jsonl" => Some(Self::Jsonl),
            "yaml" => Some(Self::Yaml),
            "readout" => Some(Self::Readout),
            "display" => Some(Self::Display),
            _ => None,
        }
    }
}

/// Render an ANode to text in the given vernacular.
pub fn render(node: &ANode, vernacular: Vernacular) -> String {
    match node {
        ANode::MNode(m) => render_mnode(m, vernacular),
        ANode::PNode(p) => render_pnode(p, vernacular),
    }
}

fn render_mnode(m: &MNode, vernacular: Vernacular) -> String {
    match vernacular {
        Vernacular::Cddl => mnode::vernacular::to_cddl(m),
        Vernacular::CddlValue => render_mnode_cddl_value(m),
        Vernacular::Sql | Vernacular::Sqlite => mnode::vernacular::to_sql(m),
        Vernacular::SqlSchema | Vernacular::SqliteSchema => mnode::vernacular::to_sql_schema(m),
        Vernacular::Cql => mnode::vernacular::to_cql(m),
        Vernacular::CqlSchema => mnode::vernacular::to_cql_schema(m),
        Vernacular::Json => render_mnode_json(m, true),
        Vernacular::Jsonl => render_mnode_json(m, false),
        Vernacular::Yaml => render_mnode_yaml(m),
        Vernacular::Readout => render_mnode_readout(m, 0),
        Vernacular::Display => format!("{}", m),
    }
}

fn render_pnode(p: &pnode::PNode, vernacular: Vernacular) -> String {
    match vernacular {
        Vernacular::Cddl | Vernacular::CddlValue => pnode::vernacular::to_cddl(p),
        Vernacular::Sql | Vernacular::Sqlite
        | Vernacular::SqlSchema | Vernacular::SqliteSchema => pnode::vernacular::to_sql(p),
        Vernacular::Cql | Vernacular::CqlSchema => pnode::vernacular::to_cql(p),
        Vernacular::Json => render_pnode_json(p, true),
        Vernacular::Jsonl => render_pnode_json(p, false),
        Vernacular::Yaml => render_pnode_yaml(p, 0),
        Vernacular::Readout => render_pnode_readout(p, 0),
        Vernacular::Display => format!("{}", p),
    }
}

// -- CDDL Value ---------------------------------------------------------------

fn render_mnode_cddl_value(m: &MNode) -> String {
    let mut lines = Vec::with_capacity(m.fields.len());
    for (name, value) in &m.fields {
        lines.push(format!("  {} : {}", name, mnode::vernacular::to_cddl_value(value)));
    }
    format!("{{\n{}\n}}", lines.join(",\n"))
}

// -- JSON ---------------------------------------------------------------------

fn render_mnode_json(m: &MNode, pretty: bool) -> String {
    let obj = mvalue_map_to_json(m);
    if pretty {
        format_json_pretty(&obj, 0)
    } else {
        format_json_compact(&obj)
    }
}

fn render_pnode_json(p: &pnode::PNode, pretty: bool) -> String {
    let obj = pnode_to_json(p);
    if pretty {
        format_json_pretty(&obj, 0)
    } else {
        format_json_compact(&obj)
    }
}

/// Simple JSON value type for rendering without serde dependency.
enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

fn mvalue_to_json(v: &MValue) -> JsonValue {
    match v {
        MValue::Text(s) | MValue::EnumStr(s) | MValue::Ascii(s)
        | MValue::Date(s) | MValue::Time(s) | MValue::DateTime(s) => {
            JsonValue::Str(s.clone())
        }
        MValue::Int(n) | MValue::Millis(n) => JsonValue::Int(*n),
        MValue::Int32(n) | MValue::EnumOrd(n) => JsonValue::Int(*n as i64),
        MValue::Short(n) => JsonValue::Int(*n as i64),
        MValue::Float(n) => JsonValue::Float(*n),
        MValue::Float32(n) => JsonValue::Float(*n as f64),
        MValue::Half(n) => JsonValue::Int(*n as i64),
        MValue::Bool(b) => JsonValue::Bool(*b),
        MValue::Null => JsonValue::Null,
        MValue::Bytes(b) => {
            JsonValue::Str(b.iter().map(|byte| format!("{:02x}", byte)).collect())
        }
        MValue::Nanos { epoch_seconds, nano_adjust } => {
            JsonValue::Object(vec![
                ("epoch_seconds".into(), JsonValue::Int(*epoch_seconds)),
                ("nano_adjust".into(), JsonValue::Int(*nano_adjust as i64)),
            ])
        }
        MValue::UuidV1(b) | MValue::UuidV7(b) => {
            JsonValue::Str(format!(
                "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
            ))
        }
        MValue::Ulid(b) => {
            JsonValue::Str(b.iter().map(|byte| format!("{:02x}", byte)).collect())
        }
        MValue::List(items) | MValue::Set(items) => {
            JsonValue::Array(items.iter().map(mvalue_to_json).collect())
        }
        MValue::Array(_, items) => {
            JsonValue::Array(items.iter().map(mvalue_to_json).collect())
        }
        MValue::Map(node) => mvalue_map_to_json(node),
        MValue::TypedMap(entries) => {
            JsonValue::Object(entries.iter().map(|(k, v)| {
                (format!("{}", k), mvalue_to_json(v))
            }).collect())
        }
    }
}

fn mvalue_map_to_json(m: &MNode) -> JsonValue {
    JsonValue::Object(
        m.fields.iter().map(|(k, v)| (k.clone(), mvalue_to_json(v))).collect()
    )
}

fn comparand_to_json(c: &pnode::Comparand) -> JsonValue {
    match c {
        pnode::Comparand::Int(v) => JsonValue::Int(*v),
        pnode::Comparand::Float(v) => JsonValue::Float(*v),
        pnode::Comparand::Text(s) => JsonValue::Str(s.clone()),
        pnode::Comparand::Bool(b) => JsonValue::Bool(*b),
        pnode::Comparand::Bytes(b) => JsonValue::Str(
            b.iter().map(|x| format!("{:02x}", x)).collect(),
        ),
        pnode::Comparand::Null => JsonValue::Null,
    }
}

fn pnode_to_json(p: &pnode::PNode) -> JsonValue {
    match p {
        pnode::PNode::Predicate(pred) => {
            let field_val = match &pred.field {
                pnode::FieldRef::Index(i) => JsonValue::Str(format!("field_{}", i)),
                pnode::FieldRef::Named(s) => JsonValue::Str(s.clone()),
            };
            let mut obj = vec![
                ("type".into(), JsonValue::Str("predicate".into())),
                ("field".into(), field_val),
                ("op".into(), JsonValue::Str(pred.op.symbol().into())),
            ];
            if pred.comparands.len() == 1 {
                obj.push(("value".into(), comparand_to_json(&pred.comparands[0])));
            } else {
                obj.push(("values".into(), JsonValue::Array(
                    pred.comparands.iter().map(comparand_to_json).collect()
                )));
            }
            JsonValue::Object(obj)
        }
        pnode::PNode::Conjugate(conj) => {
            let op = match conj.conjugate_type {
                pnode::ConjugateType::And => "and",
                pnode::ConjugateType::Or => "or",
                _ => "unknown",
            };
            JsonValue::Object(vec![
                ("type".into(), JsonValue::Str(op.into())),
                ("children".into(), JsonValue::Array(
                    conj.children.iter().map(pnode_to_json).collect()
                )),
            ])
        }
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\x20' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn format_json_compact(val: &JsonValue) -> String {
    match val {
        JsonValue::Null => "null".into(),
        JsonValue::Bool(b) => if *b { "true".into() } else { "false".into() },
        JsonValue::Int(n) => n.to_string(),
        JsonValue::Float(n) => {
            let s = n.to_string();
            if s.contains('.') { s } else { format!("{}.0", s) }
        }
        JsonValue::Str(s) => format!("\"{}\"", json_escape(s)),
        JsonValue::Array(items) => {
            let inner: Vec<String> = items.iter().map(format_json_compact).collect();
            format!("[{}]", inner.join(","))
        }
        JsonValue::Object(entries) => {
            let inner: Vec<String> = entries.iter()
                .map(|(k, v)| format!("\"{}\":{}", json_escape(k), format_json_compact(v)))
                .collect();
            format!("{{{}}}", inner.join(","))
        }
    }
}

fn format_json_pretty(val: &JsonValue, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    let pad_inner = "  ".repeat(indent + 1);
    match val {
        JsonValue::Null | JsonValue::Bool(_) | JsonValue::Int(_)
        | JsonValue::Float(_) | JsonValue::Str(_) => format_json_compact(val),
        JsonValue::Array(items) if items.is_empty() => "[]".into(),
        JsonValue::Array(items) => {
            let inner: Vec<String> = items.iter()
                .map(|v| format!("{}{}", pad_inner, format_json_pretty(v, indent + 1)))
                .collect();
            format!("[\n{}\n{}]", inner.join(",\n"), pad)
        }
        JsonValue::Object(entries) if entries.is_empty() => "{}".into(),
        JsonValue::Object(entries) => {
            let inner: Vec<String> = entries.iter()
                .map(|(k, v)| format!("{}\"{}\": {}", pad_inner, json_escape(k), format_json_pretty(v, indent + 1)))
                .collect();
            format!("{{\n{}\n{}}}", inner.join(",\n"), pad)
        }
    }
}

// -- YAML ---------------------------------------------------------------------

fn render_mnode_yaml(m: &MNode) -> String {
    let mut lines = Vec::new();
    for (name, value) in &m.fields {
        render_yaml_field(&mut lines, name, value, 0);
    }
    lines.join("\n")
}

fn render_yaml_field(lines: &mut Vec<String>, name: &str, value: &MValue, indent: usize) {
    let pad = "  ".repeat(indent);
    match value {
        MValue::Map(node) => {
            lines.push(format!("{}{}:", pad, name));
            for (k, v) in &node.fields {
                render_yaml_field(lines, k, v, indent + 1);
            }
        }
        MValue::List(items) | MValue::Set(items) | MValue::Array(_, items) => {
            lines.push(format!("{}{}:", pad, name));
            for item in items {
                lines.push(format!("{}- {}", "  ".repeat(indent + 1), yaml_scalar(item)));
            }
        }
        MValue::TypedMap(entries) => {
            lines.push(format!("{}{}:", pad, name));
            for (k, v) in entries {
                lines.push(format!("{}{}: {}", "  ".repeat(indent + 1), yaml_scalar(k), yaml_scalar(v)));
            }
        }
        _ => {
            lines.push(format!("{}{}: {}", pad, name, yaml_scalar(value)));
        }
    }
}

fn yaml_scalar(v: &MValue) -> String {
    match v {
        MValue::Text(s) | MValue::EnumStr(s) | MValue::Ascii(s)
        | MValue::Date(s) | MValue::Time(s) | MValue::DateTime(s) => {
            if s.contains(':') || s.contains('#') || s.contains('\'')
                || s.contains('"') || s.starts_with(' ') || s.is_empty()
            {
                format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
            } else {
                s.clone()
            }
        }
        MValue::Int(n) | MValue::Millis(n) => n.to_string(),
        MValue::Int32(n) | MValue::EnumOrd(n) => n.to_string(),
        MValue::Short(n) => n.to_string(),
        MValue::Float(n) => n.to_string(),
        MValue::Float32(n) => n.to_string(),
        MValue::Half(n) => n.to_string(),
        MValue::Bool(b) => if *b { "true".into() } else { "false".into() },
        MValue::Null => "null".into(),
        MValue::Bytes(b) => format!("0x{}", b.iter().map(|byte| format!("{:02x}", byte)).collect::<String>()),
        MValue::Nanos { epoch_seconds, nano_adjust } => format!("{}.{}", epoch_seconds, nano_adjust),
        MValue::UuidV1(b) | MValue::UuidV7(b) => format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
        ),
        MValue::Ulid(b) => format!("0x{}", b.iter().map(|byte| format!("{:02x}", byte)).collect::<String>()),
        _ => format!("{}", v),
    }
}

/// Format a PNode comparand for YAML output.
fn yaml_comparand(c: &pnode::Comparand) -> String {
    match c {
        pnode::Comparand::Text(s) => {
            if s.contains(':') || s.contains('#') || s.contains('\'')
                || s.contains('"') || s.starts_with(' ') || s.is_empty()
            {
                format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
            } else {
                s.clone()
            }
        }
        _ => format!("{}", c),
    }
}

fn render_pnode_yaml(p: &pnode::PNode, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match p {
        pnode::PNode::Predicate(pred) => {
            let field = match &pred.field {
                pnode::FieldRef::Index(i) => format!("field_{}", i),
                pnode::FieldRef::Named(s) => s.clone(),
            };
            if pred.comparands.len() == 1 {
                format!("{}field: {}\n{}op: {}\n{}value: {}",
                    pad, field, pad, pred.op.symbol(), pad, yaml_comparand(&pred.comparands[0]))
            } else {
                let vals: Vec<String> = pred.comparands.iter().map(|v| format!("{}  - {}", pad, yaml_comparand(v))).collect();
                format!("{}field: {}\n{}op: {}\n{}values:\n{}",
                    pad, field, pad, pred.op.symbol(), pad, vals.join("\n"))
            }
        }
        pnode::PNode::Conjugate(conj) => {
            let op = match conj.conjugate_type {
                pnode::ConjugateType::And => "and",
                pnode::ConjugateType::Or => "or",
                _ => "unknown",
            };
            let children: Vec<String> = conj.children.iter()
                .map(|c| format!("{}-\n{}", "  ".repeat(indent + 1), render_pnode_yaml(c, indent + 2)))
                .collect();
            format!("{}type: {}\n{}children:\n{}", pad, op, pad, children.join("\n"))
        }
    }
}

// -- Readout ------------------------------------------------------------------

fn render_mnode_readout(m: &MNode, indent: usize) -> String {
    let pad = "\t".repeat(indent);
    let max_name_len = m.fields.keys().map(|k| k.len()).max().unwrap_or(0);
    let mut lines = Vec::new();
    for (name, value) in &m.fields {
        let padded_name = format!("{:<width$}", name, width = max_name_len);
        match value {
            MValue::Map(inner) => {
                lines.push(format!("{}{} :", pad, padded_name));
                lines.push(render_mnode_readout(inner, indent + 1));
            }
            MValue::List(items) | MValue::Set(items) | MValue::Array(_, items) => {
                lines.push(format!("{}{} :", pad, padded_name));
                for item in items {
                    lines.push(format!("{}\t- {}", pad, item));
                }
            }
            MValue::TypedMap(entries) => {
                lines.push(format!("{}{} :", pad, padded_name));
                for (k, v) in entries {
                    lines.push(format!("{}\t{} : {}", pad, k, v));
                }
            }
            _ => {
                lines.push(format!("{}{} : {}", pad, padded_name, value));
            }
        }
    }
    lines.join("\n")
}

fn render_pnode_readout(p: &pnode::PNode, indent: usize) -> String {
    let pad = "\t".repeat(indent);
    match p {
        pnode::PNode::Predicate(pred) => {
            let field = match &pred.field {
                pnode::FieldRef::Index(i) => format!("field_{}", i),
                pnode::FieldRef::Named(s) => s.clone(),
            };
            if pred.comparands.len() == 1 {
                format!("{}{} {} {}", pad, field, pred.op.symbol(), pred.comparands[0])
            } else {
                let vals: Vec<String> = pred.comparands.iter().map(|v| format!("{}", v)).collect();
                format!("{}{} {} ({})", pad, field, pred.op.symbol(), vals.join(", "))
            }
        }
        pnode::PNode::Conjugate(conj) => {
            let op = match conj.conjugate_type {
                pnode::ConjugateType::And => "AND",
                pnode::ConjugateType::Or => "OR",
                _ => "???",
            };
            let children: Vec<String> = conj.children.iter()
                .map(|c| render_pnode_readout(c, indent + 1))
                .collect();
            format!("{}{}:\n{}", pad, op, children.join(&format!("\n{}{}\n", pad, op)))
        }
    }
}

// -- Parse (reverse direction) ------------------------------------------------

/// Parse text in the given vernacular into an ANode.
///
/// Currently supports JSON, SQL VALUES, CQL VALUES, CDDL group, YAML subset,
/// and readout formats. Returns an error for unsupported or unrecognized input.
pub fn parse(text: &str, vernacular: Vernacular) -> Result<ANode, String> {
    match vernacular {
        Vernacular::Json | Vernacular::Jsonl => parse_json(text),
        Vernacular::Sql | Vernacular::Sqlite => parse_sql_values(text),
        Vernacular::Cql => parse_cql_values(text),
        Vernacular::Cddl => parse_cddl_group(text),
        Vernacular::Yaml => parse_yaml(text),
        Vernacular::Readout => parse_readout(text),
        _ => Err(format!("{:?} parse not yet supported by ANode encoding", vernacular)),
    }
}

fn parse_json(text: &str) -> Result<ANode, String> {
    let val: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| format!("JSON parse error: {}", e))?;
    match val {
        serde_json::Value::Object(map) => {
            let mut node = MNode::new();
            for (k, v) in map {
                node.insert(k, json_to_mvalue(&v));
            }
            Ok(ANode::MNode(node))
        }
        _ => Err("JSON input must be an object".into()),
    }
}

fn json_to_mvalue(v: &serde_json::Value) -> MValue {
    match v {
        serde_json::Value::Null => MValue::Null,
        serde_json::Value::Bool(b) => MValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                MValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                MValue::Float(f)
            } else {
                MValue::Null
            }
        }
        serde_json::Value::String(s) => MValue::Text(s.clone()),
        serde_json::Value::Array(arr) => {
            MValue::List(arr.iter().map(json_to_mvalue).collect())
        }
        serde_json::Value::Object(map) => {
            let mut node = MNode::new();
            for (k, v) in map {
                node.insert(k.clone(), json_to_mvalue(v));
            }
            MValue::Map(node)
        }
    }
}

fn parse_sql_values(text: &str) -> Result<ANode, String> {
    let trimmed = text.trim();
    let inner = if trimmed.starts_with('(') && trimmed.ends_with(')') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };
    let values = split_values(inner)?;
    let mut node = MNode::new();
    for (i, val) in values.iter().enumerate() {
        let name = format!("col_{}", i);
        node.insert(name, parse_sql_literal(val.trim()));
    }
    Ok(ANode::MNode(node))
}

fn parse_cql_values(text: &str) -> Result<ANode, String> {
    let trimmed = text.trim();
    let inner = if trimmed.starts_with('(') && trimmed.ends_with(')') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };
    let values = split_values(inner)?;
    let mut node = MNode::new();
    for (i, val) in values.iter().enumerate() {
        let name = format!("col_{}", i);
        node.insert(name, parse_cql_literal(val.trim()));
    }
    Ok(ANode::MNode(node))
}

fn split_values(s: &str) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut escape_next = false;
    let mut depth = 0;

    for c in s.chars() {
        if escape_next {
            current.push(c);
            escape_next = false;
            continue;
        }
        match c {
            '\'' if !in_string => {
                in_string = true;
                current.push(c);
            }
            '\'' if in_string => {
                current.push(c);
                in_string = false;
            }
            '(' if !in_string => {
                depth += 1;
                current.push(c);
            }
            ')' if !in_string => {
                depth -= 1;
                current.push(c);
            }
            ',' if !in_string && depth == 0 => {
                result.push(current.clone());
                current.clear();
            }
            '\\' if in_string => {
                escape_next = true;
                current.push(c);
            }
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    Ok(result)
}

fn parse_sql_literal(s: &str) -> MValue {
    let upper = s.to_uppercase();
    if upper == "NULL" {
        return MValue::Null;
    }
    if upper == "TRUE" {
        return MValue::Bool(true);
    }
    if upper == "FALSE" {
        return MValue::Bool(false);
    }
    if s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2 {
        let inner = &s[1..s.len() - 1];
        return MValue::Text(inner.replace("''", "'"));
    }
    if let Ok(i) = s.parse::<i64>() {
        return MValue::Int(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return MValue::Float(f);
    }
    MValue::Text(s.into())
}

fn parse_cql_literal(s: &str) -> MValue {
    let lower = s.to_lowercase();
    if lower == "null" {
        return MValue::Null;
    }
    if lower == "true" {
        return MValue::Bool(true);
    }
    if lower == "false" {
        return MValue::Bool(false);
    }
    if s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2 {
        let inner = &s[1..s.len() - 1];
        return MValue::Text(inner.replace("''", "'"));
    }
    if let Ok(i) = s.parse::<i64>() {
        return MValue::Int(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return MValue::Float(f);
    }
    MValue::Text(s.into())
}

fn parse_cddl_group(text: &str) -> Result<ANode, String> {
    let trimmed = text.trim();
    let inner = if trimmed.starts_with('{') && trimmed.ends_with('}') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };

    let mut node = MNode::new();
    for line in inner.split(',') {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(colon_pos) = line.find(':') {
            let key = line[..colon_pos].trim().trim_matches('"');
            let type_str = line[colon_pos + 1..].trim();
            let value = cddl_type_to_null_mvalue(type_str);
            node.insert(key.into(), value);
        }
    }
    Ok(ANode::MNode(node))
}

fn cddl_type_to_null_mvalue(type_str: &str) -> MValue {
    match type_str {
        "tstr" => MValue::Text(String::new()),
        "int" => MValue::Int(0),
        "float" | "float16" => MValue::Float(0.0),
        "bool" => MValue::Bool(false),
        "bstr" => MValue::Bytes(Vec::new()),
        "null" => MValue::Null,
        _ => MValue::Null,
    }
}

fn parse_yaml(text: &str) -> Result<ANode, String> {
    let mut node = MNode::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(colon_pos) = line.find(':') {
            let key = line[..colon_pos].trim();
            let val_str = line[colon_pos + 1..].trim();
            if val_str.is_empty() {
                node.insert(key.into(), MValue::Null);
            } else {
                node.insert(key.into(), parse_yaml_scalar(val_str));
            }
        }
    }
    Ok(ANode::MNode(node))
}

fn parse_yaml_scalar(s: &str) -> MValue {
    if s == "null" || s == "~" {
        return MValue::Null;
    }
    if s == "true" {
        return MValue::Bool(true);
    }
    if s == "false" {
        return MValue::Bool(false);
    }
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        return MValue::Text(s[1..s.len() - 1].into());
    }
    if let Ok(i) = s.parse::<i64>() {
        return MValue::Int(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return MValue::Float(f);
    }
    MValue::Text(s.into())
}

fn parse_readout(text: &str) -> Result<ANode, String> {
    let mut node = MNode::new();
    for line in text.lines() {
        let line = line.trim_start_matches('\t');
        if line.is_empty() {
            continue;
        }
        if let Some(colon_pos) = line.find(" : ") {
            let key = line[..colon_pos].trim();
            let val_str = line[colon_pos + 3..].trim();
            node.insert(key.into(), parse_readout_value(val_str));
        }
    }
    Ok(ANode::MNode(node))
}

fn parse_readout_value(s: &str) -> MValue {
    if s == "NULL" {
        return MValue::Null;
    }
    if s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2 {
        return MValue::Text(s[1..s.len() - 1].into());
    }
    if let Ok(i) = s.parse::<i64>() {
        return MValue::Int(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return MValue::Float(f);
    }
    if s == "true" {
        return MValue::Bool(true);
    }
    if s == "false" {
        return MValue::Bool(false);
    }
    MValue::Text(s.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::mnode::{MNode, MValue};
    use crate::formats::pnode::{Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode};
    use crate::formats::anode::ANode;

    fn sample_mnode() -> MNode {
        let mut node = MNode::new();
        node.insert("name".into(), MValue::Text("alice".into()));
        node.insert("age".into(), MValue::Int(30));
        node.insert("score".into(), MValue::Float(99.5));
        node.insert("active".into(), MValue::Bool(true));
        node.insert("empty".into(), MValue::Null);
        node
    }

    fn sample_pnode() -> PNode {
        PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("age".into()),
                    op: OpType::Gt,
                    comparands: vec![Comparand::Int(18)],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("status".into()),
                    op: OpType::In,
                    comparands: vec![Comparand::Int(1), Comparand::Int(2), Comparand::Int(3)],
                }),
            ],
        })
    }

    #[test]
    fn test_render_mnode_cddl() {
        let m = sample_mnode();
        let text = render(&ANode::MNode(m), Vernacular::Cddl);
        assert!(text.contains("name : tstr"));
        assert!(text.contains("age : int"));
    }

    #[test]
    fn test_render_mnode_sql() {
        let m = sample_mnode();
        let text = render(&ANode::MNode(m), Vernacular::Sql);
        assert!(text.contains("'alice'"));
        assert!(text.contains("30"));
        assert!(text.contains("TRUE"));
    }

    #[test]
    fn test_render_mnode_cql() {
        let m = sample_mnode();
        let text = render(&ANode::MNode(m), Vernacular::Cql);
        assert!(text.contains("'alice'"));
        assert!(text.contains("true"));
    }

    #[test]
    fn test_render_mnode_json_roundtrip() {
        let m = sample_mnode();
        let json_text = render(&ANode::MNode(m.clone()), Vernacular::Json);
        assert!(json_text.contains("\"name\""));
        assert!(json_text.contains("\"alice\""));

        let parsed = parse(&json_text, Vernacular::Json).unwrap();
        match parsed {
            ANode::MNode(parsed_m) => {
                assert_eq!(parsed_m.fields["name"], MValue::Text("alice".into()));
                assert_eq!(parsed_m.fields["age"], MValue::Int(30));
                assert_eq!(parsed_m.fields["active"], MValue::Bool(true));
                assert_eq!(parsed_m.fields["empty"], MValue::Null);
            }
            _ => panic!("expected MNode"),
        }
    }

    #[test]
    fn test_render_mnode_jsonl() {
        let m = sample_mnode();
        let text = render(&ANode::MNode(m), Vernacular::Jsonl);
        assert!(!text.contains('\n'));
        assert!(text.contains("\"name\":\"alice\""));
    }

    #[test]
    fn test_render_mnode_yaml() {
        let m = sample_mnode();
        let text = render(&ANode::MNode(m), Vernacular::Yaml);
        assert!(text.contains("name: alice"));
        assert!(text.contains("age: 30"));
    }

    #[test]
    fn test_render_mnode_readout() {
        let m = sample_mnode();
        let text = render(&ANode::MNode(m), Vernacular::Readout);
        assert!(text.contains("name"));
        assert!(text.contains("'alice'"));
        assert!(text.contains("30"));
    }

    #[test]
    fn test_render_mnode_display() {
        let m = sample_mnode();
        let text = render(&ANode::MNode(m), Vernacular::Display);
        assert!(text.contains("name: 'alice'"));
    }

    #[test]
    fn test_render_pnode_sql() {
        let p = sample_pnode();
        let text = render(&ANode::PNode(p), Vernacular::Sql);
        assert!(text.contains("age > 18"));
        assert!(text.contains("AND"));
    }

    #[test]
    fn test_render_pnode_json() {
        let p = sample_pnode();
        let text = render(&ANode::PNode(p), Vernacular::Json);
        assert!(text.contains("\"and\""));
        assert!(text.contains("\"age\""));
    }

    #[test]
    fn test_parse_sql_values() {
        let parsed = parse("('alice', 42, TRUE, NULL)", Vernacular::Sql).unwrap();
        match parsed {
            ANode::MNode(m) => {
                assert_eq!(m.fields["col_0"], MValue::Text("alice".into()));
                assert_eq!(m.fields["col_1"], MValue::Int(42));
                assert_eq!(m.fields["col_2"], MValue::Bool(true));
                assert_eq!(m.fields["col_3"], MValue::Null);
            }
            _ => panic!("expected MNode"),
        }
    }

    #[test]
    fn test_parse_cql_values() {
        let parsed = parse("('bob', 99, true, null)", Vernacular::Cql).unwrap();
        match parsed {
            ANode::MNode(m) => {
                assert_eq!(m.fields["col_0"], MValue::Text("bob".into()));
                assert_eq!(m.fields["col_1"], MValue::Int(99));
                assert_eq!(m.fields["col_2"], MValue::Bool(true));
                assert_eq!(m.fields["col_3"], MValue::Null);
            }
            _ => panic!("expected MNode"),
        }
    }

    #[test]
    fn test_parse_cddl_group() {
        let parsed = parse("{ name : tstr, count : int }", Vernacular::Cddl).unwrap();
        match parsed {
            ANode::MNode(m) => {
                assert_eq!(m.fields.len(), 2);
                assert!(m.fields.contains_key("name"));
                assert!(m.fields.contains_key("count"));
            }
            _ => panic!("expected MNode"),
        }
    }

    #[test]
    fn test_parse_yaml() {
        let parsed = parse("name: alice\nage: 30\nactive: true", Vernacular::Yaml).unwrap();
        match parsed {
            ANode::MNode(m) => {
                assert_eq!(m.fields["name"], MValue::Text("alice".into()));
                assert_eq!(m.fields["age"], MValue::Int(30));
                assert_eq!(m.fields["active"], MValue::Bool(true));
            }
            _ => panic!("expected MNode"),
        }
    }

    #[test]
    fn test_parse_readout() {
        let parsed = parse("name : 'alice'\nage  : 30", Vernacular::Readout).unwrap();
        match parsed {
            ANode::MNode(m) => {
                assert_eq!(m.fields["name"], MValue::Text("alice".into()));
                assert_eq!(m.fields["age"], MValue::Int(30));
            }
            _ => panic!("expected MNode"),
        }
    }

    #[test]
    fn test_unsupported_parse() {
        assert!(parse("foo", Vernacular::Display).is_err());
        assert!(parse("foo", Vernacular::SqlSchema).is_err());
    }

    #[test]
    fn test_vernacular_from_str() {
        assert_eq!(Vernacular::from_str("json"), Some(Vernacular::Json));
        assert_eq!(Vernacular::from_str("SQL"), Some(Vernacular::Sql));
        assert_eq!(Vernacular::from_str("cddl"), Some(Vernacular::Cddl));
        assert_eq!(Vernacular::from_str("unknown"), None);
    }
}
