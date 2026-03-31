// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Vernacular adapters for MNode — human-readable representations in SQL, CQL,
//! and CDDL syntax.
//!
//! These adapters are used for troubleshooting, visualization, and schema
//! documentation. They convert MNode records and their type tags into the
//! native syntax of each query language or schema language.

use super::{MNode, MValue, TypeTag, format_uuid, hex_encode};

// -- SQL vernacular -----------------------------------------------------------

/// Render an MNode as a SQL-style `INSERT VALUES (...)` fragment
pub fn to_sql(node: &MNode) -> String {
    let mut parts = Vec::with_capacity(node.fields.len());
    for (_name, value) in &node.fields {
        parts.push(sql_value(value));
    }
    format!("({})", parts.join(", "))
}

/// Render an MNode's schema as SQL `CREATE TABLE` column definitions
pub fn to_sql_schema(node: &MNode) -> String {
    let mut cols = Vec::with_capacity(node.fields.len());
    for (name, value) in &node.fields {
        cols.push(format!("  {} {}", name, sql_type(&value.tag())));
    }
    format!("(\n{}\n)", cols.join(",\n"))
}

fn sql_value(value: &MValue) -> String {
    match value {
        MValue::Text(s) | MValue::EnumStr(s) | MValue::Ascii(s) => {
            format!("'{}'", s.replace('\'', "''"))
        }
        MValue::Int(v) | MValue::Millis(v) => format!("{}", v),
        MValue::Float(v) => format!("{}", v),
        MValue::Bool(v) => if *v { "TRUE".into() } else { "FALSE".into() },
        MValue::Bytes(v) => format!("X'{}'", hex_encode(v)),
        MValue::Null => "NULL".into(),
        MValue::EnumOrd(v) | MValue::Int32(v) => format!("{}", v),
        MValue::Short(v) => format!("{}", v),
        MValue::Float32(v) => format!("{}", v),
        MValue::Half(v) => format!("{}", v),
        MValue::Nanos { epoch_seconds, nano_adjust } => {
            format!("TIMESTAMP '{}.{}'", epoch_seconds, nano_adjust)
        }
        MValue::Date(s) => format!("DATE '{}'", s),
        MValue::Time(s) => format!("TIME '{}'", s),
        MValue::DateTime(s) => format!("TIMESTAMP '{}'", s),
        MValue::UuidV1(b) | MValue::UuidV7(b) => format!("'{}'", format_uuid(b)),
        MValue::Ulid(b) => format!("'{}'", hex_encode(b)),
        MValue::List(items) | MValue::Set(items) => {
            let inner: Vec<String> = items.iter().map(sql_value).collect();
            format!("ARRAY[{}]", inner.join(", "))
        }
        MValue::Array(_, items) => {
            let inner: Vec<String> = items.iter().map(sql_value).collect();
            format!("ARRAY[{}]", inner.join(", "))
        }
        MValue::Map(node) => to_sql(node),
        MValue::TypedMap(entries) => {
            let inner: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{} => {}", sql_value(k), sql_value(v)))
                .collect();
            format!("MAP({})", inner.join(", "))
        }
    }
}

fn sql_type(tag: &TypeTag) -> &'static str {
    match tag {
        TypeTag::Text | TypeTag::TextValidated | TypeTag::EnumStr => "TEXT",
        TypeTag::Ascii => "VARCHAR",
        TypeTag::Int | TypeTag::Millis => "BIGINT",
        TypeTag::Int32 | TypeTag::EnumOrd => "INT",
        TypeTag::Short => "SMALLINT",
        TypeTag::Float => "DOUBLE PRECISION",
        TypeTag::Float32 => "REAL",
        TypeTag::Half => "SMALLINT",
        TypeTag::Bool => "BOOLEAN",
        TypeTag::Bytes | TypeTag::Decimal | TypeTag::Varint => "BLOB",
        TypeTag::Null => "TEXT",
        TypeTag::Date => "DATE",
        TypeTag::Time => "TIME",
        TypeTag::DateTime | TypeTag::Nanos => "TIMESTAMP",
        TypeTag::UuidV1 | TypeTag::UuidV7 => "UUID",
        TypeTag::Ulid => "CHAR(26)",
        TypeTag::List | TypeTag::Array | TypeTag::Set => "TEXT[]",
        TypeTag::Map | TypeTag::TypedMap => "JSONB",
    }
}

// -- CQL vernacular -----------------------------------------------------------

/// Render an MNode as a CQL-style `VALUES (...)` fragment
pub fn to_cql(node: &MNode) -> String {
    let mut parts = Vec::with_capacity(node.fields.len());
    for (_name, value) in &node.fields {
        parts.push(cql_value(value));
    }
    format!("({})", parts.join(", "))
}

/// Render an MNode's schema as CQL column definitions
pub fn to_cql_schema(node: &MNode) -> String {
    let mut cols = Vec::with_capacity(node.fields.len());
    for (name, value) in &node.fields {
        cols.push(format!("  {} {}", name, cql_type(&value.tag())));
    }
    format!("(\n{}\n)", cols.join(",\n"))
}

fn cql_value(value: &MValue) -> String {
    match value {
        MValue::Text(s) | MValue::EnumStr(s) | MValue::Ascii(s) => {
            format!("'{}'", s.replace('\'', "''"))
        }
        MValue::Int(v) | MValue::Millis(v) => format!("{}", v),
        MValue::Float(v) => format!("{}", v),
        MValue::Bool(v) => if *v { "true".into() } else { "false".into() },
        MValue::Bytes(v) => format!("0x{}", hex_encode(v)),
        MValue::Null => "null".into(),
        MValue::EnumOrd(v) | MValue::Int32(v) => format!("{}", v),
        MValue::Short(v) => format!("{}", v),
        MValue::Float32(v) => format!("{}", v),
        MValue::Half(v) => format!("{}", v),
        MValue::Nanos { epoch_seconds, nano_adjust } => {
            format!("'{}.{}'", epoch_seconds, nano_adjust)
        }
        MValue::Date(s) | MValue::Time(s) | MValue::DateTime(s) => {
            format!("'{}'", s)
        }
        MValue::UuidV1(b) | MValue::UuidV7(b) => format_uuid(b),
        MValue::Ulid(b) => format!("0x{}", hex_encode(b)),
        MValue::List(items) => {
            let inner: Vec<String> = items.iter().map(cql_value).collect();
            format!("[{}]", inner.join(", "))
        }
        MValue::Set(items) => {
            let inner: Vec<String> = items.iter().map(cql_value).collect();
            format!("{{{}}}", inner.join(", "))
        }
        MValue::Array(_, items) => {
            let inner: Vec<String> = items.iter().map(cql_value).collect();
            format!("[{}]", inner.join(", "))
        }
        MValue::Map(node) => {
            let inner: Vec<String> = node
                .fields
                .iter()
                .map(|(k, v)| format!("'{}': {}", k, cql_value(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        MValue::TypedMap(entries) => {
            let inner: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}: {}", cql_value(k), cql_value(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

fn cql_type(tag: &TypeTag) -> &'static str {
    match tag {
        TypeTag::Text | TypeTag::TextValidated | TypeTag::EnumStr => "text",
        TypeTag::Ascii => "ascii",
        TypeTag::Int | TypeTag::Millis => "bigint",
        TypeTag::Int32 | TypeTag::EnumOrd => "int",
        TypeTag::Short => "smallint",
        TypeTag::Float => "double",
        TypeTag::Float32 => "float",
        TypeTag::Half => "smallint",
        TypeTag::Bool => "boolean",
        TypeTag::Bytes => "blob",
        TypeTag::Decimal => "decimal",
        TypeTag::Varint => "varint",
        TypeTag::Null => "text",
        TypeTag::Date => "date",
        TypeTag::Time => "time",
        TypeTag::DateTime | TypeTag::Nanos => "timestamp",
        TypeTag::UuidV1 => "timeuuid",
        TypeTag::UuidV7 => "uuid",
        TypeTag::Ulid => "blob",
        TypeTag::List | TypeTag::Array => "list<text>",
        TypeTag::Set => "set<text>",
        TypeTag::Map | TypeTag::TypedMap => "map<text, text>",
    }
}

// -- CDDL vernacular ----------------------------------------------------------

/// Render an MNode's schema as a CDDL type definition
///
/// CDDL (Concise Data Definition Language, RFC 8610) is used for describing
/// CBOR and JSON data structures. This adapter generates CDDL group syntax
/// for an MNode's field structure.
pub fn to_cddl(node: &MNode) -> String {
    let mut lines = Vec::with_capacity(node.fields.len());
    for (name, value) in &node.fields {
        lines.push(format!("  {} : {}", cddl_key(name), cddl_type(&value.tag())));
    }
    format!("{{\n{}\n}}", lines.join(",\n"))
}

/// Render an MValue as a CDDL literal
pub fn to_cddl_value(value: &MValue) -> String {
    match value {
        MValue::Text(s) | MValue::EnumStr(s) | MValue::Ascii(s)
        | MValue::Date(s) | MValue::Time(s) | MValue::DateTime(s) => {
            format!("\"{}\"", s)
        }
        MValue::Int(v) | MValue::Millis(v) => format!("{}", v),
        MValue::Float(v) => format!("{}", v),
        MValue::Bool(v) => if *v { "true".into() } else { "false".into() },
        MValue::Bytes(v) => format!("h'{}'", hex_encode(v)),
        MValue::Null => "null".into(),
        MValue::EnumOrd(v) | MValue::Int32(v) => format!("{}", v),
        MValue::Short(v) => format!("{}", v),
        MValue::Float32(v) => format!("{}", v),
        MValue::Half(v) => format!("{}", v),
        MValue::Nanos { epoch_seconds, nano_adjust } => {
            format!("{}.{}", epoch_seconds, nano_adjust)
        }
        MValue::UuidV1(b) | MValue::UuidV7(b) => format!("\"{}\"", format_uuid(b)),
        MValue::Ulid(b) => format!("h'{}'", hex_encode(b)),
        MValue::List(items) | MValue::Set(items) => {
            let inner: Vec<String> = items.iter().map(to_cddl_value).collect();
            format!("[{}]", inner.join(", "))
        }
        MValue::Array(_, items) => {
            let inner: Vec<String> = items.iter().map(to_cddl_value).collect();
            format!("[{}]", inner.join(", "))
        }
        MValue::Map(node) => to_cddl(node),
        MValue::TypedMap(entries) => {
            let inner: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{} => {}", to_cddl_value(k), to_cddl_value(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

fn cddl_key(name: &str) -> String {
    // CDDL keys can be bare identifiers if they match [a-zA-Z_][a-zA-Z0-9_-]*
    if name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        && name.starts_with(|c: char| c.is_alphabetic() || c == '_')
    {
        name.to_string()
    } else {
        format!("\"{}\"", name)
    }
}

fn cddl_type(tag: &TypeTag) -> &'static str {
    match tag {
        TypeTag::Text | TypeTag::TextValidated | TypeTag::EnumStr | TypeTag::Ascii => "tstr",
        TypeTag::Int | TypeTag::Millis => "int",
        TypeTag::Int32 | TypeTag::EnumOrd => "int",
        TypeTag::Short => "int",
        TypeTag::Float | TypeTag::Float32 => "float",
        TypeTag::Half => "float16",
        TypeTag::Bool => "bool",
        TypeTag::Bytes | TypeTag::Decimal | TypeTag::Varint => "bstr",
        TypeTag::Null => "null",
        TypeTag::Date | TypeTag::Time | TypeTag::DateTime => "tstr",
        TypeTag::Nanos => "int",
        TypeTag::UuidV1 | TypeTag::UuidV7 => "tstr",
        TypeTag::Ulid => "bstr",
        TypeTag::List | TypeTag::Array => "[* any]",
        TypeTag::Set => "[* any]",
        TypeTag::Map | TypeTag::TypedMap => "{* any => any}",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sql_output() {
        let mut node = MNode::new();
        node.insert("name".into(), MValue::Text("alice".into()));
        node.insert("age".into(), MValue::Int(30));
        node.insert("active".into(), MValue::Bool(true));

        let sql = to_sql(&node);
        assert!(sql.contains("'alice'"));
        assert!(sql.contains("30"));
        assert!(sql.contains("TRUE"));
    }

    #[test]
    fn test_sql_schema() {
        let mut node = MNode::new();
        node.insert("name".into(), MValue::Text("x".into()));
        node.insert("id".into(), MValue::Int(1));

        let schema = to_sql_schema(&node);
        assert!(schema.contains("name TEXT"));
        assert!(schema.contains("id BIGINT"));
    }

    #[test]
    fn test_cql_output() {
        let mut node = MNode::new();
        node.insert("name".into(), MValue::Text("bob".into()));
        node.insert("score".into(), MValue::Float(99.5));

        let cql = to_cql(&node);
        assert!(cql.contains("'bob'"));
        assert!(cql.contains("99.5"));
    }

    #[test]
    fn test_cql_schema() {
        let mut node = MNode::new();
        node.insert("name".into(), MValue::Text("x".into()));
        node.insert("id".into(), MValue::Int(1));

        let schema = to_cql_schema(&node);
        assert!(schema.contains("name text"));
        assert!(schema.contains("id bigint"));
    }

    #[test]
    fn test_cddl_output() {
        let mut node = MNode::new();
        node.insert("name".into(), MValue::Text("x".into()));
        node.insert("count".into(), MValue::Int32(42));

        let cddl = to_cddl(&node);
        assert!(cddl.contains("name : tstr"));
        assert!(cddl.contains("count : int"));
    }

    #[test]
    fn test_sql_escaping() {
        let mut node = MNode::new();
        node.insert("val".into(), MValue::Text("it's a test".into()));

        let sql = to_sql(&node);
        assert!(sql.contains("it''s a test"));
    }
}
