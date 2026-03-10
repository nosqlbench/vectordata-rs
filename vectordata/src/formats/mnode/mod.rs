// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! MNode — self-describing binary metadata record format.
//!
//! An MNode is a structured key-value record where each field carries its own
//! type tag. This is the wire format used for `metadata_content` facets in
//! predicated datasets.
//!
//! ## Wire format
//!
//! Payload (no length prefix):
//! ```text
//! [dialect_leader: u8 = 0x01]
//! [field_count: u16 LE]
//! per field:
//!   [name_len: u16 LE][name_utf8: N bytes]
//!   [type_tag: u8]
//!   [value_bytes: variable per tag]
//! ```
//!
//! Framed (for embedding in a stream):
//! ```text
//! [payload_len: u32 LE][payload...]
//! ```

/// Dialect leader byte identifying MNode records.
pub const DIALECT_MNODE: u8 = 0x01;

mod tags;
pub mod scan;
pub mod vernacular;

pub use tags::TypeTag;

use std::fmt;
use std::io::{self, Cursor, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use indexmap::IndexMap;

/// A single typed value in an MNode field
#[derive(Debug, Clone, PartialEq)]
pub enum MValue {
    Text(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Bytes(Vec<u8>),
    Null,
    EnumStr(String),
    EnumOrd(i32),
    List(Vec<MValue>),
    Map(MNode),
    Ascii(String),
    Int32(i32),
    Short(i16),
    Float32(f32),
    Half(u16),
    Millis(i64),
    Nanos { epoch_seconds: i64, nano_adjust: i32 },
    Date(String),
    Time(String),
    DateTime(String),
    UuidV1([u8; 16]),
    UuidV7([u8; 16]),
    Ulid([u8; 16]),
    Array(TypeTag, Vec<MValue>),
    Set(Vec<MValue>),
    TypedMap(Vec<(MValue, MValue)>),
}

impl MValue {
    /// Create a structural fingerprint by replacing the value with a
    /// type-default placeholder.  Preserves the type tag (and for
    /// containers, the recursive structure) but strips all payload data.
    pub fn fingerprint(&self) -> MValue {
        match self {
            MValue::Text(_) => MValue::Text(String::new()),
            MValue::Int(_) => MValue::Int(0),
            MValue::Float(_) => MValue::Float(0.0),
            MValue::Bool(_) => MValue::Bool(false),
            MValue::Bytes(_) => MValue::Bytes(Vec::new()),
            MValue::Null => MValue::Null,
            MValue::EnumStr(_) => MValue::EnumStr(String::new()),
            MValue::EnumOrd(_) => MValue::EnumOrd(0),
            MValue::List(items) => MValue::List(items.iter().map(|v| v.fingerprint()).collect()),
            MValue::Map(node) => MValue::Map(node.fingerprint()),
            MValue::Ascii(_) => MValue::Ascii(String::new()),
            MValue::Int32(_) => MValue::Int32(0),
            MValue::Short(_) => MValue::Short(0),
            MValue::Float32(_) => MValue::Float32(0.0),
            MValue::Half(_) => MValue::Half(0),
            MValue::Millis(_) => MValue::Millis(0),
            MValue::Nanos { .. } => MValue::Nanos { epoch_seconds: 0, nano_adjust: 0 },
            MValue::Date(_) => MValue::Date(String::new()),
            MValue::Time(_) => MValue::Time(String::new()),
            MValue::DateTime(_) => MValue::DateTime(String::new()),
            MValue::UuidV1(_) => MValue::UuidV1([0u8; 16]),
            MValue::UuidV7(_) => MValue::UuidV7([0u8; 16]),
            MValue::Ulid(_) => MValue::Ulid([0u8; 16]),
            MValue::Array(tag, items) => MValue::Array(*tag, items.iter().map(|v| v.fingerprint()).collect()),
            MValue::Set(items) => MValue::Set(items.iter().map(|v| v.fingerprint()).collect()),
            MValue::TypedMap(entries) => MValue::TypedMap(
                entries.iter().map(|(k, v)| (k.fingerprint(), v.fingerprint())).collect(),
            ),
        }
    }

    /// Returns the type tag for this value
    pub fn tag(&self) -> TypeTag {
        match self {
            MValue::Text(_) => TypeTag::Text,
            MValue::Int(_) => TypeTag::Int,
            MValue::Float(_) => TypeTag::Float,
            MValue::Bool(_) => TypeTag::Bool,
            MValue::Bytes(_) => TypeTag::Bytes,
            MValue::Null => TypeTag::Null,
            MValue::EnumStr(_) => TypeTag::EnumStr,
            MValue::EnumOrd(_) => TypeTag::EnumOrd,
            MValue::List(_) => TypeTag::List,
            MValue::Map(_) => TypeTag::Map,
            MValue::Ascii(_) => TypeTag::Ascii,
            MValue::Int32(_) => TypeTag::Int32,
            MValue::Short(_) => TypeTag::Short,
            MValue::Float32(_) => TypeTag::Float32,
            MValue::Half(_) => TypeTag::Half,
            MValue::Millis(_) => TypeTag::Millis,
            MValue::Nanos { .. } => TypeTag::Nanos,
            MValue::Date(_) => TypeTag::Date,
            MValue::Time(_) => TypeTag::Time,
            MValue::DateTime(_) => TypeTag::DateTime,
            MValue::UuidV1(_) => TypeTag::UuidV1,
            MValue::UuidV7(_) => TypeTag::UuidV7,
            MValue::Ulid(_) => TypeTag::Ulid,
            MValue::Array(_, _) => TypeTag::Array,
            MValue::Set(_) => TypeTag::Set,
            MValue::TypedMap(_) => TypeTag::TypedMap,
        }
    }
}

impl fmt::Display for MValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MValue::Text(s) | MValue::EnumStr(s) | MValue::Ascii(s) => {
                write!(f, "'{}'", s)
            }
            MValue::Int(v) => write!(f, "{}", v),
            MValue::Float(v) => write!(f, "{}", v),
            MValue::Bool(v) => write!(f, "{}", v),
            MValue::Bytes(v) => write!(f, "0x{}", hex_encode(v)),
            MValue::Null => write!(f, "NULL"),
            MValue::EnumOrd(v) => write!(f, "enum({})", v),
            MValue::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            MValue::Map(node) => write!(f, "{}", node),
            MValue::Int32(v) => write!(f, "{}", v),
            MValue::Short(v) => write!(f, "{}", v),
            MValue::Float32(v) => write!(f, "{}", v),
            MValue::Half(v) => write!(f, "half(0x{:04x})", v),
            MValue::Millis(v) => write!(f, "millis({})", v),
            MValue::Nanos { epoch_seconds, nano_adjust } => {
                write!(f, "nanos({}.{})", epoch_seconds, nano_adjust)
            }
            MValue::Date(s) | MValue::Time(s) | MValue::DateTime(s) => {
                write!(f, "'{}'", s)
            }
            MValue::UuidV1(b) | MValue::UuidV7(b) => write!(f, "{}", format_uuid(b)),
            MValue::Ulid(b) => write!(f, "ulid({})", hex_encode(b)),
            MValue::Array(_, items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            MValue::Set(items) => {
                write!(f, "{{")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "}}")
            }
            MValue::TypedMap(entries) => {
                write!(f, "{{")?;
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// An MNode: an ordered map of named, typed fields.
///
/// Uses `IndexMap` to preserve insertion order, matching the wire format's
/// field ordering.
#[derive(Debug, Clone, PartialEq)]
pub struct MNode {
    pub fields: IndexMap<String, MValue>,
}

impl MNode {
    /// Create an empty MNode
    pub fn new() -> Self {
        MNode {
            fields: IndexMap::new(),
        }
    }

    /// Add a field
    pub fn insert(&mut self, name: String, value: MValue) {
        self.fields.insert(name, value);
    }

    /// Create a structural fingerprint by preserving field names and type
    /// tags while replacing all values with type-default placeholders.
    ///
    /// Two MNodes with equal fingerprints are *structurally congruent* —
    /// they have the same fields in the same order with the same types,
    /// differing only in values.
    pub fn fingerprint(&self) -> MNode {
        MNode {
            fields: self
                .fields
                .iter()
                .map(|(name, value)| (name.clone(), value.fingerprint()))
                .collect(),
        }
    }

    /// Check whether two MNodes are structurally congruent — same field
    /// names, same order, same value types, differing only in values.
    pub fn is_congruent(&self, other: &Self) -> bool {
        self.fingerprint().to_string() == other.fingerprint().to_string()
    }

    /// Encode to bytes (payload only, no length prefix).
    ///
    /// Prepends the `DIALECT_MNODE` leader byte before the field data.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(DIALECT_MNODE);
        self.write_payload(&mut buf).expect("write to Vec should not fail");
        buf
    }

    /// Encode with a 4-byte LE length prefix (for stream framing)
    pub fn encode(&self) -> Vec<u8> {
        let payload = self.to_bytes();
        let mut buf = Vec::with_capacity(4 + payload.len());
        buf.write_u32::<LittleEndian>(payload.len() as u32).unwrap();
        buf.extend_from_slice(&payload);
        buf
    }

    /// Decode from bytes (payload only, no length prefix).
    ///
    /// Verifies and strips the `DIALECT_MNODE` leader byte before decoding.
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "empty mnode data"));
        }
        if data[0] != DIALECT_MNODE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected MNode dialect leader 0x{:02x}, got 0x{:02x}", DIALECT_MNODE, data[0]),
            ));
        }
        let mut cursor = Cursor::new(&data[1..]);
        Self::read_payload(&mut cursor)
    }

    /// Decode from a framed stream (reads 4-byte length prefix first)
    pub fn from_buffer(reader: &mut impl Read) -> io::Result<Self> {
        let len = reader.read_u32::<LittleEndian>()? as usize;
        let mut payload = vec![0u8; len];
        reader.read_exact(&mut payload)?;
        Self::from_bytes(&payload)
    }

    fn write_payload(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_u16::<LittleEndian>(self.fields.len() as u16)?;
        for (name, value) in &self.fields {
            // Field name
            let name_bytes = name.as_bytes();
            w.write_u16::<LittleEndian>(name_bytes.len() as u16)?;
            w.write_all(name_bytes)?;
            // Type tag + value
            write_tagged_value(w, value)?;
        }
        Ok(())
    }

    fn read_payload(r: &mut Cursor<&[u8]>) -> io::Result<Self> {
        let field_count = r.read_u16::<LittleEndian>()? as usize;
        let mut fields = IndexMap::with_capacity(field_count);
        for _ in 0..field_count {
            let name_len = r.read_u16::<LittleEndian>()? as usize;
            let mut name_buf = vec![0u8; name_len];
            r.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let value = read_tagged_value(r)?;
            fields.insert(name, value);
        }
        Ok(MNode { fields })
    }
}

impl Default for MNode {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, (name, value)) in self.fields.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", name, value)?;
        }
        write!(f, "}}")
    }
}

// -- encoding helpers ---------------------------------------------------------

fn write_tagged_value(w: &mut impl Write, value: &MValue) -> io::Result<()> {
    w.write_u8(value.tag() as u8)?;
    write_value(w, value)
}

fn write_value(w: &mut impl Write, value: &MValue) -> io::Result<()> {
    match value {
        MValue::Text(s) | MValue::EnumStr(s) | MValue::Ascii(s)
        | MValue::Date(s) | MValue::Time(s) | MValue::DateTime(s) => {
            let bytes = s.as_bytes();
            w.write_u32::<LittleEndian>(bytes.len() as u32)?;
            w.write_all(bytes)?;
        }
        MValue::Int(v) | MValue::Millis(v) => {
            w.write_i64::<LittleEndian>(*v)?;
        }
        MValue::Float(v) => {
            w.write_f64::<LittleEndian>(*v)?;
        }
        MValue::Bool(v) => {
            w.write_u8(if *v { 1 } else { 0 })?;
        }
        MValue::Bytes(v) => {
            w.write_u32::<LittleEndian>(v.len() as u32)?;
            w.write_all(v)?;
        }
        MValue::Null => {}
        MValue::EnumOrd(v) | MValue::Int32(v) => {
            w.write_i32::<LittleEndian>(*v)?;
        }
        MValue::List(items) | MValue::Set(items) => {
            w.write_u32::<LittleEndian>(items.len() as u32)?;
            for item in items {
                write_tagged_value(w, item)?;
            }
        }
        MValue::Map(node) => {
            let payload = node.to_bytes();
            w.write_u32::<LittleEndian>(payload.len() as u32)?;
            w.write_all(&payload)?;
        }
        MValue::Short(v) => {
            w.write_i16::<LittleEndian>(*v)?;
        }
        MValue::Float32(v) => {
            w.write_f32::<LittleEndian>(*v)?;
        }
        MValue::Half(v) => {
            w.write_u16::<LittleEndian>(*v)?;
        }
        MValue::Nanos { epoch_seconds, nano_adjust } => {
            w.write_i64::<LittleEndian>(*epoch_seconds)?;
            w.write_i32::<LittleEndian>(*nano_adjust)?;
        }
        MValue::UuidV1(b) | MValue::UuidV7(b) | MValue::Ulid(b) => {
            w.write_all(b)?;
        }
        MValue::Array(elem_tag, items) => {
            w.write_u8(*elem_tag as u8)?;
            w.write_u32::<LittleEndian>(items.len() as u32)?;
            for item in items {
                write_value(w, item)?;
            }
        }
        MValue::TypedMap(entries) => {
            w.write_u32::<LittleEndian>(entries.len() as u32)?;
            for (k, v) in entries {
                write_tagged_value(w, k)?;
                write_tagged_value(w, v)?;
            }
        }
    }
    Ok(())
}

// -- decoding helpers ---------------------------------------------------------

fn read_tagged_value(r: &mut Cursor<&[u8]>) -> io::Result<MValue> {
    let tag_byte = r.read_u8()?;
    let tag = TypeTag::from_u8(tag_byte).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, format!("unknown type tag: {}", tag_byte))
    })?;
    read_value(r, tag)
}

fn read_value(r: &mut Cursor<&[u8]>, tag: TypeTag) -> io::Result<MValue> {
    match tag {
        TypeTag::Text => Ok(MValue::Text(read_len_prefixed_string(r)?)),
        TypeTag::Int => Ok(MValue::Int(r.read_i64::<LittleEndian>()?)),
        TypeTag::Float => Ok(MValue::Float(r.read_f64::<LittleEndian>()?)),
        TypeTag::Bool => Ok(MValue::Bool(r.read_u8()? != 0)),
        TypeTag::Bytes => {
            let len = r.read_u32::<LittleEndian>()? as usize;
            let mut buf = vec![0u8; len];
            r.read_exact(&mut buf)?;
            Ok(MValue::Bytes(buf))
        }
        TypeTag::Null => Ok(MValue::Null),
        TypeTag::EnumStr => Ok(MValue::EnumStr(read_len_prefixed_string(r)?)),
        TypeTag::EnumOrd => Ok(MValue::EnumOrd(r.read_i32::<LittleEndian>()?)),
        TypeTag::List => {
            let count = r.read_u32::<LittleEndian>()? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_tagged_value(r)?);
            }
            Ok(MValue::List(items))
        }
        TypeTag::Map => {
            let len = r.read_u32::<LittleEndian>()? as usize;
            let mut payload = vec![0u8; len];
            r.read_exact(&mut payload)?;
            let node = MNode::from_bytes(&payload)?;
            Ok(MValue::Map(node))
        }
        TypeTag::TextValidated => Ok(MValue::Text(read_len_prefixed_string(r)?)),
        TypeTag::Ascii => Ok(MValue::Ascii(read_len_prefixed_string(r)?)),
        TypeTag::Int32 => Ok(MValue::Int32(r.read_i32::<LittleEndian>()?)),
        TypeTag::Short => Ok(MValue::Short(r.read_i16::<LittleEndian>()?)),
        TypeTag::Decimal => {
            let _scale = r.read_i32::<LittleEndian>()?;
            let len = r.read_u32::<LittleEndian>()? as usize;
            let mut buf = vec![0u8; len];
            r.read_exact(&mut buf)?;
            // Store as raw bytes for now — full BigDecimal support can be added later
            Ok(MValue::Bytes(buf))
        }
        TypeTag::Varint => {
            let len = r.read_u32::<LittleEndian>()? as usize;
            let mut buf = vec![0u8; len];
            r.read_exact(&mut buf)?;
            Ok(MValue::Bytes(buf))
        }
        TypeTag::Float32 => Ok(MValue::Float32(r.read_f32::<LittleEndian>()?)),
        TypeTag::Half => Ok(MValue::Half(r.read_u16::<LittleEndian>()?)),
        TypeTag::Millis => Ok(MValue::Millis(r.read_i64::<LittleEndian>()?)),
        TypeTag::Nanos => {
            let epoch_seconds = r.read_i64::<LittleEndian>()?;
            let nano_adjust = r.read_i32::<LittleEndian>()?;
            Ok(MValue::Nanos { epoch_seconds, nano_adjust })
        }
        TypeTag::Date => Ok(MValue::Date(read_len_prefixed_string(r)?)),
        TypeTag::Time => Ok(MValue::Time(read_len_prefixed_string(r)?)),
        TypeTag::DateTime => Ok(MValue::DateTime(read_len_prefixed_string(r)?)),
        TypeTag::UuidV1 => {
            let mut buf = [0u8; 16];
            r.read_exact(&mut buf)?;
            Ok(MValue::UuidV1(buf))
        }
        TypeTag::UuidV7 => {
            let mut buf = [0u8; 16];
            r.read_exact(&mut buf)?;
            Ok(MValue::UuidV7(buf))
        }
        TypeTag::Ulid => {
            let mut buf = [0u8; 16];
            r.read_exact(&mut buf)?;
            Ok(MValue::Ulid(buf))
        }
        TypeTag::Array => {
            let elem_tag_byte = r.read_u8()?;
            let elem_tag = TypeTag::from_u8(elem_tag_byte).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "unknown array element tag")
            })?;
            let count = r.read_u32::<LittleEndian>()? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_value(r, elem_tag)?);
            }
            Ok(MValue::Array(elem_tag, items))
        }
        TypeTag::Set => {
            let count = r.read_u32::<LittleEndian>()? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_tagged_value(r)?);
            }
            Ok(MValue::Set(items))
        }
        TypeTag::TypedMap => {
            let count = r.read_u32::<LittleEndian>()? as usize;
            let mut entries = Vec::with_capacity(count);
            for _ in 0..count {
                let k = read_tagged_value(r)?;
                let v = read_tagged_value(r)?;
                entries.push((k, v));
            }
            Ok(MValue::TypedMap(entries))
        }
    }
}

fn read_len_prefixed_string(r: &mut Cursor<&[u8]>) -> io::Result<String> {
    let len = r.read_u32::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn format_uuid(bytes: &[u8; 16]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11],
        bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnode_roundtrip() {
        let mut node = MNode::new();
        node.insert("name".into(), MValue::Text("alice".into()));
        node.insert("age".into(), MValue::Int(30));
        node.insert("score".into(), MValue::Float(99.5));
        node.insert("active".into(), MValue::Bool(true));
        node.insert("empty".into(), MValue::Null);

        let bytes = node.to_bytes();
        let decoded = MNode::from_bytes(&bytes).unwrap();
        assert_eq!(node, decoded);
    }

    #[test]
    fn test_mnode_framed_roundtrip() {
        let mut node = MNode::new();
        node.insert("x".into(), MValue::Int32(42));
        node.insert("y".into(), MValue::Float32(3.14));

        let encoded = node.encode();
        let mut cursor = Cursor::new(encoded.as_slice());
        let decoded = MNode::from_buffer(&mut cursor).unwrap();
        assert_eq!(node, decoded);
    }

    #[test]
    fn test_mnode_nested() {
        let mut inner = MNode::new();
        inner.insert("key".into(), MValue::Text("val".into()));

        let mut outer = MNode::new();
        outer.insert("nested".into(), MValue::Map(inner));
        outer.insert("list".into(), MValue::List(vec![
            MValue::Int(1),
            MValue::Text("two".into()),
        ]));

        let bytes = outer.to_bytes();
        let decoded = MNode::from_bytes(&bytes).unwrap();
        assert_eq!(outer, decoded);
    }

    #[test]
    fn test_mnode_display() {
        let mut node = MNode::new();
        node.insert("name".into(), MValue::Text("bob".into()));
        node.insert("age".into(), MValue::Int(25));
        let s = format!("{}", node);
        assert!(s.contains("name: 'bob'"));
        assert!(s.contains("age: 25"));
    }

    #[test]
    fn test_mvalue_fingerprint() {
        assert_eq!(MValue::Text("hello".into()).fingerprint(), MValue::Text(String::new()));
        assert_eq!(MValue::Int(42).fingerprint(), MValue::Int(0));
        assert_eq!(MValue::Float(3.14).fingerprint(), MValue::Float(0.0));
        assert_eq!(MValue::Bool(true).fingerprint(), MValue::Bool(false));
        assert_eq!(MValue::Null.fingerprint(), MValue::Null);
    }

    #[test]
    fn test_mnode_fingerprint() {
        let mut a = MNode::new();
        a.insert("name".into(), MValue::Text("alice".into()));
        a.insert("age".into(), MValue::Int(30));

        let mut b = MNode::new();
        b.insert("name".into(), MValue::Text("bob".into()));
        b.insert("age".into(), MValue::Int(25));

        assert!(a.is_congruent(&b));
        assert_eq!(a.fingerprint(), b.fingerprint());

        // Different field set — not congruent
        let mut c = MNode::new();
        c.insert("name".into(), MValue::Text("carol".into()));
        c.insert("score".into(), MValue::Float(99.0));
        assert!(!a.is_congruent(&c));
    }

    #[test]
    fn test_mnode_fingerprint_nested() {
        let mut inner = MNode::new();
        inner.insert("k".into(), MValue::Int(999));

        let mut node = MNode::new();
        node.insert("nested".into(), MValue::Map(inner));
        node.insert("items".into(), MValue::List(vec![MValue::Text("x".into())]));

        let fp = node.fingerprint();
        match fp.fields.get("nested").unwrap() {
            MValue::Map(m) => {
                assert_eq!(*m.fields.get("k").unwrap(), MValue::Int(0));
            }
            _ => panic!("expected Map"),
        }
        match fp.fields.get("items").unwrap() {
            MValue::List(items) => {
                assert_eq!(items[0], MValue::Text(String::new()));
            }
            _ => panic!("expected List"),
        }
    }
}
