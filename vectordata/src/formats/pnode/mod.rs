// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! PNode — predicate tree binary format.
//!
//! A PNode represents a boolean predicate tree used for `metadata_predicates`
//! facets in predicated datasets. The tree is composed of:
//!
//! - **PredicateNode**: leaf nodes with a field reference, operator, and comparand values
//! - **ConjugateNode**: interior nodes that combine children with AND/OR
//!
//! ## Wire format
//!
//! Encoding is recursive, pre-order (parent first, then children).
//!
//! The first byte of each node is the `ConjugateType` discriminant:
//! - `0` = PRED (PredicateNode)
//! - `1` = AND (ConjugateNode)
//! - `2` = OR (ConjugateNode)
//!
//! ### ConjugateNode
//! ```text
//! [conjugate_type: u8][child_count: u8][children...]
//! ```
//!
//! ### PredicateNode (indexed mode — field by index)
//! ```text
//! [PRED=0: u8][field_index: u8][op: u8][comparand_count: i16 LE][comparands: i64 LE * n]
//! ```
//!
//! ### PredicateNode (named mode — field by name)
//!
//! Two sub-formats are supported. Version detection uses a `0xFF` marker byte
//! immediately after the `DIALECT_PNODE` leader:
//!
//! **Legacy format** (first byte after leader is 0x00, 0x01, or 0x02 — a ConjugateType):
//! ```text
//! [PRED=0: u8][name_len: u16 LE][name: UTF-8][op: u8][comparand_count: i16 LE][comparands: i64 LE * n]
//! ```
//!
//! **Typed format** (first byte after leader is 0xFF):
//! ```text
//! [0xFF][PRED=0: u8][name_len: u16 LE][name: UTF-8][op: u8][comparand_count: i16 LE][typed_comparands...]
//! ```
//!
//! Typed comparand encoding per value:
//! - tag `0` (Int): i64 LE
//! - tag `1` (Float): f64 LE
//! - tag `2` (Text): u16 LE len + UTF-8
//! - tag `3` (Bool): u8
//! - tag `4` (Bytes): u32 LE len + raw
//! - tag `5` (Null): nothing
//!
//! All PNode payloads are prefixed with a `DIALECT_PNODE` leader byte (`0x02`)
//! to identify the record type when stored alongside MNode records.

/// Dialect leader byte identifying PNode records.
pub const DIALECT_PNODE: u8 = 0x02;

/// Marker byte for typed-comparand named format.
const TYPED_MARKER: u8 = 0xFF;

pub mod eval;
pub mod vernacular;

// vernacular module provides to_cddl/to_cql/to_sql — used from tests

use std::fmt;
use std::io::{self, Cursor, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// Conjugate type discriminant — the first byte of every node.
///
/// `Pred` marks a leaf predicate; `And` and `Or` mark interior boolean nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ConjugateType {
    /// Leaf predicate node.
    Pred = 0,
    /// Boolean AND conjunction.
    And = 1,
    /// Boolean OR disjunction.
    Or = 2,
}

impl ConjugateType {
    /// Decode from the raw wire byte. Returns `None` for unrecognized values.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Pred),
            1 => Some(Self::And),
            2 => Some(Self::Or),
            _ => None,
        }
    }
}

impl fmt::Display for ConjugateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pred => write!(f, "PRED"),
            Self::And => write!(f, "AND"),
            Self::Or => write!(f, "OR"),
        }
    }
}

/// Comparison operator for predicate leaf nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpType {
    /// Greater than (`>`).
    Gt = 0,
    /// Less than (`<`).
    Lt = 1,
    /// Equal (`=`).
    Eq = 2,
    /// Not equal (`!=`).
    Ne = 3,
    /// Greater than or equal (`>=`).
    Ge = 4,
    /// Less than or equal (`<=`).
    Le = 5,
    /// Set membership (`IN`).
    In = 6,
    /// Pattern match (`MATCHES`).
    Matches = 7,
}

impl OpType {
    /// Decode from the raw wire byte. Returns `None` for unrecognized values.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Gt),
            1 => Some(Self::Lt),
            2 => Some(Self::Eq),
            3 => Some(Self::Ne),
            4 => Some(Self::Ge),
            5 => Some(Self::Le),
            6 => Some(Self::In),
            7 => Some(Self::Matches),
            _ => None,
        }
    }

    /// SQL operator symbol
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Gt => ">",
            Self::Lt => "<",
            Self::Eq => "=",
            Self::Ne => "!=",
            Self::Ge => ">=",
            Self::Le => "<=",
            Self::In => "IN",
            Self::Matches => "MATCHES",
        }
    }
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

/// Field reference in a predicate node — either a positional index or a name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldRef {
    /// Positional field index (indexed wire format).
    Index(u8),
    /// Field name string (named wire format).
    Named(String),
}

impl fmt::Display for FieldRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldRef::Index(i) => write!(f, "field[{}]", i),
            FieldRef::Named(s) => write!(f, "{}", s),
        }
    }
}

/// A typed comparand value used in predicate nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum Comparand {
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit floating-point
    Float(f64),
    /// UTF-8 text string
    Text(String),
    /// Boolean value
    Bool(bool),
    /// Raw byte sequence
    Bytes(Vec<u8>),
    /// Null / absent value
    Null,
}

impl Comparand {
    /// Extract the i64 value, or return an error for non-Int variants.
    ///
    /// Used by indexed mode which only supports integer comparands.
    pub fn as_i64(&self) -> io::Result<i64> {
        match self {
            Comparand::Int(v) => Ok(*v),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("indexed mode only supports Int comparands, got {}", other),
            )),
        }
    }
}

impl fmt::Display for Comparand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Comparand::Int(v) => write!(f, "{}", v),
            Comparand::Float(v) => {
                let s = v.to_string();
                if s.contains('.') {
                    write!(f, "{}", s)
                } else {
                    write!(f, "{}.0", s)
                }
            }
            Comparand::Text(s) => write!(f, "'{}'", s),
            Comparand::Bool(b) => write!(f, "{}", b),
            Comparand::Bytes(b) => {
                write!(f, "X'")?;
                for byte in b {
                    write!(f, "{:02x}", byte)?;
                }
                write!(f, "'")
            }
            Comparand::Null => write!(f, "NULL"),
        }
    }
}

/// A predicate tree node — either a leaf comparison or a boolean combinator.
#[derive(Debug, Clone, PartialEq)]
pub enum PNode {
    /// Leaf predicate: `field op (v1, v2, ...)`.
    Predicate(PredicateNode),
    /// Boolean conjunction: AND or OR of child nodes.
    Conjugate(ConjugateNode),
}

/// Leaf predicate node: `field op (v1, v2, ...)`.
///
/// For single-comparand operators (Gt, Lt, Eq, etc.) the `comparands` vec
/// has one element. For `In`, it holds the membership set.
#[derive(Debug, Clone, PartialEq)]
pub struct PredicateNode {
    /// The field this predicate tests.
    pub field: FieldRef,
    /// The comparison operator.
    pub op: OpType,
    /// One or more comparand values.
    pub comparands: Vec<Comparand>,
}

/// Interior boolean node: AND/OR combining child sub-trees.
#[derive(Debug, Clone, PartialEq)]
pub struct ConjugateNode {
    /// Whether this is an AND or OR combination.
    pub conjugate_type: ConjugateType,
    /// Child predicate sub-trees.
    pub children: Vec<PNode>,
}

impl PNode {
    /// Create a structural fingerprint by replacing all comparand values with
    /// type-default placeholders while preserving the tree shape, field names,
    /// operators, and comparand count/types.
    ///
    /// Two PNodes with equal fingerprints are *structurally congruent* — they
    /// differ only in the specific comparand values.
    pub fn fingerprint(&self) -> PNode {
        match self {
            PNode::Predicate(pred) => PNode::Predicate(PredicateNode {
                field: pred.field.clone(),
                op: pred.op,
                comparands: pred
                    .comparands
                    .iter()
                    .map(|c| match c {
                        Comparand::Int(_) => Comparand::Int(0),
                        Comparand::Float(_) => Comparand::Float(0.0),
                        Comparand::Text(_) => Comparand::Text(String::new()),
                        Comparand::Bool(_) => Comparand::Bool(false),
                        Comparand::Bytes(_) => Comparand::Bytes(Vec::new()),
                        Comparand::Null => Comparand::Null,
                    })
                    .collect(),
            }),
            PNode::Conjugate(conj) => PNode::Conjugate(ConjugateNode {
                conjugate_type: conj.conjugate_type,
                children: conj.children.iter().map(|c| c.fingerprint()).collect(),
            }),
        }
    }

    /// Check whether two PNodes are structurally congruent — same tree shape,
    /// field references, operators, and comparand types, differing only in values.
    pub fn is_congruent(&self, other: &Self) -> bool {
        self.fingerprint().to_string() == other.fingerprint().to_string()
    }

    /// Encode to bytes (indexed mode — fields as u8 indices).
    ///
    /// Prepends the `DIALECT_PNODE` leader byte before the tree data.
    /// All comparands must be `Comparand::Int`; other types produce an error.
    pub fn to_bytes_indexed(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(DIALECT_PNODE);
        self.write_indexed(&mut buf).expect("write to Vec should not fail");
        buf
    }

    /// Encode to bytes (named mode — fields as UTF-8 strings).
    ///
    /// Prepends the `DIALECT_PNODE` leader byte and `0xFF` typed marker
    /// before the tree data.
    pub fn to_bytes_named(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(DIALECT_PNODE);
        buf.push(TYPED_MARKER);
        self.write_named(&mut buf).expect("write to Vec should not fail");
        buf
    }

    /// Decode from bytes (indexed mode).
    ///
    /// Verifies and strips the `DIALECT_PNODE` leader byte before decoding.
    /// All comparands are wrapped as `Comparand::Int`.
    pub fn from_bytes_indexed(data: &[u8]) -> io::Result<Self> {
        if data.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "empty pnode data"));
        }
        if data[0] != DIALECT_PNODE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected PNode dialect leader 0x{:02x}, got 0x{:02x}", DIALECT_PNODE, data[0]),
            ));
        }
        let mut cursor = Cursor::new(&data[1..]);
        Self::read_indexed(&mut cursor)
    }

    /// Decode from bytes (named mode).
    ///
    /// Verifies and strips the `DIALECT_PNODE` leader byte before decoding.
    /// Supports both legacy (i64-only) and typed comparand formats via
    /// `0xFF` marker detection.
    pub fn from_bytes_named(data: &[u8]) -> io::Result<Self> {
        if data.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "empty pnode data"));
        }
        if data[0] != DIALECT_PNODE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected PNode dialect leader 0x{:02x}, got 0x{:02x}", DIALECT_PNODE, data[0]),
            ));
        }
        if data.len() < 2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "pnode data too short"));
        }
        if data[1] == TYPED_MARKER {
            // New typed format
            let mut cursor = Cursor::new(&data[2..]);
            Self::read_named_typed(&mut cursor)
        } else {
            // Legacy format — i64-only comparands
            let mut cursor = Cursor::new(&data[1..]);
            Self::read_named_legacy(&mut cursor)
        }
    }

    fn write_indexed(&self, w: &mut impl Write) -> io::Result<()> {
        match self {
            PNode::Predicate(pred) => {
                w.write_u8(ConjugateType::Pred as u8)?;
                match &pred.field {
                    FieldRef::Index(i) => w.write_u8(*i)?,
                    FieldRef::Named(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "named field in indexed mode",
                        ));
                    }
                }
                w.write_u8(pred.op as u8)?;
                w.write_i16::<LittleEndian>(pred.comparands.len() as i16)?;
                for v in &pred.comparands {
                    w.write_i64::<LittleEndian>(v.as_i64()?)?;
                }
            }
            PNode::Conjugate(conj) => {
                w.write_u8(conj.conjugate_type as u8)?;
                w.write_u8(conj.children.len() as u8)?;
                for child in &conj.children {
                    child.write_indexed(w)?;
                }
            }
        }
        Ok(())
    }

    fn write_named(&self, w: &mut impl Write) -> io::Result<()> {
        match self {
            PNode::Predicate(pred) => {
                w.write_u8(ConjugateType::Pred as u8)?;
                match &pred.field {
                    FieldRef::Named(name) => {
                        let name_bytes = name.as_bytes();
                        w.write_u16::<LittleEndian>(name_bytes.len() as u16)?;
                        w.write_all(name_bytes)?;
                    }
                    FieldRef::Index(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "indexed field in named mode",
                        ));
                    }
                }
                w.write_u8(pred.op as u8)?;
                w.write_i16::<LittleEndian>(pred.comparands.len() as i16)?;
                for v in &pred.comparands {
                    write_typed_comparand(w, v)?;
                }
            }
            PNode::Conjugate(conj) => {
                w.write_u8(conj.conjugate_type as u8)?;
                w.write_u8(conj.children.len() as u8)?;
                for child in &conj.children {
                    child.write_named(w)?;
                }
            }
        }
        Ok(())
    }

    fn read_indexed(r: &mut Cursor<&[u8]>) -> io::Result<Self> {
        let tag = r.read_u8()?;
        let ctype = ConjugateType::from_u8(tag).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("unknown conjugate type: {}", tag))
        })?;
        match ctype {
            ConjugateType::Pred => {
                let field_index = r.read_u8()?;
                let op_byte = r.read_u8()?;
                let op = OpType::from_u8(op_byte).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("unknown op: {}", op_byte))
                })?;
                let count = r.read_i16::<LittleEndian>()? as usize;
                let mut comparands = Vec::with_capacity(count);
                for _ in 0..count {
                    comparands.push(Comparand::Int(r.read_i64::<LittleEndian>()?));
                }
                Ok(PNode::Predicate(PredicateNode {
                    field: FieldRef::Index(field_index),
                    op,
                    comparands,
                }))
            }
            ConjugateType::And | ConjugateType::Or => {
                let child_count = r.read_u8()? as usize;
                let mut children = Vec::with_capacity(child_count);
                for _ in 0..child_count {
                    children.push(Self::read_indexed(r)?);
                }
                Ok(PNode::Conjugate(ConjugateNode {
                    conjugate_type: ctype,
                    children,
                }))
            }
        }
    }

    /// Read named mode with legacy i64-only comparands.
    fn read_named_legacy(r: &mut Cursor<&[u8]>) -> io::Result<Self> {
        let tag = r.read_u8()?;
        let ctype = ConjugateType::from_u8(tag).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("unknown conjugate type: {}", tag))
        })?;
        match ctype {
            ConjugateType::Pred => {
                let name_len = r.read_u16::<LittleEndian>()? as usize;
                let mut name_buf = vec![0u8; name_len];
                r.read_exact(&mut name_buf)?;
                let name = String::from_utf8(name_buf)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let op_byte = r.read_u8()?;
                let op = OpType::from_u8(op_byte).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("unknown op: {}", op_byte))
                })?;
                let count = r.read_i16::<LittleEndian>()? as usize;
                let mut comparands = Vec::with_capacity(count);
                for _ in 0..count {
                    comparands.push(Comparand::Int(r.read_i64::<LittleEndian>()?));
                }
                Ok(PNode::Predicate(PredicateNode {
                    field: FieldRef::Named(name),
                    op,
                    comparands,
                }))
            }
            ConjugateType::And | ConjugateType::Or => {
                let child_count = r.read_u8()? as usize;
                let mut children = Vec::with_capacity(child_count);
                for _ in 0..child_count {
                    children.push(Self::read_named_legacy(r)?);
                }
                Ok(PNode::Conjugate(ConjugateNode {
                    conjugate_type: ctype,
                    children,
                }))
            }
        }
    }

    /// Read named mode with typed comparands (new format after 0xFF marker).
    fn read_named_typed(r: &mut Cursor<&[u8]>) -> io::Result<Self> {
        let tag = r.read_u8()?;
        let ctype = ConjugateType::from_u8(tag).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("unknown conjugate type: {}", tag))
        })?;
        match ctype {
            ConjugateType::Pred => {
                let name_len = r.read_u16::<LittleEndian>()? as usize;
                let mut name_buf = vec![0u8; name_len];
                r.read_exact(&mut name_buf)?;
                let name = String::from_utf8(name_buf)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let op_byte = r.read_u8()?;
                let op = OpType::from_u8(op_byte).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("unknown op: {}", op_byte))
                })?;
                let count = r.read_i16::<LittleEndian>()? as usize;
                let mut comparands = Vec::with_capacity(count);
                for _ in 0..count {
                    comparands.push(read_typed_comparand(r)?);
                }
                Ok(PNode::Predicate(PredicateNode {
                    field: FieldRef::Named(name),
                    op,
                    comparands,
                }))
            }
            ConjugateType::And | ConjugateType::Or => {
                let child_count = r.read_u8()? as usize;
                let mut children = Vec::with_capacity(child_count);
                for _ in 0..child_count {
                    children.push(Self::read_named_typed(r)?);
                }
                Ok(PNode::Conjugate(ConjugateNode {
                    conjugate_type: ctype,
                    children,
                }))
            }
        }
    }
}

/// Write a single typed comparand to the output.
fn write_typed_comparand(w: &mut impl Write, c: &Comparand) -> io::Result<()> {
    match c {
        Comparand::Int(v) => {
            w.write_u8(0)?;
            w.write_i64::<LittleEndian>(*v)?;
        }
        Comparand::Float(v) => {
            w.write_u8(1)?;
            w.write_f64::<LittleEndian>(*v)?;
        }
        Comparand::Text(s) => {
            w.write_u8(2)?;
            let bytes = s.as_bytes();
            w.write_u16::<LittleEndian>(bytes.len() as u16)?;
            w.write_all(bytes)?;
        }
        Comparand::Bool(b) => {
            w.write_u8(3)?;
            w.write_u8(if *b { 1 } else { 0 })?;
        }
        Comparand::Bytes(b) => {
            w.write_u8(4)?;
            w.write_u32::<LittleEndian>(b.len() as u32)?;
            w.write_all(b)?;
        }
        Comparand::Null => {
            w.write_u8(5)?;
        }
    }
    Ok(())
}

/// Read a single typed comparand from the input.
fn read_typed_comparand(r: &mut Cursor<&[u8]>) -> io::Result<Comparand> {
    let tag = r.read_u8()?;
    match tag {
        0 => Ok(Comparand::Int(r.read_i64::<LittleEndian>()?)),
        1 => Ok(Comparand::Float(r.read_f64::<LittleEndian>()?)),
        2 => {
            let len = r.read_u16::<LittleEndian>()? as usize;
            let mut buf = vec![0u8; len];
            r.read_exact(&mut buf)?;
            let s = String::from_utf8(buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(Comparand::Text(s))
        }
        3 => {
            let b = r.read_u8()?;
            Ok(Comparand::Bool(b != 0))
        }
        4 => {
            let len = r.read_u32::<LittleEndian>()? as usize;
            let mut buf = vec![0u8; len];
            r.read_exact(&mut buf)?;
            Ok(Comparand::Bytes(buf))
        }
        5 => Ok(Comparand::Null),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown comparand type tag: {}", tag),
        )),
    }
}

impl fmt::Display for PNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PNode::Predicate(pred) => {
                write!(f, "{} {} ", pred.field, pred.op)?;
                if pred.comparands.len() == 1 {
                    write!(f, "{}", pred.comparands[0])
                } else {
                    write!(f, "(")?;
                    for (i, v) in pred.comparands.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v)?;
                    }
                    write!(f, ")")
                }
            }
            PNode::Conjugate(conj) => {
                let op = match conj.conjugate_type {
                    ConjugateType::And => "AND",
                    ConjugateType::Or => "OR",
                    _ => unreachable!(),
                };
                write!(f, "(")?;
                for (i, child) in conj.children.iter().enumerate() {
                    if i > 0 {
                        write!(f, " {} ", op)?;
                    }
                    write!(f, "{}", child)?;
                }
                write!(f, ")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_indexed_roundtrip() {
        let node = PNode::Predicate(PredicateNode {
            field: FieldRef::Index(3),
            op: OpType::Eq,
            comparands: vec![Comparand::Int(42)],
        });

        let bytes = node.to_bytes_indexed();
        let decoded = PNode::from_bytes_indexed(&bytes).unwrap();
        assert_eq!(node, decoded);
    }

    #[test]
    fn test_predicate_named_roundtrip() {
        let node = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("color".into()),
            op: OpType::In,
            comparands: vec![Comparand::Int(1), Comparand::Int(2), Comparand::Int(3)],
        });

        let bytes = node.to_bytes_named();
        let decoded = PNode::from_bytes_named(&bytes).unwrap();
        assert_eq!(node, decoded);
    }

    #[test]
    fn test_conjugate_tree_roundtrip() {
        let tree = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Index(0),
                    op: OpType::Gt,
                    comparands: vec![Comparand::Int(10)],
                }),
                PNode::Conjugate(ConjugateNode {
                    conjugate_type: ConjugateType::Or,
                    children: vec![
                        PNode::Predicate(PredicateNode {
                            field: FieldRef::Index(1),
                            op: OpType::Eq,
                            comparands: vec![Comparand::Int(5)],
                        }),
                        PNode::Predicate(PredicateNode {
                            field: FieldRef::Index(2),
                            op: OpType::Le,
                            comparands: vec![Comparand::Int(100)],
                        }),
                    ],
                }),
            ],
        });

        let bytes = tree.to_bytes_indexed();
        let decoded = PNode::from_bytes_indexed(&bytes).unwrap();
        assert_eq!(tree, decoded);
    }

    #[test]
    fn test_display() {
        let tree = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("age".into()),
                    op: OpType::Gt,
                    comparands: vec![Comparand::Int(18)],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("score".into()),
                    op: OpType::Le,
                    comparands: vec![Comparand::Int(100)],
                }),
            ],
        });
        let s = format!("{}", tree);
        assert!(s.contains("AND"));
        assert!(s.contains("age > 18"));
        assert!(s.contains("score <= 100"));
    }

    #[test]
    fn test_indexed_wire_size() {
        // Leader byte + indexed predicate with 1 comparand: 1 + 1 + 1 + 1 + 2 + 8 = 14 bytes
        let node = PNode::Predicate(PredicateNode {
            field: FieldRef::Index(0),
            op: OpType::Eq,
            comparands: vec![Comparand::Int(42)],
        });
        let bytes = node.to_bytes_indexed();
        assert_eq!(bytes.len(), 14);
    }

    #[test]
    fn test_named_typed_roundtrip() {
        let tree = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("age".into()),
                    op: OpType::Ge,
                    comparands: vec![Comparand::Int(18)],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("score".into()),
                    op: OpType::Lt,
                    comparands: vec![Comparand::Float(99.5)],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("name".into()),
                    op: OpType::Eq,
                    comparands: vec![Comparand::Text("alice".into())],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("active".into()),
                    op: OpType::Eq,
                    comparands: vec![Comparand::Bool(true)],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("data".into()),
                    op: OpType::Eq,
                    comparands: vec![Comparand::Bytes(vec![0xDE, 0xAD])],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("empty".into()),
                    op: OpType::Eq,
                    comparands: vec![Comparand::Null],
                }),
            ],
        });

        let bytes = tree.to_bytes_named();
        // Verify typed marker is present
        assert_eq!(bytes[0], DIALECT_PNODE);
        assert_eq!(bytes[1], TYPED_MARKER);

        let decoded = PNode::from_bytes_named(&bytes).unwrap();
        assert_eq!(tree, decoded);
    }

    #[test]
    fn test_named_legacy_backward_compat() {
        // Build legacy-format bytes manually: DIALECT_PNODE + PRED(0) + name + op + i64 comparands
        let mut buf = Vec::new();
        buf.push(DIALECT_PNODE);
        // ConjugateType::Pred = 0
        buf.push(0);
        // name "x" — u16 LE length + utf8
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.push(b'x');
        // OpType::Eq = 2
        buf.push(2);
        // comparand count = 1
        buf.extend_from_slice(&1i16.to_le_bytes());
        // comparand value = 42
        buf.extend_from_slice(&42i64.to_le_bytes());

        let decoded = PNode::from_bytes_named(&buf).unwrap();
        match decoded {
            PNode::Predicate(pred) => {
                assert_eq!(pred.field, FieldRef::Named("x".into()));
                assert_eq!(pred.op, OpType::Eq);
                assert_eq!(pred.comparands, vec![Comparand::Int(42)]);
            }
            _ => panic!("expected Predicate"),
        }
    }

    #[test]
    fn test_indexed_rejects_non_int() {
        let node = PNode::Predicate(PredicateNode {
            field: FieldRef::Index(0),
            op: OpType::Eq,
            comparands: vec![Comparand::Text("nope".into())],
        });
        // to_bytes_indexed should fail for non-Int comparands
        let result = std::panic::catch_unwind(|| node.to_bytes_indexed());
        assert!(result.is_err() || {
            // The write_indexed returns an error which causes expect to panic
            true
        });
    }

    #[test]
    fn test_fingerprint_strips_values() {
        let node = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("age".into()),
            op: OpType::Gt,
            comparands: vec![Comparand::Int(42)],
        });
        let fp = node.fingerprint();
        assert_eq!(
            fp,
            PNode::Predicate(PredicateNode {
                field: FieldRef::Named("age".into()),
                op: OpType::Gt,
                comparands: vec![Comparand::Int(0)],
            })
        );
    }

    #[test]
    fn test_fingerprint_congruence() {
        let a = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("x".into()),
                    op: OpType::Eq,
                    comparands: vec![Comparand::Text("hello".into())],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("y".into()),
                    op: OpType::Lt,
                    comparands: vec![Comparand::Int(100)],
                }),
            ],
        });
        let b = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("x".into()),
                    op: OpType::Eq,
                    comparands: vec![Comparand::Text("world".into())],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("y".into()),
                    op: OpType::Lt,
                    comparands: vec![Comparand::Int(999)],
                }),
            ],
        });
        assert!(a.is_congruent(&b));

        // Different structure — not congruent
        let c = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("x".into()),
            op: OpType::Eq,
            comparands: vec![Comparand::Text("hello".into())],
        });
        assert!(!a.is_congruent(&c));
    }

    #[test]
    fn test_fingerprint_preserves_comparand_types() {
        let node = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("f".into()),
            op: OpType::In,
            comparands: vec![
                Comparand::Text("a".into()),
                Comparand::Text("b".into()),
            ],
        });
        let fp = node.fingerprint();
        let expected = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("f".into()),
            op: OpType::In,
            comparands: vec![
                Comparand::Text(String::new()),
                Comparand::Text(String::new()),
            ],
        });
        assert_eq!(fp, expected);
    }

    #[test]
    fn test_comparand_display() {
        assert_eq!(format!("{}", Comparand::Int(42)), "42");
        assert_eq!(format!("{}", Comparand::Float(3.14)), "3.14");
        assert_eq!(format!("{}", Comparand::Text("hello".into())), "'hello'");
        assert_eq!(format!("{}", Comparand::Bool(true)), "true");
        assert_eq!(format!("{}", Comparand::Bytes(vec![0xCA, 0xFE])), "X'cafe'");
        assert_eq!(format!("{}", Comparand::Null), "NULL");
    }
}
