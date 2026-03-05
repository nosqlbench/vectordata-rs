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
//! ```text
//! [PRED=0: u8][name_len: u16 LE][name: UTF-8][op: u8][comparand_count: i16 LE][comparands: i64 LE * n]
//! ```
//!
//! All PNode payloads are prefixed with a `DIALECT_PNODE` leader byte (`0x02`)
//! to identify the record type when stored alongside MNode records.

/// Dialect leader byte identifying PNode records.
pub const DIALECT_PNODE: u8 = 0x02;

pub mod vernacular;

// vernacular module provides to_cddl/to_cql/to_sql — used from tests

use std::fmt;
use std::io::{self, Cursor, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// Conjugate type discriminant — the first byte of every node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ConjugateType {
    Pred = 0,
    And = 1,
    Or = 2,
}

impl ConjugateType {
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

/// Comparison operator type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpType {
    Gt = 0,
    Lt = 1,
    Eq = 2,
    Ne = 3,
    Ge = 4,
    Le = 5,
    In = 6,
    Matches = 7,
}

impl OpType {
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

/// Field reference — either by index or by name
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldRef {
    Index(u8),
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

/// A predicate tree node
#[derive(Debug, Clone, PartialEq)]
pub enum PNode {
    /// Leaf predicate: field op comparands
    Predicate(PredicateNode),
    /// Boolean conjunction: AND or OR of children
    Conjugate(ConjugateNode),
}

/// Leaf node: `field op (v1, v2, ...)`
#[derive(Debug, Clone, PartialEq)]
pub struct PredicateNode {
    pub field: FieldRef,
    pub op: OpType,
    pub comparands: Vec<i64>,
}

/// Interior node: AND/OR of child nodes
#[derive(Debug, Clone, PartialEq)]
pub struct ConjugateNode {
    pub conjugate_type: ConjugateType,
    pub children: Vec<PNode>,
}

impl PNode {
    /// Encode to bytes (indexed mode — fields as u8 indices).
    ///
    /// Prepends the `DIALECT_PNODE` leader byte before the tree data.
    pub fn to_bytes_indexed(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(DIALECT_PNODE);
        self.write_indexed(&mut buf).expect("write to Vec should not fail");
        buf
    }

    /// Encode to bytes (named mode — fields as UTF-8 strings).
    ///
    /// Prepends the `DIALECT_PNODE` leader byte before the tree data.
    pub fn to_bytes_named(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(DIALECT_PNODE);
        self.write_named(&mut buf).expect("write to Vec should not fail");
        buf
    }

    /// Decode from bytes (indexed mode).
    ///
    /// Verifies and strips the `DIALECT_PNODE` leader byte before decoding.
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
        let mut cursor = Cursor::new(&data[1..]);
        Self::read_named(&mut cursor)
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
                    w.write_i64::<LittleEndian>(*v)?;
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
                    w.write_i64::<LittleEndian>(*v)?;
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
                    comparands.push(r.read_i64::<LittleEndian>()?);
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

    fn read_named(r: &mut Cursor<&[u8]>) -> io::Result<Self> {
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
                    comparands.push(r.read_i64::<LittleEndian>()?);
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
                    children.push(Self::read_named(r)?);
                }
                Ok(PNode::Conjugate(ConjugateNode {
                    conjugate_type: ctype,
                    children,
                }))
            }
        }
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
            comparands: vec![42],
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
            comparands: vec![1, 2, 3],
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
                    comparands: vec![10],
                }),
                PNode::Conjugate(ConjugateNode {
                    conjugate_type: ConjugateType::Or,
                    children: vec![
                        PNode::Predicate(PredicateNode {
                            field: FieldRef::Index(1),
                            op: OpType::Eq,
                            comparands: vec![5],
                        }),
                        PNode::Predicate(PredicateNode {
                            field: FieldRef::Index(2),
                            op: OpType::Le,
                            comparands: vec![100],
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
                    comparands: vec![18],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("score".into()),
                    op: OpType::Le,
                    comparands: vec![100],
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
            comparands: vec![42],
        });
        let bytes = node.to_bytes_indexed();
        assert_eq!(bytes.len(), 14);
    }
}
