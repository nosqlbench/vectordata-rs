// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! ANode — unified wrapper for MNode and PNode records.
//!
//! Provides a single `ANode` enum that can hold either an MNode or a PNode,
//! along with a Stage 1 binary codec that auto-detects the dialect from the
//! leader byte.
//!
//! ## Dialect leader bytes
//!
//! | Byte | Dialect |
//! |------|---------|
//! | 0x00 | Invalid |
//! | 0x01 | MNode   |
//! | 0x02 | PNode   |

use std::fmt;

use super::mnode::{self, MNode};
use super::pnode::{self, PNode};

/// Dialect leader byte: invalid / unrecognized.
pub const DIALECT_INVALID: u8 = 0x00;

/// Dialect leader byte: MNode (metadata record).
pub const DIALECT_MNODE: u8 = mnode::DIALECT_MNODE;

/// Dialect leader byte: PNode (predicate tree).
pub const DIALECT_PNODE: u8 = pnode::DIALECT_PNODE;

/// Any Node — unified wrapper for MNode and PNode records.
#[derive(Debug, Clone, PartialEq)]
pub enum ANode {
    /// Metadata record.
    MNode(MNode),
    /// Predicate tree.
    PNode(PNode),
}

impl fmt::Display for ANode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ANode::MNode(node) => write!(f, "{}", node),
            ANode::PNode(node) => write!(f, "{}", node),
        }
    }
}

/// Decode raw bytes, auto-detecting dialect from the leader byte.
pub fn decode(data: &[u8]) -> Result<ANode, String> {
    if data.is_empty() {
        return Err("empty data".into());
    }
    match data[0] {
        DIALECT_MNODE => decode_mnode(data),
        DIALECT_PNODE => decode_pnode(data),
        other => Err(format!("unknown dialect leader byte: 0x{:02x}", other)),
    }
}

/// Decode raw bytes, forcing MNode interpretation.
///
/// The data must still carry the `DIALECT_MNODE` leader byte.
pub fn decode_mnode(data: &[u8]) -> Result<ANode, String> {
    MNode::from_bytes(data)
        .map(ANode::MNode)
        .map_err(|e| format!("MNode decode error: {}", e))
}

/// Decode raw bytes, forcing PNode interpretation (named mode).
///
/// The data must still carry the `DIALECT_PNODE` leader byte.
pub fn decode_pnode(data: &[u8]) -> Result<ANode, String> {
    PNode::from_bytes_named(data)
        .map(ANode::PNode)
        .map_err(|e| format!("PNode decode error: {}", e))
}

/// Encode an ANode to bytes (includes leader byte).
pub fn encode(node: &ANode) -> Vec<u8> {
    match node {
        ANode::MNode(m) => m.to_bytes(),
        ANode::PNode(p) => p.to_bytes_named(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::mnode::{MNode, MValue};
    use crate::formats::pnode::{Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode};

    #[test]
    fn test_dialect_constants() {
        assert_eq!(DIALECT_INVALID, 0x00);
        assert_eq!(DIALECT_MNODE, 0x01);
        assert_eq!(DIALECT_PNODE, 0x02);
    }

    #[test]
    fn test_auto_detect_mnode() {
        let mut node = MNode::new();
        node.insert("x".into(), MValue::Int(42));
        let bytes = node.to_bytes();
        assert_eq!(bytes[0], DIALECT_MNODE);

        let decoded = decode(&bytes).unwrap();
        match decoded {
            ANode::MNode(m) => assert_eq!(m.fields["x"], MValue::Int(42)),
            _ => panic!("expected MNode"),
        }
    }

    #[test]
    fn test_auto_detect_pnode() {
        let node = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("age".into()),
            op: OpType::Gt,
            comparands: vec![Comparand::Int(18)],
        });
        let bytes = node.to_bytes_named();
        assert_eq!(bytes[0], DIALECT_PNODE);

        let decoded = decode(&bytes).unwrap();
        match decoded {
            ANode::PNode(p) => assert_eq!(p, node),
            _ => panic!("expected PNode"),
        }
    }

    #[test]
    fn test_encode_decode_roundtrip_mnode() {
        let mut node = MNode::new();
        node.insert("name".into(), MValue::Text("alice".into()));
        node.insert("score".into(), MValue::Float(99.5));
        let anode = ANode::MNode(node.clone());

        let bytes = encode(&anode);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded, anode);
    }

    #[test]
    fn test_encode_decode_roundtrip_pnode() {
        let tree = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("x".into()),
                    op: OpType::Eq,
                    comparands: vec![Comparand::Int(1)],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("y".into()),
                    op: OpType::Lt,
                    comparands: vec![Comparand::Int(10)],
                }),
            ],
        });
        let anode = ANode::PNode(tree);

        let bytes = encode(&anode);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded, anode);
    }

    #[test]
    fn test_unknown_dialect() {
        let data = vec![0xFF, 0x00, 0x01];
        assert!(decode(&data).is_err());
    }

    #[test]
    fn test_empty_data() {
        assert!(decode(&[]).is_err());
    }

    #[test]
    fn test_display() {
        let mut node = MNode::new();
        node.insert("k".into(), MValue::Int(1));
        let anode = ANode::MNode(node);
        let s = format!("{}", anode);
        assert!(s.contains("k: 1"));
    }
}
