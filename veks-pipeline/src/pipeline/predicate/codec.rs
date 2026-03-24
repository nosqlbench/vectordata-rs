// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Binary serialization for predicate trees.
//!
//! Uses a length-prefixed binary format:
//! - Each `PNode` starts with a 1-byte type tag (0 = Predicate, 1 = Conjugate).
//! - `PredicateNode`: field_index (u32 LE), op_tag (u8), value_count (u32 LE),
//!   then value_count × i64 LE values.
//! - `ConjugateNode`: logic_op (u8), child_count (u32 LE), then child_count
//!   serialized child `PNode`s.
//!
//! A sequence of predicates (one per query) is written as:
//! query_count (u32 LE), then for each query: byte_length (u32 LE) + serialized PNode.

use std::io::{self, Read, Write};

use super::predicate::{ConjugateNode, LogicOp, OpType, PNode, PredicateNode};

const TAG_PREDICATE: u8 = 0;
const TAG_CONJUGATE: u8 = 1;

/// Serialize a `PNode` to a writer.
pub fn write_pnode(w: &mut impl Write, pnode: &PNode) -> io::Result<()> {
    match pnode {
        PNode::Predicate(pred) => {
            w.write_all(&[TAG_PREDICATE])?;
            w.write_all(&pred.field_index.to_le_bytes())?;
            w.write_all(&[pred.op.to_tag()])?;
            w.write_all(&(pred.values.len() as u32).to_le_bytes())?;
            for &v in &pred.values {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        PNode::Conjugate(conj) => {
            w.write_all(&[TAG_CONJUGATE])?;
            w.write_all(&[conj.op.to_tag()])?;
            w.write_all(&(conj.children.len() as u32).to_le_bytes())?;
            for child in &conj.children {
                write_pnode(w, child)?;
            }
        }
    }
    Ok(())
}

/// Deserialize a `PNode` from a reader.
pub fn read_pnode(r: &mut impl Read) -> io::Result<PNode> {
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag)?;

    match tag[0] {
        TAG_PREDICATE => {
            let mut buf4 = [0u8; 4];
            r.read_exact(&mut buf4)?;
            let field_index = u32::from_le_bytes(buf4);

            let mut op_tag = [0u8; 1];
            r.read_exact(&mut op_tag)?;
            let op = OpType::from_tag(op_tag[0]).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid op tag")
            })?;

            r.read_exact(&mut buf4)?;
            let value_count = u32::from_le_bytes(buf4) as usize;

            let mut values = Vec::with_capacity(value_count);
            let mut buf8 = [0u8; 8];
            for _ in 0..value_count {
                r.read_exact(&mut buf8)?;
                values.push(i64::from_le_bytes(buf8));
            }

            Ok(PNode::Predicate(PredicateNode {
                field_index,
                op,
                values,
            }))
        }
        TAG_CONJUGATE => {
            let mut logic_tag = [0u8; 1];
            r.read_exact(&mut logic_tag)?;
            let op = LogicOp::from_tag(logic_tag[0]).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid logic op tag")
            })?;

            let mut buf4 = [0u8; 4];
            r.read_exact(&mut buf4)?;
            let child_count = u32::from_le_bytes(buf4) as usize;

            let mut children = Vec::with_capacity(child_count);
            for _ in 0..child_count {
                children.push(read_pnode(r)?);
            }

            Ok(PNode::Conjugate(ConjugateNode { op, children }))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown PNode tag: {}", tag[0]),
        )),
    }
}

/// Serialize a PNode to a byte vector.
pub fn encode_pnode(pnode: &PNode) -> Vec<u8> {
    let mut buf = Vec::new();
    write_pnode(&mut buf, pnode).expect("in-memory write cannot fail");
    buf
}

/// Deserialize a PNode from a byte slice.
pub fn decode_pnode(data: &[u8]) -> io::Result<PNode> {
    let mut cursor = io::Cursor::new(data);
    read_pnode(&mut cursor)
}

/// Write a sequence of predicates (one per query) to a writer.
///
/// Format: query_count (u32 LE), then for each:
///   byte_length (u32 LE) + serialized PNode bytes.
pub fn write_predicates(w: &mut impl Write, predicates: &[PNode]) -> io::Result<()> {
    w.write_all(&(predicates.len() as u32).to_le_bytes())?;
    for pred in predicates {
        let encoded = encode_pnode(pred);
        w.write_all(&(encoded.len() as u32).to_le_bytes())?;
        w.write_all(&encoded)?;
    }
    Ok(())
}

/// Read a sequence of predicates from a reader.
pub fn read_predicates(r: &mut impl Read) -> io::Result<Vec<PNode>> {
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let count = u32::from_le_bytes(buf4) as usize;

    let mut predicates = Vec::with_capacity(count);
    for _ in 0..count {
        r.read_exact(&mut buf4)?;
        let len = u32::from_le_bytes(buf4) as usize;
        let mut data = vec![0u8; len];
        r.read_exact(&mut data)?;
        predicates.push(decode_pnode(&data)?);
    }
    Ok(predicates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnode_roundtrip_simple() {
        let pred = PNode::Predicate(PredicateNode {
            field_index: 3,
            op: OpType::Ge,
            values: vec![42],
        });

        let encoded = encode_pnode(&pred);
        let decoded = decode_pnode(&encoded).unwrap();

        if let PNode::Predicate(p) = decoded {
            assert_eq!(p.field_index, 3);
            assert_eq!(p.op, OpType::Ge);
            assert_eq!(p.values, vec![42]);
        } else {
            panic!("expected Predicate");
        }
    }

    #[test]
    fn test_pnode_roundtrip_conjugate() {
        let pred = PNode::Conjugate(ConjugateNode {
            op: LogicOp::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field_index: 0,
                    op: OpType::Eq,
                    values: vec![10],
                }),
                PNode::Predicate(PredicateNode {
                    field_index: 1,
                    op: OpType::In,
                    values: vec![1, 2, 3],
                }),
            ],
        });

        let encoded = encode_pnode(&pred);
        let decoded = decode_pnode(&encoded).unwrap();

        if let PNode::Conjugate(c) = decoded {
            assert_eq!(c.op, LogicOp::And);
            assert_eq!(c.children.len(), 2);
        } else {
            panic!("expected Conjugate");
        }
    }

    #[test]
    fn test_predicates_sequence_roundtrip() {
        let preds = vec![
            PNode::Predicate(PredicateNode {
                field_index: 0,
                op: OpType::Eq,
                values: vec![1],
            }),
            PNode::Conjugate(ConjugateNode {
                op: LogicOp::Or,
                children: vec![
                    PNode::Predicate(PredicateNode {
                        field_index: 0,
                        op: OpType::Lt,
                        values: vec![5],
                    }),
                    PNode::Predicate(PredicateNode {
                        field_index: 1,
                        op: OpType::Gt,
                        values: vec![10],
                    }),
                ],
            }),
        ];

        let mut buf = Vec::new();
        write_predicates(&mut buf, &preds).unwrap();

        let mut cursor = io::Cursor::new(&buf);
        let decoded = read_predicates(&mut cursor).unwrap();
        assert_eq!(decoded.len(), 2);
    }

    #[test]
    fn test_in_predicate_values_preserved() {
        let pred = PNode::Predicate(PredicateNode {
            field_index: 2,
            op: OpType::In,
            values: vec![100, 200, 300, 400, 500],
        });

        let decoded = decode_pnode(&encode_pnode(&pred)).unwrap();
        if let PNode::Predicate(p) = decoded {
            assert_eq!(p.values, vec![100, 200, 300, 400, 500]);
        } else {
            panic!("expected Predicate");
        }
    }
}
