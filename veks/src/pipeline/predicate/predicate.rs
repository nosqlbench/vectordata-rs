// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Predicate tree types and evaluation for predicated datasets.
//!
//! Defines the predicate expression tree (`PNode`) composed of leaf
//! `PredicateNode` (field comparisons) and internal `ConjugateNode`
//! (AND/OR combinators), plus the `evaluate` function that tests
//! a predicate against attribute columns at a given vector index.

use super::attribute::AttributeColumn;

/// Comparison operator for a predicate leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpType {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Greater than.
    Gt,
    /// Less than.
    Lt,
    /// Greater than or equal.
    Ge,
    /// Less than or equal.
    Le,
    /// Set membership (value IN set).
    In,
}

impl OpType {
    /// Parse an operator from a string.
    pub fn from_str(s: &str) -> Option<OpType> {
        match s.to_uppercase().as_str() {
            "EQ" | "=" | "==" => Some(OpType::Eq),
            "NE" | "!=" | "<>" => Some(OpType::Ne),
            "GT" | ">" => Some(OpType::Gt),
            "LT" | "<" => Some(OpType::Lt),
            "GE" | ">=" => Some(OpType::Ge),
            "LE" | "<=" => Some(OpType::Le),
            "IN" => Some(OpType::In),
            _ => None,
        }
    }

    /// Encode operator as a u8 tag for binary serialization.
    pub fn to_tag(&self) -> u8 {
        match self {
            OpType::Eq => 0,
            OpType::Ne => 1,
            OpType::Gt => 2,
            OpType::Lt => 3,
            OpType::Ge => 4,
            OpType::Le => 5,
            OpType::In => 6,
        }
    }

    /// Decode operator from a u8 tag.
    pub fn from_tag(tag: u8) -> Option<OpType> {
        match tag {
            0 => Some(OpType::Eq),
            1 => Some(OpType::Ne),
            2 => Some(OpType::Gt),
            3 => Some(OpType::Lt),
            4 => Some(OpType::Ge),
            5 => Some(OpType::Le),
            6 => Some(OpType::In),
            _ => None,
        }
    }
}

/// Leaf predicate: a comparison on a single field.
#[derive(Debug, Clone)]
pub struct PredicateNode {
    /// Index into the attribute schema's field list.
    pub field_index: u32,
    /// Comparison operator.
    pub op: OpType,
    /// Comparison value(s). For scalar ops (Eq, Ne, Gt, Lt, Ge, Le),
    /// only `values[0]` is used. For In, all values form the membership set.
    pub values: Vec<i64>,
}

/// Logical combinator type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogicOp {
    /// All children must be true.
    And,
    /// At least one child must be true.
    Or,
}

impl LogicOp {
    /// Encode as u8 tag.
    pub fn to_tag(&self) -> u8 {
        match self {
            LogicOp::And => 0,
            LogicOp::Or => 1,
        }
    }

    /// Decode from u8 tag.
    pub fn from_tag(tag: u8) -> Option<LogicOp> {
        match tag {
            0 => Some(LogicOp::And),
            1 => Some(LogicOp::Or),
            _ => None,
        }
    }
}

/// Internal combinator node with children.
#[derive(Debug, Clone)]
pub struct ConjugateNode {
    /// Logical operator.
    pub op: LogicOp,
    /// Child predicate nodes.
    pub children: Vec<PNode>,
}

/// A predicate expression tree node.
#[derive(Debug, Clone)]
pub enum PNode {
    /// Leaf: field comparison.
    Predicate(PredicateNode),
    /// Internal: logical combinator.
    Conjugate(ConjugateNode),
}

/// Evaluate a predicate tree against attribute columns at a given vector index.
///
/// Returns `true` if the vector at `vector_idx` satisfies the predicate.
pub fn evaluate(pnode: &PNode, columns: &[AttributeColumn], vector_idx: usize) -> bool {
    match pnode {
        PNode::Predicate(pred) => evaluate_leaf(pred, columns, vector_idx),
        PNode::Conjugate(conj) => match conj.op {
            LogicOp::And => conj
                .children
                .iter()
                .all(|c| evaluate(c, columns, vector_idx)),
            LogicOp::Or => conj
                .children
                .iter()
                .any(|c| evaluate(c, columns, vector_idx)),
        },
    }
}

/// Evaluate a leaf predicate against columns.
fn evaluate_leaf(pred: &PredicateNode, columns: &[AttributeColumn], idx: usize) -> bool {
    let fi = pred.field_index as usize;
    if fi >= columns.len() {
        return false;
    }

    match &columns[fi] {
        AttributeColumn::Int(vals) => {
            let val = vals[idx] as i64;
            eval_scalar(val, &pred.op, &pred.values)
        }
        AttributeColumn::Long(vals) => {
            let val = vals[idx];
            eval_scalar(val, &pred.op, &pred.values)
        }
        AttributeColumn::Enum(vals) => {
            let val = vals[idx] as i64;
            eval_scalar(val, &pred.op, &pred.values)
        }
        AttributeColumn::EnumSet(vals) => {
            let bitmask = &vals[idx];
            eval_enum_set(bitmask, &pred.op, &pred.values)
        }
    }
}

/// Evaluate a scalar comparison.
fn eval_scalar(val: i64, op: &OpType, operands: &[i64]) -> bool {
    match op {
        OpType::Eq => !operands.is_empty() && val == operands[0],
        OpType::Ne => !operands.is_empty() && val != operands[0],
        OpType::Gt => !operands.is_empty() && val > operands[0],
        OpType::Lt => !operands.is_empty() && val < operands[0],
        OpType::Ge => !operands.is_empty() && val >= operands[0],
        OpType::Le => !operands.is_empty() && val <= operands[0],
        OpType::In => operands.contains(&val),
    }
}

/// Evaluate an enum-set comparison.
///
/// For enum sets, the predicate values are enum indices. We check whether
/// the bitmask has the corresponding bit set.
fn eval_enum_set(bitmask: &[u8], op: &OpType, operands: &[i64]) -> bool {
    match op {
        OpType::Eq | OpType::In => {
            // All specified values must be present in the set
            operands.iter().all(|&v| {
                let byte_idx = v as usize / 8;
                let bit_idx = v as usize % 8;
                byte_idx < bitmask.len() && (bitmask[byte_idx] & (1 << bit_idx)) != 0
            })
        }
        OpType::Ne => {
            // At least one specified value must be absent
            operands.iter().any(|&v| {
                let byte_idx = v as usize / 8;
                let bit_idx = v as usize % 8;
                byte_idx >= bitmask.len() || (bitmask[byte_idx] & (1 << bit_idx)) == 0
            })
        }
        // Gt/Lt/Ge/Le: compare popcount (number of set bits)
        OpType::Gt | OpType::Lt | OpType::Ge | OpType::Le => {
            let popcount: i64 = bitmask.iter().map(|b| b.count_ones() as i64).sum();
            eval_scalar(popcount, op, operands)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_type_roundtrip() {
        for op in &[
            OpType::Eq,
            OpType::Ne,
            OpType::Gt,
            OpType::Lt,
            OpType::Ge,
            OpType::Le,
            OpType::In,
        ] {
            assert_eq!(OpType::from_tag(op.to_tag()).as_ref(), Some(op));
        }
    }

    #[test]
    fn test_evaluate_int_eq() {
        let columns = vec![AttributeColumn::Int(vec![10, 20, 30])];
        let pred = PNode::Predicate(PredicateNode {
            field_index: 0,
            op: OpType::Eq,
            values: vec![20],
        });
        assert!(!evaluate(&pred, &columns, 0));
        assert!(evaluate(&pred, &columns, 1));
        assert!(!evaluate(&pred, &columns, 2));
    }

    #[test]
    fn test_evaluate_int_range() {
        let columns = vec![AttributeColumn::Int(vec![5, 15, 25])];
        let gt10 = PNode::Predicate(PredicateNode {
            field_index: 0,
            op: OpType::Gt,
            values: vec![10],
        });
        assert!(!evaluate(&gt10, &columns, 0));
        assert!(evaluate(&gt10, &columns, 1));
        assert!(evaluate(&gt10, &columns, 2));
    }

    #[test]
    fn test_evaluate_in() {
        let columns = vec![AttributeColumn::Enum(vec![0, 1, 2, 3])];
        let pred = PNode::Predicate(PredicateNode {
            field_index: 0,
            op: OpType::In,
            values: vec![1, 3],
        });
        assert!(!evaluate(&pred, &columns, 0));
        assert!(evaluate(&pred, &columns, 1));
        assert!(!evaluate(&pred, &columns, 2));
        assert!(evaluate(&pred, &columns, 3));
    }

    #[test]
    fn test_evaluate_and() {
        let columns = vec![
            AttributeColumn::Int(vec![5, 15, 25]),
            AttributeColumn::Int(vec![100, 200, 300]),
        ];
        let pred = PNode::Conjugate(ConjugateNode {
            op: LogicOp::And,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field_index: 0,
                    op: OpType::Gt,
                    values: vec![10],
                }),
                PNode::Predicate(PredicateNode {
                    field_index: 1,
                    op: OpType::Lt,
                    values: vec![250],
                }),
            ],
        });
        assert!(!evaluate(&pred, &columns, 0)); // 5 > 10 fails
        assert!(evaluate(&pred, &columns, 1)); // 15 > 10 && 200 < 250
        assert!(!evaluate(&pred, &columns, 2)); // 300 < 250 fails
    }

    #[test]
    fn test_evaluate_or() {
        let columns = vec![AttributeColumn::Int(vec![5, 15, 25])];
        let pred = PNode::Conjugate(ConjugateNode {
            op: LogicOp::Or,
            children: vec![
                PNode::Predicate(PredicateNode {
                    field_index: 0,
                    op: OpType::Eq,
                    values: vec![5],
                }),
                PNode::Predicate(PredicateNode {
                    field_index: 0,
                    op: OpType::Eq,
                    values: vec![25],
                }),
            ],
        });
        assert!(evaluate(&pred, &columns, 0));
        assert!(!evaluate(&pred, &columns, 1));
        assert!(evaluate(&pred, &columns, 2));
    }

    #[test]
    fn test_evaluate_enum_set() {
        // Bitmask: bit 0 and bit 2 set = 0b00000101
        let columns = vec![AttributeColumn::EnumSet(vec![
            vec![0b00000101], // {0, 2}
            vec![0b00000010], // {1}
        ])];

        // Check if value 0 is in the set
        let pred = PNode::Predicate(PredicateNode {
            field_index: 0,
            op: OpType::In,
            values: vec![0],
        });
        assert!(evaluate(&pred, &columns, 0)); // {0,2} contains 0
        assert!(!evaluate(&pred, &columns, 1)); // {1} does not contain 0
    }
}
