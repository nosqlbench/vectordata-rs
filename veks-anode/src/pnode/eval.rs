// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Evaluate a [`PNode`] predicate tree against an [`MNode`] record.
//!
//! The core entry point is [`evaluate`], which recursively walks the predicate
//! tree and tests each leaf against the corresponding field in the MNode.

use crate::mnode::MValue;
use crate::mnode::MNode;
use crate::pnode::{
    Comparand, ConjugateType, FieldRef, OpType, PNode,
};

/// Evaluate a predicate tree against a single metadata record.
///
/// Returns `true` if the record satisfies the predicate, `false` otherwise.
pub fn evaluate(pnode: &PNode, mnode: &MNode) -> bool {
    match pnode {
        PNode::Conjugate(conj) => match conj.conjugate_type {
            ConjugateType::And => conj.children.iter().all(|c| evaluate(c, mnode)),
            ConjugateType::Or => conj.children.iter().any(|c| evaluate(c, mnode)),
            ConjugateType::Pred => false, // invalid tree structure
        },
        PNode::Predicate(pred) => {
            let value = match &pred.field {
                FieldRef::Named(name) => mnode.fields.get(name),
                FieldRef::Index(i) => mnode.fields.get_index(*i as usize).map(|(_, v)| v),
            };

            match value {
                None => {
                    // Missing field: matches Eq Null, mismatches everything else
                    match pred.op {
                        OpType::Eq => pred.comparands.iter().any(|c| matches!(c, Comparand::Null)),
                        OpType::In => pred.comparands.iter().any(|c| matches!(c, Comparand::Null)),
                        OpType::Ne => pred.comparands.iter().all(|c| !matches!(c, Comparand::Null)),
                        _ => false,
                    }
                }
                Some(mv) => eval_predicate(mv, pred.op, &pred.comparands),
            }
        }
    }
}

/// Evaluate a single predicate leaf: `field_value op comparands`.
fn eval_predicate(mv: &MValue, op: OpType, comparands: &[Comparand]) -> bool {
    match op {
        OpType::In => comparands.iter().any(|c| compare_eq(mv, c)),
        OpType::Matches => false, // unsupported
        OpType::Eq => {
            assert!(!comparands.is_empty());
            compare_eq(mv, &comparands[0])
        }
        OpType::Ne => {
            assert!(!comparands.is_empty());
            !compare_eq(mv, &comparands[0])
        }
        OpType::Gt | OpType::Lt | OpType::Ge | OpType::Le => {
            assert!(!comparands.is_empty());
            match compare_ord(mv, &comparands[0]) {
                None => false,
                Some(ord) => match op {
                    OpType::Gt => ord == std::cmp::Ordering::Greater,
                    OpType::Lt => ord == std::cmp::Ordering::Less,
                    OpType::Ge => ord != std::cmp::Ordering::Less,
                    OpType::Le => ord != std::cmp::Ordering::Greater,
                    _ => unreachable!(),
                },
            }
        }
    }
}

/// Test equality between an MValue and a Comparand, with type coercion.
fn compare_eq(mv: &MValue, c: &Comparand) -> bool {
    match (mv, c) {
        // Null
        (MValue::Null, Comparand::Null) => true,
        (MValue::Null, _) | (_, Comparand::Null) => false,

        // Bool — only exact match
        (MValue::Bool(a), Comparand::Bool(b)) => a == b,

        // Bytes — only exact match
        (MValue::Bytes(a), Comparand::Bytes(b)) => a == b,

        // Text types vs Text comparand
        (MValue::Text(a), Comparand::Text(b))
        | (MValue::Ascii(a), Comparand::Text(b))
        | (MValue::EnumStr(a), Comparand::Text(b)) => a == b,

        // Int types vs Int comparand
        (mv, Comparand::Int(ci)) if is_int_type(mv) => {
            extract_i64(mv) == Some(*ci)
        }

        // Float types vs Float comparand
        (mv, Comparand::Float(cf)) if is_float_type(mv) => {
            extract_f64(mv) == Some(*cf)
        }

        // Cross-type numeric: Int MValue vs Float comparand
        (mv, Comparand::Float(cf)) if is_int_type(mv) => {
            extract_i64(mv).map(|v| v as f64) == Some(*cf)
        }

        // Cross-type numeric: Float MValue vs Int comparand
        (mv, Comparand::Int(ci)) if is_float_type(mv) => {
            extract_f64(mv) == Some(*ci as f64)
        }

        // All other type combinations are mismatches
        _ => false,
    }
}

/// Compare an MValue with a Comparand for ordering, with type coercion.
///
/// Returns `None` for incompatible types or types that don't support ordering.
fn compare_ord(mv: &MValue, c: &Comparand) -> Option<std::cmp::Ordering> {
    match (mv, c) {
        // Text types vs Text comparand
        (MValue::Text(a), Comparand::Text(b))
        | (MValue::Ascii(a), Comparand::Text(b))
        | (MValue::EnumStr(a), Comparand::Text(b)) => Some(a.as_str().cmp(b.as_str())),

        // Int types vs Int comparand
        (mv, Comparand::Int(ci)) if is_int_type(mv) => {
            extract_i64(mv).map(|v| v.cmp(ci))
        }

        // Float types vs Float comparand
        (mv, Comparand::Float(cf)) if is_float_type(mv) => {
            extract_f64(mv).and_then(|v| v.partial_cmp(cf))
        }

        // Cross-type numeric: Int MValue vs Float comparand
        (mv, Comparand::Float(cf)) if is_int_type(mv) => {
            extract_i64(mv).and_then(|v| (v as f64).partial_cmp(cf))
        }

        // Cross-type numeric: Float MValue vs Int comparand
        (mv, Comparand::Int(ci)) if is_float_type(mv) => {
            extract_f64(mv).and_then(|v| v.partial_cmp(&(*ci as f64)))
        }

        // All other combinations — no ordering
        _ => None,
    }
}

/// Check if an MValue is an integer-family type.
fn is_int_type(mv: &MValue) -> bool {
    matches!(mv, MValue::Int(_) | MValue::Int32(_) | MValue::Short(_) | MValue::Millis(_))
}

/// Check if an MValue is a float-family type.
fn is_float_type(mv: &MValue) -> bool {
    matches!(mv, MValue::Float(_) | MValue::Float32(_))
}

/// Extract the i64 representation from an integer-family MValue.
fn extract_i64(mv: &MValue) -> Option<i64> {
    match mv {
        MValue::Int(v) => Some(*v),
        MValue::Int32(v) => Some(*v as i64),
        MValue::Short(v) => Some(*v as i64),
        MValue::Millis(v) => Some(*v),
        _ => None,
    }
}

/// Extract the f64 representation from a float-family MValue.
fn extract_f64(mv: &MValue) -> Option<f64> {
    match mv {
        MValue::Float(v) => Some(*v),
        MValue::Float32(v) => Some(*v as f64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pnode::{ConjugateNode, PredicateNode};

    /// Build a simple leaf predicate with a named field.
    fn pred(name: &str, op: OpType, comparands: Vec<Comparand>) -> PNode {
        PNode::Predicate(PredicateNode {
            field: FieldRef::Named(name.to_string()),
            op,
            comparands,
        })
    }

    /// Build a single-field MNode.
    fn mnode_one(name: &str, val: MValue) -> MNode {
        let mut m = MNode::new();
        m.insert(name.to_string(), val);
        m
    }

    // -- Eq --

    #[test]
    fn test_eq_int() {
        let p = pred("x", OpType::Eq, vec![Comparand::Int(42)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Int(42))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Int(99))));
    }

    #[test]
    fn test_eq_float() {
        let p = pred("x", OpType::Eq, vec![Comparand::Float(3.14)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Float(3.14))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Float(2.71))));
    }

    #[test]
    fn test_eq_text() {
        let p = pred("x", OpType::Eq, vec![Comparand::Text("hello".into())]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Text("hello".into()))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Text("world".into()))));
    }

    #[test]
    fn test_eq_bool() {
        let p = pred("x", OpType::Eq, vec![Comparand::Bool(true)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Bool(true))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Bool(false))));
    }

    #[test]
    fn test_eq_null() {
        let p = pred("x", OpType::Eq, vec![Comparand::Null]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Null)));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Int(1))));
    }

    #[test]
    fn test_eq_bytes() {
        let p = pred("x", OpType::Eq, vec![Comparand::Bytes(vec![1, 2, 3])]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Bytes(vec![1, 2, 3]))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Bytes(vec![4, 5]))));
    }

    // -- Ne --

    #[test]
    fn test_ne_int() {
        let p = pred("x", OpType::Ne, vec![Comparand::Int(42)]);
        assert!(!evaluate(&p, &mnode_one("x", MValue::Int(42))));
        assert!(evaluate(&p, &mnode_one("x", MValue::Int(99))));
    }

    #[test]
    fn test_ne_null() {
        let p = pred("x", OpType::Ne, vec![Comparand::Null]);
        assert!(!evaluate(&p, &mnode_one("x", MValue::Null)));
        assert!(evaluate(&p, &mnode_one("x", MValue::Int(1))));
    }

    // -- Gt / Lt / Ge / Le --

    #[test]
    fn test_gt_int() {
        let p = pred("x", OpType::Gt, vec![Comparand::Int(10)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Int(11))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Int(10))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Int(9))));
    }

    #[test]
    fn test_lt_float() {
        let p = pred("x", OpType::Lt, vec![Comparand::Float(5.0)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Float(4.9))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Float(5.0))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Float(5.1))));
    }

    #[test]
    fn test_ge_int() {
        let p = pred("x", OpType::Ge, vec![Comparand::Int(10)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Int(10))));
        assert!(evaluate(&p, &mnode_one("x", MValue::Int(11))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Int(9))));
    }

    #[test]
    fn test_le_float() {
        let p = pred("x", OpType::Le, vec![Comparand::Float(5.0)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Float(5.0))));
        assert!(evaluate(&p, &mnode_one("x", MValue::Float(4.9))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Float(5.1))));
    }

    #[test]
    fn test_gt_text() {
        let p = pred("x", OpType::Gt, vec![Comparand::Text("bob".into())]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Text("charlie".into()))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Text("alice".into()))));
    }

    // -- In --

    #[test]
    fn test_in_int() {
        let p = pred("x", OpType::In, vec![
            Comparand::Int(1), Comparand::Int(3), Comparand::Int(5),
        ]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Int(3))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Int(2))));
    }

    #[test]
    fn test_in_text() {
        let p = pred("x", OpType::In, vec![
            Comparand::Text("a".into()), Comparand::Text("b".into()),
        ]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Text("b".into()))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Text("c".into()))));
    }

    // -- Missing field --

    #[test]
    fn test_missing_field_eq_null() {
        let p = pred("missing", OpType::Eq, vec![Comparand::Null]);
        let m = mnode_one("other", MValue::Int(1));
        assert!(evaluate(&p, &m));
    }

    #[test]
    fn test_missing_field_eq_int() {
        let p = pred("missing", OpType::Eq, vec![Comparand::Int(42)]);
        let m = mnode_one("other", MValue::Int(1));
        assert!(!evaluate(&p, &m));
    }

    #[test]
    fn test_missing_field_ne_null() {
        let p = pred("missing", OpType::Ne, vec![Comparand::Null]);
        let m = mnode_one("other", MValue::Int(1));
        assert!(!evaluate(&p, &m));
    }

    #[test]
    fn test_missing_field_gt() {
        let p = pred("missing", OpType::Gt, vec![Comparand::Int(0)]);
        let m = mnode_one("other", MValue::Int(1));
        assert!(!evaluate(&p, &m));
    }

    // -- Explicit MValue::Null --

    #[test]
    fn test_explicit_null_eq_null() {
        let p = pred("x", OpType::Eq, vec![Comparand::Null]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Null)));
    }

    #[test]
    fn test_explicit_null_ne_null() {
        let p = pred("x", OpType::Ne, vec![Comparand::Null]);
        assert!(!evaluate(&p, &mnode_one("x", MValue::Null)));
    }

    #[test]
    fn test_explicit_null_gt() {
        let p = pred("x", OpType::Gt, vec![Comparand::Int(0)]);
        assert!(!evaluate(&p, &mnode_one("x", MValue::Null)));
    }

    // -- Conjugate trees --

    #[test]
    fn test_and_conjugate() {
        let tree = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                pred("x", OpType::Gt, vec![Comparand::Int(5)]),
                pred("y", OpType::Eq, vec![Comparand::Text("ok".into())]),
            ],
        });
        let mut m = MNode::new();
        m.insert("x".into(), MValue::Int(10));
        m.insert("y".into(), MValue::Text("ok".into()));
        assert!(evaluate(&tree, &m));

        let mut m2 = MNode::new();
        m2.insert("x".into(), MValue::Int(10));
        m2.insert("y".into(), MValue::Text("no".into()));
        assert!(!evaluate(&tree, &m2));
    }

    #[test]
    fn test_or_conjugate() {
        let tree = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::Or,
            children: vec![
                pred("x", OpType::Eq, vec![Comparand::Int(1)]),
                pred("x", OpType::Eq, vec![Comparand::Int(2)]),
            ],
        });
        assert!(evaluate(&tree, &mnode_one("x", MValue::Int(1))));
        assert!(evaluate(&tree, &mnode_one("x", MValue::Int(2))));
        assert!(!evaluate(&tree, &mnode_one("x", MValue::Int(3))));
    }

    // -- Cross-width coercion --

    #[test]
    fn test_int32_vs_comparand_int() {
        let p = pred("x", OpType::Eq, vec![Comparand::Int(42)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Int32(42))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Int32(99))));
    }

    #[test]
    fn test_short_vs_comparand_int() {
        let p = pred("x", OpType::Gt, vec![Comparand::Int(10)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Short(11))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Short(10))));
    }

    #[test]
    fn test_millis_vs_comparand_int() {
        let p = pred("x", OpType::Le, vec![Comparand::Int(1000)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Millis(1000))));
        assert!(evaluate(&p, &mnode_one("x", MValue::Millis(999))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Millis(1001))));
    }

    #[test]
    fn test_float32_vs_comparand_float() {
        let p = pred("x", OpType::Lt, vec![Comparand::Float(5.0)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Float32(4.5))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Float32(5.5))));
    }

    // -- Cross-type numeric --

    #[test]
    fn test_int_vs_comparand_float() {
        let p = pred("x", OpType::Eq, vec![Comparand::Float(42.0)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Int(42))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Int(43))));
    }

    #[test]
    fn test_float_vs_comparand_int() {
        let p = pred("x", OpType::Eq, vec![Comparand::Int(42)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Float(42.0))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Float(42.5))));
    }

    #[test]
    fn test_int_vs_float_ordering() {
        let p = pred("x", OpType::Gt, vec![Comparand::Float(10.5)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Int(11))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Int(10))));
    }

    #[test]
    fn test_float_vs_int_ordering() {
        let p = pred("x", OpType::Lt, vec![Comparand::Int(10)]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Float(9.5))));
        assert!(!evaluate(&p, &mnode_one("x", MValue::Float(10.5))));
    }

    // -- Text subtypes --

    #[test]
    fn test_ascii_vs_text() {
        let p = pred("x", OpType::Eq, vec![Comparand::Text("hello".into())]);
        assert!(evaluate(&p, &mnode_one("x", MValue::Ascii("hello".into()))));
    }

    #[test]
    fn test_enumstr_vs_text() {
        let p = pred("x", OpType::Eq, vec![Comparand::Text("red".into())]);
        assert!(evaluate(&p, &mnode_one("x", MValue::EnumStr("red".into()))));
    }

    // -- Cross-type mismatch --

    #[test]
    fn test_text_vs_int_mismatch() {
        let p = pred("x", OpType::Eq, vec![Comparand::Int(42)]);
        assert!(!evaluate(&p, &mnode_one("x", MValue::Text("42".into()))));
    }

    #[test]
    fn test_bool_vs_int_mismatch() {
        let p = pred("x", OpType::Eq, vec![Comparand::Int(1)]);
        assert!(!evaluate(&p, &mnode_one("x", MValue::Bool(true))));
    }

    // -- FieldRef::Index --

    #[test]
    fn test_field_ref_index() {
        let p = PNode::Predicate(PredicateNode {
            field: FieldRef::Index(0),
            op: OpType::Eq,
            comparands: vec![Comparand::Int(42)],
        });
        let mut m = MNode::new();
        m.insert("first".into(), MValue::Int(42));
        m.insert("second".into(), MValue::Int(99));
        assert!(evaluate(&p, &m));

        let p1 = PNode::Predicate(PredicateNode {
            field: FieldRef::Index(1),
            op: OpType::Eq,
            comparands: vec![Comparand::Int(99)],
        });
        assert!(evaluate(&p1, &m));
    }

    // -- Matches (unsupported) --

    #[test]
    fn test_matches_unsupported() {
        let p = pred("x", OpType::Matches, vec![Comparand::Text(".*".into())]);
        assert!(!evaluate(&p, &mnode_one("x", MValue::Text("anything".into()))));
    }
}
