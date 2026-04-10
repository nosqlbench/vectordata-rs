// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Vernacular adapters for PNode — human-readable representations in SQL, CQL,
//! and CDDL syntax.
//!
//! These adapters convert predicate trees into native query language syntax for
//! troubleshooting, visualization, and schema documentation.

use super::{Comparand, ConjugateType, FieldRef, OpType, PNode};

// -- SQL vernacular -----------------------------------------------------------

/// Render a PNode as a SQL WHERE clause expression
pub fn to_sql(node: &PNode) -> String {
    sql_expr(node)
}

/// Format a comparand for SQL output.
fn sql_comparand(c: &Comparand) -> String {
    match c {
        Comparand::Int(v) => v.to_string(),
        Comparand::Float(v) => {
            let s = v.to_string();
            if s.contains('.') { s } else { format!("{}.0", s) }
        }
        Comparand::Text(s) => format!("'{}'", s.replace('\'', "''")),
        Comparand::Bool(b) => if *b { "TRUE".into() } else { "FALSE".into() },
        Comparand::Bytes(b) => format!("X'{}'", b.iter().map(|x| format!("{:02x}", x)).collect::<String>()),
        Comparand::Null => "NULL".into(),
    }
}

fn sql_expr(node: &PNode) -> String {
    match node {
        PNode::Predicate(pred) => {
            let field = sql_field(&pred.field);
            match pred.op {
                OpType::In => {
                    let vals: Vec<String> = pred.comparands.iter().map(sql_comparand).collect();
                    format!("{} IN ({})", field, vals.join(", "))
                }
                OpType::Matches => {
                    if let Some(v) = pred.comparands.first() {
                        format!("{} ~ {}", field, sql_comparand(v))
                    } else {
                        format!("{} ~ ''", field)
                    }
                }
                _ => {
                    if pred.comparands.len() == 1 {
                        format!("{} {} {}", field, pred.op.symbol(), sql_comparand(&pred.comparands[0]))
                    } else {
                        let vals: Vec<String> =
                            pred.comparands.iter().map(sql_comparand).collect();
                        format!("{} {} ({})", field, pred.op.symbol(), vals.join(", "))
                    }
                }
            }
        }
        PNode::Conjugate(conj) => {
            let op = match conj.conjugate_type {
                ConjugateType::And => "AND",
                ConjugateType::Or => "OR",
                _ => unreachable!(),
            };
            let parts: Vec<String> = conj.children.iter().map(sql_expr).collect();
            if parts.len() == 1 {
                parts[0].clone()
            } else {
                format!("({})", parts.join(&format!(" {} ", op)))
            }
        }
    }
}

fn sql_field(field: &FieldRef) -> String {
    match field {
        FieldRef::Index(i) => format!("field_{}", i),
        FieldRef::Named(s) => s.clone(),
    }
}

// -- CQL vernacular -----------------------------------------------------------

/// Render a PNode as a CQL WHERE clause expression
pub fn to_cql(node: &PNode) -> String {
    cql_expr(node)
}

/// Format a comparand for CQL output.
fn cql_comparand(c: &Comparand) -> String {
    match c {
        Comparand::Int(v) => v.to_string(),
        Comparand::Float(v) => {
            let s = v.to_string();
            if s.contains('.') { s } else { format!("{}.0", s) }
        }
        Comparand::Text(s) => format!("'{}'", s.replace('\'', "''")),
        Comparand::Bool(b) => if *b { "true".into() } else { "false".into() },
        Comparand::Bytes(b) => format!("0x{}", b.iter().map(|x| format!("{:02x}", x)).collect::<String>()),
        Comparand::Null => "null".into(),
    }
}

fn cql_expr(node: &PNode) -> String {
    match node {
        PNode::Predicate(pred) => {
            let field = cql_field(&pred.field);
            match pred.op {
                OpType::In => {
                    let vals: Vec<String> = pred.comparands.iter().map(cql_comparand).collect();
                    format!("{} IN ({})", field, vals.join(", "))
                }
                OpType::Ne => {
                    // CQL doesn't have != directly; use < and > with OR
                    if pred.comparands.len() == 1 {
                        let v = cql_comparand(&pred.comparands[0]);
                        format!(
                            "({} < {} OR {} > {})",
                            field, v, field, v
                        )
                    } else {
                        format!("{} != {}", field, cql_comparand(&pred.comparands[0]))
                    }
                }
                OpType::Matches => {
                    // CQL uses LIKE for pattern matching
                    if let Some(v) = pred.comparands.first() {
                        format!("{} LIKE {}", field, cql_comparand(v))
                    } else {
                        format!("{} LIKE ''", field)
                    }
                }
                _ => {
                    if pred.comparands.len() == 1 {
                        format!("{} {} {}", field, pred.op.symbol(), cql_comparand(&pred.comparands[0]))
                    } else {
                        let vals: Vec<String> =
                            pred.comparands.iter().map(cql_comparand).collect();
                        format!("{} {} ({})", field, pred.op.symbol(), vals.join(", "))
                    }
                }
            }
        }
        PNode::Conjugate(conj) => {
            let op = match conj.conjugate_type {
                ConjugateType::And => "AND",
                ConjugateType::Or => "OR",
                _ => unreachable!(),
            };
            let parts: Vec<String> = conj.children.iter().map(cql_expr).collect();
            if parts.len() == 1 {
                parts[0].clone()
            } else {
                format!("({})", parts.join(&format!(" {} ", op)))
            }
        }
    }
}

fn cql_field(field: &FieldRef) -> String {
    match field {
        FieldRef::Index(i) => format!("field_{}", i),
        FieldRef::Named(s) => s.clone(),
    }
}

// -- CDDL vernacular ----------------------------------------------------------

/// Render a PNode tree as a CDDL type definition
///
/// Produces a CDDL group describing the predicate structure, useful for
/// documenting the expected shape of predicate data.
pub fn to_cddl(node: &PNode) -> String {
    cddl_expr(node)
}

/// Format a comparand for CDDL output.
fn cddl_comparand(c: &Comparand) -> String {
    match c {
        Comparand::Int(v) => v.to_string(),
        Comparand::Float(v) => {
            let s = v.to_string();
            if s.contains('.') { s } else { format!("{}.0", s) }
        }
        Comparand::Text(s) => format!("\"{}\"", s),
        Comparand::Bool(b) => if *b { "true".into() } else { "false".into() },
        Comparand::Bytes(b) => format!("h'{}'", b.iter().map(|x| format!("{:02x}", x)).collect::<String>()),
        Comparand::Null => "null".into(),
    }
}

fn cddl_expr(node: &PNode) -> String {
    match node {
        PNode::Predicate(pred) => {
            let field = match &pred.field {
                FieldRef::Index(i) => format!("field_{}", i),
                FieldRef::Named(s) => s.clone(),
            };
            let op_name = match pred.op {
                OpType::Gt => "gt",
                OpType::Lt => "lt",
                OpType::Eq => "eq",
                OpType::Ne => "ne",
                OpType::Ge => "ge",
                OpType::Le => "le",
                OpType::In => "in",
                OpType::Matches => "matches",
            };
            if pred.comparands.len() == 1 {
                format!("{{ field: \"{}\", op: \"{}\", value: {} }}", field, op_name, cddl_comparand(&pred.comparands[0]))
            } else {
                let vals: Vec<String> = pred.comparands.iter().map(cddl_comparand).collect();
                format!(
                    "{{ field: \"{}\", op: \"{}\", values: [{}] }}",
                    field,
                    op_name,
                    vals.join(", ")
                )
            }
        }
        PNode::Conjugate(conj) => {
            let op = match conj.conjugate_type {
                ConjugateType::And => "and",
                ConjugateType::Or => "or",
                _ => unreachable!(),
            };
            let children: Vec<String> = conj.children.iter().map(cddl_expr).collect();
            format!("{{ {}: [{}] }}", op, children.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;

    fn sample_tree() -> PNode {
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
    fn test_sql_output() {
        let sql = to_sql(&sample_tree());
        assert!(sql.contains("age > 18"));
        assert!(sql.contains("status IN (1, 2, 3)"));
        assert!(sql.contains("AND"));
    }

    #[test]
    fn test_cql_output() {
        let cql = to_cql(&sample_tree());
        assert!(cql.contains("age > 18"));
        assert!(cql.contains("status IN (1, 2, 3)"));
        assert!(cql.contains("AND"));
    }

    #[test]
    fn test_cddl_output() {
        let cddl = to_cddl(&sample_tree());
        assert!(cddl.contains("and"));
        assert!(cddl.contains("\"age\""));
        assert!(cddl.contains("\"gt\""));
    }

    #[test]
    fn test_sql_typed_comparands() {
        let node = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("name".into()),
            op: OpType::Eq,
            comparands: vec![Comparand::Text("alice".into())],
        });
        let sql = to_sql(&node);
        assert_eq!(sql, "name = 'alice'");

        let node = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("active".into()),
            op: OpType::Eq,
            comparands: vec![Comparand::Bool(true)],
        });
        let sql = to_sql(&node);
        assert_eq!(sql, "active = TRUE");
    }

    #[test]
    fn test_cql_typed_comparands() {
        let node = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("name".into()),
            op: OpType::Eq,
            comparands: vec![Comparand::Text("bob".into())],
        });
        let cql = to_cql(&node);
        assert_eq!(cql, "name = 'bob'");

        let node = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("active".into()),
            op: OpType::Eq,
            comparands: vec![Comparand::Bool(false)],
        });
        let cql = to_cql(&node);
        assert_eq!(cql, "active = false");
    }
}
