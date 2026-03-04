// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Vernacular adapters for PNode — human-readable representations in SQL, CQL,
//! and CDDL syntax.
//!
//! These adapters convert predicate trees into native query language syntax for
//! troubleshooting, visualization, and schema documentation.

use super::{ConjugateType, FieldRef, OpType, PNode};

// -- SQL vernacular -----------------------------------------------------------

/// Render a PNode as a SQL WHERE clause expression
pub fn to_sql(node: &PNode) -> String {
    sql_expr(node)
}

fn sql_expr(node: &PNode) -> String {
    match node {
        PNode::Predicate(pred) => {
            let field = sql_field(&pred.field);
            match pred.op {
                OpType::In => {
                    let vals: Vec<String> = pred.comparands.iter().map(|v| v.to_string()).collect();
                    format!("{} IN ({})", field, vals.join(", "))
                }
                OpType::Matches => {
                    if let Some(v) = pred.comparands.first() {
                        format!("{} ~ '{}'", field, v)
                    } else {
                        format!("{} ~ ''", field)
                    }
                }
                _ => {
                    if pred.comparands.len() == 1 {
                        format!("{} {} {}", field, pred.op.symbol(), pred.comparands[0])
                    } else {
                        let vals: Vec<String> =
                            pred.comparands.iter().map(|v| v.to_string()).collect();
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

fn cql_expr(node: &PNode) -> String {
    match node {
        PNode::Predicate(pred) => {
            let field = cql_field(&pred.field);
            match pred.op {
                OpType::In => {
                    let vals: Vec<String> = pred.comparands.iter().map(|v| v.to_string()).collect();
                    format!("{} IN ({})", field, vals.join(", "))
                }
                OpType::Ne => {
                    // CQL doesn't have != directly; use < and > with OR
                    if pred.comparands.len() == 1 {
                        format!(
                            "({} < {} OR {} > {})",
                            field, pred.comparands[0], field, pred.comparands[0]
                        )
                    } else {
                        format!("{} != {}", field, pred.comparands[0])
                    }
                }
                OpType::Matches => {
                    // CQL uses LIKE for pattern matching
                    if let Some(v) = pred.comparands.first() {
                        format!("{} LIKE '{}'", field, v)
                    } else {
                        format!("{} LIKE ''", field)
                    }
                }
                _ => {
                    if pred.comparands.len() == 1 {
                        format!("{} {} {}", field, pred.op.symbol(), pred.comparands[0])
                    } else {
                        let vals: Vec<String> =
                            pred.comparands.iter().map(|v| v.to_string()).collect();
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
                format!("{{ field: \"{}\", op: \"{}\", value: {} }}", field, op_name, pred.comparands[0])
            } else {
                let vals: Vec<String> = pred.comparands.iter().map(|v| v.to_string()).collect();
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
                    comparands: vec![18],
                }),
                PNode::Predicate(PredicateNode {
                    field: FieldRef::Named("status".into()),
                    op: OpType::In,
                    comparands: vec![1, 2, 3],
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
}
