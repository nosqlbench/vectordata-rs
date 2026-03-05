// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Random attribute and predicate generation with selectivity control.
//!
//! Generates random attribute columns and query predicates that target
//! a configurable selectivity (fraction of base vectors satisfying the
//! predicate).

use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::attribute::{AttributeColumn, AttributeSchema, FieldDescriptor, FieldType};
use super::predicate::{
    ConjugateNode, LogicOp, OpType, PNode, PredicateNode, evaluate,
};

/// Generate attribute columns from a schema for `count` vectors.
///
/// Each column is filled with uniformly random values within the
/// cardinality range, using the provided seeded PRNG.
pub fn generate_columns(
    schema: &AttributeSchema,
    count: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<AttributeColumn> {
    schema
        .fields
        .iter()
        .map(|field| generate_column(field, count, rng))
        .collect()
}

/// Generate a single attribute column.
fn generate_column(
    field: &FieldDescriptor,
    count: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> AttributeColumn {
    let card = field.cardinality.max(1) as i64;
    match field.field_type {
        FieldType::Int => {
            let vals: Vec<i32> = (0..count)
                .map(|_| rng.random_range(0..card) as i32)
                .collect();
            AttributeColumn::Int(vals)
        }
        FieldType::Long => {
            let vals: Vec<i64> = (0..count)
                .map(|_| rng.random_range(0..card))
                .collect();
            AttributeColumn::Long(vals)
        }
        FieldType::Enum => {
            let vals: Vec<u32> = (0..count)
                .map(|_| rng.random_range(0..card as u32))
                .collect();
            AttributeColumn::Enum(vals)
        }
        FieldType::EnumSet => {
            let num_bytes = ((card + 7) / 8) as usize;
            let vals: Vec<Vec<u8>> = (0..count)
                .map(|_| {
                    let mut bitmask = vec![0u8; num_bytes];
                    // Each bit has ~50% chance of being set
                    for byte in &mut bitmask {
                        *byte = rng.random::<u8>();
                    }
                    // Mask off unused high bits in the last byte
                    let used_bits = (card % 8) as u8;
                    if used_bits > 0 && !bitmask.is_empty() {
                        let last = bitmask.len() - 1;
                        bitmask[last] &= (1u8 << used_bits) - 1;
                    }
                    bitmask
                })
                .collect();
            AttributeColumn::EnumSet(vals)
        }
    }
}

/// Generate a default attribute schema with the specified field count,
/// types, and cardinalities.
pub fn generate_schema(
    field_count: usize,
    type_strs: &[&str],
    cardinalities: &[u32],
) -> AttributeSchema {
    let mut fields = Vec::with_capacity(field_count);
    for i in 0..field_count {
        let ft_str = type_strs.get(i % type_strs.len()).copied().unwrap_or("int");
        let ft = FieldType::from_str(ft_str).unwrap_or(FieldType::Int);
        let card = cardinalities
            .get(i % cardinalities.len())
            .copied()
            .unwrap_or(10);

        let name = format!("field_{}", i);
        let enum_values = if ft == FieldType::Enum || ft == FieldType::EnumSet {
            (0..card).map(|j| format!("{}_{}", name, j)).collect()
        } else {
            vec![]
        };

        fields.push(FieldDescriptor {
            name,
            field_type: ft,
            cardinality: card,
            enum_values,
        });
    }
    AttributeSchema { fields }
}

/// Generate a simple predicate for a single query targeting approximate selectivity.
///
/// For simple predicates, generates one leaf predicate per field connected by AND.
/// The range of each predicate is tuned to achieve the target selectivity.
pub fn generate_predicate_simple(
    schema: &AttributeSchema,
    target_selectivity: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> PNode {
    if schema.fields.is_empty() {
        return PNode::Predicate(PredicateNode {
            field_index: 0,
            op: OpType::Eq,
            values: vec![0],
        });
    }

    // For AND of N fields, each field should pass with probability
    // selectivity^(1/N) to achieve overall selectivity
    let n = schema.fields.len() as f64;
    let per_field_sel = target_selectivity.powf(1.0 / n);

    let children: Vec<PNode> = schema
        .fields
        .iter()
        .enumerate()
        .map(|(fi, field)| generate_field_predicate(fi as u32, field, per_field_sel, rng))
        .collect();

    if children.len() == 1 {
        children.into_iter().next().unwrap()
    } else {
        PNode::Conjugate(ConjugateNode {
            op: LogicOp::And,
            children,
        })
    }
}

/// Generate a compound predicate with nested AND/OR.
pub fn generate_predicate_compound(
    schema: &AttributeSchema,
    target_selectivity: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> PNode {
    if schema.fields.len() < 2 {
        return generate_predicate_simple(schema, target_selectivity, rng);
    }

    // Split fields into two groups, one AND and one OR
    let mid = schema.fields.len() / 2;

    let and_children: Vec<PNode> = schema.fields[..mid]
        .iter()
        .enumerate()
        .map(|(fi, field)| {
            let sel = target_selectivity.powf(1.0 / mid as f64);
            generate_field_predicate(fi as u32, field, sel, rng)
        })
        .collect();

    let or_children: Vec<PNode> = schema.fields[mid..]
        .iter()
        .enumerate()
        .map(|(fi, field)| {
            // For OR, each child passes with lower probability
            let sel = target_selectivity / (schema.fields.len() - mid) as f64;
            generate_field_predicate((mid + fi) as u32, field, sel.min(1.0), rng)
        })
        .collect();

    let or_node = PNode::Conjugate(ConjugateNode {
        op: LogicOp::Or,
        children: or_children,
    });

    let mut all_children = and_children;
    all_children.push(or_node);

    PNode::Conjugate(ConjugateNode {
        op: LogicOp::And,
        children: all_children,
    })
}

/// Generate a predicate for a single field targeting the given selectivity.
fn generate_field_predicate(
    field_index: u32,
    field: &FieldDescriptor,
    target_sel: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> PNode {
    let card = field.cardinality.max(1);

    match field.field_type {
        FieldType::Int | FieldType::Long => {
            // Use a range predicate: value >= low AND value < high
            // Range width = card * target_sel
            let range_width = ((card as f64 * target_sel).ceil() as i64).max(1);
            let max_start = (card as i64 - range_width).max(0);
            let start = rng.random_range(0..=max_start);

            if range_width == 1 {
                PNode::Predicate(PredicateNode {
                    field_index,
                    op: OpType::Eq,
                    values: vec![start],
                })
            } else {
                // AND of >= start and < start+range_width
                PNode::Conjugate(ConjugateNode {
                    op: LogicOp::And,
                    children: vec![
                        PNode::Predicate(PredicateNode {
                            field_index,
                            op: OpType::Ge,
                            values: vec![start],
                        }),
                        PNode::Predicate(PredicateNode {
                            field_index,
                            op: OpType::Lt,
                            values: vec![start + range_width],
                        }),
                    ],
                })
            }
        }
        FieldType::Enum => {
            // IN predicate with subset of values
            let num_in = ((card as f64 * target_sel).ceil() as usize).max(1).min(card as usize);
            let mut all_vals: Vec<i64> = (0..card as i64).collect();
            // Partial Fisher-Yates to select num_in values
            for i in 0..num_in {
                let j = rng.random_range(i..all_vals.len());
                all_vals.swap(i, j);
            }
            let selected: Vec<i64> = all_vals[..num_in].to_vec();

            if selected.len() == 1 {
                PNode::Predicate(PredicateNode {
                    field_index,
                    op: OpType::Eq,
                    values: selected,
                })
            } else {
                PNode::Predicate(PredicateNode {
                    field_index,
                    op: OpType::In,
                    values: selected,
                })
            }
        }
        FieldType::EnumSet => {
            // Check for specific enum value membership
            let check_val = rng.random_range(0..card as i64);
            PNode::Predicate(PredicateNode {
                field_index,
                op: OpType::In,
                values: vec![check_val],
            })
        }
    }
}

/// Measure the actual selectivity of a predicate against columns.
pub fn measure_selectivity(
    pred: &PNode,
    columns: &[AttributeColumn],
    count: usize,
) -> f64 {
    if count == 0 {
        return 0.0;
    }
    let passing = (0..count)
        .filter(|&i| evaluate(pred, columns, i))
        .count();
    passing as f64 / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::rng;

    #[test]
    fn test_generate_schema() {
        let schema = generate_schema(3, &["int", "enum", "enum_set"], &[12, 30, 28]);
        assert_eq!(schema.fields.len(), 3);
        assert_eq!(schema.fields[0].field_type, FieldType::Int);
        assert_eq!(schema.fields[0].cardinality, 12);
        assert_eq!(schema.fields[1].field_type, FieldType::Enum);
        assert_eq!(schema.fields[1].cardinality, 30);
        assert_eq!(schema.fields[1].enum_values.len(), 30);
        assert_eq!(schema.fields[2].field_type, FieldType::EnumSet);
    }

    #[test]
    fn test_generate_columns() {
        let schema = generate_schema(2, &["int", "enum"], &[10, 5]);
        let mut rng_inst = rng::seeded_rng(42);
        let columns = generate_columns(&schema, 100, &mut rng_inst);
        assert_eq!(columns.len(), 2);
        assert_eq!(columns[0].len(), 100);
        assert_eq!(columns[1].len(), 100);

        // Check int values are in range
        if let AttributeColumn::Int(vals) = &columns[0] {
            for &v in vals {
                assert!(v >= 0 && v < 10);
            }
        } else {
            panic!("expected Int column");
        }
    }

    #[test]
    fn test_generate_predicate_selectivity() {
        let schema = generate_schema(2, &["int", "enum"], &[100, 20]);
        let mut rng_inst = rng::seeded_rng(42);
        let columns = generate_columns(&schema, 10_000, &mut rng_inst);

        let mut rng_inst = rng::seeded_rng(99);
        let pred = generate_predicate_simple(&schema, 0.1, &mut rng_inst);

        let sel = measure_selectivity(&pred, &columns, 10_000);
        // Should be approximately 0.1 (within reasonable tolerance)
        assert!(
            sel > 0.01 && sel < 0.5,
            "selectivity {} out of expected range",
            sel
        );
    }

    #[test]
    fn test_generate_predicate_compound() {
        let schema = generate_schema(4, &["int", "enum", "int", "enum"], &[50, 10, 50, 10]);
        let mut rng_inst = rng::seeded_rng(42);
        let columns = generate_columns(&schema, 5_000, &mut rng_inst);

        let mut rng_inst = rng::seeded_rng(99);
        let pred = generate_predicate_compound(&schema, 0.2, &mut rng_inst);

        let sel = measure_selectivity(&pred, &columns, 5_000);
        // Compound predicates may not hit exact selectivity, but should be non-trivial
        assert!(sel > 0.0 && sel < 1.0, "selectivity {} is degenerate", sel);
    }
}
