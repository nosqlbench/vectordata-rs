// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Decode a predicate slab and print operator/field distribution.
//! Used for end-to-end calibration checks of `generate predicates`
//! output.

use std::collections::BTreeMap;

use slabtastic::SlabReader;
use veks_core::formats::pnode::{FieldRef, OpType, PNode};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: inspect_predicate_slab <slab>");
    let r = SlabReader::open(path).expect("open slab");
    let mut by_op: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_field: BTreeMap<String, usize> = BTreeMap::new();
    let mut samples: Vec<String> = Vec::new();
    let total = r.total_records() as i64;
    for ord in 0..total {
        let data = r.get(ord).unwrap();
        let p = PNode::from_bytes_named(&data).unwrap();
        record_node(&p, &mut by_op, &mut by_field);
        if (ord as usize) < 12 {
            samples.push(format!("{}", p));
        }
    }
    println!("total: {} predicates", total);
    println!("operator distribution:");
    for (op, c) in &by_op { println!("  {op:>10}: {c}"); }
    println!("field distribution:");
    let mut field_v: Vec<(&String, &usize)> = by_field.iter().collect();
    field_v.sort_by(|a, b| b.1.cmp(a.1));
    for (f, c) in field_v { println!("  {f}: {c}"); }
    println!("first 12 predicates:");
    for s in samples { println!("  {s}"); }
}

fn record_node(
    node: &PNode,
    by_op: &mut BTreeMap<String, usize>,
    by_field: &mut BTreeMap<String, usize>,
) {
    match node {
        PNode::Predicate(p) => {
            let op_name = match p.op {
                OpType::Gt => "Gt",
                OpType::Lt => "Lt",
                OpType::Eq => "Eq",
                OpType::Ne => "Ne",
                OpType::Ge => "Ge",
                OpType::Le => "Le",
                OpType::In => "In",
                OpType::Matches => "Matches",
            };
            *by_op.entry(op_name.to_string()).or_insert(0) += 1;
            let field_name = match &p.field {
                FieldRef::Named(n) => n.clone(),
                FieldRef::Index(o) => format!("#{o}"),
            };
            *by_field.entry(field_name).or_insert(0) += 1;
        }
        PNode::Conjugate(c) => {
            for child in &c.children { record_node(child, by_op, by_field); }
        }
    }
}
