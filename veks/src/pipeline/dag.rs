// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! DAG construction and topological sorting for pipeline steps.
//!
//! Builds a directed acyclic graph from step definitions, respecting both
//! explicit `after` dependencies and implicit file-based dependencies (where
//! one step's input matches another step's output). Validates for:
//! - Cycles
//! - Dangling inputs (references to undefined step IDs)
//! - Output collisions (two steps producing the same file)

use std::collections::HashMap;

use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};

use super::schema::StepDef;

/// A resolved step in the DAG, carrying its original definition and index.
#[derive(Debug, Clone)]
pub struct ResolvedStep {
    /// The effective step ID (explicit or derived).
    pub id: String,
    /// The step definition.
    pub def: StepDef,
}

/// Result of DAG construction: steps in topological (execution) order.
#[derive(Debug)]
pub struct PipelineDag {
    /// Steps sorted in topological order (dependencies before dependents).
    pub steps: Vec<ResolvedStep>,
}

/// Build a DAG from a list of step definitions and return steps in execution order.
///
/// Validates for duplicate IDs, cycles, dangling `after` references, and
/// output path collisions.
pub fn build_dag(steps: &[StepDef]) -> Result<PipelineDag, String> {
    if steps.is_empty() {
        return Ok(PipelineDag { steps: vec![] });
    }

    // 1. Resolve effective IDs and check for duplicates
    let mut id_to_index: HashMap<String, usize> = HashMap::new();
    let mut resolved: Vec<ResolvedStep> = Vec::with_capacity(steps.len());

    for (i, step) in steps.iter().enumerate() {
        let id = step.effective_id();
        if let Some(prev) = id_to_index.get(&id) {
            return Err(format!(
                "duplicate step ID '{}' (steps {} and {})",
                id, prev, i
            ));
        }
        id_to_index.insert(id.clone(), i);
        resolved.push(ResolvedStep {
            id,
            def: step.clone(),
        });
    }

    // 2. Check for output collisions
    let mut output_to_step: HashMap<String, &str> = HashMap::new();
    for step in &resolved {
        if let Some(output) = step.def.output_path() {
            if let Some(prev_id) = output_to_step.get(&output) {
                return Err(format!(
                    "output collision: steps '{}' and '{}' both produce '{}'",
                    prev_id, step.id, output
                ));
            }
            output_to_step.insert(output, &step.id);
        }
    }

    // 3. Build graph with explicit `after` edges
    let mut graph = DiGraph::<usize, ()>::new();
    let mut node_indices: Vec<NodeIndex> = Vec::with_capacity(resolved.len());

    for i in 0..resolved.len() {
        node_indices.push(graph.add_node(i));
    }

    for (i, step) in resolved.iter().enumerate() {
        for dep_id in &step.def.after {
            let dep_idx = id_to_index.get(dep_id.as_str()).ok_or_else(|| {
                format!(
                    "step '{}' depends on '{}' which is not defined",
                    step.id, dep_id
                )
            })?;
            // Edge from dependency to dependent (dep must run first)
            graph.add_edge(node_indices[*dep_idx], node_indices[i], ());
        }
    }

    // 4. Build implicit file-based dependency edges
    // If step B has an option value matching step A's output, B depends on A.
    let output_to_idx: HashMap<String, usize> = resolved
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.def.output_path().map(|o| (o, i)))
        .collect();

    for (i, step) in resolved.iter().enumerate() {
        for (_key, value) in &step.def.options {
            let val_str = match value {
                serde_yaml::Value::String(s) => s.as_str(),
                _ => continue,
            };
            if let Some(&producer_idx) = output_to_idx.get(val_str) {
                if producer_idx != i {
                    // Avoid duplicate edges
                    let from = node_indices[producer_idx];
                    let to = node_indices[i];
                    if !graph.contains_edge(from, to) {
                        graph.add_edge(from, to, ());
                    }
                }
            }
        }
    }

    // 5. Topological sort (detects cycles)
    let sorted = toposort(&graph, None).map_err(|cycle| {
        let step_idx = graph[cycle.node_id()];
        format!(
            "cycle detected involving step '{}'",
            resolved[step_idx].id
        )
    })?;

    let ordered_steps: Vec<ResolvedStep> = sorted
        .into_iter()
        .map(|node_idx| {
            let step_idx = graph[node_idx];
            resolved[step_idx].clone()
        })
        .collect();

    Ok(PipelineDag {
        steps: ordered_steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use crate::pipeline::schema::OnPartial;

    fn step(id: &str, run: &str, after: Vec<&str>, output: Option<&str>) -> StepDef {
        let mut options = IndexMap::new();
        if let Some(out) = output {
            options.insert(
                "output".to_string(),
                serde_yaml::Value::String(out.to_string()),
            );
        }
        StepDef {
            id: Some(id.to_string()),
            run: run.to_string(),
            description: None,
            after: after.into_iter().map(String::from).collect(),
            profiles: vec![],
            per_profile: false,
            on_partial: OnPartial::default(),
            options,
        }
    }

    #[test]
    fn test_empty_dag() {
        let dag = build_dag(&[]).unwrap();
        assert!(dag.steps.is_empty());
    }

    #[test]
    fn test_single_step() {
        let steps = vec![step("a", "generate vectors", vec![], Some("out.fvec"))];
        let dag = build_dag(&steps).unwrap();
        assert_eq!(dag.steps.len(), 1);
        assert_eq!(dag.steps[0].id, "a");
    }

    #[test]
    fn test_explicit_ordering() {
        let steps = vec![
            step("a", "generate vectors", vec![], Some("base.fvec")),
            step("b", "compute knn", vec!["a"], Some("knn.ivec")),
        ];
        let dag = build_dag(&steps).unwrap();
        assert_eq!(dag.steps.len(), 2);
        assert_eq!(dag.steps[0].id, "a");
        assert_eq!(dag.steps[1].id, "b");
    }

    #[test]
    fn test_implicit_file_dependency() {
        let mut b_opts = IndexMap::new();
        b_opts.insert(
            "source".to_string(),
            serde_yaml::Value::String("shuffle.ivec".to_string()),
        );
        b_opts.insert(
            "output".to_string(),
            serde_yaml::Value::String("extracted.fvec".to_string()),
        );

        let steps = vec![
            step("a", "generate ivec-shuffle", vec![], Some("shuffle.ivec")),
            StepDef {
                id: Some("b".to_string()),
                run: "generate fvec-extract".to_string(),
                description: None,
                after: vec![],
                profiles: vec![],
                per_profile: false,
                on_partial: OnPartial::default(),
                options: b_opts,
            },
        ];
        let dag = build_dag(&steps).unwrap();
        assert_eq!(dag.steps[0].id, "a");
        assert_eq!(dag.steps[1].id, "b");
    }

    #[test]
    fn test_cycle_detection() {
        let steps = vec![
            step("a", "cmd-a", vec!["b"], Some("a.out")),
            step("b", "cmd-b", vec!["a"], Some("b.out")),
        ];
        let result = build_dag(&steps);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cycle"));
    }

    #[test]
    fn test_duplicate_id() {
        let steps = vec![
            step("dup", "cmd-a", vec![], Some("a.out")),
            step("dup", "cmd-b", vec![], Some("b.out")),
        ];
        let result = build_dag(&steps);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("duplicate"));
    }

    #[test]
    fn test_dangling_reference() {
        let steps = vec![step("a", "cmd-a", vec!["nonexistent"], Some("a.out"))];
        let result = build_dag(&steps);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not defined"));
    }

    #[test]
    fn test_output_collision() {
        let steps = vec![
            step("a", "cmd-a", vec![], Some("same.fvec")),
            step("b", "cmd-b", vec![], Some("same.fvec")),
        ];
        let result = build_dag(&steps);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("collision"));
    }

    #[test]
    fn test_derived_id() {
        let steps = vec![StepDef {
            id: None,
            run: "generate ivec-shuffle".to_string(),
            description: None,
            after: vec![],
            profiles: vec![],
            per_profile: false,
            on_partial: OnPartial::default(),
            options: IndexMap::new(),
        }];
        let dag = build_dag(&steps).unwrap();
        assert_eq!(dag.steps[0].id, "generate-ivec-shuffle");
    }

    #[test]
    fn test_diamond_dag() {
        // A -> B, A -> C, B -> D, C -> D
        let steps = vec![
            step("a", "cmd-a", vec![], Some("a.out")),
            step("b", "cmd-b", vec!["a"], Some("b.out")),
            step("c", "cmd-c", vec!["a"], Some("c.out")),
            step("d", "cmd-d", vec!["b", "c"], Some("d.out")),
        ];
        let dag = build_dag(&steps).unwrap();
        assert_eq!(dag.steps.len(), 4);
        // A must be first, D must be last
        assert_eq!(dag.steps[0].id, "a");
        assert_eq!(dag.steps[3].id, "d");
    }
}
