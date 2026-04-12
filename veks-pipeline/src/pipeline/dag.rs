// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! DAG construction and topological sorting for pipeline steps.
//!
//! Builds a directed acyclic graph from step definitions, respecting both
//! explicit `after` dependencies and implicit file-based dependencies (where
//! one step's input matches another step's output). Validates for:
//! - Cycles
//! - Dangling inputs (references to undefined step IDs)
//! - Output collisions (two steps producing the same file)
//!
//! The topological sort is **order-stable**: when multiple steps are ready
//! (all dependencies satisfied), they are emitted in their original definition
//! order. This lets authors control execution priority by listing steps
//! earlier in the YAML.

use std::collections::HashMap;

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

    // 5. Order-stable topological sort (Kahn's algorithm).
    //
    // When multiple steps are ready (in-degree 0), they are emitted in
    // their original definition order. This lets authors control execution
    // priority by listing steps earlier in the YAML.
    let ordered_steps = {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;

        // Compute in-degrees
        let mut in_degree: Vec<usize> = vec![0; resolved.len()];
        for edge in graph.edge_indices() {
            let (_, target) = graph.edge_endpoints(edge).unwrap();
            let target_step = graph[target];
            in_degree[target_step] += 1;
        }

        // Seed with zero-in-degree nodes, ordered by original index (lowest first)
        let mut ready: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                ready.push(Reverse(i));
            }
        }

        let mut result: Vec<ResolvedStep> = Vec::with_capacity(resolved.len());
        while let Some(Reverse(step_idx)) = ready.pop() {
            // For each step that depends on this one, decrement in-degree
            let node = node_indices[step_idx];
            for neighbor in graph.neighbors(node) {
                let dep_step = graph[neighbor];
                in_degree[dep_step] -= 1;
                if in_degree[dep_step] == 0 {
                    ready.push(Reverse(dep_step));
                }
            }
            result.push(resolved[step_idx].clone());
        }

        if result.len() != resolved.len() {
            // Trace the cycle for a useful error message.
            // Walk from any stuck node following edges among stuck nodes
            // until we revisit a node, then collect the cycle.
            let stuck = in_degree.iter().position(|&d| d > 0).unwrap_or(0);
            let mut visited = vec![false; resolved.len()];
            let mut cur = stuck;

            // Phase 1: walk until we revisit a node (find cycle entry)
            loop {
                if visited[cur] { break; }
                visited[cur] = true;
                let node = node_indices[cur];
                let mut found_next = false;
                for neighbor in graph.neighbors(node) {
                    let dep = graph[neighbor];
                    if in_degree[dep] > 0 {
                        cur = dep;
                        found_next = true;
                        break;
                    }
                }
                if !found_next { break; }
            }

            // Phase 2: collect the cycle starting from `cur`
            let entry = cur;
            let mut cycle_steps: Vec<String> = vec![resolved[cur].id.clone()];
            let node = node_indices[cur];
            let mut found_next = false;
            for neighbor in graph.neighbors(node) {
                let dep = graph[neighbor];
                if in_degree[dep] > 0 {
                    cur = dep;
                    found_next = true;
                    break;
                }
            }
            if found_next {
                while cur != entry {
                    cycle_steps.push(resolved[cur].id.clone());
                    let node = node_indices[cur];
                    let mut next = false;
                    for neighbor in graph.neighbors(node) {
                        let dep = graph[neighbor];
                        if in_degree[dep] > 0 {
                            cur = dep;
                            next = true;
                            break;
                        }
                    }
                    if !next { break; }
                }
                cycle_steps.push(resolved[entry].id.clone()); // close the cycle
            }
            // Also list all stuck steps with their dependencies
            let stuck_info: Vec<String> = in_degree.iter().enumerate()
                .filter(|&(_, &d)| d > 0)
                .map(|(i, _)| {
                    let step = &resolved[i];
                    let deps: Vec<&str> = step.def.after.iter().map(|s| s.as_str()).collect();
                    format!("  {} (profiles: {:?}, after: {:?})", step.id, step.def.profiles, deps)
                })
                .collect();
            return Err(format!(
                "cycle detected: {}\n\nStuck steps:\n{}",
                cycle_steps.join(" → "),
                stuck_info.join("\n"),
            ));
        }

        result
    };

    Ok(PipelineDag {
        steps: ordered_steps,
    })
}

/// Build a DAG from a subset of steps, silently ignoring `after` references
/// to step IDs not present in the provided list.
///
/// This is used for finalization passes where finalize steps may reference
/// compute steps via `after` that are not in the finalize-only DAG. Those
/// compute steps are guaranteed to have already completed, so the
/// dependency is trivially satisfied.
pub fn build_dag_partial(steps: &[StepDef]) -> Result<PipelineDag, String> {
    if steps.is_empty() {
        return Ok(PipelineDag { steps: vec![] });
    }

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

    let mut graph = DiGraph::<usize, ()>::new();
    let mut node_indices: Vec<NodeIndex> = Vec::with_capacity(resolved.len());
    for i in 0..resolved.len() {
        node_indices.push(graph.add_node(i));
    }

    for (i, step) in resolved.iter().enumerate() {
        for dep_id in &step.def.after {
            // Silently skip references to steps not in this subset
            if let Some(&dep_idx) = id_to_index.get(dep_id.as_str()) {
                graph.add_edge(node_indices[dep_idx], node_indices[i], ());
            }
        }
    }

    // Order-stable topological sort (same algorithm as build_dag)
    let ordered_steps = {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;

        let mut in_degree: Vec<usize> = vec![0; resolved.len()];
        for edge in graph.edge_indices() {
            let (_, target) = graph.edge_endpoints(edge).unwrap();
            let target_step = graph[target];
            in_degree[target_step] += 1;
        }

        let mut ready: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                ready.push(Reverse(i));
            }
        }

        let mut result: Vec<ResolvedStep> = Vec::with_capacity(resolved.len());
        while let Some(Reverse(step_idx)) = ready.pop() {
            let node = node_indices[step_idx];
            for neighbor in graph.neighbors(node) {
                let dep_step = graph[neighbor];
                in_degree[dep_step] -= 1;
                if in_degree[dep_step] == 0 {
                    ready.push(Reverse(dep_step));
                }
            }
            result.push(resolved[step_idx].clone());
        }

        if result.len() != resolved.len() {
            return Err("cycle detected among finalization steps".to_string());
        }

        result
    };

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
            phase: 0,
            finalize: false,
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
                phase: 0,
                finalize: false,
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
            phase: 0,
            finalize: false,
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

    #[test]
    fn test_order_stability() {
        // Two independent chains: vectors→knn and metadata→survey.
        // "vectors" (idx 0) and "metadata" (idx 2) are both roots, but
        // vectors is listed first. After vectors completes, knn (idx 1)
        // becomes ready and has a lower index than metadata (idx 2), so
        // it runs next. The stable order is: vectors, knn, metadata, survey.
        let steps = vec![
            step("vectors", "import vectors", vec![], Some("v.fvec")),
            step("knn", "compute knn", vec!["vectors"], Some("k.ivec")),
            step("metadata", "import metadata", vec![], Some("m.slab")),
            step("survey", "survey", vec!["metadata"], Some("s.json")),
        ];
        let dag = build_dag(&steps).unwrap();
        let ids: Vec<&str> = dag.steps.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["vectors", "knn", "metadata", "survey"]);
    }

    #[test]
    fn test_order_stability_many_independent() {
        // Five independent steps — must come out in definition order.
        let steps = vec![
            step("e", "cmd-e", vec![], Some("e.out")),
            step("d", "cmd-d", vec![], Some("d.out")),
            step("c", "cmd-c", vec![], Some("c.out")),
            step("b", "cmd-b", vec![], Some("b.out")),
            step("a", "cmd-a", vec![], Some("a.out")),
        ];
        let dag = build_dag(&steps).unwrap();
        let ids: Vec<&str> = dag.steps.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["e", "d", "c", "b", "a"]);
    }

    // ── build_dag_partial tests ───────────────────────────────────

    #[test]
    fn test_partial_empty() {
        let dag = build_dag_partial(&[]).unwrap();
        assert!(dag.steps.is_empty());
    }

    #[test]
    fn test_partial_ignores_dangling_after_refs() {
        // Finalize steps reference compute steps that are NOT in the
        // partial subset. build_dag_partial should silently ignore them.
        let steps = vec![
            step("gen-json", "generate dataset-json", vec!["compute-knn"], None),
            step("gen-merkle", "merkle create", vec!["gen-json", "verify-knn"], None),
            step("gen-catalog", "catalog generate", vec!["gen-merkle"], None),
        ];
        let dag = build_dag_partial(&steps).unwrap();
        let ids: Vec<&str> = dag.steps.iter().map(|s| s.id.as_str()).collect();
        // Internal ordering preserved: gen-json → gen-merkle → gen-catalog
        assert_eq!(ids, vec!["gen-json", "gen-merkle", "gen-catalog"]);
    }

    #[test]
    fn test_partial_preserves_internal_ordering() {
        // Two finalize steps with an internal dependency.
        // The dangling reference "extract-base" is ignored.
        let steps = vec![
            step("gen-vars", "generate variables-json", vec!["extract-base"], None),
            step("gen-merkle", "merkle create", vec!["gen-vars"], None),
        ];
        let dag = build_dag_partial(&steps).unwrap();
        assert_eq!(dag.steps[0].id, "gen-vars");
        assert_eq!(dag.steps[1].id, "gen-merkle");
    }

    #[test]
    fn test_partial_all_dangling_refs_become_roots() {
        // When ALL after references are dangling, the steps become
        // independent roots and emit in definition order.
        let steps = vec![
            step("c", "cmd-c", vec!["nonexistent-1"], None),
            step("b", "cmd-b", vec!["nonexistent-2"], None),
            step("a", "cmd-a", vec!["nonexistent-3"], None),
        ];
        let dag = build_dag_partial(&steps).unwrap();
        let ids: Vec<&str> = dag.steps.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["c", "b", "a"]);
    }

    #[test]
    fn test_partial_detects_cycles() {
        let steps = vec![
            step("a", "cmd-a", vec!["b"], None),
            step("b", "cmd-b", vec!["a"], None),
        ];
        let result = build_dag_partial(&steps);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cycle"));
    }

    #[test]
    fn test_partial_detects_duplicate_ids() {
        let steps = vec![
            step("dup", "cmd-a", vec![], None),
            step("dup", "cmd-b", vec![], None),
        ];
        let result = build_dag_partial(&steps);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("duplicate"));
    }

    #[test]
    fn test_partial_mixed_dangling_and_internal_deps() {
        // gen-json depends on "compute-knn" (dangling) and nothing else.
        // gen-vars depends on "gen-json" (internal).
        // gen-log depends on "gen-json" (internal).
        // gen-merkle depends on "gen-vars" and "gen-log" (both internal)
        //   plus "verify-knn" (dangling).
        // gen-catalog depends on "gen-merkle" (internal).
        let steps = vec![
            step("gen-json", "generate dataset-json", vec!["compute-knn"], None),
            step("gen-vars", "generate variables-json", vec!["gen-json"], None),
            step("gen-log", "generate dataset-log-jsonl", vec!["gen-json"], None),
            step("gen-merkle", "merkle create", vec!["gen-vars", "gen-log", "verify-knn"], None),
            step("gen-catalog", "catalog generate", vec!["gen-merkle"], None),
        ];
        let dag = build_dag_partial(&steps).unwrap();
        let ids: Vec<&str> = dag.steps.iter().map(|s| s.id.as_str()).collect();

        // gen-json must be first (root after dangling ref removed)
        assert_eq!(ids[0], "gen-json");
        // gen-catalog must be last
        assert_eq!(ids[4], "gen-catalog");
        // gen-vars and gen-log must come before gen-merkle
        let merkle_pos = ids.iter().position(|&id| id == "gen-merkle").unwrap();
        let vars_pos = ids.iter().position(|&id| id == "gen-vars").unwrap();
        let log_pos = ids.iter().position(|&id| id == "gen-log").unwrap();
        assert!(vars_pos < merkle_pos, "gen-vars must precede gen-merkle");
        assert!(log_pos < merkle_pos, "gen-log must precede gen-merkle");
    }
}
