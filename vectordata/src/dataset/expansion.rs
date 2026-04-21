// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Per-profile step expansion for `dataset.yaml` pipelines.
//!
//! Transforms `per_profile: true` template steps into concrete per-profile
//! steps with resolved IDs, dependencies, and option values. This is a core
//! part of the dataset definition semantics — any module that reads
//! `dataset.yaml` must apply this expansion to get the complete step set.
//!
//! The expansion is a pure data transformation: no I/O, no command execution.

use std::collections::HashSet;

use super::config::DatasetConfig;
use super::pipeline::StepDef;
use super::profile::DSProfileGroup;

/// Collect all raw steps from a `DatasetConfig`.
pub fn collect_all_steps(config: &DatasetConfig) -> Vec<StepDef> {
    let mut steps = Vec::new();
    if let Some(ref pipeline) = config.upstream {
        if let Some(ref shared_steps) = pipeline.steps {
            steps.extend(shared_steps.clone());
        }
    }
    steps
}

/// Filter steps to only those that should run for the given profile.
///
/// A step runs if:
/// - It has no `profiles` field (shared step, always runs), OR
/// - Its `profiles` list contains the active profile name.
pub fn filter_steps_for_profile(steps: Vec<StepDef>, profile: &str) -> Vec<StepDef> {
    steps
        .into_iter()
        .filter(|step| step.profiles.is_empty() || step.profiles.iter().any(|p| p == profile))
        .collect()
}

/// Expand `per_profile` template steps into concrete steps for each profile.
///
/// For each profile (sized ascending by base_count, then default), template
/// steps are cloned with:
/// - ID suffixed with `-{profile_name}` (no suffix for default)
/// - `profiles: [profile_name]` set
/// - `after` references to other template steps similarly suffixed
/// - Option values with `${profile_dir}`, `${base_count}`, `${base_end}`,
///   `${query_count}`, `${profile_name}` resolved
/// - Output paths auto-prefixed with `profiles/{name}/`
///
/// Template steps (per_profile=true) are removed; their expansions replace them.
/// Non-template steps pass through unchanged.
/// Map a pipeline command name to the facet code it implements.
/// Returns None for commands that aren't facet-specific (e.g., state set).
fn command_facet(run: &str) -> Option<char> {
    match run {
        "compute knn" | "compute knn-blas" | "compute knn-faiss" => Some('G'),
        "verify knn-consolidated" | "verify dataset-knnutils" | "verify knn-groundtruth" => Some('G'),
        "compute evaluate-predicates" => Some('R'),
        "verify predicates-sqlite" | "verify predicate-results"
            | "verify predicates-consolidated" => Some('R'),
        "compute filtered-knn" => Some('F'),
        "verify filtered-knn-consolidated" => Some('F'),
        "generate predicates" => Some('P'),
        _ => None,
    }
}

pub fn expand_per_profile_steps(
    steps: Vec<StepDef>,
    profiles: &DSProfileGroup,
    query_count: u64,
) -> Vec<StepDef> {
    expand_per_profile_steps_scoped(steps, profiles, query_count, None)
}

/// Expand per_profile templates, optionally restricting partition profiles
/// to a sub-facet scope.
///
/// When `oracle_sub_facets` is Some (e.g., "BQG"), partition profiles
/// only get templates whose command maps to a facet in the sub-set.
/// Sized profiles and the default profile always get all templates.
pub fn expand_per_profile_steps_scoped(
    steps: Vec<StepDef>,
    profiles: &DSProfileGroup,
    query_count: u64,
    oracle_sub_facets: Option<&str>,
) -> Vec<StepDef> {
    let (templates, regular): (Vec<_>, Vec<_>) = steps
        .into_iter()
        .partition(|s| s.per_profile);

    if templates.is_empty() {
        return regular;
    }

    let template_ids: HashSet<String> = templates
        .iter()
        .map(|s| s.effective_id())
        .collect();

    // Collect bare output filenames from templates for auto-prefix matching.
    let template_output_names: HashSet<String> = templates
        .iter()
        .flat_map(|s| s.output_paths())
        .map(|p| p.strip_prefix("${profile_dir}").unwrap_or(&p).to_string())
        .collect();

    // Check if there are already explicit default-gated steps
    let has_explicit_default_steps = regular
        .iter()
        .any(|s| s.profiles.iter().any(|p| p == "default"));

    let mut result = regular;

    // Collect all profiles to expand: natural name order, then default last.
    // Natural order ensures label_0, label_1, ..., label_9, label_10
    // (not ASCII order or size order which is arbitrary for partitions).
    let mut non_default: Vec<(&str, Option<u64>)> = profiles.profiles.iter()
        .filter(|(name, _)| name.as_str() != "default")
        .map(|(name, p)| (name.as_str(), p.base_count))
        .collect();
    non_default.sort_by(|(a, _), (b, _)| super::profile::natural_cmp(a, b));

    let mut all_profiles: Vec<(&str, Option<u64>)> = non_default;
    if profiles.profiles.contains_key("default") && !has_explicit_default_steps {
        all_profiles.push(("default", None));
    }

    // Expand by phase first, then by profile within each phase.
    // This ensures all phase-0 steps (compute) for all profiles are emitted
    // before any phase-1 steps (verify), enabling correct phase ordering.
    let max_phase = templates.iter().map(|t| t.phase).max().unwrap_or(0);

    // Classic layout: the default profile's outputs go in the dataset root
    // instead of profiles/default/. Detected by checking whether ANY profile
    // view uses a "profiles/" prefix. Source files (base_vectors, query_vectors)
    // may live at root even in standard layout — only computed artifact paths
    // are reliable signals.
    let classic = profiles.profiles.get("default")
        .map(|p| !p.views.values().any(|v| v.source.path.contains("profiles/")))
        .unwrap_or(false);

    for phase in 0..=max_phase {
    for (profile_name, base_count_opt) in &all_profiles {
        let profile_dir = if classic && *profile_name == "default" {
            String::new()
        } else {
            format!("profiles/{}/", profile_name)
        };
        let suffix = if *profile_name == "default" {
            String::new()
        } else {
            format!("-{}", profile_name)
        };

        // `${base_end}` is the file-index upper bound for steps that
        // read directly from the COMBINED source (queries at
        // `[0..query_count)`, base at `[query_count..query_count+base_count)`).
        // That's why query_count is added — used by extract-base to
        // know where to stop consuming base vectors from the combined
        // source. Compute-knn does NOT use this variable; it gets a
        // separate range injected below sized to its own semantics.
        let base_end_str = match base_count_opt {
            Some(bc) => (query_count + bc).to_string(),
            None => "${vector_count}".to_string(),
        };

        // Detect if this is a partition profile (own base_vectors, not
        // windowed from default) vs a sized profile (subset of default).
        let is_partition = profiles.profiles.get(*profile_name)
            .map(|p| p.partition)
            .unwrap_or(false);

        for template in templates.iter().filter(|t| t.phase == phase) {
            let template_id = template.effective_id();

            // Templates with "partition" in the ID only expand for partition
            // profiles — never for default or sized profiles.
            if !is_partition && template_id.contains("partition") {
                continue;
            }

            // For partition profiles, skip templates outside the oracle scope.
            // E.g., with oracle scope "BQG", skip evaluate-predicates (R),
            // compute-filtered-knn (F), etc.
            if is_partition {
                if let Some(scope) = oracle_sub_facets {
                    if let Some(facet) = command_facet(&template.run) {
                        if !scope.contains(facet) {
                            continue;
                        }
                    }
                }
            }

            let expanded_id = if suffix.is_empty() {
                template_id.clone()
            } else {
                format!("{}{}", template_id, suffix)
            };

            // Rewrite after references
            let expanded_after: Vec<String> = template.after.iter().map(|dep| {
                if template_ids.contains(dep.as_str()) {
                    if suffix.is_empty() { dep.clone() }
                    else { format!("{}{}", dep, suffix) }
                } else {
                    dep.clone()
                }
            }).collect();

            // Rewrite option values
            let mut expanded_options = template.options.clone();
            for (key, v) in expanded_options.iter_mut() {
                if let serde_yaml::Value::String(s) = v {
                    if s.contains("${profile_dir}") {
                        *s = s.replace("${profile_dir}", &profile_dir);
                    } else {
                        let bare = s.as_str();
                        let is_output_key = key == "output"
                            || key == "indices"
                            || key == "distances";
                        let should_prefix = !bare.contains('/')
                            && !bare.contains("${")
                            && (is_output_key || template_output_names.contains(bare));
                        if should_prefix {
                            *s = format!("{}{}", profile_dir, s);
                        }
                    }

                    *s = s
                        .replace("${profile_name}", profile_name)
                        .replace("${base_end}", &base_end_str)
                        .replace("${query_count}", &query_count.to_string());
                    if let Some(bc) = base_count_opt {
                        *s = s.replace("${base_count}", &bc.to_string());
                    }
                }
            }

            // Auto-inject range for sized profiles (not partition profiles —
            // partition base vectors are already the exact subset, no windowing needed).
            //
            // Syntax note: use `..` not `,`. The source-string grammar
            // (`vectordata/src/dataset/source.rs::parse_window`) treats
            // `,` as a MULTI-INTERVAL separator: `"0,N"` parses as two
            // intervals `[0,0)` and `[0,N)`, and any consumer that reads
            // `window.0[0]` picks up the empty `[0,0)` → base_n = 0 →
            // divide-by-zero downstream. `..` is the interval separator,
            // matching the inline authoring form `file.fvec[0..N)`.
            if base_count_opt.is_some() && !is_partition && !expanded_options.contains_key("range") {
                // base_end = base_count. See `base_end_str` above for
                // why query_count is NOT added — by compute-knn time
                // the base file holds only base vectors.
                let base_end = base_count_opt.unwrap();
                expanded_options.insert(
                    "range".to_string(),
                    serde_yaml::Value::String(format!("[0..{})", base_end)),
                );
            }

            result.push(StepDef {
                id: Some(expanded_id),
                run: template.run.clone(),
                description: template.description.clone(),
                after: expanded_after,
                profiles: vec![profile_name.to_string()],
                per_profile: false,
                phase: template.phase,
                finalize: template.finalize,
                on_partial: template.on_partial.clone(),
                options: expanded_options,
            });
        }
    }
    } // close for phase

    // Phase-aware profile ordering.
    //
    // Within each phase, profiles execute smallest → largest → default.
    // Phase N+1 starts only after phase N completes for ALL profiles.
    // This prevents I/O thrashing between compute and verification.
    //
    // Example with 2 profiles (1m, default) and 2 phases:
    //   Phase 0: compute-knn-1m → compute-knn (default)
    //   Phase 1: verify-knn-1m → verify-knn (default)
    //   Phase 1 depends on phase 0's last step.

    for phase in 0..=max_phase {
        // Templates in this phase
        let phase_templates: Vec<&StepDef> = templates.iter()
            .filter(|t| t.phase == phase)
            .collect();
        if phase_templates.is_empty() { continue; }

        // Chain profiles within phase 0 to prevent I/O thrashing between
        // compute-heavy steps. Verify phases (phase > 0) leave profiles
        // independent so they can run concurrently.
        if phase == 0 {
            let mut prev_last_id: Option<String> = None;

            for (profile_name, _) in &all_profiles {
                let suffix = if *profile_name == "default" { String::new() }
                    else { format!("-{}", profile_name) };

                let first_id = phase_templates.first().map(|t| {
                    let tid = t.effective_id();
                    if suffix.is_empty() { tid } else { format!("{}{}", tid, suffix) }
                });

                let last_id = phase_templates.last().map(|t| {
                    let tid = t.effective_id();
                    if suffix.is_empty() { tid } else { format!("{}{}", tid, suffix) }
                });

                // Add dependency on previous profile's last step in this phase
                if let (Some(prev), Some(first)) = (&prev_last_id, &first_id) {
                    for step in result.iter_mut() {
                        if step.effective_id() == *first {
                            if !step.after.contains(prev) {
                                step.after.push(prev.clone());
                            }
                            break;
                        }
                    }
                }

                prev_last_id = last_id;
            }
        }

    }

    // Post-hoc phase ordering: for each phase > 0, the first emitted step
    // of that phase depends on the last emitted step of the previous phase.
    // This is done after all expansion so we reference actual emitted steps,
    // not template names that may have been skipped.
    // Collect the IDs of steps that were produced by per_profile expansion
    // (they have a non-empty profiles field). Phase ordering only applies
    // to these — regular steps like partition-profiles are not part of the
    // per_profile phase sequence.
    let expanded_ids: std::collections::HashSet<String> = result.iter()
        .filter(|s| !s.profiles.is_empty())
        .map(|s| s.effective_id())
        .collect();

    for phase in 1..=max_phase {
        let prev_phase_last = result.iter()
            .filter(|s| s.phase == phase - 1 && !s.finalize && expanded_ids.contains(&s.effective_id()))
            .last()
            .map(|s| s.effective_id());
        let this_phase_first_idx = result.iter()
            .position(|s| s.phase == phase && !s.finalize && expanded_ids.contains(&s.effective_id()));

        if let (Some(prev_id), Some(idx)) = (prev_phase_last, this_phase_first_idx) {
            if !result[idx].after.contains(&prev_id) {
                result[idx].after.push(prev_id);
            }
        }
    }

    result
}

/// Collect, expand, and return the fully resolved step list for a dataset.
///
/// This is the canonical entry point for any module that needs the complete
/// pipeline step set. It:
/// 1. Collects raw steps from the config
/// 2. Reads `query_count` from upstream defaults
/// 3. Expands per_profile templates into concrete per-profile steps
pub fn resolve_steps(config: &DatasetConfig) -> Vec<StepDef> {
    let raw_steps = collect_all_steps(config);
    let query_count: u64 = config.upstream.as_ref()
        .and_then(|p| p.defaults.as_ref())
        .and_then(|d| d.get("query_count"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    expand_per_profile_steps(raw_steps, &config.profiles, query_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_empty() {
        let config: DatasetConfig = serde_yaml::from_str(
            "name: test\nprofiles:\n  default:\n    base_vectors: base.fvec\n"
        ).unwrap();
        assert!(collect_all_steps(&config).is_empty());
    }

    #[test]
    fn test_filter_shared_steps_pass() {
        let step = StepDef {
            id: Some("s1".into()),
            run: "import".into(),
            description: None,
            after: vec![],
            profiles: vec![],
            per_profile: false,
            phase: 0,
            finalize: false,
            on_partial: crate::dataset::pipeline::OnPartial::default(),
            options: Default::default(),
        };
        let filtered = filter_steps_for_profile(vec![step], "any");
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_filter_profile_gated() {
        let step = StepDef {
            id: Some("s1".into()),
            run: "compute knn".into(),
            description: None,
            after: vec![],
            profiles: vec!["10m".into()],
            per_profile: false,
            phase: 0,
            finalize: false,
            on_partial: crate::dataset::pipeline::OnPartial::default(),
            options: Default::default(),
        };
        assert_eq!(filter_steps_for_profile(vec![step.clone()], "10m").len(), 1);
        assert_eq!(filter_steps_for_profile(vec![step], "20m").len(), 0);
    }
}
