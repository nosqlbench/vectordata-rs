// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: tag the profiles of an existing dataset (PL-12).
//!
//! A dataset written before tags existed is given, idempotently and by
//! textual edit of its own `dataset.yaml`, the tags its generators
//! would write today (PS-12): `size` on every profile with a count,
//! and on every profile whose predicate facet a completed generator
//! step produced, the facet's `family`, `selectivity_ladder`, `forms`
//! and `predicates` class. Every profile other than `default` and the
//! partitions names its parent, `inherits: default`. No facet and no
//! file moves, a re-run writes nothing, and what was written is
//! recorded in the step record (PS-13) so a hand edit is reported.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use vectordata::dataset::profile::{DSView, size_rung};
use vectordata::dataset::{DatasetConfig, catalog::strip_window_suffix};

use crate::pipeline::command::{
    ArtifactManifest, AttributeWrite, CommandDoc, CommandOp, CommandResult, OptionDesc, Options,
    Status, StreamContext, render_options_table,
};
use crate::pipeline::progress::StepRecord;

pub struct TagProfilesOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(TagProfilesOp)
}

impl CommandOp for TagProfilesOp {
    fn command_path(&self) -> &str {
        "config tag-profiles"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_CONFIG
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_SECONDARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Tag every profile of an existing dataset the way its generators would today".into(),
            body: format!(
                r#"# config tag-profiles

Give the profiles of a dataset written before tags existed the tags a
selector reads (`docs/design/srd-profile-selectors.md`), by textual
edit of `dataset.yaml`: `size` on every profile with a count, the
`family`, `selectivity_ladder`, `forms` and `predicates` class of a
predicate facet a completed generator step produced, and
`inherits: default` on every profile other than `default` and the
partitions. Nothing else in the file changes, a re-run writes nothing,
and what was written is recorded beside the step so a hand edit is
reported as stale rather than silently kept.

Runs in the finalize pass, after every producing step, and before the
steps that publish the dataset's definition.

## Options

{}
"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, _options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();
        let yaml_path = ctx.workspace.join("dataset.yaml");
        let text = match std::fs::read_to_string(&yaml_path) {
            Ok(t) => t,
            Err(e) => return error(format!("read {}: {e}", yaml_path.display()), start),
        };
        let config = match DatasetConfig::load(&yaml_path) {
            Ok(c) => c,
            Err(e) => return error(format!("load {}: {e}", yaml_path.display()), start),
        };
        let plan = match plan_tags(&config, &ctx.workspace, &ctx.progress.steps) {
            Ok(p) => p,
            Err(e) => return error(e, start),
        };
        for note in &plan.notes {
            ctx.ui.log(&format!("  note: {note}"));
        }

        // Parents are named by textual edit here; the tags go through
        // the runner, which writes and records them once the step has
        // succeeded (PS-13).
        let mut edited = text.clone();
        for profile in &plan.parents {
            edited = match vectordata::dataset::yaml_edit::set_profile_inherits(&edited, profile, "default") {
                Ok(t) => t,
                Err(e) => return error(format!("profile '{profile}': {e}"), start),
            };
        }
        let parents_written = edited != text;
        if parents_written && !ctx.dry_run {
            let tmp = yaml_path.with_extension("yaml.tmp");
            if let Err(e) = std::fs::write(&tmp, &edited).and_then(|_| std::fs::rename(&tmp, &yaml_path)) {
                return error(format!("write {}: {e}", yaml_path.display()), start);
            }
        }
        let tagged = plan.writes.iter().map(|w| w.profile.as_str()).collect::<std::collections::BTreeSet<_>>().len();
        // The file it edited is its output when it changed anything, so
        // the finalize steps that publish the definition see a newer
        // input and run again; a run that wrote nothing produces
        // nothing and leaves them fresh.
        let config_now = DatasetConfig::load(&yaml_path).ok();
        let writes_anything = parents_written
            || plan.writes.iter().any(|w| {
                config_now
                    .as_ref()
                    .and_then(|c| c.profiles.profile(&w.profile))
                    .and_then(|p| p.attributes.get(&w.key))
                    != Some(&w.value)
            });
        let produced = if writes_anything { vec![yaml_path.clone()] } else { vec![] };
        ctx.attributes.extend(plan.writes);

        CommandResult {
            status: Status::Ok,
            message: format!(
                "tags on {tagged} profile(s); parents {} on {} profile(s)",
                if parents_written { "named" } else { "already named" },
                plan.parents.len()
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![]
    }

    /// The step edits `dataset.yaml` in place and produces no artifact
    /// of its own.
    fn project_artifacts(&self, step_id: &str, _options: &Options) -> ArtifactManifest {
        ArtifactManifest {
            step_id: step_id.to_string(),
            command: self.command_path().to_string(),
            inputs: vec![],
            outputs: vec![],
            intermediates: vec![],
        }
    }
}

fn error(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

/// What the step will write: the tags per profile and the profiles
/// that gain `inherits: default`.
pub struct TagPlan {
    pub writes: Vec<AttributeWrite>,
    pub parents: Vec<String>,
    pub notes: Vec<String>,
}

fn facet_relpath(view: &DSView) -> String {
    let clean = strip_window_suffix(&view.source.path);
    crate::pipeline::dataset_lookup::strip_namespace(clean)
        .trim_start_matches("./")
        .to_string()
}

/// The tags the generators would write today (PS-12), from the
/// dataset's declarations and the completed steps that produced its
/// predicate facets.
pub fn plan_tags(
    config: &DatasetConfig,
    workspace: &Path,
    steps: &std::collections::HashMap<String, StepRecord>,
) -> Result<TagPlan, String> {
    let group = &config.profiles;
    let mut writes: Vec<AttributeWrite> = Vec::new();
    let mut parents: Vec<String> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    // The facet-describing tags of each predicate slab, computed once.
    let mut facet_tags: BTreeMap<String, Vec<(String, serde_yaml::Value)>> = BTreeMap::new();

    for (name, profile) in &group.profiles {
        let mut tags: Vec<(String, serde_yaml::Value)> = Vec::new();

        // size: a member whose name spells its own count — `64mi`,
        // `10m` — carries that spelling as its rung (PS-20); the
        // default's is the rung of what its base holds; any other
        // profile with a count, the decimal rung of that count.
        let size = if name == "default" {
            group.effective_view(name, "base_vectors")
                .and_then(|v| v.record_count.or(v.source.declared_count))
                .or(profile.base_count)
                .map(size_rung)
        } else {
            profile.base_count.map(|count| {
                if vectordata::dataset::selector::parse_number(name) == Some(count as f64) {
                    name.clone()
                } else {
                    size_rung(count)
                }
            })
        };
        if let Some(size) = size {
            tags.push(("size".to_string(), serde_yaml::Value::from(size)));
        }

        // The class of the predicate facet this profile reads, from the
        // step that produced it and a census of what it holds. A layer
        // in a layered dataset holds no predicate group and carries no
        // class of its own (PL-1, PL-11).
        if !(group.layered && group.is_layer(name))
            && let Some(view) = group.effective_view(name, "metadata_predicates")
        {
            let rel = facet_relpath(view);
            if !facet_tags.contains_key(&rel) {
                facet_tags.insert(rel.clone(), predicate_facet_tags(workspace, &rel, steps, &mut notes)?);
            }
            tags.extend(facet_tags[&rel].iter().cloned());
        }

        // Every tag the plan holds is asked for, present or not: the
        // write is idempotent, and the record then carries the whole
        // plan (PS-13), so a hand edit is reported after a run that
        // wrote nothing as much as after one that did.
        for (key, value) in tags {
            writes.push(AttributeWrite { profile: name.clone(), key, value });
        }
        if name != "default" && !profile.partition && profile.inherits.is_none() {
            parents.push(name.clone());
        }
    }
    Ok(TagPlan { writes, parents, notes })
}

/// `family`, `selectivity_ladder`, `forms` and `predicates` for one
/// predicate slab: the family and ladder from the record of the step
/// that produced it, the form count and class from a census of the
/// file (PS-23).
fn predicate_facet_tags(
    workspace: &Path,
    rel: &str,
    steps: &std::collections::HashMap<String, StepRecord>,
    notes: &mut Vec<String>,
) -> Result<Vec<(String, serde_yaml::Value)>, String> {
    let mut tags: Vec<(String, serde_yaml::Value)> = Vec::new();
    let producer = steps.values().find(|r| {
        r.status == Status::Ok
            && r.outputs.iter().any(|o| o.path.trim_start_matches("./") == rel)
            && r.resolved_options.contains_key("strategy")
    });
    match producer {
        Some(record) => {
            let strategy = record.resolved_options["strategy"].clone();
            tags.push(("family".to_string(), serde_yaml::Value::from(strategy.clone())));
            if strategy == "stratified" {
                let spec = record
                    .resolved_options
                    .get("decades")
                    .map(String::as_str)
                    .unwrap_or(super::gen_predicates_stratified::DEFAULT_DECADES);
                let decades = super::gen_predicates_stratified::parse_decades(spec)?;
                let ladder: Vec<serde_yaml::Value> =
                    decades.iter().map(|&d| serde_yaml::Value::from(10f64.powi(d))).collect();
                tags.push(("selectivity_ladder".to_string(), serde_yaml::Value::Sequence(ladder)));
            }
        }
        None => notes.push(format!(
            "{rel}: no completed generator step records this facet; family and ladder not tagged"
        )),
    }
    let path = workspace.join(rel);
    if !path.exists() {
        notes.push(format!("{rel}: not present; forms and predicates not tagged"));
        return Ok(tags);
    }
    let classes = super::analyze_predicate_forms::facet_form_classes(&path)?;
    let class = match (classes.forms, classes.parts) {
        (1, Some(parts)) => format!("uniform-{parts}"),
        _ => "mixed".to_string(),
    };
    tags.push(("forms".to_string(), serde_yaml::Value::from(classes.forms as u64)));
    tags.push(("predicates".to_string(), serde_yaml::Value::from(class)));
    Ok(tags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::progress::{OutputRecord, ProgressLog};
    use chrono::Utc;
    use std::collections::HashMap;

    const YAML: &str = "format_version: 2\nname: t\n\nprofiles:\n  default:\n    maxk: 10\n    base_vectors:\n      source: profiles/base/b__NNNN.fvecs\n      shard_stride: 100000000\n      shard_count: 5\n      record_count: 495930736\n    metadata_predicates: profiles/base/predicates.slab\n  10m:\n    base_count: 10000000  # ten million\n    neighbor_indices: profiles/10m/gt.ivecs\n  64mi:\n    base_count: 67108864\n  odd:\n    base_count: 1234567\n  part-0:\n    partition: true\n    base_count: 5\n    base_vectors: profiles/part-0/b.fvecs\n";

    fn ctx_at(workspace: &Path) -> StreamContext {
        StreamContext {
            attributes: Vec::new(),
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            cache: workspace.join(".cache"),
            defaults: indexmap::IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    fn generator_record() -> StepRecord {
        let mut opts = HashMap::new();
        opts.insert("strategy".to_string(), "stratified".to_string());
        opts.insert("output".to_string(), "profiles/base/predicates.slab".to_string());
        StepRecord {
            attributes: Vec::new(),
            status: Status::Ok,
            message: String::new(),
            completed_at: Utc::now(),
            elapsed_secs: 1.0,
            outputs: vec![OutputRecord {
                path: "profiles/base/predicates.slab".to_string(),
                size: 1,
                mtime: None,
            }],
            resolved_options: opts,
            error: None,
            resource_summary: None,
            provenance: None,
        }
    }

    /// **The plan is what the generators would write today** (PS-12):
    /// the default's rung from what its base holds, a member's from its
    /// count, the facet's family and ladder from the producing step,
    /// parents on everything but default and the partitions.
    #[test]
    fn the_plan_reads_the_declarations_and_the_producing_step() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("dataset.yaml"), YAML).unwrap();
        let config = DatasetConfig::load(&tmp.path().join("dataset.yaml")).unwrap();
        let mut log = ProgressLog::new();
        log.steps.insert("generate-predicates".to_string(), generator_record());
        let plan = plan_tags(&config, tmp.path(), &log.steps).unwrap();

        let get = |profile: &str, key: &str| -> Option<serde_yaml::Value> {
            plan.writes.iter().find(|w| w.profile == profile && w.key == key).map(|w| w.value.clone())
        };
        assert_eq!(get("default", "size"), Some(serde_yaml::Value::from("495m")));
        assert_eq!(get("10m", "size"), Some(serde_yaml::Value::from("10m")));
        assert_eq!(get("64mi", "size"), Some(serde_yaml::Value::from("64mi")), "a name that spells its count is the rung");
        assert_eq!(get("odd", "size"), Some(serde_yaml::Value::from("1m")), "any other count takes the decimal rung");
        assert_eq!(get("part-0", "size"), Some(serde_yaml::Value::from("5")));
        assert_eq!(get("default", "family"), Some(serde_yaml::Value::from("stratified")));
        assert_eq!(get("10m", "family"), Some(serde_yaml::Value::from("stratified")), "the member reads the default's facet");
        assert!(get("part-0", "family").is_none(), "a partition reads no shared facet");
        let ladder = get("10m", "selectivity_ladder").unwrap();
        assert_eq!(ladder.as_sequence().unwrap().len(), 7);
        assert!(get("default", "predicates").is_none(), "no slab present, no census");
        assert!(plan.notes.iter().any(|n| n.contains("not present")));
        assert_eq!(plan.parents, vec!["10m".to_string(), "64mi".to_string(), "odd".to_string()]);
    }

    /// **A re-run writes nothing** (PL-12): once the tags and parents
    /// are in the file, the plan is empty and the file is untouched,
    /// comments included.
    #[test]
    fn a_second_run_writes_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml_path = tmp.path().join("dataset.yaml");
        std::fs::write(&yaml_path, YAML).unwrap();
        let mut log = ProgressLog::new();
        log.steps.insert("generate-predicates".to_string(), generator_record());
        let config = DatasetConfig::load(&yaml_path).unwrap();
        let plan = plan_tags(&config, tmp.path(), &log.steps).unwrap();

        // Apply as the runner would: parents by the step, tags after it.
        let mut text = std::fs::read_to_string(&yaml_path).unwrap();
        for p in &plan.parents {
            text = vectordata::dataset::yaml_edit::set_profile_inherits(&text, p, "default").unwrap();
        }
        let mut by_profile: BTreeMap<String, Vec<(String, serde_yaml::Value)>> = BTreeMap::new();
        for w in &plan.writes {
            by_profile.entry(w.profile.clone()).or_default().push((w.key.clone(), w.value.clone()));
        }
        for (p, attrs) in &by_profile {
            text = vectordata::dataset::yaml_edit::set_profile_attributes(&text, p, attrs).unwrap();
        }
        std::fs::write(&yaml_path, &text).unwrap();
        assert!(text.contains("    base_count: 10000000  # ten million\n"), "comments survive:\n{text}");
        assert!(
            text.contains("    inherits: default\n    base_count: 10000000  # ten million\n"),
            "the parent is named under the profile:\n{text}"
        );

        let again = DatasetConfig::load(&yaml_path).unwrap();
        let plan = plan_tags(&again, tmp.path(), &log.steps).unwrap();
        assert!(plan.writes.iter().all(|w| {
            again.profiles.profile(&w.profile).and_then(|p| p.attributes.get(&w.key)) == Some(&w.value)
        }), "a second plan asks for what is there, so the record holds the whole plan: {:?}", plan.writes);
        assert!(plan.parents.is_empty());
        let mut op = TagProfilesOp;
        let mut ctx = ctx_at(tmp.path());
        let r = op.execute(&Options::default(), &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);
        assert_eq!(std::fs::read_to_string(&yaml_path).unwrap(), text, "a re-run writes nothing");
        assert!(!ctx.attributes.is_empty(), "and still records its plan");
        assert!(r.produced.is_empty(), "nothing written, nothing produced");
    }
}
