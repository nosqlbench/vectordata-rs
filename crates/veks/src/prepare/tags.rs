// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks prepare tags` — edit the tags of the profiles a selector names,
//! as a change of plan (PS-11, PS-13).
//!
//! A tag is a plan, and a plan may be revised. The edit is textual and
//! idempotent, touching only the profiles' own `attributes:` lines, and
//! it is **recorded**: every step record holding the edited tag takes
//! the new value, and every record naming `dataset.yaml` as its output
//! takes the file's new size, so no step reads the edit as a hand
//! change it must undo and no compute step turns stale. What does
//! refresh is the published definition — dataset.json, the catalog,
//! the docs — because those are derived from the file and must carry
//! the tags a selector will read.

use std::path::PathBuf;

use serde_yaml::Value as Yaml;

use vectordata::dataset::profile::naming_tags;
use vectordata::dataset::yaml_edit::{render_scalar, set_profile_attributes, unset_profile_attribute};
use vectordata::dataset::DatasetConfig;

use crate::pipeline::progress::ProgressLog;

pub struct TagsArgs {
    pub path: PathBuf,
    pub profile: String,
    pub set: Vec<String>,
    pub unset: Vec<String>,
    pub dry_run: bool,
}

/// One edit: a key set to a value, or removed.
#[derive(Debug, Clone, PartialEq)]
pub struct Edit {
    pub profile: String,
    pub key: String,
    pub value: Option<Yaml>,
}

fn exit_with(msg: String) -> ! {
    eprintln!("Error: {msg}");
    std::process::exit(1);
}

pub fn run(args: TagsArgs) {
    let dataset_path = if args.path.is_file() { args.path.clone() } else { args.path.join("dataset.yaml") };
    let text = std::fs::read_to_string(&dataset_path)
        .unwrap_or_else(|e| exit_with(format!("{}: {e}", crate::check::rel_display(&dataset_path))));
    let config = DatasetConfig::load(&dataset_path).unwrap_or_else(|e| exit_with(e));
    let (out, edits, notes) = plan(&text, &config, &args).unwrap_or_else(|e| exit_with(e));
    for n in &notes {
        println!("  {n}");
    }
    if edits.is_empty() {
        println!("Nothing to change.");
        return;
    }
    for e in &edits {
        match &e.value {
            Some(v) => println!("  {}: {} = {}", e.profile, e.key, render_scalar(v).unwrap_or_default()),
            None => println!("  {}: {} removed", e.profile, e.key),
        }
    }
    if args.dry_run {
        println!("Dry run: {} edit(s) planned, nothing written.", edits.len());
        return;
    }
    if out != text {
        match crate::check::fix::create_backup(&dataset_path) {
            Ok(bp) => println!("  Backed up {} → {}", crate::check::rel_display(&dataset_path), crate::check::rel_display(&bp)),
            Err(e) => eprintln!("  Warning: backup failed: {e}"),
        }
        let tmp = dataset_path.with_extension("yaml.tmp");
        if let Err(e) = std::fs::write(&tmp, &out).and_then(|_| std::fs::rename(&tmp, &dataset_path)) {
            exit_with(format!("failed to write {}: {e}", crate::check::rel_display(&dataset_path)));
        }
    }
    let log_path = ProgressLog::path_for_dataset(&dataset_path);
    if log_path.exists() {
        match ProgressLog::load(&log_path) {
            Ok((mut log, _)) => {
                let touched = record(&mut log, &edits, out.len() as u64);
                if let Err(e) = log.save() {
                    exit_with(format!("failed to save the progress log: {e}"));
                }
                println!("  Recorded the plan in {touched} step record(s).");
            }
            Err(e) => eprintln!("  Warning: progress log not updated: {e}"),
        }
    }
    println!("Edited tags on {} profile(s).", edits.iter().map(|e| e.profile.as_str()).collect::<std::collections::BTreeSet<_>>().len());
}

/// Parse `key=value` with the value read as YAML, so a list or a
/// number is what it looks like and a single-quoted value keeps its
/// spelling; a bare word that YAML would misread is kept as text.
pub fn parse_set(spec: &str) -> Result<(String, Yaml), String> {
    let Some((key, value)) = spec.split_once('=') else {
        return Err(format!("--set `{spec}` is not `key=value`"));
    };
    let key = key.trim();
    if key.is_empty() {
        return Err(format!("--set `{spec}` names no key"));
    }
    let value = value.trim();
    let parsed: Yaml = serde_yaml::from_str(value).unwrap_or_else(|_| Yaml::from(value));
    let parsed = match parsed {
        Yaml::Null => return Err(format!("--set `{spec}` has no value; use --unset to remove a tag")),
        Yaml::Mapping(_) => return Err(format!("--set `{spec}`: a tag cannot be a map (PS-15)")),
        other => other,
    };
    Ok((key.to_string(), parsed))
}

/// The edited yaml, the edits applied, and notes: a naming tag edited
/// on a profile whose name was derived from its tags is noted, since
/// the name stays put (PS-22) and no longer spells the plan.
pub fn plan(text: &str, config: &DatasetConfig, args: &TagsArgs) -> Result<(String, Vec<Edit>, Vec<String>), String> {
    let profiles = config.profiles.select(Some(&args.profile)).map_err(|e| e.to_string())?;
    let mut sets: Vec<(String, Yaml)> = Vec::new();
    for s in &args.set {
        sets.push(parse_set(s)?);
    }
    for u in &args.unset {
        if u.trim().is_empty() {
            return Err("--unset names no key".to_string());
        }
    }
    if sets.is_empty() && args.unset.is_empty() {
        return Err("nothing to edit: give --set key=value or --unset key".to_string());
    }
    let naming: Vec<&str> = naming_tags(&config.profile_tags);
    let mut out = text.to_string();
    let mut edits: Vec<Edit> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    for profile in &profiles {
        let current = &config.profiles.profiles[profile].attributes;
        let generated = current.get("family").and_then(|v| v.as_str()) == Some("uniform")
            || profile.ends_with("-mixed");
        let changed: Vec<(String, Yaml)> = sets
            .iter()
            .filter(|(k, v)| current.get(k) != Some(v))
            .cloned()
            .collect();
        for (k, _) in &changed {
            if generated && naming.contains(&k.as_str()) {
                notes.push(format!(
                    "note: `{k}` names `{profile}`; the name stays as it is (PS-22) and no longer spells this tag"
                ));
            }
        }
        if !changed.is_empty() {
            out = set_profile_attributes(&out, profile, &changed)?;
            edits.extend(changed.into_iter().map(|(key, value)| Edit { profile: profile.clone(), key, value: Some(value) }));
        }
        for key in &args.unset {
            if current.contains_key(key.trim()) {
                out = unset_profile_attribute(&out, profile, key.trim())?;
                edits.push(Edit { profile: profile.clone(), key: key.trim().to_string(), value: None });
            }
        }
    }
    Ok((out, edits, notes))
}

/// Record the edits as the plan (PS-13): every record holding an edited
/// tag takes its new value or drops it, and every record naming
/// `dataset.yaml` as an output takes the file's new size. Returns how
/// many records changed.
pub fn record(log: &mut ProgressLog, edits: &[Edit], yaml_size: u64) -> usize {
    let mut touched = 0usize;
    for rec in log.steps.values_mut() {
        let mut changed = false;
        for e in edits {
            match &e.value {
                Some(v) => {
                    let rendered = render_scalar(v).unwrap_or_default();
                    for a in rec.attributes.iter_mut().filter(|a| a.profile == e.profile && a.key == e.key) {
                        if a.value != rendered {
                            a.value = rendered.clone();
                            changed = true;
                        }
                    }
                }
                None => {
                    let before = rec.attributes.len();
                    rec.attributes.retain(|a| !(a.profile == e.profile && a.key == e.key));
                    changed |= rec.attributes.len() != before;
                }
            }
        }
        if changed {
            touched += 1;
        }
    }
    let before: Vec<u64> = log.steps.values().flat_map(|r| r.outputs.iter().filter(|o| o.path.trim_start_matches("./") == "dataset.yaml").map(|o| o.size)).collect();
    log.update_output_size("dataset.yaml", yaml_size);
    touched + before.iter().filter(|s| **s != yaml_size).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::progress::{AttributeRecord, OutputRecord, StepRecord};
    use chrono::Utc;
    use std::collections::HashMap;

    const YAML: &str = "format_version: 3\nname: t\nprofile_tags:\n  size: ~\n  predicates: ~\n  selectivity: ~\nprofiles:\n  default:\n    base_vectors: b.fvec\n    attributes:\n      size: 1k\n      predicates: mixed\n      selectivity_ladder: [0.1, 0.01]\n  10m:\n    inherits: default\n    base_count: 10\n    attributes:\n      size: 10m\n      predicates: mixed  # planned\n      selectivity_ladder: [0.1, 0.01]\n  10m-uniform-2-1e-4:\n    inherits: default\n    base_count: 10\n    attributes:\n      size: 10m\n      predicates: uniform-2\n      selectivity: '1e-4'\n      family: uniform\n";

    fn args(profile: &str, set: &[&str], unset: &[&str]) -> TagsArgs {
        TagsArgs { path: PathBuf::from("."), profile: profile.into(), set: set.iter().map(|s| s.to_string()).collect(), unset: unset.iter().map(|s| s.to_string()).collect(), dry_run: false }
    }

    fn rec(attrs: &[(&str, &str, &str)], out: Option<(&str, u64)>) -> StepRecord {
        StepRecord {
            attributes: attrs.iter().map(|(p, k, v)| AttributeRecord { profile: p.to_string(), key: k.to_string(), value: v.to_string() }).collect(),
            status: crate::pipeline::command::Status::Ok,
            message: String::new(),
            completed_at: Utc::now(),
            elapsed_secs: 1.0,
            outputs: out.map(|(p, s)| vec![OutputRecord { path: p.to_string(), size: s, mtime: None }]).unwrap_or_default(),
            resolved_options: HashMap::new(),
            error: None,
            resource_summary: None,
            provenance: None,
        }
    }

    /// **A tag edit is a change of plan** (PS-13): the profiles a
    /// selector names take the value, comments survive, the edit is
    /// idempotent, and every record holding the tag takes it too, so
    /// no step reads the edit as stale.
    #[test]
    fn tags_are_edited_by_selector_and_recorded() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("dataset.yaml");
        std::fs::write(&p, YAML).unwrap();
        let config = DatasetConfig::load(&p).unwrap();
        let a = args("predicates=mixed", &["selectivity=[0.1, 0.01]", "note=back-ported"], &[]);
        let (out, edits, notes) = plan(YAML, &config, &a).unwrap();
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(edits.len(), 4, "{edits:?}");
        assert!(out.contains("      predicates: mixed  # planned\n      selectivity_ladder: [0.1, 0.01]\n      selectivity: [0.1, 0.01]\n      note: back-ported\n"), "{out}");
        assert!(!out.contains("      selectivity: [0.1, 0.01]\n      family: uniform"), "the uniform set is untouched");
        std::fs::write(&p, &out).unwrap();
        let again = DatasetConfig::load(&p).unwrap();
        let (out2, edits2, _) = plan(&out, &again, &a).unwrap();
        assert_eq!(out2, out);
        assert!(edits2.is_empty(), "a second edit writes nothing");

        let mut log = ProgressLog::new();
        log.steps.insert("tag-profiles".into(), rec(&[("10m", "predicates", "mixed"), ("10m", "note", "old")], Some(("dataset.yaml", 10))));
        log.steps.insert("compute-knn-10m".into(), rec(&[], Some(("profiles/10m/g.ivec", 5))));
        let touched = record(&mut log, &edits, out.len() as u64);
        assert_eq!(touched, 2, "the tag record and the yaml size");
        let tp = &log.steps["tag-profiles"];
        assert!(tp.attributes.iter().any(|a| a.profile == "10m" && a.key == "note" && a.value == "back-ported"));
        assert_eq!(tp.outputs[0].size, out.len() as u64);
        assert_eq!(log.steps["compute-knn-10m"].outputs[0].size, 5, "a compute record is untouched");
        assert!(log.check_step_freshness("tag-profiles", None, Some(tmp.path())).is_none(), "the edit is the plan, not a stale output");
    }

    /// A naming tag edited on a generated set is noted, never renamed;
    /// unset removes the key and its record; a map is refused.
    #[test]
    fn naming_tags_are_noted_and_unset_removes() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("dataset.yaml");
        std::fs::write(&p, YAML).unwrap();
        let config = DatasetConfig::load(&p).unwrap();
        let (out, edits, notes) = plan(YAML, &config, &args("family=uniform", &["selectivity='1e-5'"], &["form"])).unwrap();
        assert_eq!(edits.len(), 1, "form was not present, nothing to unset: {edits:?}");
        assert!(notes.iter().any(|n| n.contains("`selectivity` names `10m-uniform-2-1e-4`")), "{notes:?}");
        assert!(out.contains("  10m-uniform-2-1e-4:\n"), "the name stays");
        assert!(out.contains("      selectivity: '1e-5'\n"), "{out}");
        let (out, edits, _) = plan(YAML, &config, &args("10m", &[], &["selectivity_ladder"])).unwrap();
        assert_eq!(edits, vec![Edit { profile: "10m".into(), key: "selectivity_ladder".into(), value: None }]);
        assert!(!out.contains("  10m:\n    inherits: default\n    base_count: 10\n    attributes:\n      size: 10m\n      predicates: mixed  # planned\n      selectivity_ladder"), "{out}");
        assert!(out.contains("      predicates: mixed  # planned\n  10m-uniform"), "{out}");
        assert!(parse_set("k={a: 1}").is_err());
        assert!(parse_set("k=").is_err());
        assert_eq!(parse_set("k='1e-6'").unwrap().1, Yaml::from("1e-6"));
    }
}
