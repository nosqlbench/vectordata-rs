// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks prepare predicate-sets` — declare uniform predicate sets
//! (PL-2, PL-9, PL-11).
//!
//! For each size and each level, one set profile is declared: it
//! names the size layer as its parent, restates the count, carries the
//! tags a selector reads (`size`, `predicates: uniform-<n>`,
//! `selectivity`, `family`, `form`), and declares the predicate group
//! whole under `profiles/<set>/`. A rung that carries a predicate group
//! of its own — tessera's `10m` — gets a hand-named layer beside it,
//! `<rung>-unfiltered`, declaring the same ground-truth files (PL-11);
//! a rung that is already a layer is used as it is. One `generate
//! predicates --strategy uniform` step per set fills its slab; the
//! per-profile templates then evaluate it and compute its filtered
//! ground truth, because the set declares those facets. Every edit is
//! textual and idempotent: a set already declared is left alone.
//! Naming a layer other than `default` needs format version 3, which
//! is stated, with the standard tag schema, when the file is below it.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde_yaml::Value as Yaml;

use vectordata::dataset::profile::{profile_name_from_tags, size_rung, PREDICATE_GROUP};
use vectordata::dataset::selector::parse_number;
use vectordata::dataset::yaml_edit::{
    append_profile, append_step, render_scalar, set_format_version, set_profile_tags_schema,
};
use vectordata::dataset::DatasetConfig;

use crate::pipeline::commands::gen_predicates_uniform::Form;

pub struct PredicateSetsArgs {
    pub path: PathBuf,
    pub form: String,
    pub levels: String,
    pub sizes: Option<String>,
    pub count: Option<String>,
    pub layer_suffix: String,
}

fn exit_with(msg: String) -> ! {
    eprintln!("Error: {msg}");
    std::process::exit(1);
}

pub fn run(args: PredicateSetsArgs) {
    let dataset_path = if args.path.is_file() { args.path.clone() } else { args.path.join("dataset.yaml") };
    if !dataset_path.exists() {
        exit_with(format!("no dataset.yaml at {}", crate::check::rel_display(&args.path)));
    }
    let text = std::fs::read_to_string(&dataset_path)
        .unwrap_or_else(|e| exit_with(format!("{}: {e}", crate::check::rel_display(&dataset_path))));
    let config = DatasetConfig::load(&dataset_path).unwrap_or_else(|e| exit_with(e));
    match plan(&text, &config, &args) {
        Ok((out, summary)) => {
            if out == text {
                println!("Nothing to declare: every set named is already there.");
                return;
            }
            match crate::check::fix::create_backup(&dataset_path) {
                Ok(bp) => println!("  Backed up {} → {}", crate::check::rel_display(&dataset_path), crate::check::rel_display(&bp)),
                Err(e) => eprintln!("  Warning: backup failed: {e}"),
            }
            let tmp = dataset_path.with_extension("yaml.tmp");
            if let Err(e) = std::fs::write(&tmp, &out).and_then(|_| std::fs::rename(&tmp, &dataset_path)) {
                exit_with(format!("failed to write {}: {e}", crate::check::rel_display(&dataset_path)));
            }
            for line in summary {
                println!("  {line}");
            }
            println!();
            println!("To generate and evaluate the sets, run:");
            println!("  veks run");
        }
        Err(e) => exit_with(e),
    }
}

/// The edited yaml and what was added, or why nothing can be.
pub fn plan(text: &str, config: &DatasetConfig, args: &PredicateSetsArgs) -> Result<(String, Vec<String>), String> {
    let form = Form::parse(&args.form)?;
    let levels: Vec<String> = args
        .levels
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if levels.is_empty() {
        return Err("levels: give at least one, e.g. `1e-2,1e-3`".to_string());
    }
    for l in &levels {
        match l.parse::<f64>() {
            Ok(v) if v > 0.0 && v < 1.0 => {}
            _ => return Err(format!("level `{l}` is not a fraction in (0, 1)")),
        }
    }
    let group = &config.profiles;
    let unstated: Vec<&String> = group
        .profiles
        .iter()
        .filter(|(n, p)| n.as_str() != "default" && !p.partition && p.inherits.is_none())
        .map(|(n, _)| n)
        .collect();
    if !unstated.is_empty() {
        return Err(format!(
            "a set names its layer, which needs format version 3, and these profiles state no parent: {}; run the `config tag-profiles` step first",
            unstated.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")
        ));
    }
    let Some(default) = group.profile("default") else {
        return Err("the dataset has no default profile".to_string());
    };
    let standard: IndexMap<String, Yaml> = [("size", Yaml::Null), ("predicates", Yaml::Null), ("selectivity", Yaml::Null)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    let schema = if config.profile_tags.is_empty() { standard.clone() } else { config.profile_tags.clone() };

    let mut out = text.to_string();
    let mut summary: Vec<String> = Vec::new();
    if config.format_version < vectordata::model::FORMAT_VERSION_TAGGED {
        out = set_format_version(&out, vectordata::model::FORMAT_VERSION_TAGGED);
        summary.push(format!("format_version: {}", vectordata::model::FORMAT_VERSION_TAGGED));
        if config.profile_tags.is_empty() {
            out = set_profile_tags_schema(&out, &standard)?;
            summary.push("profile_tags: size, predicates, selectivity".to_string());
        }
    }

    // The sizes: named, or every sized profile that is not itself a set.
    let sizes: Vec<String> = match &args.sizes {
        Some(s) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
        None => group
            .profiles
            .iter()
            .filter(|(n, p)| {
                n.as_str() != "default"
                    && !p.partition
                    && p.base_count.is_some()
                    && p.inherits.as_deref() == Some("default")
                    && !n.ends_with("-mixed")
                    && !n.ends_with(&format!("-{}", args.layer_suffix))
            })
            .map(|(n, _)| n.clone())
            .collect(),
    };
    if sizes.is_empty() {
        return Err("no sized profile to declare sets under".to_string());
    }

    // The generator step every set's step follows: the survey and the
    // steps it waits on.
    let base_generator = config
        .upstream
        .as_ref()
        .and_then(|u| u.steps.as_ref())
        .and_then(|steps| steps.iter().find(|s| s.run == "generate predicates" && s.profiles.is_empty()));
    let survey = base_generator
        .and_then(|s| s.options.get("survey"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| "the pipeline has no `generate predicates` step with a `survey` option to follow".to_string())?;
    let after: Vec<String> = base_generator.map(|s| s.after.clone()).unwrap_or_default();
    let count = args.count.clone().unwrap_or_else(|| "${query_count}".to_string());

    // The predicate group's file names, as default declares them.
    let group_files: Vec<(&str, String)> = PREDICATE_GROUP
        .iter()
        .filter_map(|f| {
            default.views.get(*f).and_then(|v| {
                Path::new(vectordata::dataset::catalog::strip_window_suffix(&v.source.path))
                    .file_name()
                    .map(|n| (*f, n.to_string_lossy().to_string()))
            })
        })
        .collect();
    if group_files.is_empty() {
        return Err("the default profile declares no predicate group; a set has nothing to mirror".to_string());
    }

    let mut declared: Vec<String> = Vec::new();
    for size in &sizes {
        let Some(sp) = group.profile(size) else {
            return Err(format!("profile `{size}` is not declared"));
        };
        let Some(bc) = sp.base_count else {
            return Err(format!("profile `{size}` has no base_count; sets are declared under sized profiles"));
        };
        let size_tag = match sp.attributes.get("size").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None if parse_number(size) == Some(bc as f64) => size.clone(),
            None => size_rung(bc),
        };
        // The layer: the rung itself when it holds no predicate group,
        // else a hand-named one beside it declaring the same files.
        let layer = if group.is_layer(size) { size.clone() } else { format!("{size}-{}", args.layer_suffix) };
        if layer != *size && group.profile(&layer).is_none() && !declared.contains(&layer) {
            let mut body = vec![
                "inherits: default".to_string(),
                format!("base_count: {bc}"),
                "attributes:".to_string(),
                format!("  size: {}", render_scalar(&Yaml::from(size_tag.as_str()))?),
            ];
            for facet in ["neighbor_indices", "neighbor_distances"] {
                if let Some(v) = group.effective_view(size, facet) {
                    body.push(format!("{facet}: {}", v.source.path));
                }
            }
            out = append_profile(&out, &layer, &body)?;
            declared.push(layer.clone());
            summary.push(format!("layer {layer}: the unfiltered {size} (PL-11)"));
        }
        for level in &levels {
            let mut tags: IndexMap<String, Yaml> = IndexMap::new();
            tags.insert("size".to_string(), Yaml::from(size_tag.as_str()));
            tags.insert("predicates".to_string(), Yaml::from(format!("uniform-{}", form.arity())));
            tags.insert("selectivity".to_string(), Yaml::from(level.as_str()));
            tags.insert("form".to_string(), Yaml::from(form.id()));
            let name = profile_name_from_tags(&schema, &tags, false)?;
            if group.profile(&name).is_some() || declared.contains(&name) {
                summary.push(format!("set {name}: already declared"));
                continue;
            }
            let mut body = vec![
                format!("inherits: {}", render_scalar(&Yaml::from(layer.as_str()))?),
                format!("base_count: {bc}"),
                "attributes:".to_string(),
                format!("  size: {}", render_scalar(&Yaml::from(size_tag.as_str()))?),
                format!("  predicates: uniform-{}", form.arity()),
                format!("  selectivity: {}", render_scalar(&Yaml::from(level.as_str()))?),
                "  family: uniform".to_string(),
                format!("  form: {}", render_scalar(&Yaml::from(form.id()))?),
            ];
            for (facet, file) in &group_files {
                body.push(format!("{facet}: profiles/{name}/{file}"));
            }
            out = append_profile(&out, &name, &body)?;
            let mut step = vec![
                format!("id: generate-predicates-{name}"),
                "run: generate predicates".to_string(),
                format!("description: Uniform predicate set {} at {level} for {name}", args.form),
            ];
            if !after.is_empty() {
                step.push("after:".to_string());
                step.extend(after.iter().map(|a| format!("- {a}")));
            }
            step.push("profiles:".to_string());
            step.push(format!("- {}", render_scalar(&Yaml::from(name.as_str()))?));
            step.push("strategy: uniform".to_string());
            step.push(format!("form: {}", render_scalar(&Yaml::from(args.form.as_str()))?));
            step.push(format!("selectivity: {}", render_scalar(&Yaml::from(level.as_str()))?));
            step.push(format!("count: {}", render_scalar(&Yaml::from(count.as_str()))?));
            step.push(format!("survey: {}", render_scalar(&Yaml::from(survey.as_str()))?));
            step.push(format!("output: profiles/{name}/{}", group_files.iter().find(|(f, _)| *f == "metadata_predicates").map(|(_, n)| n.as_str()).unwrap_or("predicates.slab")));
            out = append_step(&out, &step)?;
            declared.push(name.clone());
            summary.push(format!("set {name}: inherits {layer}, form {} at {level}", form.id()));
        }
    }
    Ok((out, summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAML: &str = "format_version: 2\nname: t\nupstream:\n  defaults:\n    query_count: '3'\n  steps:\n  - id: survey\n    run: analyze survey\n  - id: generate-predicates\n    run: generate predicates\n    after:\n    - survey\n    strategy: stratified\n    survey: .cache/survey.json\n    output: profiles/base/predicates.slab\n\nprofiles:\n  default:\n    base_vectors: profiles/base/b.fvec\n    metadata_predicates: profiles/base/predicates.slab\n    metadata_results: profiles/default/metadata_results.slab\n    prefiltered_neighbor_indices: profiles/default/prefiltered_neighbor_indices.ivec\n    neighbor_indices: profiles/default/neighbor_indices.ivecs\n  10m:\n    inherits: default\n    base_count: 10000000\n    attributes:\n      size: 10m\n      predicates: mixed\n    neighbor_indices: profiles/10m/neighbor_indices.ivecs\n    metadata_results: profiles/10m/metadata_results.slab\n    prefiltered_neighbor_indices: profiles/10m/prefiltered_neighbor_indices.ivec\n";

    fn args(sizes: Option<&str>) -> PredicateSetsArgs {
        PredicateSetsArgs {
            path: PathBuf::from("."),
            form: "topic_l3.eq+citation_percentile.range".into(),
            levels: "1e-2,1e-3".into(),
            sizes: sizes.map(str::to_string),
            count: None,
            layer_suffix: "unfiltered".into(),
        }
    }

    /// **A rung carrying its own predicate group gets a hand-named
    /// layer, and one set per level under it** (PL-11, PL-13), the file
    /// lifted to version 3 with the standard schema; the edit is
    /// idempotent and the result loads.
    #[test]
    fn sets_are_declared_under_a_layer_beside_the_rung() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("dataset.yaml");
        std::fs::write(&p, YAML).unwrap();
        let config = DatasetConfig::load(&p).unwrap();
        let (out, summary) = plan(YAML, &config, &args(Some("10m"))).unwrap();
        assert!(out.starts_with("format_version: 3\n"), "{out}");
        assert!(out.contains("profile_tags:\n  size: ~\n  predicates: ~\n  selectivity: ~\n"), "{out}");
        assert!(out.contains("  10m-unfiltered:\n    inherits: default\n    base_count: 10000000\n    attributes:\n      size: 10m\n    neighbor_indices: profiles/10m/neighbor_indices.ivecs\n"), "{out}");
        assert!(out.contains("  10m-uniform-2-1e-2:\n    inherits: 10m-unfiltered\n    base_count: 10000000\n    attributes:\n      size: 10m\n      predicates: uniform-2\n      selectivity: '1e-2'\n      family: uniform\n      form: citation_percentile.range_topic_l3.eq\n    metadata_predicates: profiles/10m-uniform-2-1e-2/predicates.slab\n    metadata_results: profiles/10m-uniform-2-1e-2/metadata_results.slab\n    prefiltered_neighbor_indices: profiles/10m-uniform-2-1e-2/prefiltered_neighbor_indices.ivec\n"), "{out}");
        assert!(out.contains("  - id: generate-predicates-10m-uniform-2-1e-3\n    run: generate predicates\n"), "{out}");
        assert!(out.contains("    profiles:\n    - 10m-uniform-2-1e-3\n    strategy: uniform\n    form: 'topic_l3.eq+citation_percentile.range'\n    selectivity: '1e-3'\n    count: '${query_count}'\n    survey: '.cache/survey.json'\n    output: profiles/10m-uniform-2-1e-3/predicates.slab\n"), "{out}");
        assert!(summary.iter().any(|s| s.starts_with("layer 10m-unfiltered")), "{summary:?}");

        std::fs::write(&p, &out).unwrap();
        let reloaded = DatasetConfig::load(&p).unwrap();
        assert_eq!(reloaded.format_version, 3);
        assert!(reloaded.profiles.is_layer("10m-unfiltered"));
        assert!(!reloaded.profiles.is_layer("10m-uniform-2-1e-2"));
        assert_eq!(reloaded.profiles.profile("10m-uniform-2-1e-2").unwrap().inherits.as_deref(), Some("10m-unfiltered"));
        let (again, summary) = plan(&out, &reloaded, &args(Some("10m"))).unwrap();
        assert_eq!(again, out, "a second declaration writes nothing");
        assert!(summary.iter().all(|s| s.contains("already declared")), "{summary:?}");
    }

    /// A rung that is already a layer is used as the parent directly.
    #[test]
    fn a_layer_is_its_own_parent() {
        let yaml = YAML.replace("format_version: 2", "format_version: 3").replace(
            "    metadata_results: profiles/10m/metadata_results.slab\n    prefiltered_neighbor_indices: profiles/10m/prefiltered_neighbor_indices.ivec\n",
            "",
        );
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("dataset.yaml");
        std::fs::write(&p, &yaml).unwrap();
        let config = DatasetConfig::load(&p).unwrap();
        let (out, _) = plan(&yaml, &config, &args(Some("10m"))).unwrap();
        assert!(!out.contains("10m-unfiltered"), "{out}");
        assert!(out.contains("  10m-uniform-2-1e-2:\n    inherits: 10m\n"), "{out}");
    }
}
