// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks prepare predicate-sets` — declare uniform predicate sets
//! (PL-2, PL-9, PL-11) as a grid of sizes and levels.
//!
//! A level's predicates are the same at every size: they are drawn from
//! the census of the whole base, so one `generate predicates --strategy
//! uniform` step per level writes one slab, under
//! `profiles/base/uniform-<n>-<level>/`, and every set at the level
//! declares it — the way every mixed rung declares the stratified slab.
//! The evaluation of a set then shares its segment cache with the sets
//! of the smaller rungs. Each set profile names its size layer as its
//! parent, restates the count, carries the tags a selector reads
//! (`size`, `predicates: uniform-<n>`, `selectivity`, `family`, `form`),
//! and declares the rest of the predicate group under `profiles/<set>/`.
//! A rung that carries a predicate group of its own — tessera's `10m` —
//! gets a hand-named layer beside it, `<rung>-unfiltered`, declaring the
//! same ground-truth files (PL-11); a rung that is already a layer is
//! used as it is.
//!
//! A cell is declared only where the level meets the **floor**: the
//! matches a median predicate of the level is expected to find in the
//! rung's rows, from the census, must reach `--min-matches` (default
//! 100, the reliability floor of the stratified generator; 0 declares
//! every cell). Below it most predicates would match nothing.
//!
//! Every edit is textual and idempotent: a set already declared is left
//! alone, and a set declared with a slab of its own is **trued up** to
//! the level's slab — its view re-pointed, its own generator step
//! removed, the files it no longer reads named for removal. Naming a
//! layer other than `default` needs format version 3, which is stated,
//! with the standard tag schema, when the file is below it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde_yaml::Value as Yaml;

use vectordata::dataset::pipeline::StepDef;
use vectordata::dataset::profile::{profile_name_from_tags, size_rung, PREDICATE_GROUP};
use vectordata::dataset::selector::parse_number;
use vectordata::dataset::yaml_edit::{
    add_step_profile, append_profile, append_step, remove_step, render_scalar, set_format_version,
    set_profile_tags_schema, set_profile_view,
};
use vectordata::dataset::DatasetConfig;

use crate::pipeline::commands::gen_predicates_uniform::{level_census, Form, LevelCensus};

pub struct PredicateSetsArgs {
    pub path: PathBuf,
    pub form: String,
    pub levels: String,
    pub sizes: Option<String>,
    pub count: Option<String>,
    pub layer_suffix: String,
    /// The floor of expected matches per predicate a cell must meet; 0
    /// declares every cell.
    pub min_matches: u64,
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
    let form = Form::parse(&args.form).unwrap_or_else(|e| exit_with(e));
    let levels = parse_levels(&args.levels).unwrap_or_else(|e| exit_with(e));
    let censuses = if args.min_matches > 0 {
        censuses_for(&config, &dataset_path, &form, &levels).unwrap_or_else(|e| exit_with(e))
    } else {
        HashMap::new()
    };
    match plan(&text, &config, &args, &censuses) {
        Ok((out, summary)) => {
            for line in &summary {
                println!("  {line}");
            }
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
            println!();
            println!("To generate and evaluate the sets, run:");
            println!("  veks run");
        }
        Err(e) => exit_with(e),
    }
}

/// The levels as spelled and as read: each a fraction in (0, 1).
pub fn parse_levels(spec: &str) -> Result<Vec<(String, f64)>, String> {
    let mut out = Vec::new();
    for l in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match l.parse::<f64>() {
            Ok(v) if v > 0.0 && v < 1.0 => out.push((l.to_string(), v)),
            _ => return Err(format!("level `{l}` is not a fraction in (0, 1)")),
        }
    }
    if out.is_empty() {
        return Err("levels: give at least one, e.g. `1e-2,1e-3`".to_string());
    }
    Ok(out)
}

/// The generator step every set's step follows: the shared `generate
/// predicates` step, whose survey and upstreams the level steps take.
fn base_generator(config: &DatasetConfig) -> Option<&StepDef> {
    config
        .upstream
        .as_ref()
        .and_then(|u| u.steps.as_ref())
        .and_then(|steps| steps.iter().find(|s| s.run == "generate predicates" && s.profiles.is_empty()))
}

fn survey_option(config: &DatasetConfig) -> Result<String, String> {
    base_generator(config)
        .and_then(|s| s.options.get("survey"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| "the pipeline has no `generate predicates` step with a `survey` option to follow".to_string())
}

/// A step option with its variables read: `${cache}` is the workspace
/// cache, any other `${name}` a dataset variable. What stays unresolved
/// is an error naming it.
fn resolve_vars(s: &str, config: &DatasetConfig) -> Result<String, String> {
    let mut out = s.replace("${cache}", ".cache");
    for (k, v) in &config.variables {
        out = out.replace(&format!("${{{k}}}"), v);
    }
    if out.contains("${") {
        return Err(format!("`{s}` names a variable the dataset does not hold; pass --min-matches 0 to declare without the census"));
    }
    Ok(out)
}

/// The census of each level from the survey the base generator names,
/// keyed by the level as spelled.
pub fn censuses_for(
    config: &DatasetConfig,
    dataset_path: &Path,
    form: &Form,
    levels: &[(String, f64)],
) -> Result<HashMap<String, Option<LevelCensus>>, String> {
    let survey = resolve_vars(&survey_option(config)?, config)?;
    let workspace = dataset_path.parent().unwrap_or(Path::new("."));
    let path = workspace.join(survey);
    if !path.exists() {
        return Err(format!(
            "the survey {} is not there yet; run the pipeline through its survey step, or pass --min-matches 0",
            crate::check::rel_display(&path)
        ));
    }
    let mut out = HashMap::new();
    for (text, value) in levels {
        out.insert(text.clone(), level_census(&path, form, *value)?);
    }
    Ok(out)
}

/// The edited yaml and what was declared, skipped or trued up, or why
/// nothing can be. `censuses` holds each level's census when a floor
/// applies; a level absent from it is declared at every size.
pub fn plan(
    text: &str,
    config: &DatasetConfig,
    args: &PredicateSetsArgs,
    censuses: &HashMap<String, Option<LevelCensus>>,
) -> Result<(String, Vec<String>), String> {
    let form = Form::parse(&args.form)?;
    let levels = parse_levels(&args.levels)?;
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

    let survey = survey_option(config)?;
    let after: Vec<String> = base_generator(config).map(|s| s.after.clone()).unwrap_or_default();
    let count = args.count.clone().unwrap_or_else(|| "${query_count}".to_string());
    let declared_steps: Vec<String> = config
        .upstream
        .as_ref()
        .and_then(|u| u.steps.as_ref())
        .map(|steps| steps.iter().map(|s| s.effective_id()).collect())
        .unwrap_or_default();

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
    let Some((_, slab_file)) = group_files.iter().find(|(f, _)| *f == "metadata_predicates") else {
        return Err("the default profile declares no `metadata_predicates`; a set has no slab to mirror".to_string());
    };
    let arity = form.arity();

    let mut declared: Vec<String> = Vec::new();
    let mut new_steps: Vec<String> = Vec::new();
    for (level, level_value) in &levels {
        let census = censuses.get(level).cloned().flatten();
        if args.min_matches > 0 && censuses.contains_key(level) && census.is_none() {
            summary.push(format!(
                "level {level}: the form is not a censused pair, so the floor cannot be read; declaring it at every size"
            ));
        }
        let level_dir = format!("profiles/base/uniform-{arity}-{level}");
        let shared = format!("{level_dir}/{slab_file}");
        let level_step = format!("generate-predicates-uniform-{arity}-{level}");
        let mut members: Vec<String> = Vec::new();
        for size in &sizes {
            let Some(sp) = group.profile(size) else {
                return Err(format!("profile `{size}` is not declared"));
            };
            let Some(bc) = sp.base_count else {
                return Err(format!("profile `{size}` has no base_count; sets are declared under sized profiles"));
            };
            if let Some(c) = &census {
                let expected = c.expected_matches(bc);
                if expected < args.min_matches as f64 {
                    summary.push(format!(
                        "skipped {size} at {level}: a median predicate expects {expected:.0} match(es) in {bc} rows, below the floor of {}",
                        args.min_matches
                    ));
                    continue;
                }
            }
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
            let mut tags: IndexMap<String, Yaml> = IndexMap::new();
            tags.insert("size".to_string(), Yaml::from(size_tag.as_str()));
            tags.insert("predicates".to_string(), Yaml::from(format!("uniform-{arity}")));
            tags.insert("selectivity".to_string(), Yaml::from(level.as_str()));
            tags.insert("form".to_string(), Yaml::from(form.id()));
            let name = profile_name_from_tags(&schema, &tags, false)?;
            if let Some(existing) = group.profile(&name) {
                // Declared already: trued up to the level's slab when it
                // reads one of its own.
                let own = existing
                    .views
                    .get("metadata_predicates")
                    .map(|v| vectordata::dataset::catalog::strip_window_suffix(&v.source.path).to_string());
                match own {
                    Some(p) if p != shared => {
                        out = set_profile_view(&out, &name, "metadata_predicates", &shared)?;
                        let own_step = format!("generate-predicates-{name}");
                        out = remove_step(&out, &own_step)?;
                        let report = Path::new(&p).with_extension("json").to_string_lossy().to_string();
                        summary.push(format!(
                            "set {name}: now reads {shared}; its own step {own_step} is removed, and {p} with {report} are unreferenced — remove them once the shared slab is generated"
                        ));
                    }
                    _ => summary.push(format!("set {name}: already declared")),
                }
                members.push(name);
                continue;
            }
            if declared.contains(&name) {
                summary.push(format!("set {name}: already declared"));
                members.push(name);
                continue;
            }
            let mut body = vec![
                format!("inherits: {}", render_scalar(&Yaml::from(layer.as_str()))?),
                format!("base_count: {bc}"),
                "attributes:".to_string(),
                format!("  size: {}", render_scalar(&Yaml::from(size_tag.as_str()))?),
                format!("  predicates: uniform-{arity}"),
                format!("  selectivity: {}", render_scalar(&Yaml::from(level.as_str()))?),
                "  family: uniform".to_string(),
                format!("  form: {}", render_scalar(&Yaml::from(form.id()))?),
            ];
            for (facet, file) in &group_files {
                if *facet == "metadata_predicates" {
                    body.push(format!("{facet}: {shared}"));
                } else {
                    body.push(format!("{facet}: profiles/{name}/{file}"));
                }
            }
            out = append_profile(&out, &name, &body)?;
            declared.push(name.clone());
            summary.push(format!("set {name}: inherits {layer}, form {} at {level}", form.id()));
            members.push(name);
        }
        if members.is_empty() {
            continue;
        }
        if declared_steps.contains(&level_step) || new_steps.contains(&level_step) {
            for m in &members {
                out = add_step_profile(&out, &level_step, m)?;
            }
            continue;
        }
        let mut step = vec![
            format!("id: {level_step}"),
            "run: generate predicates".to_string(),
            format!("description: Uniform predicate set {} at {level}, one slab shared by every set at the level", args.form),
        ];
        if !after.is_empty() {
            step.push("after:".to_string());
            step.extend(after.iter().map(|a| format!("- {a}")));
        }
        step.push("profiles:".to_string());
        for m in &members {
            step.push(format!("- {}", render_scalar(&Yaml::from(m.as_str()))?));
        }
        step.push("strategy: uniform".to_string());
        step.push(format!("form: {}", render_scalar(&Yaml::from(args.form.as_str()))?));
        step.push(format!("selectivity: {}", render_scalar(&Yaml::from(level.as_str()))?));
        step.push(format!("count: {}", render_scalar(&Yaml::from(count.as_str()))?));
        step.push(format!("survey: {}", render_scalar(&Yaml::from(survey.as_str()))?));
        step.push(format!("output: {shared}"));
        out = append_step(&out, &step)?;
        new_steps.push(level_step.clone());
        summary.push(format!("step {level_step}: writes {shared} for {} set(s)", members.len()));
        let _ = level_value;
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
            min_matches: 0,
        }
    }

    fn none() -> HashMap<String, Option<LevelCensus>> {
        HashMap::new()
    }

    /// **A rung carrying its own predicate group gets a hand-named
    /// layer, and one set per level under it, every set at a level
    /// reading the level's one slab** (PL-11, PL-13); the file is lifted
    /// to version 3 with the standard schema, the edit is idempotent and
    /// the result loads.
    #[test]
    fn sets_are_declared_under_a_layer_beside_the_rung() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("dataset.yaml");
        std::fs::write(&p, YAML).unwrap();
        let config = DatasetConfig::load(&p).unwrap();
        let (out, summary) = plan(YAML, &config, &args(Some("10m")), &none()).unwrap();
        assert!(out.starts_with("format_version: 3\n"), "{out}");
        assert!(out.contains("profile_tags:\n  size: ~\n  predicates: ~\n  selectivity: ~\n"), "{out}");
        assert!(out.contains("  10m-unfiltered:\n    inherits: default\n    base_count: 10000000\n    attributes:\n      size: 10m\n    neighbor_indices: profiles/10m/neighbor_indices.ivecs\n"), "{out}");
        assert!(out.contains("  10m-uniform-2-1e-2:\n    inherits: 10m-unfiltered\n    base_count: 10000000\n    attributes:\n      size: 10m\n      predicates: uniform-2\n      selectivity: '1e-2'\n      family: uniform\n      form: citation_percentile.range_topic_l3.eq\n    metadata_predicates: profiles/base/uniform-2-1e-2/predicates.slab\n    metadata_results: profiles/10m-uniform-2-1e-2/metadata_results.slab\n    prefiltered_neighbor_indices: profiles/10m-uniform-2-1e-2/prefiltered_neighbor_indices.ivec\n"), "{out}");
        assert!(out.contains("  - id: generate-predicates-uniform-2-1e-3\n    run: generate predicates\n"), "{out}");
        assert!(out.contains("    profiles:\n    - 10m-uniform-2-1e-3\n    strategy: uniform\n    form: 'topic_l3.eq+citation_percentile.range'\n    selectivity: '1e-3'\n    count: '${query_count}'\n    survey: '.cache/survey.json'\n    output: profiles/base/uniform-2-1e-3/predicates.slab\n"), "{out}");
        assert!(!out.contains("generate-predicates-10m-uniform"), "no step per set: {out}");
        assert!(summary.iter().any(|s| s.starts_with("layer 10m-unfiltered")), "{summary:?}");

        std::fs::write(&p, &out).unwrap();
        let reloaded = DatasetConfig::load(&p).unwrap();
        assert_eq!(reloaded.format_version, 3);
        assert!(reloaded.profiles.is_layer("10m-unfiltered"));
        assert!(!reloaded.profiles.is_layer("10m-uniform-2-1e-2"));
        assert_eq!(reloaded.profiles.profile("10m-uniform-2-1e-2").unwrap().inherits.as_deref(), Some("10m-unfiltered"));
        let (again, summary) = plan(&out, &reloaded, &args(Some("10m")), &none()).unwrap();
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
        let (out, _) = plan(&yaml, &config, &args(Some("10m")), &none()).unwrap();
        assert!(!out.contains("10m-unfiltered"), "{out}");
        assert!(out.contains("  10m-uniform-2-1e-2:\n    inherits: 10m\n"), "{out}");
    }

    /// **A grid grows by rung and is floored by the census** (PL-11): a
    /// second rung joins the level's step, a cell whose median predicate
    /// would find too few matches is skipped and said so, and a floor of
    /// 0 declares it.
    #[test]
    fn a_grid_grows_and_is_floored() {
        let yaml = YAML.replace("format_version: 2", "format_version: 3")
            + "  1m:\n    inherits: default\n    base_count: 1000000\n    attributes:\n      size: 1m\n      predicates: mixed\n    neighbor_indices: profiles/1m/neighbor_indices.ivecs\n    metadata_results: profiles/1m/metadata_results.slab\n    prefiltered_neighbor_indices: profiles/1m/prefiltered_neighbor_indices.ivec\n";
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("dataset.yaml");
        std::fs::write(&p, &yaml).unwrap();
        let config = DatasetConfig::load(&p).unwrap();
        let mut a = args(None);
        a.levels = "1e-3".into();
        a.min_matches = 100;
        // A median predicate matches 1,000 of 100m rows: 100 at 10m, 10 at 1m.
        let mut censuses = HashMap::new();
        censuses.insert("1e-3".to_string(), Some(LevelCensus { pairs: 50, median_count: 1000, population: 100_000_000 }));
        let (out, summary) = plan(&yaml, &config, &a, &censuses).unwrap();
        assert!(summary.iter().any(|s| s.starts_with("skipped 1m at 1e-3") && s.contains("10 match")), "{summary:?}");
        assert!(out.contains("  10m-uniform-2-1e-3:\n"), "{out}");
        assert!(!out.contains("  1m-uniform-2-1e-3:\n"), "{out}");
        assert!(out.contains("    profiles:\n    - 10m-uniform-2-1e-3\n    strategy: uniform\n"), "{out}");
        // Without the floor the cell is declared and joins the level's step.
        std::fs::write(&p, &out).unwrap();
        let reloaded = DatasetConfig::load(&p).unwrap();
        a.min_matches = 0;
        let (grown, summary) = plan(&out, &reloaded, &a, &none()).unwrap();
        assert!(grown.contains("  1m-uniform-2-1e-3:\n    inherits: 1m-unfiltered\n"), "{grown}");
        assert!(grown.contains("    profiles:\n    - 10m-uniform-2-1e-3\n    - 1m-uniform-2-1e-3\n    strategy: uniform\n"), "{grown}");
        assert_eq!(grown.matches("- id: generate-predicates-uniform-2-1e-3\n").count(), 1, "one step per level: {grown}");
        assert!(summary.iter().any(|s| s == "set 10m-uniform-2-1e-3: already declared"), "{summary:?}");
        std::fs::write(&p, &grown).unwrap();
        DatasetConfig::load(&p).unwrap();
    }

    /// **A set declared with a slab of its own is trued up to the level's
    /// slab**: its view re-pointed, its own step removed, its files named
    /// for removal, and the level's step lists it.
    #[test]
    fn a_set_with_its_own_slab_is_trued_up() {
        let yaml = YAML.replace("format_version: 2", "format_version: 3\nprofile_tags:\n  size: ~\n  predicates: ~\n  selectivity: ~")
            .replace(
                "    output: profiles/base/predicates.slab\n\n",
                "    output: profiles/base/predicates.slab\n  - id: generate-predicates-10m-uniform-2-1e-2\n    run: generate predicates\n    after:\n    - survey\n    profiles:\n    - 10m-uniform-2-1e-2\n    strategy: uniform\n    form: 'topic_l3.eq+citation_percentile.range'\n    selectivity: '1e-2'\n    count: '${query_count}'\n    survey: '.cache/survey.json'\n    output: profiles/10m-uniform-2-1e-2/predicates.slab\n\n",
            )
            + "  10m-unfiltered:\n    inherits: default\n    base_count: 10000000\n    attributes:\n      size: 10m\n    neighbor_indices: profiles/10m/neighbor_indices.ivecs\n  10m-uniform-2-1e-2:\n    inherits: 10m-unfiltered\n    base_count: 10000000\n    attributes:\n      size: 10m\n      predicates: uniform-2\n      selectivity: '1e-2'\n      family: uniform\n      form: citation_percentile.range_topic_l3.eq\n    metadata_predicates: profiles/10m-uniform-2-1e-2/predicates.slab\n    metadata_results: profiles/10m-uniform-2-1e-2/metadata_results.slab\n    prefiltered_neighbor_indices: profiles/10m-uniform-2-1e-2/prefiltered_neighbor_indices.ivec\n";
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("dataset.yaml");
        std::fs::write(&p, &yaml).unwrap();
        let config = DatasetConfig::load(&p).unwrap();
        let mut a = args(Some("10m"));
        a.levels = "1e-2".into();
        let (out, summary) = plan(&yaml, &config, &a, &none()).unwrap();
        assert!(out.contains("    metadata_predicates: profiles/base/uniform-2-1e-2/predicates.slab\n    metadata_results: profiles/10m-uniform-2-1e-2/metadata_results.slab\n"), "{out}");
        assert!(!out.contains("generate-predicates-10m-uniform-2-1e-2"), "the set's own step is gone: {out}");
        assert!(out.contains("  - id: generate-predicates-uniform-2-1e-2\n"), "{out}");
        assert!(out.contains("    profiles:\n    - 10m-uniform-2-1e-2\n    strategy: uniform\n"), "{out}");
        let note = summary.iter().find(|s| s.starts_with("set 10m-uniform-2-1e-2: now reads")).expect("a true-up note");
        assert!(note.contains("profiles/10m-uniform-2-1e-2/predicates.slab") && note.contains("profiles/10m-uniform-2-1e-2/predicates.json"), "{note}");
        std::fs::write(&p, &out).unwrap();
        let reloaded = DatasetConfig::load(&p).unwrap();
        let (again, summary) = plan(&out, &reloaded, &a, &none()).unwrap();
        assert_eq!(again, out, "trued up once");
        assert!(summary.iter().any(|s| s == "set 10m-uniform-2-1e-2: already declared"), "{summary:?}");
    }

    /// The floor scales the census to the rung; a step option's
    /// variables are read, and one the dataset lacks is named.
    #[test]
    fn the_floor_scales_and_variables_resolve() {
        let c = LevelCensus { pairs: 3, median_count: 824, population: 495_930_736 };
        assert!((c.expected_matches(100_000_000) - 166.1).abs() < 0.1);
        assert_eq!(LevelCensus { pairs: 0, median_count: 1, population: 0 }.expected_matches(5), 0.0);
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("dataset.yaml");
        std::fs::write(&p, YAML.replace(".cache/survey.json", "${cache}/survey.json")).unwrap();
        let config = DatasetConfig::load(&p).unwrap();
        assert_eq!(resolve_vars("${cache}/survey.json", &config).unwrap(), ".cache/survey.json");
        assert!(resolve_vars("${nope}/x", &config).unwrap_err().contains("${nope}/x"));
        let e = censuses_for(&config, &p, &Form::parse("a.eq+b.range").unwrap(), &[("1e-3".into(), 1e-3)]).unwrap_err();
        assert!(e.contains("not there yet"), "{e}");
    }
}
