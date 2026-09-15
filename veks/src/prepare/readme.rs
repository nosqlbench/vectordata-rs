// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks prepare readme` — scaffold the dataset's `README.md`.
//!
//! `README.md` at the dataset root is how a dataset is documented: the
//! narrative a reader needs before the generated reference in `docs/`
//! makes sense — what the data is, how it was made, under what terms
//! it is shared, what its profiles, tags and predicate sets are for,
//! and what a representative query looks like. The generated
//! `docs/dataset.md` is the reference; the README is the explanation.
//! It is static payload: shipped, checksummed and merkled with the
//! data, never regenerated, never cleaned.
//!
//! The scaffold writes the standard sections with every fact the
//! definition already holds filled in, and a fill-in marker where a
//! person has to write. `veks check` refuses a dataset without a README
//! and one whose markers are still there.

use std::path::PathBuf;

use vectordata::dataset::DatasetConfig;

/// The marker the scaffold leaves where a person has to write; the
/// check refuses a README that still carries one.
pub const FILL_IN: &str = "<!-- veks: fill in -->";

/// The sections a README carries, in order.
pub const SECTIONS: [&str; 6] = [
    "What this dataset is",
    "How it was made",
    "License and attribution",
    "Profiles",
    "Tags and selectors",
    "Predicates and example queries",
];

pub struct ReadmeArgs {
    pub path: PathBuf,
    pub force: bool,
}

fn exit_with(msg: String) -> ! {
    eprintln!("Error: {msg}");
    std::process::exit(1);
}

pub fn run(args: ReadmeArgs) {
    let dir = if args.path.is_file() { args.path.parent().map(|p| p.to_path_buf()).unwrap_or_default() } else { args.path.clone() };
    let dataset_path = dir.join("dataset.yaml");
    if !dataset_path.exists() {
        exit_with(format!("no dataset.yaml at {}", crate::check::rel_display(&dir)));
    }
    let readme = dir.join("README.md");
    if readme.exists() && !args.force {
        exit_with(format!(
            "{} exists; a README is written by hand and kept — pass --force to replace it with a fresh scaffold (a backup is taken)",
            crate::check::rel_display(&readme)
        ));
    }
    let config = DatasetConfig::load(&dataset_path).unwrap_or_else(|e| exit_with(e));
    let text = scaffold(&config);
    if readme.exists() {
        match crate::check::fix::create_backup(&readme) {
            Ok(bp) => println!("  Backed up {} → {}", crate::check::rel_display(&readme), crate::check::rel_display(&bp)),
            Err(e) => eprintln!("  Warning: backup failed: {e}"),
        }
    }
    if let Err(e) = std::fs::write(&readme, &text) {
        exit_with(format!("failed to write {}: {e}", crate::check::rel_display(&readme)));
    }
    let markers = text.matches(FILL_IN).count();
    println!("Wrote {} with {markers} place(s) to fill in; `veks check` refuses the dataset until they are written.", crate::check::rel_display(&readme));
}

/// The README scaffold for a definition: the standard sections, every
/// fact the definition holds already written, a fill-in marker where
/// a person must.
pub fn scaffold(config: &DatasetConfig) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {}\n\n", config.name));
    if let Some(d) = config.description.as_deref().map(str::trim).filter(|d| !d.is_empty()) {
        out.push_str(&format!("{d}\n\n"));
    }
    let attrs = config.attributes.as_ref();
    let attr = |f: fn(&vectordata::dataset::DatasetAttributes) -> Option<String>| attrs.and_then(f);

    out.push_str(&format!("## {}\n\n", SECTIONS[0]));
    out.push_str(&format!("{FILL_IN} What the vectors and the metadata are, where the records came from, and what a row means.\n\n"));
    let mut facts: Vec<(&str, String)> = Vec::new();
    if let Some(m) = attr(|a| a.model.clone()) {
        facts.push(("Embedding model", m));
    }
    if let Some(r) = attr(|a| a.model_revision.clone()) {
        facts.push(("Model revision", format!("`{r}`")));
    }
    if let Some(d) = attr(|a| a.distance_function.clone()) {
        facts.push(("Distance", d));
    }
    if let Some(n) = attr(|a| a.is_normalized.map(|b| b.to_string())) {
        facts.push(("Normalized", n));
    }
    if let Some(bc) = config.profiles.profile("default").and_then(|p| {
        config
            .profiles
            .effective_view("default", "base_vectors")
            .and_then(|v| {
                v.record_count
                    .or(v.source.declared_count)
                    .or_else(|| v.effective_window().0.last().map(|i| i.max_excl))
            })
            .or(p.base_count)
    }) {
        facts.push(("Base vectors", bc.to_string()));
    }
    if !facts.is_empty() {
        out.push_str("| Property | Value |\n|---|---|\n");
        for (k, v) in &facts {
            out.push_str(&format!("| {k} | {v} |\n"));
        }
        out.push('\n');
    }

    out.push_str(&format!("## {}\n\n", SECTIONS[1]));
    out.push_str(&format!("{FILL_IN} The pipeline in prose: how the base and queries were drawn, how the metadata was enriched, how ground truth was computed and verified.\n\n"));
    let steps: Vec<(String, String)> = config
        .upstream
        .as_ref()
        .and_then(|u| u.steps.as_ref())
        .map(|s| {
            s.iter()
                .filter(|st| st.description.is_some())
                .map(|st| (st.effective_id(), st.description.clone().unwrap_or_default()))
                .collect()
        })
        .unwrap_or_default();
    if !steps.is_empty() {
        out.push_str("The pipeline's steps, as `dataset.yaml` declares them:\n\n");
        for (id, d) in &steps {
            out.push_str(&format!("- `{id}` — {d}\n"));
        }
        out.push('\n');
    }

    out.push_str(&format!("## {}\n\n", SECTIONS[2]));
    let mut lic: Vec<(&str, String)> = Vec::new();
    if let Some(l) = attr(|a| a.license.clone()) {
        lic.push(("License", l));
    }
    if let Some(v) = attr(|a| a.vendor.clone()) {
        lic.push(("Vendor", v));
    }
    if let Some(n) = attr(|a| a.notes.clone()) {
        lic.push(("Notes", n));
    }
    if lic.is_empty() {
        out.push_str(&format!("{FILL_IN} The terms the data is shared under, the upstream sources and what they require, and the `license`, `vendor` and `notes` attributes of `dataset.yaml`.\n\n"));
    } else {
        out.push_str("| Property | Value |\n|---|---|\n");
        for (k, v) in &lic {
            out.push_str(&format!("| {k} | {v} |\n"));
        }
        out.push_str(&format!("\n{FILL_IN} What the terms require of a user, and where the full text is (`LICENSE.md` beside this file, when present).\n\n"));
    }

    out.push_str(&format!("## {}\n\n", SECTIONS[3]));
    let g = &config.profiles;
    let mut rungs = 0usize;
    let mut layers = 0usize;
    let mut sets = 0usize;
    let mut partitions = 0usize;
    for (name, p) in &g.profiles {
        if name == "default" {
            continue;
        }
        if p.partition {
            partitions += 1;
        } else if g.is_layer(name) {
            layers += 1;
        } else if p.inherits.as_deref().is_some_and(|i| i != "default") {
            sets += 1;
        } else {
            rungs += 1;
        }
    }
    out.push_str(&format!(
        "The definition declares {} profile(s): `default`, {rungs} sized rung(s), {layers} layer(s), {sets} set(s) and {partitions} partition(s). Every profile is listed with its files in `docs/dataset.md`.\n\n{FILL_IN} What the rungs are for, what a layer and a set are here, and which profile a reader should start with.\n\n",
        g.profiles.len()
    ));

    out.push_str(&format!("## {}\n\n", SECTIONS[4]));
    if config.profile_tags.is_empty() {
        out.push_str(&format!("{FILL_IN} The dataset declares no tag schema; say how its profiles are named and addressed.\n\n"));
    } else {
        out.push_str("The tag schema (`profile_tags:` in `dataset.yaml`), in naming order; `~` marks a naming tag:\n\n");
        for (k, v) in &config.profile_tags {
            let d = match v {
                serde_yaml::Value::Null => "naming".to_string(),
                other => format!("default `{}`", vectordata::dataset::yaml_edit::render_scalar(other).unwrap_or_default()),
            };
            out.push_str(&format!("- `{k}` — {d}\n"));
        }
        out.push_str(&format!(
            "\nA profile is addressed as `{}:<selector>`, where a selector is a name or an expression over these tags — `size=10m,predicates=mixed`, `selectivity=1e-6`, `profile=*`.\n\n{FILL_IN} Two or three selectors a reader of this dataset would actually use.\n\n",
            config.name
        ));
    }

    out.push_str(&format!("## {}\n\n", SECTIONS[5]));
    let has_predicates = g.profile("default").is_some_and(|p| p.views.contains_key("metadata_predicates"))
        || g.profiles.values().any(|p| p.views.contains_key("metadata_predicates"));
    if has_predicates {
        out.push_str(&format!("{FILL_IN} The metadata fields a predicate may name, how the predicate sets were drawn (families, selectivity ladder, forms), and a handful of representative predicates written out.\n\n"));
    } else {
        out.push_str("The dataset declares no predicate facet; its ground truth is unfiltered nearest neighbours.\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAML: &str = "format_version: 3\nname: t\ndescription: some passages\nattributes:\n  distance_function: COSINE\n  is_normalized: true\n  is_zero_vector_free: true\n  is_duplicate_vector_free: true\n  model: acme/embed\n  license: CC-BY-4.0\n  vendor: Acme\nprofile_tags:\n  size: ~\n  predicates: ~\n  selectivity: ~\n  family: stratified\nupstream:\n  steps:\n  - id: compute-knn\n    run: compute knn\n    description: exact neighbours\nprofiles:\n  default:\n    base_vectors: profiles/base/b.fvec[0..1000)\n    metadata_predicates: profiles/base/p.slab\n    neighbor_indices: g.ivec\n  100:\n    inherits: default\n    base_count: 100\n    neighbor_indices: profiles/100/g.ivec\n  100-mixed:\n    inherits: '100'\n    base_count: 100\n    metadata_results: profiles/100-mixed/r.slab\n";

    /// **The scaffold carries every standard section, the facts the
    /// definition holds, and a marker wherever a person must write.**
    #[test]
    fn the_scaffold_has_the_sections_and_the_facts() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("dataset.yaml");
        std::fs::write(&p, YAML).unwrap();
        let config = DatasetConfig::load(&p).unwrap();
        let text = scaffold(&config);
        assert!(text.starts_with("# t\n\nsome passages\n\n"), "{text}");
        for s in SECTIONS {
            assert!(text.contains(&format!("## {s}\n")), "section {s} missing:\n{text}");
        }
        assert!(text.contains("| Embedding model | acme/embed |"), "{text}");
        assert!(text.contains("| Base vectors | 1000 |"), "{text}");
        assert!(text.contains("- `compute-knn` — exact neighbours\n"), "{text}");
        assert!(text.contains("| License | CC-BY-4.0 |"), "{text}");
        assert!(text.contains("3 profile(s): `default`, 0 sized rung(s), 1 layer(s), 1 set(s) and 0 partition(s)"), "{text}");
        assert!(text.contains("- `size` — naming\n") && text.contains("- `family` — default `stratified`\n"), "{text}");
        assert!(text.contains("`t:<selector>`"), "{text}");
        assert!(text.matches(FILL_IN).count() >= 5, "{text}");
        let check = crate::check::readme_report(tmp.path(), &text);
        assert!(check.iter().any(|m| m.contains("place(s) to fill in")), "{check:?}");
    }
}
