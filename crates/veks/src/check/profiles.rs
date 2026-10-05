// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks check` over profile selectors and tags (PS-15, PS-23, V-25).
//!
//! A selector is only as good as the attributes it reads, so this
//! check reports what would make a profile unaddressable: a tag that
//! shadows a structural key, a value a selector cannot compare, a
//! family whose members do not all carry the key that tells them
//! apart, two profiles no selector can tell apart, and a predicate
//! facet without the `predicates` class or with one its form census
//! contradicts. Going forward every `dataset.yaml` states its version
//! (V-25): one that does not is refused here, and one below 3 is
//! reported as predating tag schemas.

use std::path::{Path, PathBuf};

use super::{rel_display, CheckResult};

pub fn check(dataset_files: &[PathBuf]) -> CheckResult {
    let mut failures: Vec<String> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    for ds_path in dataset_files {
        let ds_dir = ds_path.parent().unwrap_or(Path::new("."));
        let ds_rel = rel_display(ds_dir);
        let Ok(text) = std::fs::read_to_string(ds_path) else { continue };
        if !states_version(&text) {
            failures.push(format!(
                "{ds_rel}: dataset.yaml declares no format_version (V-25); \
                 every dataset states its version, so declare `format_version: 1` \
                 for a plain dataset or the version its content needs"
            ));
            continue;
        }
        let config = match vectordata::dataset::DatasetConfig::load(ds_path) {
            Ok(c) => c,
            Err(e) => {
                failures.push(format!("{ds_rel}: {e}"));
                continue;
            }
        };
        let tagged = config.format_version >= vectordata::model::FORMAT_VERSION_TAGGED;
        if !tagged {
            notes.push(format!(
                "{ds_rel}: format_version {} predates tag schemas; attributes are plain data \
                 (migrate to 3 to declare profile_tags)",
                config.format_version
            ));
        }
        check_profiles(&config, ds_dir, &ds_rel, tagged, &mut failures, &mut notes);
        check_layers(&config, ds_dir, &ds_rel, tagged, &mut failures, &mut notes);
    }
    if failures.is_empty() {
        let mut r = CheckResult::ok("profile-selectors");
        r.messages.push("every profile is addressable by selector".to_string());
        r.messages.extend(notes);
        r
    } else {
        failures.extend(notes);
        CheckResult::fail("profile-selectors", failures)
    }
}

/// Whether the file states a version at all — read from the text,
/// because the loader holds an absent field to 1 and cannot tell the
/// two apart afterwards (V-24).
fn states_version(text: &str) -> bool {
    text.lines().any(|l| {
        !l.starts_with(' ') && l.trim_start().starts_with("format_version:")
    })
}

fn check_profiles(
    config: &vectordata::dataset::DatasetConfig,
    ds_dir: &Path,
    ds_rel: &str,
    tagged: bool,
    failures: &mut Vec<String>,
    notes: &mut Vec<String>,
) {
    let profiles = &config.profiles;
    for (name, profile) in &profiles.profiles {
        for (key, value) in &profile.attributes {
            if vectordata::dataset::selector::STRUCTURAL_KEYS
                .iter()
                .any(|k| k.eq_ignore_ascii_case(key))
            {
                failures.push(format!(
                    "{ds_rel}: profile '{name}' attribute '{key}' shadows a structural key \
                     a selector reads first (PS-7); rename it"
                ));
            }
            if value.is_mapping() {
                failures.push(format!(
                    "{ds_rel}: profile '{name}' attribute '{key}' is a map, which a selector \
                     cannot compare; use dotted keys"
                ));
            }
        }
        // The class is required where the facet is declared, not where
        // it is inherited: a layer reads its parent's slab and carries
        // no class of its own (PL-1, PS-23).
        if profiles.declares(name, "metadata_predicates")
            && let Some(view) = profile.views.get("metadata_predicates")
        {
            match profile.attributes.get("predicates") {
                None if tagged => failures.push(format!(
                    "{ds_rel}: profile '{name}' declares metadata_predicates without the \
                     `predicates` tag (PS-23); tag it `mixed` or `uniform-<n>`"
                )),
                None => notes.push(format!(
                    "{ds_rel}: profile '{name}' declares metadata_predicates without the \
                     `predicates` tag (PS-23, required from format_version 3)"
                )),
                Some(value) => {
                    let text = value.as_str().unwrap_or("").to_ascii_lowercase();
                    let facet = ds_dir.join(
                        vectordata::dataset::catalog::strip_window_suffix(&view.source.path),
                    );
                    verify_predicates_class(&text, name, &facet, ds_rel, failures, notes);
                }
            }
        }
    }

    // A family is addressable only if every member carries the keys
    // that distinguish them; and two profiles alike on every key
    // cannot be told apart at all.
    for (spec, members) in &profiles.series_by_spec {
        let mut keys: Vec<&String> = members
            .iter()
            .filter_map(|m| profiles.profile(m))
            .flat_map(|p| p.attributes.keys())
            .collect();
        keys.sort();
        keys.dedup();
        for member in members {
            let Some(p) = profiles.profile(member) else { continue };
            let missing: Vec<&str> = keys
                .iter()
                .filter(|k| !p.attributes.contains_key(k.as_str()))
                .map(|k| k.as_str())
                .collect();
            if !missing.is_empty() {
                failures.push(format!(
                    "{ds_rel}: family '{spec}' member '{member}' lacks {} that other members \
                     carry, so the family is addressable only by name (PS-15)",
                    missing.join(", ")
                ));
            }
        }
    }
    if tagged {
        let named: Vec<(&String, &vectordata::dataset::DSProfile)> = profiles.profiles.iter().collect();
        for (i, (a, pa)) in named.iter().enumerate() {
            for (b, pb) in named.iter().skip(i + 1) {
                if pa.attributes == pb.attributes {
                    failures.push(format!(
                        "{ds_rel}: profiles '{a}' and '{b}' carry identical attributes, which no \
                         selector can tell apart (PS-15)"
                    ));
                }
            }
        }
    }
}

/// The layer rules (PL-3, PL-4, PL-6, PS-19): parents below 3 as
/// advisories, a parent whose declaration a direct child overrides, a
/// predicate group whose members do not belong together, and a
/// version-3 profile lacking a defaulted schema tag.
fn check_layers(
    config: &vectordata::dataset::DatasetConfig,
    ds_dir: &Path,
    ds_rel: &str,
    tagged: bool,
    failures: &mut Vec<String>,
    notes: &mut Vec<String>,
) {
    let group = &config.profiles;

    // PL-6 below 3: what version 3 will refuse, seen first.
    let facts: Vec<vectordata::dataset::parents::ParentFacts<'_>> = group
        .profiles
        .iter()
        .map(|(name, p)| vectordata::dataset::parents::ParentFacts {
            name,
            partition: p.partition,
            inherits: p.inherits.as_deref(),
        })
        .collect();
    for a in vectordata::dataset::parents::parent_advisories(config.format_version, &facts) {
        notes.push(format!("{ds_rel}: {a}"));
    }

    // PL-4: a parent is a decoy when a direct child overrides a facet
    // it declares that would have crossed the step.
    for (child, cp) in &group.profiles {
        let Some(parent) = group.parent_name(child) else { continue };
        let pp = &group.profiles[&parent];
        let size_step = cp.base_count.is_some() && cp.base_count != pp.base_count;
        for facet in cp.views.keys() {
            let crosses = !size_step
                || vectordata::dataset::profile::facet_role(facet)
                    != vectordata::dataset::profile::FacetRole::PerProfile;
            if crosses && group.declares(child, facet) && group.declares(&parent, facet) {
                failures.push(format!(
                    "{ds_rel}: profile '{parent}' declares {facet} that its direct child '{child}' \
                     overrides; a declaration no child inherits is a decoy (PL-4)"
                ));
            }
        }
    }

    // PS-19 at version 3: every defaulted schema tag is on every profile.
    if tagged {
        for (name, p) in &group.profiles {
            let missing: Vec<&str> = config
                .profile_tags
                .iter()
                .filter(|(k, v)| !v.is_null() && !p.attributes.contains_key(k.as_str()))
                .map(|(k, _)| k.as_str())
                .collect();
            if !missing.is_empty() {
                failures.push(format!(
                    "{ds_rel}: profile '{name}' lacks the schema tag(s) {}; `veks run` fills them (PS-19)",
                    missing.join(", ")
                ));
            }
        }
    }

    // PL-3: the predicate group holds together by content.
    let progress = {
        let log_path = crate::pipeline::progress::ProgressLog::path_for_dataset(&ds_dir.join("dataset.yaml"));
        crate::pipeline::progress::ProgressLog::load(&log_path).ok().map(|(l, _)| l)
    };
    let relpath = |v: &vectordata::dataset::DSView| -> String {
        vectordata::dataset::catalog::strip_window_suffix(&v.source.path)
            .split_once(':')
            .map(|(p, _)| p)
            .unwrap_or(vectordata::dataset::catalog::strip_window_suffix(&v.source.path))
            .trim_start_matches("./")
            .to_string()
    };
    for name in group.profiles.keys() {
        if group.layered && group.is_layer(name) {
            continue;
        }
        let Some(results) = group.effective_view(name, "metadata_results") else { continue };
        let results_rel = relpath(results);
        let results_path = ds_dir.join(&results_rel);
        let slab_rel = group.effective_view(name, "metadata_predicates").map(relpath);
        if let Some(slab_rel) = &slab_rel {
            let slab_path = ds_dir.join(slab_rel);
            if results_path.exists() && slab_path.exists() {
                match (slab_rows(&results_path), slab_rows(&slab_path)) {
                    (Some(r), Some(p)) if r != p => failures.push(format!(
                        "{ds_rel}: profile '{name}' results index {results_rel} holds {r} rows but its \
                         predicate slab {slab_rel} holds {p} predicates; the group does not belong \
                         together (PL-3)"
                    )),
                    _ => {}
                }
            }
        }
        let query_count = group
            .effective_view(name, "query_vectors")
            .map(relpath)
            .map(|q| ds_dir.join(q))
            .filter(|q| q.exists() && !q.to_string_lossy().contains("NNNN"))
            .and_then(|q| vectordata::io::open_vec::<f32>(&q.to_string_lossy()).ok().map(|r| r.count()));
        for facet in ["prefiltered_neighbor_indices", "postfiltered_neighbor_indices"] {
            let Some(view) = group.effective_view(name, facet) else { continue };
            let rel = relpath(view);
            let path = ds_dir.join(&rel);
            if !path.exists() || rel.contains("NNNN") {
                continue;
            }
            if let (Some(q), Ok(reader)) =
                (query_count, vectordata::io::open_vec::<i32>(&path.to_string_lossy()))
                && reader.count() != q
            {
                failures.push(format!(
                    "{ds_rel}: profile '{name}' {facet} {rel} holds {} rows for {q} queries (PL-3)",
                    reader.count()
                ));
            }
            if let Some(log) = &progress {
                let producer = log.steps.values().find(|r| {
                    r.status == crate::pipeline::command::Status::Ok
                        && r.outputs.iter().any(|o| o.path.trim_start_matches("./") == rel)
                });
                match producer {
                    Some(record) => {
                        if let Some(from) = record.resolved_options.get("metadata-indices")
                            && from.trim_start_matches("./") != results_rel
                        {
                            failures.push(format!(
                                "{ds_rel}: profile '{name}' {facet} was computed from {from}, not from \
                                 its results index {results_rel} (PL-3)"
                            ));
                        }
                    }
                    None => notes.push(format!(
                        "{ds_rel}: profile '{name}' {facet} has no completed step record; its \
                         derivation is unverified"
                    )),
                }
            }
        }
    }
}

fn slab_rows(path: &Path) -> Option<u64> {
    slabtastic::SlabReader::open(path).ok().map(|r| r.total_records())
}

/// Hold a `predicates` class to the facet's form census (PS-23).
fn verify_predicates_class(
    class: &str,
    profile: &str,
    facet: &Path,
    ds_rel: &str,
    failures: &mut Vec<String>,
    notes: &mut Vec<String>,
) {
    let uniform_parts = class
        .strip_prefix("uniform-")
        .and_then(|n| n.parse::<usize>().ok());
    if class != "mixed" && uniform_parts.is_none() {
        failures.push(format!(
            "{ds_rel}: profile '{profile}' predicates tag '{class}' is neither `mixed` nor \
             `uniform-<n>` (PS-23)"
        ));
        return;
    }
    if !facet.exists() {
        notes.push(format!(
            "{ds_rel}: profile '{profile}' predicate facet {} is not present; its \
             `predicates` tag is unverified",
            rel_display(facet)
        ));
        return;
    }
    let census = match crate::pipeline::commands::analyze_predicate_forms::facet_form_classes(facet) {
        Ok(c) => c,
        Err(e) => {
            failures.push(format!("{ds_rel}: profile '{profile}': {e}"));
            return;
        }
    };
    match (class, uniform_parts, census.forms, census.parts) {
        ("mixed", _, 1, _) => failures.push(format!(
            "{ds_rel}: profile '{profile}' is tagged predicates=mixed but its facet holds \
             exactly one form; a set uniform in fact must say so (PS-23)"
        )),
        ("mixed", _, 0, _) => notes.push(format!(
            "{ds_rel}: profile '{profile}' predicate facet is empty; `predicates` unverified"
        )),
        (_, Some(n), forms, parts) if forms != 1 || parts != Some(n) => failures.push(format!(
            "{ds_rel}: profile '{profile}' is tagged predicates=uniform-{n} but its facet holds \
             {forms} form(s){} (PS-23)",
            parts
                .map(|p| format!(" of {p} part(s)"))
                .unwrap_or_default()
        )),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(dir: &Path, yaml: &str) -> PathBuf {
        let p = dir.join("dataset.yaml");
        std::fs::write(&p, yaml).unwrap();
        p
    }

    /// **An unversioned dataset.yaml is refused** (V-25), and one below
    /// 3 is reported as predating tag schemas without failing.
    #[test]
    fn a_dataset_must_state_its_version() {
        let tmp = tempfile::tempdir().unwrap();
        let p = dataset(tmp.path(), "name: t\nprofiles:\n  default:\n    base_vectors: b.fvec\n");
        let r = check(std::slice::from_ref(&p));
        assert!(!r.passed);
        assert!(r.messages.iter().any(|m| m.contains("no format_version")), "{:?}", r.messages);

        dataset(tmp.path(), "format_version: 1\nname: t\nprofiles:\n  default:\n    base_vectors: b.fvec\n");
        let r = check(&[p]);
        assert!(r.passed, "{:?}", r.messages);
        assert!(r.messages.iter().any(|m| m.contains("predates tag schemas")), "{:?}", r.messages);
    }

    /// **What makes a profile unaddressable is reported** (PS-15,
    /// PS-23): a shadowing key, a map value, a family member lacking a
    /// key, two alike profiles, and a predicate facet without its class.
    #[test]
    fn unaddressable_profiles_are_reported() {
        let tmp = tempfile::tempdir().unwrap();
        let p = dataset(
            tmp.path(),
            "format_version: 3\nname: t\nprofile_tags:\n  size: ~\nprofiles:\n  default:\n    \
             base_vectors: b.fvec\n    metadata_predicates: p.slab\n    attributes:\n      size: 1k\n      \
             maxk: 10\n      nested:\n        a: 1\n  a:\n    inherits: default\n    base_count: 100\n    attributes:\n      \
             size: 100\n  b:\n    inherits: default\n    base_count: 100\n    attributes:\n      size: 100\n",
        );
        let r = check(&[p]);
        assert!(!r.passed);
        let all = r.messages.join("\n");
        assert!(all.contains("'maxk' shadows a structural key"), "{all}");
        assert!(all.contains("'nested' is a map"), "{all}");
        assert!(all.contains("without the `predicates` tag"), "{all}");
        assert!(all.contains("'a' and 'b' carry identical attributes"), "{all}");
    }

    /// **A parent whose declaration a direct child overrides is a
    /// decoy** (PL-4); an override two levels down is not, and a size
    /// step's own ground truth never is.
    #[test]
    fn a_direct_override_of_a_parents_facet_is_reported() {
        let tmp = tempfile::tempdir().unwrap();
        let p = dataset(
            tmp.path(),
            "format_version: 3\nname: t\nprofiles:\n  default:\n    base_vectors: b.fvec\n    \
             neighbor_indices: profiles/default/g.ivec\n    metadata_predicates: profiles/base/p.slab\n  \
             10m:\n    inherits: default\n    base_count: 10\n    neighbor_indices: profiles/10m/g.ivec\n    \
             metadata_results: profiles/10m/r.slab\n  10m-set:\n    inherits: 10m\n    base_count: 10\n    \
             metadata_results: profiles/10m-set/r.slab\n    metadata_predicates: profiles/10m-set/p.slab\n",
        );
        let r = check(&[p]);
        let all = r.messages.join("\n");
        assert!(all.contains("profile '10m' declares metadata_results that its direct child '10m-set' overrides"), "{all}");
        assert!(!all.contains("'default' declares neighbor_indices that"), "ground truth does not cross a size step: {all}");
        assert!(!all.contains("'default' declares metadata_predicates that"), "two levels down is not a decoy: {all}");
        assert!(!all.contains("'10m' declares metadata_predicates without"), "a layer inheriting the slab needs no class: {all}");
    }

    /// **Below 3 an implicit parent is an advisory, not a failure**
    /// (PL-6, case 15).
    #[test]
    fn implicit_parents_are_advisory_below_three() {
        let tmp = tempfile::tempdir().unwrap();
        let p = dataset(tmp.path(), "format_version: 2\nname: t\nprofiles:\n  default:\n    base_vectors: b.fvec\n  10m:\n    base_count: 10\n");
        let r = check(&[p]);
        assert!(r.passed, "{:?}", r.messages);
        assert!(r.messages.iter().any(|m| m.contains("`10m` names no parent") && m.contains("refused from format_version 3")), "{:?}", r.messages);
    }

    /// **A version-3 profile lacking a defaulted schema tag is refused**
    /// (PS-19, case 17); a naming tag may be absent.
    #[test]
    fn a_missing_defaulted_schema_tag_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let p = dataset(
            tmp.path(),
            "format_version: 3\nname: t\nprofile_tags:\n  size: ~\n  family: stratified\nprofiles:\n  default:\n    \
             base_vectors: b.fvec\n    attributes:\n      family: stratified\n  10m:\n    inherits: default\n    \
             base_count: 10\n    attributes:\n      size: 10m\n",
        );
        let r = check(&[p]);
        assert!(!r.passed);
        assert!(r.messages.iter().any(|m| m.contains("profile '10m' lacks the schema tag(s) family")), "{:?}", r.messages);
    }

    fn write_slab(path: &Path, records: usize) {
        let config = slabtastic::WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut w = slabtastic::SlabWriter::new(path, config).unwrap();
        for i in 0..records {
            w.add_record(&(i as u32).to_le_bytes()).unwrap();
        }
        w.finish().unwrap();
    }

    fn write_ivec(path: &Path, rows: usize, dim: usize) {
        let mut bytes = Vec::new();
        for r in 0..rows {
            bytes.extend_from_slice(&(dim as i32).to_le_bytes());
            for d in 0..dim {
                bytes.extend_from_slice(&((r * dim + d) as i32).to_le_bytes());
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn write_fvec(path: &Path, rows: usize, dim: usize) {
        let mut bytes = Vec::new();
        for _ in 0..rows {
            bytes.extend_from_slice(&(dim as i32).to_le_bytes());
            for d in 0..dim {
                bytes.extend_from_slice(&(d as f32).to_le_bytes());
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// **The predicate group holds together by content** (PL-3, case
    /// 5): a results index with the wrong predicate count and a filtered
    /// ground truth sized to other queries are reported naming the
    /// profile and the facets.
    #[test]
    fn a_predicate_group_that_disagrees_is_reported() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        std::fs::create_dir_all(d.join("profiles/base")).unwrap();
        std::fs::create_dir_all(d.join("profiles/default")).unwrap();
        write_fvec(&d.join("profiles/base/q.fvec"), 3, 2);
        write_slab(&d.join("profiles/base/p.slab"), 2);
        write_slab(&d.join("profiles/default/r.slab"), 3);
        write_ivec(&d.join("profiles/default/f.ivec"), 2, 2);
        let p = dataset(
            d,
            "format_version: 2\nname: t\nprofiles:\n  default:\n    base_vectors: profiles/base/b.fvec\n    \
             query_vectors: profiles/base/q.fvec\n    metadata_predicates: profiles/base/p.slab\n    \
             metadata_results: profiles/default/r.slab\n    prefiltered_neighbor_indices: profiles/default/f.ivec\n    \
             attributes:\n      predicates: mixed\n",
        );
        let r = check(&[p]);
        assert!(!r.passed);
        let all = r.messages.join("\n");
        assert!(all.contains("results index profiles/default/r.slab holds 3 rows but its predicate slab profiles/base/p.slab holds 2 predicates"), "{all}");
        assert!(all.contains("prefiltered_neighbor_indices profiles/default/f.ivec holds 2 rows for 3 queries"), "{all}");
    }

    /// A tagged dataset whose profiles are all distinct passes.
    #[test]
    fn distinct_tagged_profiles_pass() {
        let tmp = tempfile::tempdir().unwrap();
        let p = dataset(
            tmp.path(),
            "format_version: 3\nname: t\nprofile_tags:\n  size: ~\nprofiles:\n  default:\n    \
             base_vectors: b.fvec\n    attributes:\n      size: 1k\n  100:\n    inherits: default\n    base_count: 100\n    \
             attributes:\n      size: 100\n",
        );
        let r = check(&[p]);
        assert!(r.passed, "{:?}", r.messages);
    }
}
