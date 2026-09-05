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
        if let Some(view) = profile.views.get("metadata_predicates") {
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
             maxk: 10\n      nested:\n        a: 1\n  a:\n    base_count: 100\n    attributes:\n      \
             size: 100\n  b:\n    base_count: 100\n    attributes:\n      size: 100\n",
        );
        let r = check(&[p]);
        assert!(!r.passed);
        let all = r.messages.join("\n");
        assert!(all.contains("'maxk' shadows a structural key"), "{all}");
        assert!(all.contains("'nested' is a map"), "{all}");
        assert!(all.contains("without the `predicates` tag"), "{all}");
        assert!(all.contains("'a' and 'b' carry identical attributes"), "{all}");
    }

    /// A tagged dataset whose profiles are all distinct passes.
    #[test]
    fn distinct_tagged_profiles_pass() {
        let tmp = tempfile::tempdir().unwrap();
        let p = dataset(
            tmp.path(),
            "format_version: 3\nname: t\nprofile_tags:\n  size: ~\nprofiles:\n  default:\n    \
             base_vectors: b.fvec\n    attributes:\n      size: 1k\n  100:\n    base_count: 100\n    \
             attributes:\n      size: 100\n",
        );
        let r = check(&[p]);
        assert!(r.passed, "{:?}", r.messages);
    }
}
