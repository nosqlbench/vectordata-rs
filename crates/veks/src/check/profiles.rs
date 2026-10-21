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
    check_with(dataset_files, &mut super::progress::StderrSink::default())
}

/// [`check`], reporting the scans it runs to `sink`.
pub fn check_with(dataset_files: &[PathBuf], sink: &mut dyn super::progress::ScanSink) -> CheckResult {
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
        check_quantization(&config, ds_dir, &ds_rel, &mut failures);
        check_ground_truth(&config, ds_dir, &ds_rel, &mut failures, sink);
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
        for facet in ["prefiltered_neighbor_indices", "postfiltered_neighbor_indices"] {
            let Some(view) = group.effective_view(name, facet) else { continue };
            let rel = relpath(view);
            let path = ds_dir.join(&rel);
            if !path.exists() || rel.contains("NNNN") {
                continue;
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

/// EA-33: a profile's facets belong together — vectordata's
/// verification (`TestDataViewExt::verify_report`, V1–V7): a ground
/// truth has one row per query, its distances one row per row of
/// indices and its width, every index names a base vector or, in a
/// filtered truth, is the `-1` sentinel of a short row; the queries
/// have the base's dimension; a declared base count is the base's.
/// Facets not written yet are skipped, as a dataset being built has
/// them. The index scan reports progress.
fn check_ground_truth(
    config: &vectordata::dataset::DatasetConfig,
    ds_dir: &Path,
    ds_rel: &str,
    failures: &mut Vec<String>,
    sink: &mut dyn super::progress::ScanSink,
) {
    use vectordata::TestDataViewExt;
    let group = match vectordata::TestDataGroup::load(&ds_dir.to_string_lossy()) {
        Ok(g) => g,
        Err(e) => {
            failures.push(format!("{ds_rel}: cannot open the dataset to verify its facets: {e} (EA-33)"));
            return;
        }
    };
    let options = vectordata::verify::VerifyOptions { skip_absent: true, ..Default::default() };
    for name in config.profiles.profiles.keys() {
        if config.profiles.layered && config.profiles.is_layer(name) {
            continue;
        }
        let Some(view) = group.profile(name) else { continue };
        let mut progress = |label: &str, done: u64, total: u64| {
            sink.report(&format!("{ds_rel}: {label} of profile '{name}'"), done, total)
        };
        for mut failure in view.verify_report(&options, &mut progress).failures {
            // Named as the dataset names it: relative to its directory.
            failure.file = failure.file.map(|f| match Path::new(&f).strip_prefix(ds_dir) {
                Ok(rel) => rel.to_string_lossy().into_owned(),
                Err(_) => f,
            });
            failures.push(format!("{ds_rel}: profile '{name}' {failure} (EA-33)"));
        }
    }
}

/// EA-26: a quantized base travels with its three parameter columns
/// and the scheme that applies them, and every column holds one value
/// per base vector.
fn check_quantization(
    config: &vectordata::dataset::DatasetConfig,
    ds_dir: &Path,
    ds_rel: &str,
    failures: &mut Vec<String>,
) {
    use vectordata::dataset::{Quantization, StandardFacet};
    let group = &config.profiles;
    let relpath = |v: &vectordata::dataset::DSView| -> String {
        vectordata::dataset::catalog::strip_window_suffix(&v.source.path)
            .trim_start_matches("./")
            .to_string()
    };
    let count_of = |rel: &str| -> Option<Result<usize, String>> {
        let path = ds_dir.join(rel);
        if !path.exists() || rel.contains("NNNN") {
            return None;
        }
        Some(
            vectordata::io::open_vec_native(&path.to_string_lossy())
                .map(|r| r.count())
                .map_err(|e| e.to_string()),
        )
    };
    for name in group.profiles.keys() {
        if group.layered && group.is_layer(name) {
            continue;
        }
        let present: Vec<&str> = StandardFacet::QUANT
            .iter()
            .map(|f| f.key())
            .filter(|key| group.effective_view(name, key).is_some())
            .collect();
        let attribute = group
            .profiles
            .get(name)
            .and_then(|p| p.attributes.get("quantization"))
            .or_else(|| config.attributes.as_ref().and_then(|a| a.other.get("quantization")));
        if present.is_empty() && attribute.is_none() {
            continue;
        }
        if !present.is_empty() && present.len() != StandardFacet::QUANT.len() {
            failures.push(format!(
                "{ds_rel}: profile '{name}' declares {} of the three quantization columns ({}); \
                 quant_scale, quant_zero_point and quant_sum travel together (EA-26)",
                present.len(),
                present.join(", ")
            ));
        }
        let scheme = match attribute {
            None => {
                failures.push(format!(
                    "{ds_rel}: profile '{name}' holds quantization columns but no `quantization` \
                     attribute states the scheme (EA-26)"
                ));
                None
            }
            Some(value) => match Quantization::from_value(value) {
                Ok(q) => Some(q),
                Err(e) => {
                    failures.push(format!(
                        "{ds_rel}: profile '{name}': the quantization attribute does not parse: {e} (EA-26)"
                    ));
                    None
                }
            },
        };
        if present.is_empty() {
            failures.push(format!(
                "{ds_rel}: profile '{name}' states a quantization scheme but declares no \
                 quantization columns (EA-26)"
            ));
            continue;
        }
        let Some(base) = group.effective_view(name, "base_vectors") else {
            failures.push(format!(
                "{ds_rel}: profile '{name}' holds quantization columns but no base_vectors to \
                 describe (EA-26)"
            ));
            continue;
        };
        let base_rel = relpath(base);
        if let Some(q) = &scheme
            && let Some(ty) = vectordata::ElementType::from_extension(base_rel.rsplit('.').next().unwrap_or(""))
            && ty != q.codes
        {
            failures.push(format!(
                "{ds_rel}: profile '{name}' quantization codes are {} but base_vectors {base_rel} \
                 holds {ty} (EA-26)",
                q.codes
            ));
        }
        let Some(Ok(base_count)) = count_of(&base_rel) else { continue };
        for key in present {
            let rel = relpath(group.effective_view(name, key).expect("present"));
            match count_of(&rel) {
                Some(Ok(n)) if n != base_count => failures.push(format!(
                    "{ds_rel}: profile '{name}' {key} {rel} holds {n} values for {base_count} base \
                     vectors (EA-26)"
                )),
                Some(Err(e)) => failures.push(format!("{ds_rel}: profile '{name}' {key} {rel}: {e} (EA-26)")),
                _ => {}
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

    /// A u8 base of `n` vectors of dimension 2 and three columns of
    /// `scale`, `zp` and `sum` values beside it.
    fn quantized_files(dir: &Path, n: usize, scale: usize, zp: usize, sum: usize) {
        let mut base = Vec::new();
        for i in 0..n {
            base.extend_from_slice(&2i32.to_le_bytes());
            base.extend_from_slice(&[i as u8, (i * 2) as u8]);
        }
        std::fs::write(dir.join("base_vectors.bvecs"), base).unwrap();
        let f32s = |k: usize| (0..k).flat_map(|i| (i as f32 * 0.5).to_le_bytes()).collect::<Vec<u8>>();
        let i32s = |k: usize| (0..k).flat_map(|i| (i as i32).to_le_bytes()).collect::<Vec<u8>>();
        std::fs::write(dir.join("quant_scale.f32"), f32s(scale)).unwrap();
        std::fs::write(dir.join("quant_zero_point.i32"), i32s(zp)).unwrap();
        std::fs::write(dir.join("quant_sum.i32"), i32s(sum)).unwrap();
    }

    /// A base of `base` f32 vectors, `queries` queries, a ground truth of
    /// `gt_rows` rows of `k` i32 indices drawn from `ids(row, col)`, and
    /// `dist_rows` rows of distances, under the facet keys `idx` and
    /// `dist`.
    #[allow(clippy::too_many_arguments)]
    fn ground_truth_files(
        dir: &Path, base: usize, queries: usize, gt_rows: usize, k: usize, dist_rows: usize,
        idx: &str, dist: &str, ids: impl Fn(usize, usize) -> i32,
    ) -> String {
        let fvec = |n: usize| -> Vec<u8> {
            (0..n).flat_map(|i| [2i32.to_le_bytes(), (i as f32).to_le_bytes(), (i as f32 * 0.5).to_le_bytes()].concat()).collect()
        };
        std::fs::write(dir.join("base_vectors.fvecs"), fvec(base)).unwrap();
        std::fs::write(dir.join("query_vectors.fvecs"), fvec(queries)).unwrap();
        let mut gt = Vec::new();
        for r in 0..gt_rows {
            gt.extend_from_slice(&(k as i32).to_le_bytes());
            for c in 0..k { gt.extend_from_slice(&ids(r, c).to_le_bytes()); }
        }
        std::fs::write(dir.join(format!("{idx}.ivecs")), gt).unwrap();
        let mut d = Vec::new();
        for r in 0..dist_rows {
            d.extend_from_slice(&(k as i32).to_le_bytes());
            for c in 0..k { d.extend_from_slice(&((r * k + c) as f32).to_le_bytes()); }
        }
        std::fs::write(dir.join(format!("{dist}.fvecs")), d).unwrap();
        format!(
            "format_version: 1\nname: gt\nprofiles:\n  default:\n    base_vectors: base_vectors.fvecs\n    \
             query_vectors: query_vectors.fvecs\n    {idx}: {idx}.ivecs\n    {dist}: {dist}.fvecs\n"
        )
    }

    /// **An index at the base count is reported with its ordinal**
    /// (EA-33): every ground-truth index names a base vector.
    #[test]
    fn an_index_at_the_base_count_is_reported_with_its_ordinal() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = ground_truth_files(tmp.path(), 10, 3, 3, 4, 3, "neighbor_indices", "neighbor_distances",
            |r, c| if (r, c) == (1, 2) { 10 } else { ((r + c) % 10) as i32 });
        let p = dataset(tmp.path(), &yaml);
        let r = check(&[p]);
        assert!(!r.passed, "{:?}", r.messages);
        assert!(
            r.messages.iter().any(|m| m.contains("neighbor_indices") && m.contains("row 1") && m.contains("10") && m.contains("base vectors")),
            "{:?}", r.messages
        );
    }

    /// **Queries of another dimension than the base are reported**
    /// (V2): the check is vectordata's verification, which proves the
    /// facets belong together, not only the ground truth's shape.
    #[test]
    fn queries_of_another_dimension_than_the_base_are_reported() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = ground_truth_files(tmp.path(), 10, 3, 3, 4, 3, "neighbor_indices", "neighbor_distances",
            |r, c| ((r + c) % 10) as i32);
        let queries: Vec<u8> = (0..3).flat_map(|i| [3i32.to_le_bytes(), (i as f32).to_le_bytes(), 0f32.to_le_bytes(), 1f32.to_le_bytes()].concat()).collect();
        std::fs::write(tmp.path().join("query_vectors.fvecs"), queries).unwrap();
        let p = dataset(tmp.path(), &yaml);
        let r = check(&[p]);
        assert!(!r.passed, "{:?}", r.messages);
        assert!(
            r.messages.iter().any(|m| m.contains("query_vectors") && m.contains("has dimension 3; the base has 2")),
            "{:?}", r.messages
        );
    }

    /// **A ground truth with more rows than queries is reported**, and
    /// so are distances whose rows differ from the indices'.
    #[test]
    fn ground_truth_rows_must_match_the_queries_and_the_distances() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = ground_truth_files(tmp.path(), 10, 3, 4, 2, 4, "neighbor_indices", "neighbor_distances", |r, c| ((r + c) % 10) as i32);
        let p = dataset(tmp.path(), &yaml);
        let r = check(std::slice::from_ref(&p));
        assert!(!r.passed, "{:?}", r.messages);
        assert!(r.messages.iter().any(|m| m.contains("neighbor_indices") && m.contains("4 rows for 3 queries")), "{:?}", r.messages);

        let yaml = ground_truth_files(tmp.path(), 10, 3, 3, 2, 2, "neighbor_indices", "neighbor_distances", |r, c| ((r + c) % 10) as i32);
        dataset(tmp.path(), &yaml);
        let r = check(&[p]);
        assert!(!r.passed, "{:?}", r.messages);
        assert!(r.messages.iter().any(|m| m.contains("neighbor_distances") && m.contains("2 rows") && m.contains("3 rows")), "{:?}", r.messages);
    }

    /// **A filtered truth allows the `-1` sentinel of a short row** —
    /// a pre-filtered row when fewer than K base vectors pass the
    /// predicate, a post-filtered row whenever the intersection is
    /// short — **and the plain truth does not**: it ranks every base
    /// vector.
    #[test]
    fn a_filtered_truth_allows_the_minus_one_sentinel_and_a_plain_one_does_not() {
        let tmp = tempfile::tempdir().unwrap();
        let sparse = |r: usize, c: usize| if c == 1 { -1 } else { ((r + c) % 10) as i32 };
        let p = tmp.path().join("dataset.yaml");
        for (idx, dist) in [
            ("postfiltered_neighbor_indices", "postfiltered_neighbor_distances"),
            ("prefiltered_neighbor_indices", "prefiltered_neighbor_distances"),
        ] {
            let yaml = ground_truth_files(tmp.path(), 10, 3, 3, 2, 3, idx, dist, sparse);
            dataset(tmp.path(), &yaml);
            let r = check(std::slice::from_ref(&p));
            assert!(r.passed, "{idx}: {:?}", r.messages);
        }

        let yaml = ground_truth_files(tmp.path(), 10, 3, 3, 2, 3, "neighbor_indices", "neighbor_distances", sparse);
        dataset(tmp.path(), &yaml);
        let r = check(&[p]);
        assert!(!r.passed, "{:?}", r.messages);
        assert!(r.messages.iter().any(|m| m.contains("neighbor_indices") && m.contains("-1") && m.contains("sentinel")), "{:?}", r.messages);
    }

    /// **The scan reports progress**: a capturing sink hears the start,
    /// every block and the end of each index facet's scan, labelled.
    #[test]
    fn the_scan_reports_progress() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = ground_truth_files(tmp.path(), 10, 3, 3, 4, 3, "neighbor_indices", "neighbor_distances", |r, c| ((r * 3 + c) % 10) as i32);
        let p = dataset(tmp.path(), &yaml);
        let mut lines: Vec<String> = Vec::new();
        let r = check_with(std::slice::from_ref(&p), &mut lines);
        assert!(r.passed, "{:?}", r.messages);
        let scan: Vec<&String> = lines.iter().filter(|l| l.contains("scanning neighbor_indices of profile 'default'")).collect();
        assert!(scan.len() >= 2, "{lines:?}");
        assert!(scan.first().unwrap().ends_with(": 0/3"), "{lines:?}");
        assert!(scan.last().unwrap().ends_with(": 3/3"), "{lines:?}");
    }

    /// A uniform vector file of `rows` records of `dim` elements, each
    /// element from `value(row, column)`, named `<stem>.<T's extension>`.
    fn write_vectors<T: vectordata::typed_access::Element>(
        dir: &Path, stem: &str, rows: usize, dim: usize, value: impl Fn(usize, usize) -> T,
    ) -> String {
        use vectordata::typed_access::Container;
        let name = format!("{stem}.{}", T::TYPE.extension(Container::Vec));
        let mut bytes = Vec::new();
        for r in 0..rows {
            bytes.extend_from_slice(&(dim as i32).to_le_bytes());
            for c in 0..dim {
                let at = bytes.len();
                bytes.resize(at + T::SIZE, 0);
                value(r, c).write_le_bytes(&mut bytes[at..]);
            }
        }
        std::fs::write(dir.join(&name), bytes).unwrap();
        name
    }

    /// A base of ten f32 vectors, three queries, a ground truth of three
    /// rows of four indices of type `I` from `ids(row, column)`, and the
    /// distances as `D`; the dataset.yaml naming all four.
    fn typed_ground_truth<I: vectordata::typed_access::Element, D: vectordata::typed_access::Element>(
        dir: &Path, ids: impl Fn(usize, usize) -> I, dist: impl Fn(usize, usize) -> D,
    ) -> PathBuf {
        write_vectors::<f32>(dir, "base_vectors", 10, 2, |r, c| r as f32 + c as f32 * 0.5);
        write_vectors::<f32>(dir, "query_vectors", 3, 2, |r, c| r as f32 + c as f32 * 0.5);
        let idx = write_vectors::<I>(dir, "neighbor_indices", 3, 4, ids);
        let dst = write_vectors::<D>(dir, "neighbor_distances", 3, 4, dist);
        dataset(
            dir,
            &format!(
                "format_version: 1\nname: gt\nprofiles:\n  default:\n    base_vectors: base_vectors.f32vecs\n    \
                 query_vectors: query_vectors.f32vecs\n    neighbor_indices: {idx}\n    neighbor_distances: {dst}\n"
            ),
        )
    }

    /// **A ground truth passes at every integer index type the check
    /// reads as i64** — every one `conversion_to(I64)` admits, which is
    /// all eight — when its indices are in range, and at every float
    /// distance type.
    #[test]
    fn a_ground_truth_passes_at_every_integer_index_type_and_every_float_distance_type() {
        use vectordata::typed_access::{Element, ElementConversion, ElementType};
        let index_types: Vec<ElementType> = ElementType::ALL
            .iter()
            .copied()
            .filter(|t| t.is_integer())
            .collect();
        for &ty in &index_types {
            assert_ne!(ty.conversion_to(ElementType::I64), ElementConversion::Refused, "{ty}");
        }
        for &ty in &index_types {
            let tmp = tempfile::tempdir().unwrap();
            let p = veks_datatypes::dispatch_element!(ty, I => typed_ground_truth::<I, f32>(
                tmp.path(),
                |r, c| I::from_value(vectordata::typed_access::Value::Int(((r * 3 + c) % 10) as i128)).unwrap(),
                |r, c| (r * 4 + c) as f32,
            ));
            let r = check(&[p]);
            assert!(r.passed, "{ty} indices: {:?}", r.messages);
        }
        for &ty in ElementType::ALL.iter().filter(|t| t.is_float()) {
            let tmp = tempfile::tempdir().unwrap();
            let p = veks_datatypes::dispatch_element!(ty, D => typed_ground_truth::<i32, D>(
                tmp.path(),
                |r, c| ((r * 3 + c) % 10) as i32,
                |r, c| D::from_value(vectordata::typed_access::Value::Int((r * 4 + c) as i128)).unwrap(),
            ));
            let r = check(&[p]);
            assert!(r.passed, "{ty} distances: {:?}", r.messages);
        }
    }

    /// **A u64 index above `i64::MAX` is reported** (EA-33): the check
    /// reads every integer index type as i64, and a value the
    /// cross-sign conversion refuses is a failure naming the facet, the
    /// value and the type it would not fit — never a silent end of the
    /// scan that leaves the dataset passing.
    #[test]
    fn a_u64_index_above_i64_max_is_reported_as_not_fitting_i64() {
        let tmp = tempfile::tempdir().unwrap();
        let p = typed_ground_truth::<u64, f32>(
            tmp.path(),
            |r, c| if (r, c) == (2, 1) { i64::MAX as u64 + 1 } else { ((r * 3 + c) % 10) as u64 },
            |r, c| (r * 4 + c) as f32,
        );
        let r = check(&[p]);
        assert!(!r.passed, "{:?}", r.messages);
        assert!(
            r.messages.iter().any(|m| {
                m.contains("neighbor_indices")
                    && m.contains(&(i64::MAX as u64 + 1).to_string())
                    && m.contains("does not fit in i64")
            }),
            "{:?}",
            r.messages
        );
    }

    /// **A consistent ground truth passes.**
    #[test]
    fn a_consistent_ground_truth_passes() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = ground_truth_files(tmp.path(), 10, 3, 3, 4, 3, "neighbor_indices", "neighbor_distances", |r, c| ((r * 3 + c) % 10) as i32);
        let p = dataset(tmp.path(), &yaml);
        let r = check(&[p]);
        assert!(r.passed, "{:?}", r.messages);
    }

    const QUANT_ATTR: &str = "attributes:\n  quantization:\n    scheme: affine\n    granularity: per-vector\n    codes: u8\n    range: [0, 255]\n";
    const QUANT_VIEWS: &str = "    base_vectors: base_vectors.bvecs\n    quant_scale: quant_scale.f32\n    quant_zero_point: quant_zero_point.i32\n    quant_sum: quant_sum.i32\n";

    /// **A consistent quantized dataset passes** (EA-26): three
    /// columns, one value per base vector, and the scheme beside them.
    #[test]
    fn a_consistent_quantized_dataset_passes() {
        let tmp = tempfile::tempdir().unwrap();
        quantized_files(tmp.path(), 5, 5, 5, 5);
        let p = dataset(tmp.path(), &format!("format_version: 1\nname: q\n{QUANT_ATTR}profiles:\n  default:\n{QUANT_VIEWS}"));
        let r = check(&[p]);
        assert!(r.passed, "{:?}", r.messages);
    }

    /// **A signed-code corpus checks as its own type** (EA-26): an i8
    /// base with `codes: i8` passes, and `codes: u8` against the same
    /// base is reported as naming a type the base does not hold.
    #[test]
    fn an_i8_coded_dataset_passes_at_i8_and_is_reported_at_u8() {
        let tmp = tempfile::tempdir().unwrap();
        quantized_files(tmp.path(), 4, 4, 4, 4);
        let mut base = Vec::new();
        for i in 0..4i8 {
            base.extend_from_slice(&2i32.to_le_bytes());
            base.extend_from_slice(&[(-i) as u8, (i * 30) as u8]);
        }
        std::fs::write(tmp.path().join("base_vectors.i8vecs"), base).unwrap();
        let views = QUANT_VIEWS.replace("base_vectors.bvecs", "base_vectors.i8vecs");
        let signed = QUANT_ATTR.replace("codes: u8", "codes: i8").replace("range: [0, 255]", "range: [-127, 127]");
        let p = dataset(tmp.path(), &format!("format_version: 1\nname: q\n{signed}profiles:\n  default:\n{views}"));
        let r = check(std::slice::from_ref(&p));
        assert!(r.passed, "{:?}", r.messages);

        dataset(tmp.path(), &format!("format_version: 1\nname: q\n{QUANT_ATTR}profiles:\n  default:\n{views}"));
        let r = check(&[p]);
        assert!(!r.passed, "u8 codes over an i8 base: {:?}", r.messages);
        assert!(r.messages.iter().any(|m| m.contains("u8") && m.contains("i8")), "{:?}", r.messages);
    }

    /// **Columns without the attribute are reported**, as is an
    /// incomplete set of columns.
    #[test]
    fn columns_without_the_attribute_are_reported() {
        let tmp = tempfile::tempdir().unwrap();
        quantized_files(tmp.path(), 5, 5, 5, 5);
        let p = dataset(tmp.path(), &format!("format_version: 1\nname: q\nprofiles:\n  default:\n{QUANT_VIEWS}"));
        let r = check(std::slice::from_ref(&p));
        assert!(!r.passed);
        assert!(r.messages.iter().any(|m| m.contains("no `quantization` attribute")), "{:?}", r.messages);

        dataset(
            tmp.path(),
            &format!("format_version: 1\nname: q\n{QUANT_ATTR}profiles:\n  default:\n    base_vectors: base_vectors.bvecs\n    quant_scale: quant_scale.f32\n"),
        );
        let r = check(&[p]);
        assert!(!r.passed);
        assert!(r.messages.iter().any(|m| m.contains("1 of the three quantization columns")), "{:?}", r.messages);
    }

    /// **The attribute without columns is reported**, and so is a
    /// scheme whose code type is not the base's.
    #[test]
    fn the_attribute_without_columns_is_reported() {
        let tmp = tempfile::tempdir().unwrap();
        quantized_files(tmp.path(), 5, 5, 5, 5);
        let p = dataset(tmp.path(), &format!("format_version: 1\nname: q\n{QUANT_ATTR}profiles:\n  default:\n    base_vectors: base_vectors.bvecs\n"));
        let r = check(std::slice::from_ref(&p));
        assert!(!r.passed);
        assert!(r.messages.iter().any(|m| m.contains("declares no quantization columns")), "{:?}", r.messages);

        let wrong = QUANT_ATTR.replace("codes: u8", "codes: i8");
        dataset(tmp.path(), &format!("format_version: 1\nname: q\n{wrong}profiles:\n  default:\n{QUANT_VIEWS}"));
        let r = check(&[p]);
        assert!(!r.passed);
        assert!(r.messages.iter().any(|m| m.contains("codes are i8 but base_vectors")), "{:?}", r.messages);
    }

    /// **An i8-coded base passes with `codes: i8`, and `codes: u8`
    /// against it is refused** (EA-26): the scheme's code type is the
    /// base file's element type, read from its extension.
    #[test]
    fn an_i8_coded_base_passes_with_i8_codes_and_is_refused_with_u8_codes() {
        let tmp = tempfile::tempdir().unwrap();
        write_vectors::<i8>(tmp.path(), "base_vectors", 5, 2, |r, c| (r as i8 - 2) * (c as i8 + 1));
        let f32s = |k: usize| (0..k).flat_map(|i| (i as f32 * 0.5).to_le_bytes()).collect::<Vec<u8>>();
        let i32s = |k: usize| (0..k).flat_map(|i| (i as i32).to_le_bytes()).collect::<Vec<u8>>();
        std::fs::write(tmp.path().join("quant_scale.f32"), f32s(5)).unwrap();
        std::fs::write(tmp.path().join("quant_zero_point.i32"), i32s(5)).unwrap();
        std::fs::write(tmp.path().join("quant_sum.i32"), i32s(5)).unwrap();
        let views = "    base_vectors: base_vectors.i8vecs\n    quant_scale: quant_scale.f32\n    \
                     quant_zero_point: quant_zero_point.i32\n    quant_sum: quant_sum.i32\n";
        let attr = |codes: &str| {
            format!("attributes:\n  quantization:\n    scheme: affine\n    granularity: per-vector\n    codes: {codes}\n    range: [-128, 127]\n")
        };

        let p = dataset(tmp.path(), &format!("format_version: 1\nname: q\n{}profiles:\n  default:\n{views}", attr("i8")));
        let r = check(std::slice::from_ref(&p));
        assert!(r.passed, "{:?}", r.messages);

        dataset(tmp.path(), &format!("format_version: 1\nname: q\n{}profiles:\n  default:\n{views}", attr("u8")));
        let r = check(&[p]);
        assert!(!r.passed, "{:?}", r.messages);
        assert!(
            r.messages.iter().any(|m| m.contains("codes are u8 but base_vectors base_vectors.i8vecs holds i8")),
            "{:?}",
            r.messages
        );
    }

    /// **A column whose count disagrees with the base is reported**,
    /// naming the column, its count and the base count.
    #[test]
    fn a_column_count_that_disagrees_with_the_base_is_reported() {
        let tmp = tempfile::tempdir().unwrap();
        quantized_files(tmp.path(), 5, 5, 4, 5);
        let p = dataset(tmp.path(), &format!("format_version: 1\nname: q\n{QUANT_ATTR}profiles:\n  default:\n{QUANT_VIEWS}"));
        let r = check(&[p]);
        assert!(!r.passed);
        assert!(
            r.messages.iter().any(|m| m.contains("quant_zero_point quant_zero_point.i32 holds 4 values for 5 base vectors")),
            "{:?}",
            r.messages
        );
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
