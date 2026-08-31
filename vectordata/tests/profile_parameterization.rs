// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Profiles as the parameterization axis.
//!
//! A dataset needs precomputed filtered ground truth at several
//! selectivities at once. Those facets co-vary: at a given selectivity
//! the predicates, the result index, and the filtered ground truth
//! derived from them only mean anything together. Pairing 1% predicates
//! with 10% results is not a degraded answer — it is a meaningless one
//! that produces numbers looking like measurements.
//!
//! A profile is already that: a named group of facets, selected whole,
//! sharing invariants with its parent by inheritance. What was missing
//! was somewhere to record the parameter value, an inheritance rule
//! that follows the axis a family varies along, and a way to ask what a
//! family's members are.
//!
//! The acceptance cases are numbered as in
//! `docs/design/srd-profile-parameterization.md` §10.

use vectordata::dataset::DatasetConfig;

/// The SRD's §8 declaration, with a `1m` size profile and two
/// selectivities parameterizing it.
fn selectivity_family() -> &'static str {
    "name: amazon-reviews-2023\n\
     profiles:\n\
    \x20 default:\n\
    \x20   base_vectors: base_vectors.fvec\n\
    \x20   query_vectors: query_vectors.fvec\n\
    \x20   metadata_content: metadata_content.ivecs\n\
    \x20 1m:\n\
    \x20   base_count: 1000000\n\
    \x20   neighbor_indices: profiles/1m/neighbor_indices.ivecs\n\
    \x20   neighbor_distances: profiles/1m/neighbor_distances.fvecs\n\
    \x20 1m-sel001:\n\
    \x20   inherits: 1m\n\
    \x20   attributes:\n\
    \x20     selectivity: 0.0012\n\
    \x20     predicate_count: 1000\n\
    \x20     k: 100\n\
    \x20   metadata_predicates: profiles/1m-sel001/metadata_predicates.ivecs\n\
    \x20   postfiltered_neighbor_indices: profiles/1m-sel001/postfiltered_neighbor_indices.ivecs\n\
    \x20 1m-sel010:\n\
    \x20   inherits: 1m\n\
    \x20   attributes:\n\
    \x20     selectivity: 0.0104\n\
    \x20     predicate_count: 1000\n\
    \x20     k: 100\n\
    \x20   metadata_predicates: profiles/1m-sel010/metadata_predicates.ivecs\n\
    \x20   postfiltered_neighbor_indices: profiles/1m-sel010/postfiltered_neighbor_indices.ivecs\n"
}

fn load(yaml: &str) -> DatasetConfig {
    serde_yaml::from_str(yaml).expect("loads")
}

// ── case 1, 7: the selectivity axis ────────────────────────────────

/// **Case 1** — shared facets are inherited, not duplicated. A member
/// declares only what varies.
#[test]
fn a_selectivity_member_inherits_the_shared_facets() {
    let cfg = load(selectivity_family());
    let member = cfg.profiles.profile("1m-sel001").expect("1m-sel001");

    assert_eq!(
        member.view("base_vectors").map(|v| v.source.path.as_str()),
        Some("base_vectors.fvec"),
        "the corpus comes from default without being restated"
    );
    assert_eq!(
        member.view("query_vectors").map(|v| v.source.path.as_str()),
        Some("query_vectors.fvec")
    );
}

/// **Case 7** — on a selectivity axis the unfiltered ground truth is
/// invariant, so it inherits from the sized parent rather than being
/// restated per member.
///
/// This is the change P-2 exists for. Under the old fixed rule
/// `neighbor_indices` never inherited, so every member of a selectivity
/// family had to repeat the same path — a duplication that drifts the
/// moment one is edited.
#[test]
fn a_selectivity_member_inherits_ground_truth_from_its_sized_parent() {
    let cfg = load(selectivity_family());
    for name in ["1m-sel001", "1m-sel010"] {
        let member = cfg.profiles.profile(name).expect(name);
        assert_eq!(
            member.view("neighbor_indices").map(|v| v.source.path.as_str()),
            Some("profiles/1m/neighbor_indices.ivecs"),
            "{name} must share its parent's ground truth"
        );
        assert_eq!(
            member.base_count,
            Some(1_000_000),
            "{name} is the same corpus as its parent"
        );
        // What it does declare is its own.
        assert!(
            member
                .view("metadata_predicates")
                .is_some_and(|v| v.source.path.contains(name)),
            "{name} keeps its own predicates"
        );
    }
}

/// **Case 8** — the size axis is unchanged: a sized profile still owns
/// its ground truth, because that ground truth is derived from
/// `base_count` and cannot be shared.
#[test]
fn a_size_profile_still_owns_its_ground_truth() {
    let cfg = load(
        "name: sizes\n\
         profiles:\n\
        \x20 default:\n\
        \x20   base_vectors: base.fvec\n\
        \x20   neighbor_indices: neighbor_indices.ivecs\n\
        \x20 1m:\n\
        \x20   base_count: 1000000\n",
    );
    let sized = cfg.profiles.profile("1m").expect("1m");
    assert!(
        sized.view("neighbor_indices").is_none(),
        "inheriting default's full-base ground truth would mis-route every verify"
    );
    assert!(
        sized.view("base_vectors").is_some(),
        "the shared corpus still inherits"
    );
}

// ── cases 2, 3, 4: attributes ──────────────────────────────────────

/// **Case 2** — `attributes:` round-trips through load and save.
#[test]
fn profile_attributes_round_trip_through_save() {
    let cfg = load(selectivity_family());
    let member = cfg.profiles.profile("1m-sel001").unwrap();
    assert_eq!(
        member.attributes.get("selectivity").and_then(|v| v.as_f64()),
        Some(0.0012)
    );
    assert_eq!(
        member.attributes.get("predicate_count").and_then(|v| v.as_u64()),
        Some(1000)
    );

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("dataset.yaml");
    let saved = cfg.to_expanded_yaml_string(&path).expect("renders");
    let round_tripped: DatasetConfig = serde_yaml::from_str(&saved)
        .unwrap_or_else(|e| panic!("saved output must reload: {e}\n{saved}"));
    let after = round_tripped.profiles.profile("1m-sel001").expect("survives");
    assert_eq!(
        after.attributes.get("selectivity").and_then(|v| v.as_f64()),
        Some(0.0012),
        "attributes must survive a save:\n{saved}"
    );
    assert_eq!(
        after.inherits.as_deref(),
        Some("1m"),
        "the parent must survive too, or the family dissolves:\n{saved}"
    );
}

/// **Case 3** — a profile with no attributes reports as undescribed,
/// never as zero-valued. Absent is not `selectivity: 0.0` (P-7).
#[test]
fn an_undescribed_profile_is_empty_not_zero() {
    let cfg = load(selectivity_family());
    let plain = cfg.profiles.profile("1m").expect("1m");
    assert!(
        plain.attributes.is_empty(),
        "a profile that recorded nothing has no attributes"
    );
    assert!(
        plain.attributes.get("selectivity").is_none(),
        "and specifically does not have a selectivity of zero"
    );
}

/// **Case 4** — recorded attributes are what a run realized, not what
/// it targeted (P-6). The fixture's sweep aimed at 0.1% and 1% and
/// achieved 0.12% and 1.04%; those are the numbers on record.
#[test]
fn attributes_carry_the_realized_value_not_the_target() {
    let cfg = load(selectivity_family());
    let a = cfg.profiles.profile("1m-sel001").unwrap();
    let b = cfg.profiles.profile("1m-sel010").unwrap();
    assert_eq!(a.attributes["selectivity"].as_f64(), Some(0.0012));
    assert_eq!(b.attributes["selectivity"].as_f64(), Some(0.0104));
    // The name says 001; the measurement says 0.0012. A consumer
    // choosing by value must get the measurement.
    assert_ne!(
        a.attributes["selectivity"].as_f64(),
        Some(0.001),
        "the target is not what is recorded"
    );
}

// ── case 5: families ───────────────────────────────────────────────

/// **Case 5** — a family's members and their parameter values are
/// enumerable without parsing names (P-8).
#[test]
fn family_members_and_values_are_enumerable() {
    let cfg = load(
        "name: sized\n\
         profiles:\n\
        \x20 default:\n\
        \x20   base_vectors: base.fvec\n\
        \x20 sized: [1m..3m/1m]\n",
    );
    let families: Vec<(&str, &[String])> = cfg.profiles.families().collect();
    assert!(!families.is_empty(), "a generated family must be exposed");

    let (spec, members) = families[0];
    assert!(spec.contains("1m"), "the spec is reported verbatim: {spec}");
    assert!(members.len() >= 2, "members: {members:?}");

    // Every member resolves, and its attributes come back with it —
    // empty here, which is undescribed rather than zero.
    let listed = cfg.profiles.family_attributes(&members[0]);
    assert_eq!(listed.len(), members.len());
    for (name, attrs) in &listed {
        assert!(members.contains(name));
        assert!(attrs.is_empty(), "{name} recorded nothing");
    }

    // And membership is answerable from any member.
    let (spec_again, _) = cfg
        .profiles
        .family_of(&members[1])
        .expect("a member knows its family");
    assert_eq!(spec_again, spec);
}

/// A hand-declared profile is a family of itself, which is a real
/// answer rather than a missing one.
#[test]
fn a_hand_declared_profile_has_no_generator_family() {
    let cfg = load(selectivity_family());
    assert!(cfg.profiles.family_of("1m-sel001").is_none());
    let listed = cfg.profiles.family_attributes("1m-sel001");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].0, "1m-sel001");
    assert_eq!(listed[0].1["selectivity"].as_f64(), Some(0.0012));
}

// ── case 11: validation ────────────────────────────────────────────

/// **Case 11** — a family member reading different base data is
/// reported (P-11). It is not a parameterization of the same corpus,
/// and comparing results across it reports one number about two
/// datasets.
#[test]
fn a_family_member_reading_a_different_corpus_is_reported() {
    let cfg = load(
        "name: drift\n\
         profiles:\n\
        \x20 default:\n\
        \x20   base_vectors: base.fvec\n\
        \x20 sized: [1m..2m/1m]\n",
    );
    let (_, members) = cfg.profiles.families().next().expect("a family");
    let members: Vec<String> = members.to_vec();
    assert!(members.len() >= 2);

    // Point the second member at a different file.
    let mut broken = cfg.clone();
    if let Some(p) = broken.profiles.profiles.get_mut(&members[1])
        && let Some(v) = p.views.get_mut("base_vectors")
    {
        v.source.path = "somewhere_else.fvec".to_string();
    }
    let violations = vectordata::dataset::conformance::validate_conformance(&broken)
        .expect_err("a member on different base data is not conformant");
    assert!(
        violations
            .iter()
            .any(|v| v.profile == members[1] && v.key == "base_vectors"),
        "the drifting member must be named: {violations:?}"
    );
}

/// An `inherits:` naming a profile the dataset does not declare is
/// reported rather than silently ignored. The load still succeeds —
/// the facets the profile does declare are readable, and taking a whole
/// dataset out of reach over one profile would be worse.
#[test]
fn an_unresolvable_parent_loads_but_is_reported() {
    let cfg = load(
        "name: orphan\n\
         profiles:\n\
        \x20 default:\n\
        \x20   base_vectors: base.fvec\n\
        \x20 child:\n\
        \x20   inherits: nonexistent\n\
        \x20   query_vectors: q.fvec\n",
    );
    let child = cfg.profiles.profile("child").expect("still loads");
    assert!(child.view("query_vectors").is_some(), "what it declared is readable");

    let violations = vectordata::dataset::conformance::validate_conformance(&cfg)
        .expect_err("an unresolvable parent is reported");
    assert!(
        violations
            .iter()
            .any(|v| v.profile == "child" && v.key == "inherits"),
        "{violations:?}"
    );
}

/// A cycle leaves both profiles with what they declared, and is
/// reported. It must not hang the loader.
#[test]
fn an_inheritance_cycle_terminates_and_is_reported() {
    let cfg = load(
        "name: loop\n\
         profiles:\n\
        \x20 default:\n\
        \x20   base_vectors: base.fvec\n\
        \x20 a:\n\
        \x20   inherits: b\n\
        \x20   query_vectors: qa.fvec\n\
        \x20 b:\n\
        \x20   inherits: a\n\
        \x20   query_terms: qb.fvec\n",
    );
    assert!(cfg.profiles.profile("a").is_some());
    let violations = vectordata::dataset::conformance::validate_conformance(&cfg)
        .expect_err("a cycle is reported");
    assert!(
        violations
            .iter()
            .any(|v| v.key == "inherits" && v.detail.contains("cycle")),
        "{violations:?}"
    );
}

// ── case 12: the gate ──────────────────────────────────────────────

/// **Case 12 (P-13)** — every dataset predating this loads and reads
/// unchanged. Inheritance is what every sized dataset in existence
/// depends on, so this is the case the whole change ships behind.
#[test]
fn datasets_predating_parameterized_profiles_are_unchanged() {
    let cfg = load(
        "name: legacy\n\
         profiles:\n\
        \x20 default:\n\
        \x20   base_vectors: base.fvec\n\
        \x20   query_vectors: query.fvec\n\
        \x20   metadata_content: meta.ivecs\n\
        \x20   neighbor_indices: gt.ivecs\n\
        \x20   neighbor_distances: gtd.fvecs\n\
        \x20 1m:\n\
        \x20   base_count: 1000000\n\
        \x20   neighbor_indices: profiles/1m/neighbor_indices.ivecs\n\
        \x20 100k:\n\
        \x20   base_count: 100000\n",
    );

    for name in ["1m", "100k"] {
        let p = cfg.profiles.profile(name).unwrap();
        // Shared facets inherit, windowed to the profile's base count.
        let base = p.view("base_vectors").expect("inherits base_vectors");
        assert_eq!(base.source.path, "base.fvec");
        assert!(
            !base.source.window.is_empty(),
            "{name} clips the shared base to its own count"
        );
        assert_eq!(
            p.view("query_vectors").map(|v| v.source.path.as_str()),
            Some("query.fvec")
        );
    }

    // The size axis still withholds ground truth from a profile that
    // did not declare it.
    assert!(
        cfg.profiles.profile("100k").unwrap().view("neighbor_indices").is_none(),
        "a sized profile must not pick up the default's full-base ground truth"
    );
    assert_eq!(
        cfg.profiles
            .profile("1m")
            .unwrap()
            .view("neighbor_indices")
            .map(|v| v.source.path.as_str()),
        Some("profiles/1m/neighbor_indices.ivecs"),
        "and keeps the one it declared"
    );
    // Nothing here names a parent, so nothing changed.
    assert!(cfg.profiles.profile("1m").unwrap().inherits.is_none());
}

// ── cases 6, 9: the generator's override set ───────────────────────

/// **Case 6** — a generator's facet templates *are* its override set:
/// what they name is written per member, and what they do not name
/// inherits (P-3).
///
/// This is the general mechanism `sized:` turns out to be one instance
/// of. The override set is a property of the spec, not of a fixed facet
/// table.
#[test]
fn the_generator_templates_are_the_override_set() {
    let cfg = load(
        "name: gen\n\
         profiles:\n\
        \x20 default:\n\
        \x20   base_vectors: base.fvec\n\
        \x20   query_vectors: query.fvec\n\
        \x20   metadata_content: meta.ivecs\n\
        \x20 sized:\n\
        \x20   ranges: [1m..2m/1m]\n\
        \x20   facets:\n\
        \x20     neighbor_indices: profiles/${profile}/neighbor_indices.ivecs\n",
    );

    let (_, members) = cfg.profiles.families().next().expect("a family");
    assert!(members.len() >= 2, "{members:?}");

    for name in members {
        let p = cfg.profiles.profile(name).unwrap_or_else(|| panic!("{name}"));
        // In the override set: written per member, naming the member.
        let gt = p.view("neighbor_indices").expect("declared by the template");
        assert_eq!(
            gt.source.path,
            format!("profiles/{name}/neighbor_indices.ivecs"),
            "the template's facet is the member's own"
        );
        // Outside it: inherited, and not restated per member.
        assert_eq!(
            p.view("query_vectors").map(|v| v.source.path.as_str()),
            Some("query.fvec"),
            "{name} inherits what the generator does not vary"
        );
        assert_eq!(
            p.view("base_vectors").map(|v| v.source.path.as_str()),
            Some("base.fvec")
        );
    }
}

/// **Case 9** — an all-digit generated name is refused when it would be
/// interpolated into a **sharded** filename (P-9, SH-101).
#[test]
fn an_all_digit_name_interpolated_into_a_sharded_template_is_refused() {
    let err = serde_yaml::from_str::<DatasetConfig>(
        "name: ambiguous\n\
         profiles:\n\
        \x20 default:\n\
        \x20   base_vectors: base.fvec\n\
        \x20 sized:\n\
        \x20   ranges: [\"5..10/5\"]\n\
        \x20   facets:\n\
        \x20     metadata_results: p__${profile}__NNNN.ivvec\n",
    )
    .expect_err("an ambiguous derived filename must not load");
    let msg = err.to_string();
    assert!(msg.contains("all digits"), "{msg}");
    assert!(msg.contains("two readings"), "{msg}");
}

/// The same name is fine when nothing shards it. An all-digit profile
/// name is not ambiguous by itself, and strata specs have always
/// generated names like `5` into ordinary paths (P-13).
#[test]
fn an_all_digit_name_is_fine_in_an_unsharded_template() {
    let cfg = load(
        "name: fine\n\
         profiles:\n\
        \x20 default:\n\
        \x20   base_vectors: base.fvec\n\
        \x20 sized:\n\
        \x20   ranges: [\"5..10/5\"]\n\
        \x20   facets:\n\
        \x20     metadata_results: p__${profile}.ivvec\n",
    );
    let names = cfg.profiles.profile_names();
    assert!(
        names
            .iter()
            .any(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())),
        "an all-digit name must still generate when nothing shards it: {names:?}"
    );
}
