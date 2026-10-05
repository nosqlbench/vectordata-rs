// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Stated parents (PL-6).
//!
//! From `format_version` 3 every profile other than `default` states
//! `inherits:`, and a stated 3 is a claim that every parent is real
//! and every profile has one meaning: an absent parent, an unknown or
//! self parent, a cycle, and `partition: true` beside `inherits:` are
//! load refusals naming the profiles involved. Versions 1 and 2 keep
//! their fallbacks, and `veks check` reports the same conditions on
//! them as advisories so the coming rule is seen before it refuses.
//! One pure rule, called by both loaders, so they cannot disagree.

use std::collections::{BTreeSet, HashSet};

use crate::model::FORMAT_VERSION_TAGGED;

/// What the rule reads of a profile.
#[derive(Debug, Clone)]
pub struct ParentFacts<'a> {
    pub name: &'a str,
    pub partition: bool,
    pub inherits: Option<&'a str>,
}

/// Refuse a version-3 dataset whose parents are not all stated and
/// real (PL-6). Below 3 nothing is refused.
pub fn check_parents(version: u32, profiles: &[ParentFacts<'_>]) -> Result<(), String> {
    if version < FORMAT_VERSION_TAGGED {
        return Ok(());
    }
    let faults = faults(profiles, true);
    if faults.is_empty() {
        Ok(())
    } else {
        Err(faults.join("; "))
    }
}

/// The same conditions on a dataset below version 3, phrased as what
/// version 3 will refuse (PL-6), for `veks check`.
pub fn parent_advisories(version: u32, profiles: &[ParentFacts<'_>]) -> Vec<String> {
    if version >= FORMAT_VERSION_TAGGED {
        return Vec::new();
    }
    faults(profiles, false)
        .into_iter()
        .map(|f| format!("{f} (refused from format_version 3)"))
        .collect()
}

fn faults(profiles: &[ParentFacts<'_>], strict: bool) -> Vec<String> {
    let mut sorted: Vec<&ParentFacts<'_>> = profiles.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(b.name));
    let names: HashSet<&str> = profiles.iter().map(|p| p.name).collect();
    let mut faults: Vec<String> = Vec::new();
    for p in &sorted {
        if p.name == "default" {
            if p.inherits.is_some() {
                faults.push("profile `default` is the base layer and names no parent".to_string());
            }
            continue;
        }
        match (p.partition, p.inherits) {
            (true, Some(_)) => faults.push(format!(
                "profile `{}` states both `partition: true` and `inherits:`; a partition builds on nothing",
                p.name
            )),
            (true, None) => {}
            (false, None) => faults.push(format!(
                "profile `{}` names no parent; state `inherits: default` or the layer it builds on",
                p.name
            )),
            (false, Some(parent)) if parent == p.name => {
                faults.push(format!("profile `{}` names itself as its parent", p.name))
            }
            (false, Some(parent)) if !names.contains(parent) => faults.push(format!(
                "profile `{}` names an unknown parent `{parent}`",
                p.name
            )),
            (false, Some(_)) => {}
        }
    }
    // A cycle is reported once, by its sorted members.
    let mut cycles: BTreeSet<Vec<&str>> = BTreeSet::new();
    for p in &sorted {
        if p.partition || p.name == "default" {
            continue;
        }
        let mut seen: Vec<&str> = vec![p.name];
        let mut current = p.name;
        loop {
            let parent = match profiles.iter().find(|q| q.name == current) {
                Some(q) if !q.partition && q.name != "default" => match q.inherits {
                    Some(i) => i,
                    None if strict => break,
                    None => "default",
                },
                _ => break,
            };
            if parent == p.name {
                let mut members = seen.clone();
                members.sort();
                cycles.insert(members);
                break;
            }
            if seen.contains(&parent) || !names.contains(parent) {
                break;
            }
            seen.push(parent);
            current = parent;
        }
    }
    for members in cycles {
        faults.push(format!(
            "profiles {} close an inheritance cycle",
            members.iter().map(|m| format!("`{m}`")).collect::<Vec<_>>().join(", ")
        ));
    }
    faults
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts<'a>(rows: &'a [(&'a str, bool, Option<&'a str>)]) -> Vec<ParentFacts<'a>> {
        rows.iter()
            .map(|(n, p, i)| ParentFacts { name: n, partition: *p, inherits: *i })
            .collect()
    }

    /// **Version 3 refuses an unstated, unknown, self or cyclic parent
    /// and a partition that names one** (PL-6); a partition alone is
    /// accepted, and the same shapes pass below 3.
    #[test]
    fn version_three_states_every_parent() {
        let ok = facts(&[("default", false, None), ("10m", false, Some("default")), ("part", true, None), ("set", false, Some("10m"))]);
        assert!(check_parents(3, &ok).is_ok());

        let unstated = facts(&[("default", false, None), ("10m-mixed", false, None)]);
        let e = check_parents(3, &unstated).unwrap_err();
        assert!(e.contains("`10m-mixed` names no parent") && e.contains("inherits: default"), "{e}");
        assert!(check_parents(2, &unstated).is_ok(), "below 3 the fallback stands");

        let both = facts(&[("default", false, None), ("p", true, Some("default"))]);
        assert!(check_parents(3, &both).unwrap_err().contains("both `partition: true` and `inherits:`"));

        let unknown = facts(&[("default", false, None), ("a", false, Some("nope"))]);
        assert!(check_parents(3, &unknown).unwrap_err().contains("unknown parent `nope`"));

        let selfish = facts(&[("default", false, None), ("a", false, Some("a"))]);
        assert!(check_parents(3, &selfish).unwrap_err().contains("names itself"));

        let cyclic = facts(&[("default", false, None), ("a", false, Some("b")), ("b", false, Some("a"))]);
        let e = check_parents(3, &cyclic).unwrap_err();
        assert!(e.contains("`a`, `b` close an inheritance cycle"), "{e}");
        assert_eq!(e.matches("cycle").count(), 1, "one cycle, reported once: {e}");
    }

    /// Below 3 the conditions are advisories naming the version that
    /// will refuse them; at 3 there are none, the loader refuses.
    #[test]
    fn advisories_precede_the_refusal() {
        let rows = facts(&[("default", false, None), ("10m", false, None), ("a", false, Some("nope"))]);
        let notes = parent_advisories(2, &rows);
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(notes.iter().all(|n| n.ends_with("(refused from format_version 3)")), "{notes:?}");
        assert!(parent_advisories(3, &rows).is_empty());
    }
}
