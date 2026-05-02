// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Structured per-step provenance and selectable staleness checks.
//!
//! Replaces the v4 single-hash `fingerprint: Option<String>` field on
//! [`super::progress::StepRecord`]. Each step now records a
//! [`ProvenanceMap`] capturing every component that *could* be used
//! to decide whether the step is stale: identity (id, command path),
//! the binary version decomposed by axis (major / minor / patch / git
//! hash / dirty), the resolved options as a sorted map, and the full
//! provenance maps of every upstream step.
//!
//! At staleness-check time, the runner picks a [`ProvenanceSelector`]
//! and only the selected components contribute to the comparison.
//! That lets a user say "I just upgraded the binary and I know the
//! import logic didn't change — match by major version + options +
//! upstream" without losing the safety of a content-addressed cache
//! key, and without dropping the components from storage so that a
//! later run can pick a stricter selector and re-validate.
//!
//! The selector is a [`ProvenanceFlags`] bitset; presets (`STRICT`,
//! `VERSION_AWARE`, `CONFIG_ONLY`) are provided for the common cases.
//! The default is `STRICT` — current v4 behaviour.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::progress::FnvHasher;

/// Structured provenance of a single step's execution.
///
/// Stored on [`super::progress::StepRecord`] in place of v4's opaque
/// `fingerprint: String`. Every component is captured here verbatim;
/// the staleness hash is computed *on demand* under a
/// [`ProvenanceSelector`] so a single stored record can answer
/// strict-equality, major-version-equality, options-equality, etc.
/// queries without a re-run of the producing step.
///
/// `upstream` carries the **full** provenance map of each upstream
/// step (not just its hash) so the selector cascades correctly: when
/// a user picks a relaxed selector for the head step, that selector
/// is applied recursively to every upstream contribution. Without
/// the recursive structure, the head's hash would still pull in the
/// strictly-computed leaves and the relaxation would be useless.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceMap {
    /// Step identifier — the YAML `id` field.
    pub step_id: String,
    /// Fully-qualified command path (e.g., `compute knn-metal`).
    pub command_path: String,

    /// Binary's major version at the time the step ran.
    pub binary_version_major: u32,
    /// Binary's minor version.
    pub binary_version_minor: u32,
    /// Binary's patch version.
    pub binary_version_patch: u32,
    /// Git short hash the binary was built from. Empty when the
    /// build wasn't done in a git checkout.
    #[serde(default)]
    pub binary_git_hash: String,
    /// Whether the working tree had uncommitted changes at build
    /// time. Captured but typically excluded from the staleness hash
    /// so local development doesn't cascade everything stale on
    /// every save.
    #[serde(default)]
    pub binary_dirty: bool,

    /// Resolved options for this step. Sorted for deterministic
    /// hashing; `BTreeMap` preserves that on serialise/deserialise.
    #[serde(default)]
    pub options: BTreeMap<String, String>,

    /// Per-upstream provenance, keyed by upstream `step_id`. The
    /// recursive shape is what makes the selector cascade — when the
    /// head's hash is computed under selector S, each upstream
    /// contribution is also computed under S.
    #[serde(default)]
    pub upstream: BTreeMap<String, ProvenanceMap>,
}

impl ProvenanceMap {
    /// Build a `ProvenanceMap` for a step given the resolved option
    /// map, the upstream provenance maps, and a parsed
    /// [`BinaryVersion`]. The binary version is extracted from the
    /// command's [`super::command::CommandOp::build_version`] string
    /// at the call site — see [`BinaryVersion::parse`].
    pub fn build(
        step_id: &str,
        command_path: &str,
        binary: &BinaryVersion,
        options: &std::collections::HashMap<String, String>,
        upstream: BTreeMap<String, ProvenanceMap>,
    ) -> Self {
        let mut sorted_opts: BTreeMap<String, String> = BTreeMap::new();
        for (k, v) in options { sorted_opts.insert(k.clone(), v.clone()); }
        ProvenanceMap {
            step_id: step_id.to_string(),
            command_path: command_path.to_string(),
            binary_version_major: binary.major,
            binary_version_minor: binary.minor,
            binary_version_patch: binary.patch,
            binary_git_hash: binary.git_hash.clone(),
            binary_dirty: binary.dirty,
            options: sorted_opts,
            upstream,
        }
    }

    /// Compute the staleness hash under `selector`. Only the
    /// components selected by the bitset contribute to the hash.
    /// Identity components (`STEP_ID`, `COMMAND_PATH`) are usually
    /// always included — when comparing two stored maps for the
    /// same step, those won't differ anyway, so excluding them is
    /// just a small bit of metadata savings.
    pub fn hash(&self, selector: ProvenanceFlags) -> String {
        let mut h = FnvHasher::new();
        self.hash_into(selector, &mut h);
        format!("{:016x}", h.finish())
    }

    fn hash_into(&self, selector: ProvenanceFlags, h: &mut FnvHasher) {
        if selector.contains(ProvenanceFlags::STEP_ID) {
            h.write(b"step_id=");
            h.write(self.step_id.as_bytes());
            h.write(b"\0");
        }
        if selector.contains(ProvenanceFlags::COMMAND_PATH) {
            h.write(b"command_path=");
            h.write(self.command_path.as_bytes());
            h.write(b"\0");
        }
        if selector.contains(ProvenanceFlags::VERSION_MAJOR) {
            h.write(b"vmaj=");
            h.write(self.binary_version_major.to_string().as_bytes());
            h.write(b"\0");
        }
        if selector.contains(ProvenanceFlags::VERSION_MINOR) {
            h.write(b"vmin=");
            h.write(self.binary_version_minor.to_string().as_bytes());
            h.write(b"\0");
        }
        if selector.contains(ProvenanceFlags::VERSION_PATCH) {
            h.write(b"vpat=");
            h.write(self.binary_version_patch.to_string().as_bytes());
            h.write(b"\0");
        }
        if selector.contains(ProvenanceFlags::GIT_HASH) {
            h.write(b"git=");
            h.write(self.binary_git_hash.as_bytes());
            h.write(b"\0");
        }
        if selector.contains(ProvenanceFlags::DIRTY_FLAG) {
            h.write(b"dirty=");
            h.write(if self.binary_dirty { b"1" } else { b"0" });
            h.write(b"\0");
        }
        if selector.contains(ProvenanceFlags::OPTIONS) {
            h.write(b"options:");
            for (k, v) in &self.options {
                h.write(k.as_bytes());
                h.write(b"=");
                h.write(v.as_bytes());
                h.write(b"\0");
            }
        }
        if selector.contains(ProvenanceFlags::UPSTREAM) {
            h.write(b"upstream:");
            for (id, up) in &self.upstream {
                h.write(id.as_bytes());
                h.write(b":");
                let up_hash = up.hash(selector);
                h.write(up_hash.as_bytes());
                h.write(b"\0");
            }
        }
    }

    /// Diff two provenance maps, returning the set of components
    /// whose values differ. Used by `--explain-staleness` to tell
    /// the user *which* axes pushed a step into the stale bucket.
    /// Upstream differences are reported by upstream id, recursing
    /// only one level (the cascade is implicit — if upstream A is
    /// stale, this step is stale, even if no other component
    /// changed).
    pub fn diff(&self, other: &ProvenanceMap) -> Vec<ProvenanceDiff> {
        let mut out = Vec::new();
        macro_rules! check {
            ($flag:expr, $field:ident, $label:expr) => {
                if self.$field != other.$field {
                    out.push(ProvenanceDiff::Component {
                        flag: $flag,
                        label: $label.to_string(),
                        old: format!("{:?}", other.$field),
                        new: format!("{:?}", self.$field),
                    });
                }
            };
        }
        check!(ProvenanceFlags::STEP_ID, step_id, "step_id");
        check!(ProvenanceFlags::COMMAND_PATH, command_path, "command_path");
        check!(ProvenanceFlags::VERSION_MAJOR, binary_version_major, "binary_version_major");
        check!(ProvenanceFlags::VERSION_MINOR, binary_version_minor, "binary_version_minor");
        check!(ProvenanceFlags::VERSION_PATCH, binary_version_patch, "binary_version_patch");
        check!(ProvenanceFlags::GIT_HASH, binary_git_hash, "binary_git_hash");
        check!(ProvenanceFlags::DIRTY_FLAG, binary_dirty, "binary_dirty");

        // Per-option diff.
        let mut opt_keys: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for k in self.options.keys() { opt_keys.insert(k.as_str()); }
        for k in other.options.keys() { opt_keys.insert(k.as_str()); }
        for k in opt_keys {
            let new = self.options.get(k);
            let old = other.options.get(k);
            if new != old {
                out.push(ProvenanceDiff::Component {
                    flag: ProvenanceFlags::OPTIONS,
                    label: format!("option '{k}'"),
                    old: old.cloned().unwrap_or_else(|| "<unset>".into()),
                    new: new.cloned().unwrap_or_else(|| "<unset>".into()),
                });
            }
        }

        // Per-upstream diff (one level).
        let mut up_keys: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for k in self.upstream.keys() { up_keys.insert(k.as_str()); }
        for k in other.upstream.keys() { up_keys.insert(k.as_str()); }
        for k in up_keys {
            let new = self.upstream.get(k);
            let old = other.upstream.get(k);
            match (new, old) {
                (Some(n), Some(o)) if n == o => {}
                (None, None) => {}
                _ => {
                    out.push(ProvenanceDiff::UpstreamChanged(k.to_string()));
                }
            }
        }
        out
    }
}

/// One axis of difference between two `ProvenanceMap`s.
#[derive(Debug, Clone)]
pub enum ProvenanceDiff {
    /// A specific component diverged (version, an option, etc.).
    Component { flag: ProvenanceFlags, label: String, old: String, new: String },
    /// An upstream step's provenance changed (likely because that
    /// upstream re-ran or its own provenance shifted).
    UpstreamChanged(String),
}

impl std::fmt::Display for ProvenanceDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProvenanceDiff::Component { label, old, new, .. } => {
                write!(f, "{label}: {old} → {new}")
            }
            ProvenanceDiff::UpstreamChanged(id) => {
                write!(f, "upstream '{id}' provenance changed")
            }
        }
    }
}

/// Bitset selecting which provenance components contribute to the
/// staleness hash. Use the `STRICT`, `VERSION_AWARE`, `CONFIG_ONLY`
/// constants for typical configurations, or build a custom set with
/// `from_components`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceFlags(u32);

impl ProvenanceFlags {
    pub const STEP_ID:        Self = Self(1 << 0);
    pub const COMMAND_PATH:   Self = Self(1 << 1);
    pub const VERSION_MAJOR:  Self = Self(1 << 2);
    pub const VERSION_MINOR:  Self = Self(1 << 3);
    pub const VERSION_PATCH:  Self = Self(1 << 4);
    pub const GIT_HASH:       Self = Self(1 << 5);
    pub const DIRTY_FLAG:     Self = Self(1 << 6);
    pub const OPTIONS:        Self = Self(1 << 7);
    pub const UPSTREAM:       Self = Self(1 << 8);

    pub const fn empty() -> Self { Self(0) }

    /// Strict / current v4 behaviour: every component.
    pub const STRICT: Self = Self(
        Self::STEP_ID.0 | Self::COMMAND_PATH.0
            | Self::VERSION_MAJOR.0 | Self::VERSION_MINOR.0 | Self::VERSION_PATCH.0
            | Self::GIT_HASH.0 | Self::DIRTY_FLAG.0
            | Self::OPTIONS.0 | Self::UPSTREAM.0
    );

    /// Major-version aware: ignore minor/patch/git/dirty changes.
    /// Suitable when a user trusts that minor/patch releases don't
    /// affect a step's outputs but a major-version bump might.
    pub const VERSION_AWARE: Self = Self(
        Self::STEP_ID.0 | Self::COMMAND_PATH.0
            | Self::VERSION_MAJOR.0
            | Self::OPTIONS.0 | Self::UPSTREAM.0
    );

    /// Ignore binary version entirely; only re-run when the step's
    /// own configuration or one of its upstreams changes.
    pub const CONFIG_ONLY: Self = Self(
        Self::STEP_ID.0 | Self::COMMAND_PATH.0
            | Self::OPTIONS.0 | Self::UPSTREAM.0
    );

    pub fn contains(&self, other: ProvenanceFlags) -> bool {
        (self.0 & other.0) == other.0 && other.0 != 0
    }

    pub fn bits(&self) -> u32 { self.0 }

    /// Parse a comma-separated component list (case-insensitive,
    /// hyphens or underscores accepted) or one of the named presets.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let s = spec.trim().to_lowercase().replace('-', "_");
        match s.as_str() {
            "strict" | "all"           => return Ok(Self::STRICT),
            "version_aware"            => return Ok(Self::VERSION_AWARE),
            "config_only" | "config"   => return Ok(Self::CONFIG_ONLY),
            _ => {}
        }
        let mut out = Self::empty();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() { continue; }
            let bit = match part {
                "step_id"        => Self::STEP_ID,
                "command_path"   => Self::COMMAND_PATH,
                "version_major"  => Self::VERSION_MAJOR,
                "version_minor"  => Self::VERSION_MINOR,
                "version_patch"  => Self::VERSION_PATCH,
                "git_hash"       => Self::GIT_HASH,
                "dirty_flag" | "dirty" => Self::DIRTY_FLAG,
                "options"        => Self::OPTIONS,
                "upstream"       => Self::UPSTREAM,
                other => return Err(format!(
                    "unknown provenance component '{other}'. Known: \
                     step_id, command_path, version_major, version_minor, \
                     version_patch, git_hash, dirty, options, upstream. \
                     Or presets: strict, version-aware, config-only."
                )),
            };
            out.0 |= bit.0;
        }
        Ok(out)
    }

    /// Render the selector as a human-readable comma-separated list.
    pub fn describe(&self) -> String {
        if *self == Self::STRICT { return "strict".into(); }
        if *self == Self::VERSION_AWARE { return "version-aware".into(); }
        if *self == Self::CONFIG_ONLY { return "config-only".into(); }
        let mut parts: Vec<&str> = Vec::new();
        if self.contains(Self::STEP_ID)        { parts.push("step_id"); }
        if self.contains(Self::COMMAND_PATH)   { parts.push("command_path"); }
        if self.contains(Self::VERSION_MAJOR)  { parts.push("version_major"); }
        if self.contains(Self::VERSION_MINOR)  { parts.push("version_minor"); }
        if self.contains(Self::VERSION_PATCH)  { parts.push("version_patch"); }
        if self.contains(Self::GIT_HASH)       { parts.push("git_hash"); }
        if self.contains(Self::DIRTY_FLAG)     { parts.push("dirty"); }
        if self.contains(Self::OPTIONS)        { parts.push("options"); }
        if self.contains(Self::UPSTREAM)       { parts.push("upstream"); }
        parts.join(",")
    }
}

impl Default for ProvenanceFlags {
    fn default() -> Self { Self::STRICT }
}

/// Decomposed binary version. Parsed from
/// [`super::command::CommandOp::build_version`]'s
/// `{CARGO_PKG_VERSION}+{git_short}[+dirty]` format. Never fails —
/// missing components default to zero / empty / false so an
/// unrecognised string still produces a usable map.
#[derive(Debug, Clone, Default)]
pub struct BinaryVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub git_hash: String,
    pub dirty: bool,
}

impl BinaryVersion {
    /// Parse `{CARGO_PKG_VERSION}+{git_short}[+dirty]`. Forgiving:
    /// any component that doesn't parse is left at its default.
    pub fn parse(s: &str) -> Self {
        let mut out = Self::default();
        let mut parts = s.split('+');
        if let Some(ver) = parts.next() {
            let mut nums = ver.split('.');
            out.major = nums.next().and_then(|p| p.parse().ok()).unwrap_or(0);
            out.minor = nums.next().and_then(|p| p.parse().ok()).unwrap_or(0);
            out.patch = nums.next().and_then(|p| p.parse().ok()).unwrap_or(0);
        }
        if let Some(hash) = parts.next() {
            out.git_hash = hash.to_string();
        }
        if parts.any(|p| p == "dirty") {
            out.dirty = true;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> std::collections::HashMap<String, String> {
        let mut m = std::collections::HashMap::new();
        m.insert("k".into(), "100".into());
        m.insert("metric".into(), "L2".into());
        m
    }

    fn make_map(version: &str, opts_in: std::collections::HashMap<String, String>) -> ProvenanceMap {
        ProvenanceMap::build(
            "compute-knn",
            "compute knn",
            &BinaryVersion::parse(version),
            &opts_in,
            BTreeMap::new(),
        )
    }

    #[test]
    fn binary_version_parse_full() {
        let v = BinaryVersion::parse("1.2.3+abcd1234+dirty");
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert_eq!(v.git_hash, "abcd1234");
        assert!(v.dirty);
    }

    #[test]
    fn binary_version_parse_clean() {
        let v = BinaryVersion::parse("0.25.0+abcd1234");
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 25);
        assert_eq!(v.patch, 0);
        assert_eq!(v.git_hash, "abcd1234");
        assert!(!v.dirty);
    }

    #[test]
    fn flags_parse_presets() {
        assert_eq!(ProvenanceFlags::parse("strict").unwrap(), ProvenanceFlags::STRICT);
        assert_eq!(ProvenanceFlags::parse("version-aware").unwrap(), ProvenanceFlags::VERSION_AWARE);
        assert_eq!(ProvenanceFlags::parse("config-only").unwrap(), ProvenanceFlags::CONFIG_ONLY);
    }

    #[test]
    fn flags_parse_custom_list() {
        let f = ProvenanceFlags::parse("step_id,command_path,version_major,options").unwrap();
        assert!(f.contains(ProvenanceFlags::STEP_ID));
        assert!(f.contains(ProvenanceFlags::COMMAND_PATH));
        assert!(f.contains(ProvenanceFlags::VERSION_MAJOR));
        assert!(f.contains(ProvenanceFlags::OPTIONS));
        assert!(!f.contains(ProvenanceFlags::VERSION_MINOR));
        assert!(!f.contains(ProvenanceFlags::UPSTREAM));
    }

    #[test]
    fn flags_parse_unknown_errors() {
        let err = ProvenanceFlags::parse("step_id,bogus").unwrap_err();
        assert!(err.contains("bogus"));
    }

    #[test]
    fn version_bump_stale_under_strict_fresh_under_config_only() {
        let a = make_map("1.0.1+abcd1234", opts());
        let b = make_map("0.25.0+ffff0000", opts());

        // Strict: differs (binary version + git differ).
        assert_ne!(a.hash(ProvenanceFlags::STRICT), b.hash(ProvenanceFlags::STRICT));

        // Config-only: same (only options + identity matter).
        assert_eq!(
            a.hash(ProvenanceFlags::CONFIG_ONLY),
            b.hash(ProvenanceFlags::CONFIG_ONLY),
            "binary version change must not invalidate under CONFIG_ONLY"
        );
    }

    #[test]
    fn major_bump_invalidates_under_version_aware_minor_does_not() {
        let a = make_map("1.0.1+abcd", opts());
        let b = make_map("2.0.0+abcd", opts());
        let c = make_map("1.5.7+abcd", opts());

        // Major bump: stale under VERSION_AWARE.
        assert_ne!(a.hash(ProvenanceFlags::VERSION_AWARE),
                   b.hash(ProvenanceFlags::VERSION_AWARE));
        // Minor bump: still fresh under VERSION_AWARE.
        assert_eq!(a.hash(ProvenanceFlags::VERSION_AWARE),
                   c.hash(ProvenanceFlags::VERSION_AWARE),
                   "minor-version bump must not invalidate under VERSION_AWARE");
    }

    #[test]
    fn options_change_invalidates_everywhere() {
        let mut o2 = opts();
        o2.insert("k".into(), "200".into());
        let a = make_map("1.0.1+abcd", opts());
        let b = make_map("1.0.1+abcd", o2);
        assert_ne!(a.hash(ProvenanceFlags::CONFIG_ONLY), b.hash(ProvenanceFlags::CONFIG_ONLY));
        assert_ne!(a.hash(ProvenanceFlags::STRICT), b.hash(ProvenanceFlags::STRICT));
    }

    #[test]
    fn upstream_change_cascades_under_upstream_flag() {
        let mut up_a = ProvenanceMap::build(
            "extract", "transform extract", &BinaryVersion::parse("1.0.0+abcd"),
            &opts(), BTreeMap::new(),
        );
        let up_b = up_a.clone();
        up_a.binary_version_major = 1;
        let mut a_upstreams = BTreeMap::new();
        a_upstreams.insert("extract".to_string(), up_a);
        let mut b_upstreams = BTreeMap::new();
        let mut up_b_changed = up_b.clone();
        up_b_changed.binary_version_major = 2;
        b_upstreams.insert("extract".to_string(), up_b_changed);

        let head_a = ProvenanceMap::build(
            "knn", "compute knn", &BinaryVersion::parse("1.0.0+abcd"),
            &opts(), a_upstreams,
        );
        let head_b = ProvenanceMap::build(
            "knn", "compute knn", &BinaryVersion::parse("1.0.0+abcd"),
            &opts(), b_upstreams,
        );

        // Upstream's major changed; under VERSION_AWARE+UPSTREAM
        // (which VERSION_AWARE preset includes) the head should be
        // stale because the upstream's contribution differs.
        assert_ne!(head_a.hash(ProvenanceFlags::VERSION_AWARE),
                   head_b.hash(ProvenanceFlags::VERSION_AWARE),
                   "upstream version change must cascade to head");

        // If we drop UPSTREAM from the selector, the head's own
        // components are identical → fresh.
        let no_up = ProvenanceFlags(ProvenanceFlags::CONFIG_ONLY.0 & !ProvenanceFlags::UPSTREAM.0);
        assert_eq!(head_a.hash(no_up), head_b.hash(no_up));
    }

    #[test]
    fn diff_reports_changed_components() {
        let a = make_map("1.0.1+abcd", opts());
        let b = make_map("2.0.0+ffff", opts());
        let diffs = a.diff(&b);
        let labels: Vec<String> = diffs.iter().map(|d| match d {
            ProvenanceDiff::Component { label, .. } => label.clone(),
            ProvenanceDiff::UpstreamChanged(id) => format!("upstream:{id}"),
        }).collect();
        assert!(labels.iter().any(|l| l == "binary_version_major"));
        assert!(labels.iter().any(|l| l == "binary_git_hash"));
    }
}
