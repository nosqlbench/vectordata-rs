// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Structured per-step provenance and selectable staleness checks.
//!
//! Each step records a [`ProvenanceNode`] capturing every component
//! that *could* be used to decide whether the step is stale: identity
//! (id, command path), the binary version decomposed by axis (major /
//! minor / patch / git hash / dirty), the resolved options as a sorted
//! map, and — by **address** — the node of every upstream step it was
//! built on.
//!
//! Nodes live in a [`ProvenanceGraph`], a flat table keyed by content
//! address: a node's address is its hash under
//! [`ProvenanceFlags::STRICT`], so equal content has equal address, a
//! subtree shared by many dependents is stored once, and a step that
//! runs again with different inputs gets a *new* node while the node
//! its dependents were built on stays in the table for as long as
//! they reference it. Nothing nests: the graph is the same shape in
//! memory, in the progress log and in a sidecar, and its depth on disk
//! is constant however long an `after:` chain runs.
//!
//! At staleness-check time, the runner picks a selector and hashes the
//! recorded and the current node under it; the hash recurses through
//! upstream addresses, so a relaxed selector cascades: when the head
//! is hashed under selector S, each upstream contribution is also
//! computed under S. That lets a user say "I just upgraded the binary
//! and I know the import logic didn't change — match by major version,
//! options and upstream" without losing a content-addressed cache key,
//! and without dropping components from storage so that a later run
//! can pick a stricter selector and re-validate.
//!
//! The selector is a [`ProvenanceFlags`] bitset; presets (`STRICT`,
//! `VERSION_AWARE`, `CONFIG_ONLY`) are provided for the common cases.
//! The default is [`ProvenanceFlags::DEFAULT`].

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::progress::FnvHasher;

/// Extension appended to the cached provenance path for an artifact.
/// Sidecars are centralised under `<cache>/provenance/` (never beside
/// the dataset artifact) — see [`ProvenanceGraph::sidecar_path`] for the
/// 1-1 affine mapping.
pub const SIDECAR_EXT: &str = "provenance.json";

/// The content address of a [`ProvenanceNode`]: its hash under
/// [`ProvenanceFlags::STRICT`], sixteen hex digits.
pub type Address = String;

/// Structured provenance of a single step's execution.
///
/// Every component is captured verbatim; the staleness hash is
/// computed *on demand* by [`ProvenanceGraph::hash`] under a selector,
/// so one stored node can answer strict-equality,
/// major-version-equality, options-equality, etc. queries without a
/// re-run of the producing step.
///
/// `upstream` names each upstream by the **address** of the node the
/// step was built on. The node itself is in the graph; two dependents
/// of one upstream share it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceNode {
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
    /// Per-upstream node address, keyed by upstream `step_id` (or by
    /// input role, for a command keying a cache on its inputs).
    #[serde(default)]
    pub upstream: BTreeMap<String, Address>,
}

impl ProvenanceNode {
    /// Build a node for a step given the resolved option map, the
    /// upstream addresses, and a parsed [`BinaryVersion`]. The binary
    /// version is extracted from the command's
    /// [`super::command::CommandOp::build_version`] string at the
    /// call site — see [`BinaryVersion::parse`].
    pub fn build(
        step_id: &str,
        command_path: &str,
        binary: &BinaryVersion,
        options: &HashMap<String, String>,
        upstream: BTreeMap<String, Address>,
    ) -> Self {
        ProvenanceNode {
            step_id: step_id.to_string(),
            command_path: command_path.to_string(),
            binary_version_major: binary.major,
            binary_version_minor: binary.minor,
            binary_version_patch: binary.patch,
            binary_git_hash: binary.git_hash.clone(),
            binary_dirty: binary.dirty,
            options: options.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            upstream,
        }
    }

    /// The node recorded for an upstream whose record is missing or
    /// carries no provenance: a known-distinct sentinel under any
    /// selector that includes `UPSTREAM`, so the dependent counts as
    /// stale until the upstream has a real record.
    pub fn unknown(step_id: &str) -> Self {
        ProvenanceNode {
            step_id: step_id.to_string(),
            command_path: String::new(),
            binary_version_major: 0,
            binary_version_minor: 0,
            binary_version_patch: 0,
            binary_git_hash: String::new(),
            binary_dirty: false,
            options: BTreeMap::new(),
            upstream: BTreeMap::new(),
        }
    }

    /// Build a degenerate node from a file's path, size, and mtime
    /// when no sidecar is available. The cascade is weaker here — two
    /// files with identical size/mtime collide — but it preserves the
    /// *cheap and sticky* contract: an overwrite, regenerate, or touch
    /// invalidates downstream caches without ever reading the file's
    /// content.
    ///
    /// The resulting node has empty `binary_*` and `upstream` fields;
    /// its load-bearing distinguishers are the path and the file
    /// metadata, recorded under `options`. The synthetic step id
    /// `degenerate:<filename>` makes it clear in `diff` output that
    /// this entry didn't come from an upstream pipeline step.
    pub fn degenerate_from_artifact(artifact: &Path) -> std::io::Result<Self> {
        let meta = std::fs::metadata(artifact)?;
        let size = meta.len();
        // `mtime` only — `ctime`/`atime` swing too much to be a
        // stable cache signal. The literal seconds + nanos make a
        // string that diff prints cleanly.
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| format!("{}.{:09}", d.as_secs(), d.subsec_nanos()))
            .unwrap_or_else(|| "0".to_string());
        let name = artifact
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string());
        let mut options: BTreeMap<String, String> = BTreeMap::new();
        options.insert("path".into(), artifact.to_string_lossy().into_owned());
        options.insert("size".into(), size.to_string());
        options.insert("mtime".into(), mtime);
        Ok(ProvenanceNode {
            step_id: format!("degenerate:{name}"),
            command_path: "degenerate".into(),
            binary_version_major: 0,
            binary_version_minor: 0,
            binary_version_patch: 0,
            binary_git_hash: String::new(),
            binary_dirty: false,
            options,
            upstream: BTreeMap::new(),
        })
    }

    /// The node's content address: its hash under
    /// [`ProvenanceFlags::STRICT`], to which each upstream contributes
    /// its own address. Because an address *is* the strict hash,
    /// [`ProvenanceGraph::hash`] under `STRICT` returns the address
    /// unchanged.
    pub fn address(&self) -> Address {
        let mut h = FnvHasher::new();
        self.hash_own_into(ProvenanceFlags::STRICT, &mut h);
        h.write(b"upstream:");
        for (id, address) in &self.upstream {
            h.write(id.as_bytes());
            h.write(b":");
            h.write(address.as_bytes());
            h.write(b"\0");
        }
        format!("{:016x}", h.finish())
    }

    /// Hash the node's own components (everything but `upstream`)
    /// under `selector` into `h`.
    fn hash_own_into(&self, selector: ProvenanceFlags, h: &mut FnvHasher) {
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
    }
}

/// A table of [`ProvenanceNode`]s by address — the whole provenance of
/// a progress log, or the part of it a sidecar carries.
///
/// The graph is **closed**: a node can only be inserted once every
/// address it names is present, so a hash never has to guess at a
/// missing upstream. It serialises as the bare address → node map.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct ProvenanceGraph {
    nodes: BTreeMap<Address, ProvenanceNode>,
}

impl ProvenanceGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn get(&self, address: &str) -> Option<&ProvenanceNode> {
        self.nodes.get(address)
    }

    pub fn contains(&self, address: &str) -> bool {
        self.nodes.contains_key(address)
    }

    /// Insert `node`, returning its address. Every upstream address
    /// the node names must already be in the graph.
    pub fn insert(&mut self, node: ProvenanceNode) -> Result<Address, String> {
        for (id, address) in &node.upstream {
            if !self.nodes.contains_key(address) {
                return Err(format!(
                    "provenance of '{}' names upstream '{}' at {}, which the graph does not hold",
                    node.step_id, id, address
                ));
            }
        }
        let address = node.address();
        self.nodes.entry(address.clone()).or_insert(node);
        Ok(address)
    }

    /// Take every node of `other` — the way a sidecar's nodes join
    /// the graph of the run that reads it.
    pub fn absorb(&mut self, other: &ProvenanceGraph) {
        for (address, node) in &other.nodes {
            self.nodes
                .entry(address.clone())
                .or_insert_with(|| node.clone());
        }
    }

    /// The staleness hash of the node at `address` under `selector`,
    /// recursing through upstream addresses so a relaxed selector
    /// applies all the way down. `None` if the graph has no such node.
    pub fn hash(&self, address: &str, selector: ProvenanceFlags) -> Option<String> {
        let mut memo: HashMap<&str, String> = HashMap::new();
        self.hash_memo(address, selector, &mut memo)
    }

    fn hash_memo<'a>(
        &'a self,
        address: &'a str,
        selector: ProvenanceFlags,
        memo: &mut HashMap<&'a str, String>,
    ) -> Option<String> {
        if let Some(done) = memo.get(address) {
            return Some(done.clone());
        }
        let node = self.nodes.get(address)?;
        let mut h = FnvHasher::new();
        node.hash_own_into(selector, &mut h);
        if selector.contains(ProvenanceFlags::UPSTREAM) {
            h.write(b"upstream:");
            for (id, up_address) in &node.upstream {
                h.write(id.as_bytes());
                h.write(b":");
                // A closed graph always resolves; the literal address
                // is the deterministic stand-in should it not.
                let up_hash = self
                    .hash_memo(up_address, selector, memo)
                    .unwrap_or_else(|| up_address.clone());
                h.write(up_hash.as_bytes());
                h.write(b"\0");
            }
        }
        let out = format!("{:016x}", h.finish());
        memo.insert(address, out.clone());
        Some(out)
    }

    /// The addresses reachable from `roots`, roots included.
    fn reachable_from<'a>(&self, roots: impl IntoIterator<Item = &'a str>) -> HashSet<String> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut stack: Vec<String> = roots.into_iter().map(str::to_string).collect();
        while let Some(address) = stack.pop() {
            if !seen.insert(address.clone()) {
                continue;
            }
            if let Some(node) = self.nodes.get(&address) {
                stack.extend(node.upstream.values().cloned());
            }
        }
        seen
    }

    /// The self-contained subgraph reachable from `root`.
    pub fn reachable(&self, root: &str) -> ProvenanceGraph {
        let keep = self.reachable_from([root]);
        ProvenanceGraph {
            nodes: self
                .nodes
                .iter()
                .filter(|(a, _)| keep.contains(a.as_str()))
                .map(|(a, n)| (a.clone(), n.clone()))
                .collect(),
        }
    }

    /// Drop every node not reachable from `roots` — the versions no
    /// record references any more.
    pub fn retain_reachable<'a>(&mut self, roots: impl IntoIterator<Item = &'a str>) {
        let keep = self.reachable_from(roots);
        self.nodes.retain(|a, _| keep.contains(a.as_str()));
    }

    /// Diff the node at `new` against the node at `old`, returning the
    /// components whose values differ. Used by `--explain-staleness`
    /// to tell the user *which* axes pushed a step into the stale
    /// bucket. Upstream differences are reported by upstream id,
    /// one level deep (the cascade is implicit — if upstream A is
    /// stale, this step is stale, even if no other component
    /// changed). An address the graph does not hold diffs as if every
    /// component were absent.
    pub fn diff(&self, new: &str, old: &str) -> Vec<ProvenanceDiff> {
        let (Some(new), Some(old)) = (self.nodes.get(new), self.nodes.get(old)) else {
            return vec![ProvenanceDiff::Component {
                flag: ProvenanceFlags::STEP_ID,
                label: "record".into(),
                old: if self.nodes.contains_key(old) { "present".into() } else { "<absent>".into() },
                new: if self.nodes.contains_key(new) { "present".into() } else { "<absent>".into() },
            }];
        };
        let mut out = Vec::new();
        macro_rules! check {
            ($flag:expr, $field:ident, $label:expr) => {
                if new.$field != old.$field {
                    out.push(ProvenanceDiff::Component {
                        flag: $flag,
                        label: $label.to_string(),
                        old: format!("{:?}", old.$field),
                        new: format!("{:?}", new.$field),
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
        let opt_keys: std::collections::BTreeSet<&str> = new
            .options
            .keys()
            .chain(old.options.keys())
            .map(String::as_str)
            .collect();
        for k in opt_keys {
            let n = new.options.get(k);
            let o = old.options.get(k);
            if n != o {
                out.push(ProvenanceDiff::Component {
                    flag: ProvenanceFlags::OPTIONS,
                    label: format!("option '{k}'"),
                    old: o.cloned().unwrap_or_else(|| "<unset>".into()),
                    new: n.cloned().unwrap_or_else(|| "<unset>".into()),
                });
            }
        }
        let up_keys: std::collections::BTreeSet<&str> = new
            .upstream
            .keys()
            .chain(old.upstream.keys())
            .map(String::as_str)
            .collect();
        for k in up_keys {
            if new.upstream.get(k) != old.upstream.get(k) {
                out.push(ProvenanceDiff::UpstreamChanged(k.to_string()));
            }
        }
        out
    }

    // ── Sidecars ────────────────────────────────────────────────────

    /// Path of the provenance sidecar **co-located** with a cache
    /// segment (e.g. `<cache>/run-3.slab` →
    /// `<cache>/run-3.slab.provenance.json`). Used only for
    /// intermediate cache files that already live under `.cache/` —
    /// dataset artifacts use [`cached_sidecar_path`](Self::cached_sidecar_path).
    pub fn sidecar_path(artifact: &Path) -> PathBuf {
        let mut p = artifact.as_os_str().to_os_string();
        p.push(".");
        p.push(SIDECAR_EXT);
        PathBuf::from(p)
    }

    /// Path of the provenance sidecar for a **dataset artifact**.
    ///
    /// Provenance is staleness *metadata*, not dataset content, so it
    /// must never pollute the dataset storage layer. Dataset-artifact
    /// sidecars live under the cache directory in a dedicated
    /// `provenance/` subdirectory, at a **1-1 affine mapping** of the
    /// artifact's workspace-relative path:
    ///
    /// ```text
    /// profiles/base/base_vectors.fvecs
    ///   → <cache>/provenance/profiles/base/base_vectors.fvecs.provenance.json
    /// ```
    ///
    /// The transform is structure-preserving and reversible (strip the
    /// `<cache>/provenance/` prefix and the `.provenance.json` suffix to
    /// recover the artifact's relative path). `rel_artifact` is taken
    /// relative to the dataset workspace; any leading root/prefix is
    /// dropped so an absolute path still nests under `provenance/`.
    pub fn cached_sidecar_path(cache_dir: &Path, rel_artifact: &Path) -> PathBuf {
        use std::path::Component;
        let rel: PathBuf = rel_artifact
            .components()
            .filter(|c| !matches!(c, Component::RootDir | Component::Prefix(_)))
            .collect();
        let mut p = cache_dir.join("provenance").join(&rel).into_os_string();
        p.push(".");
        p.push(SIDECAR_EXT);
        PathBuf::from(p)
    }

    /// Write the subgraph reachable from `root` to `path` as a
    /// [`ProvenanceSidecar`], creating parent dirs. Pretty-printed JSON
    /// so the file is greppable in the field.
    fn write_to(&self, root: &str, path: &Path) -> std::io::Result<()> {
        let sidecar = ProvenanceSidecar {
            root: root.to_string(),
            nodes: self.reachable(root),
        };
        let body = serde_json::to_vec_pretty(&sidecar)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, body)
    }

    /// Write the provenance rooted at `root` as the co-located sidecar
    /// of a cache segment (see [`sidecar_path`](Self::sidecar_path)).
    pub fn write_sidecar(&self, root: &str, artifact: &Path) -> std::io::Result<()> {
        self.write_to(root, &Self::sidecar_path(artifact))
    }

    /// Write the provenance rooted at `root` as the cached sidecar of a
    /// dataset artifact (see [`cached_sidecar_path`](Self::cached_sidecar_path)).
    pub fn write_cached_sidecar(
        &self,
        root: &str,
        cache_dir: &Path,
        rel_artifact: &Path,
    ) -> std::io::Result<()> {
        self.write_to(root, &Self::cached_sidecar_path(cache_dir, rel_artifact))
    }

    fn read_from(path: &Path) -> std::io::Result<Option<ProvenanceSidecar>> {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice::<ProvenanceSidecar>(&bytes)
                .map(Some)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Read the co-located sidecar of a cache segment. Returns
    /// `Ok(None)` if absent — consumers should fall back to
    /// [`ProvenanceNode::degenerate_from_artifact`] so hand-curated or
    /// pre-existing files still cascade *something* into the
    /// consumer's hash.
    pub fn read_sidecar(artifact: &Path) -> std::io::Result<Option<ProvenanceSidecar>> {
        Self::read_from(&Self::sidecar_path(artifact))
    }

    /// Read the cached sidecar of a dataset artifact (the counterpart of
    /// [`write_cached_sidecar`](Self::write_cached_sidecar)). `Ok(None)`
    /// when absent.
    pub fn read_cached_sidecar(
        cache_dir: &Path,
        rel_artifact: &Path,
    ) -> std::io::Result<Option<ProvenanceSidecar>> {
        Self::read_from(&Self::cached_sidecar_path(cache_dir, rel_artifact))
    }

    /// Take the provenance of an input artifact into this graph for an
    /// upstream cascade and return its address, checking both sidecar
    /// homes before degrading: the cached (relocated) sidecar of a
    /// dataset artifact, then the co-located sidecar of a cache
    /// segment, then a degenerate `(path, size, mtime)` node from the
    /// file itself. `workspace` is the dataset root used to derive the
    /// artifact's relative path.
    pub fn for_input(
        &mut self,
        cache_dir: &Path,
        workspace: &Path,
        artifact: &Path,
    ) -> std::io::Result<Address> {
        let rel = artifact.strip_prefix(workspace).unwrap_or(artifact);
        if let Some(sidecar) = Self::read_cached_sidecar(cache_dir, rel)? {
            return Ok(sidecar.absorb_into(self));
        }
        if let Some(sidecar) = Self::read_sidecar(artifact)? {
            return Ok(sidecar.absorb_into(self));
        }
        let node = ProvenanceNode::degenerate_from_artifact(artifact)?;
        self.insert(node)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// A sidecar: the provenance of one artifact, self-contained — the
/// address of its node and every node that address reaches.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceSidecar {
    pub root: Address,
    pub nodes: ProvenanceGraph,
}

impl ProvenanceSidecar {
    /// Join the sidecar's nodes to `graph` and return the root address.
    pub fn absorb_into(self, graph: &mut ProvenanceGraph) -> Address {
        graph.absorb(&self.nodes);
        self.root
    }
}

/// One axis of difference between two provenance nodes.
#[derive(Debug, Clone)]
pub enum ProvenanceDiff {
    /// A specific component diverged (version, an option, etc.).
    Component {
        flag: ProvenanceFlags,
        label: String,
        old: String,
        new: String,
    },
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
/// presets or compose from the individual flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    /// Strict: every component.
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

    /// What a run uses when nothing asks for anything else.
    ///
    /// `CONFIG_ONLY`: a step is stale when its own options or an
    /// upstream changed, not merely because the binary did. See the
    /// `--provenance` flag for why this is the default rather than
    /// [`Self::STRICT`].
    pub const DEFAULT: Self = Self::CONFIG_ONLY;

    /// The name of [`Self::DEFAULT`], for the CLI and for the callers
    /// that build a run programmatically. One spelling, so a change
    /// here reaches all of them.
    pub const DEFAULT_NAME: &'static str = "config-only";

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
/// unrecognised string still produces a usable node.
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

    fn opts() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("k".into(), "100".into());
        m.insert("metric".into(), "L2".into());
        m
    }

    fn node(version: &str, opts_in: HashMap<String, String>) -> ProvenanceNode {
        ProvenanceNode::build(
            "compute-knn",
            "compute knn",
            &BinaryVersion::parse(version),
            &opts_in,
            BTreeMap::new(),
        )
    }

    /// A graph holding one node; returns the graph and the address.
    fn single(version: &str, opts_in: HashMap<String, String>) -> (ProvenanceGraph, Address) {
        let mut g = ProvenanceGraph::new();
        let a = g.insert(node(version, opts_in)).unwrap();
        (g, a)
    }

    fn hash_of(version: &str, opts_in: HashMap<String, String>, sel: ProvenanceFlags) -> String {
        let (g, a) = single(version, opts_in);
        g.hash(&a, sel).unwrap()
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

    /// An address is the strict hash, so hashing under STRICT returns
    /// the address itself — the property the cache keys rely on.
    #[test]
    fn the_address_is_the_strict_hash() {
        let (g, a) = single("1.0.1+abcd1234", opts());
        assert_eq!(g.hash(&a, ProvenanceFlags::STRICT).unwrap(), a);
        assert_eq!(g.get(&a).unwrap().address(), a);
    }

    #[test]
    fn version_bump_stale_under_strict_fresh_under_config_only() {
        let a = "1.0.1+abcd1234";
        let b = "0.25.0+ffff0000";
        assert_ne!(hash_of(a, opts(), ProvenanceFlags::STRICT), hash_of(b, opts(), ProvenanceFlags::STRICT));
        assert_eq!(
            hash_of(a, opts(), ProvenanceFlags::CONFIG_ONLY),
            hash_of(b, opts(), ProvenanceFlags::CONFIG_ONLY),
            "binary version change must not invalidate under CONFIG_ONLY"
        );
    }

    #[test]
    fn major_bump_invalidates_under_version_aware_minor_does_not() {
        let sel = ProvenanceFlags::VERSION_AWARE;
        assert_ne!(hash_of("1.0.1+abcd", opts(), sel), hash_of("2.0.0+abcd", opts(), sel));
        assert_eq!(
            hash_of("1.0.1+abcd", opts(), sel),
            hash_of("1.5.7+abcd", opts(), sel),
            "minor-version bump must not invalidate under VERSION_AWARE"
        );
    }

    #[test]
    fn options_change_invalidates_everywhere() {
        let mut o2 = opts();
        o2.insert("k".into(), "200".into());
        assert_ne!(hash_of("1.0.1+abcd", opts(), ProvenanceFlags::CONFIG_ONLY), hash_of("1.0.1+abcd", o2.clone(), ProvenanceFlags::CONFIG_ONLY));
        assert_ne!(hash_of("1.0.1+abcd", opts(), ProvenanceFlags::STRICT), hash_of("1.0.1+abcd", o2, ProvenanceFlags::STRICT));
    }

    /// Two heads built on two versions of one upstream: the selector
    /// that includes UPSTREAM tells them apart, the one that drops it
    /// does not.
    #[test]
    fn upstream_change_cascades_under_upstream_flag() {
        let mut g = ProvenanceGraph::new();
        let up_v1 = g
            .insert(ProvenanceNode::build("extract", "transform extract", &BinaryVersion::parse("1.0.0+abcd"), &opts(), BTreeMap::new()))
            .unwrap();
        let up_v2 = g
            .insert(ProvenanceNode::build("extract", "transform extract", &BinaryVersion::parse("2.0.0+abcd"), &opts(), BTreeMap::new()))
            .unwrap();
        assert_ne!(up_v1, up_v2);
        let head = |g: &mut ProvenanceGraph, up: &str| {
            g.insert(ProvenanceNode::build(
                "knn", "compute knn", &BinaryVersion::parse("1.0.0+abcd"), &opts(),
                BTreeMap::from([("extract".to_string(), up.to_string())]),
            )).unwrap()
        };
        let head_a = head(&mut g, &up_v1);
        let head_b = head(&mut g, &up_v2);
        assert_ne!(
            g.hash(&head_a, ProvenanceFlags::VERSION_AWARE).unwrap(),
            g.hash(&head_b, ProvenanceFlags::VERSION_AWARE).unwrap(),
            "upstream version change must cascade to head"
        );
        let no_up = ProvenanceFlags(ProvenanceFlags::CONFIG_ONLY.0 & !ProvenanceFlags::UPSTREAM.0);
        assert_eq!(g.hash(&head_a, no_up).unwrap(), g.hash(&head_b, no_up).unwrap());
        // Under CONFIG_ONLY the upstream's version is ignored all the
        // way down, so the two heads agree: the relaxation cascades.
        assert_eq!(
            g.hash(&head_a, ProvenanceFlags::CONFIG_ONLY).unwrap(),
            g.hash(&head_b, ProvenanceFlags::CONFIG_ONLY).unwrap(),
        );
        assert_eq!(g.len(), 4, "both versions of the upstream stay while heads reference them");
    }

    /// The graph is closed: a node naming an address it does not hold
    /// is refused.
    #[test]
    fn insert_refuses_a_dangling_upstream() {
        let mut g = ProvenanceGraph::new();
        let err = g
            .insert(ProvenanceNode::build(
                "knn", "compute knn", &BinaryVersion::parse("1.0.0+abcd"), &opts(),
                BTreeMap::from([("extract".to_string(), "deadbeefdeadbeef".to_string())]),
            ))
            .unwrap_err();
        assert!(err.contains("deadbeefdeadbeef"), "{err}");
        assert!(g.is_empty());
    }

    /// A chain of 400 steps hashes in linear time with a shared subtree
    /// stored once, and pruning keeps what a root reaches.
    #[test]
    fn a_long_chain_is_linear_and_prunable() {
        let mut g = ProvenanceGraph::new();
        let binary = BinaryVersion::parse("1.2.3+abc");
        let mut prev: Option<Address> = None;
        let mut addresses = Vec::new();
        for i in 0..400 {
            let mut up = BTreeMap::new();
            if let Some(p) = prev.take() {
                up.insert(format!("s{}", i - 1), p);
            }
            let a = g.insert(ProvenanceNode::build(&format!("s{i}"), "compute x", &binary, &opts(), up)).unwrap();
            addresses.push(a.clone());
            prev = Some(a);
        }
        let head = addresses.last().unwrap();
        assert_eq!(g.len(), 400);
        assert_eq!(g.hash(head, ProvenanceFlags::STRICT).unwrap(), *head);
        assert!(g.hash(head, ProvenanceFlags::CONFIG_ONLY).is_some());
        assert_eq!(g.reachable(&addresses[9]).len(), 10);
        g.retain_reachable([addresses[199].as_str()]);
        assert_eq!(g.len(), 200);
    }

    /// Sidecar round-trip — write beside an artifact, read back, the
    /// root and every node it reaches come back. The cache layer's
    /// upstream cascade depends on this.
    #[test]
    fn sidecar_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let artifact = tmp.path().join("preds.slab");
        std::fs::write(&artifact, b"placeholder").unwrap();
        let mut g = ProvenanceGraph::new();
        let base = g.insert(node("1.0.0+abcd", opts())).unwrap();
        let head = g
            .insert(ProvenanceNode::build("keys", "compute keys", &BinaryVersion::parse("1.0.0+abcd"), &opts(), BTreeMap::from([("source".to_string(), base.clone())])))
            .unwrap();
        g.write_sidecar(&head, &artifact).unwrap();
        let sidecar_path = ProvenanceGraph::sidecar_path(&artifact);
        assert!(sidecar_path.exists(), "sidecar must be written at <artifact>.provenance.json");
        let sidecar = ProvenanceGraph::read_sidecar(&artifact).unwrap().expect("sidecar should be readable");
        assert_eq!(sidecar.root, head);
        assert_eq!(sidecar.nodes.len(), 2);
        let mut other = ProvenanceGraph::new();
        let root = sidecar.absorb_into(&mut other);
        assert_eq!(other.hash(&root, ProvenanceFlags::STRICT).unwrap(), head);
        assert_eq!(other.hash(&root, ProvenanceFlags::CONFIG_ONLY), g.hash(&head, ProvenanceFlags::CONFIG_ONLY));
    }

    /// Missing sidecar is `Ok(None)`, not an error — lets consumers
    /// fall through cleanly to the degenerate-from-file path for
    /// hand-curated dataset files that pre-date the sidecar
    /// convention.
    #[test]
    fn sidecar_absent_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let artifact = tmp.path().join("preds.slab");
        std::fs::write(&artifact, b"placeholder").unwrap();
        assert!(ProvenanceGraph::read_sidecar(&artifact).unwrap().is_none(), "absent sidecar must surface as Ok(None)");
    }

    /// Degenerate provenance captures path/size/mtime so two
    /// different files at the same path with different sizes
    /// produce different addresses; `for_input` takes that route when
    /// no sidecar exists and the sidecar route when one does.
    #[test]
    fn for_input_prefers_a_sidecar_and_degrades_to_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("a.slab");
        let b = tmp.path().join("b.slab");
        std::fs::write(&a, b"short").unwrap();
        std::fs::write(&b, b"a much longer payload, surely different size").unwrap();
        let pa = ProvenanceNode::degenerate_from_artifact(&a).unwrap();
        let pb = ProvenanceNode::degenerate_from_artifact(&b).unwrap();
        assert_ne!(pa.address(), pb.address(), "different files should produce different degenerate provenances");

        let mut g = ProvenanceGraph::new();
        let from_file = g.for_input(tmp.path(), tmp.path(), &a).unwrap();
        assert_eq!(from_file, pa.address());
        // Now give `a` a sidecar: the producer's node wins over the file.
        let mut producer = ProvenanceGraph::new();
        let root = producer.insert(node("1.0.0+abcd", opts())).unwrap();
        producer.write_sidecar(&root, &a).unwrap();
        let from_sidecar = g.for_input(tmp.path(), tmp.path(), &a).unwrap();
        assert_eq!(from_sidecar, root);
        assert!(g.contains(&root));
    }

    #[test]
    fn diff_reports_changed_components() {
        let mut g = ProvenanceGraph::new();
        let a = g.insert(node("1.0.1+abcd", opts())).unwrap();
        let b = g.insert(node("2.0.0+ffff", opts())).unwrap();
        let diffs = g.diff(&a, &b);
        let labels: Vec<String> = diffs.iter().map(|d| match d {
            ProvenanceDiff::Component { label, .. } => label.clone(),
            ProvenanceDiff::UpstreamChanged(id) => format!("upstream:{id}"),
        }).collect();
        assert!(labels.iter().any(|l| l == "binary_version_major"));
        assert!(labels.iter().any(|l| l == "binary_git_hash"));
    }

    /// **The default ignores the binary version.** Rebuilding the tool
    /// must not invalidate completed steps: a dataset whose base facet
    /// took nine hours to extract should not lose it because an
    /// unrelated command was recompiled.
    #[test]
    fn the_default_is_config_only() {
        assert_eq!(ProvenanceFlags::DEFAULT, ProvenanceFlags::CONFIG_ONLY);
        for ignored in [
            ProvenanceFlags::VERSION_MAJOR,
            ProvenanceFlags::VERSION_MINOR,
            ProvenanceFlags::VERSION_PATCH,
            ProvenanceFlags::GIT_HASH,
            ProvenanceFlags::DIRTY_FLAG,
        ] {
            assert!(!ProvenanceFlags::DEFAULT.contains(ignored));
        }
        for consulted in [
            ProvenanceFlags::STEP_ID,
            ProvenanceFlags::COMMAND_PATH,
            ProvenanceFlags::OPTIONS,
            ProvenanceFlags::UPSTREAM,
        ] {
            assert!(ProvenanceFlags::DEFAULT.contains(consulted));
        }
    }

    /// The name and the flags agree, and the name parses back — the
    /// CLI's `default_value` is a literal clap requires, so this is
    /// what keeps the two from drifting apart.
    #[test]
    fn the_default_name_round_trips() {
        assert_eq!(ProvenanceFlags::DEFAULT_NAME, "config-only");
        assert_eq!(
            ProvenanceFlags::parse(ProvenanceFlags::DEFAULT_NAME).unwrap(),
            ProvenanceFlags::DEFAULT
        );
    }

    /// **What `veks run` actually defaults to**, read out of the CLI
    /// definition rather than trusted. `default_value` has to be a
    /// literal, so it is the one spelling that can drift from
    /// [`ProvenanceFlags::DEFAULT_NAME`] without anything noticing.
    #[test]
    fn the_cli_default_is_the_named_default() {
        let literal = crate::pipeline::run_args_provenance_default();
        assert_eq!(literal, ProvenanceFlags::DEFAULT_NAME);
        assert_eq!(ProvenanceFlags::parse(literal).unwrap(), ProvenanceFlags::DEFAULT);
    }

    /// A rebuilt binary leaves a step fresh under the default and
    /// stale under `strict` — the behaviour change this default is.
    #[test]
    fn a_rebuild_is_fresh_by_default_and_stale_under_strict() {
        assert_eq!(
            hash_of("2.0.0+aaaaaaaa", opts(), ProvenanceFlags::DEFAULT),
            hash_of("2.0.1+bbbbbbbb", opts(), ProvenanceFlags::DEFAULT),
            "a rebuild must not invalidate a completed step"
        );
        assert_ne!(
            hash_of("2.0.0+aaaaaaaa", opts(), ProvenanceFlags::STRICT),
            hash_of("2.0.1+bbbbbbbb", opts(), ProvenanceFlags::STRICT),
            "strict still notices, for callers that ask for it"
        );
    }

    /// What the default *does* catch: a changed option.
    #[test]
    fn a_changed_option_is_still_stale_by_default() {
        let mut other = opts();
        other.insert("range".to_string(), "[0,999)".to_string());
        assert_ne!(
            hash_of("2.0.0+aaaaaaaa", opts(), ProvenanceFlags::DEFAULT),
            hash_of("2.0.0+aaaaaaaa", other, ProvenanceFlags::DEFAULT),
        );
    }
}
