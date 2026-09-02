// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Progress log for command stream pipelines.
//!
//! Tracks the status of each pipeline step in a persistent YAML file
//! (`.cache/.upstream.progress.yaml`) in the workspace cache directory.
//! This enables skip-if-fresh semantics: completed steps are skipped on
//! re-run unless their inputs have changed.
//!
//! Per-step staleness is governed by structured provenance — see
//! [`super::provenance`] for selector-driven hashing. Every record's
//! provenance is one address into the log's [`ProvenanceGraph`], a flat
//! table shared by the whole log, so the file is linear in the number
//! of steps however deep the `after:` chains run. A schema-5 log, which
//! nested each upstream's full map inside every dependent, is migrated
//! on load: the nested copies are skipped unparsed and the graph is
//! rebuilt from the records they duplicated.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::command::Status;
use super::provenance::{Address, BinaryVersion, ProvenanceFlags, ProvenanceGraph, ProvenanceNode};

/// Schema version for the progress log.
///
/// Bump this whenever cache key algorithms, segment naming conventions,
/// or other internal formats change. On load, if the stored version does
/// not match, all step records are cleared (the user is notified).
///
/// History:
/// - v3: fingerprint chains for DAG-based staleness
/// - v4: build_version in StepRecord, mtime-based staleness removed,
///   fingerprint now includes command build version
/// - v5: structured provenance replaces opaque fingerprint string;
///   staleness check is selector-driven (see `ProvenanceFlags`)
/// - v6: provenance stored as one flat, content-addressed node table
///   per log; records hold an address (v5 nested every upstream's full
///   map inside every dependent, which grew quadratically and became
///   unreadable past 64 chained steps). v5 logs are migrated, not cleared.
const PROGRESS_SCHEMA_VERSION: u32 = 6;

/// The last schema that nested each upstream's full provenance inside
/// every dependent record. A log at this version is migrated on load
/// rather than cleared: its records are intact, only their shape was
/// untenable.
const NESTED_PROVENANCE_SCHEMA_VERSION: u32 = 5;

/// Persistent progress log for a pipeline execution.
///
/// Stored as `.cache/.upstream.progress.yaml` in the workspace cache directory.
#[derive(Debug, Clone, Default)]
pub struct ProgressLog {
    /// Path to the progress file on disk.
    path: Option<PathBuf>,
    /// Schema version — used to auto-invalidate when cache key algorithms
    /// or internal formats change.
    pub schema_version: u32,
    /// Per-step execution records, keyed by step ID.
    pub steps: HashMap<String, StepRecord>,
    /// Every provenance node any record references, plus the current
    /// nodes built during a run. Pruned to what records reach on save.
    pub provenance: ProvenanceGraph,
    /// The schema version `load` migrated this log from, if any. The
    /// first `save` keeps the original file beside the rewritten one.
    migrated_from: Option<u32>,
}

/// The on-disk log: the graph, then the records by step id in sorted
/// order so the file diffs cleanly between runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProgressLogFile {
    schema_version: u32,
    #[serde(default, skip_serializing_if = "ProvenanceGraph::is_empty")]
    provenance: ProvenanceGraph,
    #[serde(default)]
    steps: BTreeMap<String, StepRecord>,
}

/// A schema-5 record's provenance read leniently: its own components,
/// and for each upstream the **snapshot** the record was built on — that
/// snapshot's own components and the names of *its* upstreams. Anything
/// nested deeper is skipped without being parsed, which is what keeps a
/// log with deep `after:` chains readable: the YAML parser refuses to
/// descend past 128 levels but skips any depth.
#[derive(Deserialize)]
struct NestedProvenance {
    step_id: String,
    command_path: String,
    binary_version_major: u32,
    binary_version_minor: u32,
    binary_version_patch: u32,
    #[serde(default)]
    binary_git_hash: String,
    #[serde(default)]
    binary_dirty: bool,
    #[serde(default)]
    options: BTreeMap<String, String>,
    #[serde(default)]
    upstream: BTreeMap<String, NestedSnapshot>,
}

/// One level of a schema-5 nested copy: the upstream as the dependent
/// saw it when it ran. Its own upstreams are named, their copies skipped.
#[derive(Deserialize)]
struct NestedSnapshot {
    step_id: String,
    command_path: String,
    binary_version_major: u32,
    binary_version_minor: u32,
    binary_version_patch: u32,
    #[serde(default)]
    binary_git_hash: String,
    #[serde(default)]
    binary_dirty: bool,
    #[serde(default)]
    options: BTreeMap<String, String>,
    #[serde(default)]
    upstream: BTreeMap<String, serde::de::IgnoredAny>,
}

impl NestedSnapshot {
    /// The node this snapshot describes, given the addresses of its
    /// upstreams.
    fn into_node(self, upstream: BTreeMap<String, Address>) -> ProvenanceNode {
        ProvenanceNode {
            step_id: self.step_id,
            command_path: self.command_path,
            binary_version_major: self.binary_version_major,
            binary_version_minor: self.binary_version_minor,
            binary_version_patch: self.binary_version_patch,
            binary_git_hash: self.binary_git_hash,
            binary_dirty: self.binary_dirty,
            options: self.options,
            upstream,
        }
    }
}

/// A schema-5 record: the same fields as [`StepRecord`] with the
/// nested provenance in place of an address.
#[derive(Deserialize)]
struct NestedStepRecord {
    status: Status,
    message: String,
    completed_at: DateTime<Utc>,
    elapsed_secs: f64,
    #[serde(default)]
    outputs: Vec<OutputRecord>,
    #[serde(default)]
    resolved_options: HashMap<String, String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    resource_summary: Option<ResourceSummary>,
    #[serde(default)]
    provenance: Option<NestedProvenance>,
}

/// A schema-5 log as read for migration.
#[derive(Deserialize)]
struct NestedProgressLogFile {
    #[serde(default)]
    steps: HashMap<String, NestedStepRecord>,
}

/// Just the schema version, so the loader can pick a reader without
/// descending into the records.
#[derive(Deserialize)]
struct SchemaProbe {
    #[serde(default)]
    schema_version: u32,
}

/// Resource consumption summary for a single step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSummary {
    /// Peak RSS in bytes observed during step execution.
    pub peak_rss_bytes: u64,
    /// CPU user time in seconds consumed during step execution.
    pub cpu_user_secs: f64,
    /// CPU system time in seconds consumed during step execution.
    pub cpu_system_secs: f64,
    /// Total bytes read from disk during step execution.
    pub io_read_bytes: u64,
    /// Total bytes written to disk during step execution.
    pub io_write_bytes: u64,
}

/// Record of a single step's execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// Status of the last execution.
    pub status: Status,
    /// Human-readable message from the last execution.
    pub message: String,
    /// Timestamp when the step completed.
    pub completed_at: DateTime<Utc>,
    /// Wall-clock elapsed time in seconds.
    pub elapsed_secs: f64,
    /// Output files produced and their sizes at completion.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<OutputRecord>,
    /// Resolved options that were used (for cache invalidation).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub resolved_options: HashMap<String, String>,
    /// Error detail if the step failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Resource consumption summary (peak RSS, CPU, I/O).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_summary: Option<ResourceSummary>,
    /// Address of this step's provenance node in the log's graph. The
    /// node captures every component (identity, binary version
    /// components, resolved options, upstream addresses) so the
    /// staleness hash can be recomputed under any [`ProvenanceFlags`]
    /// selector at check time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Address>,
}

/// Record of a single output artifact at completion time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputRecord {
    /// Path to the output file, relative to the workspace when it lies
    /// inside it (absolute otherwise), so the record reads the same from
    /// any working directory.
    pub path: String,
    /// File size in bytes at completion.
    pub size: u64,
    /// Modification time (as RFC 3339 timestamp).
    pub mtime: Option<String>,
}

impl ProgressLog {
    /// Create an empty progress log (in-memory only, no file backing).
    pub fn new() -> Self {
        ProgressLog {
            schema_version: PROGRESS_SCHEMA_VERSION,
            ..ProgressLog::default()
        }
    }

    /// Load a progress log from a file, or create a new one if the file
    /// does not exist.
    ///
    /// A log at the current schema is read as is. A log at
    /// [`NESTED_PROVENANCE_SCHEMA_VERSION`] is migrated: every record is
    /// kept and its provenance rebuilt from the records of its upstreams
    /// (see [`migrate_nested`]); the message in the second tuple element
    /// says so, and the first `save` keeps the original file beside the
    /// rewritten one. Any other stored version clears all step records,
    /// as before.
    pub fn load(path: &Path) -> Result<(Self, Option<String>), String> {
        let empty = || ProgressLog {
            path: Some(path.to_path_buf()),
            schema_version: PROGRESS_SCHEMA_VERSION,
            steps: HashMap::new(),
            provenance: ProvenanceGraph::new(),
            migrated_from: None,
        };
        if !path.exists() {
            return Ok((empty(), None));
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read progress log {}: {}", path.display(), e))?;
        let parse_err = |e: serde_yaml::Error| {
            format!("Failed to parse progress log {}: {}", path.display(), e)
        };
        let probe: SchemaProbe = serde_yaml::from_str(&content).map_err(parse_err)?;
        if probe.schema_version == PROGRESS_SCHEMA_VERSION {
            let file: ProgressLogFile = serde_yaml::from_str(&content).map_err(parse_err)?;
            for (id, record) in &file.steps {
                if let Some(address) = &record.provenance
                    && !file.provenance.contains(address)
                {
                    return Err(format!(
                        "Failed to parse progress log {}: step '{}' names provenance {} which the log does not hold",
                        path.display(),
                        id,
                        address
                    ));
                }
            }
            return Ok((
                ProgressLog {
                    steps: file.steps.into_iter().collect(),
                    provenance: file.provenance,
                    ..empty()
                },
                None,
            ));
        }
        if probe.schema_version == NESTED_PROVENANCE_SCHEMA_VERSION {
            let file: NestedProgressLogFile = serde_yaml::from_str(&content).map_err(parse_err)?;
            let count = file.steps.len();
            let mut graph = ProvenanceGraph::new();
            let workspace = path.parent().and_then(Path::parent);
            let (steps, migration) = migrate_nested(file.steps, &mut graph, workspace);
            let mut msg = format!(
                "Progress log migrated: schema version {} → {}, {} step record(s) kept, provenance now stored flat",
                NESTED_PROVENANCE_SCHEMA_VERSION, PROGRESS_SCHEMA_VERSION, count,
            );
            if !migration.earlier_upstream.is_empty() {
                msg.push_str(&format!(
                    "; {} record(s) were built on an earlier version of an upstream and keep that snapshot: {}",
                    migration.earlier_upstream.len(),
                    migration.earlier_upstream.join(", "),
                ));
            }
            if migration.outputs_measured > 0 {
                msg.push_str(&format!(
                    "; {} sharded output(s) recorded under their logical name were re-measured as their shards",
                    migration.outputs_measured,
                ));
            }
            return Ok((
                ProgressLog {
                    steps,
                    provenance: graph,
                    migrated_from: Some(NESTED_PROVENANCE_SCHEMA_VERSION),
                    ..empty()
                },
                Some(msg),
            ));
        }
        // Any other stored version: the cache key algorithms or internal
        // formats may have changed. Clear all records so steps re-run.
        let count = serde_yaml::from_str::<NestedProgressLogFile>(&content)
            .map(|f| f.steps.len())
            .unwrap_or(0);
        Ok((
            empty(),
            Some(format!(
                "Progress log invalidated: schema version changed ({} → {}), {} step records cleared",
                probe.schema_version, PROGRESS_SCHEMA_VERSION, count,
            )),
        ))
    }

    /// The schema version this log was migrated from on load, if any.
    pub fn migrated_from(&self) -> Option<u32> {
        self.migrated_from
    }

    /// Build the provenance node of a step as it would run now, insert
    /// it into the graph and return its address. Each upstream
    /// contributes the address its record holds; a missing or legacy
    /// record contributes [`ProvenanceNode::unknown`], a known-distinct
    /// sentinel under any selector that includes `UPSTREAM`.
    pub fn build_provenance(
        &mut self,
        step_id: &str,
        command_path: &str,
        resolved_options: &HashMap<String, String>,
        upstream_ids: &[&str],
        build_version: &str,
    ) -> Address {
        let binary = BinaryVersion::parse(build_version);
        let mut upstream: BTreeMap<String, Address> = BTreeMap::new();
        for up_id in upstream_ids {
            let recorded = self
                .steps
                .get(*up_id)
                .and_then(|r| r.provenance.clone())
                .filter(|a| self.provenance.contains(a));
            let address = match recorded {
                Some(a) => a,
                None => self
                    .provenance
                    .insert(ProvenanceNode::unknown(up_id))
                    .expect("a node without upstreams always inserts"),
            };
            upstream.insert((*up_id).to_string(), address);
        }
        let node = ProvenanceNode::build(step_id, command_path, &binary, resolved_options, upstream);
        self.provenance
            .insert(node)
            .expect("every upstream address was just taken from or put into this graph")
    }

    /// Check whether a step's recorded provenance matches `current`
    /// (an address in this log's graph, from [`build_provenance`](Self::build_provenance))
    /// under `selector`. Returns `Some(reason)` if stale, `None` if
    /// fresh.
    pub fn check_provenance(
        &self,
        step_id: &str,
        current: &str,
        selector: ProvenanceFlags,
    ) -> Option<String> {
        match self.steps.get(step_id) {
            Some(record) => match record.provenance.as_deref() {
                Some(stored) => {
                    let stored_hash = self.provenance.hash(stored, selector);
                    let current_hash = self.provenance.hash(current, selector);
                    if stored_hash.is_some() && stored_hash == current_hash {
                        None
                    } else {
                        let diffs = self.provenance.diff(current, stored);
                        let summary: Vec<String> =
                            diffs.iter().take(3).map(|d| d.to_string()).collect();
                        let extra = if diffs.len() > 3 {
                            format!(" (+{} more)", diffs.len() - 3)
                        } else {
                            String::new()
                        };
                        Some(if summary.is_empty() {
                            format!(
                                "provenance changed under selector '{}'",
                                selector.describe()
                            )
                        } else {
                            format!("provenance changed: {}{}", summary.join("; "), extra)
                        })
                    }
                }
                None => None, // legacy record without provenance — trust it
            },
            None => Some("not recorded".to_string()),
        }
    }

    /// Derive the progress log path from a dataset.yaml path.
    ///
    /// Returns `<dir>/.cache/.upstream.progress.yaml` where `<dir>` is the
    /// directory containing the dataset file. The progress log lives in the
    /// cache directory because it is an expensive-to-recompute workspace
    /// artifact, not a publishable dataset file.
    ///
    /// If a progress log exists at the old location (`<dir>/.upstream.progress.yaml`)
    /// and not at the new location, it is migrated automatically.
    pub fn path_for_dataset(dataset_path: &Path) -> PathBuf {
        let dir = dataset_path.parent().unwrap_or(Path::new("."));
        let new_path = dir.join(".cache").join(".upstream.progress.yaml");
        let old_path = dir.join(".upstream.progress.yaml");

        // Migrate from old location if needed
        if old_path.exists() {
            if !new_path.exists() {
                let cache_dir = dir.join(".cache");
                if std::fs::create_dir_all(&cache_dir).is_ok()
                    && std::fs::rename(&old_path, &new_path).is_ok() {
                        log::info!(
                            "Migrated progress log: {} → {}",
                            old_path.display(),
                            new_path.display(),
                        );
                    }
            } else {
                let _ = std::fs::remove_file(&old_path);
            }
        }

        new_path
    }

    /// Record a step's execution result.
    ///
    /// Side-effect: writes a co-located
    /// `<output>.provenance.json` sidecar for every `OutputRecord`
    /// path on the record, *if* the step has a provenance map.
    /// Sidecars are best-effort: failures are logged at warn level
    /// but do not abort the record. Consumers downstream rely on
    /// these sidecars to populate their own `upstream` cascade
    /// (see [`ProvenanceGraph::read_sidecar`]).
    pub fn record_step(&mut self, step_id: &str, record: StepRecord) {
        self.write_sidecars(&record);
        self.steps.insert(step_id.to_string(), record);
    }

    /// Write the provenance sidecar of every output of `record` under
    /// `<cache>/provenance/`, when the record has a provenance address
    /// and this log is backed by a file (the cache directory is the
    /// log's own parent; a purely in-memory log has nowhere to persist
    /// them).
    fn write_sidecars(&self, record: &StepRecord) {
        // Provenance sidecars for dataset outputs are staleness metadata,
        // not dataset content — they belong under `<cache>/provenance/`,
        // never beside the artifact.
        let (Some(root), Some(cache_dir)) =
            (record.provenance.as_deref(), self.path.as_deref().and_then(Path::parent))
        else {
            return;
        };
        let workspace = cache_dir.parent();
        for out in &record.outputs {
            let artifact = Path::new(&out.path);
            // A derived sidecar (a `.mref` the merkle step just produced,
            // or a provenance sidecar) must not itself get a provenance
            // sidecar, or the suffixes compound.
            let is_sidecar = artifact
                .file_name()
                .and_then(|n| n.to_str())
                .map(veks_core::filters::is_derived_sidecar)
                .unwrap_or(false);
            if is_sidecar {
                continue;
            }
            // Map to a workspace-relative path so the provenance
            // mirror under `<cache>/provenance/` stays clean (the
            // runner may record outputs as absolute or relative paths).
            let rel = workspace
                .and_then(|ws| artifact.strip_prefix(ws).ok())
                .unwrap_or(artifact);
            // Migrate away any sidecar a previous version wrote beside
            // the artifact — resolve the artifact to absolute first so a
            // relative output path still points at the dataset file.
            let abs_artifact = match (artifact.is_absolute(), workspace) {
                (false, Some(ws)) => ws.join(artifact),
                _ => artifact.to_path_buf(),
            };
            let stale = ProvenanceGraph::sidecar_path(&abs_artifact);
            if stale.exists() {
                let _ = std::fs::remove_file(&stale);
            }
            if let Err(e) = self.provenance.write_cached_sidecar(root, cache_dir, rel) {
                log::warn!(
                    "provenance: failed to write sidecar for {}: {}",
                    out.path, e,
                );
            }
        }
    }

    /// Rewrite the sidecars of every recorded step's outputs, returning
    /// how many records were visited. Used after a migration so the
    /// sidecars match the log they mirror.
    pub fn rewrite_sidecars(&self) -> usize {
        let mut visited = 0;
        for record in self.steps.values() {
            if record.provenance.is_some() {
                self.write_sidecars(record);
                visited += 1;
            }
        }
        visited
    }

    /// Update the recorded output size for a file across all steps.
    ///
    /// When a downstream step modifies a file that was produced by an
    /// upstream step (e.g., overlap removal rewrites query_vectors.fvec),
    /// the stored size becomes stale. This updates ALL step records that
    /// reference the given path so the freshness check passes.
    pub fn update_output_size(&mut self, path: &str, new_size: u64) {
        for record in self.steps.values_mut() {
            for output in &mut record.outputs {
                if output.path == path {
                    output.size = new_size;
                }
            }
        }
    }

    /// Remove a step's progress record, forcing it to re-execute.
    pub fn clear_step(&mut self, step_id: &str) {
        self.steps.remove(step_id);
    }

    /// Get the record for a step, if any.
    pub fn get_step(&self, step_id: &str) -> Option<&StepRecord> {
        self.steps.get(step_id)
    }

    /// Check whether the recorded outputs for a step still match disk state
    /// and the resolved options haven't changed.
    pub fn is_step_fresh(
        &self,
        step_id: &str,
        current_options: Option<&HashMap<String, String>>,
    ) -> bool {
        self.check_step_freshness(step_id, current_options, None).is_none()
    }

    /// Check whether a step needs to be re-run, returning a reason if stale.
    ///
    /// Returns `None` if the step is fresh (locally), or `Some(reason)`
    /// describing why it is stale (options changed, outputs missing/corrupted,
    /// etc.). Upstream / build / config staleness is handled by
    /// `check_provenance` — not by this method.
    pub fn check_step_freshness(
        &self,
        step_id: &str,
        current_options: Option<&HashMap<String, String>>,
        workspace: Option<&Path>,
    ) -> Option<String> {
        let record = match self.get_step(step_id) {
            Some(r) if r.status == Status::Ok => r,
            _ => return Some("not recorded or failed".to_string()),
        };

        // If the step was skipped by the bound checker (never actually ran),
        // don't trust the record — force the bound checker to re-validate.
        if record.elapsed_secs == 0.0 && record.outputs.is_empty() {
            return Some("previous run was a bound-check skip, re-validating".to_string());
        }

        if let Some(current) = current_options
            && !record.resolved_options.is_empty() && &record.resolved_options != current {
                return Some("options changed".to_string());
            }

        for output in &record.outputs {
            let path = resolve_path(&output.path, workspace);
            let filename = std::path::Path::new(&output.path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            let is_catalog = filename == "catalog.json" || filename == "catalog.yaml";
            match std::fs::metadata(&path) {
                Ok(meta) => {
                    if !is_catalog && meta.len() != output.size {
                        return Some(format!(
                            "output '{}' size changed ({} → {})",
                            output.path, output.size, meta.len()
                        ));
                    }
                }
                Err(_) => return Some(format!("output '{}' missing", output.path)),
            }
        }

        None
    }

    /// Persist the progress log to disk in the current schema, with the
    /// graph pruned to the nodes the records reach.
    ///
    /// Writes atomically by writing to a temp file and renaming. When
    /// the log was migrated on load, the original file is kept once as
    /// `<name>.v<schema>.yaml` beside it before the first overwrite.
    pub fn save(&self) -> Result<(), String> {
        let path = self.path.as_ref().ok_or("progress log has no file path")?;
        if let Some(from) = self.migrated_from {
            let backup = path.with_extension(format!("v{}.yaml", from));
            if path.exists() && !backup.exists() {
                std::fs::copy(path, &backup).map_err(|e| {
                    format!("Failed to keep {} before rewriting: {}", backup.display(), e)
                })?;
            }
        }
        let mut provenance = self.provenance.clone();
        provenance.retain_reachable(self.steps.values().filter_map(|r| r.provenance.as_deref()));
        let file = ProgressLogFile {
            schema_version: self.schema_version,
            provenance,
            steps: self.steps.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        };
        let content = serde_yaml::to_string(&file)
            .map_err(|e| format!("Failed to serialize progress log: {}", e))?;
        let tmp_path = path.with_extension("yaml.tmp");
        std::fs::write(&tmp_path, &content)
            .map_err(|e| format!("Failed to write {}: {}", tmp_path.display(), e))?;
        std::fs::rename(&tmp_path, path)
            .map_err(|e| format!("Failed to rename progress log: {}", e))?;
        Ok(())
    }

}

/// FNV-1a 64-bit hasher for deterministic fingerprinting.
///
/// What [`migrate_nested`] found worth reporting.
#[derive(Debug, Default)]
struct NestedMigration {
    /// Records whose nested copy of an upstream differs from that
    /// upstream's own record — they were built on an earlier version of
    /// it, and keep that version as their upstream node.
    earlier_upstream: Vec<String>,
    /// Outputs that schema 5 recorded under a series' logical name with
    /// no size, now recorded as the shards they resolve to.
    outputs_measured: usize,
}

/// Rebuild schema-5 records into `graph`.
///
/// A record's provenance carried, for each upstream, a full copy of the
/// upstream's provenance **as it was when the record ran**. The copy's
/// own components are read (one level; deeper copies are skipped) and
/// become the upstream node the record references, with *that* node's
/// upstreams taken from the current records of the steps it names. When
/// the copy matches the upstream's own record it is the same node —
/// same content, same address — and nothing is duplicated; when it
/// differs, both versions live in the graph, so the record hashes
/// exactly as it did before under every selector: fresh under one that
/// ignores what changed, stale under one that does not.
///
/// Schema 5 also recorded a sharded output under its logical name with
/// size 0 and no mtime, because the logical file never existed. Such an
/// output is re-measured as the shards it resolves to — what a current
/// run records — so the freshness check can see it.
fn migrate_nested(
    records: HashMap<String, NestedStepRecord>,
    graph: &mut ProvenanceGraph,
    workspace: Option<&Path>,
) -> (HashMap<String, StepRecord>, NestedMigration) {
    let mut report = NestedMigration::default();

    // Pass 1: every record's own node, with its upstream copies set
    // aside, so a copy can be compared against the record it copied.
    let mut own: HashMap<String, NestedProvenance> = HashMap::new();
    let mut plain: HashMap<String, StepRecord> = HashMap::new();
    for (id, record) in records {
        let outputs = measure_series_outputs(record.outputs, workspace, &mut report.outputs_measured);
        if let Some(p) = record.provenance {
            own.insert(id.clone(), p);
        }
        plain.insert(
            id,
            StepRecord {
                status: record.status,
                message: record.message,
                completed_at: record.completed_at,
                elapsed_secs: record.elapsed_secs,
                outputs,
                resolved_options: record.resolved_options,
                error: record.error,
                resource_summary: record.resource_summary,
                provenance: None,
            },
        );
    }

    /// The node of step `id` as its own record describes it, built
    /// upstream-first so every address it names is in the graph.
    fn current_node(
        id: &str,
        own: &HashMap<String, NestedProvenance>,
        graph: &mut ProvenanceGraph,
        memo: &mut HashMap<String, Address>,
        visiting: &mut Vec<String>,
    ) -> Option<Address> {
        if let Some(done) = memo.get(id) {
            return Some(done.clone());
        }
        let nested = own.get(id)?;
        if visiting.iter().any(|v| v == id) {
            return None;
        }
        visiting.push(id.to_string());
        let mut upstream = BTreeMap::new();
        for up_id in nested.upstream.keys() {
            let address = current_node(up_id, own, graph, memo, visiting).unwrap_or_else(|| {
                graph
                    .insert(ProvenanceNode::unknown(up_id))
                    .expect("a node without upstreams always inserts")
            });
            upstream.insert(up_id.clone(), address);
        }
        visiting.pop();
        let node = ProvenanceNode {
            step_id: nested.step_id.clone(),
            command_path: nested.command_path.clone(),
            binary_version_major: nested.binary_version_major,
            binary_version_minor: nested.binary_version_minor,
            binary_version_patch: nested.binary_version_patch,
            binary_git_hash: nested.binary_git_hash.clone(),
            binary_dirty: nested.binary_dirty,
            options: nested.options.clone(),
            upstream,
        };
        let address = graph
            .insert(node)
            .expect("every upstream address was just put into this graph");
        memo.insert(id.to_string(), address.clone());
        Some(address)
    }

    // Pass 2: each record's own node — the same as pass 1's, except
    // that each upstream is the snapshot the record carried rather than
    // the upstream's current node.
    let mut memo: HashMap<String, Address> = HashMap::new();
    let ids: Vec<String> = own.keys().cloned().collect();
    let mut snapshots: Vec<(String, BTreeMap<String, Address>)> = Vec::new();
    for id in &ids {
        let nested = &own[id];
        let mut upstream = BTreeMap::new();
        let mut earlier = false;
        for (up_id, copy) in &nested.upstream {
            // The snapshot's own upstreams: the current nodes of the
            // steps it names (its deeper copies were not read).
            let mut copy_upstream = BTreeMap::new();
            for deeper in copy.upstream.keys() {
                let address = current_node(deeper, &own, graph, &mut memo, &mut Vec::new())
                    .unwrap_or_else(|| {
                        graph
                            .insert(ProvenanceNode::unknown(deeper))
                            .expect("a node without upstreams always inserts")
                    });
                copy_upstream.insert(deeper.clone(), address);
            }
            let snapshot = ProvenanceNode {
                step_id: copy.step_id.clone(),
                command_path: copy.command_path.clone(),
                binary_version_major: copy.binary_version_major,
                binary_version_minor: copy.binary_version_minor,
                binary_version_patch: copy.binary_version_patch,
                binary_git_hash: copy.binary_git_hash.clone(),
                binary_dirty: copy.binary_dirty,
                options: copy.options.clone(),
                upstream: copy_upstream,
            };
            let address = graph
                .insert(snapshot)
                .expect("every upstream address was just put into this graph");
            let current = current_node(up_id, &own, graph, &mut memo, &mut Vec::new());
            if current.as_deref() != Some(address.as_str()) {
                earlier = true;
            }
            upstream.insert(up_id.clone(), address);
        }
        if earlier {
            report.earlier_upstream.push(id.clone());
        }
        snapshots.push((id.clone(), upstream));
    }
    for (id, upstream) in snapshots {
        let nested = &own[&id];
        let node = ProvenanceNode {
            step_id: nested.step_id.clone(),
            command_path: nested.command_path.clone(),
            binary_version_major: nested.binary_version_major,
            binary_version_minor: nested.binary_version_minor,
            binary_version_patch: nested.binary_version_patch,
            binary_git_hash: nested.binary_git_hash.clone(),
            binary_dirty: nested.binary_dirty,
            options: nested.options.clone(),
            upstream,
        };
        let address = graph
            .insert(node)
            .expect("every upstream address was just put into this graph");
        if let Some(record) = plain.get_mut(&id) {
            record.provenance = Some(address);
        }
    }
    report.earlier_upstream.sort();
    (plain, report)
}

/// Replace an output recorded under a series' logical name with no
/// measured size by the shards it resolves to, each with its size —
/// what a current run records. Any other output is kept as it is.
fn measure_series_outputs(
    outputs: Vec<OutputRecord>,
    workspace: Option<&Path>,
    measured: &mut usize,
) -> Vec<OutputRecord> {
    let mut out = Vec::with_capacity(outputs.len());
    for output in outputs {
        let unmeasured = output.size == 0 && output.mtime.is_none();
        let full = resolve_path(&output.path, workspace);
        if !unmeasured || full.exists() {
            out.push(output);
            continue;
        }
        let shards = vectordata::dataset::shards::discover_shards(&full);
        if shards.is_empty() {
            out.push(output);
            continue;
        }
        *measured += 1;
        let base = Path::new(&output.path);
        for shard in shards {
            let Ok(meta) = std::fs::metadata(&shard) else { continue };
            // Keep the record's own form of the path (relative or
            // absolute) for the shard, as the runner would have.
            let path = match (base.parent(), shard.file_name()) {
                (Some(dir), Some(name)) if !dir.as_os_str().is_empty() => {
                    dir.join(name).to_string_lossy().into_owned()
                }
                (_, Some(name)) => name.to_string_lossy().into_owned(),
                _ => shard.to_string_lossy().into_owned(),
            };
            let mtime = meta
                .modified()
                .ok()
                .map(|m| DateTime::<Utc>::from(m).to_rfc3339());
            out.push(OutputRecord { path, size: meta.len(), mtime });
        }
    }
    out
}

/// Used by [`super::provenance`] to compute selector-driven staleness
/// hashes. No external dependency.
pub struct FnvHasher {
    state: u64,
}

impl Default for FnvHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl FnvHasher {
    pub fn new() -> Self {
        FnvHasher { state: 0xcbf29ce484222325 } // FNV offset basis
    }

    pub fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= byte as u64;
            self.state = self.state.wrapping_mul(0x100000001b3); // FNV prime
        }
    }

    pub fn finish(&self) -> u64 {
        self.state
    }
}

fn resolve_path(value: &str, workspace: Option<&Path>) -> PathBuf {
    let p = Path::new(value);
    if p.is_absolute() {
        p.to_path_buf()
    } else if let Some(ws) = workspace {
        ws.join(p)
    } else {
        p.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(provenance: Option<Address>) -> StepRecord {
        StepRecord {
            status: Status::Ok,
            message: "done".into(),
            completed_at: Utc::now(),
            elapsed_secs: 1.0,
            outputs: vec![],
            resolved_options: HashMap::new(),
            error: None,
            resource_summary: None,
            provenance,
        }
    }

    #[test]
    fn test_progress_log_path() {
        let path = ProgressLog::path_for_dataset(Path::new("/data/my-dataset/dataset.yaml"));
        assert_eq!(
            path,
            PathBuf::from("/data/my-dataset/.cache/.upstream.progress.yaml")
        );
    }

    #[test]
    fn test_record_and_check() {
        let mut log = ProgressLog::new();
        assert!(log.get_step("step1").is_none());
        log.record_step("step1", rec(None));
        assert!(log.get_step("step1").is_some());
        assert!(log.get_step("step2").is_none());
    }

    /// `record_step` writes a co-located `<output>.provenance.json`
    /// for every `OutputRecord` path when the record has a
    /// provenance. This is the producer half of the
    /// upstream-cascade contract: downstream consumers find the
    /// sidecar via [`ProvenanceGraph::read_sidecar`] and merge it
    /// into their own `upstream` map.
    /// A progress log wired to a dataset's `.cache/.upstream.progress.yaml`,
    /// so `record_step` can derive the cache dir for provenance placement.
    fn log_for_dataset(dataset: &Path) -> ProgressLog {
        let cache = dataset.join(".cache");
        std::fs::create_dir_all(&cache).unwrap();
        let mut log = ProgressLog::new();
        log.path = Some(cache.join(".upstream.progress.yaml"));
        log
    }

    #[test]
    fn record_step_writes_sidecars_under_cache_provenance() {
        let ds = tempfile::tempdir().unwrap();
        let cache = ds.path().join(".cache");
        let mut log = log_for_dataset(ds.path());
        let address = log.build_provenance("test-step", "test command", &HashMap::new(), &[], "1.0.0+abcd");
        let mut r = rec(Some(address.clone()));
        // Workspace-relative output path (as the runner records it).
        let rel = "profiles/base/out.slab";
        r.outputs.push(OutputRecord { path: rel.to_string(), size: 12, mtime: None });
        log.record_step("test-step", r);
        // Sidecar lands under <cache>/provenance/, mirroring the artifact
        // path — and NOT beside the dataset artifact.
        let cached = ProvenanceGraph::cached_sidecar_path(&cache, Path::new(rel));
        assert!(cached.exists(), "sidecar should be at {}", cached.display());
        assert!(!ds.path().join("profiles/base/out.slab.provenance.json").exists(),
            "provenance must not pollute the dataset storage layer");
        // …and carries the node under its address.
        let recovered = ProvenanceGraph::read_cached_sidecar(&cache, Path::new(rel))
            .unwrap().unwrap();
        assert_eq!(recovered.root, address);
        assert_eq!(recovered.nodes.get(&address), log.provenance.get(&address));
    }

    /// A stale sidecar a previous version wrote *beside* the artifact is
    /// migrated away (removed) on the next `record_step`.
    #[test]
    fn record_step_migrates_legacy_colocated_sidecar() {
        let ds = tempfile::tempdir().unwrap();
        let rel = "profiles/base/out.slab";
        let legacy = ds.path().join("profiles/base/out.slab.provenance.json");
        std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        std::fs::write(&legacy, b"{}").unwrap();
        let mut log = log_for_dataset(ds.path());
        let address = log.build_provenance("s", "c", &HashMap::new(), &[], "1.0.0+abcd");
        let mut r = rec(Some(address));
        r.outputs.push(OutputRecord { path: rel.to_string(), size: 1, mtime: None });
        log.record_step("s", r);
        assert!(!legacy.exists(), "legacy co-located sidecar must be removed");
        assert!(ProvenanceGraph::cached_sidecar_path(&ds.path().join(".cache"), Path::new(rel))
            .exists(), "new sidecar must be under cache/provenance");
    }

    /// A derived sidecar output (a `.mref` from the merkle step, or a provenance
    /// sidecar) must NOT itself get a provenance sidecar — otherwise suffixes
    /// compound without bound (`…fvecs.provenance.json.mref.provenance.json`).
    #[test]
    fn record_step_skips_sidecar_for_derived_outputs() {
        let ds = tempfile::tempdir().unwrap();
        let cache = ds.path().join(".cache");
        for derived in ["base.fvec.mref", "base.fvec.provenance.json", "data.mrkl"] {
            let mut log = log_for_dataset(ds.path());
            let address = log.build_provenance("merkle-step", "merkle create", &HashMap::new(), &[], "1.0.0+abcd");
            let mut r = rec(Some(address));
            r.outputs.push(OutputRecord { path: derived.to_string(), size: 7, mtime: None });
            log.record_step("merkle-step", r);
            assert!(
                !ProvenanceGraph::cached_sidecar_path(&cache, Path::new(derived)).exists(),
                "{derived} must not get a provenance sidecar"
            );
        }
    }

    /// Producer steps without a provenance (legacy records or
    /// commands that haven't been wired up yet) must NOT crash on
    /// `record_step`, and write no sidecar.
    #[test]
    fn record_step_without_provenance_does_not_write_sidecar() {
        let ds = tempfile::tempdir().unwrap();
        let rel = "legacy.slab";
        let mut r = rec(None);
        r.outputs.push(OutputRecord { path: rel.to_string(), size: 12, mtime: None });
        let mut log = log_for_dataset(ds.path());
        log.record_step("legacy-step", r);
        assert!(!ProvenanceGraph::cached_sidecar_path(&ds.path().join(".cache"), Path::new(rel))
            .exists(), "no provenance → no sidecar (best-effort, not mandatory)");
    }

    #[test]
    fn test_roundtrip() {
        let dir = scratch_dir();
        let path = dir.path().join(".cache/.upstream.progress.yaml");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut log = ProgressLog::load(&path).unwrap().0;
        let mut r = rec(None);
        r.outputs.push(OutputRecord {
            path: "output.fvec".into(),
            size: 1024,
            mtime: None,
        });
        log.record_step("step1", r);
        log.save().unwrap();
        let parsed = ProgressLog::load(&path).unwrap().0;
        assert!(parsed.get_step("step1").is_some());
        assert_eq!(parsed.steps["step1"].outputs.len(), 1);
    }

    fn scratch_dir() -> tempfile::TempDir {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("target/tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    /// A chain of `n` steps, each `after` the previous, recorded the
    /// way the runner records them.
    fn record_chain(log: &mut ProgressLog, n: usize, build: &str) {
        for i in 0..n {
            let id = format!("knn-{i}");
            let upstream: Vec<String> = if i == 0 { vec![] } else { vec![format!("knn-{}", i - 1)] };
            let ups: Vec<&str> = upstream.iter().map(String::as_str).collect();
            let mut opts = HashMap::new();
            opts.insert("k".to_string(), "100".to_string());
            opts.insert("profile".to_string(), id.clone());
            let address = log.build_provenance(&id, "compute knn", &opts, &ups, build);
            let mut r = rec(Some(address));
            r.resolved_options = opts;
            log.record_step(&id, r);
        }
    }

    /// The whole point: a chain far deeper than the YAML parser's
    /// 128-level limit round-trips through disk with every hash intact,
    /// and the file stays flat and linear.
    #[test]
    fn a_deep_chain_round_trips_flat() {
        let dir = scratch_dir();
        let path = dir.path().join(".cache/.upstream.progress.yaml");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut log = ProgressLog::load(&path).unwrap().0;
        record_chain(&mut log, 300, "2.0.0+abc");
        log.save().unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let deepest = text
            .lines()
            .map(|l| l.len() - l.trim_start().len())
            .max()
            .unwrap();
        assert!(deepest <= 8, "nesting on disk must be constant, saw indent {deepest}");
        assert!(text.len() < 300 * 2_000, "file must be linear in steps, was {} bytes", text.len());

        let (back, msg) = ProgressLog::load(&path).unwrap();
        assert!(msg.is_none(), "{msg:?}");
        assert_eq!(back.steps.len(), 300);
        assert_eq!(back.provenance.len(), 300, "one node per step, nothing duplicated");
        for (id, record) in &log.steps {
            let a = record.provenance.as_deref().unwrap();
            let b = back.steps[id].provenance.as_deref().unwrap();
            assert_eq!(a, b, "{id}");
            assert_eq!(
                log.provenance.hash(a, ProvenanceFlags::CONFIG_ONLY),
                back.provenance.hash(b, ProvenanceFlags::CONFIG_ONLY)
            );
        }
        // The head is fresh against a provenance built from the reloaded log.
        let mut back = back;
        let head = "knn-299";
        let opts = back.steps[head].resolved_options.clone();
        let current = back.build_provenance(head, "compute knn", &opts, &["knn-298"], "2.0.0+abc");
        assert!(back.check_provenance(head, &current, ProvenanceFlags::STRICT).is_none());
    }

    /// The schema-5 shape a log was written in before: each record's
    /// provenance nested in full.
    #[derive(Serialize, Deserialize)]
    struct NestedMapForTest {
        step_id: String,
        command_path: String,
        binary_version_major: u32,
        binary_version_minor: u32,
        binary_version_patch: u32,
        binary_git_hash: String,
        binary_dirty: bool,
        options: BTreeMap<String, String>,
        upstream: BTreeMap<String, NestedMapForTest>,
    }

    fn nest(graph: &ProvenanceGraph, address: &str) -> NestedMapForTest {
        let n = graph.get(address).unwrap();
        NestedMapForTest {
            step_id: n.step_id.clone(),
            command_path: n.command_path.clone(),
            binary_version_major: n.binary_version_major,
            binary_version_minor: n.binary_version_minor,
            binary_version_patch: n.binary_version_patch,
            binary_git_hash: n.binary_git_hash.clone(),
            binary_dirty: n.binary_dirty,
            options: n.options.clone(),
            upstream: n.upstream.iter().map(|(id, a)| (id.clone(), nest(graph, a))).collect(),
        }
    }

    #[derive(Serialize)]
    struct NestedRecordForTest {
        status: Status,
        message: String,
        completed_at: DateTime<Utc>,
        elapsed_secs: f64,
        outputs: Vec<OutputRecord>,
        resolved_options: HashMap<String, String>,
        provenance: Option<NestedMapForTest>,
    }

    #[derive(Serialize)]
    struct NestedLogForTest {
        schema_version: u32,
        steps: BTreeMap<String, NestedRecordForTest>,
    }

    /// `log` in the shape schema 5 wrote.
    fn nested_log_for_test(log: &ProgressLog) -> NestedLogForTest {
        let steps = log
            .steps
            .iter()
            .map(|(id, r)| {
                (
                    id.clone(),
                    NestedRecordForTest {
                        status: r.status.clone(),
                        message: r.message.clone(),
                        completed_at: r.completed_at,
                        elapsed_secs: r.elapsed_secs,
                        outputs: r.outputs.clone(),
                        resolved_options: r.resolved_options.clone(),
                        provenance: r.provenance.as_deref().map(|a| nest(&log.provenance, a)),
                    },
                )
            })
            .collect();
        NestedLogForTest { schema_version: 5, steps }
    }

    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct NestedLogReadBack {
        steps: BTreeMap<String, serde_yaml::Value>,
    }

    /// Write `log` to `path` the way schema 5 did.
    fn write_nested_v5(log: &ProgressLog, path: &Path) {
        std::fs::write(path, serde_yaml::to_string(&nested_log_for_test(log)).unwrap()).unwrap();
    }

    /// A schema-5 log with a chain deeper than the parser limit is
    /// migrated with every record kept, every node rebuilt at the same
    /// address, the original kept beside the rewrite on first save.
    #[test]
    fn a_nested_v5_log_deeper_than_the_parser_limit_is_migrated() {
        let dir = scratch_dir();
        let path = dir.path().join(".cache/.upstream.progress.yaml");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut log = ProgressLog::load(&path).unwrap().0;
        record_chain(&mut log, 200, "2.0.0+abc");
        write_nested_v5(&log, &path);
        // Sanity: the nested form really is past the parser's limit.
        let err = serde_yaml::from_str::<NestedLogReadBack>(&std::fs::read_to_string(&path).unwrap())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("recursion limit"), "{err}");

        let (migrated, msg) = ProgressLog::load(&path).unwrap();
        let msg = msg.unwrap();
        assert!(msg.contains("migrated"), "{msg}");
        assert!(!msg.contains("re-run"), "{msg}");
        assert_eq!(migrated.migrated_from(), Some(5));
        assert_eq!(migrated.steps.len(), 200);
        for (id, record) in &log.steps {
            assert_eq!(migrated.steps[id].provenance, record.provenance, "{id}");
        }
        assert_eq!(migrated.provenance.len(), 200);

        migrated.save().unwrap();
        let backup = path.with_extension("v5.yaml");
        assert!(backup.exists(), "the original is kept once");
        assert!(std::fs::read_to_string(&backup).unwrap().contains("schema_version: 5"));
        let (again, msg) = ProgressLog::load(&path).unwrap();
        assert!(msg.is_none(), "{msg:?}");
        assert_eq!(again.migrated_from(), None);
        assert_eq!(again.steps["knn-199"].provenance, log.steps["knn-199"].provenance);
    }

    /// A dependent built on an earlier version of its upstream keeps
    /// that version: fresh under a selector that ignores what changed,
    /// stale under one that does not — exactly as before the migration.
    #[test]
    fn migration_keeps_the_snapshot_a_dependent_was_built_on() {
        let dir = scratch_dir();
        let path = dir.path().join(".cache/.upstream.progress.yaml");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut log = ProgressLog::load(&path).unwrap().0;
        record_chain(&mut log, 2, "2.0.0+abc");
        // knn-0 ran again afterwards with a rebuilt binary; knn-1's
        // record still nests the old knn-0.
        let mut nested_v5 = nested_log_for_test(&log);
        let opts0 = log.steps["knn-0"].resolved_options.clone();
        let new_knn0 = log.build_provenance("knn-0", "compute knn", &opts0, &[], "2.0.1+def");
        nested_v5.steps.get_mut("knn-0").unwrap().provenance = Some(nest(&log.provenance, &new_knn0));
        std::fs::write(&path, serde_yaml::to_string(&nested_v5).unwrap()).unwrap();

        let (mut migrated, msg) = ProgressLog::load(&path).unwrap();
        let msg = msg.unwrap();
        assert!(msg.contains("earlier version") && msg.contains("knn-1"), "{msg}");
        assert_eq!(migrated.provenance.len(), 3, "old knn-0, new knn-0, knn-1");
        let opts1 = migrated.steps["knn-1"].resolved_options.clone();
        let current = migrated.build_provenance("knn-1", "compute knn", &opts1, &["knn-0"], "2.0.0+abc");
        assert!(migrated.check_provenance("knn-1", &current, ProvenanceFlags::CONFIG_ONLY).is_none(),
            "only the upstream's binary changed: fresh by default");
        assert!(migrated.check_provenance("knn-1", &current, ProvenanceFlags::STRICT).is_some(),
            "and stale under strict");
    }

    /// The same, when the upstream's *options* changed: stale under
    /// every selector that consults upstreams.
    #[test]
    fn migration_keeps_a_stale_dependent_stale() {
        let dir = scratch_dir();
        let path = dir.path().join(".cache/.upstream.progress.yaml");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut log = ProgressLog::load(&path).unwrap().0;
        record_chain(&mut log, 2, "2.0.0+abc");
        let mut nested_v5 = nested_log_for_test(&log);
        let mut opts0 = log.steps["knn-0"].resolved_options.clone();
        opts0.insert("k".to_string(), "200".to_string());
        let new_knn0 = log.build_provenance("knn-0", "compute knn", &opts0, &[], "2.0.0+abc");
        nested_v5.steps.get_mut("knn-0").unwrap().provenance = Some(nest(&log.provenance, &new_knn0));
        std::fs::write(&path, serde_yaml::to_string(&nested_v5).unwrap()).unwrap();

        let (mut migrated, _) = ProgressLog::load(&path).unwrap();
        let opts1 = migrated.steps["knn-1"].resolved_options.clone();
        let current = migrated.build_provenance("knn-1", "compute knn", &opts1, &["knn-0"], "2.0.0+abc");
        assert!(migrated.check_provenance("knn-1", &current, ProvenanceFlags::CONFIG_ONLY).is_some());
    }

    /// Schema 5 recorded a sharded output under its logical name with
    /// size 0; the migration records the shards, so the step is fresh
    /// rather than re-run.
    #[test]
    fn migration_measures_a_sharded_output_recorded_by_its_logical_name() {
        let dir = scratch_dir();
        let path = dir.path().join(".cache/.upstream.progress.yaml");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let base = dir.path().join("profiles/base");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("base_vectors__0000.fvecs"), [1u8; 40]).unwrap();
        std::fs::write(base.join("base_vectors__0001.fvecs"), [2u8; 24]).unwrap();
        let mut log = ProgressLog::load(&path).unwrap().0;
        let address = log.build_provenance("extract-base", "transform extract", &HashMap::new(), &[], "1.6.0+abc");
        let mut r = rec(Some(address));
        r.outputs.push(OutputRecord { path: "profiles/base/base_vectors.fvecs".into(), size: 0, mtime: None });
        log.record_step("extract-base", r);
        let nested_v5 = nested_log_for_test(&log);
        std::fs::write(&path, serde_yaml::to_string(&nested_v5).unwrap()).unwrap();

        let (migrated, msg) = ProgressLog::load(&path).unwrap();
        assert!(msg.unwrap().contains("1 sharded output"));
        let outputs = &migrated.steps["extract-base"].outputs;
        assert_eq!(
            outputs.iter().map(|o| (o.path.as_str(), o.size)).collect::<Vec<_>>(),
            vec![("profiles/base/base_vectors__0000.fvecs", 40), ("profiles/base/base_vectors__0001.fvecs", 24)]
        );
        assert!(migrated.check_step_freshness("extract-base", None, Some(dir.path())).is_none());
    }

    /// An unknown older schema still clears the log, as before.
    #[test]
    fn an_older_schema_is_still_cleared() {
        let dir = scratch_dir();
        let path = dir.path().join(".cache/.upstream.progress.yaml");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "schema_version: 4\nsteps:\n  a:\n    status: ok\n    message: m\n    completed_at: 2026-01-01T00:00:00Z\n    elapsed_secs: 1.0\n").unwrap();
        let (log, msg) = ProgressLog::load(&path).unwrap();
        assert!(log.steps.is_empty());
        assert!(msg.unwrap().contains("invalidated"));
    }

    #[test]
    fn test_load_nonexistent() {
        let (log, _) = ProgressLog::load(Path::new("/nonexistent/.cache/.upstream.progress.yaml")).unwrap();
        assert!(log.steps.is_empty());
    }

    #[test]
    fn test_save_and_load() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let cache_dir = tmp_dir.path().join(".cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let path = cache_dir.join(".upstream.progress.yaml");

        let (mut log, _) = ProgressLog::load(&path).unwrap();
        log.record_step("test-step", rec(None));
        log.save().unwrap();

        let (reloaded, _) = ProgressLog::load(&path).unwrap();
        assert!(reloaded.get_step("test-step").is_some());
    }

    #[test]
    fn test_build_provenance_basic() {
        let mut log = ProgressLog::new();
        let mut opts = HashMap::new();
        opts.insert("source".to_string(), "base.fvec".to_string());
        opts.insert("output".to_string(), "out.fvec".to_string());

        let p1 = log.build_provenance("step1", "transform extract", &opts, &[], "1.0.0+abc");
        let p2 = log.build_provenance("step1", "transform extract", &opts, &[], "1.0.0+abc");
        assert_eq!(p1, p2, "same inputs should produce the same address");
        assert_eq!(log.provenance.len(), 1, "and one node");

        opts.insert("source".to_string(), "other.fvec".to_string());
        let p3 = log.build_provenance("step1", "transform extract", &opts, &[], "1.0.0+abc");
        assert_ne!(p1, p3, "options change should change the address");
        assert_ne!(
            log.provenance.hash(&p1, ProvenanceFlags::STRICT),
            log.provenance.hash(&p3, ProvenanceFlags::STRICT),
        );
    }

    #[test]
    fn test_build_provenance_chains_through_upstream() {
        let mut log = ProgressLog::new();
        let opts = HashMap::new();

        // Record upstream with one provenance.
        let up_a = log.build_provenance("upstream", "transform extract", &opts, &[], "1.0.0+aaa");
        log.record_step("upstream", rec(Some(up_a)));
        let head_a = log.build_provenance("downstream", "compute knn", &opts, &["upstream"], "1.0.0+abc");

        // Re-record upstream with a different binary git hash.
        let up_b = log.build_provenance("upstream", "transform extract", &opts, &[], "1.0.0+bbb");
        log.record_step("upstream", rec(Some(up_b)));
        let head_b = log.build_provenance("downstream", "compute knn", &opts, &["upstream"], "1.0.0+abc");

        assert_ne!(
            log.provenance.hash(&head_a, ProvenanceFlags::STRICT),
            log.provenance.hash(&head_b, ProvenanceFlags::STRICT),
            "upstream change should cascade to head under STRICT"
        );

        // Under CONFIG_ONLY, the upstream's git hash doesn't matter.
        assert_eq!(
            log.provenance.hash(&head_a, ProvenanceFlags::CONFIG_ONLY),
            log.provenance.hash(&head_b, ProvenanceFlags::CONFIG_ONLY),
            "upstream-only-binary change must not cascade under CONFIG_ONLY"
        );
    }

    #[test]
    fn test_check_provenance_fresh_under_strict() {
        let mut log = ProgressLog::new();
        let opts = HashMap::new();
        let p = log.build_provenance("step1", "compute knn", &opts, &[], "1.0.0+abc");
        log.record_step("step1", rec(Some(p.clone())));

        assert!(log.check_provenance("step1", &p, ProvenanceFlags::STRICT).is_none());

        let p2 = log.build_provenance("step1", "compute knn", &opts, &[], "2.0.0+xyz");
        assert!(log.check_provenance("step1", &p2, ProvenanceFlags::STRICT).is_some());
    }

    #[test]
    fn test_check_provenance_relaxed_selector() {
        let mut log = ProgressLog::new();
        let opts = HashMap::new();
        let stored = log.build_provenance("step1", "compute knn", &opts, &[], "1.0.0+abc");
        log.record_step("step1", rec(Some(stored)));

        // Different binary version, same options.
        let current = log.build_provenance("step1", "compute knn", &opts, &[], "1.5.0+xyz");

        // STRICT: stale.
        assert!(log.check_provenance("step1", &current, ProvenanceFlags::STRICT).is_some());

        // CONFIG_ONLY: fresh — version doesn't matter.
        assert!(log.check_provenance("step1", &current, ProvenanceFlags::CONFIG_ONLY).is_none());

        // VERSION_AWARE: fresh because major version is the same.
        assert!(log.check_provenance("step1", &current, ProvenanceFlags::VERSION_AWARE).is_none());
    }

    #[test]
    fn test_check_provenance_legacy_record() {
        let mut log = ProgressLog::new();
        log.record_step("step1", rec(None)); // legacy record, no provenance
        let opts = HashMap::new();
        let current = log.build_provenance("step1", "compute knn", &opts, &[], "1.0.0+abc");

        assert!(
            log.check_provenance("step1", &current, ProvenanceFlags::STRICT).is_none(),
            "legacy record without provenance should be trusted"
        );
    }

    #[test]
    fn test_check_step_freshness_output_files_verified() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let output_path = tmp_dir.path().join("output.ivec");
        std::fs::write(&output_path, "result").unwrap();

        let mut log = ProgressLog::new();
        let mut r = rec(None);
        r.outputs.push(OutputRecord {
            path: output_path.to_string_lossy().into_owned(),
            size: 6,
            mtime: None,
        });
        log.record_step("step1", r);

        let reason = log.check_step_freshness("step1", None, None);
        assert!(reason.is_none(), "should be fresh: {:?}", reason);

        std::fs::write(&output_path, "longer result").unwrap();
        let reason = log.check_step_freshness("step1", None, None);
        assert!(reason.is_some(), "should be stale after size change");
        assert!(reason.unwrap().contains("size changed"));

        std::fs::remove_file(&output_path).unwrap();
        let reason = log.check_step_freshness("step1", None, None);
        assert!(reason.is_some(), "should be stale when output missing");
        assert!(reason.unwrap().contains("missing"));
    }

    #[test]
    fn test_schema_version_invalidation() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let cache_dir = tmp_dir.path().join(".cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let path = cache_dir.join(".upstream.progress.yaml");

        // Write a progress log with an old schema version.
        let old_content = "schema_version: 1\nsteps:\n  step1:\n    status: ok\n    message: done\n    completed_at: '2026-01-01T00:00:00Z'\n    elapsed_secs: 1.0\n";
        std::fs::write(&path, old_content).unwrap();

        let (log, msg) = ProgressLog::load(&path).unwrap();
        assert!(msg.is_some(), "expected schema version invalidation");
        assert!(msg.unwrap().contains("schema version changed"));
        assert!(log.steps.is_empty(), "steps should be cleared on version mismatch");
        assert_eq!(log.schema_version, PROGRESS_SCHEMA_VERSION);
    }
}
