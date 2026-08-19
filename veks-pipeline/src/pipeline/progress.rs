// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Progress log for command stream pipelines.
//!
//! Tracks the status of each pipeline step in a persistent YAML file
//! (`.cache/.upstream.progress.yaml`) in the workspace cache directory.
//! This enables skip-if-fresh semantics: completed steps are skipped on
//! re-run unless their inputs have changed.
//!
//! Per-step staleness is governed by a structured
//! [`ProvenanceMap`](super::provenance::ProvenanceMap) — see that module
//! for details on selector-driven hashing.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::command::Status;
use super::provenance::{BinaryVersion, ProvenanceFlags, ProvenanceMap};

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
/// - v5: structured `ProvenanceMap` replaces opaque fingerprint string;
///   staleness check is selector-driven (see `ProvenanceFlags`)
const PROGRESS_SCHEMA_VERSION: u32 = 5;

/// Persistent progress log for a pipeline execution.
///
/// Stored as `.cache/.upstream.progress.yaml` in the workspace cache directory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProgressLog {
    /// Path to the progress file on disk.
    #[serde(skip)]
    path: Option<PathBuf>,

    /// Schema version — used to auto-invalidate when cache key algorithms
    /// or internal formats change.
    #[serde(default)]
    pub schema_version: u32,

    /// Per-step execution records, keyed by step ID.
    #[serde(default)]
    pub steps: HashMap<String, StepRecord>,
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
    /// Structured provenance for staleness checks. The recorded map
    /// captures every component (identity, binary version components,
    /// resolved options, upstream provenance) so the staleness hash can
    /// be recomputed under any [`ProvenanceFlags`] selector at check
    /// time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<ProvenanceMap>,
}

/// Record of a single output artifact at completion time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputRecord {
    /// Path to the output file.
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
    /// If the stored schema version does not match the current
    /// `PROGRESS_SCHEMA_VERSION`, all step records are cleared and a
    /// message is returned via the second tuple element.
    pub fn load(path: &Path) -> Result<(Self, Option<String>), String> {
        if path.exists() {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read progress log {}: {}", path.display(), e))?;
            let mut log: ProgressLog = serde_yaml::from_str(&content).map_err(|e| {
                format!(
                    "Failed to parse progress log {}: {}",
                    path.display(),
                    e
                )
            })?;
            log.path = Some(path.to_path_buf());

            // Schema version check: if the stored version differs from the
            // current code's version, the cache key algorithms or internal
            // formats may have changed. Clear all records so steps re-run.
            let invalidation_msg = if log.schema_version != PROGRESS_SCHEMA_VERSION {
                let count = log.steps.len();
                let old_ver = log.schema_version;
                log.steps.clear();
                log.schema_version = PROGRESS_SCHEMA_VERSION;
                Some(format!(
                    "Progress log invalidated: schema version changed ({} → {}), {} step records cleared",
                    old_ver, PROGRESS_SCHEMA_VERSION, count,
                ))
            } else {
                None
            };

            Ok((log, invalidation_msg))
        } else {
            Ok((ProgressLog {
                path: Some(path.to_path_buf()),
                schema_version: PROGRESS_SCHEMA_VERSION,
                steps: HashMap::new(),
            }, None))
        }
    }

    /// Build a [`ProvenanceMap`] for a step. Pulls the upstream maps
    /// from the stored records of the named upstream steps; missing or
    /// legacy records contribute an empty default map (the resulting
    /// hash will treat them as a known-distinct sentinel under any
    /// selector that includes `UPSTREAM`).
    pub fn build_provenance(
        &self,
        step_id: &str,
        command_path: &str,
        resolved_options: &HashMap<String, String>,
        upstream_ids: &[&str],
        build_version: &str,
    ) -> ProvenanceMap {
        let binary = BinaryVersion::parse(build_version);
        let mut upstream: BTreeMap<String, ProvenanceMap> = BTreeMap::new();
        for up_id in upstream_ids {
            let up_map = self
                .steps
                .get(*up_id)
                .and_then(|r| r.provenance.clone())
                .unwrap_or_else(|| ProvenanceMap {
                    step_id: (*up_id).to_string(),
                    command_path: String::new(),
                    binary_version_major: 0,
                    binary_version_minor: 0,
                    binary_version_patch: 0,
                    binary_git_hash: String::new(),
                    binary_dirty: false,
                    options: BTreeMap::new(),
                    upstream: BTreeMap::new(),
                });
            upstream.insert((*up_id).to_string(), up_map);
        }
        ProvenanceMap::build(step_id, command_path, &binary, resolved_options, upstream)
    }

    /// Check whether a step's stored provenance matches the current
    /// computed provenance under `selector`. Returns `Some(reason)` if
    /// stale, `None` if fresh.
    pub fn check_provenance(
        &self,
        step_id: &str,
        current: &ProvenanceMap,
        selector: ProvenanceFlags,
    ) -> Option<String> {
        match self.steps.get(step_id) {
            Some(record) => match record.provenance.as_ref() {
                Some(stored) => {
                    let stored_hash = stored.hash(selector);
                    let current_hash = current.hash(selector);
                    if stored_hash == current_hash {
                        None
                    } else {
                        let diffs = current.diff(stored);
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
    /// (see [`ProvenanceMap::read_sidecar`]).
    pub fn record_step(&mut self, step_id: &str, record: StepRecord) {
        // Provenance sidecars for dataset outputs are staleness metadata,
        // not dataset content — they belong under `<cache>/provenance/`,
        // never beside the artifact. The cache directory is the progress
        // log's own parent (`<dataset>/.cache`); without a progress path
        // (purely in-memory log) there's nowhere to persist them, so skip.
        if let (Some(prov), Some(cache_dir)) =
            (record.provenance.as_ref(), self.path.as_deref().and_then(Path::parent))
        {
            // Migrate away any sidecar a previous version wrote beside the
            // artifact, so re-running cleans up the dataset layer.
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
                let stale = ProvenanceMap::sidecar_path(&abs_artifact);
                if stale.exists() {
                    let _ = std::fs::remove_file(&stale);
                }
                if let Err(e) = prov.write_cached_sidecar(cache_dir, rel) {
                    log::warn!(
                        "provenance: failed to write sidecar for {}: {}",
                        out.path, e,
                    );
                }
            }
        }
        self.steps.insert(step_id.to_string(), record);
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

    /// Persist the progress log to disk.
    ///
    /// Writes atomically by writing to a temp file and renaming.
    pub fn save(&self) -> Result<(), String> {
        let path = self.path.as_ref().ok_or("progress log has no file path")?;
        let content = serde_yaml::to_string(self)
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

    fn rec(provenance: Option<ProvenanceMap>) -> StepRecord {
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
    /// `ProvenanceMap`. This is the producer half of the
    /// upstream-cascade contract: downstream consumers find the
    /// sidecar via [`ProvenanceMap::read_sidecar`] and merge it
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
        use super::super::provenance::BinaryVersion;
        let ds = tempfile::tempdir().unwrap();
        let cache = ds.path().join(".cache");

        let prov = ProvenanceMap::build(
            "test-step",
            "test command",
            &BinaryVersion::parse("1.0.0+abcd"),
            &HashMap::new(),
            std::collections::BTreeMap::new(),
        );
        let mut r = rec(Some(prov.clone()));
        // Workspace-relative output path (as the runner records it).
        let rel = "profiles/base/out.slab";
        r.outputs.push(OutputRecord { path: rel.to_string(), size: 12, mtime: None });

        let mut log = log_for_dataset(ds.path());
        log.record_step("test-step", r);

        // Sidecar lands under <cache>/provenance/, mirroring the artifact
        // path — and NOT beside the dataset artifact.
        let cached = ProvenanceMap::cached_sidecar_path(&cache, Path::new(rel));
        assert!(cached.exists(), "sidecar should be at {}", cached.display());
        assert!(!ds.path().join("profiles/base/out.slab.provenance.json").exists(),
            "provenance must not pollute the dataset storage layer");

        // …and round-trips to a structurally-equal map.
        let recovered = ProvenanceMap::read_cached_sidecar(&cache, Path::new(rel))
            .unwrap().unwrap();
        assert_eq!(recovered.hash(super::super::provenance::ProvenanceFlags::STRICT),
                   prov.hash(super::super::provenance::ProvenanceFlags::STRICT));
    }

    /// A stale sidecar a previous version wrote *beside* the artifact is
    /// migrated away (removed) on the next `record_step`.
    #[test]
    fn record_step_migrates_legacy_colocated_sidecar() {
        use super::super::provenance::BinaryVersion;
        let ds = tempfile::tempdir().unwrap();
        let rel = "profiles/base/out.slab";
        let legacy = ds.path().join("profiles/base/out.slab.provenance.json");
        std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        std::fs::write(&legacy, b"{}").unwrap();

        let prov = ProvenanceMap::build("s", "c",
            &BinaryVersion::parse("1.0.0+abcd"), &HashMap::new(),
            std::collections::BTreeMap::new());
        let mut r = rec(Some(prov));
        r.outputs.push(OutputRecord { path: rel.to_string(), size: 1, mtime: None });

        let mut log = log_for_dataset(ds.path());
        log.record_step("s", r);

        assert!(!legacy.exists(), "legacy co-located sidecar must be removed");
        assert!(ProvenanceMap::cached_sidecar_path(&ds.path().join(".cache"), Path::new(rel))
            .exists(), "new sidecar must be under cache/provenance");
    }

    /// A derived sidecar output (a `.mref` from the merkle step, or a provenance
    /// sidecar) must NOT itself get a provenance sidecar — otherwise suffixes
    /// compound without bound (`…fvecs.provenance.json.mref.provenance.json`).
    #[test]
    fn record_step_skips_sidecar_for_derived_outputs() {
        let ds = tempfile::tempdir().unwrap();
        let cache = ds.path().join(".cache");
        let prov = ProvenanceMap::build(
            "merkle-step",
            "merkle create",
            &BinaryVersion::parse("1.0.0+abcd"),
            &HashMap::new(),
            std::collections::BTreeMap::new(),
        );
        for derived in ["base.fvec.mref", "base.fvec.provenance.json", "data.mrkl"] {
            let mut r = rec(Some(prov.clone()));
            r.outputs.push(OutputRecord { path: derived.to_string(), size: 7, mtime: None });
            let mut log = log_for_dataset(ds.path());
            log.record_step("merkle-step", r);
            assert!(
                !ProvenanceMap::cached_sidecar_path(&cache, Path::new(derived)).exists(),
                "{derived} must not get a provenance sidecar"
            );
        }
    }

    /// Producer steps without a `ProvenanceMap` (legacy records or
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
        assert!(!ProvenanceMap::cached_sidecar_path(&ds.path().join(".cache"), Path::new(rel))
            .exists(), "no provenance → no sidecar (best-effort, not mandatory)");
    }

    #[test]
    fn test_roundtrip() {
        let mut log = ProgressLog::new();
        let mut r = rec(None);
        r.outputs.push(OutputRecord {
            path: "output.fvec".into(),
            size: 1024,
            mtime: None,
        });
        log.record_step("step1", r);

        let yaml = serde_yaml::to_string(&log).unwrap();
        let parsed: ProgressLog = serde_yaml::from_str(&yaml).unwrap();
        assert!(parsed.get_step("step1").is_some());
        assert_eq!(parsed.steps["step1"].outputs.len(), 1);
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
        let log = ProgressLog::new();
        let mut opts = HashMap::new();
        opts.insert("source".to_string(), "base.fvec".to_string());
        opts.insert("output".to_string(), "out.fvec".to_string());

        let p1 = log.build_provenance("step1", "transform extract", &opts, &[], "1.0.0+abc");
        let p2 = log.build_provenance("step1", "transform extract", &opts, &[], "1.0.0+abc");
        assert_eq!(
            p1.hash(ProvenanceFlags::STRICT),
            p2.hash(ProvenanceFlags::STRICT),
            "same inputs should produce same provenance hash"
        );

        opts.insert("source".to_string(), "other.fvec".to_string());
        let p3 = log.build_provenance("step1", "transform extract", &opts, &[], "1.0.0+abc");
        assert_ne!(
            p1.hash(ProvenanceFlags::STRICT),
            p3.hash(ProvenanceFlags::STRICT),
            "options change should change hash"
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
            head_a.hash(ProvenanceFlags::STRICT),
            head_b.hash(ProvenanceFlags::STRICT),
            "upstream change should cascade to head under STRICT"
        );

        // Under CONFIG_ONLY, the upstream's git hash doesn't matter.
        assert_eq!(
            head_a.hash(ProvenanceFlags::CONFIG_ONLY),
            head_b.hash(ProvenanceFlags::CONFIG_ONLY),
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
