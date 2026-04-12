// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Progress log for command stream pipelines.
//!
//! Tracks the status of each pipeline step in a persistent YAML file
//! (`.cache/.upstream.progress.yaml`) in the workspace cache directory. This enables
//! skip-if-fresh semantics: completed steps are skipped on re-run unless
//! their inputs have changed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::command::Status;

/// Schema version for the progress log.
///
/// Bump this whenever cache key algorithms, segment naming conventions,
/// or other internal formats change. On load, if the stored version does
/// not match, all step records are cleared (the user is notified).
///
/// History:
/// - v3: fingerprint chains for DAG-based staleness
/// - v4: build_version in StepRecord, mtime-based staleness removed,
///        fingerprint now includes command build version
const PROGRESS_SCHEMA_VERSION: u32 = 4;

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
    /// Configuration fingerprint: hash of this step's identity (id, command,
    /// resolved options, build version) plus the fingerprints of all upstream
    /// dependencies. When the fingerprint changes, this step and all
    /// downstream dependents must re-execute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
    /// Build version of the command that produced this result.
    /// When the current command's build_version differs from this stored
    /// value, the step is stale — the command logic may have changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_version: Option<String>,
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

    /// Compute the configuration fingerprint for a step.
    ///
    /// The fingerprint is a hash of the step's identity (id, command,
    /// sorted resolved options) plus the stored fingerprints of all
    /// upstream dependencies. If any upstream step has no stored
    /// fingerprint (legacy record), it contributes "unknown".
    ///
    /// This creates a Merkle-like chain: changing any step's options
    /// invalidates that step and cascades to all dependents.
    pub fn compute_fingerprint(
        &self,
        step_id: &str,
        run_command: &str,
        resolved_options: &HashMap<String, String>,
        upstream_ids: &[&str],
        build_version: &str,
    ) -> String {
        let mut hasher = FnvHasher::new();
        hasher.write(step_id.as_bytes());
        hasher.write(b"\0");
        hasher.write(run_command.as_bytes());
        hasher.write(b"\0");
        hasher.write(build_version.as_bytes());
        hasher.write(b"\0");

        // Sort options for deterministic ordering
        let mut sorted: Vec<(&str, &str)> = resolved_options
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        sorted.sort();
        for (k, v) in sorted {
            hasher.write(k.as_bytes());
            hasher.write(b"=");
            hasher.write(v.as_bytes());
            hasher.write(b"\0");
        }

        // Chain upstream fingerprints
        hasher.write(b"upstream:");
        let mut up_sorted: Vec<&str> = upstream_ids.to_vec();
        up_sorted.sort();
        for up_id in up_sorted {
            let up_fp = self.steps.get(up_id)
                .and_then(|r| r.fingerprint.as_deref())
                .unwrap_or("unknown");
            hasher.write(up_id.as_bytes());
            hasher.write(b":");
            hasher.write(up_fp.as_bytes());
            hasher.write(b"\0");
        }

        format!("{:016x}", hasher.finish())
    }

    /// Check whether a step's stored fingerprint matches its current
    /// computed fingerprint. Returns `Some(reason)` if stale.
    pub fn check_fingerprint(
        &self,
        step_id: &str,
        current_fingerprint: &str,
    ) -> Option<String> {
        match self.steps.get(step_id) {
            Some(record) => {
                match record.fingerprint.as_deref() {
                    Some(stored) if stored == current_fingerprint => None,
                    Some(stored) => Some(format!(
                        "fingerprint changed ({} → {})",
                        &stored[..8.min(stored.len())],
                        &current_fingerprint[..8.min(current_fingerprint.len())],
                    )),
                    None => None, // legacy record without fingerprint — trust it
                }
            }
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
                // Ensure .cache/ exists
                let cache_dir = dir.join(".cache");
                if std::fs::create_dir_all(&cache_dir).is_ok() {
                    if std::fs::rename(&old_path, &new_path).is_ok() {
                        log::info!(
                            "Migrated progress log: {} → {}",
                            old_path.display(),
                            new_path.display(),
                        );
                    }
                }
            } else {
                // Both exist — new location is authoritative, remove orphan
                let _ = std::fs::remove_file(&old_path);
            }
        }

        new_path
    }

    /// Record a step's execution result.
    pub fn record_step(&mut self, step_id: &str, record: StepRecord) {
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
    ///
    /// Returns `true` if the step is fresh, `false` if it needs re-running.
    /// Use [`check_step_freshness`] for a detailed reason string.
    pub fn is_step_fresh(
        &self,
        step_id: &str,
        current_options: Option<&HashMap<String, String>>,
    ) -> bool {
        self.check_step_freshness(step_id, current_options, None).is_none()
    }

    /// Check whether a step needs to be re-run, returning a reason if stale.
    ///
    /// Returns `None` if the step is fresh, or `Some(reason)` describing why
    /// it is stale (options changed, outputs missing/corrupted, etc.).
    ///
    /// Staleness from upstream changes is handled by the fingerprint chain
    /// (see `compute_fingerprint` and `check_fingerprint`) — not by mtime
    /// comparisons. This method only checks local state: options match,
    /// outputs exist with correct sizes.
    ///
    /// When `workspace` is provided, relative paths in options are resolved
    /// against it for output file checks.
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

        // Check whether resolved options changed since the last run
        if let Some(current) = current_options {
            if !record.resolved_options.is_empty() && &record.resolved_options != current {
                return Some("options changed".to_string());
            }
        }

        // Check output files exist with matching sizes.
        // Catalog files are regenerable artifacts that may be updated
        // externally (e.g., `veks prepare catalog generate` from a parent
        // directory) — skip size verification for them.
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
/// Used by the pipeline executor to compute per-step configuration
/// fingerprints. No external dependency.
pub struct FnvHasher {
    state: u64,
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

        log.record_step(
            "step1",
            StepRecord {
                status: Status::Ok,
                message: "done".to_string(),
                completed_at: Utc::now(),
                elapsed_secs: 1.5,
                outputs: vec![],
                resolved_options: HashMap::new(),
                error: None,
                resource_summary: None,
                fingerprint: None,
                build_version: None,
            },
        );

        assert!(log.get_step("step1").is_some());
        assert!(log.get_step("step2").is_none());
    }

    #[test]
    fn test_roundtrip() {
        let mut log = ProgressLog::new();
        log.record_step(
            "step1",
            StepRecord {
                status: Status::Ok,
                message: "completed".to_string(),
                completed_at: Utc::now(),
                elapsed_secs: 2.0,
                outputs: vec![OutputRecord {
                    path: "output.fvec".to_string(),
                    size: 1024,
                    mtime: None,
                }],
                resolved_options: HashMap::new(),
                error: None,
                resource_summary: None,
                fingerprint: None,
                build_version: None,
            },
        );

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
        log.record_step(
            "test-step",
            StepRecord {
                status: Status::Ok,
                message: "ok".to_string(),
                completed_at: Utc::now(),
                elapsed_secs: 0.5,
                outputs: vec![],
                resolved_options: HashMap::new(),
                error: None,
                resource_summary: None,
                fingerprint: None,
                build_version: None,
            },
        );
        log.save().unwrap();

        let (reloaded, _) = ProgressLog::load(&path).unwrap();
        assert!(reloaded.get_step("test-step").is_some());
    }

    #[test]
    fn test_fingerprint_basic() {
        let log = ProgressLog::new();
        let mut opts = HashMap::new();
        opts.insert("source".to_string(), "base.fvec".to_string());
        opts.insert("output".to_string(), "out.fvec".to_string());

        let fp1 = log.compute_fingerprint("step1", "transform extract", &opts, &[], "");
        let fp2 = log.compute_fingerprint("step1", "transform extract", &opts, &[], "");
        assert_eq!(fp1, fp2, "same inputs should produce same fingerprint");

        // Change an option
        opts.insert("source".to_string(), "other.fvec".to_string());
        let fp3 = log.compute_fingerprint("step1", "transform extract", &opts, &[], "");
        assert_ne!(fp1, fp3, "different options should produce different fingerprint");
    }

    #[test]
    fn test_fingerprint_cascades_through_upstream() {
        let mut log = ProgressLog::new();

        // Record upstream step with a fingerprint
        log.record_step("upstream", StepRecord {
            status: Status::Ok,
            message: "done".into(),
            completed_at: Utc::now(),
            elapsed_secs: 1.0,
            outputs: vec![],
            resolved_options: HashMap::new(),
            error: None,
            resource_summary: None,
            fingerprint: Some("aaaa".to_string()),
            build_version: None,
        });

        let opts = HashMap::new();
        let fp1 = log.compute_fingerprint("downstream", "compute knn", &opts, &["upstream"], "");

        // Change upstream fingerprint
        log.record_step("upstream", StepRecord {
            status: Status::Ok,
            message: "done".into(),
            completed_at: Utc::now(),
            elapsed_secs: 1.0,
            outputs: vec![],
            resolved_options: HashMap::new(),
            error: None,
            resource_summary: None,
            fingerprint: Some("bbbb".to_string()),
            build_version: None,
        });

        let fp2 = log.compute_fingerprint("downstream", "compute knn", &opts, &["upstream"], "");
        assert_ne!(fp1, fp2, "upstream fingerprint change should cascade");
    }

    #[test]
    fn test_check_fingerprint_fresh() {
        let mut log = ProgressLog::new();
        log.record_step("step1", StepRecord {
            status: Status::Ok,
            message: "done".into(),
            completed_at: Utc::now(),
            elapsed_secs: 1.0,
            outputs: vec![],
            resolved_options: HashMap::new(),
            error: None,
            resource_summary: None,
            fingerprint: Some("abc123".to_string()),
            build_version: None,
        });

        assert!(log.check_fingerprint("step1", "abc123").is_none(), "matching fingerprint should be fresh");
        assert!(log.check_fingerprint("step1", "xyz789").is_some(), "mismatched fingerprint should be stale");
    }

    #[test]
    fn test_check_fingerprint_legacy_record() {
        let mut log = ProgressLog::new();
        log.record_step("step1", StepRecord {
            status: Status::Ok,
            message: "done".into(),
            completed_at: Utc::now(),
            elapsed_secs: 1.0,
            outputs: vec![],
            resolved_options: HashMap::new(),
            error: None,
            resource_summary: None,
            fingerprint: None, // legacy — no fingerprint
            build_version: None,
        });

        // Legacy records without fingerprint should be trusted (not forced stale)
        assert!(log.check_fingerprint("step1", "anything").is_none(),
            "legacy record without fingerprint should be trusted");
    }

    #[test]
    fn test_staleness_from_upstream_via_fingerprint() {
        // When an upstream step changes its fingerprint (re-executed),
        // the downstream step's computed fingerprint changes, making it stale.
        // This replaces the old mtime-based staleness detection.
        let mut log = ProgressLog::new();

        // Record upstream step with fingerprint "aaa"
        log.record_step("extract", StepRecord {
            status: Status::Ok,
            message: "done".into(),
            completed_at: Utc::now(),
            elapsed_secs: 1.0,
            outputs: vec![],
            resolved_options: HashMap::new(),
            error: None,
            resource_summary: None,
            fingerprint: Some("aaa".to_string()),
            build_version: Some("0.1.0+abc".to_string()),
        });

        // Compute downstream fingerprint with upstream at "aaa"
        let opts = HashMap::new();
        let fp1 = log.compute_fingerprint("knn", "compute knn", &opts, &["extract"], "0.1.0+abc");

        // Record downstream step with this fingerprint
        log.record_step("knn", StepRecord {
            status: Status::Ok,
            message: "done".into(),
            completed_at: Utc::now(),
            elapsed_secs: 2.0,
            outputs: vec![],
            resolved_options: opts.clone(),
            error: None,
            resource_summary: None,
            fingerprint: Some(fp1.clone()),
            build_version: Some("0.1.0+abc".to_string()),
        });

        // Fingerprint should match — step is fresh
        assert!(log.check_fingerprint("knn", &fp1).is_none(), "should be fresh initially");

        // Now upstream re-executes with a new fingerprint
        log.record_step("extract", StepRecord {
            status: Status::Ok,
            message: "re-ran".into(),
            completed_at: Utc::now(),
            elapsed_secs: 1.0,
            outputs: vec![],
            resolved_options: HashMap::new(),
            error: None,
            resource_summary: None,
            fingerprint: Some("bbb".to_string()),
            build_version: Some("0.1.0+abc".to_string()),
        });

        // Recompute downstream fingerprint — it should differ
        let fp2 = log.compute_fingerprint("knn", "compute knn", &opts, &["extract"], "0.1.0+abc");
        assert_ne!(fp1, fp2, "upstream change should produce different fingerprint");
        assert!(log.check_fingerprint("knn", &fp2).is_some(),
            "stored fingerprint should be stale after upstream change");
    }

    #[test]
    fn test_staleness_from_build_version_change() {
        // When the command's build version changes (recompilation), the
        // fingerprint changes, making the step stale.
        let log = ProgressLog::new();
        let opts = HashMap::new();

        let fp_v1 = log.compute_fingerprint("step1", "compute knn", &opts, &[], "0.17.0+abc123");
        let fp_v2 = log.compute_fingerprint("step1", "compute knn", &opts, &[], "0.17.0+def456");

        assert_ne!(fp_v1, fp_v2,
            "different build versions should produce different fingerprints");
    }

    #[test]
    fn test_check_step_freshness_output_files_verified() {
        // Freshness check verifies output files exist with correct sizes.
        // Missing or size-changed outputs make the step stale.
        let tmp_dir = tempfile::tempdir().unwrap();
        let output_path = tmp_dir.path().join("output.ivec");

        // Create output file
        std::fs::write(&output_path, "result").unwrap();

        let mut log = ProgressLog::new();
        log.record_step("step1", StepRecord {
            status: Status::Ok,
            message: "done".into(),
            completed_at: Utc::now(),
            elapsed_secs: 1.0,
            outputs: vec![OutputRecord {
                path: output_path.to_string_lossy().into_owned(),
                size: 6, // matches "result" (6 bytes)
                mtime: None,
            }],
            resolved_options: HashMap::new(),
            error: None,
            resource_summary: None,
            fingerprint: None,
            build_version: None,
        });

        // Fresh when output exists with correct size
        let reason = log.check_step_freshness("step1", None, None);
        assert!(reason.is_none(), "should be fresh: {:?}", reason);

        // Stale when output size changes
        std::fs::write(&output_path, "longer result").unwrap();
        let reason = log.check_step_freshness("step1", None, None);
        assert!(reason.is_some(), "should be stale after size change");
        assert!(reason.unwrap().contains("size changed"));

        // Stale when output is deleted
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

        // Write a progress log with an old schema version
        let old_content = "schema_version: 1\nsteps:\n  step1:\n    status: ok\n    message: done\n    completed_at: '2026-01-01T00:00:00Z'\n    elapsed_secs: 1.0\n";
        std::fs::write(&path, old_content).unwrap();

        let (log, msg) = ProgressLog::load(&path).unwrap();
        assert!(msg.is_some(), "expected schema version invalidation");
        assert!(msg.unwrap().contains("schema version changed"));
        assert!(log.steps.is_empty(), "steps should be cleared on version mismatch");
        assert_eq!(log.schema_version, PROGRESS_SCHEMA_VERSION);
    }

}
