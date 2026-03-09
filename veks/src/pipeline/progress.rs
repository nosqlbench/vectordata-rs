// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Progress log for command stream pipelines.
//!
//! Tracks the status of each pipeline step in a persistent YAML file
//! (`.upstream.progress.yaml`) next to the `dataset.yaml`. This enables
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
const PROGRESS_SCHEMA_VERSION: u32 = 3;

/// Persistent progress log for a pipeline execution.
///
/// Stored as `.upstream.progress.yaml` next to the dataset.yaml file.
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

    /// Check if a source file (e.g. dataset.yaml) is newer than this progress
    /// log. If so, clear all entries and return a message describing the
    /// invalidation.
    ///
    /// Returns `Some(message)` if the log was invalidated, `None` otherwise.
    pub fn invalidate_if_stale(&mut self, source_path: &Path) -> Option<String> {
        let log_path = self.path.as_ref()?;
        let log_mtime = std::fs::metadata(log_path).ok()?.modified().ok()?;
        let source_mtime = std::fs::metadata(source_path).ok()?.modified().ok()?;
        if source_mtime > log_mtime {
            let count = self.steps.len();
            self.steps.clear();
            Some(format!(
                "Progress log invalidated: {} is newer than {} ({} step records cleared)",
                source_path.display(),
                log_path.display(),
                count,
            ))
        } else {
            None
        }
    }

    /// Derive the progress log path from a dataset.yaml path.
    ///
    /// Returns `<dir>/.upstream.progress.yaml` where `<dir>` is the directory
    /// containing the dataset file.
    pub fn path_for_dataset(dataset_path: &Path) -> PathBuf {
        let dir = dataset_path.parent().unwrap_or(Path::new("."));
        dir.join(".upstream.progress.yaml")
    }

    /// Record a step's execution result.
    pub fn record_step(&mut self, step_id: &str, record: StepRecord) {
        self.steps.insert(step_id.to_string(), record);
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
    /// it is stale (input newer than output, options changed, etc.).
    ///
    /// When `workspace` is provided, relative paths in options are resolved
    /// against it for mtime comparisons.
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

        // Check whether resolved options changed since the last run
        if let Some(current) = current_options {
            if !record.resolved_options.is_empty() && &record.resolved_options != current {
                return Some("options changed".to_string());
            }
        }

        // Check output files exist with matching sizes
        for output in &record.outputs {
            let path = resolve_path(&output.path, workspace);
            match std::fs::metadata(&path) {
                Ok(meta) => {
                    if meta.len() != output.size {
                        return Some(format!(
                            "output '{}' size changed ({} → {})",
                            output.path, output.size, meta.len()
                        ));
                    }
                }
                Err(_) => return Some(format!("output '{}' missing", output.path)),
            }
        }

        // Check input file mtimes: if any input is newer than the step's
        // completion time, the step is stale.
        if let Some(current) = current_options {
            let completed = record.completed_at;
            let completed_systime = std::time::SystemTime::UNIX_EPOCH
                + std::time::Duration::from_secs(completed.timestamp() as u64);
            for (key, value) in current {
                // Skip "output" — that's what we produce, not what we consume
                if key == "output" {
                    continue;
                }
                let path = resolve_path(value, workspace);
                if path.is_file() {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        if let Ok(mtime) = meta.modified() {
                            if mtime > completed_systime {
                                return Some(format!(
                                    "input '{}' ({}) is newer than last run",
                                    key, value,
                                ));
                            }
                        }
                    }
                }
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

/// Resolve a path that may be relative, using the workspace as the base
/// directory when provided.
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
            PathBuf::from("/data/my-dataset/.upstream.progress.yaml")
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
            },
        );

        let yaml = serde_yaml::to_string(&log).unwrap();
        let parsed: ProgressLog = serde_yaml::from_str(&yaml).unwrap();
        assert!(parsed.get_step("step1").is_some());
        assert_eq!(parsed.steps["step1"].outputs.len(), 1);
    }

    #[test]
    fn test_load_nonexistent() {
        let (log, _) = ProgressLog::load(Path::new("/nonexistent/.upstream.progress.yaml")).unwrap();
        assert!(log.steps.is_empty());
    }

    #[test]
    fn test_save_and_load() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join(".upstream.progress.yaml");

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
            },
        );
        log.save().unwrap();

        let (reloaded, _) = ProgressLog::load(&path).unwrap();
        assert!(reloaded.get_step("test-step").is_some());
    }

    #[test]
    fn test_invalidate_if_stale() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let progress_path = tmp_dir.path().join(".upstream.progress.yaml");
        let dataset_path = tmp_dir.path().join("dataset.yaml");

        // Create progress log first
        let (mut log, _) = ProgressLog::load(&progress_path).unwrap();
        log.record_step(
            "step1",
            StepRecord {
                status: Status::Ok,
                message: "ok".to_string(),
                completed_at: Utc::now(),
                elapsed_secs: 0.5,
                outputs: vec![],
                resolved_options: HashMap::new(),
                error: None,
                resource_summary: None,
            },
        );
        log.save().unwrap();

        // Create dataset.yaml after progress log (newer mtime)
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&dataset_path, "name: test\n").unwrap();

        // Reload and check invalidation
        let (mut log, _) = ProgressLog::load(&progress_path).unwrap();
        assert!(log.get_step("step1").is_some());
        let msg = log.invalidate_if_stale(&dataset_path);
        assert!(msg.is_some(), "expected invalidation");
        assert!(log.steps.is_empty(), "steps should be cleared");
    }

    #[test]
    fn test_invalidate_if_stale_no_invalidation() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let progress_path = tmp_dir.path().join(".upstream.progress.yaml");
        let dataset_path = tmp_dir.path().join("dataset.yaml");

        // Create dataset.yaml first
        std::fs::write(&dataset_path, "name: test\n").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Then create progress log (newer)
        let (mut log, _) = ProgressLog::load(&progress_path).unwrap();
        log.record_step(
            "step1",
            StepRecord {
                status: Status::Ok,
                message: "ok".to_string(),
                completed_at: Utc::now(),
                elapsed_secs: 0.5,
                outputs: vec![],
                resolved_options: HashMap::new(),
                error: None,
                resource_summary: None,
            },
        );
        log.save().unwrap();

        let (mut log, _) = ProgressLog::load(&progress_path).unwrap();
        let msg = log.invalidate_if_stale(&dataset_path);
        assert!(msg.is_none(), "should not invalidate when progress is newer");
        assert!(log.get_step("step1").is_some());
    }

    #[test]
    fn test_check_step_freshness_input_mtime() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let input_path = tmp_dir.path().join("input.fvec");
        let output_path = tmp_dir.path().join("output.ivec");

        // Create input file
        std::fs::write(&input_path, "data").unwrap();

        // Record step as completed "in the past"
        let mut log = ProgressLog::new();
        let past = Utc::now() - chrono::Duration::seconds(60);
        log.record_step(
            "knn",
            StepRecord {
                status: Status::Ok,
                message: "ok".to_string(),
                completed_at: past,
                elapsed_secs: 1.0,
                outputs: vec![OutputRecord {
                    path: output_path.to_string_lossy().into_owned(),
                    size: 6,
                    mtime: None,
                }],
                resolved_options: {
                    let mut m = HashMap::new();
                    m.insert("source".to_string(), input_path.to_string_lossy().into_owned());
                    m.insert("output".to_string(), output_path.to_string_lossy().into_owned());
                    m
                },
                error: None,
                resource_summary: None,
            },
        );

        // Create output file with correct size
        std::fs::write(&output_path, "result").unwrap();

        // Input is newer than the step's completed_at — should be stale
        let opts: HashMap<String, String> = [
            ("source".to_string(), input_path.to_string_lossy().into_owned()),
            ("output".to_string(), output_path.to_string_lossy().into_owned()),
        ].into_iter().collect();

        let reason = log.check_step_freshness("knn", Some(&opts), None);
        assert!(reason.is_some(), "expected stale due to input mtime");
        assert!(reason.unwrap().contains("input"), "reason should mention input");
    }

    #[test]
    fn test_schema_version_invalidation() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join(".upstream.progress.yaml");

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
