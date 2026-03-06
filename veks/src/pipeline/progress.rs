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

/// Persistent progress log for a pipeline execution.
///
/// Stored as `.upstream.progress.yaml` next to the dataset.yaml file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProgressLog {
    /// Path to the progress file on disk.
    #[serde(skip)]
    path: Option<PathBuf>,

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
        ProgressLog::default()
    }

    /// Load a progress log from a file, or create a new one if the file
    /// does not exist.
    pub fn load(path: &Path) -> Result<Self, String> {
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
            Ok(log)
        } else {
            Ok(ProgressLog {
                path: Some(path.to_path_buf()),
                steps: HashMap::new(),
            })
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
    /// Returns `true` if the step was recorded as OK, all output files exist
    /// with sizes matching the recorded values, and the resolved options
    /// match the current step options.
    pub fn is_step_fresh(
        &self,
        step_id: &str,
        current_options: Option<&HashMap<String, String>>,
    ) -> bool {
        let record = match self.get_step(step_id) {
            Some(r) if r.status == Status::Ok => r,
            _ => return false,
        };

        // Check whether resolved options changed since the last run
        if let Some(current) = current_options {
            if !record.resolved_options.is_empty() && &record.resolved_options != current {
                return false;
            }
        }

        for output in &record.outputs {
            let path = Path::new(&output.path);
            match std::fs::metadata(path) {
                Ok(meta) => {
                    if meta.len() != output.size {
                        return false;
                    }
                }
                Err(_) => return false,
            }
        }

        true
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
        let log = ProgressLog::load(Path::new("/nonexistent/.upstream.progress.yaml")).unwrap();
        assert!(log.steps.is_empty());
    }

    #[test]
    fn test_save_and_load() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join(".upstream.progress.yaml");

        let mut log = ProgressLog::load(&path).unwrap();
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

        let reloaded = ProgressLog::load(&path).unwrap();
        assert!(reloaded.get_step("test-step").is_some());
    }
}
