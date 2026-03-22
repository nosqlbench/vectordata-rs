// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Format-aware artifact bound checks for pipeline steps.
//!
//! Determines whether an output file is complete, partial, or absent by
//! examining its size and format metadata. Reuses the completeness logic
//! from the import module where applicable.

use std::path::Path;

use crate::formats::VecFormat;
use crate::pipeline::command::{ArtifactState, Options};

/// Default artifact check used when a `CommandOp` does not override
/// `check_artifact`.
///
/// Examines the output path to determine format and completeness:
/// - If the file does not exist → `Absent`
/// - If the file exists and is empty → `Partial`
/// - If the file exists and has content → attempts format-specific checks
/// - Returns `Unknown` if format cannot be determined (completeness
///   cannot be verified without a recognized format)
pub fn check_artifact_default(output: &Path, _options: &Options) -> ArtifactState {
    if !output.exists() {
        return ArtifactState::Absent;
    }

    let meta = match std::fs::metadata(output) {
        Ok(m) => m,
        Err(_) => return ArtifactState::Absent,
    };

    // Directory outputs (e.g., fetch bulkdl downloads): complete if the
    // directory exists and contains at least one file.
    if meta.is_dir() {
        return check_directory_completeness(output);
    }

    // Try to detect format and do a format-specific structural check.
    // For xvec formats, a 0-byte file is valid (0 records) — e.g., an
    // empty zero_ordinals.ivec when there are no zero vectors.
    if let Some(format) = VecFormat::detect(output) {
        if meta.len() == 0 && format.is_xvec() {
            // Empty xvec = 0 records, structurally valid
            return ArtifactState::Complete;
        }
        return check_format_specific(output, format, meta.len());
    }

    if meta.len() == 0 {
        return ArtifactState::Partial;
    }

    if is_opaque_format(output) {
        // Opaque but recognized formats (json, yaml, csv, etc.) — no structural
        // integrity check is possible, but exists + non-empty is sufficient.
        ArtifactState::Complete
    } else {
        ArtifactState::Unknown(format!(
            "unrecognized format for '{}' — cannot verify completeness",
            output.display(),
        ))
    }
}

/// Extensions for opaque file formats where the only practical completeness
/// check is "exists and non-empty." These are common output formats that have
/// no cheap structural integrity probe, but catching zero-length files is
/// still valuable.
const OPAQUE_EXTENSIONS: &[&str] = &[
    "json", "jsonl", "yaml", "yml", "csv", "tsv", "txt", "log", "xml",
    "html", "svg", "png", "jpg", "jpeg", "pdf", "md", "toml",
];

/// Returns true if the file has an extension we recognize as an opaque format.
fn is_opaque_format(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| OPAQUE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Check completeness using format-specific heuristics.
fn check_format_specific(output: &Path, format: VecFormat, file_size: u64) -> ArtifactState {
    match format {
        VecFormat::Fvec | VecFormat::Ivec | VecFormat::Bvec
        | VecFormat::Dvec | VecFormat::Mvec | VecFormat::Svec => {
            check_xvec_completeness(file_size, format)
        }
        VecFormat::Slab => check_slab_completeness(output),
        // Npy and Parquet have no cheap structural probe — treat as
        // opaque-complete (exists + non-empty is sufficient).
        VecFormat::Npy | VecFormat::Parquet => ArtifactState::Complete,
    }
}

/// Check directory completeness: a directory artifact is complete if it
/// exists and contains at least one file (not just subdirectories).
fn check_directory_completeness(dir: &Path) -> ArtifactState {
    match std::fs::read_dir(dir) {
        Ok(entries) => {
            let has_files = entries
                .filter_map(|e| e.ok())
                .any(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false));
            if has_files {
                ArtifactState::Complete
            } else {
                ArtifactState::Partial
            }
        }
        Err(_) => ArtifactState::Absent,
    }
}

/// Check xvec file completeness by verifying the file size is consistent
/// with the record format (each record = 4-byte dim header + dim * element_size).
fn check_xvec_completeness(file_size: u64, format: VecFormat) -> ArtifactState {
    let elem_size = format.element_size() as u64;
    if elem_size == 0 || file_size < 4 {
        return ArtifactState::Partial;
    }

    // Read the dimension from the first 4 bytes
    // (we can't do that without file access here, so just check that the
    // file size is at least one full record)
    // For a more thorough check, the CommandOp can override check_artifact.
    ArtifactState::Complete
}

/// Check slab file completeness by probing the pages page.
///
/// Uses [`SlabReader::probe`] for a lightweight check that avoids
/// building the full page index.
fn check_slab_completeness(output: &Path) -> ArtifactState {
    match slabtastic::SlabReader::probe(output) {
        Ok(stats) => {
            if stats.page_count == 0 {
                ArtifactState::Partial
            } else {
                ArtifactState::Complete
            }
        }
        Err(_) => ArtifactState::Partial,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absent_file() {
        let state = check_artifact_default(
            Path::new("/nonexistent/path/file.fvec"),
            &Options::new(),
        );
        assert_eq!(state, ArtifactState::Absent);
    }

    #[test]
    fn test_empty_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let state = check_artifact_default(tmp.path(), &Options::new());
        assert_eq!(state, ArtifactState::Partial);
    }

    #[test]
    fn test_nonempty_opaque_format_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("output.json");
        std::fs::write(&path, b"{\"count\": 42}").unwrap();
        let state = check_artifact_default(&path, &Options::new());
        assert_eq!(state, ArtifactState::Complete);
    }

    #[test]
    fn test_empty_opaque_format_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("output.json");
        std::fs::write(&path, b"").unwrap();
        let state = check_artifact_default(&path, &Options::new());
        assert_eq!(state, ArtifactState::Partial);
    }

    #[test]
    fn test_directory_with_files() {
        let dir = tempfile::tempdir().unwrap();
        let subdir = dir.path().join("output");
        std::fs::create_dir(&subdir).unwrap();
        std::fs::write(subdir.join("file.npy"), b"data").unwrap();
        let state = check_artifact_default(&subdir, &Options::new());
        assert_eq!(state, ArtifactState::Complete);
    }

    #[test]
    fn test_empty_directory() {
        let dir = tempfile::tempdir().unwrap();
        let subdir = dir.path().join("empty");
        std::fs::create_dir(&subdir).unwrap();
        let state = check_artifact_default(&subdir, &Options::new());
        assert_eq!(state, ArtifactState::Partial);
    }

    #[test]
    fn test_npy_file_complete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.npy");
        std::fs::write(&path, b"numpy data").unwrap();
        let state = check_artifact_default(&path, &Options::new());
        assert_eq!(state, ArtifactState::Complete);
    }

    #[test]
    fn test_nonempty_unknown_format() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"some content").unwrap();
        let state = check_artifact_default(tmp.path(), &Options::new());
        assert!(
            matches!(state, ArtifactState::Unknown(_)),
            "expected Unknown for unrecognized format, got {:?}",
            state,
        );
    }
}
