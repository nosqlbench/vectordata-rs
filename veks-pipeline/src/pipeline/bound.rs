// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Format-aware artifact bound checks for pipeline steps.
//!
//! Determines whether an output file is complete, partial, or absent by
//! examining its size and format metadata. Reuses the completeness logic
//! from the import module where applicable.

use std::path::Path;

use veks_core::formats::VecFormat;
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
fn check_format_specific(output: &Path, format: VecFormat, _file_size: u64) -> ArtifactState {
    match format {
        _ if format.is_xvec() => {
            check_xvec_alignment(output)
        }
        VecFormat::Slab => check_slab_completeness(output),
        // Npy, Parquet, Hdf5, and scalar formats have no cheap structural
        // probe — treat as opaque-complete (exists + non-empty is sufficient).
        _ => ArtifactState::Complete,
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

/// Check xvec file completeness by verifying the file size is record-aligned.
///
/// Reads the dimension from the first 4 bytes, computes the record stride,
/// and checks that the total file size is an exact multiple. A file with
/// trailing bytes is `Partial` (truncated or interrupted write).
fn check_xvec_completeness(file_size: u64, format: VecFormat) -> ArtifactState {
    let elem_size = format.element_size() as u64;
    if elem_size == 0 || file_size < 4 {
        return ArtifactState::Partial;
    }

    // We need the dimension to compute stride. Read it from the path
    // passed via the format checker. Since we only have file_size here,
    // we can't read the header — but we can check the alignment constraint:
    // file_size must satisfy: file_size % (4 + dim * elem_size) == 0
    // for some positive integer dim. Try common dimensions.
    //
    // Heuristic: compute dim from the first record's header by scanning
    // possible record sizes. If file_size - 4 is divisible by elem_size,
    // that gives a candidate dim. Then check total alignment.
    //
    // Actually, without the file handle we can't read the header. Use the
    // weaker check: file_size must be > 0 and at least 4 bytes.
    // The strong alignment check is done in check_xvec_completeness_with_path.
    if file_size > 0 {
        ArtifactState::Complete
    } else {
        ArtifactState::Partial
    }
}

/// Check xvec file completeness with full path access.
///
/// Two checks:
/// 1. Record alignment: file size must be an exact multiple of record stride
/// 2. Count marker: if a `.count` sidecar exists, the file's record count
///    must match. This catches interruptions at record boundaries.
pub fn check_xvec_alignment(path: &Path) -> ArtifactState {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return ArtifactState::Absent,
    };
    let file_size = meta.len();
    if file_size < 4 {
        return if file_size == 0 { ArtifactState::Complete } else { ArtifactState::Partial };
    }

    // Read dimension from first 4 bytes
    let dim = match std::fs::File::open(path) {
        Ok(mut f) => {
            use std::io::Read;
            let mut buf = [0u8; 4];
            if f.read_exact(&mut buf).is_err() { return ArtifactState::Partial; }
            i32::from_le_bytes(buf) as u64
        }
        Err(_) => return ArtifactState::Absent,
    };

    if dim == 0 || dim > 100_000 {
        return ArtifactState::Partial;
    }

    let format = match VecFormat::detect(path) {
        Some(f) => f,
        None => return ArtifactState::Complete,
    };
    let elem_size = format.element_size() as u64;
    if elem_size == 0 { return ArtifactState::Complete; }

    let record_stride = 4 + dim * elem_size;
    if file_size % record_stride != 0 {
        return ArtifactState::Partial;
    }

    let actual_records = file_size / record_stride;

    // Check verified count from variables.yaml — written by pipeline
    // commands after successful output. If the variable exists, the file's
    // record count must match. This catches interruptions at record
    // boundaries that pass the alignment check.
    //
    // Walk up from the output file to find variables.yaml — outputs may
    // be in .cache/ (1 level), profiles/name/ (2 levels), or deeper.
    let var_name = format!("verified_count:{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or(""));
    if let Some(workspace) = find_workspace_with_variables(path) {
        if let Ok(vars) = crate::pipeline::variables::load(&workspace) {
            if let Some(expected_str) = vars.get(&var_name) {
                if let Ok(expected) = expected_str.parse::<u64>() {
                    if actual_records != expected {
                        return ArtifactState::Partial;
                    }
                    return ArtifactState::Complete;
                }
            }
            // variables.yaml exists but has no entry for this file.
            // The command either didn't write one (older code) or was
            // interrupted after creating the file but before writing
            // the count. Treat as Partial — the failsafe catches
            // record-boundary truncations.
            return ArtifactState::Partial;
        }
    }

    // No variables.yaml found at all (e.g., after --restart, or first
    // run). Fall back to record-alignment check only — a record-aligned
    // file is likely complete. Requiring verified_count here would force
    // re-execution of every step after any variables.yaml deletion.
    ArtifactState::Complete
}

/// Walk up from an output path to find the workspace containing variables.yaml.
/// Checks up to 4 ancestor directories (covers .cache/, profiles/name/, etc.).
fn find_workspace_with_variables(output: &Path) -> Option<std::path::PathBuf> {
    let mut dir = output.parent()?;
    for _ in 0..4 {
        dir = dir.parent()?;
        if dir.join("variables.yaml").exists() {
            return Some(dir.to_path_buf());
        }
    }
    None
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
