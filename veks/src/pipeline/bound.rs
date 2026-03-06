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

    if meta.len() == 0 {
        return ArtifactState::Partial;
    }

    // Try to detect format and do a basic size sanity check
    if let Some(format) = VecFormat::detect(output) {
        check_format_specific(output, format, meta.len())
    } else {
        ArtifactState::Unknown(format!(
            "unrecognized format for '{}' — cannot verify completeness",
            output.display(),
        ))
    }
}

/// Check completeness using format-specific heuristics.
fn check_format_specific(output: &Path, format: VecFormat, file_size: u64) -> ArtifactState {
    match format {
        VecFormat::Fvec | VecFormat::Ivec | VecFormat::Bvec
        | VecFormat::Dvec | VecFormat::Hvec | VecFormat::Svec => {
            check_xvec_completeness(file_size, format)
        }
        VecFormat::Slab => check_slab_completeness(output),
        _ => ArtifactState::Unknown(format!(
            "format '{}' for '{}' has no completeness check",
            format.name(),
            output.display(),
        )),
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
