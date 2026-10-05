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
        // A facet written across shards has nothing at the unsharded
        // name (SH-35). It is present — as much as any facet is — and
        // reporting it Absent would make the step that produced it
        // look like it did nothing. Each shard is an ordinary file of
        // the format, so each is checked as one.
        let shards = vectordata::dataset::discover_shards(output);
        if !shards.is_empty() {
            return check_series(output, &shards, _options);
        }
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
    // empty zero_ordinals.ivecs when there are no zero vectors.
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
    match xvec_record_count(path) {
        Ok(records) => verified_count_state(path, records),
        Err(state) => state,
    }
}

/// A facet written across shards (SH-35): every shard must be a
/// structurally complete file of the format, and the series as a
/// whole — not any one shard — is held to the verified count recorded
/// for its unsharded name. Holding each shard to the series' count, or
/// to an entry of its own that no producer writes, called every
/// complete series partial.
fn check_series(series: &Path, shards: &[std::path::PathBuf], options: &Options) -> ArtifactState {
    let xvec = VecFormat::detect(series).is_some_and(|f| f.is_xvec());
    if !xvec {
        return shards
            .iter()
            .map(|s| check_artifact_default(s, options))
            .find(|state| *state != ArtifactState::Complete)
            .unwrap_or(ArtifactState::Complete);
    }
    let mut total = 0u64;
    for shard in shards {
        match xvec_record_count(shard) {
            Ok(records) => total += records,
            Err(state) => return state,
        }
    }
    verified_count_state(series, total)
}

/// The records in an xvec file whose size is record-aligned; the state
/// it is in otherwise.
fn xvec_record_count(path: &Path) -> Result<u64, ArtifactState> {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return Err(ArtifactState::Absent),
    };
    let file_size = meta.len();
    if file_size < 4 {
        return if file_size == 0 { Ok(0) } else { Err(ArtifactState::Partial) };
    }

    // Read dimension from first 4 bytes
    let dim = match std::fs::File::open(path) {
        Ok(mut f) => {
            use std::io::Read;
            let mut buf = [0u8; 4];
            if f.read_exact(&mut buf).is_err() {
                return Err(ArtifactState::Partial);
            }
            i32::from_le_bytes(buf) as u64
        }
        Err(_) => return Err(ArtifactState::Absent),
    };

    if dim == 0 || dim > 100_000 {
        return Err(ArtifactState::Partial);
    }

    let format = match VecFormat::detect(path) {
        Some(f) => f,
        None => return Ok(0),
    };
    let elem_size = format.element_size() as u64;
    if elem_size == 0 {
        return Ok(0);
    }

    let record_stride = 4 + dim * elem_size;
    if file_size % record_stride != 0 {
        return Err(ArtifactState::Partial);
    }
    Ok(file_size / record_stride)
}

/// Hold an xvec artifact's record count to the `verified_count:<name>`
/// its producer recorded in `variables.yaml`, when it recorded one — a
/// count written after the output, so a mismatch is an interruption at
/// a record boundary that alignment cannot see. No entry is no
/// evidence: an older producer wrote none, a shard of a series has
/// none of its own, and the runner's own record already guards a
/// completed step against interruption.
///
/// `variables.yaml` is looked for up to four directories above the
/// artifact — outputs may be in `.cache/`, `profiles/name/`, or
/// deeper.
fn verified_count_state(path: &Path, actual_records: u64) -> ArtifactState {
    let var_name = format!("verified_count:{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or(""));
    if let Some(workspace) = find_workspace_with_variables(path)
        && let Ok(vars) = crate::pipeline::variables::load(&workspace)
        && let Some(expected_str) = vars.get(&var_name)
        && let Ok(expected) = expected_str.parse::<u64>()
        && actual_records != expected
    {
        return ArtifactState::Partial;
    }
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

    fn write_fvecs(path: &Path, dim: usize, records: usize) {
        let mut bytes = Vec::new();
        for r in 0..records {
            bytes.extend_from_slice(&(dim as i32).to_le_bytes());
            for d in 0..dim {
                bytes.extend_from_slice(&((r * dim + d) as f32).to_le_bytes());
            }
        }
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    fn workspace_with(vars: &[(&str, &str)]) -> tempfile::TempDir {
        let ws = tempfile::tempdir().unwrap();
        let mut map = indexmap::IndexMap::new();
        for (k, v) in vars {
            map.insert((*k).to_string(), (*v).to_string());
        }
        crate::pipeline::variables::save(ws.path(), &map).unwrap();
        ws
    }

    /// A sharded series is complete when every shard aligns and the
    /// shards together hold the count verified for the series' own
    /// name; no shard is held to that count, and none needs an entry
    /// of its own.
    #[test]
    fn a_sharded_series_is_judged_as_a_whole() {
        let ws = workspace_with(&[("verified_count:base.fvecs", "6"), ("base_count", "6")]);
        let series = ws.path().join("profiles/base/base.fvecs");
        write_fvecs(&ws.path().join("profiles/base/base__0000.fvecs"), 2, 4);
        write_fvecs(&ws.path().join("profiles/base/base__0001.fvecs"), 2, 2);
        assert_eq!(check_artifact_default(&series, &Options::new()), ArtifactState::Complete);

        // Fewer records than verified: an interruption at a shard boundary.
        std::fs::remove_file(ws.path().join("profiles/base/base__0001.fvecs")).unwrap();
        assert_eq!(check_artifact_default(&series, &Options::new()), ArtifactState::Partial);

        // A misaligned shard is partial whatever the count says.
        write_fvecs(&ws.path().join("profiles/base/base__0001.fvecs"), 2, 2);
        let mut bytes = std::fs::read(ws.path().join("profiles/base/base__0001.fvecs")).unwrap();
        bytes.pop();
        std::fs::write(ws.path().join("profiles/base/base__0001.fvecs"), bytes).unwrap();
        assert_eq!(check_artifact_default(&series, &Options::new()), ArtifactState::Partial);
    }

    /// A record-aligned xvec with no verified count on file is
    /// complete: the absence of an entry is not evidence.
    #[test]
    fn an_aligned_xvec_without_a_verified_count_is_complete() {
        let ws = workspace_with(&[("base_count", "3")]);
        let path = ws.path().join(".cache/other.fvecs");
        write_fvecs(&path, 3, 3);
        assert_eq!(check_artifact_default(&path, &Options::new()), ArtifactState::Complete);
        // With an entry that disagrees, it is partial.
        let ws = workspace_with(&[("verified_count:other.fvecs", "4")]);
        let path = ws.path().join(".cache/other.fvecs");
        write_fvecs(&path, 3, 3);
        assert_eq!(check_artifact_default(&path, &Options::new()), ArtifactState::Partial);
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
