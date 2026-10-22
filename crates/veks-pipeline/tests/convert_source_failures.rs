// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `transform convert` fails on a source file it cannot read
//! (`docs/design/srd-crate-boundaries.md` PR-8). The npy directory
//! reader skipped a file that failed to load, so the stream ended early
//! and the conversion reported success with the records before it.

mod support;

use std::path::Path;

use veks_io::VecFormat;
use veks_pipeline::pipeline::command::Status;

use support::{options, run_command};

const DIM: usize = 2;
const ROWS: usize = 4;

/// An f32 npy file of `ROWS × DIM`, of which `keep` data bytes are
/// written: a short `keep` makes a file whose header scans and whose
/// load fails.
fn write_npy(path: &Path, first: f32, keep: usize) {
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({ROWS}, {DIM}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    let body: Vec<u8> = (0..ROWS * DIM).flat_map(|i| (first + i as f32).to_le_bytes()).collect();
    bytes.extend_from_slice(&body[..keep.min(body.len())]);
    std::fs::write(path, bytes).unwrap();
}

/// Three npy files, the middle one cut short after its header.
fn corrupt_npy_dir(root: &Path) -> std::path::PathBuf {
    let dir = root.join("embeddings");
    std::fs::create_dir_all(&dir).unwrap();
    write_npy(&dir.join("part_0.npy"), 0.0, usize::MAX);
    write_npy(&dir.join("part_1.npy"), 100.0, 5);
    write_npy(&dir.join("part_2.npy"), 200.0, usize::MAX);
    dir
}

/// **Converting a corrupt npy directory into a slab** — the streaming
/// path, record by record — fails, naming the file, and reports no
/// success with the records before it.
#[test]
fn converting_a_corrupt_npy_directory_through_the_stream_fails() {
    let tmp = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let source = corrupt_npy_dir(tmp.path());
    let output = tmp.path().join("base.slab");
    let run = run_command(
        "transform convert",
        tmp.path(),
        &options(&[
            ("source", &source.display().to_string()),
            ("output", &output.display().to_string()),
            ("from", "npy"),
        ]),
    );
    assert_eq!(run.result.status, Status::Error, "the conversion failed: {}", run.result.message);
    assert!(run.result.message.contains("part_1.npy"), "the error names the file: {}", run.result.message);
    assert!(
        veks_io::open_format(&output, VecFormat::Slab).is_err(),
        "no slab of the records before the corrupt file is left as the output"
    );
}

/// **Converting a corrupt npy directory into an fvec** — the parallel
/// path, which loads the files itself — fails too.
#[test]
fn converting_a_corrupt_npy_directory_in_parallel_fails() {
    let tmp = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let source = corrupt_npy_dir(tmp.path());
    let output = tmp.path().join("base.fvec");
    let run = run_command(
        "transform convert",
        tmp.path(),
        &options(&[
            ("source", &source.display().to_string()),
            ("output", &output.display().to_string()),
            ("from", "npy"),
        ]),
    );
    assert_eq!(run.result.status, Status::Error, "the conversion failed: {}", run.result.message);
}
