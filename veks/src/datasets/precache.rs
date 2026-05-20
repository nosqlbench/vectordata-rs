// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks datasets precache` — thin delegate over
//! [`vectordata::datasets::precache::run`].
//!
//! The implementation (spec resolution, live progress meter,
//! per-facet + aggregate rendering) lives in the vectordata crate so
//! `vectordata datasets precache …` and `veks datasets precache …`
//! produce identical behaviour.

use std::path::Path;

/// Entry point for `veks datasets precache`. Exits the process
/// with the code returned by the shared implementation.
pub fn run(
    dataset_spec: &str,
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
    cache_dir: Option<&Path>,
) {
    let code = vectordata::datasets::precache::run(
        dataset_spec, configdir, extra_catalogs, at, cache_dir);
    if code != 0 { std::process::exit(code); }
}

#[cfg(test)]
mod tests {
    #[test]
    fn local_dataset_directory_prebuffers_via_canonical_path() {
        // A local dataset with a dataset.yaml — every facet should
        // resolve to Storage::Mmap and precache should be a no-op.
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();

        std::fs::write(ws.join("base.fvec"), {
            let mut buf = Vec::new();
            buf.extend(&2i32.to_le_bytes());
            buf.extend(&1.0f32.to_le_bytes());
            buf.extend(&2.0f32.to_le_bytes());
            buf.extend(&2i32.to_le_bytes());
            buf.extend(&3.0f32.to_le_bytes());
            buf.extend(&4.0f32.to_le_bytes());
            buf
        }).unwrap();
        std::fs::write(ws.join("dataset.yaml"), "\
name: test
profiles:
  default:
    base_vectors: base.fvec
").unwrap();

        let group = vectordata::TestDataGroup::load(
            ws.join("dataset.yaml").to_str().unwrap(),
        ).unwrap();
        let view = group.profile("default").unwrap();
        view.prebuffer_all().unwrap();

        // The base.fvec original must still be exactly where we put
        // it — no shadow copy in any cache directory.
        assert!(ws.join("base.fvec").is_file());
    }
}
