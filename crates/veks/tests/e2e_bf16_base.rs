// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end: a dataset built over a bf16 base (SRD element-access
//! §13.10, "one e2e with a bf16 base whose KNN equals the f32 build's").
//!
//! The base values are chosen exactly representable in bf16 — half
//! steps below 128, where bf16's eight significant bits are exact — so
//! the bf16 build and the f32 build of the same values answer the same
//! neighbours, and the comparison is of the pipeline's type handling,
//! not of rounding.
//!
//! This runs the `veks` binary, as `e2e_shard_sizing` does, so it covers
//! the whole path from the bootstrap to the facets on disk.

use std::path::{Path, PathBuf};
use std::process::Command;

use veks::prepare::import::ImportArgs;
use vectordata::TestDataViewExt;
use vectordata::typed_access::{Element, bf16};

#[path = "support/catalog_mirror.rs"]
mod catalog_mirror;
#[path = "support/samples.rs"]
mod samples;

fn veks_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_veks"))
}

/// Base vector `i`, element `d`: distinct half-integer values below 128.
fn base_value(i: usize, d: usize) -> f32 {
    (i * 4 + d) as f32 * 0.5
}

/// Query vector `q`, element `d`: quarter-integer values below 32, so no
/// query coincides with a base vector.
fn query_value(q: usize, d: usize) -> f32 {
    (q * 4 + d) as f32 * 0.5 + 0.25
}

fn args_for(name: &str, output: &Path, base: &Path, query: &Path) -> ImportArgs {
    ImportArgs {
        name: name.to_string(),
        output: output.to_path_buf(),
        base_vectors: Some(base.to_path_buf()),
        query_vectors: Some(query.to_path_buf()),
        self_search: false,
        query_count: 10,
        metadata: None,
        ground_truth: None,
        ground_truth_distances: None,
        metric: "L2".to_string(),
        neighbors: 5,
        seed: 0,
        description: None,
        no_dedup: true,
        no_filtered: true,
        no_zero_check: true,
        duplicate_count: Some(0),
        zero_count: Some(0),
        normalize: false,
        force: true,
        base_convert_format: None,
        query_convert_format: None,
        compress_cache: false,
        sized_profiles: None,
        base_fraction: 1.0,
        required_facets: Some("BQGD".to_string()),
        round_digits: 10,
        max_shard_bytes: vectordata::dataset::DEFAULT_MAX_SHARD_BYTES,
        pedantic_dedup: false,
        selectivity: 0.0001,
        predicate_count: 100,
        predicate_strategy: "eq".to_string(),
        provided_facets: None,
        classic: false,
        personality: "native".to_string(),
        synthesize_metadata: false,
        synthesis_mode: "simple-int-eq".to_string(),
        synthesis_format: "slab".to_string(),
        metadata_fields: 3,
        metadata_range_min: 0,
        metadata_range_max: 1000,
        predicate_range_min: 0,
        predicate_range_max: 1000,
        verify_knn_sample: 0,
        partition_oracles: false,
        max_partitions: 100,
        on_undersized: "error".to_string(),
        cosine_mode: None,
    }
}

fn run_pipeline(dataset_yaml: &Path) -> (bool, String) {
    catalog_mirror::mark_publishable(dataset_yaml.parent().expect("dataset.yaml has a directory"));
    let out = Command::new(veks_bin())
        .arg("run")
        .arg("--output").arg("batch")
        .arg("--threads").arg("2")
        .arg(dataset_yaml)
        .output()
        .expect("failed to execute veks");
    let text = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if out.status.success() {
        catalog_mirror::assert_projections_mirror_dataset_yaml(dataset_yaml.parent().unwrap());
    }
    (out.status.success(), text)
}

/// Bootstrap and build a dataset whose base is 60 vectors of dimension
/// 4 at `T` and whose queries are 10 f32 vectors, returning the dataset
/// directory.
fn build<T: Element>(tmp: &Path, name: &str) -> PathBuf {
    let base = tmp.join(format!("base.{}", samples::vec_extension::<T>()));
    let query = tmp.join("query.f32vecs");
    samples::write_vectors::<T>(&base, 60, 4, |i, d| {
        T::from_value(vectordata::typed_access::Value::Float(base_value(i, d) as f64)).unwrap()
    });
    samples::write_vectors::<f32>(&query, 10, 4, query_value);
    let out = tmp.join(name);
    veks::prepare::import::run(args_for(name, &out, &base, &query));
    let yaml = out.join("dataset.yaml");
    assert!(yaml.exists(), "{name}: bootstrap produced no dataset.yaml");
    let (ok, log) = run_pipeline(&yaml);
    assert!(ok, "{name}: pipeline failed:\n{log}");
    out
}

/// **A bf16 base builds, loads as f32, and its ground truth is the f32
/// build's.** The base facet is a `.bf16vecs` file; the dataset opens
/// it through `base_vectors_as::<f32>` by the widening the conversion
/// table grants, and the neighbour indices the KNN wrote over it equal
/// those written over the same values held as f32.
#[test]
fn a_bf16_base_builds_loads_as_f32_and_answers_the_f32_builds_neighbours() {
    let tmp = tempfile::tempdir().unwrap();
    let as_bf16 = build::<bf16>(tmp.path(), "bf16-base");
    let as_f32 = build::<f32>(tmp.path(), "f32-base");

    // Every bf16 base value is the f32 value, exactly.
    let group = vectordata::TestDataGroup::load(as_bf16.to_str().unwrap()).expect("the bf16 dataset loads");
    let view = group.profile("default").expect("default profile");
    let base = view.base_vectors_as::<f32>().expect("a bf16 base opens as f32");
    assert_eq!((base.dim(), base.count()), (4, 60));
    for i in 0..60 {
        let got = base.get(i).unwrap();
        let want: Vec<f32> = (0..4).map(|d| base_value(i, d)).collect();
        assert_eq!(got, want, "record {i}");
    }

    // And the neighbours agree with the f32 build's, row for row.
    let f32_group = vectordata::TestDataGroup::load(as_f32.to_str().unwrap()).unwrap();
    let f32_view = f32_group.profile("default").unwrap();
    let a = view.neighbor_indices_as::<i32>().expect("bf16 build's neighbours");
    let b = f32_view.neighbor_indices_as::<i32>().expect("f32 build's neighbours");
    assert_eq!((a.dim(), a.count()), (b.dim(), b.count()));
    assert_eq!(a.count(), 10, "one row per query");
    for q in 0..a.count() {
        assert_eq!(a.get(q).unwrap(), b.get(q).unwrap(), "query {q}");
    }
}
