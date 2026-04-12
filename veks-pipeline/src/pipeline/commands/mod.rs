// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Built-in pipeline command implementations.
//!
//! Each command wraps an existing veks operation or provides a new Rust-native
//! implementation behind the `CommandOp` trait, extracting options from the
//! pipeline `Options` map and delegating to the underlying logic.

// Removed: barrier, cleanup_cleanfvec
pub mod catalog_generate;
pub mod catalog_stats;
pub mod require;
pub mod source_window;
pub mod analyze_checkendian;
pub mod analyze_compare;
#[cfg(feature = "knnutils")]
pub mod analyze_fvecs_check_knnutils;
pub mod analyze_explore;
pub mod analyze_find_duplicates;
pub mod analyze_find_zeros;
pub mod analyze_find;
pub mod analyze_histogram;
pub mod analyze_modeldiff;
pub mod analyze_plot;
pub mod analyze_select;
pub mod analyze_slice;
pub mod analyze_stats;
pub mod analyze_verifyknn;
pub mod analyze_verifyprofiles;
pub mod analyze_normals;
#[cfg(feature = "knnutils")]
pub mod analyze_normals_knnutils;
pub mod analyze_norms;
pub mod analyze_overlap;
pub mod analyze_zeros;
pub mod cleanup_overlap;
pub mod clean_ordinals;
pub mod compute_dedup;
pub mod dataset_json;
pub mod compute_filtered_knn;
pub mod compute_knn;
#[cfg(feature = "knnutils")]
pub mod compute_knn_blas;
#[cfg(feature = "faiss")]
pub mod compute_knn_faiss;
#[cfg(feature = "knnutils")]
pub mod compute_sort_knnutils;
pub mod compute_sort;
pub mod config;
mod convert;
mod describe;
pub mod fetch_bulkdl;
pub mod fetch_dlhf;
pub mod gen_dataset;
pub mod gen_derive;
pub mod gen_extract;
pub mod gen_metadata;
pub mod gen_from_model;
pub mod gen_predicate_keys;
pub mod inspect_filtered_knn;
pub mod gen_predicates;
pub mod gen_shuffle;
#[cfg(feature = "knnutils")]
pub mod gen_shuffle_knnutils;
pub mod gen_sketch;
pub mod gen_dataset_log_jsonl;
pub mod gen_variables_json;
pub mod gen_vectors;
pub mod gen_vvec_index;
mod import;
pub mod info_compute;
pub mod info_file;
pub mod inspect_predicate;
pub mod json_jjq;
pub mod json_rjq;
pub mod merkle;
pub mod set_variable;
pub mod slab;
pub mod verify_consolidated;
#[cfg(feature = "knnutils")]
pub mod transform_normalize_knnutils;
#[cfg(feature = "knnutils")]
pub mod transform_remove_zeros_knnutils;
#[cfg(feature = "knnutils")]
pub mod verify_dataset_knnutils;
pub mod verify_knn;
pub mod verify_predicates;
pub mod verify_predicates_sqlite;

use super::registry::CommandRegistry;

/// Register all built-in commands with the given registry.
///
/// Command paths follow the verb-noun convention established in SRD §1.5:
/// group names are verbs, subcommand names are nouns/noun-phrases.
pub fn register_all(registry: &mut CommandRegistry) {
    // ── analyze ──────────────────────────────────────────────────────
    registry.register("analyze check-endian", analyze_checkendian::factory);
    registry.register("analyze compare-files", analyze_compare::factory);
    #[cfg(feature = "knnutils")]
    registry.register("analyze fvecs-check-knnutils", analyze_fvecs_check_knnutils::factory);
    registry.register("analyze compute-info", info_compute::factory);
    registry.register("analyze describe", describe::factory);
    registry.register("analyze file", info_file::factory);
    registry.register("analyze find", analyze_find::factory);
    registry.register("analyze find-duplicates", analyze_find_duplicates::factory);
    registry.register("analyze find-zeros", analyze_find_zeros::factory);
    registry.register("analyze display-histogram", analyze_histogram::factory);
    registry.register("analyze model-diff", analyze_modeldiff::factory);
    registry.register("analyze explain-predicates", inspect_predicate::factory);
    registry.register("analyze explain-filtered-knn", inspect_filtered_knn::factory);
    registry.register("analyze select", analyze_select::factory);
    registry.register("analyze slice", analyze_slice::factory);
    registry.register("analyze stats", analyze_stats::factory);
    registry.register("analyze survey", slab::survey_factory);
    registry.register("analyze verify-knn", analyze_verifyknn::factory);
    registry.register("analyze verify-profiles", analyze_verifyprofiles::factory);
    registry.register("analyze measure-normals", analyze_normals::factory);
    #[cfg(feature = "knnutils")]
    registry.register("analyze check-normalization-knnutils", analyze_normals_knnutils::factory);
    registry.register("analyze display-norms", analyze_norms::factory);
    registry.register("analyze overlap", analyze_overlap::factory);
    registry.register("analyze zeros", analyze_zeros::factory);

    // ── cleanup ─────────────────────────────────────────────────────
    registry.register("cleanup overlap", cleanup_overlap::factory);

    // ── compute ──────────────────────────────────────────────────────
    registry.register("compute evaluate-predicates", gen_predicate_keys::factory);
    registry.register("compute filtered-knn", compute_filtered_knn::factory);
    registry.register("compute knn", compute_knn::factory);
    #[cfg(feature = "knnutils")]
    registry.register("compute knn-blas", compute_knn_blas::factory);
    #[cfg(feature = "faiss")]
    registry.register("compute knn-faiss", compute_knn_faiss::factory);
    registry.register("compute sort", compute_dedup::factory);
    #[cfg(feature = "knnutils")]
    registry.register("compute sort-knnutils", compute_sort_knnutils::factory);

    // ── download ─────────────────────────────────────────────────────
    registry.register("download bulk", fetch_bulkdl::factory);
    registry.register("download huggingface", fetch_dlhf::factory);

    // ── generate ─────────────────────────────────────────────────────
    registry.register("generate dataset", gen_dataset::factory);
    registry.register("generate dataset-json", dataset_json::factory);
    registry.register("generate derive", gen_derive::factory);
    registry.register("generate from-model", gen_from_model::factory);
    registry.register("generate metadata", gen_metadata::factory);
    registry.register("generate predicates", gen_predicates::factory);
    registry.register("generate shuffle", gen_shuffle::factory);
    #[cfg(feature = "knnutils")]
    registry.register("generate shuffle-knnutils", gen_shuffle_knnutils::factory);
    registry.register("generate sketch", gen_sketch::factory);
    registry.register("generate dataset-log-jsonl", gen_dataset_log_jsonl::factory);
    registry.register("generate variables-json", gen_variables_json::factory);
    registry.register("generate vectors", gen_vectors::factory);
    registry.register("generate vvec-index", gen_vvec_index::factory);

    // ── merkle ───────────────────────────────────────────────────────
    registry.register("merkle create", merkle::create_factory);
    registry.register("merkle diff", merkle::diff_factory);
    registry.register("merkle path", merkle::path_factory);
    registry.register("merkle spoilbits", merkle::spoilbits_factory);
    registry.register("merkle spoilchunks", merkle::spoilchunks_factory);
    registry.register("merkle summary", merkle::summary_factory);
    registry.register("merkle treeview", merkle::treeview_factory);
    registry.register("merkle verify", merkle::verify_factory);

    // ── query ────────────────────────────────────────────────────────
    registry.register("query json", json_jjq::factory);
    registry.register("query records", json_rjq::factory);

    // ── state ────────────────────────────────────────────────────────
    registry.register("state set", set_variable::factory);
    registry.register("state clear", set_variable::clear_factory);

    // ── transform ────────────────────────────────────────────────────
    registry.register("transform convert", convert::factory);
    registry.register("transform extract", gen_extract::extract_factory);
    registry.register("transform ordinals", clean_ordinals::factory);
    #[cfg(feature = "knnutils")]
    registry.register("transform normalize-knnutils", transform_normalize_knnutils::factory);
    #[cfg(feature = "knnutils")]
    registry.register("transform remove-zeros-knnutils", transform_remove_zeros_knnutils::factory);

    // ── verify ───────────────────────────────────────────────────────
    #[cfg(feature = "knnutils")]
    registry.register("verify dataset-knnutils", verify_dataset_knnutils::factory);
    registry.register("verify knn-groundtruth", verify_knn::factory);
    registry.register("verify predicate-results", verify_predicates::factory);
    registry.register("verify knn-consolidated", verify_consolidated::knn_consolidated_factory);
    registry.register("verify filtered-knn-consolidated", verify_consolidated::filtered_knn_consolidated_factory);
    registry.register("verify predicates-consolidated", verify_consolidated::predicates_consolidated_factory);
    registry.register("verify predicates-sqlite", verify_predicates_sqlite::factory);

    // ── catalog ──────────────────────────────────────────────────────
    registry.register("catalog generate", catalog_generate::factory);
    registry.register("catalog stats", catalog_stats::factory);

    // ── pipeline orchestration ─────────────────────────────────────
    registry.register("pipeline require", require::factory);
}
