// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Built-in pipeline command implementations.
//!
//! Each command wraps an existing veks operation or provides a new Rust-native
//! implementation behind the `CommandOp` trait, extracting options from the
//! pipeline `Options` map and delegating to the underlying logic.

pub mod analyze_explore;
pub mod analyze_checkendian;
pub mod analyze_find;
pub mod analyze_flamegraph;
pub mod analyze_modeldiff;
pub mod analyze_plot;
pub mod analyze_profile;
pub mod analyze_verifyprofiles;
pub mod analyze_compare;
pub mod analyze_histogram;
pub mod analyze_select;
pub mod analyze_slice;
pub mod analyze_stats;
pub mod analyze_verifyknn;
pub mod analyze_zeros;
pub mod cleanup_cleanfvec;
pub mod compute_knn;
pub mod compute_sort;
mod config;
mod convert;
mod catalog;
pub mod datasets_cache;
pub mod datasets_curlify;
pub mod datasets_list;
pub mod datasets_plan;
pub mod datasets_prebuffer;
pub mod fetch_dlhf;
mod describe;
pub mod gen_dataset;
pub mod gen_derive;
pub mod gen_extract;
pub mod gen_from_model;
pub mod gen_predicated;
pub mod gen_shuffle;
pub mod gen_sketch;
pub mod gen_vectors;
pub mod info_compute;
pub mod info_file;
mod import;
pub mod json_jjq;
pub mod json_rjq;
pub mod merkle;
pub mod slab;

use super::registry::CommandRegistry;

/// Register all built-in commands with the given registry.
pub fn register_all(registry: &mut CommandRegistry) {
    // Phase 2: wrappers for existing commands
    registry.register("import facet", import::factory);
    registry.register("convert file", convert::factory);
    registry.register("analyze describe", describe::factory);

    // Phase 3: generation commands
    registry.register("generate vectors", gen_vectors::factory);
    registry.register("generate ivec-shuffle", gen_shuffle::factory);
    registry.register("generate fvec-extract", gen_extract::fvec_factory);
    registry.register("generate ivec-extract", gen_extract::ivec_factory);
    registry.register("generate sketch", gen_sketch::factory);
    registry.register("generate from-model", gen_from_model::factory);

    // Phase 4: compute commands
    registry.register("compute knn", compute_knn::factory);
    registry.register("compute sort", compute_sort::factory);

    // Phase 5: analysis, info, and cleanup commands
    registry.register("analyze verify-knn", analyze_verifyknn::factory);
    registry.register("analyze stats", analyze_stats::factory);
    registry.register("analyze histogram", analyze_histogram::factory);
    registry.register("info file", info_file::factory);
    registry.register("info compute", info_compute::factory);
    registry.register("cleanup cleanfvec", cleanup_cleanfvec::factory);

    // Phase 5 batch 2: merkle, datasets, json
    registry.register("merkle create", merkle::create_factory);
    registry.register("merkle verify", merkle::verify_factory);
    registry.register("merkle diff", merkle::diff_factory);
    registry.register("merkle summary", merkle::summary_factory);
    registry.register("merkle treeview", merkle::treeview_factory);
    registry.register("datasets list", datasets_list::factory);
    registry.register("json jjq", json_jjq::factory);
    registry.register("json rjq", json_rjq::factory);

    // Phase 5 batch 3: additional analyze commands
    registry.register("analyze slice", analyze_slice::factory);
    registry.register("analyze check-endian", analyze_checkendian::factory);
    registry.register("analyze zeros", analyze_zeros::factory);
    registry.register("analyze compare", analyze_compare::factory);

    // Phase 5 batch 4: generate dataset, merkle path, config
    registry.register("generate dataset", gen_dataset::factory);
    registry.register("generate derive", gen_derive::factory);
    registry.register("merkle path", merkle::path_factory);
    registry.register("config show", config::show_factory);
    registry.register("config init", config::init_factory);

    // Phase 5 batch 5: analyze select, merkle spoil, config list-mounts
    registry.register("analyze select", analyze_select::factory);
    registry.register("merkle spoilbits", merkle::spoilbits_factory);
    registry.register("merkle spoilchunks", merkle::spoilchunks_factory);
    registry.register("config list-mounts", config::list_mounts_factory);

    // Phase 5 batch 7: analyze find/profile/model-diff/verifyprofiles, datasets, catalog
    registry.register("analyze explore", analyze_explore::factory);
    registry.register("analyze find", analyze_find::factory);
    registry.register("analyze profile", analyze_profile::factory);
    registry.register("analyze model-diff", analyze_modeldiff::factory);
    registry.register("analyze verify-profiles", analyze_verifyprofiles::factory);
    registry.register("datasets plan", datasets_plan::factory);
    registry.register("datasets cache", datasets_cache::factory);
    registry.register("catalog generate", catalog::factory);
    registry.register("datasets curlify", datasets_curlify::factory);
    registry.register("datasets prebuffer", datasets_prebuffer::factory);
    registry.register("fetch dlhf", fetch_dlhf::factory);

    // Phase 5 batch 6: slab commands
    registry.register("slab import", slab::import_factory);
    registry.register("slab export", slab::export_factory);
    registry.register("slab append", slab::append_factory);
    registry.register("slab rewrite", slab::rewrite_factory);
    registry.register("slab check", slab::check_factory);
    registry.register("slab get", slab::get_factory);
    registry.register("slab analyze", slab::analyze_factory);
    registry.register("slab explain", slab::explain_factory);
    registry.register("slab namespaces", slab::namespaces_factory);
    registry.register("slab inspect", slab::inspect_factory);

    // Phase 6: visualization and predicated datasets
    registry.register("analyze plot", analyze_plot::factory);
    registry.register("analyze flamegraph", analyze_flamegraph::factory);
    registry.register("generate predicated", gen_predicated::factory);
}
