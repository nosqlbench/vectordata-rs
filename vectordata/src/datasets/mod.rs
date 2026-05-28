// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Canonical implementation of `<binary> datasets …` subcommands.
//!
//! Both the `vectordata` binary and the `veks` CLI dispatch into
//! this module — there is exactly one implementation of each
//! command. The submodules return *exit codes* (`i32`) rather than
//! `Result`s so the dispatch layer in either binary can simply
//! `std::process::exit(code)`.

pub mod browser;
pub mod cache;
pub mod curlify;
pub mod derive;
pub mod drop_cache;
pub mod filter;
pub mod list;
pub mod precache;
pub mod ping;
