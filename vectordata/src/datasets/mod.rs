// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Canonical implementation of `<binary> datasets …` subcommands.
//!
//! Both the `vectordata` binary and the `veks` CLI dispatch into
//! this module — there is exactly one implementation of each
//! command. The submodules return *exit codes* (`i32`) rather than
//! `Result`s so the dispatch layer in either binary can simply
//! `std::process::exit(code)`.

pub mod prebuffer;
