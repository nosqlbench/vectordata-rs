// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Helpers shared between the predicate-generation commands.
//!
//! `gen_predicates` (survey-driven, selectivity-targeted) and
//! `gen_simple_predicates` (config-only, smoke-test grade) share
//! their option helper, path resolution, and error wrapping.

use std::path::Path;
use std::time::Instant;

use crate::pipeline::command::{
    CommandResult, OptionDesc, OptionRole, Status,
};

pub(crate) fn resolve_path(path_str: &str, workspace: &Path) -> std::path::PathBuf {
    let p = std::path::PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

pub(crate) fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

pub(crate) fn opt(
    name: &str,
    type_name: &str,
    required: bool,
    default: Option<&str>,
    desc: &str,
    role: OptionRole,
) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
        extended_description: None,
        role,
    }
}
