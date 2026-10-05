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
    if p.is_absolute() {
        return p;
    }
    // Strip leading `.` and `..` aren't a problem (they belong),
    // but redundant `./` components turn into duplicate dots once
    // we join — `Path::from(".").join("./foo") == "././foo"`. The
    // result is functionally the same path, but it surfaces in
    // user-facing error messages, so we normalise here.
    let joined = workspace.join(&p);
    normalise_dot_components(&joined)
}

/// Collapse redundant `CurDir` (`.`) components so that joining
/// a `.` workspace with a `./foo` arg produces `./foo`, not
/// `././foo`. Preserves a leading `./` (matching the user's
/// notation when they typed one) and preserves `..` components
/// (which actually shift the path).
fn normalise_dot_components(path: &Path) -> std::path::PathBuf {
    use std::path::Component;
    let mut out = std::path::PathBuf::new();
    let mut wrote_leading_dot = false;
    let mut wrote_anything_else = false;
    for c in path.components() {
        match c {
            Component::CurDir => {
                if !wrote_leading_dot && !wrote_anything_else {
                    out.push(".");
                    wrote_leading_dot = true;
                }
                // Any subsequent CurDir is dropped — they're noise.
            }
            other => {
                wrote_anything_else = true;
                out.push(other.as_os_str());
            }
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Joining a `.` workspace with `./foo` must not produce
    /// `././foo` — the user-facing path strings should be clean.
    #[test]
    fn resolve_path_collapses_doubled_dot_prefix() {
        let p = resolve_path("./predicates.slab", Path::new("."));
        assert_eq!(p, Path::new("./predicates.slab"));
    }

    /// Bare relative names join naturally and pick up the
    /// leading `.` from the workspace.
    #[test]
    fn resolve_path_keeps_workspace_dot() {
        let p = resolve_path("predicates.slab", Path::new("."));
        assert_eq!(p, Path::new("./predicates.slab"));
    }

    /// Absolute inputs ignore the workspace.
    #[test]
    fn resolve_path_absolute_passes_through() {
        let p = resolve_path("/abs/predicates.slab", Path::new("/ws"));
        assert_eq!(p, Path::new("/abs/predicates.slab"));
    }

    /// `..` components are preserved (they actually shift the
    /// path); only `.` components are collapsed.
    #[test]
    fn resolve_path_keeps_parent_components() {
        let p = resolve_path("../sibling/x.slab", Path::new("./profiles/base"));
        assert_eq!(p, Path::new("./profiles/base/../sibling/x.slab"));
    }
}
