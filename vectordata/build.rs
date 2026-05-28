// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0
//
// Emit build-time metadata for use by the `vectordata` binary's
// `--version` output. Three values are exposed via `env!` at compile
// time:
//
//   VECTORDATA_GIT_DESCRIBE — output of `git describe --tags --always
//     --dirty`, or the literal string "unknown" when not inside a git
//     work tree (e.g. building from a published crate tarball).
//   VECTORDATA_BUILD_PROFILE — Cargo's PROFILE env var ("release",
//     "debug", "profiling", …), surfaced so bug-report version output
//     records which build the user is running.
//   VECTORDATA_BUILD_DATE — UTC date the binary was compiled. Coarse
//     enough not to bust caches (date, not minute), but useful when
//     triaging "what version did I install last week".
//
// Cargo invalidates this script + the binary whenever HEAD moves or
// the working tree dirties, via the `rerun-if-changed` directives.

use std::process::Command;

fn main() {
    let describe = Command::new("git")
        .args(["describe", "--tags", "--always", "--dirty=+dirty"])
        .output()
        .ok()
        .and_then(|o| if o.status.success() {
            String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
        } else {
            None
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=VECTORDATA_GIT_DESCRIBE={describe}");

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=VECTORDATA_BUILD_PROFILE={profile}");

    // Build date — UTC, day-resolution. `date -u +%Y-%m-%d` is portable
    // across Linux + macOS; on Windows runners cargo will see PROFILE
    // but `date` is likely absent, so fall back to "unknown".
    let date = Command::new("date")
        .args(["-u", "+%Y-%m-%d"])
        .output()
        .ok()
        .and_then(|o| if o.status.success() {
            String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
        } else {
            None
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=VECTORDATA_BUILD_DATE={date}");

    // Rerun this script when HEAD or the index moves, so version
    // strings stay in sync. Adding `build.rs` itself catches changes
    // to the script.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/index");
}
