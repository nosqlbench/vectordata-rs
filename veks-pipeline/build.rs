// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

fn main() {
    // Link system BLAS for knnutils personality commands.
    // On Ubuntu: `apt install libopenblas-dev` (or `libmkl-dev` for MKL).
    // The system's libblas.so resolves to whichever BLAS is configured
    // via update-alternatives.
    #[cfg(feature = "knnutils")]
    println!("cargo:rustc-link-lib=blas");

    // Inject build metadata as compile-time environment variables.
    // VEKS_BUILD_HASH: short git SHA of the current commit (or "unknown").
    // VEKS_BUILD_TIMESTAMP: UTC timestamp of the build.
    // These are used by CommandOp::build_version() for provenance tracking.
    let git_hash = std::process::Command::new("git")
        .args(["rev-parse", "--short=10", "HEAD"])
        .output()
        .ok()
        .and_then(|o| if o.status.success() {
            String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
        } else {
            None
        })
        .unwrap_or_else(|| "unknown".to_string());

    // Check for dirty working tree
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    let build_hash = if dirty {
        format!("{}+dirty", git_hash)
    } else {
        git_hash
    };

    println!("cargo:rustc-env=VEKS_BUILD_HASH={}", build_hash);

    // Rebuild if git HEAD changes (new commit)
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/refs/");
}
