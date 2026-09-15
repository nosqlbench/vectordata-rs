// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

fn main() {
    // Link system BLAS for knnutils personality commands.
    // On Ubuntu: `apt install libopenblas-dev` (or `libmkl-dev` for MKL).
    // The system's libblas.so resolves to whichever BLAS is configured
    // via update-alternatives.
    //
    // Unix-only: every `cblas_sgemm` call site in the source is gated
    // on `cfg(all(feature = "knnutils", unix))`, so on non-Unix targets
    // the FFI symbols are never referenced and the link directive
    // would just produce a "library not found" error from the linker.
    // `CARGO_CFG_UNIX` is set in build.rs when the *target* is Unix
    // (cargo populates these per the target triple, not the host).
    #[cfg(feature = "knnutils")]
    if std::env::var_os("CARGO_CFG_UNIX").is_some() {
        println!("cargo:rustc-link-lib=blas");
    }

    // Inject build metadata as compile-time environment variables.
    // VEKS_BUILD_HASH: short git SHA (or "unknown"), `+dirty` when the
    // working tree has uncommitted changes, then `+<profile>` — the cargo
    // build profile, `debug` or `release`.
    // VEKS_BUILD_TIMESTAMP: UTC timestamp of the build.
    // These are used by CommandOp::build_version() for provenance tracking.
    let workspace = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = std::path::Path::new(&workspace).parent().unwrap_or(std::path::Path::new("."));

    let git_hash = std::process::Command::new("git")
        .args(["rev-parse", "--short=10", "HEAD"])
        .current_dir(workspace_root)
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
        .current_dir(workspace_root)
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    // The profile the binary was built under, stated always: a debug
    // build's unoptimized hot loops run several times slower than a
    // release build's, and the stamp in every run log is where that
    // has to be visible.
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    let build_hash = if dirty {
        format!("{git_hash}+dirty+{profile}")
    } else {
        format!("{git_hash}+{profile}")
    };

    let build_number = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string());

    println!("cargo:rustc-env=VEKS_BUILD_HASH={}", build_hash);
    println!("cargo:rustc-env=VEKS_BUILD_NUMBER={}", build_number);

    // Rebuild if git HEAD changes (new commit)
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/refs/");
}
