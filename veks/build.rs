// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

/// Build script: creates test temp directory and injects build metadata.
fn main() {
    // Ensure the test temp directory exists.
    // `.cargo/config.toml` sets `TMPDIR=target/test-tmp` (relative) so that
    // test temp files go to a project-local directory instead of `/tmp`.
    let workspace = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let test_tmp = std::path::Path::new(&workspace).join("../target/test-tmp");
    let _ = std::fs::create_dir_all(&test_tmp);

    // Inject build metadata as compile-time environment variables.
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
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/refs/");
}
