// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

/// Build script: creates test temp directory and injects build metadata.
fn main() {
    // Ensure the test temp directory exists.
    // `.cargo/config.toml` sets `TMPDIR=target/test-tmp` (relative) so that
    // test temp files go to a project-local directory instead of `/tmp`.
    let workspace = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = std::path::Path::new(&workspace).parent().unwrap_or(std::path::Path::new("."));
    let test_tmp = workspace_root.join("target/test-tmp");
    let _ = std::fs::create_dir_all(&test_tmp);

    // Inject build metadata as compile-time environment variables.
    // VEKS_BUILD_HASH: short git SHA (or "unknown"), `+dirty` when the
    // working tree has uncommitted changes, then `+<profile>` — the cargo
    // build profile, `debug` or `release`.
    // VEKS_BUILD_NUMBER: epoch seconds — changes every build regardless of git state.
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

    // Capture rustc version for --version output.
    let rustc_version = std::process::Command::new("rustc")
        .args(["--version"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=VEKS_BUILD_HASH={}", build_hash);
    println!("cargo:rustc-env=VEKS_BUILD_NUMBER={}", build_number);
    println!("cargo:rustc-env=VEKS_RUSTC_VERSION={}", rustc_version);
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/refs/");
}
