// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

/// Ensure the test temp directory exists.
///
/// `.cargo/config.toml` sets `TMPDIR=target/test-tmp` (relative) so that
/// test temp files go to a project-local directory instead of `/tmp`.
/// The `tempfile` crate requires this directory to exist before creating
/// temp files, so we create it at build time.
fn main() {
    let workspace = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let test_tmp = std::path::Path::new(&workspace).join("../target/test-tmp");
    let _ = std::fs::create_dir_all(&test_tmp);
}
