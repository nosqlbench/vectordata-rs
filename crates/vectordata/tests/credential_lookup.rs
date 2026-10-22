// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **A read token is one in-memory lookup per request, never I/O**
//! (`docs/design/srd-crate-boundaries.md` PR-7). Credentials are keyed
//! by URL prefix, so two paths on one origin carry their own tokens; the
//! store is read once per process and again only on
//! `reload_credentials`, so a request never touches the file.
//!
//! The store is the process's, found through `$VECTORDATA_HOME`; the
//! lookups run in a child process given that variable, so this
//! process's environment is never changed.

use std::path::Path;

use vectordata::transport::Credentials;

const CHILD_HOME: &str = "VECTORDATA_TEST_LOOKUP_HOME";

fn store(a: &str, b: &str) -> String {
    format!(
        "[\"https://h.example/a\"]\ntoken = \"{a}\"\n\n[\"https://h.example/b\"]\ntoken = \"{b}\"\n"
    )
}

/// **Each request's token comes from the store as loaded**: the file
/// rewritten under a running process changes nothing until
/// `reload_credentials`, and each prefix keeps its own token.
#[test]
fn a_read_token_is_looked_up_in_memory_per_prefix() {
    let home = tempfile::tempdir().unwrap();
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["lookup_in_a_child", "--exact", "--nocapture", "--test-threads=1"])
        .env("VECTORDATA_HOME", home.path())
        .env(CHILD_HOME, home.path())
        .env_remove("VECTORDATA_TOKEN")
        .output()
        .expect("run the child");
    assert!(
        out.status.success() && String::from_utf8_lossy(&out.stdout).contains("1 passed"),
        "the lookup child failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The child half of the test above; does nothing unless launched by it.
#[test]
fn lookup_in_a_child() {
    let Ok(home) = std::env::var(CHILD_HOME) else { return };
    let file = Path::new(&home).join("credentials.toml");
    std::fs::write(&file, store("first-a", "first-b")).unwrap();
    let creds = vectordata::credentials::StoredCredentials;
    let token = |path: &str| creds.read_token(&format!("https://h.example/{path}/base.fvec"));

    assert_eq!(token("a").as_deref(), Some("first-a"));
    assert_eq!(token("b").as_deref(), Some("first-b"), "one origin, two prefixes, two tokens");
    assert_eq!(token("c"), None);

    std::fs::write(&file, store("second-a", "second-b")).unwrap();
    assert_eq!(token("a").as_deref(), Some("first-a"), "a request read the credentials file");

    vectordata::credentials::reload_credentials();
    assert_eq!(token("a").as_deref(), Some("second-a"));
    assert_eq!(token("b").as_deref(), Some("second-b"));
}
