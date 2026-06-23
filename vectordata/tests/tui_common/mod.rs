// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Shared harness for the `vectordata explore` TUI end-to-end tests:
//! drives the compiled binary inside a headless PTY (`shadow-terminal`).
//!
//! Files in this subdirectory are a module, not a test binary, so each
//! TUI test file (`explore_tui*.rs`) includes it with `mod tui_common;`.
//! Each of those is its own binary → its own process → its own
//! `$VECTORDATA_HOME`, so the env mutation here never races.

#![allow(dead_code)]

use shadow_terminal::shadow_terminal::Config;
use shadow_terminal::steppable_terminal::SteppableTerminal;

/// Ctrl-G — opens the settings/config pane.
pub const CTRL_G: &str = "\u{07}";
/// Down-arrow ANSI sequence.
pub const DOWN: &str = "\u{1b}[B";
/// Carriage return = Enter in raw mode.
pub const ENTER: &str = "\r";

/// An isolated client home with a cache + settings.yaml (no catalogs yet).
pub fn init_home() -> tempfile::TempDir {
    let home = tempfile::tempdir().unwrap();
    let cache = home.path().join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::write(
        home.path().join("settings.yaml"),
        format!("cache_dir: {}\n", cache.display()),
    )
    .unwrap();
    home
}

/// Write the home's catalogs.yaml.
pub fn write_catalogs(home: &std::path::Path, content: &str) {
    std::fs::write(home.join("catalogs.yaml"), content).unwrap();
}

/// Read the home's catalogs.yaml back (for asserting conversions).
pub fn read_catalogs(home: &std::path::Path) -> String {
    std::fs::read_to_string(home.join("catalogs.yaml")).unwrap_or_default()
}

/// Create a local catalog directory that resolves to exactly one dataset
/// (a `knn_entries.yaml` with one `name:profile` entry). The facet files
/// need not exist — listing a catalog never reads them.
pub fn resolving_catalog(home: &std::path::Path, name: &str) -> std::path::PathBuf {
    let d = home.join(name);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(
        d.join("knn_entries.yaml"),
        format!(r#""{name}ds:default": {{ base: base.fvecs, query: query.fvecs, gt: gt.ivecs }}"#),
    )
    .unwrap();
    d
}

/// Launch `vectordata explore` in a PTY against `home`. The PTY inherits
/// this process's env, so `$VECTORDATA_HOME` isolates the child; `CI=1`
/// suppresses the update check.
pub async fn start_explore(home: &std::path::Path) -> SteppableTerminal {
    unsafe {
        std::env::set_var("VECTORDATA_HOME", home);
        std::env::remove_var("VECTORDATA_TOKEN");
        std::env::set_var("CI", "1");
    }
    let bin = env!("CARGO_BIN_EXE_vectordata");
    let cfg = Config {
        width: 110,
        height: 40,
        command: vec![bin.into(), "explore".into()],
        scrollback_size: 1000,
        scrollback_step: 100,
    };
    SteppableTerminal::start(cfg).await.expect("start vectordata explore in a PTY")
}

/// Wait until `needle` is **absent** from the screen (the dual of
/// `wait_for_string`), e.g. after removing a catalog.
pub async fn wait_for_absence(term: &mut SteppableTerminal, needle: &str, timeout_ms: u32) {
    for i in 0..=timeout_ms {
        term.render_all_output().await.unwrap();
        if !term.screen_as_string().unwrap().contains(needle) {
            return;
        }
        if i == timeout_ms {
            let _ = term.dump_screen();
            panic!("'{needle}' still present after {timeout_ms}ms");
        }
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
}
