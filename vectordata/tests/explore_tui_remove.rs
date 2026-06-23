// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

// shadow-terminal drives a real PTY; ConPTY automation on Windows is unreliable
// (Ctrl-G / prompt timing races, e.g. a 5s `wait_for_string` that never sees the
// pane render), so this end-to-end TUI test runs on Unix (Linux + macOS CI) only
// — it must never gate the Windows release binary in build.yml's `Test vectordata`.
#![cfg(unix)]

//! E2E: removing a catalog from the config pane — `x` on the highlighted
//! catalog, `y` to confirm — drops it from the pane and from catalogs.yaml.

mod tui_common;
use tui_common::{
    init_home, read_catalogs, resolving_catalog, start_explore, wait_for_absence, write_catalogs,
    CTRL_G, DOWN,
};

use shadow_terminal::steppable_terminal::Input;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remove_catalog_from_config_pane() {
    let home = init_home();
    let good = resolving_catalog(home.path(), "goodcat");
    // goodcat resolves (listed first); aaa_broken resolves to 0 datasets so
    // it sorts into the trailing group → cursor index 1. Two catalogs total,
    // so the "can't remove the only one" guard doesn't fire.
    write_catalogs(
        home.path(),
        &format!("goodcat: {}\naaa_broken: {}/nope\n", good.display(), home.path().display()),
    );

    let mut term = start_explore(home.path()).await;
    term.wait_for_string("goodcatds", Some(10_000)).await.unwrap();
    term.send_input(Input::Event(CTRL_G.to_string())).unwrap();
    term.wait_for_string("aaa_broken", Some(5_000))
        .await
        .expect("the 0-dataset catalog is listed");

    // Move to aaa_broken (cursor 0 = goodcat) and remove it.
    term.send_input(Input::Event(DOWN.to_string())).unwrap();
    term.send_input(Input::Characters("x".to_string())).unwrap();
    term.wait_for_string("Remove catalog", Some(5_000))
        .await
        .expect("the remove confirm overlay appears");
    term.send_input(Input::Characters("y".to_string())).unwrap();

    // The reload reopens the pane without the removed catalog…
    wait_for_absence(&mut term, "aaa_broken", 8_000).await;
    // …and it's gone from catalogs.yaml on disk.
    let yaml = read_catalogs(home.path());
    assert!(
        !yaml.contains("aaa_broken"),
        "catalogs.yaml still has the removed entry:\n{yaml}"
    );

    term.kill().unwrap();
}
