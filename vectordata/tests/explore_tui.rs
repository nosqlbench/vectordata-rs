// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! E2E: the catalog **configuration pane** (Ctrl-G) lists every configured
//! catalog — including one that resolves to zero datasets (the regression
//! guarded here) — driving the compiled `vectordata explore` in a PTY.

mod tui_common;
use tui_common::{init_home, resolving_catalog, start_explore, write_catalogs, CTRL_G};

use shadow_terminal::steppable_terminal::Input;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explore_config_pane_lists_all_configured_catalogs() {
    let home = init_home();
    let good = resolving_catalog(home.path(), "goodcat");
    // One resolving catalog + one whose location does not exist (0 datasets);
    // both MUST appear in the pane.
    write_catalogs(
        home.path(),
        &format!("goodcat: {}\nbrokencat: {}/nope\n", good.display(), home.path().display()),
    );

    let mut term = start_explore(home.path()).await;

    term.wait_for_string("goodcatds", Some(10_000))
        .await
        .expect("picker should list the resolving catalog's dataset");
    term.send_input(Input::Event(CTRL_G.to_string())).expect("send Ctrl-G");
    term.wait_for_string("Catalogs", Some(5_000))
        .await
        .expect("config pane shows the Catalogs tab");
    term.wait_for_string("goodcat", Some(5_000))
        .await
        .expect("resolving catalog appears");
    term.wait_for_string("brokencat", Some(5_000))
        .await
        .expect("a 0-dataset catalog must still appear in the config pane");

    term.kill().expect("shut down the PTY");
}
