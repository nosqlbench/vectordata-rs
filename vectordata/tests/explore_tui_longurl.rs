// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! E2E: a long URL typed into the add-catalog field stays visible at the
//! caret end (the field scrolls / the box widens) instead of being clipped.

mod tui_common;
use tui_common::{init_home, resolving_catalog, start_explore, write_catalogs, CTRL_G, ENTER};

use shadow_terminal::steppable_terminal::Input;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn long_url_tail_stays_visible_in_add_field() {
    let home = init_home();
    let good = resolving_catalog(home.path(), "goodcat");
    write_catalogs(home.path(), &format!("goodcat: {}\n", good.display()));

    let mut term = start_explore(home.path()).await;
    term.wait_for_string("goodcatds", Some(10_000)).await.unwrap();
    term.send_input(Input::Event(CTRL_G.to_string())).unwrap();
    term.wait_for_string("Catalogs", Some(5_000)).await.unwrap();

    // Open the add modal and advance to the URL field.
    term.send_input(Input::Characters("a".to_string())).unwrap();
    term.wait_for_string("Add catalog", Some(5_000)).await.unwrap();
    term.send_input(Input::Characters("ln".to_string())).unwrap(); // name
    term.send_input(Input::Characters(ENTER.to_string())).unwrap(); // → url field

    // A URL far longer than the field (130 chars), ending in a marker. Before
    // the fix this tail was clipped off the right edge; now the field scrolls
    // to the caret so the marker is on screen.
    let long = format!("https://example.com/{}ZZEND", "a".repeat(105));
    assert!(long.chars().count() > 120);
    term.send_input(Input::Characters(long)).unwrap();

    term.wait_for_string("ZZEND", Some(5_000))
        .await
        .expect("the end of a long URL must stay visible while typing");

    term.kill().unwrap();
}
