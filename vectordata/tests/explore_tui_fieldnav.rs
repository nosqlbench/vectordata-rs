// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! E2E: in the add/edit-catalog modal, ↑/↓ move between the name/url/token
//! fields (not just Tab). Verified via the masked token field: typing after
//! ↓↓ lands in the (masked) token, so the plaintext never shows; ↑↑ returns
//! to the visible name field.

mod tui_common;
use tui_common::{init_home, resolving_catalog, start_explore, wait_for_absence, write_catalogs, CTRL_G, DOWN, UP};

use shadow_terminal::steppable_terminal::Input;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn arrows_move_between_modal_fields() {
    let home = init_home();
    let good = resolving_catalog(home.path(), "goodcat");
    write_catalogs(home.path(), &format!("goodcat: {}\n", good.display()));

    let mut term = start_explore(home.path()).await;
    term.wait_for_string("goodcatds", Some(10_000)).await.unwrap();
    term.send_input(Input::Event(CTRL_G.to_string())).unwrap();
    term.wait_for_string("Catalogs", Some(5_000)).await.unwrap();

    // Open the add modal (cursor starts on the name field).
    term.send_input(Input::Characters("a".to_string())).unwrap();
    term.wait_for_string("Add catalog", Some(5_000)).await.unwrap();
    term.send_input(Input::Characters("AANAME".to_string())).unwrap();
    term.wait_for_string("AANAME", Some(5_000)).await.expect("typed into the name field");

    // ↓↓ to the token field; typed text is masked, so the plaintext is absent.
    term.send_input(Input::Event(DOWN.to_string())).unwrap();
    term.send_input(Input::Event(DOWN.to_string())).unwrap();
    term.send_input(Input::Characters("BBSECRET".to_string())).unwrap();
    wait_for_absence(&mut term, "BBSECRET", 4_000).await; // would be visible in name if ↓ failed

    // ↑↑ back to the name field; appended text is visible there.
    term.send_input(Input::Event(UP.to_string())).unwrap();
    term.send_input(Input::Event(UP.to_string())).unwrap();
    term.send_input(Input::Characters("CCBACK".to_string())).unwrap();
    term.wait_for_string("AANAMECCBACK", Some(5_000))
        .await
        .expect("↑↑ returned to the name field");

    term.kill().unwrap();
}
