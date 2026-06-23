// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! E2E: naming a catalog on a legacy **list-formatted** catalogs.yaml
//! prompts (on the suspended console) to convert it to name-based form;
//! accepting rewrites the file to the `name: url` map form.

mod tui_common;
use tui_common::{
    init_home, read_catalogs, resolving_catalog, start_explore, write_catalogs, CTRL_G, ENTER,
};

use shadow_terminal::steppable_terminal::Input;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn naming_on_list_format_prompts_to_convert() {
    let home = init_home();
    let good = resolving_catalog(home.path(), "goodcat");
    let other = resolving_catalog(home.path(), "othercat");
    // LIST form — the legacy shape that can't store names.
    write_catalogs(home.path(), &format!("- {}\n", good.display()));

    let mut term = start_explore(home.path()).await;
    term.wait_for_string("goodcatds", Some(10_000)).await.unwrap();
    term.send_input(Input::Event(CTRL_G.to_string())).unwrap();
    term.wait_for_string("Catalogs", Some(5_000)).await.unwrap();

    // Add a NAMED catalog: a → name → url → (empty token) → submit.
    term.send_input(Input::Characters("a".to_string())).unwrap();
    term.wait_for_string("Add catalog", Some(5_000)).await.expect("add modal opens");
    term.send_input(Input::Characters("newcat".to_string())).unwrap();
    term.send_input(Input::Characters(ENTER.to_string())).unwrap();
    term.send_input(Input::Characters(other.display().to_string())).unwrap();
    term.send_input(Input::Characters(ENTER.to_string())).unwrap();
    term.send_input(Input::Characters(ENTER.to_string())).unwrap();

    // The convert prompt appears on the (suspended) console.
    term.wait_for_string("list-formatted", Some(8_000))
        .await
        .expect("naming on a list-format file prompts to convert");

    // Accept the conversion.
    term.send_input(Input::Characters("y".to_string())).unwrap();
    term.send_input(Input::Characters(ENTER.to_string())).unwrap();
    term.wait_for_string("Converted", Some(8_000))
        .await
        .expect("the conversion runs after accepting");

    // catalogs.yaml is now name-based: the existing list entry got a derived
    // name and there are no `- url` lines left.
    let yaml = read_catalogs(home.path());
    assert!(
        !yaml.lines().any(|l| l.trim_start().starts_with("- ")),
        "catalogs.yaml should be map form after conversion:\n{yaml}"
    );
    assert!(
        yaml.contains("goodcat:"),
        "the existing list entry should be converted to a named entry:\n{yaml}"
    );

    term.kill().unwrap();
}
