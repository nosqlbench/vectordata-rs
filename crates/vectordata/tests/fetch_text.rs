// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata::fetch::fetch_text` and `fetch_if_modified`: the small
//! files a program reads beside a dataset, fetched by the client, auth
//! and offline rule the dataset fetches use, so a program needs no HTTP
//! client of its own for them (LS-2).

mod support;

use support::testserver::TestServer;

#[test]
fn fetch_text_reads_a_small_file_and_names_a_failure() {
    let served = tempfile::tempdir().unwrap();
    std::fs::write(served.path().join("catalog.yaml"), "datasets: []\n").unwrap();
    let server = TestServer::start(served.path()).unwrap();
    let text = vectordata::fetch::fetch_text(&format!("{}/catalog.yaml", server.base_url())).unwrap();
    assert_eq!(text, "datasets: []\n");
    let err = vectordata::fetch::fetch_text(&format!("{}/absent.yaml", server.base_url())).unwrap_err();
    assert!(err.to_string().contains("HTTP 404"), "{err}");
}

#[test]
fn fetch_if_modified_fetches_once_and_then_keeps_the_copy() {
    let served = tempfile::tempdir().unwrap();
    std::fs::write(served.path().join("list.txt"), "a\nb\n").unwrap();
    let server = TestServer::start(served.path()).unwrap();
    let local = tempfile::tempdir().unwrap();
    let copy = local.path().join("list.txt");
    let url = format!("{}/list.txt", server.base_url());
    assert!(vectordata::fetch::fetch_if_modified(&url, &copy).unwrap(), "no copy: fetched");
    assert_eq!(std::fs::read_to_string(&copy).unwrap(), "a\nb\n");
    assert!(!vectordata::fetch::fetch_if_modified(&url, &copy).unwrap(), "the copy is current");
}
