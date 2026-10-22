// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **Offline mode stops the endpoint introspection calls before the
//! network** (`docs/design/srd-crate-boundaries.md` TR-12): `whoami`
//! (behind `ping`) and `candidate_namespaces` (behind completion) make no
//! request, while the explicit server commands — here `login` and
//! `backup`, whose access view is a `whoami` of its own — are not
//! affected.
//!
//! Offline mode is the process's transport configuration, read once from
//! `$VECTORDATA_OFFLINE` and the settings that `$VECTORDATA_HOME`
//! isolates; the calls run in a child process given those variables, so
//! this process's environment is never changed.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// A server that records each request line and answers `whoami` and
/// the namespace list with an empty view, and anything else with 401.
fn recording_server() -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            let _ = reader.read_line(&mut line);
            loop {
                let mut header = String::new();
                if reader.read_line(&mut header).is_err() || header.trim().is_empty() {
                    break;
                }
            }
            log.lock().unwrap().push(line.trim().to_string());
            let (status, body) = if line.contains("/-/whoami") || line.contains("/-/namespaces") {
                ("200 OK", r#"{"namespaces": []}"#)
            } else {
                ("401 Unauthorized", "")
            };
            let _ = write!(stream, "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        }
    });
    (url, seen)
}

const CHILD_URL: &str = "VECTORDATA_TEST_ENDPOINT_URL";

/// Run [`endpoint_calls_in_a_child`] offline, with `home` as the client home.
fn run_offline_child(home: &Path, url: &str) -> std::process::Output {
    std::process::Command::new(std::env::current_exe().unwrap())
        .args(["endpoint_calls_in_a_child", "--exact", "--nocapture", "--test-threads=1"])
        .env("VECTORDATA_HOME", home)
        .env(vectordata::settings::OFFLINE_ENV, "1")
        .env_remove("VECTORDATA_TOKEN")
        .env(CHILD_URL, url)
        .output()
        .expect("run the child")
}

#[test]
fn offline_mode_refuses_endpoint_introspection_before_any_request() {
    let (url, seen) = recording_server();
    let work = tempfile::tempdir().unwrap();
    let home = work.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(home.join("settings.yaml"), format!("cache_dir: {}\n", work.path().join("cache").display())).unwrap();

    let out = run_offline_child(&home, &url);
    assert!(
        out.status.success(),
        "the offline child failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let seen = seen.lock().unwrap().clone();
    let paths: Vec<&str> = seen.iter().filter_map(|line| line.split_whitespace().nth(1)).collect();
    assert_eq!(
        paths,
        ["/auth/token", "/-/whoami"],
        "offline: only login and backup's access view reach the server, in that order: {seen:?}"
    );
}

/// The child half of the test above; does nothing unless launched by it.
#[test]
fn endpoint_calls_in_a_child() {
    let Ok(url) = std::env::var(CHILD_URL) else { return };
    let refused = vectordata::endpoint::whoami(&url, None).expect_err("whoami is refused offline");
    assert!(refused.contains("offline mode is on"), "{refused}");
    assert!(vectordata::endpoint::candidate_namespaces(&url, Some("t")).is_empty());
    // `login` reaches the server, which refuses the password.
    let login = vectordata::endpoint::login_password(&url, "u", "p", None, None).expect_err("the server says 401");
    assert!(login.contains("authentication failed"), "{login}");
    // `backup` reaches the server too: its access view lists no namespace.
    let dest = tempfile::tempdir().unwrap();
    let stats = vectordata::backup::run_backup(&url, &dest.path().display().to_string(), false, None);
    assert!(stats.is_ok(), "backup is an explicit command: {:?}", stats.err());
}
