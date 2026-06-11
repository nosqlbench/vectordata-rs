// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test of the **compiled `vecd` binary**: drive the real CLI
//! (`init` → `backends add` → `users add` → `ns add` → `bind` →
//! `tokens create`), start `vecd serve` as a child process on an
//! ephemeral port, then run the real `vectordata push` engine against it
//! over HTTP and pull the result back anonymously.
//!
//! This complements `push_against_vecd.rs` (which drives the library
//! `AppState` in-process) by exercising the binary, the config/data-dir
//! flow, and the standalone `serve` listener.

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vectordata::push::transport::TransportOptions;
use vectordata::push::{execute, ChecksumPolicy, Options};

/// Path to the binary under test, provided by cargo to integration tests.
fn vecd_bin() -> &'static str {
    env!("CARGO_BIN_EXE_vecd")
}

struct Harness {
    _work: tempfile::TempDir,
    config_dir: PathBuf,
    data_dir: PathBuf,
}

impl Harness {
    fn new() -> Self {
        let work = tempfile::tempdir().unwrap();
        let config_dir = work.path().join("config");
        let data_dir = work.path().join("data");
        std::fs::create_dir_all(&config_dir).unwrap();
        Harness { _work: work, config_dir, data_dir }
    }

    /// Run a `vecd` subcommand to completion, returning stdout. Panics on
    /// non-zero exit (with stderr) so setup failures are loud.
    fn vecd(&self, args: &[&str]) -> String {
        let out = Command::new(vecd_bin())
            .env("VECD_CONFIG", &self.config_dir)
            .arg("--data-dir")
            .arg(&self.data_dir)
            .args(args)
            .output()
            .expect("spawn vecd");
        if !out.status.success() {
            panic!(
                "vecd {:?} failed ({}):\n{}",
                args,
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
        }
        String::from_utf8_lossy(&out.stdout).to_string()
    }

    /// Start `vecd serve` on an ephemeral port; wait for it to publish its
    /// address and answer `/healthz`.
    fn serve(&self) -> Server {
        let child = Command::new(vecd_bin())
            .env("VECD_CONFIG", &self.config_dir)
            .arg("--data-dir")
            .arg(&self.data_dir)
            .args(["serve", "--bind", "127.0.0.1:0"])
            .spawn()
            .expect("spawn vecd serve");
        let addr_file = self.data_dir.join("vecd.addr");
        let base = wait_for_endpoint(&addr_file);
        Server { child, base }
    }
}

struct Server {
    child: Child,
    /// Base URL like `http://127.0.0.1:PORT`.
    base: String,
}

impl Server {
    fn url(&self, path: &str) -> String {
        format!("{}/{path}", self.base)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Poll for the `vecd.addr` file, then for `/healthz` to answer.
fn wait_for_endpoint(addr_file: &Path) -> String {
    let deadline = Instant::now() + Duration::from_secs(15);
    let addr = loop {
        if let Ok(s) = std::fs::read_to_string(addr_file)
            && !s.trim().is_empty() {
                break s.trim().to_string();
            }
        if Instant::now() > deadline {
            panic!("vecd serve never published its address");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let base = format!("http://{addr}");
    let client = reqwest::blocking::Client::new();
    loop {
        if let Ok(r) = client.get(format!("{base}/healthz")).send()
            && r.status().is_success() {
                break;
            }
        if Instant::now() > deadline {
            panic!("vecd /healthz never became ready");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    base
}

fn make_dataset(dir: &Path) {
    std::fs::write(
        dir.join("dataset.yaml"),
        "name: glove\nattributes:\n  is_zero_vector_free: true\n  is_duplicate_vector_free: true\n",
    )
    .unwrap();
    std::fs::write(dir.join("base.fvec"), b"VECDBASE").unwrap();
}

fn push_opts(src: &Path, to: String, token: Option<String>) -> Options {
    Options {
        path: src.to_path_buf(),
        to: Some(to),
        message: None,
        raw: false,
        checksums: ChecksumPolicy::Auto,
        dry_run: false,
        no_check: true,
        assume_yes: true,
        delete: false,
        abort_incomplete: false,
        concurrency: 4,
        files: None,
        transport: TransportOptions { token, profile: None, endpoint_url: None },
        cmd: "vectordata push (cli e2e)".into(),
        actor: "tester@host".into(),
    }
}

/// Extract a minted token (`token:   vd_…`) from `init`/`tokens create`
/// output.
fn token_from(output: &str) -> String {
    output
        .lines()
        .find_map(|l| l.trim().strip_prefix("token:"))
        .map(|t| t.trim().to_string())
        .filter(|t| t.starts_with("vd_"))
        .expect("a vd_ token in output")
}

/// Ensures a started daemon is stopped even if assertions fail.
struct DaemonGuard<'a> {
    h: &'a Harness,
}
impl Drop for DaemonGuard<'_> {
    fn drop(&mut self) {
        let _ = Command::new(vecd_bin())
            .env("VECD_CONFIG", &self.h.config_dir)
            .arg("--data-dir")
            .arg(&self.h.data_dir)
            .arg("stop")
            .output();
    }
}

#[test]
fn cli_daemon_start_status_stop() {
    let h = Harness::new();
    // vecd requires a config before any operational command — establish one.
    h.vecd(&["config", "auto", "--yes"]);
    h.vecd(&["init", "--superuser", "root"]);

    // status before start: not running.
    assert!(h.vecd(&["status"]).contains("not running"));

    // start daemonizes; the parent returns once the pidfile is written.
    let started = h.vecd(&["start", "--bind", "127.0.0.1:0"]);
    assert!(started.contains("started"), "{started}");
    let _guard = DaemonGuard { h: &h };

    // Wait for it to publish its address + answer /healthz.
    let base = wait_for_endpoint(&h.data_dir.join("vecd.addr"));
    let resp = reqwest::blocking::get(format!("{base}/healthz")).unwrap();
    assert_eq!(resp.status(), 200);

    // status reports running + the bound address.
    let status = h.vecd(&["status"]);
    assert!(status.contains("running"), "{status}");

    // stop drains it; status then reports not running.
    let stopped = h.vecd(&["stop"]);
    assert!(stopped.contains("stopped"), "{stopped}");

    // Give the OS a moment, then confirm it's down.
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && h.vecd(&["status"]).contains("is running") {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(h.vecd(&["status"]).contains("not running"));
}

#[test]
fn cli_init_serve_push_pull() {
    let h = Harness::new();

    // vecd requires a config before any operational command (config auto writes
    // safe defaults; --data-dir below overrides the data dir it records).
    h.vecd(&["config", "auto", "--yes"]);

    // init mints a superuser token and prints it once.
    let init_out = h.vecd(&["init", "--superuser", "root"]);
    let root_token = token_from(&init_out);
    assert!(root_token.starts_with("vd_"));

    // Set up an endpoint exactly as an operator would.
    let objects = h._work.path().join("objects");
    h.vecd(&["backends", "add", "store", "--kind", "local", "--endpoint",
             &format!("local:{}", objects.display()), "--active"]);
    h.vecd(&["users", "add", "alice", "--level", "user"]);
    h.vecd(&["ns", "add", "datasets/glove", "--owner", "alice", "--backend-config", "store", "--active"]);
    h.vecd(&["bind", "--to", "alice", "--role", "curate", "--ns", "datasets/glove"]);
    h.vecd(&["bind", "--to", "PUBLIC", "--role", "reader", "--ns", "datasets/glove"]);
    let tok_out = h.vecd(&["tokens", "create", "--user", "alice", "--description", "push key", "--expires", "30d"]);
    let token = token_from(&tok_out);

    // Listings reflect the setup.
    let ns_list = h.vecd(&["ns", "list"]);
    assert!(ns_list.contains("datasets/glove"));
    assert!(h.vecd(&["backends", "list"]).contains("store"));
    assert!(h.vecd(&["users", "list"]).contains("alice"));

    // Start the daemon and push the real dataset through it.
    let server = h.serve();
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());
    let outcome = execute(&push_opts(
        src.path(),
        format!("{}/datasets/glove/", server.base),
        Some(token.clone()),
    ))
    .expect("push through the live vecd binary");
    assert_eq!(outcome.version, 1);

    // Anonymous pull works via the PUBLIC reader binding.
    let client = reqwest::blocking::Client::new();
    let resp = client.get(server.url("datasets/glove/base.fvec")).send().unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.bytes().unwrap().as_ref(), b"VECDBASE");

    // The access log recorded the traffic.
    let log = h.vecd(&["log", "--tail", "100"]);
    assert!(log.contains("datasets/glove/base.fvec"));

    // A second init refuses to clobber the DB.
    let reinit = Command::new(vecd_bin())
        .env("VECD_CONFIG", &h.config_dir)
        .arg("--data-dir")
        .arg(&h.data_dir)
        .args(["init"])
        .output()
        .unwrap();
    assert!(!reinit.status.success(), "re-init must refuse to clobber");
}
