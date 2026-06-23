// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test of the **compiled `vecd` binary**: drive the real CLI
//! (`init` → `backends add` → `users add` → `ns add` → `bind` →
//! `tokens create`), start `vecd serve` as a child process on an
//! ephemeral port, then run the real `vectordata push` engine against it
//! over HTTP and pull results back two ways:
//!
//!   * a **public** dataset (`PUBLIC reader` bound) pulled **anonymously**, and
//!   * a **private** dataset (owner-only) that is refused anonymously and
//!     pulled with an **authenticated** `vectordata` reader after
//!     `vectordata login` records the endpoint credential — the binary-level
//!     proof of authenticated upload **and** download.
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

/// A tiny valid `.fvec` (two dim-3 f32 vectors) so the private dataset can
/// be opened + verified by the real `XvecReader` after an authenticated pull.
fn fvec_bytes() -> Vec<u8> {
    let mut b = Vec::new();
    for v in [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]] {
        b.extend_from_slice(&3i32.to_le_bytes());
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    b
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
            .args(["daemon", "stop"])
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
    assert!(h.vecd(&["daemon", "status"]).contains("not running"));

    // start daemonizes; the parent returns once the pidfile is written.
    let started = h.vecd(&["daemon", "start", "--bind", "127.0.0.1:0"]);
    assert!(started.contains("started"), "{started}");
    let _guard = DaemonGuard { h: &h };

    // Wait for it to publish its address + answer /healthz.
    let base = wait_for_endpoint(&h.data_dir.join("vecd.addr"));
    let resp = reqwest::blocking::get(format!("{base}/healthz")).unwrap();
    assert_eq!(resp.status(), 200);

    // status reports running + the bound address.
    let status = h.vecd(&["daemon", "status"]);
    assert!(status.contains("running"), "{status}");

    // stop drains it; status then reports not running.
    let stopped = h.vecd(&["daemon", "stop"]);
    assert!(stopped.contains("stopped"), "{stopped}");

    // Give the OS a moment, then confirm it's down.
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && h.vecd(&["daemon", "status"]).contains("is running") {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(h.vecd(&["daemon", "status"]).contains("not running"));
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
    h.vecd(&["store", "backends", "add", "store", "--kind", "local", "--endpoint",
             &format!("local:{}", objects.display()), "--active"]);
    h.vecd(&["access", "users", "add", "alice", "--level", "user"]);
    h.vecd(&["store", "ns", "add", "datasets/glove", "--owner", "alice", "--backend-config", "store", "--active"]);
    h.vecd(&["access", "bind", "--to", "alice", "--role", "curate", "--ns", "datasets/glove"]);
    h.vecd(&["access", "bind", "--to", "PUBLIC", "--role", "reader", "--ns", "datasets/glove"]);
    // A second namespace with NO PUBLIC binding — owner-only (private).
    // Created before `serve` so the running server sees it at startup.
    h.vecd(&["store", "ns", "add", "datasets/priv", "--owner", "alice", "--backend-config", "store", "--active"]);
    h.vecd(&["access", "bind", "--to", "alice", "--role", "curate", "--ns", "datasets/priv"]);
    let tok_out = h.vecd(&["access", "tokens", "create", "--user", "alice", "--description", "push key", "--expires", "30d"]);
    let token = token_from(&tok_out);

    // Listings reflect the setup.
    let ns_list = h.vecd(&["store", "ns", "list"]);
    assert!(ns_list.contains("datasets/glove"));
    assert!(h.vecd(&["store", "backends", "list"]).contains("store"));
    assert!(h.vecd(&["access", "users", "list"]).contains("alice"));

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

    // ── Authenticated upload + download of the PRIVATE dataset ────────
    // (namespace `datasets/priv` set up before `serve`, above.)
    // Authenticated UPLOAD: push a real fvec privately with alice's token.
    let psrc = tempfile::tempdir().unwrap();
    std::fs::write(psrc.path().join("dataset.yaml"), "name: priv\n").unwrap();
    std::fs::write(psrc.path().join("base.fvec"), fvec_bytes()).unwrap();
    let pout = execute(&push_opts(
        psrc.path(),
        format!("{}/datasets/priv/", server.base),
        Some(token.clone()),
    ))
    .expect("authenticated private push");
    assert_eq!(pout.version, 1);

    let priv_facet = server.url("datasets/priv/base.fvec");

    // Anonymous download of the private dataset is refused.
    let anon = client.get(&priv_facet).send().unwrap();
    assert!(
        !anon.status().is_success(),
        "private dataset must not be readable anonymously (got {})",
        anon.status()
    );

    // Isolate the vectordata client's credential store + cache (this only
    // affects the client side — vecd uses VECD_CONFIG/--data-dir), and make
    // sure no env token can shortcut the resolution under test.
    let vd_home = tempfile::tempdir().unwrap();
    let vd_cache = vd_home.path().join("cache");
    std::fs::create_dir_all(&vd_cache).unwrap();
    std::fs::write(
        vd_home.path().join("settings.yaml"),
        format!("cache_dir: {}\n", vd_cache.display()),
    )
    .unwrap();
    unsafe {
        std::env::set_var("VECTORDATA_HOME", vd_home.path());
        std::env::remove_var("VECTORDATA_TOKEN");
    }

    // `vectordata login` (the binary's library entry) records alice's token
    // for the catalog URL — the CLI counterpart of the auth config view.
    assert_eq!(
        vectordata::client_cli::login(
            &format!("{}/datasets/priv/", server.base),
            Some("alice"),
            Some(&token),
            None,
            None,
        ),
        0,
        "vectordata login should store the endpoint credential"
    );
    // In-session: pick up the just-written credential (the long-running
    // refresh the explorer relies on).
    vectordata::credentials::reload_credentials();

    // Authenticated DOWNLOAD via the stored credential alone — no
    // $VECTORDATA_TOKEN — returns the exact uploaded vectors.
    let reader = vectordata::XvecReader::<f32>::open(&priv_facet)
        .expect("authenticated private download via the stored login credential");
    assert_eq!(reader.count(), 2);
    assert_eq!(reader.get_slice(0), &[1.0f32, 2.0, 3.0][..]);
    assert_eq!(reader.get_slice(1), &[4.0f32, 5.0, 6.0][..]);

    unsafe {
        std::env::remove_var("VECTORDATA_HOME");
    }

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
