// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Client-driven, resumable backup and restore, end-to-end: push a
//! versioned dataset to vecd **A**, mirror it off-system (content-addressed,
//! resumable), then restore it into a fresh vecd **B** and read it back —
//! a genuine cross-store round trip.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use vecd::admin;
use vecd::db::Db;
use vecd::model::{Level, Listable};
use vecd::server::{self, AppState};

use vectordata::backup;
use vectordata::push::transport::TransportOptions;
use vectordata::push::{execute, ChecksumPolicy, Options};

struct Vecd {
    port: u16,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Vecd {
    fn start(db: Db) -> Self {
        let state = AppState::new(db).unwrap();
        let (port_tx, port_rx) = mpsc::channel();
        let (sd_tx, sd_rx) = tokio::sync::oneshot::channel::<()>();
        let thread = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                port_tx.send(listener.local_addr().unwrap().port()).unwrap();
                server::serve_listener(state, listener, async move {
                    let _ = sd_rx.await;
                })
                .await
                .unwrap();
            });
        });
        let port = port_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        Vecd { port, shutdown: Some(sd_tx), thread: Some(thread) }
    }
    fn url(&self, p: &str) -> String {
        format!("http://127.0.0.1:{}/{p}", self.port)
    }
    fn root(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}
impl Drop for Vecd {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

/// A vecd with one private curate-bound dataset namespace; returns the token.
fn server_with_ns(mem_id: &str, ns: &str) -> (Vecd, String) {
    let dir = tempfile::tempdir().unwrap();
    // Leak the tempdir so the mem backend (process-global by id) outlives it.
    let path = dir.keep().join("vecd.db");
    let mut db = Db::init(&path).unwrap();
    admin::add_backend(&mut db, "store", "mem", &format!("mem:{mem_id}"), None, None, None, true).unwrap();
    admin::add_user(&mut db, "alice", Level::User, None, None).unwrap();
    admin::add_namespace(&mut db, ns, "alice", Some("store"), true, Listable::Grantees, None, None).unwrap();
    admin::bind(&mut db, "alice", "curate", ns).unwrap();
    let token = admin::create_token(&mut db, "alice", "key", Some("30d"), None).unwrap().plaintext;
    (Vecd::start(db), token)
}

fn push_opts(src: &std::path::Path, to: String, token: &str, message: Option<&str>) -> Options {
    Options {
        path: src.to_path_buf(),
        to: Some(to),
        message: message.map(String::from),
        raw: false,
        checksums: ChecksumPolicy::Auto,
        dry_run: false,
        no_check: true,
        assume_yes: true,
        delete: false,
        abort_incomplete: false,
        concurrency: 4,
        files: None,
        transport: TransportOptions { token: Some(token.to_string()), profile: None, endpoint_url: None },
        cmd: "push".into(),
        actor: "tester".into(),
    }
}

#[test]
fn backup_then_restore_round_trip() {
    // ── vecd A: push two versions of a dataset ──────────────────────
    let (a, a_token) = server_with_ns("backup-a", "datasets/glove");
    let src = tempfile::tempdir().unwrap();
    std::fs::write(src.path().join("dataset.yaml"), "name: glove\n").unwrap();
    std::fs::write(src.path().join("base.fvec"), b"VERSION-ONE").unwrap();
    execute(&push_opts(src.path(), a.url("datasets/glove/"), &a_token, None)).expect("v1");
    std::fs::write(src.path().join("base.fvec"), b"VERSION-TWO").unwrap();
    execute(&push_opts(src.path(), a.url("datasets/glove/"), &a_token, Some("bump"))).expect("v2");

    // ── backup A → a content-addressed mirror ───────────────────────
    let mirror = tempfile::tempdir().unwrap();
    let dest = mirror.path().to_string_lossy().to_string();
    let s = backup::run_backup(&a.root(), &dest, false, Some(&a_token)).expect("backup");
    assert_eq!(s.namespaces, 1);
    assert_eq!(s.versions, 2, "both committed versions mirrored");
    assert!(s.blobs_fetched > 0);
    // The mirror is a content-addressed tree.
    assert!(mirror.path().join("blobs").read_dir().unwrap().count() > 0);
    assert!(mirror.path().join("ns/datasets/glove/versions.json").exists());
    assert!(mirror.path().join("ns/datasets/glove/@v2/manifest.json").exists());

    // ── resume: a second run re-fetches nothing ─────────────────────
    let s2 = backup::run_backup(&a.root(), &dest, true, Some(&a_token)).expect("incremental backup");
    assert_eq!(s2.versions_skipped, 2, "completed versions are skipped on resume");
    assert_eq!(s2.blobs_fetched, 0, "no blob re-fetched");

    // ── restore the mirror into a fresh vecd B ──────────────────────
    let (b, b_token) = server_with_ns("backup-b", "datasets/glove");
    let r = backup::run_restore(&dest, &b.root(), Some(&b_token)).expect("restore");
    assert_eq!(r.namespaces, 1);

    // B now serves the latest (v2) content.
    let got = reqwest::blocking::Client::new()
        .get(b.url("datasets/glove/base.fvec"))
        .bearer_auth(&b_token)
        .send()
        .unwrap();
    assert_eq!(got.status(), 200);
    assert_eq!(got.bytes().unwrap().as_ref(), b"VERSION-TWO");
}
