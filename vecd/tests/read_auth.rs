// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Read-side bearer auth, end-to-end through the **public `vectordata`
//! reader API** against a live `vecd`. A private dataset is unreadable
//! anonymously and readable once a token is resolved by origin
//! (`$VECTORDATA_TOKEN`). This is the additive read-path capability — the
//! frozen `http_storage.rs` guardrail proves the no-token path is
//! unchanged; this proves the with-token path works.
//!
//! Its own test binary, so the one `$VECTORDATA_TOKEN`/`$VECTORDATA_HOME`
//! mutation is process-local and the single test runs the steps in order.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use vecd::admin;
use vecd::db::Db;
use vecd::model::{Level, Listable};
use vecd::server::{self, AppState};

use vectordata::push::transport::TransportOptions;
use vectordata::push::{execute, ChecksumPolicy, Options};
use vectordata::XvecReader;

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

/// A tiny valid `.fvec`: two dim-3 f32 vectors.
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

#[test]
fn private_dataset_needs_a_token_to_read() {
    // Isolate the credential store + cache away from the dev machine's
    // real config. Configure a cache dir so the reader is fully set up and
    // the *only* difference between the two reads below is the token.
    let home = tempfile::tempdir().unwrap();
    let cache = home.path().join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::write(
        home.path().join("settings.yaml"),
        format!("cache_dir: {}\n", cache.display()),
    )
    .unwrap();
    unsafe {
        std::env::set_var("VECTORDATA_HOME", home.path());
        std::env::remove_var("VECTORDATA_TOKEN");
    }

    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::init(&dir.path().join("vecd.db")).unwrap();
    admin::add_backend(&mut db, "store", "mem", "mem:read-auth", None, None, None, true).unwrap();
    admin::add_user(&mut db, "alice", Level::User, None, None).unwrap();
    // PRIVATE: nothing bound to PUBLIC/KNOWN — owner-only.
    admin::add_namespace(&mut db, "datasets/priv", "alice", Some("store"), true, Listable::Grantees, None, None).unwrap();
    admin::bind(&mut db, "alice", "curate", "datasets/priv").unwrap();
    let token = admin::create_token(&mut db, "alice", "key", Some("30d"), None).unwrap().plaintext;

    let server = Vecd::start(db);

    // Push a real (tiny) fvec dataset with alice's token.
    let src = tempfile::tempdir().unwrap();
    std::fs::write(src.path().join("dataset.yaml"), "name: priv\n").unwrap();
    std::fs::write(src.path().join("base.fvec"), fvec_bytes()).unwrap();
    let opts = Options {
        path: src.path().to_path_buf(),
        to: Some(server.url("datasets/priv/")),
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
        transport: TransportOptions { token: Some(token.clone()), profile: None, endpoint_url: None },
        cmd: "push".into(),
        actor: "tester".into(),
    };
    execute(&opts).expect("push private dataset");

    let url = server.url("datasets/priv/base.fvec");

    // Anonymous read of a private dataset → fails (the reader can't open it).
    assert!(
        XvecReader::<f32>::open(&url).is_err(),
        "private dataset must not be readable without a token"
    );

    // With a token resolved by origin ($VECTORDATA_TOKEN), the read succeeds.
    unsafe { std::env::set_var("VECTORDATA_TOKEN", &token) };
    let reader = XvecReader::<f32>::open(&url).expect("private read with a token");
    assert_eq!(reader.count(), 2);
    assert_eq!(reader.get_slice(0), &[1.0f32, 2.0, 3.0][..]);
    unsafe { std::env::remove_var("VECTORDATA_TOKEN") };
}
