// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end demonstration of **authenticated dataset upload + download
//! driven by the explorer's auth configuration view**, against a live
//! `vecd`.
//!
//! The explorer's "Set auth" view records a credential via
//! [`vectordata::credentials::set_catalog_credential`] — keyed by the
//! catalog URL (so reads authenticate by URL) and tagged with the catalog
//! name (so the config view can join credentials to catalogs pairwise and
//! show the lock indicator). This test drives exactly that core:
//!
//!   1. **Upload** — a real `vectordata push` with alice's token writes a
//!      private dataset into a private (owner-only) vecd namespace.
//!   2. Anonymous download of that private dataset **fails**.
//!   3. The auth view's core records a credential for the catalog URL.
//!      [`credential_for_url`] then reports it (the lock indicator), with
//!      the catalog name-tag and user the view displays.
//!   4. **Download** — the public `vectordata` reader opens the private
//!      facet and returns the exact bytes, authenticating purely from the
//!      stored credential, with **no `$VECTORDATA_TOKEN` in the
//!      environment**. (This also exercises the in-session credential-cache
//!      refresh: the failed anonymous read initialises the process cache
//!      empty, and `set_catalog_credential` must refresh it so the read
//!      authenticates without a restart.)
//!   5. The view's "clear auth" core removes the credential and the
//!      indicator goes away.
//!
//! Companion to `read_auth.rs` (which drove the token via `$VECTORDATA_TOKEN`):
//! here the credential comes from the config-view code path. Its own test
//! binary so the one `$VECTORDATA_HOME`/`$VECTORDATA_TOKEN` mutation is
//! process-local.

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

/// An in-process vecd serving a pre-configured DB on a random port.
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
fn auth_view_credential_enables_private_upload_and_download() {
    // Isolate the credential store + cache from the dev machine's real
    // config, and make sure no env token can shortcut the resolution we're
    // testing — the credential must come from the config-view core.
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

    // A vecd with a PRIVATE (owner-only) namespace owned by alice.
    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::init(&dir.path().join("vecd.db")).unwrap();
    admin::add_backend(&mut db, "store", "mem", "mem:auth-view", None, None, None, true).unwrap();
    admin::add_user(&mut db, "alice", Level::User, None, None).unwrap();
    admin::add_namespace(&mut db, "datasets/priv", "alice", Some("store"), true, Listable::Grantees, None, None).unwrap();
    admin::bind(&mut db, "alice", "curate", "datasets/priv").unwrap();
    let token = admin::create_token(&mut db, "alice", "key", Some("30d"), None).unwrap().plaintext;

    let server = Vecd::start(db);

    // ── 1. Authenticated UPLOAD: a real `vectordata push` with the token.
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
    execute(&opts).expect("authenticated push (upload) of the private dataset");

    // The catalog URL the config view would store the credential against
    // (the dataset directory), and a facet URL underneath it.
    let catalog_url = server.url("datasets/priv/");
    let facet_url = server.url("datasets/priv/base.fvec");

    // ── 2. Anonymous download of a private dataset → fails.
    assert!(
        XvecReader::<f32>::open(&facet_url).is_err(),
        "private dataset must not be readable before a credential is recorded",
    );

    // ── 3. Auth config-view core: record the credential (URL-keyed,
    //        name-tagged). This is what the explorer's "Set auth" view runs.
    vectordata::credentials::set_catalog_credential("priv-cat", &catalog_url, &token, Some("alice"))
        .expect("auth view records the credential");

    // What the config view then displays: the lock indicator (a credential
    // resolves for the catalog's URLs) with the name-tag and user.
    let shown = vectordata::credentials::credential_for_url(&facet_url)
        .expect("indicator: a credential now resolves for the catalog URL");
    assert_eq!(shown.catalog.as_deref(), Some("priv-cat"), "credential is name-tagged for the pairwise join");
    assert_eq!(shown.user.as_deref(), Some("alice"));

    // ── 4. Authenticated DOWNLOAD via the stored credential alone — no
    //        $VECTORDATA_TOKEN in the environment. The exact uploaded bytes
    //        come back.
    let reader = XvecReader::<f32>::open(&facet_url)
        .expect("authenticated read using only the stored credential");
    assert_eq!(reader.count(), 2);
    assert_eq!(reader.get_slice(0), &[1.0f32, 2.0, 3.0][..]);
    assert_eq!(reader.get_slice(1), &[4.0f32, 5.0, 6.0][..]);

    // ── 5. Auth config-view "clear" core: remove the credential; the
    //        indicator goes away. (A subsequent read may still hit the local
    //        cache populated above, so we assert on the indicator, not a
    //        re-fetch.)
    assert!(
        vectordata::credentials::clear_catalog_credential(&catalog_url).expect("clear succeeds"),
        "a credential was present to clear",
    );
    assert!(
        vectordata::credentials::credential_for_url(&facet_url).is_none(),
        "indicator: the credential is gone after clearing",
    );

    unsafe {
        std::env::remove_var("VECTORDATA_HOME");
    }
}
