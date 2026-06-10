// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end proof: the **real** `vectordata push` engine driven against
//! an in-process `vecd` daemon. This is the Phase 1 acceptance criterion
//! from `docs/design/vecd-daemon.md` (decision #11) — "an authenticated,
//! conditional-write-honoring gateway [that] push/pull work against
//! end-to-end."
//!
//! It exercises the whole stack: the AAA cone (token → principal →
//! authorize), namespace→backend routing, and the DB-as-CAS-authority
//! conditional writes that push's single-provenance guarantee and its
//! conditional-write *probe* depend on.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use vecd::admin;
use vecd::db::Db;
use vecd::model::{Level, Listable};
use vecd::server::{self, AppState};

use vectordata::push::transport::TransportOptions;
use vectordata::push::{execute, ChecksumPolicy, Options};

/// An in-process vecd serving a pre-configured DB on a random port.
struct Vecd {
    port: u16,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Vecd {
    fn start(db: Db) -> Self {
        let state = AppState::new(db).expect("build state");
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
        let port = port_rx.recv_timeout(Duration::from_secs(5)).expect("vecd started");
        Vecd { port, shutdown: Some(sd_tx), thread: Some(thread) }
    }

    /// Publish-root URL for a namespace (trailing slash, as `.publish_url`
    /// expects).
    fn ns_url(&self, ns: &str) -> String {
        format!("http://127.0.0.1:{}/{ns}/", self.port)
    }

    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}/{path}", self.port)
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

/// A dataset directory push understands (mirrors the push test fixture).
fn make_dataset(dir: &std::path::Path) {
    std::fs::write(
        dir.join("dataset.yaml"),
        "name: glove\nattributes:\n  is_zero_vector_free: true\n  is_duplicate_vector_free: true\n",
    )
    .unwrap();
    std::fs::write(dir.join("base.fvec"), b"VECDBASE").unwrap();
}

fn opts(src: &std::path::Path, to: String, token: Option<String>) -> Options {
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
        cmd: "vectordata push (vecd test)".into(),
        actor: "tester@host".into(),
    }
}

/// Build a DB with a mem backend, one user, and a curate binding on `ns`.
/// Returns (db, the user's token plaintext). `mem_id` keeps the in-process
/// mem store distinct across tests.
fn setup(ns: &str, mem_id: &str, public_read: bool) -> (tempfile::TempDir, Db, String) {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::init(&dir.path().join("vecd.db")).unwrap();
    admin::add_backend(&mut db, "store", "mem", &format!("mem:{mem_id}"), None, None, None, true).unwrap();
    admin::add_user(&mut db, "alice", Level::User, None, None).unwrap();
    admin::add_namespace(&mut db, ns, "alice", Some("store"), true, Listable::Grantees, None, None).unwrap();
    admin::bind(&mut db, "alice", "curate", ns).unwrap();
    if public_read {
        admin::bind(&mut db, "PUBLIC", "reader", ns).unwrap();
    }
    let tok = admin::create_token(&mut db, "alice", "push key", Some("30d"), None).unwrap();
    (dir, db, tok.plaintext)
}

#[test]
fn candidate_namespaces_via_whoami_for_bound_user() {
    // A regular user with a curate binding: `/-/namespaces` is forbidden (not
    // operator), so we fall back to the writable namespaces in `whoami`.
    let (_d, db, token) = setup("datasets/glove", "vecd-ns-1", false);
    let server = Vecd::start(db);
    let nss = vectordata::endpoint::candidate_namespaces(&server.url(""), Some(&token));
    assert!(
        nss.iter().any(|n| n == "datasets/glove"),
        "writable namespace should surface via whoami: {nss:?}"
    );
}

#[test]
fn candidate_namespaces_via_namespaces_endpoint_for_superuser() {
    // The reported case: a superuser holds no explicit namespace binding (so
    // `whoami` lists none), yet `/-/namespaces` (operator+) returns the full
    // list — which is what drives the smart push default and `--to` completion.
    let (_d, mut db, _alice) = setup("datasets/glove", "vecd-ns-2", false);
    admin::add_user(&mut db, "root", Level::Superuser, None, None).unwrap();
    let root_tok = admin::create_token(&mut db, "root", "admin key", Some("30d"), None)
        .unwrap()
        .plaintext;
    let server = Vecd::start(db);

    // whoami alone gives a superuser no namespaces …
    let view = vectordata::endpoint::whoami(&server.url(""), Some(&root_tok)).unwrap();
    let whoami_count =
        view.get("namespaces").and_then(|n| n.as_array()).map(|a| a.len()).unwrap_or(0);
    // … but candidate_namespaces consults /-/namespaces and finds it.
    let nss = vectordata::endpoint::candidate_namespaces(&server.url(""), Some(&root_tok));
    assert!(
        nss.iter().any(|n| n == "datasets/glove"),
        "superuser should see the namespace via /-/namespaces (whoami listed {whoami_count}): {nss:?}"
    );
    // A backend-attached namespace IS suggested; the next test covers one without.
}

#[test]
fn push_to_backendless_namespace_explains_why() {
    // The reported case: the namespace exists and is active, but has NO storage
    // backend. vecd 404s on write — and now returns the reason in the body,
    // which the client surfaces instead of a blank status.
    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::init(&dir.path().join("vecd.db")).unwrap();
    admin::add_user(&mut db, "alice", Level::User, None, None).unwrap();
    // backend_config = None → the namespace has no storage backend.
    admin::add_namespace(&mut db, "datasets/nobk", "alice", None, true, Listable::Grantees, None, None)
        .unwrap();
    admin::bind(&mut db, "alice", "curate", "datasets/nobk").unwrap();
    let token = admin::create_token(&mut db, "alice", "k", Some("30d"), None).unwrap().plaintext;
    admin::add_user(&mut db, "root", Level::Superuser, None, None).unwrap();
    let root_tok = admin::create_token(&mut db, "root", "k", Some("30d"), None).unwrap().plaintext;
    let server = Vecd::start(db);

    // Writing 404s, but the reason is surfaced (not an opaque status).
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());
    let err = execute(&opts(src.path(), server.ns_url("datasets/nobk"), Some(token))).unwrap_err();
    match err {
        vectordata::push::Failure::Operational(m) => {
            assert!(
                m.contains("has no storage backend"),
                "the 404 should name the missing backend, not be opaque: {m}"
            );
            // No writable namespace exists anywhere → the endpoint-wide hint fires.
            assert!(
                m.contains("no namespace with a storage backend"),
                "an unconfigured endpoint should say so to an authenticated caller: {m}"
            );
        }
        other => panic!("expected an Operational error explaining the backend gap, got {other:?}"),
    }

    // The privileged namespace list omits a backendless namespace, so it's never
    // offered as a smart default / `--to` completion candidate.
    let suggested = vectordata::endpoint::candidate_namespaces(&server.url(""), Some(&root_tok));
    assert!(
        !suggested.iter().any(|n| n == "datasets/nobk"),
        "a backendless namespace must not be suggested: {suggested:?}"
    );
}

#[test]
fn dynamic_catalog_lists_datasets_under_a_namespace() {
    // No catalog was ever published, yet a GET of the namespace catalog returns
    // a live listing of the datasets stored under it — so a pushed dataset is
    // discoverable without a manual `catalog generate`/publish step.
    let (_d, db, token) = setup("datasets", "vecd-cat-1", false);
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    // Two datasets' dataset.yaml objects (as a push would land them), each with
    // the attributes + profiles a real dataset.yaml carries.
    for name in ["toy", "glove"] {
        let yaml = format!(
            "name: {name}\n\
             attributes:\n  distance_function: L2\n\
             profiles:\n  default:\n    base_vectors: profiles/base/base.fvecs\n"
        );
        client
            .put(server.url(&format!("datasets/{name}/dataset.yaml")))
            .bearer_auth(&token)
            .body(yaml)
            .send()
            .unwrap();
    }

    // The synthesized catalog lists both, with namespace-relative paths AND the
    // embedded layout (metric + profiles) read from each dataset.yaml.
    let resp = client.get(server.url("datasets/catalog.json")).bearer_auth(&token).send().unwrap();
    assert!(resp.status().is_success(), "catalog should be synthesized, got {}", resp.status());
    let cat: serde_json::Value = resp.json().unwrap();
    let arr = cat.as_array().expect("catalog is a JSON array");
    assert_eq!(arr.len(), 2, "{cat}");
    let toy = arr.iter().find(|e| e["name"] == "toy").expect("toy listed");
    assert_eq!(toy["path"], "toy/dataset.yaml", "{cat}");
    assert_eq!(toy["layout"]["attributes"]["distance_function"], "L2", "metric embedded: {cat}");
    assert!(toy["layout"]["profiles"]["default"].is_object(), "profiles embedded: {cat}");
    assert!(arr.iter().any(|e| e["name"] == "glove"), "catalog should list glove: {cat}");

    // An empty namespace has no datasets → a real 404, not an empty catalog.
    let (_d2, db2, tok2) = setup("empty", "vecd-cat-2", false);
    let server2 = Vecd::start(db2);
    let empty = reqwest::blocking::Client::new()
        .get(server2.url("empty/catalog.json"))
        .bearer_auth(&tok2)
        .send()
        .unwrap();
    assert_eq!(empty.status(), reqwest::StatusCode::NOT_FOUND);
}

#[test]
fn middleware_gives_every_error_a_body() {
    // A bare `Out::status(412)` in the handler (a CAS conflict) must still reach
    // the wire with a readable body — the response middleware guarantees it.
    let (_d, db, token) = setup("datasets/glove", "vecd-body-1", false);
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();
    let url = server.url("datasets/glove/.probe");
    let put = |b: &str| {
        client
            .put(&url)
            .bearer_auth(&token)
            .header("If-None-Match", "*")
            .body(b.to_string())
            .send()
            .unwrap()
    };

    let _ = put("1"); // conditional create
    let conflict = put("2"); // same precondition → 412
    assert_eq!(conflict.status(), reqwest::StatusCode::PRECONDITION_FAILED);
    assert!(
        !conflict.text().unwrap().trim().is_empty(),
        "the middleware must give even a bare 412 a readable body"
    );
}

#[test]
fn push_to_inactive_namespace_says_to_activate_it() {
    // The reported case: the namespace HAS a backend ("store") but is inactive
    // (config-only). vecd must name the real fix — activate it — not a backend
    // message.
    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::init(&dir.path().join("vecd.db")).unwrap();
    admin::add_backend(&mut db, "store", "mem", "mem:inactive-ns", None, None, None, true).unwrap();
    admin::add_user(&mut db, "alice", Level::User, None, None).unwrap();
    // active = false → config-only, despite having a backend.
    admin::add_namespace(
        &mut db,
        "datasets/dormant",
        "alice",
        Some("store"),
        false,
        Listable::Grantees,
        None,
        None,
    )
    .unwrap();
    admin::bind(&mut db, "alice", "curate", "datasets/dormant").unwrap();
    let token = admin::create_token(&mut db, "alice", "k", Some("30d"), None).unwrap().plaintext;
    let server = Vecd::start(db);

    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());
    let err = execute(&opts(src.path(), server.ns_url("datasets/dormant"), Some(token))).unwrap_err();
    match err {
        vectordata::push::Failure::Operational(m) => assert!(
            m.contains("is inactive") && m.contains("vecd ns set datasets/dormant --active"),
            "an inactive namespace should say to activate it: {m}"
        ),
        other => panic!("expected an Operational error telling us to activate, got {other:?}"),
    }
}

#[test]
fn push_succeeds_and_is_retrievable() {
    let (_d, db, token) = setup("datasets/glove", "vecd-e2e-1", false);
    let server = Vecd::start(db);
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());

    let outcome = execute(&opts(src.path(), server.ns_url("datasets/glove"), Some(token.clone())))
        .expect("authenticated push should succeed");
    assert_eq!(outcome.version, 1);
    // Two content files: dataset.yaml + base.fvec (SHA256SUMS/.publish_url/
    // pushlog are managed artifacts, not counted as adds).
    assert_eq!(outcome.added, 2);

    // The content is retrievable back over the wire *with* the token
    // (curate ⊇ read). The conditional-write probe and the begin→complete
    // CAS chain having succeeded is implied by the push returning Ok.
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(server.url("datasets/glove/base.fvec"))
        .bearer_auth(&token)
        .send()
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.bytes().unwrap().as_ref(), b"VECDBASE");

    // The provenance artifacts landed too.
    for key in ["SHA256SUMS", ".publish_url", "pushlog.jsonl"] {
        let r = client
            .get(server.url(&format!("datasets/glove/{key}")))
            .bearer_auth(&token)
            .send()
            .unwrap();
        assert_eq!(r.status(), 200, "{key} should be retrievable");
    }
}

#[test]
fn push_without_token_is_rejected() {
    let (_d, db, _token) = setup("datasets/glove", "vecd-e2e-2", false);
    let server = Vecd::start(db);
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());

    // No token → the namespace is private (no PUBLIC binding) → the
    // preflight is rejected → an operational auth failure.
    let err = execute(&opts(src.path(), server.ns_url("datasets/glove"), None))
        .expect_err("anonymous push must be refused");
    match err {
        vectordata::push::Failure::Operational(m) => {
            assert!(m.to_lowercase().contains("auth"), "expected auth failure, got: {m}");
        }
        other => panic!("expected operational auth failure, got {other:?}"),
    }
}

#[test]
fn read_only_token_cannot_push() {
    let (_d, mut db, _curate) = setup("datasets/glove", "vecd-e2e-3", false);
    // A delegated, read-only key for alice.
    let ro = admin::create_token(
        &mut db,
        "alice",
        "read-only",
        Some("7d"),
        Some(admin::parse_profile_spec("read datasets/glove").unwrap()),
    )
    .unwrap();
    let server = Vecd::start(db);
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());

    // The token can read but not write → the first conditional PUT (the
    // CAS probe) is forbidden, surfaced as an operational failure.
    let err = execute(&opts(src.path(), server.ns_url("datasets/glove"), Some(ro.plaintext)))
        .expect_err("read-only token must not be able to push");
    assert!(matches!(err, vectordata::push::Failure::Operational(_)), "got {err:?}");
}

#[test]
fn public_binding_enables_anonymous_pull() {
    let (_d, db, token) = setup("datasets/glove", "vecd-e2e-4", true);
    let server = Vecd::start(db);
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());
    execute(&opts(src.path(), server.ns_url("datasets/glove"), Some(token)))
        .expect("push should succeed");

    // PUBLIC reader binding → anonymous GET works (transparent public pull).
    let client = reqwest::blocking::Client::new();
    let resp = client.get(server.url("datasets/glove/base.fvec")).send().unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.bytes().unwrap().as_ref(), b"VECDBASE");

    // A namespace with no PUBLIC binding stays private: anonymous → 401.
    let priv_resp = client.get(server.url("datasets/secret/x")).send().unwrap();
    assert_eq!(priv_resp.status(), 401);
}

#[test]
fn whoami_reflects_effective_access() {
    let (_d, db, token) = setup("datasets/glove", "vecd-e2e-6", true);
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    // Authenticated: alice owns datasets/glove with full actions.
    let body = client
        .get(server.url("-/whoami"))
        .bearer_auth(&token)
        .send()
        .unwrap()
        .text()
        .unwrap();
    assert!(body.contains("\"identity\":\"alice\""), "{body}");
    assert!(body.contains("datasets/glove"), "{body}");
    assert!(body.contains("\"owner\":true"), "{body}");
    assert!(body.contains("\"authenticated\":true"), "{body}");

    // Anonymous: still sees the PUBLIC-readable namespace, unauthenticated.
    let anon = client.get(server.url("-/whoami")).send().unwrap().text().unwrap();
    assert!(anon.contains("\"authenticated\":false"), "{anon}");
    assert!(anon.contains("datasets/glove"), "{anon}");
}

#[test]
fn namespaces_endpoint_requires_privilege() {
    let (_d, mut db, alice_token) = setup("datasets/glove", "vecd-e2e-7", false);
    // An operator-level principal may read the namespace config.
    admin::add_user(&mut db, "ops", Level::Operator, None, None).unwrap();
    let ops = admin::create_token(&mut db, "ops", "ops key", Some("7d"), None).unwrap();
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    let r_user = client.get(server.url("-/namespaces")).bearer_auth(&alice_token).send().unwrap();
    assert_eq!(r_user.status(), 403, "a plain user must not read namespace config");

    let r_ops = client.get(server.url("-/namespaces")).bearer_auth(&ops.plaintext).send().unwrap();
    assert_eq!(r_ops.status(), 200);
    assert!(r_ops.text().unwrap().contains("datasets/glove"));
}

#[test]
fn list_endpoint_enumerates_keys_after_push() {
    let (_d, db, token) = setup("datasets/glove", "vecd-e2e-8", false);
    let server = Vecd::start(db);
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());
    execute(&opts(src.path(), server.ns_url("datasets/glove"), Some(token.clone())))
        .expect("push");

    let body = reqwest::blocking::Client::new()
        .get(server.url("datasets/glove/?list"))
        .bearer_auth(&token)
        .send()
        .unwrap()
        .text()
        .unwrap();
    assert!(body.contains("base.fvec"), "{body}");
    assert!(body.contains("pushlog.jsonl"), "{body}");
    assert!(body.contains("\"etag\""), "{body}");
}

#[test]
fn versions_pinned_reads_and_manifest() {
    let (_d, db, token) = setup("datasets/glove", "vecd-e2e-9", false);
    let server = Vecd::start(db);
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());

    // v1
    execute(&opts(src.path(), server.ns_url("datasets/glove"), Some(token.clone()))).expect("v1");

    // v2 — overwrite base.fvec (needs -m, since it changes remote data).
    std::fs::write(src.path().join("base.fvec"), b"VECDBASE2").unwrap();
    let mut o2 = opts(src.path(), server.ns_url("datasets/glove"), Some(token.clone()));
    o2.message = Some("bump base".into());
    let v2 = execute(&o2).expect("v2");
    assert_eq!(v2.version, 2);

    let client = reqwest::blocking::Client::new();

    // The version history lists both committed versions.
    let hist = client
        .get(server.url("-/versions/datasets/glove"))
        .bearer_auth(&token)
        .send()
        .unwrap()
        .text()
        .unwrap();
    assert!(hist.contains("\"v1\""), "{hist}");
    assert!(hist.contains("\"v2\""), "{hist}");

    // @latest serves v2 and echoes the resolved version header.
    let latest = client.get(server.url("datasets/glove/base.fvec")).bearer_auth(&token).send().unwrap();
    assert_eq!(latest.headers().get("x-vecd-version").unwrap(), "v2");
    assert_eq!(latest.bytes().unwrap().as_ref(), b"VECDBASE2");

    // A pinned read of @v1 returns the original content.
    let pinned = client
        .get(server.url("datasets/glove/@v1/base.fvec"))
        .bearer_auth(&token)
        .send()
        .unwrap();
    assert_eq!(pinned.status(), 200);
    assert_eq!(pinned.headers().get("x-vecd-version").unwrap(), "v1");
    assert_eq!(pinned.bytes().unwrap().as_ref(), b"VECDBASE");

    // The version manifest lists the keys with their content-keys.
    let manifest = client
        .get(server.url("datasets/glove/@v2/-/manifest"))
        .bearer_auth(&token)
        .send()
        .unwrap()
        .text()
        .unwrap();
    assert!(manifest.contains("base.fvec"), "{manifest}");
    assert!(manifest.contains("content_key"), "{manifest}");
    assert!(manifest.contains("\"v2\""), "{manifest}");
}

#[test]
fn expired_version_sweeps_to_stasis_and_serves_410() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("vecd.db");
    let mut db = Db::init(&db_path).unwrap();
    admin::add_backend(&mut db, "store", "mem", "mem:vecd-e2e-10", None, None, None, true).unwrap();
    admin::add_user(&mut db, "alice", Level::User, None, None).unwrap();
    admin::add_namespace(&mut db, "datasets/glove", "alice", Some("store"), true, Listable::Grantees, None, None).unwrap();
    admin::bind(&mut db, "alice", "curate", "datasets/glove").unwrap();
    admin::bind(&mut db, "PUBLIC", "reader", "datasets/glove").unwrap();
    let token = admin::create_token(&mut db, "alice", "push", Some("30d"), None).unwrap().plaintext;

    let server = Vecd::start(db);
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());
    execute(&opts(src.path(), server.ns_url("datasets/glove"), Some(token.clone()))).expect("push v1");

    let client = reqwest::blocking::Client::new();
    assert_eq!(client.get(server.url("datasets/glove/base.fvec")).send().unwrap().status(), 200);

    // Force the version's lifecycle into the past and sweep it (a second
    // connection — WAL lets the running daemon observe the change).
    let mut admin_conn = Db::open(&db_path).unwrap();
    admin_conn.conn().execute("UPDATE versions SET expires_at=1", []).unwrap();
    assert_eq!(vecd::lifetime::sweep(&mut admin_conn).unwrap(), 1);

    // @latest now reports the dataset gone (410 + lifecycle header).
    let gone = client.get(server.url("datasets/glove/base.fvec")).send().unwrap();
    assert_eq!(gone.status(), 410);
    assert_eq!(gone.headers().get("x-vecd-lifecycle").unwrap(), "stasis");

    // A pinned read of the stasis version is also 410.
    let pinned = client.get(server.url("datasets/glove/@v1/base.fvec")).bearer_auth(&token).send().unwrap();
    assert_eq!(pinned.status(), 410);

    // An admin extend restores it; @latest serves again.
    vecd::lifetime::extend(&mut admin_conn, "datasets/glove", "v1", Some(3600)).unwrap();
    let back = client.get(server.url("datasets/glove/base.fvec")).send().unwrap();
    assert_eq!(back.status(), 200);
    assert_eq!(back.bytes().unwrap().as_ref(), b"VECDBASE");
}

#[test]
fn password_login_and_delegated_token_issue() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::init(&dir.path().join("vecd.db")).unwrap();
    admin::add_backend(&mut db, "store", "mem", "mem:vecd-e2e-11", None, None, None, true).unwrap();
    admin::add_user(&mut db, "alice", Level::User, Some("s3kr3t"), None).unwrap();
    admin::add_namespace(&mut db, "datasets/glove", "alice", Some("store"), true, Listable::Grantees, None, None).unwrap();
    admin::bind(&mut db, "alice", "curate", "datasets/glove").unwrap();
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    // Wrong password → 401.
    let bad = client
        .post(server.url("auth/token"))
        .json(&serde_json::json!({"user":"alice","password":"nope"}))
        .send()
        .unwrap();
    assert_eq!(bad.status(), 401);

    // Correct password → a usable token (immediately, via reload_now).
    let login: serde_json::Value = client
        .post(server.url("auth/token"))
        .json(&serde_json::json!({"user":"alice","password":"s3kr3t"}))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let token = login["token"].as_str().unwrap().to_string();
    assert!(token.starts_with("vd_"));
    let who = client.get(server.url("-/whoami")).bearer_auth(&token).send().unwrap().text().unwrap();
    assert!(who.contains("\"identity\":\"alice\""), "{who}");

    // The session token mints a delegated, read-only key.
    let issued: serde_json::Value = client
        .post(server.url("tokens"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"description":"collaborator","profile":"read datasets/glove","expires":"7d"}))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let ro = issued["token"].as_str().unwrap().to_string();
    let ro_id = issued["id"].as_i64().unwrap();

    // The delegated key can read but not write (PUT → 403).
    let put = client.put(server.url("datasets/glove/x")).bearer_auth(&ro).body("hi").send().unwrap();
    assert_eq!(put.status(), 403);

    // The issuer revokes it; it stops working.
    let del = client.delete(server.url(&format!("tokens/{ro_id}"))).bearer_auth(&token).send().unwrap();
    assert_eq!(del.status(), 204);
    let after = client.get(server.url("-/whoami")).bearer_auth(&ro).send().unwrap();
    assert_eq!(after.status(), 401);
}

#[test]
fn repeated_auth_failures_are_throttled() {
    let (_d, db, _t) = setup("datasets/glove", "vecd-e2e-13", false);
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();
    // Hammer with an invalid token; after the failure threshold the source
    // is throttled with 429.
    let mut saw_429 = false;
    for _ in 0..20 {
        let r = client
            .get(server.url("datasets/glove/x"))
            .bearer_auth("vd_bogustoken")
            .send()
            .unwrap();
        if r.status() == 429 {
            saw_429 = true;
            break;
        }
    }
    assert!(saw_429, "repeated bad-token requests should eventually be throttled");
}

#[test]
fn metrics_endpoint_exposes_counters() {
    let (_d, db, _t) = setup("datasets/glove", "vecd-e2e-12", false);
    let server = Vecd::start(db);
    let body = reqwest::blocking::get(server.url("metrics")).unwrap().text().unwrap();
    assert!(body.contains("vecd_requests_total"), "{body}");
    assert!(body.contains("vecd_auth_failures_total"), "{body}");
}

#[test]
fn push_delete_removes_orphans_over_vecd() {
    let (_d, db, token) = setup("datasets/glove", "vecd-e2e-14", false);
    let server = Vecd::start(db);
    let src = tempfile::tempdir().unwrap();
    std::fs::write(src.path().join("dataset.yaml"), "name: glove\n").unwrap();
    std::fs::write(src.path().join("a.fvec"), b"AAAA").unwrap();
    std::fs::write(src.path().join("b.fvec"), b"BBBB").unwrap();

    // v1 publishes both files.
    execute(&opts(src.path(), server.ns_url("datasets/glove"), Some(token.clone()))).expect("v1");
    let client = reqwest::blocking::Client::new();
    assert_eq!(client.get(server.url("datasets/glove/b.fvec")).bearer_auth(&token).send().unwrap().status(), 200);

    // Remove b.fvec locally and re-push with --delete (needs -m): the
    // orphan must disappear from the current version via vecd's ?list.
    std::fs::remove_file(src.path().join("b.fvec")).unwrap();
    let mut o2 = opts(src.path(), server.ns_url("datasets/glove"), Some(token.clone()));
    o2.delete = true;
    o2.message = Some("drop b".into());
    let v2 = execute(&o2).expect("v2 with --delete");
    assert_eq!(v2.deleted, 1, "b.fvec should be detected as an orphan and deleted");

    // a.fvec remains; b.fvec is gone from @latest.
    assert_eq!(client.get(server.url("datasets/glove/a.fvec")).bearer_auth(&token).send().unwrap().status(), 200);
    assert_eq!(client.get(server.url("datasets/glove/b.fvec")).bearer_auth(&token).send().unwrap().status(), 404);
}

#[test]
fn large_object_streams_past_the_old_body_cap() {
    // A >2 MiB object once tripped axum's default body limit (413). With the
    // streaming PUT path it rides straight to the backend and round-trips
    // byte-for-byte, with an envelope ETag that is independent of content.
    let (_d, db, token) = setup("datasets/glove", "vecd-e2e-15", false);
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    // 3 MiB of non-trivial bytes (so a sparse/zero shortcut can't pass).
    let body: Vec<u8> = (0..3 * 1024 * 1024).map(|i| (i * 31 + 7) as u8).collect();
    let put = client
        .put(server.url("datasets/glove/big.bin"))
        .bearer_auth(&token)
        .body(body.clone())
        .send()
        .unwrap();
    assert_eq!(put.status(), 201, "large PUT must succeed, not 413");
    let etag = put.headers().get("etag").unwrap().to_str().unwrap().to_string();
    assert!(!etag.is_empty());

    let got = client
        .get(server.url("datasets/glove/big.bin"))
        .bearer_auth(&token)
        .send()
        .unwrap();
    assert_eq!(got.status(), 200);
    assert_eq!(got.bytes().unwrap().as_ref(), body.as_slice(), "bytes must round-trip exactly");
}

#[test]
fn push_uses_the_resumable_upload_protocol_for_content() {
    // The client detects a vecd endpoint and uploads every content file via
    // the resumable POST/PATCH/finalize protocol (≥2 splits each), not a
    // plain PUT. Proof: the server's `uploads` table carries a completed
    // resumable-upload row per content file after the push.
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("vecd.db");
    let mut db = Db::init(&db_path).unwrap();
    admin::add_backend(&mut db, "store", "mem", "mem:vecd-e2e-17", None, None, None, true).unwrap();
    admin::add_user(&mut db, "alice", Level::User, None, None).unwrap();
    admin::add_namespace(&mut db, "datasets/glove", "alice", Some("store"), true, Listable::Grantees, None, None).unwrap();
    admin::bind(&mut db, "alice", "curate", "datasets/glove").unwrap();
    let token = admin::create_token(&mut db, "alice", "push", Some("30d"), None).unwrap().plaintext;

    let server = Vecd::start(db);
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path()); // dataset.yaml + base.fvec (two content files)
    execute(&opts(src.path(), server.ns_url("datasets/glove"), Some(token.clone()))).expect("push");

    // A second connection (WAL) inspects the upload bookkeeping.
    let admin_conn = Db::open(&db_path).unwrap();
    let completed: i64 = admin_conn
        .conn()
        .query_row("SELECT COUNT(*) FROM uploads WHERE complete=1", [], |r| r.get(0))
        .unwrap();
    assert!(completed >= 2, "expected a completed resumable upload per content file, got {completed}");
    // Each completed upload recorded its committed envelope ETag.
    let with_etag: i64 = admin_conn
        .conn()
        .query_row("SELECT COUNT(*) FROM uploads WHERE complete=1 AND etag IS NOT NULL", [], |r| r.get(0))
        .unwrap();
    assert_eq!(with_etag, completed, "every completed upload should carry its ETag");
}

#[test]
fn empty_object_streams_and_round_trips() {
    // A zero-byte object must finalize too: the streaming writer materializes
    // an empty staging blob so there is something to promote. (Regression:
    // empty `.ivecs` cache artifacts 500'd before this.)
    let (_d, db, token) = setup("datasets/glove", "vecd-e2e-16", false);
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    let put = client.put(server.url("datasets/glove/empty.bin")).bearer_auth(&token).body(Vec::<u8>::new()).send().unwrap();
    assert_eq!(put.status(), 201, "empty PUT must succeed");

    let got = client.get(server.url("datasets/glove/empty.bin")).bearer_auth(&token).send().unwrap();
    assert_eq!(got.status(), 200);
    assert_eq!(got.headers().get("content-length").unwrap(), "0");
    assert!(got.bytes().unwrap().is_empty());
}

#[test]
fn healthz_is_open() {
    let (_d, db, _t) = setup("datasets/glove", "vecd-e2e-5", false);
    let server = Vecd::start(db);
    let resp = reqwest::blocking::get(server.url("healthz")).unwrap();
    assert_eq!(resp.status(), 200);
}
