// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end proof of the resumable / concurrent / sparse upload protocol
//! (the IETF "Resumable Uploads for HTTP" convention with vecd's
//! parallel-sparse extension): `POST` create → `PATCH` chunks (in order, out
//! of order, and concurrently) → `HEAD` resume → finalize-on-full → `GET`
//! returns the exact bytes. vecd writes the stream faithfully and never
//! hashes content; the ETag is envelope-only.

use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use vecd::admin;
use vecd::db::Db;
use vecd::model::{Level, Listable};
use vecd::server::{self, AppState};

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
            let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
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

/// A namespace `datasets/glove` owned by alice (curate), mem-backed.
fn setup(mem_id: &str) -> (tempfile::TempDir, Db, String) {
    setup_quota(mem_id, None)
}

/// As [`setup`], with an optional namespace byte quota (a size spec).
fn setup_quota(mem_id: &str, quota: Option<&str>) -> (tempfile::TempDir, Db, String) {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::init(&dir.path().join("vecd.db")).unwrap();
    admin::add_backend(&mut db, "store", "mem", &format!("mem:{mem_id}"), None, None, None, true).unwrap();
    admin::add_user(&mut db, "alice", Level::User, None, None).unwrap();
    admin::add_namespace(&mut db, "datasets/glove", "alice", Some("store"), true, Listable::Grantees, None, quota)
        .unwrap();
    admin::bind(&mut db, "alice", "curate", "datasets/glove").unwrap();
    let tok = admin::create_token(&mut db, "alice", "push key", Some("30d"), None).unwrap();
    (dir, db, tok.plaintext)
}

/// Deterministic, non-trivial payload (so a zero-fill shortcut can't pass).
fn payload(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 37 + 11) as u8).collect()
}

/// `POST` to create an upload of `total` bytes; returns the absolute upload
/// URL parsed from `Location`.
fn create_upload(client: &reqwest::blocking::Client, server: &Vecd, key: &str, total: usize, token: &str) -> String {
    let resp = client
        .post(server.url(key))
        .bearer_auth(token)
        .header("Upload-Length", total.to_string())
        .send()
        .unwrap();
    assert_eq!(resp.status(), 201, "create upload");
    assert_eq!(resp.headers().get("upload-offset").unwrap(), "0");
    let loc = resp.headers().get("location").unwrap().to_str().unwrap().to_string();
    server.url(loc.trim_start_matches('/'))
}

fn patch(
    client: &reqwest::blocking::Client,
    upload_url: &str,
    offset: usize,
    data: &[u8],
    token: &str,
) -> reqwest::blocking::Response {
    client
        .patch(upload_url)
        .bearer_auth(token)
        .header("Upload-Offset", offset.to_string())
        .body(data.to_vec())
        .send()
        .unwrap()
}

#[test]
fn sequential_chunks_upload_and_finalize() {
    let (_d, db, token) = setup("vecd-resumable-1");
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    let total = 2000;
    let body = payload(total);
    let upload = create_upload(&client, &server, "datasets/glove/seq.bin", total, &token);

    // Eight in-order chunks; the acked Upload-Offset advances each time.
    let chunk = 250;
    for (i, window) in body.chunks(chunk).enumerate() {
        let off = i * chunk;
        let r = patch(&client, &upload, off, window, &token);
        assert_eq!(r.status(), 204);
        let acked: usize = r.headers().get("upload-offset").unwrap().to_str().unwrap().parse().unwrap();
        assert_eq!(acked, off + window.len(), "offset advances along the prefix");
        let complete = r.headers().get("upload-complete").unwrap() == "?1";
        assert_eq!(complete, off + window.len() == total, "complete only on the last chunk");
    }

    // The object is live and byte-exact; its ETag is envelope-only (opaque).
    let got = client.get(server.url("datasets/glove/seq.bin")).bearer_auth(&token).send().unwrap();
    assert_eq!(got.status(), 200);
    assert_eq!(got.bytes().unwrap().as_ref(), body.as_slice());
}

#[test]
fn out_of_order_chunks_linearize() {
    let (_d, db, token) = setup("vecd-resumable-2");
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    let total = 1000;
    let body = payload(total);
    let upload = create_upload(&client, &server, "datasets/glove/ooo.bin", total, &token);

    // Send the tail first: the acked offset stays at 0 behind the gap.
    let r = patch(&client, &upload, 500, &body[500..1000], &token);
    assert_eq!(r.status(), 204);
    assert_eq!(r.headers().get("upload-offset").unwrap(), "0");
    assert_eq!(r.headers().get("upload-complete").unwrap(), "?0");

    // HEAD reflects the same resumable offset.
    let h = client.head(&upload).bearer_auth(&token).send().unwrap();
    assert_eq!(h.headers().get("upload-offset").unwrap(), "0");
    assert_eq!(h.headers().get("upload-length").unwrap(), "1000");

    // Fill the head: the prefix jumps 0 → 1000 and finalizes.
    let r = patch(&client, &upload, 0, &body[0..500], &token);
    assert_eq!(r.status(), 204);
    assert_eq!(r.headers().get("upload-offset").unwrap(), "1000");
    assert_eq!(r.headers().get("upload-complete").unwrap(), "?1");

    let got = client.get(server.url("datasets/glove/ooo.bin")).bearer_auth(&token).send().unwrap();
    assert_eq!(got.bytes().unwrap().as_ref(), body.as_slice());
}

#[test]
fn concurrent_sparse_patches_assemble_exactly() {
    let (_d, db, token) = setup("vecd-resumable-3");
    let server = Vecd::start(db);
    let client = Arc::new(reqwest::blocking::Client::new());

    let total = 8000;
    let body = Arc::new(payload(total));
    let upload = create_upload(&client, &server, "datasets/glove/par.bin", total, &token);

    // Sixteen disjoint chunks pushed from sixteen threads, shuffled offsets.
    let chunk = 500;
    let mut order: Vec<usize> = (0..total / chunk).collect();
    order.swap(0, 7);
    order.swap(3, 11);
    order.swap(5, 15);
    let mut handles = Vec::new();
    for i in order {
        let client = client.clone();
        let body = body.clone();
        let upload = upload.clone();
        let token = token.clone();
        handles.push(thread::spawn(move || {
            let off = i * chunk;
            let r = patch(&client, &upload, off, &body[off..off + chunk], &token);
            assert_eq!(r.status(), 204, "concurrent PATCH at {off}");
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // Exactly one PATCH finalized; the object is whole and byte-exact.
    let got = client.get(server.url("datasets/glove/par.bin")).bearer_auth(&token).send().unwrap();
    assert_eq!(got.status(), 200);
    assert_eq!(got.bytes().unwrap().as_ref(), body.as_slice());

    // HEAD on the finalized upload reports completion.
    let h = client.head(&upload).bearer_auth(&token).send().unwrap();
    assert_eq!(h.headers().get("upload-offset").unwrap(), "8000");
    assert_eq!(h.headers().get("upload-complete").unwrap(), "?1");
}

#[test]
fn resume_after_interruption() {
    let (_d, db, token) = setup("vecd-resumable-4");
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    let total = 4096;
    let body = payload(total);
    let upload = create_upload(&client, &server, "datasets/glove/resume.bin", total, &token);

    // Upload the first half, then "lose" the connection.
    let r = patch(&client, &upload, 0, &body[0..2048], &token);
    assert_eq!(r.headers().get("upload-offset").unwrap(), "2048");

    // A fresh client HEADs to learn where to resume.
    let client2 = reqwest::blocking::Client::new();
    let h = client2.head(&upload).bearer_auth(&token).send().unwrap();
    let resume: usize = h.headers().get("upload-offset").unwrap().to_str().unwrap().parse().unwrap();
    assert_eq!(resume, 2048);
    assert_eq!(h.headers().get("upload-complete").unwrap(), "?0");

    // Send only the remainder; it completes.
    let r = patch(&client2, &upload, resume, &body[resume..total], &token);
    assert_eq!(r.headers().get("upload-complete").unwrap(), "?1");

    let got = client2.get(server.url("datasets/glove/resume.bin")).bearer_auth(&token).send().unwrap();
    assert_eq!(got.bytes().unwrap().as_ref(), body.as_slice());
}

#[test]
fn delete_abandons_an_in_progress_upload() {
    let (_d, db, token) = setup("vecd-resumable-5");
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    let total = 1000;
    let body = payload(total);
    let upload = create_upload(&client, &server, "datasets/glove/gone.bin", total, &token);
    patch(&client, &upload, 0, &body[0..400], &token);

    // Abandon it.
    let del = client.delete(&upload).bearer_auth(&token).send().unwrap();
    assert_eq!(del.status(), 204);

    // The upload resource is gone, and no object was committed.
    let after = client.patch(&upload).bearer_auth(&token).header("Upload-Offset", "400").body(vec![0u8; 10]).send().unwrap();
    assert_eq!(after.status(), 404);
    let obj = client.get(server.url("datasets/glove/gone.bin")).bearer_auth(&token).send().unwrap();
    assert_eq!(obj.status(), 404);
}

#[test]
fn unauthorized_patch_is_refused() {
    let (_d, db, token) = setup("vecd-resumable-6");
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    let upload = create_upload(&client, &server, "datasets/glove/private.bin", 100, &token);
    // No token → the private namespace refuses the write.
    let anon = client.patch(&upload).header("Upload-Offset", "0").body(vec![1u8; 100]).send().unwrap();
    assert_eq!(anon.status(), 401);
}

#[test]
fn finalize_honors_if_none_match_star() {
    let (_d, db, token) = setup("vecd-resumable-8");
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    // A pre-existing object at the key.
    let pre = client.put(server.url("datasets/glove/cas.bin")).bearer_auth(&token).body(vec![1u8; 50]).send().unwrap();
    assert_eq!(pre.status(), 201);

    // An upload created with If-None-Match:* must fail CAS at finalize.
    let resp = client
        .post(server.url("datasets/glove/cas.bin"))
        .bearer_auth(&token)
        .header("Upload-Length", "50")
        .header("If-None-Match", "*")
        .send()
        .unwrap();
    assert_eq!(resp.status(), 201);
    let upload = server.url(resp.headers().get("location").unwrap().to_str().unwrap().trim_start_matches('/'));
    let done = patch(&client, &upload, 0, &vec![2u8; 50], &token);
    assert_eq!(done.status(), 412, "If-None-Match:* must reject an overwrite at finalize");

    // The original bytes are untouched.
    let got = client.get(server.url("datasets/glove/cas.bin")).bearer_auth(&token).send().unwrap();
    assert_eq!(got.bytes().unwrap().as_ref(), vec![1u8; 50].as_slice());
}

#[test]
fn finalize_enforces_quota() {
    // A 100-byte namespace cap; a 200-byte upload is rejected at finalize.
    let (_d, db, token) = setup_quota("vecd-resumable-9", Some("100"));
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    let upload = create_upload(&client, &server, "datasets/glove/toobig.bin", 200, &token);
    let r = patch(&client, &upload, 0, &payload(200), &token);
    assert_eq!(r.status(), 507, "over-quota upload must 507 at finalize");
    assert_eq!(r.headers().get("x-vecd-quota").unwrap(), "100");

    // Nothing was committed.
    let obj = client.get(server.url("datasets/glove/toobig.bin")).bearer_auth(&token).send().unwrap();
    assert_eq!(obj.status(), 404);
}

#[test]
fn oversized_chunk_is_rejected() {
    let (_d, db, token) = setup("vecd-resumable-7");
    let server = Vecd::start(db);
    let client = reqwest::blocking::Client::new();

    let upload = create_upload(&client, &server, "datasets/glove/big.bin", 100, &token);
    // A chunk that would write past the declared Upload-Length is a 400.
    let r = patch(&client, &upload, 50, &vec![7u8; 100], &token);
    assert_eq!(r.status(), 400);
}
