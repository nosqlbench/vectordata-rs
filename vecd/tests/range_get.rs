// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! HTTP `Range` support on the object read path, end-to-end against a live
//! `vecd`. The reader/precache client issues parallel ranged `GET`s and
//! verifies each response is exactly the bytes it asked for; before range
//! support landed, `vecd` ignored the `Range` header and returned the whole
//! object with `200`, so every sub-range read failed its length check. These
//! tests pin the wire contract: `Accept-Ranges`, `206` + `Content-Range` for
//! satisfiable ranges, `416` for unsatisfiable ones.
//!
//! Its own test binary so the server lifecycle is self-contained.

use std::sync::mpsc;
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

/// A deterministic 4000-byte object so slices are trivial to verify.
fn blob() -> Vec<u8> {
    (0..4000u32).map(|i| (i % 251) as u8).collect()
}

#[test]
fn range_requests_serve_exact_byte_slices() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::init(&dir.path().join("vecd.db")).unwrap();
    admin::add_backend(&mut db, "store", "mem", "mem:range-get", None, None, None, true).unwrap();
    admin::add_user(&mut db, "alice", Level::User, None, None).unwrap();
    admin::add_namespace(&mut db, "datasets/pub", "alice", Some("store"), true, Listable::Grantees, None, None).unwrap();
    admin::bind(&mut db, "alice", "curate", "datasets/pub").unwrap();
    admin::bind(&mut db, "PUBLIC", "reader", "datasets/pub").unwrap();
    let token = admin::create_token(&mut db, "alice", "key", Some("30d"), None).unwrap().plaintext;

    let server = Vecd::start(db);
    let http = reqwest::blocking::Client::new();
    let url = server.url("datasets/pub/blob.bin");
    let data = blob();

    // Upload the object (streaming PUT), authenticated as alice.
    let put = http
        .put(&url)
        .bearer_auth(&token)
        .body(data.clone())
        .send()
        .expect("PUT blob");
    assert!(put.status().is_success(), "PUT status {}", put.status());

    // Full GET → 200, whole body, and the server advertises range support.
    let full = http.get(&url).send().expect("GET full");
    assert_eq!(full.status().as_u16(), 200);
    assert_eq!(
        full.headers().get(reqwest::header::ACCEPT_RANGES).and_then(|v| v.to_str().ok()),
        Some("bytes"),
        "200 response must advertise Accept-Ranges: bytes"
    );
    assert_eq!(full.bytes().expect("full body").as_ref(), &data[..]);

    // A closed sub-range → 206 + Content-Range, body is exactly that slice.
    let mid = http
        .get(&url)
        .header(reqwest::header::RANGE, "bytes=1000-1999")
        .send()
        .expect("GET mid range");
    assert_eq!(mid.status().as_u16(), 206, "satisfiable range must be 206");
    assert_eq!(
        mid.headers().get(reqwest::header::CONTENT_RANGE).and_then(|v| v.to_str().ok()),
        Some("bytes 1000-1999/4000")
    );
    assert_eq!(mid.bytes().expect("mid body").as_ref(), &data[1000..=1999]);

    // Suffix range → last N bytes.
    let tail = http
        .get(&url)
        .header(reqwest::header::RANGE, "bytes=-100")
        .send()
        .expect("GET suffix range");
    assert_eq!(tail.status().as_u16(), 206);
    assert_eq!(tail.bytes().expect("tail body").as_ref(), &data[3900..4000]);

    // Open-ended range → from offset to EOF.
    let openr = http
        .get(&url)
        .header(reqwest::header::RANGE, "bytes=3990-")
        .send()
        .expect("GET open range");
    assert_eq!(openr.status().as_u16(), 206);
    assert_eq!(openr.bytes().expect("open body").as_ref(), &data[3990..4000]);

    // An unsatisfiable range → 416 with the size in Content-Range.
    let oob = http
        .get(&url)
        .header(reqwest::header::RANGE, "bytes=8000-9000")
        .send()
        .expect("GET out-of-bounds range");
    assert_eq!(oob.status().as_u16(), 416, "out-of-range must be 416");
    assert_eq!(
        oob.headers().get(reqwest::header::CONTENT_RANGE).and_then(|v| v.to_str().ok()),
        Some("bytes */4000")
    );

    // HEAD advertises range support too (so clients can probe before ranging).
    let head = http.head(&url).send().expect("HEAD");
    assert_eq!(head.status().as_u16(), 200);
    assert_eq!(
        head.headers().get(reqwest::header::ACCEPT_RANGES).and_then(|v| v.to_str().ok()),
        Some("bytes")
    );
}
