// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the `https`/`http` push transport and a full
//! `push::execute` over HTTP, driven against the in-repo write-capable
//! object-store mock (`support::objectstore`). These cover the transport
//! verbs (HEAD/GET/PUT/DELETE), the conditional-put CAS guard over the
//! wire, and — the mode that can't be reached with `file://` — bearer
//! auth failure and success.

mod support;

use support::objectstore::ObjectStore;

use vectordata::push::binding::parse_publish_url;
use vectordata::push::transport::{open, PushError, TransportOptions};
use vectordata::push::{execute, ChecksumPolicy, Options};

fn opts(src: &std::path::Path, base_url: String, token: Option<String>) -> Options {
    Options {
        path: src.to_path_buf(),
        to: Some(base_url),
        message: None,
        raw: false,
        checksums: ChecksumPolicy::Auto,
        dry_run: false,
        no_check: false,
        assume_yes: true,
        delete: false,
        abort_incomplete: false,
        concurrency: 4,
        files: None,
        transport: TransportOptions { token, profile: None, endpoint_url: None },
        cmd: "vectordata push (https test)".into(),
        actor: "tester@host".into(),
    }
}

fn make_dataset(dir: &std::path::Path) {
    std::fs::write(
        dir.join("dataset.yaml"),
        "name: httpds\nattributes:\n  is_zero_vector_free: true\n  is_duplicate_vector_free: true\n",
    )
    .unwrap();
    std::fs::write(dir.join("base.fvec"), b"HTTPBASE").unwrap();
}

#[test]
fn https_transport_verbs_and_conditional_put() {
    let store = ObjectStore::start().unwrap();
    let binding = parse_publish_url(&store.base_url()).unwrap();
    let tx = open(&binding, &TransportOptions::default()).unwrap();

    // Absent → None.
    assert!(tx.head("x").unwrap().is_none());
    assert!(tx.get("x").unwrap().is_none());

    // PUT with must-not-exist precondition: ok while absent, 412 once present.
    tx.put_bytes("x", b"hello", Some("")).unwrap();
    assert!(matches!(tx.put_bytes("x", b"again", Some("")), Err(PushError::PreconditionFailed)));

    // HEAD/GET reflect it, with an ETag.
    let obj = tx.head("x").unwrap().unwrap();
    assert_eq!(obj.size, 5);
    let etag = obj.etag.expect("server returns an ETag");
    assert_eq!(tx.get("x").unwrap().unwrap(), b"hello");

    // Conditional overwrite: correct etag succeeds, stale etag is refused.
    tx.put_bytes("x", b"world!", Some(&etag)).unwrap();
    assert!(matches!(tx.put_bytes("x", b"z", Some(&etag)), Err(PushError::PreconditionFailed)));

    // put_file streams a file body.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("f"), b"FILEBODY").unwrap();
    tx.put_file("dir/f", &tmp.path().join("f")).unwrap();
    assert_eq!(tx.get("dir/f").unwrap().unwrap(), b"FILEBODY");

    // DELETE removes it.
    tx.delete("dir/f").unwrap();
    assert!(tx.head("dir/f").unwrap().is_none());

    // Generic https has no object listing → --delete is unsupported, loudly.
    assert!(tx.list("").is_err());
}

#[test]
fn push_over_https_end_to_end() {
    let store = ObjectStore::start().unwrap();
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());

    let outcome = execute(&opts(src.path(), store.base_url(), None)).expect("https push");
    assert_eq!(outcome.version, 1);

    // Everything is retrievable back over the wire.
    let binding = parse_publish_url(&store.base_url()).unwrap();
    let tx = open(&binding, &TransportOptions::default()).unwrap();
    assert_eq!(tx.get("base.fvec").unwrap().unwrap(), b"HTTPBASE");
    assert!(tx.get("SHA256SUMS").unwrap().is_some());
    assert!(tx.get(".publish_url").unwrap().is_some());
    let log = tx.get("pushlog.jsonl").unwrap().unwrap();
    let log = vectordata::push::pushlog::Log::parse(&String::from_utf8_lossy(&log)).unwrap();
    assert_eq!(log.stable_version(), Some(1));
}

#[test]
fn https_auth_failure_then_success_with_token() {
    let store = ObjectStore::start_with_token("s3kr3t").unwrap();
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());

    // No token → the preflight HEAD is rejected (403) → operational auth error.
    let err = execute(&opts(src.path(), store.base_url(), None)).expect_err("no token");
    match err {
        vectordata::push::Failure::Operational(m) => {
            assert!(m.to_lowercase().contains("auth"), "{m}");
        }
        other => panic!("expected operational auth failure, got {other:?}"),
    }

    // Correct token → succeeds.
    let outcome = execute(&opts(src.path(), store.base_url(), Some("s3kr3t".into())))
        .expect("authed https push");
    assert_eq!(outcome.version, 1);
}

#[test]
fn push_refuses_endpoint_that_ignores_conditional_writes() {
    // The single-provenance guarantee depends on conditional writes; a
    // store that ignores them must be refused at the preflight probe.
    let store = ObjectStore::start_ignoring_conditionals().unwrap();
    let src = tempfile::tempdir().unwrap();
    make_dataset(src.path());
    let err = execute(&opts(src.path(), store.base_url(), None))
        .expect_err("non-conforming store must be refused");
    match err {
        vectordata::push::Failure::Operational(m) => {
            assert!(m.contains("conditional writes"), "{m}");
        }
        other => panic!("expected operational refusal, got {other:?}"),
    }
}

