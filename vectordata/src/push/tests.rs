// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the push engine, exercised against the
//! `file://` transport (the fully-local, fully-deterministic one). The
//! engine itself is transport-agnostic, so these cover the binding,
//! checksum, provenance, overwrite-gate, and convergence logic that the
//! `https`/`s3` transports also ride.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("vd-push-test-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A minimal known-good structured dataset: required publishable
/// attributes plus a couple of content files at two directory levels.
fn make_dataset(dir: &Path) {
    std::fs::write(
        dir.join("dataset.yaml"),
        "name: testds\n\
         attributes:\n  \
           is_zero_vector_free: true\n  \
           is_duplicate_vector_free: true\n",
    )
    .unwrap();
    std::fs::write(dir.join("base.fvec"), b"BASEDATA").unwrap();
    std::fs::create_dir_all(dir.join("profiles/1m")).unwrap();
    std::fs::write(dir.join("profiles/1m/neighbor.ivec"), b"NEIGHBORS").unwrap();
}

fn file_url(dir: &Path) -> String {
    format!("file://{}/", dir.display())
}

fn opts(src: &Path, remote: &Path) -> Options {
    Options {
        path: src.to_path_buf(),
        to: Some(file_url(remote)),
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
        transport: TransportOptions::default(),
        cmd: "vectordata push (test)".to_string(),
        actor: "tester@host".to_string(),
    }
}

fn remote_log(remote: &Path) -> Log {
    Log::parse(&std::fs::read_to_string(remote.join(pushlog::PUSHLOG_FILE)).unwrap()).unwrap()
}

#[test]
fn first_push_creates_all_artifacts() {
    let src = unique("first-src");
    let remote = unique("first-remote");
    make_dataset(&src);

    let outcome = execute(&opts(&src, &remote)).expect("push should succeed");
    assert_eq!(outcome.version, 1);
    assert_eq!(outcome.overwritten, 0);
    assert!(outcome.added >= 2, "added {}", outcome.added);
    assert_eq!(outcome.skipped, 0);

    // Content copied.
    assert_eq!(std::fs::read(remote.join("base.fvec")).unwrap(), b"BASEDATA");
    assert_eq!(
        std::fs::read(remote.join("profiles/1m/neighbor.ivec")).unwrap(),
        b"NEIGHBORS"
    );
    // Per-directory SHA256SUMS present at both levels.
    assert!(remote.join("SHA256SUMS").is_file());
    assert!(remote.join("profiles/1m/SHA256SUMS").is_file());
    // Binding written remotely and mirrored locally.
    assert!(remote.join(".publish_url").is_file());
    assert!(src.join(".publish_url").is_file());
    // Provenance: begin + complete for seq 1, stable at 1.
    let rl = remote_log(&remote);
    assert_eq!(rl.stable_version(), Some(1));
    assert_eq!(rl.events.len(), 2);
    assert!(rl.events[0].is_begin());
    // Local pushlog mirrors the remote.
    assert_eq!(remote_log(&remote), super::read_local_log(&src).unwrap());
}

#[test]
fn repush_unchanged_skips_everything() {
    let src = unique("repush-src");
    let remote = unique("repush-remote");
    make_dataset(&src);
    execute(&opts(&src, &remote)).unwrap();

    let outcome = execute(&opts(&src, &remote)).expect("idempotent re-push");
    assert_eq!(outcome.added, 0);
    assert_eq!(outcome.overwritten, 0);
    assert!(outcome.skipped >= 2);
    assert_eq!(outcome.version, 2, "version advances even for an additive/no-op push");
    assert_eq!(remote_log(&remote).stable_version(), Some(2));
}

#[test]
fn overwrite_without_message_is_blocked() {
    let src = unique("ow-block-src");
    let remote = unique("ow-block-remote");
    make_dataset(&src);
    execute(&opts(&src, &remote)).unwrap();

    // Change existing content; checksums auto-refresh will pick it up.
    std::fs::write(src.join("base.fvec"), b"CHANGEDDATA").unwrap();
    let err = execute(&opts(&src, &remote)).expect_err("overwrite must be gated");
    match err {
        Failure::Usage(m) => assert!(m.contains("overwrite"), "{m}"),
        other => panic!("expected Usage, got {other:?}"),
    }
    // The remote data must be untouched (no begin written either).
    assert_eq!(std::fs::read(remote.join("base.fvec")).unwrap(), b"BASEDATA");
    assert_eq!(remote_log(&remote).stable_version(), Some(1));
}

#[test]
fn overwrite_with_message_succeeds_and_logs() {
    let src = unique("ow-ok-src");
    let remote = unique("ow-ok-remote");
    make_dataset(&src);
    execute(&opts(&src, &remote)).unwrap();

    std::fs::write(src.join("base.fvec"), b"CHANGEDDATA").unwrap();
    let mut o = opts(&src, &remote);
    o.message = Some("regen base after dedup".to_string());
    let outcome = execute(&o).expect("gated overwrite with -m");
    assert_eq!(outcome.overwritten, 1);
    assert_eq!(outcome.version, 2);
    assert_eq!(std::fs::read(remote.join("base.fvec")).unwrap(), b"CHANGEDDATA");

    // The begin event carries the message and the overwrite.
    let rl = remote_log(&remote);
    let begin = rl.events.iter().find(|e| e.is_begin() && e.seq() == 2).unwrap();
    match begin {
        Event::Begin { message, overwrites, .. } => {
            assert_eq!(message.as_deref(), Some("regen base after dedup"));
            assert!(overwrites.iter().any(|o| o.key == "base.fvec"));
        }
        _ => unreachable!(),
    }
}

#[test]
fn dry_run_writes_nothing() {
    let src = unique("dry-src");
    let remote = unique("dry-remote");
    make_dataset(&src);

    let mut o = opts(&src, &remote);
    o.dry_run = true;
    let outcome = execute(&o).expect("dry run ok");
    assert!(outcome.dry_run);
    assert!(outcome.added >= 2);
    // Nothing written to the remote, and no SHA256SUMS minted locally.
    assert!(!remote.join("base.fvec").exists());
    assert!(!remote.join(pushlog::PUSHLOG_FILE).exists());
    assert!(!src.join("SHA256SUMS").exists());
}

#[test]
fn binding_conflict_is_refused() {
    let src = unique("conflict-src");
    let remote = unique("conflict-remote");
    make_dataset(&src);
    // Pre-bind the source somewhere else.
    std::fs::write(src.join(".publish_url"), "file:///somewhere/else/\n").unwrap();

    let err = execute(&opts(&src, &remote)).expect_err("binding conflict");
    assert!(matches!(err, Failure::Usage(m) if m.contains("conflict")));
}

#[test]
fn divergent_provenance_is_refused() {
    let src = unique("diverge-src");
    let remote = unique("diverge-remote");
    make_dataset(&src);
    execute(&opts(&src, &remote)).unwrap();

    // Simulate someone else pushing: remote log gains seq 2 that the
    // local copy never saw.
    let mut rl = remote_log(&remote);
    rl.events.push(Event::Begin {
        seq: 2,
        ts: "Mon, 01 Jan 2024 00:00:00 GMT".into(),
        actor: "other@host".into(),
        cmd: "vectordata push".into(),
        message: None,
        overwrites: vec![],
        added: vec![],
        deletes: vec![],
        sums: std::collections::BTreeMap::new(),
        tool_version: "x".into(),
    });
    rl.events.push(Event::Complete {
        seq: 2,
        ts: "Mon, 01 Jan 2024 00:00:01 GMT".into(),
        sums: std::collections::BTreeMap::new(),
    });
    std::fs::write(remote.join(pushlog::PUSHLOG_FILE), rl.render()).unwrap();

    let err = execute(&opts(&src, &remote)).expect_err("remote ahead → divergent");
    assert!(matches!(err, Failure::Usage(m) if m.contains("divergent")));
}

#[test]
fn open_update_tombstone_blocks() {
    let src = unique("open-src");
    let remote = unique("open-remote");
    make_dataset(&src);
    execute(&opts(&src, &remote)).unwrap();

    // Append an unmatched begin to BOTH logs (so it's not a divergence,
    // just an open/crashed update both sides agree happened).
    let extra_begin = Event::Begin {
        seq: 2,
        ts: "Mon, 01 Jan 2024 00:00:00 GMT".into(),
        actor: "crashed@host".into(),
        cmd: "vectordata push".into(),
        message: None,
        overwrites: vec![],
        added: vec![],
        deletes: vec![],
        sums: std::collections::BTreeMap::new(),
        tool_version: "x".into(),
    };
    for log_path in [remote.join(pushlog::PUSHLOG_FILE), src.join(pushlog::PUSHLOG_FILE)] {
        let mut l = Log::parse(&std::fs::read_to_string(&log_path).unwrap()).unwrap();
        l.events.push(extra_begin.clone());
        std::fs::write(&log_path, l.render()).unwrap();
    }

    // The open begin's (empty) sums don't match the current content, and
    // --abort-incomplete was not given → refuse, pointing at resume/abort.
    let err = execute(&opts(&src, &remote)).expect_err("open update blocks");
    assert!(matches!(err, Failure::Usage(m) if m.contains("incomplete push")));
}

#[test]
fn raw_mode_pushes_arbitrary_files() {
    let src = unique("raw-src");
    let remote = unique("raw-remote");
    std::fs::write(src.join("notes.txt"), b"hello").unwrap();
    std::fs::write(src.join("model.bin"), b"\x00\x01\x02").unwrap();

    let mut o = opts(&src, &remote);
    o.raw = true;
    let outcome = execute(&o).expect("raw push ok");
    assert_eq!(outcome.mode, SourceMode::Raw);
    assert_eq!(std::fs::read(remote.join("notes.txt")).unwrap(), b"hello");
    assert!(remote.join("SHA256SUMS").is_file());
}

#[test]
fn delete_removes_orphans_gated_by_message() {
    let src = unique("del-src");
    let remote = unique("del-remote");
    make_dataset(&src);
    execute(&opts(&src, &remote)).unwrap();

    // Drop a whole profile locally (file + its minted SHA256SUMS) so the
    // remote copies become orphans.
    std::fs::remove_dir_all(src.join("profiles/1m")).unwrap();

    // --delete without -m is gated (destructive).
    let mut o = opts(&src, &remote);
    o.delete = true;
    let err = execute(&o).expect_err("delete must be gated");
    assert!(matches!(err, Failure::Usage(m) if m.contains("delete")));
    // Orphan still present.
    assert!(remote.join("profiles/1m/neighbor.ivec").is_file());

    // With -m, the orphan (and its now-empty dir's SHA256SUMS) are removed.
    o.message = Some("drop 1m profile".to_string());
    let outcome = execute(&o).expect("gated delete with -m");
    assert!(outcome.deleted >= 1, "deleted {}", outcome.deleted);
    assert!(!remote.join("profiles/1m/neighbor.ivec").exists());
    // The surviving root content is untouched.
    assert_eq!(std::fs::read(remote.join("base.fvec")).unwrap(), b"BASEDATA");
    // The deletion is recorded on the begin event for audit.
    let rl = remote_log(&remote);
    let begin = rl.events.iter().rev().find(|e| e.is_begin()).unwrap();
    match begin {
        Event::Begin { deletes, .. } => {
            assert!(deletes.iter().any(|d| d.contains("neighbor.ivec")));
        }
        _ => unreachable!(),
    }
}

#[test]
fn injected_set_publishes_only_selected_files_as_one_hierarchy() {
    // Mirrors how `veks publish` delegates: a publish root holding two
    // datasets plus a root catalog and some junk the producer excludes.
    let root = unique("hier-src");
    let remote = unique("hier-remote");
    std::fs::create_dir_all(root.join("ds-a")).unwrap();
    std::fs::create_dir_all(root.join("ds-b")).unwrap();
    std::fs::write(root.join("catalog.json"), b"{}").unwrap();
    std::fs::write(root.join("ds-a/base.fvec"), b"A").unwrap();
    std::fs::write(root.join("ds-b/base.fvec"), b"B").unwrap();
    std::fs::write(root.join("ds-a/.publish"), b"").unwrap(); // producer marker, excluded
    std::fs::write(root.join("README.md"), b"not published").unwrap(); // excluded

    // The producer-selected set (relative, forward-slashed) — note the
    // catalog and both datasets, but NOT README or the .publish marker.
    let selected = vec![
        "catalog.json".to_string(),
        "ds-a/base.fvec".to_string(),
        "ds-b/base.fvec".to_string(),
    ];

    let mut o = opts(&root, &remote);
    o.files = Some(selected);
    let outcome = execute(&o).expect("hierarchy push ok");
    assert_eq!(outcome.mode, SourceMode::Hierarchy);
    assert_eq!(outcome.version, 1);

    // Exactly the selected files shipped — README did not.
    assert!(remote.join("catalog.json").is_file());
    assert!(remote.join("ds-a/base.fvec").is_file());
    assert!(remote.join("ds-b/base.fvec").is_file());
    assert!(!remote.join("README.md").exists());
    // One root pushlog covers the whole hierarchy; per-dir SHA256SUMS.
    assert_eq!(remote_log(&remote).stable_version(), Some(1));
    assert!(remote.join("SHA256SUMS").is_file());
    assert!(remote.join("ds-a/SHA256SUMS").is_file());
    // The root SHA256SUMS lists the catalog but not README.
    let root_sums = std::fs::read_to_string(remote.join("SHA256SUMS")).unwrap();
    assert!(root_sums.contains("catalog.json"));
    assert!(!root_sums.contains("README.md"));
}

/// Turn a completed remote (…begin(N), complete(N)) into a crashed one
/// (…begin(N) open), and roll the local log back to before N — exactly
/// the on-disk state a push that died after `begin` but before
/// `complete` leaves behind.
fn simulate_crash(remote: &Path, src: &Path) {
    let mut rl = remote_log(remote);
    rl.events.pop(); // drop the trailing complete → open begin remains
    std::fs::write(remote.join(pushlog::PUSHLOG_FILE), rl.render()).unwrap();
    // The crash never mirrored the new version locally.
    std::fs::write(src.join(pushlog::PUSHLOG_FILE), rl.committed().render()).unwrap();
}

#[test]
fn resume_finishes_an_interrupted_push() {
    let src = unique("resume-src");
    let remote = unique("resume-remote");
    make_dataset(&src);
    execute(&opts(&src, &remote)).unwrap(); // version 1

    // A second, additive version: a new file.
    std::fs::write(src.join("extra.fvec"), b"EXTRA").unwrap();
    execute(&opts(&src, &remote)).unwrap(); // version 2 completes…
    assert_eq!(remote_log(&remote).stable_version(), Some(2));

    // …now rewrite history to look like version 2 crashed mid-upload:
    // open begin(2) remains, local rolled back, and one object plus the
    // root checksum file never made it up.
    simulate_crash(&remote, &src);
    std::fs::remove_file(remote.join("extra.fvec")).unwrap();
    std::fs::remove_file(remote.join("SHA256SUMS")).unwrap();
    assert!(remote_log(&remote).open_update().is_some(), "precondition: open begin");

    // Re-running the same push (content unchanged) resumes seq 2 — no -m,
    // no new version — and idempotently restores the missing object.
    let outcome = execute(&opts(&src, &remote)).expect("resume should finish seq 2");
    assert!(outcome.resumed, "should have resumed");
    assert_eq!(outcome.version, 2, "resume keeps the same seq, not a new one");
    assert_eq!(std::fs::read(remote.join("extra.fvec")).unwrap(), b"EXTRA");
    let rl = remote_log(&remote);
    assert_eq!(rl.stable_version(), Some(2), "seq 2 is now complete");
    assert!(rl.open_update().is_none(), "no open update remains");
    // Local mirrors the finished remote.
    assert_eq!(rl, super::read_local_log(&src).unwrap());
}

#[test]
fn open_push_with_changed_source_refuses_then_aborts() {
    let src = unique("abort-src");
    let remote = unique("abort-remote");
    make_dataset(&src);
    execute(&opts(&src, &remote)).unwrap(); // version 1
    std::fs::write(src.join("extra.fvec"), b"EXTRA").unwrap();
    execute(&opts(&src, &remote)).unwrap(); // version 2
    simulate_crash(&remote, &src); // open begin(2)

    // The source changes after the crash → current intent no longer
    // matches the open begin's recorded content fingerprint.
    std::fs::write(src.join("extra.fvec"), b"DIFFERENT-NOW").unwrap();

    // Default: refuse, pointing at both recovery routes.
    let err = execute(&opts(&src, &remote)).expect_err("mismatched open push must refuse");
    assert!(matches!(&err, Failure::Usage(m)
        if m.contains("incomplete push") && m.contains("--abort-incomplete")), "{err:?}");

    // With --abort-incomplete: abandon seq 2, push current state as seq 3.
    // The new content overwrites the crashed v2's bytes, so -m is required
    // (the abort path is a fresh push and the gate still applies).
    let mut o = opts(&src, &remote);
    o.abort_incomplete = true;
    o.message = Some("supersede the abandoned push".to_string());
    let outcome = execute(&o).expect("abort + fresh push");
    assert!(!outcome.resumed);
    assert_eq!(outcome.version, 3, "fresh seq after the aborted one");
    assert_eq!(std::fs::read(remote.join("extra.fvec")).unwrap(), b"DIFFERENT-NOW");
    let rl = remote_log(&remote);
    assert_eq!(rl.stable_version(), Some(3));
    // The aborted seq is recorded between the old begin and the new push.
    assert!(rl.events.iter().any(|e| matches!(e, Event::Abort { seq: 2, .. })));
}

// ─── binding failure modes ──────────────────────────────────────────

#[test]
fn no_destination_is_refused() {
    let src = unique("nodest-src");
    make_dataset(&src);
    let mut o = opts(&src, Path::new("/unused"));
    o.to = None; // and no .publish_url in src
    let err = execute(&o).expect_err("no .publish_url and no --to");
    assert!(matches!(err, Failure::Usage(m) if m.contains("no destination")));
}

#[test]
fn existing_binding_is_used_without_to() {
    let src = unique("bind-src");
    let remote = unique("bind-remote");
    make_dataset(&src);
    std::fs::write(src.join(".publish_url"), format!("{}\n", file_url(&remote))).unwrap();
    let mut o = opts(&src, &remote);
    o.to = None; // rely solely on the persisted binding
    let outcome = execute(&o).expect("bound via .publish_url");
    assert_eq!(outcome.version, 1);
    assert!(remote.join("base.fvec").is_file());
}

#[test]
fn malformed_publish_url_is_a_hard_stop() {
    let src = unique("bad-url-src");
    make_dataset(&src);
    std::fs::write(src.join(".publish_url"), "ftp://nope/path\n").unwrap();
    let mut o = opts(&src, Path::new("/unused"));
    o.to = None;
    let err = execute(&o).expect_err("unsupported scheme");
    assert!(matches!(err, Failure::Usage(m) if m.contains("unsupported transport")));
}

#[test]
fn remote_bound_elsewhere_is_a_conflict() {
    let src = unique("rconf-src");
    let remote = unique("rconf-remote");
    make_dataset(&src);
    // The remote root is already bound to a different endpoint.
    std::fs::create_dir_all(&remote).unwrap();
    std::fs::write(remote.join(".publish_url"), "file:///some/other/place/\n").unwrap();
    let err = execute(&opts(&src, &remote)).expect_err("remote bound elsewhere");
    assert!(matches!(err, Failure::Usage(m) if m.contains("remote conflict")));
}

// ─── source-mode / validation failure modes ─────────────────────────

#[test]
fn no_recognized_mode_without_raw_is_refused() {
    let src = unique("nomode-src");
    let remote = unique("nomode-remote");
    std::fs::write(src.join("loose.bin"), b"x").unwrap(); // no dataset.yaml/knn_entries
    let err = execute(&opts(&src, &remote)).expect_err("no mode, no --raw");
    assert!(matches!(err, Failure::Usage(m) if m.contains("--raw")));
}

#[test]
fn structured_missing_attributes_is_refused_but_no_check_bypasses() {
    let src = unique("attrs-src");
    let remote = unique("attrs-remote");
    std::fs::write(src.join("dataset.yaml"), "name: x\n").unwrap(); // no required attrs
    std::fs::write(src.join("base.fvec"), b"B").unwrap();

    let err = execute(&opts(&src, &remote)).expect_err("missing publishable attrs");
    assert!(matches!(err, Failure::Usage(m) if m.contains("is_zero_vector_free")));

    // --no-check bypasses known-good validation (binding/provenance still apply).
    let mut o = opts(&src, &remote);
    o.no_check = true;
    assert_eq!(execute(&o).expect("no_check bypass").version, 1);
}

#[test]
fn catalog_mode_end_to_end_and_missing_file_is_refused() {
    let src = unique("knn-src");
    let remote = unique("knn-remote");
    std::fs::write(
        src.join("knn_entries.yaml"),
        "\"ds:default\":\n  base: base.fvec\n  query: query.fvec\n  gt: gt.ivec\n",
    )
    .unwrap();
    std::fs::write(src.join("base.fvec"), b"B").unwrap();
    std::fs::write(src.join("query.fvec"), b"Q").unwrap();

    // gt.ivec referenced but absent → refuse.
    let err = execute(&opts(&src, &remote)).expect_err("missing gt");
    assert!(matches!(err, Failure::Usage(m) if m.contains("missing")));

    // Provide it → catalog push succeeds.
    std::fs::write(src.join("gt.ivec"), b"G").unwrap();
    let outcome = execute(&opts(&src, &remote)).expect("catalog push");
    assert_eq!(outcome.mode, SourceMode::Catalog);
    assert!(remote.join("gt.ivec").is_file());
}

// ─── convergence / provenance failure modes ─────────────────────────

#[test]
fn forked_provenance_is_refused() {
    let src = unique("fork-src");
    let remote = unique("fork-remote");
    make_dataset(&src);
    execute(&opts(&src, &remote)).unwrap(); // v1: both [b1,c1]
    std::fs::write(src.join("extra.fvec"), b"E").unwrap();
    execute(&opts(&src, &remote)).unwrap(); // v2: both [b1,c1,b2,c2]

    // Rewrite the remote's v2 to a *different* push that shares only the
    // [b1,c1] prefix → neither log is a prefix of the other.
    let mut rl = remote_log(&remote);
    rl.events.truncate(2); // keep real [b1,c1]
    rl.events.push(Event::Begin {
        seq: 2, ts: "Mon, 01 Jan 2024 00:00:00 GMT".into(), actor: "other@host".into(),
        cmd: "x".into(), message: None, overwrites: vec![], added: vec![], deletes: vec![],
        sums: std::collections::BTreeMap::new(), tool_version: "x".into(),
    });
    rl.events.push(Event::Complete {
        seq: 2, ts: "Mon, 01 Jan 2024 00:00:01 GMT".into(),
        sums: std::collections::BTreeMap::new(),
    });
    std::fs::write(remote.join(pushlog::PUSHLOG_FILE), rl.render()).unwrap();

    let err = execute(&opts(&src, &remote)).expect_err("forked histories");
    assert!(matches!(err, Failure::Usage(m) if m.contains("forked")));
}

#[test]
fn local_ahead_carries_its_tail_up() {
    let src = unique("la-src");
    let remote = unique("la-remote");
    make_dataset(&src);
    execute(&opts(&src, &remote)).unwrap(); // v1
    std::fs::write(src.join("extra.fvec"), b"E").unwrap();
    execute(&opts(&src, &remote)).unwrap(); // v2: both [b1,c1,b2,c2]

    // Roll the REMOTE pushlog back to [b1,c1] (remote fell behind the
    // local provenance); local stays at v2.
    let mut rl = remote_log(&remote);
    rl.events.truncate(2);
    std::fs::write(remote.join(pushlog::PUSHLOG_FILE), rl.render()).unwrap();

    // A v3 push proceeds and carries the local-only b2/c2 tail back up.
    std::fs::write(src.join("more.fvec"), b"M").unwrap();
    let outcome = execute(&opts(&src, &remote)).expect("local-ahead push");
    assert_eq!(outcome.version, 3);
    let rl = remote_log(&remote);
    assert_eq!(rl.stable_version(), Some(3));
    assert!(rl.events.iter().any(|e| matches!(e, Event::Complete { seq: 2, .. })),
        "the local-only v2 tail was carried up");
}

#[test]
fn crash_after_complete_before_mirror_fast_forwards() {
    let src = unique("ff-src");
    let remote = unique("ff-remote");
    make_dataset(&src);
    execute(&opts(&src, &remote)).unwrap(); // v1 completes on remote + local

    // Simulate: the remote `complete` landed but the local mirror never
    // did → local pushlog is missing entirely. Content is unchanged.
    std::fs::remove_file(src.join(pushlog::PUSHLOG_FILE)).unwrap();

    let outcome = execute(&opts(&src, &remote)).expect("should fast-forward, not wedge");
    assert!(!outcome.resumed);
    assert_eq!(outcome.version, 1, "recognizes we're already at the remote head");
    assert_eq!(outcome.added, 0);
    // Local provenance was fast-forwarded to match the remote.
    assert_eq!(remote_log(&remote), super::read_local_log(&src).unwrap());
}

// ─── transport compare-and-swap (the pushlog concurrency guard) ──────

#[test]
fn transport_conditional_put_enforces_if_match() {
    use super::transport::{open, PushError, TransportOptions};
    let dir = unique("cas");
    let binding = binding::parse_publish_url(&file_url(&dir)).unwrap();
    let tx = open(&binding, &TransportOptions::default()).unwrap();

    // must-not-exist precondition: succeeds while absent…
    tx.put_bytes("log", b"v1", Some("")).unwrap();
    // …and fails once it exists.
    assert!(matches!(tx.put_bytes("log", b"v2", Some("")), Err(PushError::PreconditionFailed)));

    let etag = tx.head("log").unwrap().unwrap().etag.unwrap();
    // correct etag → ok; this changes the content (and thus the etag).
    tx.put_bytes("log", b"v2", Some(&etag)).unwrap();
    // the old etag is now stale → precondition fails (no silent clobber).
    assert!(matches!(tx.put_bytes("log", b"v3", Some(&etag)), Err(PushError::PreconditionFailed)));
}

/// End-to-end proof of the tombstone-on-failure property using a *real*
/// injected fault (a read-only remote subdir), rather than a manufactured
/// crash state: a push that dies mid-upload must leave an open `begin`
/// (because begin is written before any object), and a rerun must resume
/// it to completion once the fault clears.
#[cfg(unix)]
#[test]
fn real_midupload_failure_leaves_resumable_tombstone() {
    use std::os::unix::fs::PermissionsExt;

    let src = unique("fault-src");
    let remote = unique("fault-remote");
    // v1: a root file and a file in subdir p/.
    std::fs::write(
        src.join("dataset.yaml"),
        "name: f\nattributes:\n  is_zero_vector_free: true\n  is_duplicate_vector_free: true\n",
    )
    .unwrap();
    std::fs::write(src.join("base.fvec"), b"BASE").unwrap();
    std::fs::create_dir_all(src.join("p")).unwrap();
    std::fs::write(src.join("p/a.fvec"), b"AAA").unwrap();
    execute(&opts(&src, &remote)).unwrap(); // v1 completes

    // v2 adds p/b.fvec. Make the remote's p/ read-only so its upload
    // fails — but the root (pushlog, binding) stays writable so begin(2)
    // is recorded.
    std::fs::write(src.join("p/b.fvec"), b"BBB").unwrap();
    let mut perms = std::fs::metadata(remote.join("p")).unwrap().permissions();
    perms.set_mode(0o555);
    std::fs::set_permissions(remote.join("p"), perms).unwrap();

    let err = execute(&opts(&src, &remote)).expect_err("upload into read-only dir must fail");
    assert!(matches!(err, Failure::Operational(_)), "{err:?}");
    // The interrupted push left an open begin(2) on the remote…
    let rl = remote_log(&remote);
    assert_eq!(rl.open_update().map(|(s, _, _)| s), Some(2), "tombstone present");
    assert!(!remote.join("p/b.fvec").exists(), "object did not land");

    // Clear the fault and re-run: it resumes seq 2 and finishes.
    let mut perms = std::fs::metadata(remote.join("p")).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(remote.join("p"), perms).unwrap();

    let outcome = execute(&opts(&src, &remote)).expect("resume after fault cleared");
    assert!(outcome.resumed);
    assert_eq!(outcome.version, 2);
    assert_eq!(std::fs::read(remote.join("p/b.fvec")).unwrap(), b"BBB");
    assert_eq!(remote_log(&remote).stable_version(), Some(2));
    assert!(remote_log(&remote).open_update().is_none());
}

#[test]
fn keep_policy_refuses_stale_checksums() {
    let src = unique("keep-src");
    let remote = unique("keep-remote");
    make_dataset(&src);
    // No SHA256SUMS exists yet → under keep that's a hard stop.
    let mut o = opts(&src, &remote);
    o.checksums = ChecksumPolicy::Keep;
    let err = execute(&o).expect_err("keep with missing sums");
    assert!(matches!(err, Failure::Usage(m) if m.contains("SHA256SUMS")));
}
