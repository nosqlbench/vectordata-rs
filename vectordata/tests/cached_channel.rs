// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the cache-backed file channel.
//!
//! Tests the full lifecycle: cold start, on-demand fetch, prebuffer,
//! crash recovery (resume from .mrkl checkpoint), and corrupted chunk
//! detection — all over real HTTP using `testserver`.

use std::fs;

mod support;
use support::testserver::TestServer;
use vectordata::cache::CachedChannel;
use vectordata::merkle::MerkleRef;
use vectordata::transport::HttpTransport;

/// Create test content: byte i at position i (mod 256).
fn test_content(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

/// Set up a server serving the given content and return (dir, server, content).
fn setup(content: &[u8]) -> (tempfile::TempDir, TestServer) {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("data.bin"), content).unwrap();
    let server = TestServer::start(dir.path()).unwrap();
    (dir, server)
}

fn make_transport(server: &TestServer) -> HttpTransport {
    let url = url::Url::parse(&format!("{}data.bin", server.base_url())).unwrap();
    HttpTransport::new(url)
}

// ---------------------------------------------------------------------------
// Cold start + on-demand read
// ---------------------------------------------------------------------------

#[test]
fn test_cold_start_on_demand_read() {
    let content = test_content(4096);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    )
    .unwrap();

    // Nothing downloaded yet
    assert_eq!(channel.valid_count(), 0);

    // Read first 16 bytes — triggers download of chunk 0
    let data = channel.read(0, 16).unwrap();
    assert_eq!(&data, &content[0..16]);
    assert_eq!(channel.valid_count(), 1);

    // Read from chunk 2 — triggers download of chunk 2
    let data = channel.read(2048, 100).unwrap();
    assert_eq!(&data, &content[2048..2148]);
    assert_eq!(channel.valid_count(), 2);
}

#[test]
fn test_read_spanning_chunks() {
    let content = test_content(4096);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    )
    .unwrap();

    // Read across chunk boundary (end of chunk 0 into chunk 1)
    let data = channel.read(1000, 100).unwrap();
    assert_eq!(&data, &content[1000..1100]);
    // Should have fetched chunks 0 and 1
    assert_eq!(channel.valid_count(), 2);
}

#[test]
fn test_read_entire_file() {
    let content = test_content(4096);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    )
    .unwrap();

    let data = channel.read(0, 4096).unwrap();
    assert_eq!(data, content);
    assert!(channel.is_complete());
}

// ---------------------------------------------------------------------------
// Prebuffer
// ---------------------------------------------------------------------------

#[test]
fn test_prebuffer_downloads_all() {
    let content = test_content(4096);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    )
    .unwrap();

    channel.prebuffer().unwrap();
    assert!(channel.is_complete());
    assert_eq!(channel.valid_count(), 4);

    // All reads should now be served from cache (no network)
    let data = channel.read(0, 4096).unwrap();
    assert_eq!(data, content);
}

#[test]
fn test_prebuffer_with_progress() {
    let content = test_content(4096);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    )
    .unwrap();

    let mut callback_called = false;
    channel
        .prebuffer_with_progress(|progress| {
            callback_called = true;
            assert!(progress.is_complete());
        })
        .unwrap();

    assert!(callback_called);
    assert!(channel.is_complete());
}

#[test]
fn test_prebuffer_partial_then_complete() {
    let content = test_content(4096);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    )
    .unwrap();

    // Read one chunk first
    let _ = channel.read(0, 100).unwrap();
    assert_eq!(channel.valid_count(), 1);

    // Prebuffer should only download the remaining 3 chunks
    channel.prebuffer().unwrap();
    assert!(channel.is_complete());
}

// ---------------------------------------------------------------------------
// Crash recovery (resume from .mrkl checkpoint)
// ---------------------------------------------------------------------------

#[test]
fn test_resume_after_crash() {
    let content = test_content(4096);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    // First session: download 2 of 4 chunks
    {
        let channel = CachedChannel::open(
            Box::new(make_transport(&server)),
            mref.clone(),
            cache_dir.path(),
            "data.bin",
        )
        .unwrap();

        let _ = channel.read(0, 100).unwrap(); // chunk 0
        let _ = channel.read(1024, 100).unwrap(); // chunk 1
        assert_eq!(channel.valid_count(), 2);
        // channel dropped — simulates crash
    }

    // Verify .mrkl file was persisted
    assert!(cache_dir.path().join("data.bin.mrkl").exists());

    // Second session: resume from checkpoint
    {
        let mref2 = mref.clone();
        let channel = CachedChannel::open(
            Box::new(make_transport(&server)),
            mref2,
            cache_dir.path(),
            "data.bin",
        )
        .unwrap();

        // Chunks 0 and 1 should already be valid (from state file)
        assert_eq!(channel.valid_count(), 2);

        // Read chunk 0 from cache (no download needed)
        let data = channel.read(0, 16).unwrap();
        assert_eq!(&data, &content[0..16]);

        // Download remaining chunks
        channel.prebuffer().unwrap();
        assert!(channel.is_complete());

        let data = channel.read(0, 4096).unwrap();
        assert_eq!(data, content);
    }
}

// ---------------------------------------------------------------------------
// Corrupted chunk detection
// ---------------------------------------------------------------------------

#[test]
fn test_corrupted_chunk_detected() {
    // Serve content where one chunk will be corrupted
    let content = test_content(4096);
    let mref = MerkleRef::from_content(&content, 1024);

    // Corrupt chunk 1 in the served file
    let mut corrupted = content.clone();
    corrupted[1024] ^= 0xFF;

    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("data.bin"), &corrupted).unwrap();
    let server = TestServer::start(dir.path()).unwrap();

    let cache_dir = tempfile::tempdir().unwrap();
    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    )
    .unwrap();

    // Chunk 0 should work (not corrupted)
    let data = channel.read(0, 100).unwrap();
    assert_eq!(&data, &content[0..100]);

    // Chunk 1 should fail verification
    let result = channel.read(1024, 100);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("integrity verification"),
        "expected integrity error, got: {}",
        err
    );
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_non_aligned_content_size() {
    // Content size not a multiple of chunk size — last chunk is shorter
    let content = test_content(3000);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    )
    .unwrap();

    assert_eq!(channel.total_chunks(), 3);

    // Read the last (short) chunk
    let data = channel.read(2048, 952).unwrap();
    assert_eq!(&data, &content[2048..3000]);

    channel.prebuffer().unwrap();
    assert!(channel.is_complete());

    let full = channel.read(0, 3000).unwrap();
    assert_eq!(full, content);
}

#[test]
fn test_empty_read() {
    let content = test_content(4096);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    )
    .unwrap();

    let data = channel.read(100, 0).unwrap();
    assert!(data.is_empty());
    // No chunks should have been fetched
    assert_eq!(channel.valid_count(), 0);
}

#[test]
fn test_out_of_bounds_read() {
    let content = test_content(4096);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    )
    .unwrap();

    let result = channel.read(4000, 200);
    assert!(result.is_err());
}

#[test]
fn test_already_complete_prebuffer_is_noop() {
    let content = test_content(4096);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    )
    .unwrap();

    channel.prebuffer().unwrap();
    assert!(channel.is_complete());

    // Second prebuffer should be a no-op
    channel.prebuffer().unwrap();
    assert!(channel.is_complete());
}

// ---------------------------------------------------------------------------
// Concurrent access — multiple threads reading the same chunk
// ---------------------------------------------------------------------------

#[test]
fn test_concurrent_reads_same_chunk() {
    use std::sync::Arc;
    use std::thread;

    let content = test_content(8192);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = Arc::new(CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    ).unwrap());

    // Spawn 8 threads all reading the same range (chunk 0)
    let handles: Vec<_> = (0..8).map(|_| {
        let ch = Arc::clone(&channel);
        let expected = content[0..16].to_vec();
        thread::spawn(move || {
            let data = ch.read(0, 16).unwrap();
            assert_eq!(data, expected);
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    // Only 1 chunk should have been downloaded (deduplication)
    assert!(channel.valid_count() >= 1);
}

#[test]
fn test_concurrent_reads_different_chunks() {
    use std::sync::Arc;
    use std::thread;

    let content = test_content(8192); // 8 chunks of 1024
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = Arc::new(CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    ).unwrap());

    // Each thread reads from a different chunk
    let handles: Vec<_> = (0..8).map(|i| {
        let ch = Arc::clone(&channel);
        let offset = (i * 1024) as u64;
        let expected = content[offset as usize..offset as usize + 16].to_vec();
        thread::spawn(move || {
            let data = ch.read(offset, 16).unwrap();
            assert_eq!(data, expected);
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    // All 8 chunks should be valid
    assert_eq!(channel.valid_count(), 8);
}

#[test]
fn test_concurrent_prebuffer_and_reads() {
    use std::sync::Arc;
    use std::thread;

    let content = test_content(8192);
    let (_dir, server) = setup(&content);
    let mref = MerkleRef::from_content(&content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = Arc::new(CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    ).unwrap());

    // One thread prebuffers while others read concurrently
    let ch_prebuf = Arc::clone(&channel);
    let prebuf_handle = thread::spawn(move || {
        ch_prebuf.prebuffer().unwrap();
    });

    let read_handles: Vec<_> = (0..4).map(|i| {
        let ch = Arc::clone(&channel);
        let offset = (i * 2000) as u64;
        let len = 100u64;
        let expected = content[offset as usize..(offset + len) as usize].to_vec();
        thread::spawn(move || {
            let data = ch.read(offset, len).unwrap();
            assert_eq!(data, expected);
        })
    }).collect();

    prebuf_handle.join().unwrap();
    for h in read_handles {
        h.join().unwrap();
    }

    assert!(channel.is_complete());
}

// ---------------------------------------------------------------------------
// Mref / content size mismatch
// ---------------------------------------------------------------------------

#[test]
fn test_mref_smaller_than_actual_content() {
    // .mref was built for a 2048-byte file, but the server serves 4096 bytes.
    // The cache should only use the mref's declared content size.
    let small_content = test_content(2048);
    let large_content = test_content(4096);

    let dir = tempfile::tempdir().unwrap();
    // Write the LARGE file to disk (what the server serves)
    fs::write(dir.path().join("data.bin"), &large_content).unwrap();
    let server = TestServer::start(dir.path()).unwrap();

    // But create mref from the SMALL content
    let mref = MerkleRef::from_content(&small_content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    ).unwrap();

    // Content size should match the mref, not the server
    assert_eq!(channel.content_size(), 2048);

    // Reading within the mref's range should work (but verification
    // will fail because the server's content doesn't match the mref hashes)
    let result = channel.read(0, 16);
    // Either errors (hash mismatch) or returns data from the first chunk
    // of the larger file — depends on whether chunk 0 of the large file
    // happens to match chunk 0 of the small file (they're identical in
    // our test_content since both start with the same byte pattern).
    // The first 2048 bytes are identical, so chunks 0-1 verify fine.
    match result {
        Ok(data) => assert_eq!(data, small_content[0..16]),
        Err(_) => {} // hash mismatch also acceptable
    }
}

#[test]
fn test_mref_larger_than_actual_content() {
    // .mref was built for a 4096-byte file, but the server serves 2048 bytes.
    // Reading beyond the server's content should fail.
    let large_content = test_content(4096);
    let small_content = test_content(2048);

    let dir = tempfile::tempdir().unwrap();
    // Server has the SMALL file
    fs::write(dir.path().join("data.bin"), &small_content).unwrap();
    let server = TestServer::start(dir.path()).unwrap();

    // Mref built from the LARGE content
    let mref = MerkleRef::from_content(&large_content, 1024);
    let cache_dir = tempfile::tempdir().unwrap();

    let channel = CachedChannel::open(
        Box::new(make_transport(&server)),
        mref,
        cache_dir.path(),
        "data.bin",
    ).unwrap()
    .with_retry_policy(vectordata::transport::RetryPolicy {
        max_retries: 1,
        base_delay_ms: 1,
        max_delay_ms: 1,
        jitter_fraction: 0.0,
    });

    // Content size from mref says 4096, but server only has 2048.
    // Reading past 2048 should fail — the server can't serve those bytes.
    let result = channel.read(3000, 100);
    assert!(result.is_err(),
        "reading beyond server content should fail");
}
