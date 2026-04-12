// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the chunked transport layer.
//!
//! Uses `testserver` to serve test data locally and verifies that
//! `HttpTransport`, retry logic, and parallel downloads work correctly
//! over real HTTP.

use std::fs;
use std::io::Write;
use std::path::Path;

mod support;
use support::testserver::TestServer;
use vectordata::transport::{
    ChunkRequest, ChunkedTransport, DownloadProgress, HttpTransport, RetryPolicy,
};

/// Create a test file with known content: byte i at position i (mod 256).
fn write_test_file(path: &Path, size: usize) {
    let mut file = fs::File::create(path).unwrap();
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    file.write_all(&data).unwrap();
}

fn setup_transport() -> (tempfile::TempDir, TestServer, HttpTransport) {
    let dir = tempfile::tempdir().unwrap();
    write_test_file(&dir.path().join("data.bin"), 4096);

    let server = TestServer::start(dir.path()).unwrap();
    let url = url::Url::parse(&format!("{}data.bin", server.base_url())).unwrap();
    let transport = HttpTransport::new(url);

    (dir, server, transport)
}

#[test]
fn test_http_transport_content_length() {
    let (_dir, _server, transport) = setup_transport();
    assert_eq!(transport.content_length().unwrap(), 4096);
}

#[test]
fn test_http_transport_supports_range() {
    let (_dir, _server, transport) = setup_transport();
    assert!(transport.supports_range());
}

#[test]
fn test_http_transport_fetch_range() {
    let (_dir, _server, transport) = setup_transport();

    let data = transport.fetch_range(0, 16).unwrap();
    assert_eq!(data.len(), 16);
    // Verify content: byte i at position i
    for i in 0..16 {
        assert_eq!(data[i], i as u8);
    }
}

#[test]
fn test_http_transport_fetch_range_middle() {
    let (_dir, _server, transport) = setup_transport();

    let data = transport.fetch_range(100, 50).unwrap();
    assert_eq!(data.len(), 50);
    for i in 0..50 {
        assert_eq!(data[i], ((100 + i) % 256) as u8);
    }
}

#[test]
fn test_http_transport_fetch_range_end() {
    let (_dir, _server, transport) = setup_transport();

    // Last 32 bytes
    let data = transport.fetch_range(4064, 32).unwrap();
    assert_eq!(data.len(), 32);
    for i in 0..32 {
        assert_eq!(data[i], ((4064 + i) % 256) as u8);
    }
}

#[test]
fn test_parallel_chunk_download() {
    let (_dir, _server, transport) = setup_transport();
    let policy = RetryPolicy::default();
    let progress = DownloadProgress::new(4096, 4);

    let chunks: Vec<ChunkRequest> = (0..4)
        .map(|i| ChunkRequest {
            index: i,
            start: i as u64 * 1024,
            len: 1024,
        })
        .collect();

    let results =
        vectordata::transport::fetch_chunks_parallel(&transport, &chunks, &policy, &progress, 4);

    assert_eq!(results.len(), 4);
    for result in &results {
        let (idx, data) = result.as_ref().unwrap();
        assert_eq!(data.len(), 1024);
        // Verify first byte of each chunk
        let expected_first = ((*idx as usize) * 1024 % 256) as u8;
        assert_eq!(data[0], expected_first);
    }

    assert!(progress.is_complete());
    assert_eq!(progress.downloaded_bytes(), 4096);
    assert!(!progress.is_failed());
}

#[test]
fn test_parallel_download_with_concurrency_limit() {
    let (_dir, _server, transport) = setup_transport();
    let policy = RetryPolicy::default();
    let progress = DownloadProgress::new(4096, 16);

    // 16 smaller chunks, concurrency 2
    let chunks: Vec<ChunkRequest> = (0..16)
        .map(|i| ChunkRequest {
            index: i,
            start: i as u64 * 256,
            len: 256,
        })
        .collect();

    let results =
        vectordata::transport::fetch_chunks_parallel(&transport, &chunks, &policy, &progress, 2);

    assert_eq!(results.len(), 16);
    assert!(results.iter().all(|r| r.is_ok()));
    assert!(progress.is_complete());
}

#[test]
fn test_merkle_verified_download() {
    // End-to-end: create content, build merkle ref, download chunks via
    // HTTP, verify each chunk against the merkle tree.
    use vectordata::merkle::{MerkleRef, MerkleState};

    let dir = tempfile::tempdir().unwrap();
    let content: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    fs::write(dir.path().join("data.bin"), &content).unwrap();

    let server = TestServer::start(dir.path()).unwrap();
    let url = url::Url::parse(&format!("{}data.bin", server.base_url())).unwrap();
    let transport = HttpTransport::new(url);

    let mref = MerkleRef::from_content(&content, 1024);
    let mut state = MerkleState::from_ref(&mref);
    let policy = RetryPolicy::default();
    let progress = DownloadProgress::new(4096, 4);

    let chunks: Vec<ChunkRequest> = (0..4)
        .map(|i| ChunkRequest {
            index: i,
            start: i as u64 * 1024,
            len: 1024,
        })
        .collect();

    let results =
        vectordata::transport::fetch_chunks_parallel(&transport, &chunks, &policy, &progress, 4);

    for result in results {
        let (idx, data) = result.unwrap();
        assert!(
            state.verify_and_mark(idx, &data),
            "chunk {} failed verification",
            idx
        );
    }

    assert!(state.is_complete());
}
