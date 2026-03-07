// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Lightweight HTTP file server for integration tests.
//!
//! Provides a [`TestServer`] that serves files from a directory with full
//! HTTP Range request support. Designed for testing HTTP-based data access
//! layers (e.g., `HttpVectorReader`, merkle tree verification) against
//! real HTTP semantics without external infrastructure.
//!
//! # Usage
//!
//! ```no_run
//! use testserver::TestServer;
//! use std::path::Path;
//!
//! // Start serving files from a directory
//! let server = TestServer::start(Path::new("/path/to/test/data")).unwrap();
//!
//! // Get the base URL for constructing file URLs
//! let base = server.base_url(); // e.g., "http://127.0.0.1:45123/"
//!
//! // Access files via HTTP — supports Range requests
//! let file_url = format!("{}data.fvec", base);
//!
//! // Server stops when dropped
//! drop(server);
//! ```

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

/// A lightweight HTTP server that serves files from a directory.
///
/// Supports:
/// - `GET` with full file response
/// - `GET` with `Range: bytes=start-end` header (206 Partial Content)
/// - `HEAD` requests (returns Content-Length without body)
/// - Proper `Accept-Ranges: bytes` header
/// - `Content-Length` on all responses
///
/// The server binds to `127.0.0.1:0` (OS-assigned port) and runs on a
/// background thread. It stops when dropped.
pub struct TestServer {
    port: u16,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl TestServer {
    /// Start serving files from the given directory.
    ///
    /// Binds to `127.0.0.1` on a random available port. Returns immediately
    /// once the server is ready to accept connections.
    pub fn start(root: &Path) -> Result<Self, String> {
        let root = root
            .canonicalize()
            .map_err(|e| format!("cannot resolve root path: {}", e))?;

        let server = tiny_http::Server::http("127.0.0.1:0")
            .map_err(|e| format!("failed to bind: {}", e))?;

        let port = server
            .server_addr()
            .to_ip()
            .map(|a| a.port())
            .ok_or_else(|| "failed to get server port".to_string())?;

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);

        let thread = thread::Builder::new()
            .name("testserver".into())
            .spawn(move || {
                serve_loop(server, root, stop_clone);
            })
            .map_err(|e| format!("failed to spawn server thread: {}", e))?;

        Ok(TestServer {
            port,
            stop,
            thread: Some(thread),
        })
    }

    /// The base URL of the server, e.g., `"http://127.0.0.1:12345/"`.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/", self.port)
    }

    /// The port the server is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Send a dummy request to unblock the accept loop
        let _ = std::net::TcpStream::connect(format!("127.0.0.1:{}", self.port));
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

/// Main server loop — handles requests until `stop` is set.
fn serve_loop(server: tiny_http::Server, root: PathBuf, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::SeqCst) {
        // Use a timeout so we can check the stop flag periodically
        let request = match server.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(Some(req)) => req,
            Ok(None) => continue, // timeout, check stop flag
            Err(_) => break,
        };

        if stop.load(Ordering::SeqCst) {
            let _ = request.respond(tiny_http::Response::empty(503));
            break;
        }

        handle_request(request, &root);
    }
}

/// Handle a single HTTP request.
fn handle_request(request: tiny_http::Request, root: &Path) {
    let url_path = request.url().to_string();

    // Strip leading slash and decode path
    let relative = url_path.trim_start_matches('/');
    if relative.contains("..") {
        let _ = request.respond(tiny_http::Response::empty(403));
        return;
    }

    let file_path = root.join(relative);

    // Check file exists
    let metadata = match fs::metadata(&file_path) {
        Ok(m) if m.is_file() => m,
        _ => {
            let _ = request.respond(tiny_http::Response::empty(404));
            return;
        }
    };

    let file_size = metadata.len();

    // HEAD request — return metadata only
    if request.method() == &tiny_http::Method::Head {
        let response = tiny_http::Response::empty(200)
            .with_header(
                tiny_http::Header::from_bytes("Content-Length", file_size.to_string())
                    .unwrap(),
            )
            .with_header(
                tiny_http::Header::from_bytes("Accept-Ranges", "bytes").unwrap(),
            );
        let _ = request.respond(response);
        return;
    }

    // Check for Range header
    let range_header = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Range"))
        .map(|h| h.value.as_str().to_string());

    if let Some(range_str) = range_header {
        // Parse "bytes=start-end"
        if let Some(range) = parse_range(&range_str, file_size) {
            serve_range(request, &file_path, file_size, range.0, range.1);
        } else {
            // Invalid range
            let response = tiny_http::Response::empty(416).with_header(
                tiny_http::Header::from_bytes(
                    "Content-Range",
                    format!("bytes */{}", file_size),
                )
                .unwrap(),
            );
            let _ = request.respond(response);
        }
    } else {
        // Full file response
        serve_full(request, &file_path, file_size);
    }
}

/// Parse a Range header value like "bytes=0-3" or "bytes=100-199".
///
/// Returns `(start, end)` inclusive byte offsets, or None if invalid.
fn parse_range(header: &str, file_size: u64) -> Option<(u64, u64)> {
    let bytes_prefix = header.strip_prefix("bytes=")?;
    let (start_str, end_str) = bytes_prefix.split_once('-')?;

    let start: u64 = start_str.parse().ok()?;
    let end: u64 = if end_str.is_empty() {
        file_size - 1
    } else {
        end_str.parse().ok()?
    };

    if start > end || start >= file_size {
        return None;
    }

    let end = std::cmp::min(end, file_size - 1);
    Some((start, end))
}

/// Serve a byte range from a file (206 Partial Content).
fn serve_range(
    request: tiny_http::Request,
    path: &Path,
    file_size: u64,
    start: u64,
    end: u64,
) {
    let length = end - start + 1;

    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => {
            let _ = request.respond(tiny_http::Response::empty(500));
            return;
        }
    };

    use std::io::Seek;
    if file.seek(std::io::SeekFrom::Start(start)).is_err() {
        let _ = request.respond(tiny_http::Response::empty(500));
        return;
    }

    let mut data = vec![0u8; length as usize];
    if file.read_exact(&mut data).is_err() {
        let _ = request.respond(tiny_http::Response::empty(500));
        return;
    }

    let cursor = std::io::Cursor::new(data);
    let response = tiny_http::Response::new(
        tiny_http::StatusCode(206),
        vec![
            tiny_http::Header::from_bytes("Content-Length", length.to_string()).unwrap(),
            tiny_http::Header::from_bytes(
                "Content-Range",
                format!("bytes {}-{}/{}", start, end, file_size),
            )
            .unwrap(),
            tiny_http::Header::from_bytes("Accept-Ranges", "bytes").unwrap(),
        ],
        cursor,
        Some(length as usize),
        None,
    );
    let _ = request.respond(response);
}

/// Serve a full file (200 OK).
fn serve_full(request: tiny_http::Request, path: &Path, file_size: u64) {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => {
            let _ = request.respond(tiny_http::Response::empty(500));
            return;
        }
    };

    let response = tiny_http::Response::from_file(file)
        .with_header(
            tiny_http::Header::from_bytes("Accept-Ranges", "bytes").unwrap(),
        )
        .with_header(
            tiny_http::Header::from_bytes("Content-Length", file_size.to_string()).unwrap(),
        );
    let _ = request.respond(response);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn setup_test_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();

        // Create a small fvec file: 2 vectors of dimension 3
        // Each record: [dim:i32 LE][v0:f32 LE][v1:f32 LE][v2:f32 LE]
        let mut file = fs::File::create(dir.path().join("test.fvec")).unwrap();
        // Vector 0: [1.0, 2.0, 3.0]
        file.write_all(&3i32.to_le_bytes()).unwrap();
        file.write_all(&1.0f32.to_le_bytes()).unwrap();
        file.write_all(&2.0f32.to_le_bytes()).unwrap();
        file.write_all(&3.0f32.to_le_bytes()).unwrap();
        // Vector 1: [4.0, 5.0, 6.0]
        file.write_all(&3i32.to_le_bytes()).unwrap();
        file.write_all(&4.0f32.to_le_bytes()).unwrap();
        file.write_all(&5.0f32.to_le_bytes()).unwrap();
        file.write_all(&6.0f32.to_le_bytes()).unwrap();

        // Create a dataset.yaml
        fs::write(
            dir.path().join("dataset.yaml"),
            "name: test\nprofiles:\n  default:\n    base_vectors: test.fvec\n",
        )
        .unwrap();

        dir
    }

    #[test]
    fn test_server_starts_and_stops() {
        let dir = setup_test_dir();
        let server = TestServer::start(dir.path()).unwrap();
        assert!(server.port() > 0);
        let url = server.base_url();
        assert!(url.starts_with("http://127.0.0.1:"));
        drop(server);
    }

    #[test]
    fn test_full_file_get() {
        let dir = setup_test_dir();
        let server = TestServer::start(dir.path()).unwrap();

        let client = reqwest::blocking::Client::new();
        let resp = client
            .get(format!("{}test.fvec", server.base_url()))
            .send()
            .unwrap();

        assert_eq!(resp.status(), 200);
        let body = resp.bytes().unwrap();
        // 2 vectors * (4 + 3*4) = 2 * 16 = 32 bytes
        assert_eq!(body.len(), 32);
    }

    #[test]
    fn test_head_request() {
        let dir = setup_test_dir();
        let server = TestServer::start(dir.path()).unwrap();

        let client = reqwest::blocking::Client::new();
        let resp = client
            .head(format!("{}test.fvec", server.base_url()))
            .send()
            .unwrap();

        assert_eq!(resp.status(), 200);
        let content_length: u64 = resp
            .headers()
            .get("content-length")
            .unwrap()
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(content_length, 32);
    }

    #[test]
    fn test_range_request() {
        let dir = setup_test_dir();
        let server = TestServer::start(dir.path()).unwrap();

        let client = reqwest::blocking::Client::new();

        // Read just the dimension header (first 4 bytes)
        let resp = client
            .get(format!("{}test.fvec", server.base_url()))
            .header("Range", "bytes=0-3")
            .send()
            .unwrap();

        assert_eq!(resp.status(), 206);
        let body = resp.bytes().unwrap();
        assert_eq!(body.len(), 4);
        // Dimension should be 3
        let dim = i32::from_le_bytes([body[0], body[1], body[2], body[3]]);
        assert_eq!(dim, 3);
    }

    #[test]
    fn test_range_request_second_vector() {
        let dir = setup_test_dir();
        let server = TestServer::start(dir.path()).unwrap();

        let client = reqwest::blocking::Client::new();

        // Read the second vector (bytes 16-31)
        let resp = client
            .get(format!("{}test.fvec", server.base_url()))
            .header("Range", "bytes=16-31")
            .send()
            .unwrap();

        assert_eq!(resp.status(), 206);
        let body = resp.bytes().unwrap();
        assert_eq!(body.len(), 16);

        // Parse dimension
        let dim = i32::from_le_bytes([body[0], body[1], body[2], body[3]]);
        assert_eq!(dim, 3);

        // Parse first element of second vector (should be 4.0)
        let val = f32::from_le_bytes([body[4], body[5], body[6], body[7]]);
        assert_eq!(val, 4.0);
    }

    #[test]
    fn test_404_for_missing_file() {
        let dir = setup_test_dir();
        let server = TestServer::start(dir.path()).unwrap();

        let client = reqwest::blocking::Client::new();
        let resp = client
            .get(format!("{}nonexistent.fvec", server.base_url()))
            .send()
            .unwrap();

        assert_eq!(resp.status(), 404);
    }

    #[test]
    fn test_path_traversal_blocked() {
        let dir = setup_test_dir();
        let server = TestServer::start(dir.path()).unwrap();

        // reqwest normalizes ".." out of URLs before sending, so the server
        // sees a resolved path. The server's ".." check catches raw requests
        // from other clients. Here we just verify the file isn't reachable.
        let client = reqwest::blocking::Client::new();
        let resp = client
            .get(format!("{}..%2Fetc%2Fpasswd", server.base_url()))
            .send()
            .unwrap();

        // Should be 403 (contains "..") or 404 (not found) — either is safe
        let status = resp.status().as_u16();
        assert!(
            status == 403 || status == 404,
            "expected 403 or 404, got {}",
            status
        );
    }

    #[test]
    fn test_dataset_yaml_served() {
        let dir = setup_test_dir();
        let server = TestServer::start(dir.path()).unwrap();

        let client = reqwest::blocking::Client::new();
        let resp = client
            .get(format!("{}dataset.yaml", server.base_url()))
            .send()
            .unwrap();

        assert_eq!(resp.status(), 200);
        let body = resp.text().unwrap();
        assert!(body.contains("name: test"));
    }
}
