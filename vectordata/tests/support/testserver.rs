// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Lightweight HTTP file server for integration tests.
//!
//! Provides a [`TestServer`] that serves files from a directory with full
//! HTTP Range request support. Designed for testing HTTP-based data access
//! layers (e.g., `HttpVectorReader`, merkle tree verification) against
//! real HTTP semantics without external infrastructure.

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
    #[allow(dead_code)]
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
        let request = match server.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(Some(req)) => req,
            Ok(None) => continue,
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

    let relative = url_path.trim_start_matches('/');
    if relative.contains("..") {
        let _ = request.respond(tiny_http::Response::empty(403));
        return;
    }

    let file_path = root.join(relative);

    let metadata = match fs::metadata(&file_path) {
        Ok(m) if m.is_file() => m,
        _ => {
            let _ = request.respond(tiny_http::Response::empty(404));
            return;
        }
    };

    let file_size = metadata.len();

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

    let range_header = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Range"))
        .map(|h| h.value.as_str().to_string());

    if let Some(range_str) = range_header {
        if let Some(range) = parse_range(&range_str, file_size) {
            serve_range(request, &file_path, file_size, range.0, range.1);
        } else {
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
        serve_full(request, &file_path, file_size);
    }
}

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
