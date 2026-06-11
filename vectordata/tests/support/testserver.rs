// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

//! HTTP file server for integration tests.
//!
//! Uses axum + tower-http ServeDir for correct static file serving
//! with HEAD, GET, and Range request support. Counts accepted TCP
//! connections (`accepted_connections()`) so tests can assert that
//! a client is reusing a pooled connection across many requests
//! instead of opening a fresh socket each time.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// An HTTP server that serves files from a directory.
///
/// Uses axum with tower-http's `ServeDir` for standards-compliant
/// static file serving including Range requests and HEAD.
pub struct TestServer {
    port: u16,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
    accepted: Arc<AtomicU64>,
}

impl TestServer {
    /// Start serving files from the given directory.
    pub fn start(root: &Path) -> Result<Self, String> {
        Self::start_with_ranges(root, true)
    }

    /// Start a server that IGNORES `Range` headers: every GET returns
    /// the whole body with `200 OK` and no `Accept-Ranges` header.
    /// Models servers without byte-range support so tests can prove
    /// the storage layer's FullTransfer fallback.
    pub fn start_no_range(root: &Path) -> Result<Self, String> {
        Self::start_with_ranges(root, false)
    }

    fn start_with_ranges(root: &Path, ranges: bool) -> Result<Self, String> {
        let root = root.to_path_buf();
        let (port_tx, port_rx) = std::sync::mpsc::channel();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let accepted = Arc::new(AtomicU64::new(0));
        let accepted_for_thread = Arc::clone(&accepted);

        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async {
                let app = if ranges {
                    let service = tower_http::services::ServeDir::new(&root);
                    axum::Router::new().fallback_service(service)
                } else {
                    // Whole-body-only handler: no Accept-Ranges, no
                    // 206 — a Range header on the request is ignored.
                    let root = root.clone();
                    axum::Router::new().fallback(move |
                        method: axum::http::Method,
                        uri: axum::http::Uri,
                    | {
                        let root = root.clone();
                        async move {
                            let rel = uri.path().trim_start_matches('/');
                            let path = root.join(rel);
                            let Ok(bytes) = tokio::fs::read(&path).await else {
                                return axum::http::Response::builder()
                                    .status(404)
                                    .body(axum::body::Body::empty())
                                    .unwrap();
                            };
                            let len = bytes.len();
                            let body = if method == axum::http::Method::HEAD {
                                axum::body::Body::empty()
                            } else {
                                axum::body::Body::from(bytes)
                            };
                            axum::http::Response::builder()
                                .status(200)
                                .header(axum::http::header::CONTENT_LENGTH, len)
                                .body(body)
                                .unwrap()
                        }
                    })
                };

                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .unwrap();
                let addr = listener.local_addr().unwrap();
                port_tx.send(addr.port()).unwrap();

                // Custom accept loop so we can count incoming TCP
                // connections. axum::serve hides the accept layer,
                // so we drive it manually with hyper-util's
                // TokioExecutor + auto::Builder. Each accepted
                // socket bumps `accepted_for_thread`; one HTTP/1.1
                // keep-alive connection serving N requests counts
                // as 1 — exactly the signal the pooling test needs.
                use hyper_util::rt::{TokioExecutor, TokioIo};
                use hyper_util::server::conn::auto;
                use tower::Service;

                let mut shutdown_rx = shutdown_rx;
                loop {
                    tokio::select! {
                        _ = &mut shutdown_rx => break,
                        accept = listener.accept() => {
                            let (stream, _peer) = match accept {
                                Ok(s) => s,
                                Err(_) => continue,
                            };
                            accepted_for_thread.fetch_add(1, Ordering::Relaxed);
                            let app = app.clone();
                            tokio::spawn(async move {
                                let io = TokioIo::new(stream);
                                let svc = hyper::service::service_fn(move |req| {
                                    let mut app = app.clone();
                                    async move {
                                        Ok::<_, std::convert::Infallible>(
                                            app.call(req).await.unwrap()
                                        )
                                    }
                                });
                                let _ = auto::Builder::new(TokioExecutor::new())
                                    .serve_connection(io, svc)
                                    .await;
                            });
                        }
                    }
                }
            });
        });

        let port = port_rx.recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|e| format!("server failed to start: {}", e))?;

        Ok(TestServer {
            port,
            shutdown: Some(shutdown_tx),
            thread: Some(thread),
            accepted,
        })
    }

    /// The base URL of the server.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/", self.port)
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// Total number of TCP connections accepted since the server
    /// started. A reused HTTP/1.1 keep-alive connection counts as
    /// 1 regardless of how many requests it served — so tests can
    /// assert connection-pool reuse by comparing this count to the
    /// number of requests they made.
    pub fn accepted_connections(&self) -> u64 {
        self.accepted.load(Ordering::Relaxed)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}
