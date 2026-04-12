// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! HTTP file server for integration tests.
//!
//! Uses axum + tower-http ServeDir for correct static file serving
//! with HEAD, GET, and Range request support.

use std::net::SocketAddr;
use std::path::Path;

/// An HTTP server that serves files from a directory.
///
/// Uses axum with tower-http's `ServeDir` for standards-compliant
/// static file serving including Range requests and HEAD.
pub struct TestServer {
    port: u16,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl TestServer {
    /// Start serving files from the given directory.
    pub fn start(root: &Path) -> Result<Self, String> {
        let root = root.to_path_buf();
        let (port_tx, port_rx) = std::sync::mpsc::channel();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async {
                let service = tower_http::services::ServeDir::new(&root);
                let app = axum::Router::new()
                    .fallback_service(service);

                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .unwrap();
                let addr = listener.local_addr().unwrap();
                port_tx.send(addr.port()).unwrap();

                axum::serve(listener, app)
                    .with_graceful_shutdown(async { let _ = shutdown_rx.await; })
                    .await
                    .unwrap();
            });
        });

        let port = port_rx.recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|e| format!("server failed to start: {}", e))?;

        Ok(TestServer {
            port,
            shutdown: Some(shutdown_tx),
            thread: Some(thread),
        })
    }

    /// The base URL of the server.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/", self.port)
    }

    #[allow(dead_code)]
    pub fn port(&self) -> u16 {
        self.port
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
