// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

//! A minimal write-capable HTTP object store for integration tests —
//! the write-side counterpart to [`super::testserver::TestServer`]
//! (which is `ServeDir`, GET/HEAD only). It speaks the REST object
//! semantics `vectordata push`'s `HttpsTransport` expects:
//!
//! - `GET` → 200 + body + `ETag`, or 404
//! - `HEAD` → 200 + `Content-Length` + `ETag`, or 404
//! - `PUT` → store; honors `If-None-Match: *` (412 if present) and `If-Match: <etag>` (412 on mismatch); 201 + `ETag`
//! - `DELETE` → remove; 204
//!
//! With `require_token`, any request lacking `Authorization: Bearer
//! <token>` gets 403 — so auth-failure paths are exercised against the
//! real client code. Objects live in memory; the `ETag` is the SHA-256
//! of the content (matching the local transport's etag convention).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::Router;
use sha2::{Digest, Sha256};

#[derive(Clone)]
struct AppState {
    objects: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    token: Option<String>,
    /// Misbehavior knob for failure-injection tests.
    ignore_conditionals: bool,
}

impl AppState {
    fn base() -> Self {
        AppState {
            objects: Arc::new(Mutex::new(HashMap::new())),
            token: None,
            ignore_conditionals: false,
        }
    }
}

fn etag(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

async fn handle(
    State(state): State<AppState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Auth gate (when a token is required).
    if let Some(tok) = &state.token {
        let presented = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());
        if presented != Some(format!("Bearer {tok}").as_str()) {
            return (StatusCode::FORBIDDEN, "missing/invalid bearer token").into_response();
        }
    }

    let key = uri.path().trim_start_matches('/').to_string();
    let mut objects = state.objects.lock().unwrap();

    match method {
        Method::GET => match objects.get(&key) {
            Some(b) => (StatusCode::OK, [(header::ETAG, etag(b))], b.clone()).into_response(),
            None => StatusCode::NOT_FOUND.into_response(),
        },
        Method::HEAD => match objects.get(&key) {
            Some(b) => (
                StatusCode::OK,
                [
                    (header::CONTENT_LENGTH, b.len().to_string()),
                    (header::ETAG, etag(b)),
                ],
            )
                .into_response(),
            None => StatusCode::NOT_FOUND.into_response(),
        },
        Method::PUT => {
            // A non-conforming store ignores conditional headers entirely.
            if !state.ignore_conditionals {
                // If-None-Match: * → only create if absent.
                if headers.contains_key(header::IF_NONE_MATCH) && objects.contains_key(&key) {
                    return StatusCode::PRECONDITION_FAILED.into_response();
                }
                // If-Match: <etag> → only overwrite if the current etag matches.
                if let Some(im) = headers.get(header::IF_MATCH).and_then(|v| v.to_str().ok()) {
                    let want = im.trim_matches('"');
                    let ok = objects.get(&key).map(|b| etag(b)) == Some(want.to_string());
                    if !ok {
                        return StatusCode::PRECONDITION_FAILED.into_response();
                    }
                }
            }
            let tag = etag(&body);
            objects.insert(key, body.to_vec());
            (StatusCode::CREATED, [(header::ETAG, tag)]).into_response()
        }
        Method::DELETE => {
            objects.remove(&key);
            StatusCode::NO_CONTENT.into_response()
        }
        _ => StatusCode::METHOD_NOT_ALLOWED.into_response(),
    }
}

/// A running in-memory HTTP object store.
pub struct ObjectStore {
    port: u16,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl ObjectStore {
    /// Start an open (no-auth, conforming) store.
    pub fn start() -> Result<Self, String> {
        Self::start_with(AppState::base())
    }

    /// Start a store that requires `Authorization: Bearer <token>`.
    pub fn start_with_token(token: &str) -> Result<Self, String> {
        Self::start_with(AppState { token: Some(token.to_string()), ..AppState::base() })
    }

    /// Start a store that silently ignores conditional-write headers
    /// (`If-Match`/`If-None-Match`) — a non-conforming endpoint.
    pub fn start_ignoring_conditionals() -> Result<Self, String> {
        Self::start_with(AppState { ignore_conditionals: true, ..AppState::base() })
    }

    fn start_with(state: AppState) -> Result<Self, String> {
        let (port_tx, port_rx) = std::sync::mpsc::channel();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let app = Router::new().fallback(handle).with_state(state);
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                port_tx.send(listener.local_addr().unwrap().port()).unwrap();
                axum::serve(listener, app)
                    .with_graceful_shutdown(async {
                        let _ = shutdown_rx.await;
                    })
                    .await
                    .unwrap();
            });
        });

        let port = port_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|e| format!("object store failed to start: {e}"))?;
        Ok(ObjectStore { port, shutdown: Some(shutdown_tx), thread: Some(thread) })
    }

    /// Base URL, with trailing slash (a valid `https`-family publish root
    /// for `.publish_url`/`--to`).
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/", self.port)
    }
}

impl Drop for ObjectStore {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}
