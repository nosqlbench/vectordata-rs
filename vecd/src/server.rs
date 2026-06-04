// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The HTTP server — the thin `axum`/`tokio` shell over the synchronous
//! core. Per-request pipeline (`docs/design/vecd-daemon.md` § *HTTP server
//! & request pipeline*):
//!
//! ```text
//! authenticate (token → principal) → authorize (action × key vs the cone)
//!   → handler (resolve backend; honor If-Match/If-None-Match; compute ETag)
//!   → access_log
//! ```
//!
//! Authn/authz happen against the in-memory [`Snapshot`] (no DB on the hot
//! path); only the object op and the access-log append touch the
//! `Mutex<Db>`. The snapshot is **live-reloaded**: a background tick polls
//! the cheap `PRAGMA data_version` and, when `auth_generation` has
//! advanced, atomically swaps in a freshly built snapshot — so an admin's
//! `vecd bind …` / `vecd ns set …` takes effect within ~1s, no restart.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, Path, State};
use axum::http::{header, HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use futures_util::StreamExt;

use crate::auth;
use crate::authz::{Caller, Snapshot};
use crate::backend::Backend;
use crate::db::Db;
use crate::model::{Action, VecdError, PRIV_IGNORE_QUOTAS};
use crate::namespace::{self, Resolved};
use crate::session;
use crate::store::{self, Precondition, PutResult};
use crate::upload;

use vectordata::push::pushlog::{Event, Log, PUSHLOG_FILE};

/// Upper bound on a *buffered* request body (the small pushlog control file
/// that must be parsed, plus empty GET/HEAD/DELETE bodies). Object content
/// never takes this path — it streams (see [`stream_put`]).
const MAX_BUFFERED_BODY: usize = 256 * 1024 * 1024;

/// Bytes accumulated before each [`Backend::put_at`] while streaming a body
/// to staging — bounds resident memory and amortizes per-chunk backend I/O.
const UPLOAD_FLUSH_BYTES: usize = 1 << 20;

/// Lightweight Prometheus-style counters.
#[derive(Default)]
pub struct Metrics {
    pub requests: AtomicU64,
    pub auth_failures: AtomicU64,
    pub denials: AtomicU64,
    pub throttled: AtomicU64,
}

/// Per-source auth-failure tracking for abuse throttling.
struct RateEntry {
    failures: u32,
    blocked_until: i64,
}

/// After this many consecutive auth failures from one source, block it.
const RATE_FAIL_THRESHOLD: u32 = 8;
/// Base block window (seconds), grown with the failure count.
const RATE_BLOCK_BASE_SECS: i64 = 5;

/// Shared server state. Cheap to clone (everything behind an `Arc`).
#[derive(Clone)]
pub struct AppState {
    /// The CAS authority + access log. A single connection behind a mutex
    /// — SQLite is single-writer anyway, and the hot authz path never
    /// touches it.
    pub db: Arc<Mutex<Db>>,
    /// The live-reloadable control-plane snapshot.
    pub snapshot: Arc<RwLock<Arc<Snapshot>>>,
    /// Request/auth counters for `/metrics`.
    pub metrics: Arc<Metrics>,
    /// Per-source auth-failure throttle state.
    rate: Arc<Mutex<HashMap<String, RateEntry>>>,
}

impl AppState {
    /// Build state from an opened DB, loading the initial snapshot.
    pub fn new(db: Db) -> Result<Self, VecdError> {
        let snap = Snapshot::build(&db.load_control_plane()?);
        Ok(AppState {
            db: Arc::new(Mutex::new(db)),
            snapshot: Arc::new(RwLock::new(Arc::new(snap))),
            metrics: Arc::new(Metrics::default()),
            rate: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn snapshot(&self) -> Arc<Snapshot> {
        self.snapshot.read().unwrap().clone()
    }

    /// Rebuild and swap the control-plane snapshot now — used after a
    /// token-API mutation so the new credential is usable immediately
    /// (rather than after the next ~1s reload tick).
    pub fn reload_now(&self) {
        let cp = { self.db.lock().unwrap().load_control_plane() };
        if let Ok(cp) = cp {
            *self.snapshot.write().unwrap() = Arc::new(Snapshot::build(&cp));
        }
    }

    /// Is this source currently throttled for repeated auth failures?
    fn is_throttled(&self, source: &str) -> bool {
        let map = self.rate.lock().unwrap();
        map.get(source).map(|e| e.blocked_until > now_secs()).unwrap_or(false)
    }

    fn record_auth_failure(&self, source: &str) {
        let mut map = self.rate.lock().unwrap();
        let e = map.entry(source.to_string()).or_insert(RateEntry { failures: 0, blocked_until: 0 });
        e.failures = e.failures.saturating_add(1);
        if e.failures >= RATE_FAIL_THRESHOLD {
            // Exponential-ish backoff capped at ~5 minutes.
            let over = (e.failures - RATE_FAIL_THRESHOLD) as i64;
            let window = (RATE_BLOCK_BASE_SECS << over.min(6)).min(300);
            e.blocked_until = now_secs() + window;
        }
    }

    fn record_auth_success(&self, source: &str) {
        self.rate.lock().unwrap().remove(source);
    }
}

/// Current wall-clock epoch seconds.
fn now_secs() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Build the router: `/healthz` (open), the introspection endpoints, and
/// a fallback that handles object operations on arbitrary key paths.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/-/whoami", get(whoami))
        .route("/-/namespaces", get(namespaces))
        .route("/-/versions/{*ns}", get(versions_list))
        .route("/metrics", get(metrics))
        .route("/auth/token", axum::routing::post(auth_token))
        .route("/tokens", axum::routing::post(issue_token))
        .route("/tokens/{id}", axum::routing::delete(revoke_token))
        .route(
            "/-/uploads/{id}",
            axum::routing::patch(upload_patch).head(upload_head).delete(upload_delete),
        )
        .fallback(object_handler)
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// `GET /metrics` — Prometheus text exposition (open; counters only).
async fn metrics(State(state): State<AppState>) -> Response {
    let m = &state.metrics;
    let body = format!(
        "# HELP vecd_requests_total Total HTTP requests.\n\
         # TYPE vecd_requests_total counter\n\
         vecd_requests_total {}\n\
         # HELP vecd_auth_failures_total Failed authentications.\n\
         # TYPE vecd_auth_failures_total counter\n\
         vecd_auth_failures_total {}\n\
         # HELP vecd_denials_total Authorization denials.\n\
         # TYPE vecd_denials_total counter\n\
         vecd_denials_total {}\n\
         # HELP vecd_throttled_total Requests rejected by the abuse throttle.\n\
         # TYPE vecd_throttled_total counter\n\
         vecd_throttled_total {}\n",
        m.requests.load(Ordering::Relaxed),
        m.auth_failures.load(Ordering::Relaxed),
        m.denials.load(Ordering::Relaxed),
        m.throttled.load(Ordering::Relaxed),
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
        .body(axum::body::Body::from(body))
        .unwrap()
}

#[derive(serde::Deserialize)]
struct AuthTokenReq {
    user: String,
    password: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    expires: Option<String>,
}

/// `POST /auth/token` (unauthenticated) — exchange `{user, password}` for a
/// freshly minted bearer token (the user's full authority). Drives
/// `vectordata login`. Throttled per source.
async fn auth_token(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    body: Bytes,
) -> Response {
    let source = addr.ip().to_string();
    if state.is_throttled(&source) {
        state.metrics.throttled.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::TOO_MANY_REQUESTS, "slow down\n").into_response();
    }
    let req: AuthTokenReq = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(_) => return (StatusCode::BAD_REQUEST, "expected {user, password}\n").into_response(),
    };
    let presented = crate::auth::hash_token(&req.password);
    let ok = { state.db.lock().unwrap().password_ok(&req.user, &presented) };
    match ok {
        Ok(Some(_)) => {}
        Ok(None) => {
            state.metrics.auth_failures.fetch_add(1, Ordering::Relaxed);
            state.record_auth_failure(&source);
            return (StatusCode::UNAUTHORIZED, "invalid credentials\n").into_response();
        }
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}\n")).into_response(),
    }
    state.record_auth_success(&source);
    let desc = req.description.unwrap_or_else(|| format!("login from {source}"));
    mint_and_respond(&state, &req.user, &desc, req.expires.as_deref(), None)
}

#[derive(serde::Deserialize)]
struct IssueReq {
    description: String,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    expires: Option<String>,
}

/// `POST /tokens` (authenticated) — a principal mints a delegated key for
/// its own identity, optionally narrowed by a profile. The cone bounds the
/// key to the issuer's live authority regardless. Drives
/// `vectordata token issue`.
async fn issue_token(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let (_snap, caller) = match auth_or_401(&state, &headers) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let Some(user) = caller.name().map(|s| s.to_string()) else {
        return (StatusCode::UNAUTHORIZED, "a token is required to issue keys\n").into_response();
    };
    let req: IssueReq = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(_) => return (StatusCode::BAD_REQUEST, "expected {description, profile?, expires?}\n").into_response(),
    };
    let profile_json = match req.profile.as_deref().map(crate::admin::parse_profile_spec).transpose() {
        Ok(p) => p,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("{e}\n")).into_response(),
    };
    mint_and_respond(&state, &user, &req.description, req.expires.as_deref(), profile_json)
}

/// `DELETE /tokens/<id>` (authenticated) — revoke a token the caller issued
/// (same user) or, for an admin+, any token.
async fn revoke_token(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    headers: HeaderMap,
) -> Response {
    let (_snap, caller) = match auth_or_401(&state, &headers) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let Some(user) = caller.name() else {
        return (StatusCode::UNAUTHORIZED, "").into_response();
    };
    let allowed = {
        let db = state.db.lock().unwrap();
        let owner = db.token_owner(id).ok().flatten();
        let me = db.user_id(user).ok().flatten();
        let is_admin = caller.level().map(|l| l >= crate::model::Level::Admin).unwrap_or(false);
        owner.is_some() && (owner == me || is_admin)
    };
    if !allowed {
        return (StatusCode::FORBIDDEN, "not your token\n").into_response();
    }
    let r = { crate::admin::revoke_token(&mut state.db.lock().unwrap(), id) };
    match r {
        Ok(()) => {
            state.reload_now();
            (StatusCode::NO_CONTENT, "").into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}\n")).into_response(),
    }
}

/// Mint a token and return it once as JSON. Reloads the snapshot so the new
/// key is usable immediately.
fn mint_and_respond(
    state: &AppState,
    user: &str,
    description: &str,
    expires: Option<&str>,
    profile_json: Option<String>,
) -> Response {
    let created = {
        let mut db = state.db.lock().unwrap();
        crate::admin::create_token(&mut db, user, description, expires, profile_json)
    };
    match created {
        Ok(tok) => {
            state.reload_now();
            json_ok(&serde_json::json!({
                "token": tok.plaintext,
                "id": tok.id,
                "user": user,
                "expires_at": tok.expires_at,
            }))
        }
        Err(VecdError::Usage(m)) => (StatusCode::BAD_REQUEST, format!("{m}\n")).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}\n")).into_response(),
    }
}

/// Authenticate from the `Authorization` header, mapping a presented-but-
/// invalid token to a 401 response.
fn auth_or_401(state: &AppState, headers: &HeaderMap) -> Result<(Arc<Snapshot>, Caller), Response> {
    let snap = state.snapshot();
    let bearer = auth::bearer_from_header(
        headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()),
    );
    match auth::authenticate(&snap, bearer, now_secs()) {
        Ok(caller) => Ok((snap, caller)),
        Err(e) => Err((StatusCode::UNAUTHORIZED, format!("{}\n", e.message())).into_response()),
    }
}

/// `GET /-/whoami` — the caller's effective access: identity, level, and
/// the visible namespaces with their effective actions. Backs
/// `vectordata ping <url>`. Available anonymously (reflects PUBLIC).
async fn whoami(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let (snap, caller) = match auth_or_401(&state, &headers) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let (visible, hidden) = snap.visible_to(&caller);
    let namespaces: Vec<serde_json::Value> = visible
        .iter()
        .map(|n| {
            serde_json::json!({
                "path": n.path,
                "actions": n.actions.iter().map(|a| a.name()).collect::<Vec<_>>(),
                "listable": n.listable.name(),
                "owner": n.owner,
            })
        })
        .collect();
    let body = serde_json::json!({
        "endpoint": "vecd",
        "identity": caller.name(),
        "level": caller.level().map(|l| l.name()),
        "authenticated": caller.is_authenticated(),
        "token_description": caller.token_desc(),
        "namespaces": namespaces,
        "hidden": hidden,
    });
    json_ok(&body)
}

/// `GET /-/namespaces` — namespace config for a privileged caller
/// (operator and above). Others get 403.
async fn namespaces(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let (snap, caller) = match auth_or_401(&state, &headers) {
        Ok(v) => v,
        Err(r) => return r,
    };
    if caller.level().map(|l| l < crate::model::Level::Operator).unwrap_or(true) {
        return (StatusCode::FORBIDDEN, "operator level required\n").into_response();
    }
    let mut rows: Vec<serde_json::Value> = snap
        .namespaces()
        .map(|n| {
            serde_json::json!({
                "path": n.path,
                "owner": n.owner,
                "backend_config": n.backend_config,
                "active": n.active,
                "listable": n.listable.name(),
                "quota_bytes": n.quota_bytes,
                "ttl_seconds": n.ttl_seconds,
            })
        })
        .collect();
    rows.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
    json_ok(&serde_json::json!({ "namespaces": rows }))
}

/// `GET /-/versions/<ns>` — the append-only version history of a
/// namespace (authorized by READ on the namespace).
async fn versions_list(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    headers: HeaderMap,
) -> Response {
    let (snap, caller) = match auth_or_401(&state, &headers) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let ns = crate::authz::normalize(&ns);
    if !snap.can(&caller, Action::Read, &ns) {
        let status =
            if caller.is_authenticated() { StatusCode::FORBIDDEN } else { StatusCode::UNAUTHORIZED };
        return (status, "").into_response();
    }
    let versions = {
        let db = state.db.lock().unwrap();
        match session::list_versions(&db, &ns) {
            Ok(v) => v,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}\n")).into_response(),
        }
    };
    let rows: Vec<serde_json::Value> = versions
        .iter()
        .map(|v| {
            serde_json::json!({
                "seq": v.seq,
                "tag": v.tag,
                "manifest_hash": v.manifest_hash,
                "state": v.state,
                "committed_at": v.committed_at,
                "expires_at": v.expires_at,
            })
        })
        .collect();
    json_ok(&serde_json::json!({ "namespace": ns, "versions": rows }))
}

fn json_ok(value: &serde_json::Value) -> Response {
    let body = serde_json::to_vec(value).unwrap_or_default();
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CONTENT_LENGTH, body.len())
        .body(axum::body::Body::from(body))
        .unwrap()
}

/// HTTP method → data-plane action.
fn method_action(method: &Method) -> Option<Action> {
    match *method {
        Method::GET => Some(Action::Read),
        Method::HEAD => Some(Action::Read),
        Method::PUT => Some(Action::Write),
        // POST on an object path creates a resumable upload — a write.
        Method::POST => Some(Action::Write),
        Method::DELETE => Some(Action::Delete),
        _ => None,
    }
}

/// Parse the conditional-write precondition from request headers. Mirrors
/// exactly what `vectordata push`'s `HttpsTransport` sends:
/// `If-None-Match: *` (must-not-exist) and `If-Match: "<etag>"`.
fn precondition(headers: &HeaderMap) -> Precondition {
    if let Some(inm) = headers.get(header::IF_NONE_MATCH).and_then(|v| v.to_str().ok()) {
        if inm.trim() == "*" {
            return Precondition::IfNoneMatchStar;
        }
    }
    if let Some(im) = headers.get(header::IF_MATCH).and_then(|v| v.to_str().ok()) {
        return Precondition::IfMatch(im.trim().trim_matches('"').to_string());
    }
    Precondition::None
}

/// The internal result of handling a request, before it becomes a
/// [`Response`].
struct Out {
    status: StatusCode,
    etag: Option<String>,
    body: Vec<u8>,
    extra: Vec<(&'static str, String)>,
    /// HEAD must not carry a body but still reports Content-Length.
    content_length: Option<u64>,
}

impl Out {
    fn status(status: StatusCode) -> Self {
        Out { status, etag: None, body: Vec::new(), extra: Vec::new(), content_length: None }
    }
    fn etag(mut self, etag: String) -> Self {
        self.etag = Some(etag);
        self
    }
    fn body(mut self, body: Vec<u8>) -> Self {
        self.content_length = Some(body.len() as u64);
        self.body = body;
        self
    }
    fn header(mut self, k: &'static str, v: String) -> Self {
        self.extra.push((k, v));
        self
    }
    fn content_length(mut self, n: u64) -> Self {
        self.content_length = Some(n);
        self
    }
    fn set_status(mut self, status: StatusCode) -> Self {
        self.status = status;
        self
    }
}

impl IntoResponse for Out {
    fn into_response(self) -> Response {
        let mut builder = Response::builder().status(self.status);
        if let Some(etag) = &self.etag {
            builder = builder.header(header::ETAG, format!("\"{etag}\""));
        }
        if let Some(len) = self.content_length {
            builder = builder.header(header::CONTENT_LENGTH, len);
        }
        for (k, v) in &self.extra {
            builder = builder.header(*k, v);
        }
        builder.body(axum::body::Body::from(self.body)).unwrap()
    }
}

async fn object_handler(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let remote = addr.ip().to_string();
    let bearer = auth::bearer_from_header(
        headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()),
    )
    .map(|s| s.to_string());

    // Streaming write path: a `PUT` of object content (anything but the
    // parsed pushlog control file) goes straight to backend storage in
    // bounded chunks — the whole object is never buffered in memory, so
    // there is no request-size cap beyond the namespace quota. Resolution is
    // pure (snapshot only), so deciding the path needs no DB or body.
    if method == Method::PUT {
        let snap = state.snapshot();
        let parsed = parse_path(uri.path());
        if let Ok(resolved) = namespace::resolve(&snap, &parsed.logical) {
            if resolved.rel_key != PUSHLOG_FILE {
                return stream_put(state, snap, parsed.logical, headers, bearer, remote, resolved, body)
                    .await;
            }
        }
    }

    // Buffered path: GET/HEAD/DELETE (empty bodies) and the small pushlog
    // PUT (which must be parsed). The blocking core runs off the async
    // runtime so DB/backend I/O never stalls the reactor.
    let bytes = match axum::body::to_bytes(body, MAX_BUFFERED_BODY).await {
        Ok(b) => b,
        Err(_) => {
            return (StatusCode::PAYLOAD_TOO_LARGE, "request body too large to buffer\n")
                .into_response()
        }
    };
    let out = tokio::task::spawn_blocking(move || {
        handle_blocking(&state, &method, &uri, &headers, bearer.as_deref(), bytes.as_ref(), &remote)
    })
    .await
    .unwrap_or_else(|_| Out::status(StatusCode::INTERNAL_SERVER_ERROR));

    out.into_response()
}

/// The shared request preamble for object operations: count, abuse-throttle,
/// authenticate, stamp `last_used`, and authorize the action against the
/// cone. On any terminal outcome it records the access-log row itself and
/// returns the [`Out`] to send; on success it returns the authorized caller
/// (the caller logs the final status once the op completes). Used by both
/// the buffered ([`handle_blocking`]) and streaming ([`stream_put`]) paths.
fn gate(
    state: &AppState,
    snap: &Snapshot,
    method: &Method,
    path: &str,
    bearer: Option<&str>,
    remote: &str,
) -> Result<Caller, Out> {
    let action_name = method.as_str();
    state.metrics.requests.fetch_add(1, Ordering::Relaxed);

    // ── abuse throttle ──────────────────────────────────────────────
    if state.is_throttled(remote) {
        state.metrics.throttled.fetch_add(1, Ordering::Relaxed);
        return Err(Out::status(StatusCode::TOO_MANY_REQUESTS)
            .body(b"too many failed attempts; slow down\n".to_vec()));
    }

    // ── authenticate ────────────────────────────────────────────────
    let caller = match auth::authenticate(snap, bearer, now_secs()) {
        Ok(c) => c,
        Err(e) => {
            state.metrics.auth_failures.fetch_add(1, Ordering::Relaxed);
            state.record_auth_failure(remote);
            let out = Out::status(StatusCode::UNAUTHORIZED)
                .body(format!("{}\n", e.message()).into_bytes());
            log(state, &Caller::Anonymous, action_name, path, out.status, 0, remote);
            return Err(out);
        }
    };
    if caller.is_authenticated() {
        state.record_auth_success(remote);
    }

    // Stamp last_used for a real token (best-effort, no reload).
    if let Caller::User { token_id, .. } = &caller {
        if let Ok(db) = state.db.lock() {
            db.touch_token(*token_id);
        }
    }

    // ── authorize ───────────────────────────────────────────────────
    let Some(action) = method_action(method) else {
        let out = Out::status(StatusCode::METHOD_NOT_ALLOWED);
        log(state, &caller, action_name, path, out.status, 0, remote);
        return Err(out);
    };
    if !snap.can(&caller, action, path) {
        state.metrics.denials.fetch_add(1, Ordering::Relaxed);
        // 401 when the caller could authenticate to gain access; 403 when
        // an authenticated principal is simply not permitted.
        let status = if caller.is_authenticated() {
            StatusCode::FORBIDDEN
        } else {
            StatusCode::UNAUTHORIZED
        };
        let out = Out::status(status);
        log(state, &caller, action_name, path, out.status, 0, remote);
        return Err(out);
    }

    Ok(caller)
}

/// The synchronous request core for the buffered path. Returns the [`Out`]
/// to render; also appends the access-log row (the final pipeline stage).
fn handle_blocking(
    state: &AppState,
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    bearer: Option<&str>,
    body: &[u8],
    remote: &str,
) -> Out {
    let snap = state.snapshot();
    let parsed = parse_path(uri.path());
    let path = parsed.logical.as_str();
    let action_name = method.as_str();

    let caller = match gate(state, &snap, method, path, bearer, remote) {
        Ok(c) => c,
        Err(out) => return out,
    };

    // ── resolve + handle ────────────────────────────────────────────
    let out = match handle_object(state, &snap, &caller, method, &parsed, headers, body, uri.query()) {
        Ok(out) => out,
        Err(VecdError::Usage(m)) => {
            Out::status(StatusCode::BAD_REQUEST).body(format!("{m}\n").into_bytes())
        }
        Err(e) => Out::status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(format!("{e}\n").into_bytes()),
    };
    log(state, &caller, action_name, path, out.status, out.content_length.unwrap_or(0), remote);
    out
}

/// The streaming `PUT` path: gate, then stream the request body into a
/// backend staging blob in bounded chunks, then finalize (session-stage or a
/// CAS+quota lone write). Object content never lands in memory whole.
#[allow(clippy::too_many_arguments)]
async fn stream_put(
    state: AppState,
    snap: Arc<Snapshot>,
    path: String,
    headers: HeaderMap,
    bearer: Option<String>,
    remote: String,
    resolved: Resolved,
    body: Body,
) -> Response {
    // ── gate (throttle + authn + authz) before any storage I/O ──────
    let caller = {
        let state = state.clone();
        let snap = snap.clone();
        let path = path.clone();
        let bearer = bearer.clone();
        let remote = remote.clone();
        let gated = tokio::task::spawn_blocking(move || {
            gate(&state, &snap, &Method::PUT, &path, bearer.as_deref(), &remote)
        })
        .await
        .unwrap_or_else(|_| Err(Out::status(StatusCode::INTERNAL_SERVER_ERROR)));
        match gated {
            Ok(c) => c,
            Err(out) => return out.into_response(),
        }
    };

    // ── stream the body into a backend staging blob ─────────────────
    let staging_key = new_upload_id();
    let backend = resolved.backend.clone();
    let size = match stream_to_staging(backend.clone(), staging_key.clone(), 0, u64::MAX, body).await {
        Ok(n) => n,
        Err(e) => {
            let _ = backend.discard_staged(&staging_key);
            let out =
                Out::status(StatusCode::BAD_REQUEST).body(format!("{e}\n").into_bytes());
            log(&state, &caller, "PUT", &path, out.status, 0, &remote);
            return out.into_response();
        }
    };

    // ── finalize (DB commit) off the reactor ────────────────────────
    let cond = precondition(&headers);
    let ignore_quota = snap.has_system_privilege(&caller, PRIV_IGNORE_QUOTAS);
    let quota = snap
        .namespace(&resolved.storage_ns)
        .map(|n| n.quota_bytes)
        .unwrap_or(crate::model::DEFAULT_QUOTA_BYTES);
    let finalized = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || {
            finalize_streamed_put(&state, resolved, &staging_key, size, &cond, quota, ignore_quota)
        })
        .await
        .unwrap_or_else(|_| Err(VecdError::op("finalize task failed")))
    };
    let out = match finalized {
        Ok(out) => out,
        Err(VecdError::Usage(m)) => {
            Out::status(StatusCode::BAD_REQUEST).body(format!("{m}\n").into_bytes())
        }
        Err(e) => Out::status(StatusCode::INTERNAL_SERVER_ERROR).body(format!("{e}\n").into_bytes()),
    };
    log(&state, &caller, "PUT", &path, out.status, out.content_length.unwrap_or(0), &remote);
    out.into_response()
}

/// Stream a request body to a backend staging blob starting at byte
/// `base_offset`, returning the number of bytes written. The async task
/// pulls body frames and hands them, with backpressure (a bounded channel),
/// to a blocking writer that coalesces them into `UPLOAD_FLUSH_BYTES`
/// [`Backend::put_at`] calls — so resident memory stays bounded regardless
/// of object size. No byte is written at or beyond `ceiling` (the declared
/// object length for a resumable `PATCH`; `u64::MAX` for an unbounded whole
/// `PUT`); a body that would overrun it is rejected.
async fn stream_to_staging(
    backend: Arc<dyn Backend>,
    staging_key: String,
    base_offset: u64,
    ceiling: u64,
    body: Body,
) -> Result<u64, VecdError> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<Bytes>(8);
    let writer = {
        let staging_key = staging_key.clone();
        tokio::task::spawn_blocking(move || -> Result<u64, VecdError> {
            let mut offset = base_offset;
            let mut buf: Vec<u8> = Vec::new();
            while let Ok(chunk) = rx.recv() {
                buf.extend_from_slice(&chunk);
                if buf.len() >= UPLOAD_FLUSH_BYTES {
                    backend.put_at(&staging_key, offset, &buf)?;
                    offset += buf.len() as u64;
                    buf.clear();
                }
            }
            if !buf.is_empty() {
                backend.put_at(&staging_key, offset, &buf)?;
                offset += buf.len() as u64;
            }
            // Materialize the staging blob even when nothing was written, so
            // a zero-byte object (e.g. an empty `.ivecs`) still finalizes
            // (otherwise there is no staged blob to promote).
            if offset == base_offset {
                backend.put_at(&staging_key, base_offset, &[])?;
            }
            Ok(offset - base_offset)
        })
    };

    let mut stream = body.into_data_stream();
    let mut written = base_offset;
    let mut read_err = None;
    while let Some(frame) = stream.next().await {
        match frame {
            Ok(chunk) => {
                if chunk.is_empty() {
                    continue;
                }
                if written.saturating_add(chunk.len() as u64) > ceiling {
                    read_err = Some(VecdError::usage(
                        "chunk extends past the declared Upload-Length".to_string(),
                    ));
                    break;
                }
                written += chunk.len() as u64;
                if tx.send(chunk).is_err() {
                    break; // the writer stopped (its error surfaces below)
                }
            }
            Err(e) => {
                read_err = Some(VecdError::op(format!("reading request body: {e}")));
                break;
            }
        }
    }
    drop(tx); // signal end-of-stream; the writer flushes and returns
    let written = writer
        .await
        .map_err(|e| VecdError::op(format!("upload writer join: {e}")))??;
    if let Some(e) = read_err {
        return Err(e);
    }
    Ok(written)
}

/// Finalize a streamed `PUT`: inside an open session it stages (invisible to
/// readers); otherwise it is a lone write with per-object CAS + quota. The
/// bytes already live in the backend staging blob `staging_key`.
fn finalize_streamed_put(
    state: &AppState,
    resolved: Resolved,
    staging_key: &str,
    size: u64,
    cond: &Precondition,
    quota: u64,
    ignore_quota: bool,
) -> Result<Out, VecdError> {
    let mut db = state.db.lock().unwrap();
    if session::is_open(&db, &resolved.storage_ns)? {
        let etag = session::stage_put_staged(&mut db, &resolved, staging_key, size)?;
        return Ok(Out::status(StatusCode::CREATED).etag(etag));
    }
    match store::put_staged(&mut db, &resolved, staging_key, size, cond, quota, ignore_quota)? {
        PutResult::Written { etag } => Ok(Out::status(StatusCode::CREATED).etag(etag)),
        PutResult::PreconditionFailed => Ok(Out::status(StatusCode::PRECONDITION_FAILED)),
        PutResult::QuotaExceeded { quota } => {
            Ok(Out::status(StatusCode::INSUFFICIENT_STORAGE).header("X-Vecd-Quota", quota.to_string()))
        }
    }
}

/// A fresh, collision-resistant id for an upload resource / staging blob.
fn new_upload_id() -> String {
    format!("{:016x}{:016x}", rand::random::<u64>(), rand::random::<u64>())
}

/// Parse a structured-fields integer header (RFC 8941 — an sf-integer is the
/// plain decimal number). Backs `Upload-Length` / `Upload-Offset`.
fn parse_sf_integer(headers: &HeaderMap, name: &str) -> Option<u64> {
    headers.get(name).and_then(|v| v.to_str().ok()).and_then(|s| s.trim().parse::<u64>().ok())
}

/// `POST /<ns>/<key>` with `Upload-Length: <N>` — create a resumable upload
/// resource (IETF "Resumable Uploads for HTTP"). Allocates an upload id and a
/// backend staging blob, captures any CAS precondition for finalize, and
/// returns `201` with the upload URL in `Location` and `Upload-Offset: 0`.
fn handle_create_upload(state: &AppState, r: &Resolved, headers: &HeaderMap) -> Result<Out, VecdError> {
    let total = parse_sf_integer(headers, "upload-length").ok_or_else(|| {
        VecdError::usage("creating an upload requires an Upload-Length header".to_string())
    })?;
    let (if_match, if_none_match) = match precondition(headers) {
        Precondition::None => (None, false),
        Precondition::IfNoneMatchStar => (None, true),
        Precondition::IfMatch(e) => (Some(e), false),
    };
    let upload_id = new_upload_id();
    let staging_key = new_upload_id();
    {
        let db = state.db.lock().unwrap();
        upload::create(
            &db,
            &upload_id,
            &r.storage_ns,
            &r.rel_key,
            total,
            &staging_key,
            if_match.as_deref(),
            if_none_match,
            now_secs(),
        )?;
    }
    Ok(Out::status(StatusCode::CREATED)
        .header("Location", format!("/-/uploads/{upload_id}"))
        .header("Upload-Offset", "0".to_string())
        .header("Upload-Length", total.to_string())
        .header("Upload-Complete", "?0".to_string()))
}

/// Authenticate, load the upload by id, and authorize WRITE on its key.
/// Returns `(snapshot, caller, upload)` or a terminal response.
fn authorize_upload(
    state: &AppState,
    headers: &HeaderMap,
    upload_id: &str,
) -> Result<(Arc<Snapshot>, Caller, upload::Upload), Response> {
    let (snap, caller) = auth_or_401(state, headers)?;
    let found = {
        let db = state.db.lock().unwrap();
        upload::find(&db, upload_id)
    };
    let upload = match found {
        Ok(Some(u)) => u,
        Ok(None) => return Err((StatusCode::NOT_FOUND, "no such upload\n").into_response()),
        Err(e) => return Err((StatusCode::INTERNAL_SERVER_ERROR, format!("{e}\n")).into_response()),
    };
    if !snap.can(&caller, Action::Write, &upload.path()) {
        let status =
            if caller.is_authenticated() { StatusCode::FORBIDDEN } else { StatusCode::UNAUTHORIZED };
        return Err((status, "").into_response());
    }
    Ok((snap, caller, upload))
}

/// A bare resumable-upload status response: the `Upload-Offset` /
/// `Upload-Length` / `Upload-Complete` structured-field headers (and the
/// object `ETag` once finalized), with no body.
fn upload_status_response(
    status: StatusCode,
    offset: u64,
    total: u64,
    complete: bool,
    etag: Option<&str>,
) -> Response {
    let mut b = Response::builder()
        .status(status)
        .header("Upload-Offset", offset.to_string())
        .header("Upload-Length", total.to_string())
        .header("Upload-Complete", if complete { "?1" } else { "?0" });
    if let Some(e) = etag {
        b = b.header(header::ETAG, format!("\"{e}\""));
    }
    b.body(Body::empty()).unwrap()
}

/// `PATCH /-/uploads/<id>` with `Upload-Offset: <k>` and a chunk body — write
/// `[k, k+len)` sparsely to the staging blob, merge the received interval,
/// and acknowledge the linearized contiguous-prefix offset. Concurrent
/// PATCHes at disjoint offsets are allowed; the interval-merge is serialized
/// by the DB mutex. When the prefix reaches `Upload-Length`, the object is
/// committed (finalize-on-full).
async fn upload_patch(
    State(state): State<AppState>,
    Path(upload_id): Path<String>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let (snap, caller, upload) = match authorize_upload(&state, &headers, &upload_id) {
        Ok(v) => v,
        Err(r) => return r,
    };
    // Already committed → idempotent acknowledgement.
    if upload.complete {
        return upload_status_response(
            StatusCode::NO_CONTENT,
            upload.total_length,
            upload.total_length,
            true,
            upload.etag.as_deref(),
        );
    }
    let Some(k) = parse_sf_integer(&headers, "upload-offset") else {
        return (StatusCode::BAD_REQUEST, "PATCH requires an Upload-Offset header\n").into_response();
    };
    if k > upload.total_length {
        return (StatusCode::BAD_REQUEST, "Upload-Offset is past Upload-Length\n").into_response();
    }
    let resolved = match namespace::resolve(&snap, &upload.path()) {
        Ok(r) => r,
        Err(_) => return (StatusCode::NOT_FOUND, "upload target no longer resolves\n").into_response(),
    };

    // Stream the chunk to [k, ..), bounded by the declared length. Sparse
    // writes to disjoint regions proceed in parallel with other PATCHes.
    let len = match stream_to_staging(
        resolved.backend.clone(),
        upload.staging_key.clone(),
        k,
        upload.total_length,
        body,
    )
    .await
    {
        Ok(n) => n,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("{e}\n")).into_response(),
    };

    // Merge the interval (serialized by the DB mutex) and read the prefix +
    // whether this PATCH won the right to finalize the now-full upload.
    let (prefix, claimed) = {
        let mut db = state.db.lock().unwrap();
        match upload::record_chunk(&mut db, &upload_id, k, len, upload.total_length) {
            Ok(r) => r,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}\n")).into_response(),
        }
    };

    // Finalize-on-full: exactly one PATCH (the claimer) promotes the staged
    // blob to the object; concurrent fillers just report the full offset.
    if claimed {
        let cond = upload.precondition();
        let ignore_quota = snap.has_system_privilege(&caller, PRIV_IGNORE_QUOTAS);
        let quota = snap
            .namespace(&resolved.storage_ns)
            .map(|n| n.quota_bytes)
            .unwrap_or(crate::model::DEFAULT_QUOTA_BYTES);
        let total = upload.total_length;
        let staging_key = upload.staging_key.clone();
        match finalize_streamed_put(&state, resolved, &staging_key, total, &cond, quota, ignore_quota) {
            Ok(out) if out.status == StatusCode::CREATED => {
                if let Some(etag) = &out.etag {
                    let db = state.db.lock().unwrap();
                    let _ = upload::mark_complete(&db, &upload_id, etag);
                }
                return upload_status_response(StatusCode::NO_CONTENT, total, total, true, out.etag.as_deref());
            }
            // CAS / quota failure at finalize — release the claim so the
            // upload can be retried or deleted, and surface the status.
            Ok(out) => {
                let db = state.db.lock().unwrap();
                let _ = upload::reset_finalizing(&db, &upload_id);
                return out.into_response();
            }
            Err(e) => {
                let db = state.db.lock().unwrap();
                let _ = upload::reset_finalizing(&db, &upload_id);
                return match e {
                    VecdError::Usage(m) => (StatusCode::BAD_REQUEST, format!("{m}\n")).into_response(),
                    e => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}\n")).into_response(),
                };
            }
        }
    }

    // Not the finalizer: report the contiguous offset (which equals total
    // when another PATCH is finalizing — the client can HEAD for the commit).
    upload_status_response(StatusCode::NO_CONTENT, prefix, upload.total_length, false, None)
}

/// `HEAD /-/uploads/<id>` — report the resumable offset so a client can
/// resume after an interruption (re-send only the ranges at/after the
/// contiguous offset).
async fn upload_head(
    State(state): State<AppState>,
    Path(upload_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let (_snap, _caller, upload) = match authorize_upload(&state, &headers, &upload_id) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let (offset, complete) = if upload.complete {
        (upload.total_length, true)
    } else {
        let db = state.db.lock().unwrap();
        match upload::received(&db, &upload_id, upload.total_length) {
            Ok(rr) => (rr.contiguous_prefix(), false),
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}\n")).into_response(),
        }
    };
    upload_status_response(StatusCode::NO_CONTENT, offset, upload.total_length, complete, upload.etag.as_deref())
}

/// `DELETE /-/uploads/<id>` — abandon an in-progress upload: free its staging
/// blob and forget its state. Idempotent once gone.
async fn upload_delete(
    State(state): State<AppState>,
    Path(upload_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let (snap, _caller, upload) = match authorize_upload(&state, &headers, &upload_id) {
        Ok(v) => v,
        Err(r) => return r,
    };
    if !upload.complete {
        if let Ok(resolved) = namespace::resolve(&snap, &upload.path()) {
            let _ = resolved.backend.discard_staged(&upload.staging_key);
        }
    }
    {
        let db = state.db.lock().unwrap();
        let _ = upload::delete(&db, &upload_id);
    }
    (StatusCode::NO_CONTENT, "").into_response()
}

fn handle_object(
    state: &AppState,
    snap: &Snapshot,
    caller: &Caller,
    method: &Method,
    parsed: &ParsedPath,
    headers: &HeaderMap,
    body: &[u8],
    query: Option<&str>,
) -> Result<Out, VecdError> {
    let path = parsed.logical.as_str();

    // `GET <ns>/@<ver>/-/manifest` — a version's full manifest.
    if parsed.manifest && matches!(*method, Method::GET | Method::HEAD) {
        return handle_manifest(state, snap, path, parsed.selector.as_deref().unwrap_or("latest"));
    }

    // `GET <prefix>?list` — enumerate keys+etags under the prefix (READ
    // already authorized above). Enables push `--delete` over vecd.
    if *method == Method::GET && query_has_list(query) {
        return handle_list(state, snap, path);
    }

    // Routing miss (no active backend / addresses a namespace) → 404.
    let resolved = match namespace::resolve(snap, path) {
        Ok(r) => r,
        Err(VecdError::Usage(_)) => return Ok(Out::status(StatusCode::NOT_FOUND)),
        Err(e) => return Err(e),
    };

    // A pinned version selector (`@v3`/`@<tag>`/`@<hash>`) reads the
    // immutable snapshot; `@latest` (or none) reads the live manifest.
    let pinned = match parsed.selector.as_deref() {
        Some(sel) if sel != "latest" => Some(sel.to_string()),
        _ => None,
    };

    match *method {
        Method::HEAD => {
            let db = state.db.lock().unwrap();
            if let Some(sel) = &pinned {
                return read_pinned(&db, &resolved, sel, false)
                    .map(|out| if out.status == StatusCode::OK {
                        out.header("Accept-Ranges", "bytes".to_string())
                    } else {
                        out
                    });
            }
            match store::head(&db, &resolved)? {
                Some(meta) => Ok(latest_headers(&db, &resolved.storage_ns, Out::status(StatusCode::OK))?
                    .etag(meta.etag)
                    .content_length(meta.size)
                    .header("Accept-Ranges", "bytes".to_string())),
                None => Ok(absent_or_gone(&db, &resolved.storage_ns)?),
            }
        }
        Method::GET => {
            let db = state.db.lock().unwrap();
            if let Some(sel) = &pinned {
                return read_pinned(&db, &resolved, sel, true).map(|out| apply_get_range(out, headers));
            }
            match store::get(&db, &resolved)? {
                Some((bytes, meta)) => Ok(apply_get_range(
                    latest_headers(&db, &resolved.storage_ns, Out::status(StatusCode::OK))?
                        .etag(meta.etag)
                        .body(bytes),
                    headers,
                )),
                None => Ok(absent_or_gone(&db, &resolved.storage_ns)?),
            }
        }
        Method::PUT => handle_put(state, snap, caller, &resolved, headers, body),
        Method::POST => handle_create_upload(state, &resolved, headers),
        Method::DELETE => {
            let mut db = state.db.lock().unwrap();
            if session::is_open(&db, &resolved.storage_ns)? {
                // Inside a session, a delete is a staged manifest omission.
                session::stage_delete(&mut db, &resolved.storage_ns, &resolved.rel_key)?;
            } else {
                store::delete(&mut db, &resolved)?;
            }
            Ok(Out::status(StatusCode::NO_CONTENT))
        }
        _ => Ok(Out::status(StatusCode::METHOD_NOT_ALLOWED)),
    }
}

/// A request path split around an optional `@<selector>` version segment.
struct ParsedPath {
    /// The logical key/namespace path with the `@selector` segment removed.
    logical: String,
    /// The version selector, if `@…` was present (`latest`/`v3`/tag/hash).
    selector: Option<String>,
    /// True for the `…/@<ver>/-/manifest` introspection form.
    manifest: bool,
}

/// Parse `…/@<selector>/…` (and the `…/@<ver>/-/manifest` form) out of a
/// request path. Without an `@` segment, the whole path is the logical key.
fn parse_path(raw: &str) -> ParsedPath {
    let raw = raw.trim_matches('/');
    let segs: Vec<&str> = if raw.is_empty() { Vec::new() } else { raw.split('/').collect() };
    if let Some(i) = segs.iter().position(|s| s.starts_with('@')) {
        let selector = segs[i][1..].to_string();
        let before = &segs[..i];
        let after = &segs[i + 1..];
        if after == ["-", "manifest"] {
            return ParsedPath { logical: before.join("/"), selector: Some(selector), manifest: true };
        }
        let mut logical: Vec<&str> = before.to_vec();
        logical.extend_from_slice(after);
        return ParsedPath { logical: logical.join("/"), selector: Some(selector), manifest: false };
    }
    ParsedPath { logical: raw.to_string(), selector: None, manifest: false }
}

/// Decorate an `Out` with the current (`@latest`) version headers.
fn latest_headers(db: &Db, ns: &str, out: Out) -> Result<Out, VecdError> {
    match session::current_version(db, ns)? {
        Some(v) => Ok(version_headers(out, &v)),
        None => Ok(out),
    }
}

/// Apply `X-Vecd-Version`/`X-Vecd-Manifest` and lifecycle headers.
fn version_headers(mut out: Out, v: &session::VersionRow) -> Out {
    out = out.header("X-Vecd-Version", v.tag.clone()).header("X-Vecd-Manifest", v.manifest_hash.clone());
    if let Some(exp) = v.expires_at {
        out = out
            .header("X-Vecd-Expires", crate::admin::fmt_epoch(exp))
            .header("X-Vecd-Lifecycle", "live".to_string());
    }
    out
}

/// A missing `@latest` object is `404`, unless the whole namespace has
/// expired to stasis — then `410 Gone` (its data is recoverable only by an
/// admin `cleanup extend`).
fn absent_or_gone(db: &Db, ns: &str) -> Result<Out, VecdError> {
    if crate::lifetime::ns_in_stasis(db, ns)? {
        Ok(Out::status(StatusCode::GONE).header("X-Vecd-Lifecycle", "stasis".to_string()))
    } else {
        Ok(Out::status(StatusCode::NOT_FOUND))
    }
}

/// Read one key from a pinned version snapshot.
fn read_pinned(db: &Db, r: &Resolved, selector: &str, with_body: bool) -> Result<Out, VecdError> {
    let Some(v) = session::find_version(db, &r.storage_ns, selector)? else {
        return Ok(Out::status(StatusCode::NOT_FOUND));
    };
    if v.state == "stasis" {
        return Ok(Out::status(StatusCode::GONE).header("X-Vecd-Lifecycle", "stasis".to_string()));
    }
    let Some((ck, size)) = session::version_object(db, v.id, &r.rel_key)? else {
        return Ok(Out::status(StatusCode::NOT_FOUND));
    };
    let base = version_headers(Out::status(StatusCode::OK), &v).etag(ck.clone());
    if with_body {
        match r.backend.get(&store::physical(&r.rel_key, &ck))? {
            Some(bytes) => Ok(base.body(bytes)),
            None => Err(VecdError::op(format!("version {} key '{}' content missing", v.tag, r.rel_key))),
        }
    } else {
        Ok(base.content_length(size))
    }
}

/// Outcome of parsing a `Range` header against a known object size.
enum RangeOutcome {
    /// No range to honor (absent, multi-range, or unparseable) — serve the
    /// full `200` body. RFC 7233 §3.1 permits ignoring a range we don't honor.
    Full,
    /// A satisfiable single byte range, inclusive of `end`.
    Satisfiable { start: u64, end: u64 },
    /// The range lies wholly outside the object — `416`.
    Unsatisfiable,
}

/// Parse a single-range `bytes=` specifier against the object's `total` size.
/// Supports `bytes=a-b`, `bytes=a-` (to end), and `bytes=-n` (suffix). A
/// multi-range set or garbage falls back to [`RangeOutcome::Full`].
fn parse_byte_range(spec: &str, total: u64) -> RangeOutcome {
    let Some(set) = spec.trim().strip_prefix("bytes=") else {
        return RangeOutcome::Full;
    };
    // Only single ranges are honored; a comma-separated set falls back to 200.
    if set.contains(',') {
        return RangeOutcome::Full;
    }
    let Some((a, b)) = set.trim().split_once('-') else {
        return RangeOutcome::Full;
    };
    let (a, b) = (a.trim(), b.trim());
    if total == 0 {
        return RangeOutcome::Unsatisfiable;
    }
    let last = total - 1;
    let (start, end) = if a.is_empty() {
        // Suffix form `bytes=-n`: the final `n` bytes.
        match b.parse::<u64>() {
            Ok(0) => return RangeOutcome::Unsatisfiable,
            Ok(n) => (total.saturating_sub(n), last),
            Err(_) => return RangeOutcome::Full,
        }
    } else {
        let start = match a.parse::<u64>() {
            Ok(s) => s,
            Err(_) => return RangeOutcome::Full,
        };
        let end = if b.is_empty() {
            last
        } else {
            match b.parse::<u64>() {
                Ok(e) => e.min(last),
                Err(_) => return RangeOutcome::Full,
            }
        };
        (start, end)
    };
    if start > last || start > end {
        return RangeOutcome::Unsatisfiable;
    }
    RangeOutcome::Satisfiable { start, end }
}

/// Apply a request `Range` header to a fully-bodied `200 OK` object response.
///
/// Advertises `Accept-Ranges: bytes`. A satisfiable single byte range becomes
/// `206 Partial Content` with `Content-Range` and the sliced body, preserving
/// the ETag and version headers; an unsatisfiable range yields `416`. Non-`200`
/// responses (404/410/…) pass through untouched.
fn apply_get_range(out: Out, headers: &HeaderMap) -> Out {
    if out.status != StatusCode::OK {
        return out;
    }
    let out = out.header("Accept-Ranges", "bytes".to_string());
    let Some(spec) = headers.get(header::RANGE).and_then(|v| v.to_str().ok()) else {
        return out;
    };
    let total = out.body.len() as u64;
    match parse_byte_range(spec, total) {
        RangeOutcome::Full => out,
        RangeOutcome::Satisfiable { start, end } => {
            let slice = out.body[start as usize..=end as usize].to_vec();
            out.header("Content-Range", format!("bytes {start}-{end}/{total}"))
                .set_status(StatusCode::PARTIAL_CONTENT)
                .body(slice)
        }
        RangeOutcome::Unsatisfiable => out
            .header("Content-Range", format!("bytes */{total}"))
            .set_status(StatusCode::RANGE_NOT_SATISFIABLE)
            .body(Vec::new()),
    }
}

/// `GET <ns>/@<ver>/-/manifest` — a version's full manifest as JSON.
fn handle_manifest(state: &AppState, snap: &Snapshot, path: &str, selector: &str) -> Result<Out, VecdError> {
    let (storage_ns, _) = match namespace::resolve_for_list(snap, path) {
        Ok(v) => v,
        Err(VecdError::Usage(_)) => return Ok(Out::status(StatusCode::NOT_FOUND)),
        Err(e) => return Err(e),
    };
    let db = state.db.lock().unwrap();
    let Some(v) = session::find_version(&db, &storage_ns, selector)? else {
        return Ok(Out::status(StatusCode::NOT_FOUND));
    };
    if v.state == "stasis" {
        return Ok(Out::status(StatusCode::GONE).header("X-Vecd-Lifecycle", "stasis".to_string()));
    }
    let entries: Vec<serde_json::Value> = session::version_manifest(&db, v.id)?
        .into_iter()
        .map(|(key, ck, size)| serde_json::json!({ "key": key, "content_key": ck, "size": size }))
        .collect();
    let body = serde_json::to_vec(&serde_json::json!({
        "namespace": storage_ns,
        "version": v.tag,
        "seq": v.seq,
        "manifest_hash": v.manifest_hash,
        "objects": entries,
    }))
    .unwrap_or_default();
    Ok(version_headers(Out::status(StatusCode::OK), &v)
        .header("Content-Type", "application/json".to_string())
        .body(body))
}

/// Handle a `PUT`. The pushlog drives transactional sessions: a `begin`
/// tail opens a staging session, content PUTs stage into it, and a
/// `complete` tail commits (atomic pointer flip). Writes outside a session
/// mutate the live manifest with per-object CAS.
fn handle_put(
    state: &AppState,
    snap: &Snapshot,
    caller: &Caller,
    r: &Resolved,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<Out, VecdError> {
    let cond = precondition(headers);
    let ns = r.storage_ns.clone();
    let mut db = state.db.lock().unwrap();

    // The pushlog is the session signal. Its conditional writes are
    // evaluated against the *committed* pushlog (the live manifest), so the
    // single-provenance CAS guarantee holds while the new version stages
    // invisibly.
    if r.rel_key == PUSHLOG_FILE {
        if let Ok(log) = Log::parse(&String::from_utf8_lossy(body)) {
            match log.events.last() {
                Some(Event::Begin { deletes, actor, .. }) => {
                    if !committed_precondition_ok(&db, r, &cond)? {
                        return Ok(Out::status(StatusCode::PRECONDITION_FAILED));
                    }
                    let actor = actor.clone();
                    let deletes = deletes.clone();
                    session::open(&mut db, &ns, Some(&actor), &deletes)?;
                    let etag = session::stage_put_bytes(&mut db, r, body)?;
                    return Ok(Out::status(StatusCode::CREATED).etag(etag));
                }
                Some(Event::Complete { .. }) => {
                    if !committed_precondition_ok(&db, r, &cond)? {
                        return Ok(Out::status(StatusCode::PRECONDITION_FAILED));
                    }
                    // A complete with no open session (e.g. resume after the
                    // staging was lost) implicitly opens one from the live
                    // manifest so the commit is well-defined.
                    if !session::is_open(&db, &ns)? {
                        session::open(&mut db, &ns, None, &[])?;
                    }
                    let etag = session::stage_put_bytes(&mut db, r, body)?;
                    let ttl = snap.namespace(&ns).and_then(|n| n.ttl_seconds);
                    session::commit(&mut db, &ns, ttl)?;
                    return Ok(Out::status(StatusCode::CREATED).etag(etag));
                }
                Some(Event::Abort { .. }) => {
                    if !committed_precondition_ok(&db, r, &cond)? {
                        return Ok(Out::status(StatusCode::PRECONDITION_FAILED));
                    }
                    if session::is_open(&db, &ns)? {
                        session::abort(&mut db, &ns)?;
                    }
                    return Ok(Out::status(StatusCode::OK));
                }
                None => {} // empty log → fall through to a lone write
            }
        }
    }

    // Inside an open session, any object PUT stages (invisible to readers).
    if session::is_open(&db, &ns)? {
        let etag = session::stage_put_bytes(&mut db, r, body)?;
        return Ok(Out::status(StatusCode::CREATED).etag(etag));
    }

    // Otherwise a lone write to the live manifest: per-object CAS + quota.
    let quota = snap.namespace(&ns).map(|n| n.quota_bytes).unwrap_or(crate::model::DEFAULT_QUOTA_BYTES);
    let ignore_quota = snap.has_system_privilege(caller, PRIV_IGNORE_QUOTAS);
    match store::put(&mut db, r, body, &cond, quota, ignore_quota)? {
        PutResult::Written { etag } => Ok(Out::status(StatusCode::CREATED).etag(etag)),
        PutResult::PreconditionFailed => Ok(Out::status(StatusCode::PRECONDITION_FAILED)),
        PutResult::QuotaExceeded { quota } => {
            Ok(Out::status(StatusCode::INSUFFICIENT_STORAGE).header("X-Vecd-Quota", quota.to_string()))
        }
    }
}

/// Evaluate a conditional-write precondition against the *committed* (live)
/// object — used for pushlog writes during a session.
fn committed_precondition_ok(db: &Db, r: &Resolved, cond: &Precondition) -> Result<bool, VecdError> {
    let committed = store::head(db, r)?.map(|m| m.etag);
    Ok(match cond {
        Precondition::None => true,
        Precondition::IfNoneMatchStar => committed.is_none(),
        Precondition::IfMatch(e) => committed.as_deref() == Some(e.as_str()),
    })
}

/// Does the query string request a listing (`?list` or `?list=...`)?
fn query_has_list(query: Option<&str>) -> bool {
    query
        .map(|q| q.split('&').any(|kv| kv == "list" || kv.starts_with("list=")))
        .unwrap_or(false)
}

/// `GET <prefix>?list` — keys + content-keys under the prefix, as JSON
/// `{"keys":[{"key":..,"etag":..}]}`. Keys are namespace-relative (what a
/// publish-root client expects from `list("")`).
fn handle_list(state: &AppState, snap: &Snapshot, path: &str) -> Result<Out, VecdError> {
    let (storage_ns, rel_prefix) = match namespace::resolve_for_list(snap, path) {
        Ok(v) => v,
        Err(VecdError::Usage(_)) => return Ok(Out::status(StatusCode::NOT_FOUND)),
        Err(e) => return Err(e),
    };
    let entries = {
        let db = state.db.lock().unwrap();
        store::list(&db, &storage_ns)?
    };
    let keys: Vec<serde_json::Value> = entries
        .into_iter()
        .filter(|(k, _)| {
            rel_prefix.is_empty() || *k == rel_prefix || k.starts_with(&format!("{rel_prefix}/"))
        })
        .map(|(k, etag)| serde_json::json!({ "key": k, "etag": etag }))
        .collect();
    let body = serde_json::to_vec(&serde_json::json!({ "keys": keys })).unwrap_or_default();
    Ok(Out::status(StatusCode::OK)
        .header("Content-Type", "application/json".to_string())
        .body(body))
}

fn log(
    state: &AppState,
    caller: &Caller,
    action: &str,
    key: &str,
    status: StatusCode,
    bytes: u64,
    remote: &str,
) {
    if let Ok(db) = state.db.lock() {
        db.log_access(
            caller.name(),
            caller.token_desc(),
            action,
            key,
            status.as_u16(),
            bytes,
            remote,
        );
    }
}

// ── live reload ─────────────────────────────────────────────────────

/// Spawn the live-reload tick. Polls `PRAGMA data_version` (cheap, in
/// memory) on the given interval; when another connection has committed
/// *and* `auth_generation` advanced past the current snapshot, rebuilds
/// and atomically swaps the snapshot in.
pub fn spawn_reloader(state: AppState, interval: std::time::Duration) {
    tokio::spawn(async move {
        let mut last_data_version = {
            let db = state.db.lock().unwrap();
            db.data_version().unwrap_or(0)
        };
        loop {
            tokio::time::sleep(interval).await;
            let reload = {
                let db = state.db.lock().unwrap();
                match db.data_version() {
                    Ok(dv) if dv != last_data_version => {
                        last_data_version = dv;
                        let current_gen = state.snapshot.read().unwrap().generation;
                        db.auth_generation().map(|g| g > current_gen).unwrap_or(false)
                    }
                    _ => false,
                }
            };
            if reload {
                let cp = {
                    let db = state.db.lock().unwrap();
                    db.load_control_plane()
                };
                if let Ok(cp) = cp {
                    let snap = Arc::new(Snapshot::build(&cp));
                    *state.snapshot.write().unwrap() = snap;
                    log::info!("vecd: reloaded control-plane snapshot (generation {})", cp.generation);
                }
            }
        }
    });
}

/// Spawn the lifecycle sweeper: periodically move expired committed
/// versions to stasis (non-destructive) and roll each affected namespace
/// back to its newest surviving version.
pub fn spawn_sweeper(state: AppState, interval: std::time::Duration) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(interval).await;
            let db = state.db.clone();
            let swept = tokio::task::spawn_blocking(move || {
                let mut db = db.lock().unwrap();
                crate::lifetime::sweep(&mut db)
            })
            .await;
            match swept {
                Ok(Ok(n)) if n > 0 => log::info!("vecd: swept {n} expired version(s) to stasis"),
                Ok(Err(e)) => log::warn!("vecd: lifecycle sweep failed: {e}"),
                _ => {}
            }
        }
    });
}

/// TLS material for [`serve`].
pub struct TlsConfig {
    pub cert: PathBuf,
    pub key: PathBuf,
}

/// Serve until the process is signalled. `tls = None` listens plain HTTP.
/// Spawns the live-reload tick.
pub async fn serve(
    state: AppState,
    addr: SocketAddr,
    tls: Option<TlsConfig>,
) -> anyhow::Result<()> {
    spawn_reloader(state.clone(), std::time::Duration::from_secs(1));
    spawn_sweeper(state.clone(), std::time::Duration::from_secs(60));
    let app = build_router(state);
    let make = app.into_make_service_with_connect_info::<SocketAddr>();
    match tls {
        Some(tls) => {
            let config = axum_server::tls_rustls::RustlsConfig::from_pem_file(&tls.cert, &tls.key)
                .await
                .map_err(|e| anyhow::anyhow!("loading TLS cert/key: {e}"))?;
            log::info!("vecd: serving HTTPS on {addr}");
            axum_server::bind_rustls(addr, config).serve(make).await?;
        }
        None => {
            let listener = tokio::net::TcpListener::bind(addr).await?;
            log::info!("vecd: serving HTTP on {addr}");
            axum::serve(listener, make).await?;
        }
    }
    Ok(())
}

/// Serve a pre-bound listener with graceful shutdown — the entry point
/// tests use (they bind `127.0.0.1:0` to get a free port). Plain HTTP.
pub async fn serve_listener(
    state: AppState,
    listener: tokio::net::TcpListener,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    spawn_reloader(state.clone(), std::time::Duration::from_secs(1));
    let app = build_router(state);
    let make = app.into_make_service_with_connect_info::<SocketAddr>();
    axum::serve(listener, make).with_graceful_shutdown(shutdown).await?;
    Ok(())
}

#[cfg(test)]
mod range_tests {
    use super::{parse_byte_range, RangeOutcome};

    fn sat(spec: &str, total: u64) -> (u64, u64) {
        match parse_byte_range(spec, total) {
            RangeOutcome::Satisfiable { start, end } => (start, end),
            RangeOutcome::Full => panic!("expected satisfiable for {spec:?}, got Full"),
            RangeOutcome::Unsatisfiable => panic!("expected satisfiable for {spec:?}, got Unsatisfiable"),
        }
    }

    #[test]
    fn closed_range_is_inclusive() {
        // The exact slices the precache client requests for a multi-MB facet.
        assert_eq!(sat("bytes=0-1048575", 3525800), (0, 1048575));
        assert_eq!(sat("bytes=2097152-3145727", 3525800), (2097152, 3145727));
    }

    #[test]
    fn open_ended_range_runs_to_last_byte() {
        assert_eq!(sat("bytes=10-", 100), (10, 99));
    }

    #[test]
    fn suffix_range_takes_the_tail() {
        assert_eq!(sat("bytes=-20", 100), (80, 99));
        // A suffix larger than the object clamps to the whole object.
        assert_eq!(sat("bytes=-500", 100), (0, 99));
    }

    #[test]
    fn end_past_eof_clamps_to_last_byte() {
        assert_eq!(sat("bytes=90-1000", 100), (90, 99));
    }

    #[test]
    fn whitespace_is_tolerated() {
        assert_eq!(sat(" bytes=0-9 ", 100), (0, 9));
    }

    #[test]
    fn start_past_eof_is_unsatisfiable() {
        assert!(matches!(parse_byte_range("bytes=100-200", 100), RangeOutcome::Unsatisfiable));
        assert!(matches!(parse_byte_range("bytes=-0", 100), RangeOutcome::Unsatisfiable));
        assert!(matches!(parse_byte_range("bytes=0-0", 0), RangeOutcome::Unsatisfiable));
    }

    #[test]
    fn unsupported_or_garbage_falls_back_to_full() {
        // No `bytes=` unit, multi-range, and non-numeric all serve the full body.
        assert!(matches!(parse_byte_range("items=0-9", 100), RangeOutcome::Full));
        assert!(matches!(parse_byte_range("bytes=0-9,20-29", 100), RangeOutcome::Full));
        assert!(matches!(parse_byte_range("bytes=abc-def", 100), RangeOutcome::Full));
    }
}
