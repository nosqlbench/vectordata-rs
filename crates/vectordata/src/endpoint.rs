// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Client for a `vecd` endpoint's auth / introspection API — the calls
//! behind `vectordata login`, `ping`, and `token issue`/`revoke`. The AAA
//! endpoints live at the **server root**, so requests are addressed to the
//! origin (`scheme://host[:port]`) of whatever URL the user passes,
//! independent of any namespace path in it.
//!
//! Offline mode ([`settings::offline`](crate::settings::offline)) refuses
//! the introspection calls — [`whoami`] (behind `ping` and `whoami`) and
//! [`candidate_namespaces`] (behind completion and the post-login offer)
//! — before any request is made. The explicit server commands it does
//! not affect are `login` ([`login_password`]), `token issue` and
//! `revoke` ([`issue_token`], [`revoke_token`]) and `backup`'s reads
//! ([`versions`], [`manifest`], [`get_object`])
//! (`docs/design/srd-crate-boundaries.md` TR-12).

use serde::Deserialize;

use crate::credentials::origin_of_str;
use crate::transport::{Request, Response};

/// Send an introspection request through the shared transport, which
/// offline mode refuses, authorised only as the caller says: its own
/// token, or anonymous, never the read credentials.
fn send(req: Request, token: Option<&str>) -> Result<Response, crate::TransportError> {
    crate::transport::shared().send(req.bearer_or_anonymous(token))
}

/// Send an explicit server command's request (a login, a token issued
/// or revoked, a backup's read), which offline mode does not refuse;
/// authorised as [`send`] is.
fn send_command(req: Request, token: Option<&str>) -> Result<Response, crate::TransportError> {
    send(req.explicit(), token)
}

/// `req` with `body` as its JSON body.
fn with_json(req: Request, body: &serde_json::Value) -> Request {
    req.header("Content-Type", "application/json").bytes(body.to_string().into_bytes())
}

/// A response body parsed as JSON.
fn json<T: serde::de::DeserializeOwned>(resp: Response) -> Result<T, String> {
    let bytes = resp.bytes().map_err(|e| e.message)?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

/// A minted token as returned by `/auth/token` or `/tokens`.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenResp {
    /// The bearer token itself.
    pub token: String,
    /// Server-side token id, as passed to [`revoke_token`].
    #[serde(default)]
    pub id: Option<i64>,
    /// User the token was minted for.
    #[serde(default)]
    pub user: Option<String>,
    /// Expiry as Unix epoch seconds; `None` when the token does not
    /// expire or the server did not say.
    #[serde(default)]
    pub expires_at: Option<i64>,
}

/// The server-root API base (origin) for a user-supplied endpoint URL.
pub fn api_base(url: &str) -> Result<String, String> {
    origin_of_str(url).ok_or_else(|| format!("not a valid endpoint URL: {url}"))
}

/// `POST /auth/token` — exchange `{user, password}` for a bearer token.
pub fn login_password(
    url: &str,
    user: &str,
    password: &str,
    description: Option<&str>,
    expires: Option<&str>,
) -> Result<TokenResp, String> {
    let base = api_base(url)?;
    let mut body = serde_json::json!({ "user": user, "password": password });
    if let Some(d) = description {
        body["description"] = serde_json::json!(d);
    }
    if let Some(e) = expires {
        body["expires"] = serde_json::json!(e);
    }
    let resp = send_command(with_json(Request::post(format!("{base}/auth/token")), &body), None)
        .map_err(|e| format!("contacting {base}: {}", e.message))?;
    parse_token(resp, "login")
}

/// `GET /-/whoami` — the caller's effective access (JSON). `token = None`
/// reflects anonymous (`PUBLIC`) access. Refused in offline mode, with
/// the offline error's message.
pub fn whoami(url: &str, token: Option<&str>) -> Result<serde_json::Value, String> {
    let base = api_base(url)?;
    let resp = send(Request::get(format!("{base}/-/whoami")), token)
        .map_err(|e| format!("contacting {base}: {}", e.message))?;
    if resp.status() == 404 {
        return Err("not-a-vecd".to_string());
    }
    if !resp.is_success() {
        return Err(format!("{base}/-/whoami → HTTP {}", resp.status_line()));
    }
    json(resp).map_err(|e| format!("parsing whoami response: {e}"))
}

/// Namespace paths the caller could push to at `url`, best-effort.
///
/// Privileged callers (operator+) get the full *active* namespace list via
/// `GET /-/namespaces`; everyone else falls back to the namespaces `whoami`
/// reports a `write`/`publish` action for. Returns empty on any error (offline
/// mode, which makes no request; not-a-vecd; 403; …). A short per-request timeout makes this safe to call
/// from shell completion — it never hangs the prompt waiting on a dead server.
pub fn candidate_namespaces(url: &str, token: Option<&str>) -> Vec<String> {
    let Ok(base) = api_base(url) else { return Vec::new() };
    let get = |path: &str| -> Option<serde_json::Value> {
        let req = Request::get(format!("{base}{path}")).timeout(std::time::Duration::from_secs(4));
        let resp = send(req, token).ok()?;
        if !resp.is_success() {
            return None;
        }
        json(resp).ok()
    };

    // A privileged caller's `/-/namespaces` is the authoritative, backend-aware
    // list — use it even when it filters down to empty (e.g. every namespace is
    // backendless). Falling back to `whoami` here would re-suggest unwritable
    // namespaces, since `whoami` can't tell whether a backend is attached.
    if let Some(v) = get("/-/namespaces") {
        return active_namespaces_from(&v);
    }
    get("/-/whoami").map(|v| writable_namespaces_from(&v)).unwrap_or_default()
}

/// Active namespace paths from a `/-/namespaces` response (pure).
pub(crate) fn active_namespaces_from(view: &serde_json::Value) -> Vec<String> {
    view.get("namespaces")
        .and_then(|n| n.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|n| n.get("active").and_then(|a| a.as_bool()).unwrap_or(true))
                // A namespace with no storage backend can't be written — don't
                // suggest it (that's the "exists but 404s on write" trap).
                .filter(|n| n.get("backend_config").map(|b| !b.is_null()).unwrap_or(false))
                .filter_map(|n| n.get("path").and_then(|p| p.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Writable namespace paths from a `/-/whoami` response — those with a
/// `write` or `publish` effective action (pure).
pub(crate) fn writable_namespaces_from(view: &serde_json::Value) -> Vec<String> {
    view.get("namespaces")
        .and_then(|n| n.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|ns| {
                    let path = ns.get("path")?.as_str()?;
                    let actions = ns.get("actions")?.as_array()?;
                    let can_write = actions
                        .iter()
                        .filter_map(|a| a.as_str())
                        .any(|a| a == "write" || a == "publish");
                    can_write.then(|| path.to_string())
                })
                .collect()
        })
        .unwrap_or_default()
}

/// `POST /tokens` — mint a delegated key for the authenticated caller,
/// optionally narrowed by a profile spec.
pub fn issue_token(
    url: &str,
    session_token: &str,
    description: &str,
    profile: Option<&str>,
    expires: Option<&str>,
) -> Result<TokenResp, String> {
    let base = api_base(url)?;
    let mut body = serde_json::json!({ "description": description });
    if let Some(p) = profile {
        body["profile"] = serde_json::json!(p);
    }
    if let Some(e) = expires {
        body["expires"] = serde_json::json!(e);
    }
    let resp = send_command(with_json(Request::post(format!("{base}/tokens")), &body), Some(session_token))
        .map_err(|e| format!("contacting {base}: {}", e.message))?;
    parse_token(resp, "token issue")
}

/// `DELETE /tokens/<id>` — revoke a token the caller issued (or, for an
/// admin, any token).
pub fn revoke_token(url: &str, session_token: &str, id: i64) -> Result<(), String> {
    let base = api_base(url)?;
    let resp = send_command(Request::delete(format!("{base}/tokens/{id}")), Some(session_token))
        .map_err(|e| format!("contacting {base}: {}", e.message))?;
    if resp.is_success() {
        Ok(())
    } else {
        Err(format!("revoking token {id} → HTTP {}", resp.status_line()))
    }
}

/// `GET /-/versions/<ns>` — a namespace's version history (JSON).
pub fn versions(url: &str, ns: &str, token: Option<&str>) -> Result<serde_json::Value, String> {
    let base = api_base(url)?;
    json(authed_get(&format!("{base}/-/versions/{ns}"), token)?).map_err(|e| format!("parsing versions: {e}"))
}

/// `GET /<ns>/@<selector>/-/manifest` — a version's manifest (JSON).
pub fn manifest(url: &str, ns: &str, selector: &str, token: Option<&str>) -> Result<serde_json::Value, String> {
    let base = api_base(url)?;
    json(authed_get(&format!("{base}/{ns}/@{selector}/-/manifest"), token)?)
        .map_err(|e| format!("parsing manifest: {e}"))
}

/// `GET /<ns>/@<selector>/<key>` — a version's object bytes.
pub fn get_object(
    url: &str,
    ns: &str,
    selector: &str,
    key: &str,
    token: Option<&str>,
) -> Result<Vec<u8>, String> {
    let base = api_base(url)?;
    let resp = authed_get(&format!("{base}/{ns}/@{selector}/{key}"), token)?;
    resp.bytes().map_err(|e| format!("reading object body: {}", e.message))
}

/// Authenticated GET that errors on non-2xx: a read of `backup`'s, an
/// explicit server command.
fn authed_get(url: &str, token: Option<&str>) -> Result<Response, String> {
    let resp = send_command(Request::get(url), token).map_err(|e| format!("GET {url}: {}", e.message))?;
    if resp.is_success() {
        Ok(resp)
    } else {
        Err(format!("GET {url} → HTTP {}", resp.status_line()))
    }
}

fn parse_token(resp: Response, what: &str) -> Result<TokenResp, String> {
    if resp.status() == 401 {
        return Err("authentication failed (check credentials)".to_string());
    }
    if !resp.is_success() {
        let status = resp.status_line();
        let body = resp.text().unwrap_or_default();
        return Err(format!("{what} failed (HTTP {status}): {}", body.trim()));
    }
    json::<TokenResp>(resp).map_err(|e| format!("parsing {what} response: {e}"))
}

#[cfg(test)]
mod namespace_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn active_namespaces_filters_inactive_and_backendless() {
        let v = json!({"namespaces": [
            {"path": "datasets", "active": true,  "backend_config": "store"},
            {"path": "archive",  "active": false, "backend_config": "store"}, // inactive
            {"path": "general",  "active": true,  "backend_config": null},     // no backend → unwritable
            {"path": "scratch",  "backend_config": "store"},                   // no `active` ⇒ assume active
        ]});
        assert_eq!(active_namespaces_from(&v), vec!["datasets".to_string(), "scratch".to_string()]);
        assert!(active_namespaces_from(&json!({})).is_empty());
    }

    #[test]
    fn writable_namespaces_from_whoami() {
        let v = json!({"namespaces": [
            {"path": "datasets", "actions": ["read", "write"]},
            {"path": "pub",      "actions": ["publish"]},
            {"path": "ro",       "actions": ["read"]},
        ]});
        assert_eq!(writable_namespaces_from(&v), vec!["datasets".to_string(), "pub".to_string()]);
        assert!(writable_namespaces_from(&json!({})).is_empty());
    }
}
