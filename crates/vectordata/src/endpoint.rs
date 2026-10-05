// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Client for a `vecd` endpoint's auth / introspection API — the calls
//! behind `vectordata login`, `ping`, and `token issue`/`revoke`. The AAA
//! endpoints live at the **server root**, so requests are addressed to the
//! origin (`scheme://host[:port]`) of whatever URL the user passes,
//! independent of any namespace path in it.

use serde::Deserialize;

use crate::credentials::origin_of_str;
use crate::transport::shared_client_for;

/// A minted token as returned by `/auth/token` or `/tokens`.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenResp {
    pub token: String,
    #[serde(default)]
    pub id: Option<i64>,
    #[serde(default)]
    pub user: Option<String>,
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
    let resp = shared_client_for(&base)
        .post(format!("{base}/auth/token"))
        .json(&body)
        .send()
        .map_err(|e| format!("contacting {base}: {e}"))?;
    parse_token(resp, "login")
}

/// `GET /-/whoami` — the caller's effective access (JSON). `token = None`
/// reflects anonymous (`PUBLIC`) access.
pub fn whoami(url: &str, token: Option<&str>) -> Result<serde_json::Value, String> {
    let base = api_base(url)?;
    let rb = shared_client_for(&base).get(format!("{base}/-/whoami"));
    let rb = match token {
        Some(t) => rb.bearer_auth(t),
        None => rb,
    };
    let resp = rb.send().map_err(|e| format!("contacting {base}: {e}"))?;
    let status = resp.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err("not-a-vecd".to_string());
    }
    if !status.is_success() {
        return Err(format!("{base}/-/whoami → HTTP {status}"));
    }
    resp.json().map_err(|e| format!("parsing whoami response: {e}"))
}

/// Namespace paths the caller could push to at `url`, best-effort.
///
/// Privileged callers (operator+) get the full *active* namespace list via
/// `GET /-/namespaces`; everyone else falls back to the namespaces `whoami`
/// reports a `write`/`publish` action for. Returns empty on any error (offline,
/// not-a-vecd, 403, …). A short per-request timeout makes this safe to call
/// from shell completion — it never hangs the prompt waiting on a dead server.
pub fn candidate_namespaces(url: &str, token: Option<&str>) -> Vec<String> {
    let Ok(base) = api_base(url) else { return Vec::new() };
    let get = |path: &str| -> Option<serde_json::Value> {
        let rb = shared_client_for(&base)
            .get(format!("{base}{path}"))
            .timeout(std::time::Duration::from_secs(4));
        let rb = match token {
            Some(t) => rb.bearer_auth(t),
            None => rb,
        };
        let resp = rb.send().ok()?;
        if !resp.status().is_success() {
            return None;
        }
        resp.json().ok()
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
    let resp = shared_client_for(&base)
        .post(format!("{base}/tokens"))
        .bearer_auth(session_token)
        .json(&body)
        .send()
        .map_err(|e| format!("contacting {base}: {e}"))?;
    parse_token(resp, "token issue")
}

/// `DELETE /tokens/<id>` — revoke a token the caller issued (or, for an
/// admin, any token).
pub fn revoke_token(url: &str, session_token: &str, id: i64) -> Result<(), String> {
    let base = api_base(url)?;
    let resp = shared_client_for(&base)
        .delete(format!("{base}/tokens/{id}"))
        .bearer_auth(session_token)
        .send()
        .map_err(|e| format!("contacting {base}: {e}"))?;
    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("revoking token {id} → HTTP {}", resp.status()))
    }
}

/// `GET /-/versions/<ns>` — a namespace's version history (JSON).
pub fn versions(url: &str, ns: &str, token: Option<&str>) -> Result<serde_json::Value, String> {
    let base = api_base(url)?;
    authed_get(&format!("{base}/-/versions/{ns}"), token)?.json().map_err(|e| format!("parsing versions: {e}"))
}

/// `GET /<ns>/@<selector>/-/manifest` — a version's manifest (JSON).
pub fn manifest(url: &str, ns: &str, selector: &str, token: Option<&str>) -> Result<serde_json::Value, String> {
    let base = api_base(url)?;
    authed_get(&format!("{base}/{ns}/@{selector}/-/manifest"), token)?
        .json()
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
    Ok(resp.bytes().map_err(|e| format!("reading object body: {e}"))?.to_vec())
}

/// Authenticated GET that errors on non-2xx.
fn authed_get(url: &str, token: Option<&str>) -> Result<reqwest::blocking::Response, String> {
    let rb = shared_client_for(url).get(url);
    let rb = match token {
        Some(t) => rb.bearer_auth(t),
        None => rb,
    };
    let resp = rb.send().map_err(|e| format!("GET {url}: {e}"))?;
    if resp.status().is_success() {
        Ok(resp)
    } else {
        Err(format!("GET {url} → HTTP {}", resp.status()))
    }
}

fn parse_token(resp: reqwest::blocking::Response, what: &str) -> Result<TokenResp, String> {
    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err("authentication failed (check credentials)".to_string());
    }
    if !status.is_success() {
        let body = resp.text().unwrap_or_default();
        return Err(format!("{what} failed (HTTP {status}): {}", body.trim()));
    }
    resp.json::<TokenResp>().map_err(|e| format!("parsing {what} response: {e}"))
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
