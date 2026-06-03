// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Client for a `vecd` endpoint's auth / introspection API — the calls
//! behind `vectordata login`, `ping`, and `token issue`/`revoke`. The AAA
//! endpoints live at the **server root**, so requests are addressed to the
//! origin (`scheme://host[:port]`) of whatever URL the user passes,
//! independent of any namespace path in it.

use serde::Deserialize;

use crate::credentials::origin_of_str;
use crate::transport::shared_client;

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
    let resp = shared_client()
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
    let rb = shared_client().get(format!("{base}/-/whoami"));
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
    let resp = shared_client()
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
    let resp = shared_client()
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
    let rb = shared_client().get(url);
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
