// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Per-endpoint client credentials and bearer-token resolution.
//!
//! `vectordata login <url>` stores a bearer token bound to the request's
//! **origin** (`scheme://host[:port]`) in
//! `~/.config/vectordata/credentials.toml` (mode `0600`). Push and pull
//! then resolve a token transparently — no `--token` on every call — with
//! this precedence (first present wins):
//!
//! 1. an explicit `--token` / call-site override;
//! 2. `$VECTORDATA_PUSH_TOKEN` (push) / `$VECTORDATA_TOKEN`;
//! 3. the stored credential for the request's origin;
//! 4. none → anonymous.
//!
//! Public datasets keep working with no token at all — token resolution is
//! purely additive: absent everywhere, the request stays anonymous.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use url::Url;

/// Config directory, mirroring `settings.rs`: `$VECTORDATA_HOME`, else
/// `$HOME/.config/vectordata`. Tests isolate via `$VECTORDATA_HOME`.
fn config_dir() -> PathBuf {
    if let Some(root) = std::env::var_os("VECTORDATA_HOME") {
        PathBuf::from(root)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".config/vectordata")
    } else {
        PathBuf::from(".config/vectordata")
    }
}

/// Path to the credentials file.
pub fn credentials_path() -> PathBuf {
    config_dir().join("credentials.toml")
}

/// One stored credential.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    /// The bearer secret.
    pub token: String,
    /// Informational: the user the token was minted for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Informational: the token's expiry as epoch seconds (a stringified
    /// `i64`), if the server reported one at login. Used to warn before a
    /// credential lapses rather than surprising the user with a 401.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires: Option<String>,
}

/// A stored credential's standing relative to a reference time.
#[derive(Debug, PartialEq, Eq)]
pub enum Expiry {
    /// No expiry was recorded (or it didn't parse).
    Unknown,
    /// Still valid, with this many seconds left.
    Active { secs_left: i64 },
    /// Already lapsed, this many seconds ago.
    Expired { secs_ago: i64 },
}

impl Entry {
    /// Classify this credential's expiry against `now` (epoch seconds).
    /// Pure — the time is a parameter so callers/tests don't depend on a clock.
    pub fn expiry_status(&self, now: i64) -> Expiry {
        match self.expires.as_deref().and_then(|s| s.trim().parse::<i64>().ok()) {
            None => Expiry::Unknown,
            Some(exp) if exp <= now => Expiry::Expired { secs_ago: now - exp },
            Some(exp) => Expiry::Active { secs_left: exp - now },
        }
    }
}

/// The credential store — a map of origin → [`Entry`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Store {
    #[serde(flatten)]
    entries: BTreeMap<String, Entry>,
}

impl Store {
    /// Load the store, or an empty one if the file is absent/unreadable.
    pub fn load() -> Store {
        match std::fs::read_to_string(credentials_path()) {
            Ok(text) => toml::from_str(&text).unwrap_or_default(),
            Err(_) => Store::default(),
        }
    }

    /// Persist the store (creating the config dir `0700`, the file `0600`).
    pub fn save(&self) -> std::io::Result<()> {
        let dir = config_dir();
        std::fs::create_dir_all(&dir)?;
        set_mode(&dir, 0o700);
        let path = credentials_path();
        let text = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&path, text)?;
        set_mode(&path, 0o600);
        Ok(())
    }

    pub fn get(&self, origin: &str) -> Option<&Entry> {
        self.entries.get(origin)
    }

    /// The credential governing `url`, by **longest-prefix** match over the
    /// stored keys: an exact match, else the most specific stored key that is a
    /// path-prefix of `url`. This is what lets a per-catalog credential (a token
    /// scoped to `https://host/datasets`) win over an origin-wide one
    /// (`https://host`) for reads under that catalog, while still covering the
    /// rest of the origin from the origin key.
    pub fn token_for_url(&self, url: &str) -> Option<&Entry> {
        let q = credential_key(url)?;
        self.entries
            .iter()
            .filter(|(k, _)| q == **k || q.starts_with(&format!("{k}/")))
            .max_by_key(|(k, _)| k.len())
            .map(|(_, e)| e)
    }

    pub fn set(&mut self, origin: String, entry: Entry) {
        self.entries.insert(origin, entry);
    }

    /// Remove an origin's credential; returns whether one was present.
    pub fn remove(&mut self, origin: &str) -> bool {
        self.entries.remove(origin).is_some()
    }

    /// (origin, user) for every stored credential.
    pub fn list(&self) -> Vec<(String, Option<String>)> {
        self.entries.iter().map(|(o, e)| (o.clone(), e.user.clone())).collect()
    }
}

/// The origin (`scheme://host[:port]`) of a URL — the credential key.
pub fn origin_of(url: &Url) -> Option<String> {
    let scheme = url.scheme();
    let host = url.host_str()?;
    Some(match url.port() {
        Some(p) => format!("{scheme}://{host}:{p}"),
        None => format!("{scheme}://{host}"),
    })
}

/// Parse a URL string and return its origin.
pub fn origin_of_str(url: &str) -> Option<String> {
    Url::parse(url).ok().as_ref().and_then(origin_of)
}

/// The credential-store key for a URL: its origin plus any non-root path, with
/// trailing slashes trimmed. `https://h/` and `https://h` both key `https://h`
/// (origin-wide); `https://h/datasets/` keys `https://h/datasets` (catalog-
/// scoped). Used both to **record** a login and (via [`Store::token_for_url`])
/// to **match** a read to the most specific credential.
pub fn credential_key(url: &str) -> Option<String> {
    let u = Url::parse(url).ok()?;
    let origin = origin_of(&u)?;
    let path = u.path().trim_end_matches('/');
    Some(if path.is_empty() { origin } else { format!("{origin}{path}") })
}

/// The cached store for this process (so per-request reads don't re-read
/// the file). A short-lived CLI invocation loads it once.
fn cached_store() -> &'static Store {
    static STORE: OnceLock<Store> = OnceLock::new();
    STORE.get_or_init(Store::load)
}

/// Resolve a bearer token for a **read/pull** to `url`: `$VECTORDATA_TOKEN`,
/// then the credential recorded at `vectordata login` for the URL's origin,
/// else `None` (anonymous).
pub fn resolve_read_token(url: &Url) -> Option<String> {
    let env = std::env::var("VECTORDATA_TOKEN").ok().filter(|t| !t.is_empty());
    resolve_read_token_with(env.as_deref(), url, cached_store())
}

/// Pure core of [`resolve_read_token`]: `$VECTORDATA_TOKEN` (passed in) wins,
/// else the `store` credential keyed by the URL's origin. Separated so the
/// precedence can be tested without touching process env or the on-disk store.
fn resolve_read_token_with(env_token: Option<&str>, url: &Url, store: &Store) -> Option<String> {
    if let Some(t) = env_token {
        return Some(t.to_string());
    }
    store.token_for_url(url.as_str()).map(|e| e.token.clone())
}

/// Resolve a token for `url` from the store only (no env), by longest-prefix —
/// used by the CLI commands that act on a specific endpoint or catalog.
pub fn stored_token(url: &str) -> Option<String> {
    Store::load().token_for_url(url).map(|e| e.token.clone())
}

/// The endpoint a command should act on: the given `url`, else the sole
/// logged-in endpoint's origin — so commands "just work" while logged in.
/// Errors (asking for an explicit `<url>`) when none or several are stored.
pub fn resolve_endpoint(url: Option<&str>) -> Result<String, String> {
    if let Some(u) = url {
        return Ok(u.to_string());
    }
    let entries = Store::load().list();
    match entries.len() {
        0 => Err(
            "not logged in to any endpoint — give a <url>, or run `vectordata login <url>` first"
                .to_string(),
        ),
        1 => Ok(entries.into_iter().next().unwrap().0),
        _ => Err(format!(
            "logged in to several endpoints — specify one as <url>: {}",
            entries.iter().map(|(o, _)| o.as_str()).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// A bearer token resolved from a `--token` argument, with any user/expiry the
/// source carried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedToken {
    pub token: String,
    /// The user the token is for, if the source named it (a JSON token record).
    pub user: Option<String>,
    /// Expiry as epoch-seconds (stringified), if the source carried it.
    pub expires: Option<String>,
}

/// Resolve a `--token` argument that may be a **literal token**, or a path to a
/// **file**. Recognized file shapes:
///   * a JSON token record `{ "token", "user", "expires_at", … }` (as emitted
///     by `vecd tokens create --json`);
///   * a credential store `{ "<origin>": { "token", "user", "expires" }, … }`
///     (e.g. `~/.config/vecd/credentials.json`) — the entry for `for_url`'s
///     origin is used (or the sole entry when there's just one);
///   * a bare token string on its own.
/// A value that isn't an existing file is taken verbatim.
pub fn resolve_token_arg(value: &str, for_url: Option<&str>) -> Result<ResolvedToken, String> {
    let path = std::path::Path::new(value);
    if !path.is_file() {
        // Literal token string.
        return Ok(ResolvedToken { token: value.to_string(), user: None, expires: None });
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("reading token file {value}: {e}"))?;
    let trimmed = text.trim();
    if trimmed.starts_with('{') {
        let v: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| format!("parsing token JSON {value}: {e}"))?;
        // A single token record carries a top-level "token".
        if let Some(token) = v.get("token").and_then(|t| t.as_str()) {
            return Ok(ResolvedToken {
                token: token.to_string(),
                user: v.get("user").and_then(|u| u.as_str()).map(String::from),
                expires: v.get("expires_at").and_then(epoch_str),
            });
        }
        // Otherwise it's a credential store keyed by endpoint origin.
        if let Some(obj) = v.as_object() {
            return token_from_store(obj, value, for_url);
        }
        return Err(format!("token file {value}: missing a \"token\" field"));
    }
    // A bare-token file (just the secret on a line).
    if trimmed.is_empty() {
        return Err(format!("token file {value} is empty"));
    }
    Ok(ResolvedToken { token: trimmed.to_string(), user: None, expires: None })
}

/// Pull one entry out of a credential-store map: the one matching `for_url`'s
/// origin, or the sole entry when the URL is omitted and there's just one.
fn token_from_store(
    obj: &serde_json::Map<String, serde_json::Value>,
    file: &str,
    for_url: Option<&str>,
) -> Result<ResolvedToken, String> {
    let entry = match for_url {
        Some(url) => {
            let origin =
                origin_of_str(url).ok_or_else(|| format!("not a valid endpoint URL: {url}"))?;
            obj.get(&origin).ok_or_else(|| {
                let have = obj.keys().cloned().collect::<Vec<_>>().join(", ");
                format!("{file} has no credential for {origin} (it holds: {have})")
            })?
        }
        None if obj.len() == 1 => obj.values().next().unwrap(),
        None => return Err(format!("{file} holds several credentials — specify the endpoint URL")),
    };
    let token = entry
        .get("token")
        .and_then(|t| t.as_str())
        .ok_or_else(|| format!("token file {file}: the matching entry has no \"token\" field"))?
        .to_string();
    Ok(ResolvedToken {
        token,
        user: entry.get("user").and_then(|u| u.as_str()).map(String::from),
        expires: entry.get("expires").and_then(epoch_str),
    })
}

/// A JSON number or string epoch rendered as a string.
fn epoch_str(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// Credentials within this window of expiry get a heads-up warning.
const EXPIRY_WARN_SECS: i64 = 7 * 24 * 3600;

/// Print a stderr warning if the stored credential for `url`'s origin is past
/// expiry, or within [`EXPIRY_WARN_SECS`] of it — so a lapsing token surfaces
/// as a clear "go re-login" rather than an opaque 401 later. No-op when there
/// is no stored credential or it carries no expiry. Call it from commands that
/// rely on a stored credential.
pub fn warn_if_expiring(url: &str) {
    warn_if_expiring_in(url, now_secs(), &Store::load());
}

/// Testable core of [`warn_if_expiring`] (time + store injected).
fn warn_if_expiring_in(url: &str, now: i64, store: &Store) {
    let Some(origin) = origin_of_str(url) else { return };
    let Some(entry) = store.get(&origin) else { return };
    match entry.expiry_status(now) {
        Expiry::Expired { .. } => eprintln!(
            "warning: your stored credential for {origin} has expired — \
             run `vectordata login {url}` to refresh it."
        ),
        Expiry::Active { secs_left } if secs_left <= EXPIRY_WARN_SECS => {
            // Round up to whole days (secs_left > 0 in the Active arm).
            let days = (secs_left + 24 * 3600 - 1) / (24 * 3600);
            eprintln!(
                "warning: your stored credential for {origin} expires in ~{days} day(s) — \
                 run `vectordata login {url}` to refresh it."
            );
        }
        _ => {}
    }
}

/// Current time in epoch seconds.
fn now_secs() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(unix)]
fn set_mode(path: &std::path::Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode));
}
#[cfg(not(unix))]
fn set_mode(_path: &std::path::Path, _mode: u32) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_extraction() {
        let u = Url::parse("https://vecd-host:8443/datasets/glove/base.fvec").unwrap();
        assert_eq!(origin_of(&u).unwrap(), "https://vecd-host:8443");
        let u2 = Url::parse("https://example.com/x").unwrap();
        assert_eq!(origin_of(&u2).unwrap(), "https://example.com");
    }

    #[test]
    fn read_token_prefers_env_then_login_store_by_origin() {
        // A credential as `vectordata login` would record it: keyed by origin.
        let mut store = Store::default();
        store.set(
            "https://vecd-host:8443".to_string(),
            Entry { token: "login-tok".to_string(), user: Some("alice".to_string()), expires: None },
        );
        // A facet URL under that origin — what the access API reads.
        let facet =
            Url::parse("https://vecd-host:8443/datasets/toy/profiles/base/base.fvecs").unwrap();

        // $VECTORDATA_TOKEN wins when present.
        assert_eq!(
            resolve_read_token_with(Some("env-tok"), &facet, &store).as_deref(),
            Some("env-tok")
        );
        // Otherwise the token recorded at login, matched by the URL's origin —
        // so reads to that endpoint authenticate with the login credential.
        assert_eq!(
            resolve_read_token_with(None, &facet, &store).as_deref(),
            Some("login-tok")
        );
        // An origin we never logged into → anonymous (no token attached).
        let other = Url::parse("https://elsewhere:9000/x/y.fvecs").unwrap();
        assert_eq!(resolve_read_token_with(None, &other, &store), None);
    }

    #[test]
    fn per_catalog_token_wins_by_longest_prefix() {
        // Two credentials at the same origin: one origin-wide, one scoped to a
        // catalog URL (as `vectordata login https://h:8443/datasets/` records).
        let mut store = Store::default();
        store.set(
            "https://h:8443".to_string(),
            Entry { token: "origin-tok".into(), user: None, expires: None },
        );
        store.set(
            "https://h:8443/datasets".to_string(),
            Entry { token: "cat-tok".into(), user: None, expires: None },
        );

        // A read under the catalog path → the catalog-scoped token (most specific).
        let under = Url::parse("https://h:8443/datasets/toy/profiles/base/base.fvecs").unwrap();
        assert_eq!(resolve_read_token_with(None, &under, &store).as_deref(), Some("cat-tok"));
        // A read elsewhere at the origin → the origin-wide token.
        let other = Url::parse("https://h:8443/private/x.fvecs").unwrap();
        assert_eq!(resolve_read_token_with(None, &other, &store).as_deref(), Some("origin-tok"));

        // A key that only shares a string prefix must NOT match (segment boundary).
        let mut store2 = Store::default();
        store2.set(
            "https://h:8443/data".to_string(),
            Entry { token: "data-tok".into(), user: None, expires: None },
        );
        let sibling = Url::parse("https://h:8443/datasets/x.fvecs").unwrap();
        assert_eq!(
            resolve_read_token_with(None, &sibling, &store2),
            None,
            "`/data` must not match `/datasets`"
        );
    }

    #[test]
    fn resolve_token_literal_vs_file() {
        // A value that isn't a file is taken verbatim.
        assert_eq!(
            resolve_token_arg("vd_abc123", None).unwrap(),
            ResolvedToken { token: "vd_abc123".into(), user: None, expires: None }
        );
        let dir = tempfile::tempdir().unwrap();
        // JSON token record (as `vecd tokens create --json` emits).
        let jpath = dir.path().join("tok.json");
        std::fs::write(&jpath, r#"{"token":"vd_xyz","user":"alice","id":7,"expires_at":1893456000}"#).unwrap();
        assert_eq!(
            resolve_token_arg(jpath.to_str().unwrap(), None).unwrap(),
            ResolvedToken { token: "vd_xyz".into(), user: Some("alice".into()), expires: Some("1893456000".into()) }
        );
        // Bare-token file (just the secret).
        let bpath = dir.path().join("tok.txt");
        std::fs::write(&bpath, "vd_bare\n").unwrap();
        assert_eq!(
            resolve_token_arg(bpath.to_str().unwrap(), None).unwrap(),
            ResolvedToken { token: "vd_bare".into(), user: None, expires: None }
        );
        // A credential store (origin → entry), e.g. ~/.config/vecd/credentials.json:
        // pick the entry for the target url's origin.
        let spath = dir.path().join("credentials.json");
        std::fs::write(&spath, r#"{"http://h:8443":{"token":"vd_store","user":"root","expires":"1893456000"},"https://other":{"token":"vd_o","user":"bob","expires":null}}"#).unwrap();
        assert_eq!(
            resolve_token_arg(spath.to_str().unwrap(), Some("http://h:8443/")).unwrap(),
            ResolvedToken { token: "vd_store".into(), user: Some("root".into()), expires: Some("1893456000".into()) }
        );
        // Store + a url whose origin isn't present → error.
        assert!(resolve_token_arg(spath.to_str().unwrap(), Some("http://nope:1/")).is_err());
        // Store + no url, several entries → ambiguous error.
        assert!(resolve_token_arg(spath.to_str().unwrap(), None).is_err());
        // JSON with neither a top-level token nor a usable entry is an error.
        let npath = dir.path().join("bad.json");
        std::fs::write(&npath, r#"{"user":"alice"}"#).unwrap();
        assert!(resolve_token_arg(npath.to_str().unwrap(), None).is_err());
    }

    #[test]
    fn expiry_status_classifies_against_now() {
        let mk = |exp: Option<&str>| Entry {
            token: "t".into(),
            user: None,
            expires: exp.map(str::to_string),
        };
        // now = 1_000_000.
        assert_eq!(mk(None).expiry_status(1_000_000), Expiry::Unknown);
        assert_eq!(mk(Some("not-a-number")).expiry_status(1_000_000), Expiry::Unknown);
        assert_eq!(
            mk(Some("1000500")).expiry_status(1_000_000),
            Expiry::Active { secs_left: 500 }
        );
        assert_eq!(
            mk(Some("999000")).expiry_status(1_000_000),
            Expiry::Expired { secs_ago: 1000 }
        );
        // Exactly now counts as expired (boundary).
        assert_eq!(
            mk(Some("1000000")).expiry_status(1_000_000),
            Expiry::Expired { secs_ago: 0 }
        );
    }

    #[test]
    fn store_set_get_remove_roundtrip() {
        let mut s = Store::default();
        s.set(
            "https://h:1".into(),
            Entry { token: "vd_abc".into(), user: Some("alice".into()), expires: None },
        );
        assert_eq!(s.get("https://h:1").unwrap().token, "vd_abc");
        assert_eq!(s.list().len(), 1);
        assert!(s.remove("https://h:1"));
        assert!(!s.remove("https://h:1"));
    }

    #[test]
    fn toml_roundtrip_quotes_origin_keys() {
        let mut s = Store::default();
        s.set(
            "https://vecd-host:8443".into(),
            Entry { token: "vd_x".into(), user: Some("bob".into()), expires: Some("2026-01-01T00:00:00Z".into()) },
        );
        let text = toml::to_string_pretty(&s).unwrap();
        let back: Store = toml::from_str(&text).unwrap();
        assert_eq!(back.get("https://vecd-host:8443").unwrap().token, "vd_x");
    }
}
