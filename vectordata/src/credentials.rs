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
    /// Informational name-tag: the configured catalog this credential was
    /// provided for, so the explorer can join credentials to catalogs
    /// pairwise by name. Resolution still matches by URL (this field is
    /// never consulted for that) — it only labels the entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog: Option<String>,
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
            .map_err(std::io::Error::other)?;
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
/// the file). A short-lived CLI invocation loads it once; a long-running
/// one (the explorer) refreshes it via [`reload_credentials`] after
/// writing a credential, so reads pick up the change without a restart.
fn store_cache() -> &'static std::sync::RwLock<Store> {
    static STORE: OnceLock<std::sync::RwLock<Store>> = OnceLock::new();
    STORE.get_or_init(|| std::sync::RwLock::new(Store::load()))
}

/// Re-read the process credential cache from disk. Call after a write so
/// subsequent [`resolve_read_token`] calls in the same process see the
/// new credential (the interactive explorer relies on this; short-lived
/// CLI runs never need it).
pub fn reload_credentials() {
    if let Ok(mut g) = store_cache().write() {
        *g = Store::load();
    }
}

/// Resolve a bearer token for a **read/pull** to `url`: `$VECTORDATA_TOKEN`,
/// then the credential recorded at `vectordata login` for the URL's origin,
/// else `None` (anonymous).
pub fn resolve_read_token(url: &Url) -> Option<String> {
    if let Some(t) = std::env::var("VECTORDATA_TOKEN").ok().filter(|t| !t.is_empty()) {
        return Some(t);
    }
    let guard = store_cache().read().ok()?;
    resolve_read_token_with(None, url, &guard)
}

/// The stored credential governing `url` (longest-prefix match), loaded
/// fresh from disk — for the explorer's auth indicator and detail. None
/// means no credential is recorded for that URL.
pub fn credential_for_url(url: &str) -> Option<Entry> {
    Store::load().token_for_url(url).cloned()
}

/// Record a bearer credential for a catalog: stored under the catalog
/// URL's [`credential_key`] (so catalog fetch, verify, and every facet
/// read authenticate by URL) and tagged with the catalog `name` for the
/// pairwise join in the config view. Persists to `credentials.toml`
/// (`0600`) and refreshes the process cache so it takes effect at once.
pub fn set_catalog_credential(
    name: &str, url: &str, token: &str, user: Option<&str>,
) -> Result<(), String> {
    let key = credential_key(url).ok_or_else(|| format!("not a valid catalog URL: {url}"))?;
    let mut store = Store::load();
    store.set(key, Entry {
        token: token.to_string(),
        user: user.map(str::to_string).filter(|u| !u.is_empty()),
        expires: None,
        catalog: Some(name.to_string()),
    });
    store.save().map_err(|e| format!("saving {}: {e}", credentials_path().display()))?;
    reload_credentials();
    Ok(())
}

/// Whether a stored credential key and a catalog's [`credential_key`] lie on
/// the same segment-aligned path chain — i.e. the credential governs that
/// catalog (origin-wide or exact) or is scoped under it. Mirrors the
/// longest-prefix logic in [`Store::token_for_url`], so "the credential serves
/// this catalog" and "the credential is justified by this catalog" agree.
fn keys_on_same_chain(cred: &str, cat: &str) -> bool {
    cred == cat
        || cat.starts_with(&format!("{cred}/"))
        || cred.starts_with(&format!("{cat}/"))
}

/// Pure core of [`prune_orphan_credentials`]: drop every entry whose key is
/// justified by none of `catalog_keys`, returning the removed keys. Separated
/// so the matching rule is testable without the on-disk store or catalogs.
fn prune_orphans_in(store: &mut Store, catalog_keys: &[String]) -> Vec<String> {
    let orphans: Vec<String> = store
        .entries
        .keys()
        .filter(|k| !catalog_keys.iter().any(|c| keys_on_same_chain(k, c)))
        .cloned()
        .collect();
    for k in &orphans {
        store.entries.remove(k);
    }
    orphans
}

/// Reconcile `credentials.toml` with the configured catalogs: drop every stored
/// credential that corresponds (by URL, longest-prefix) to no configured
/// catalog — so the file only holds catalog-justified entries. Catalogs without
/// a URL key (local paths) contribute nothing to match against. Persists +
/// refreshes the process cache only when something changed. Returns the number
/// of credentials removed. Called after an explicit catalog **removal** (the op
/// that orphans a credential); never on add/login, so a credential created by
/// `login` before its catalog exists is left alone.
pub fn prune_orphan_credentials() -> usize {
    let cats = crate::catalog::sources::named_catalog_entries(
        &crate::catalog::sources::config_dir());
    let keys: Vec<String> = cats.iter().filter_map(|c| credential_key(&c.location)).collect();
    let mut store = Store::load();
    let removed = prune_orphans_in(&mut store, &keys);
    if !removed.is_empty() {
        let _ = store.save();
        reload_credentials();
    }
    removed.len()
}

/// Remove any credential stored for a catalog URL. Returns whether one
/// was present. Refreshes the process cache on removal.
pub fn clear_catalog_credential(url: &str) -> Result<bool, String> {
    let Some(key) = credential_key(url) else { return Ok(false) };
    let mut store = Store::load();
    let removed = store.remove(&key);
    if removed {
        store.save().map_err(|e| format!("saving {}: {e}", credentials_path().display()))?;
        reload_credentials();
    }
    Ok(removed)
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

/// The endpoint a command should act on: the given specifier (resolved by
/// [`resolve_endpoint_spec`] — a URL, or a configured catalog name/index),
/// else the sole logged-in endpoint's origin — so commands "just work" while
/// logged in. Errors (asking for an explicit endpoint) when none or several
/// are stored.
pub fn resolve_endpoint(url: Option<&str>) -> Result<String, String> {
    if let Some(u) = url {
        return resolve_endpoint_spec(u);
    }
    let entries = Store::load().list();
    match entries.len() {
        0 => Err(
            "not logged in to any endpoint — give a <url> (or a configured catalog name/index), \
             or run `vectordata login <url>` first"
                .to_string(),
        ),
        1 => Ok(entries.into_iter().next().unwrap().0),
        _ => Err(format!(
            "logged in to several endpoints — specify one as a <url>, or a configured catalog \
             name/index: {}",
            entries.iter().map(|(o, _)| o.as_str()).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Resolve an endpoint specifier to a URL: a URL (`scheme://…`) is taken
/// verbatim; otherwise it is looked up among the **configured catalogs**
/// (catalogs.yaml) as a NAME or a 1-based INDEX and resolved to that
/// catalog's location — so `vectordata logout protected` or `… 2` work
/// instead of pasting the full URL.
pub fn resolve_endpoint_spec(spec: &str) -> Result<String, String> {
    let cats = crate::catalog::sources::named_catalog_entries(
        &crate::catalog::sources::config_dir());
    resolve_endpoint_spec_in(spec, &cats)
}

/// Pure core of [`resolve_endpoint_spec`] (catalogs passed in, so it's
/// testable without touching the config dir).
fn resolve_endpoint_spec_in(
    spec: &str,
    cats: &[crate::catalog::sources::NamedCatalogSource],
) -> Result<String, String> {
    if spec.contains("://") {
        return Ok(spec.to_string());
    }
    if cats.is_empty() {
        return Err(format!(
            "'{spec}' is not a URL and no catalogs are configured to resolve it against \
             (add one with `vectordata config catalog add`)"
        ));
    }
    // A 1-based index into the configured catalogs.
    if let Ok(n) = spec.parse::<usize>()
        && (1..=cats.len()).contains(&n)
    {
        return Ok(cats[n - 1].location.clone());
    }
    // Otherwise a catalog name (covers legacy list entries named "0", "1", …).
    if let Some(c) = cats.iter().find(|c| c.name == spec) {
        return Ok(c.location.clone());
    }
    Err(format!(
        "no configured catalog named or indexed '{spec}'; configured: {}",
        cats.iter()
            .enumerate()
            .map(|(i, c)| format!("{}={}", i + 1, c.name))
            .collect::<Vec<_>>()
            .join(", ")
    ))
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
///
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
            Entry { token: "login-tok".to_string(), user: Some("alice".to_string()), expires: None, catalog: None },
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
            Entry { token: "origin-tok".into(), user: None, expires: None, catalog: None },
        );
        store.set(
            "https://h:8443/datasets".to_string(),
            Entry { token: "cat-tok".into(), user: None, expires: None, catalog: None },
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
            Entry { token: "data-tok".into(), user: None, expires: None, catalog: None },
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
            catalog: None,
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
    fn endpoint_spec_resolves_url_name_or_index() {
        use crate::catalog::sources::NamedCatalogSource;
        let cats = vec![
            NamedCatalogSource { name: "prot".into(), location: "https://h/datasets/".into() },
            NamedCatalogSource { name: "lab".into(), location: "/data/lab".into() },
        ];
        // A URL is taken verbatim.
        assert_eq!(resolve_endpoint_spec_in("https://x/y", &cats).unwrap(), "https://x/y");
        // By catalog name.
        assert_eq!(resolve_endpoint_spec_in("prot", &cats).unwrap(), "https://h/datasets/");
        // By 1-based index.
        assert_eq!(resolve_endpoint_spec_in("2", &cats).unwrap(), "/data/lab");
        // Out-of-range index and unknown name both error.
        assert!(resolve_endpoint_spec_in("9", &cats).is_err());
        assert!(resolve_endpoint_spec_in("nope", &cats).is_err());
        // A non-URL with no configured catalogs errors.
        assert!(resolve_endpoint_spec_in("prot", &[]).is_err());
        // A legacy list entry literally named "0" resolves by name even though
        // 0 isn't a valid 1-based index.
        let list = vec![NamedCatalogSource { name: "0".into(), location: "https://z/".into() }];
        assert_eq!(resolve_endpoint_spec_in("0", &list).unwrap(), "https://z/");
    }

    #[test]
    fn prune_keeps_only_catalog_justified_entries() {
        let mk = || Entry { token: "t".into(), user: None, expires: None, catalog: None };
        let mut store = Store::default();
        store.set("https://h:8443/datasets".into(), mk()); // exact catalog key
        store.set("https://h:8443".into(), mk()); // origin-wide → governs the catalog
        store.set("https://h:8443/datasets/sub".into(), mk()); // scoped under the catalog
        store.set("https://h:8443/other".into(), mk()); // sibling path — orphan
        store.set("https://elsewhere:9000".into(), mk()); // unrelated origin — orphan
        store.set("https://h:8443/data".into(), mk()); // prefix-of-string but not segment — orphan

        let catalog_keys = vec!["https://h:8443/datasets".to_string()];
        let mut removed = prune_orphans_in(&mut store, &catalog_keys);
        removed.sort();
        assert_eq!(
            removed,
            vec![
                "https://elsewhere:9000".to_string(),
                "https://h:8443/data".to_string(),
                "https://h:8443/other".to_string(),
            ]
        );
        // The three justified entries survive.
        assert!(store.get("https://h:8443/datasets").is_some());
        assert!(store.get("https://h:8443").is_some());
        assert!(store.get("https://h:8443/datasets/sub").is_some());

        // No catalog keys (e.g. all catalogs removed) → every entry is an orphan.
        let mut store2 = Store::default();
        store2.set("https://h:8443/datasets".into(), mk());
        assert_eq!(prune_orphans_in(&mut store2, &[]).len(), 1);
        assert!(store2.list().is_empty());
    }

    #[test]
    fn store_set_get_remove_roundtrip() {
        let mut s = Store::default();
        s.set(
            "https://h:1".into(),
            Entry { token: "vd_abc".into(), user: Some("alice".into()), expires: None, catalog: None },
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
            Entry { token: "vd_x".into(), user: Some("bob".into()), expires: Some("2026-01-01T00:00:00Z".into()), catalog: None },
        );
        let text = toml::to_string_pretty(&s).unwrap();
        let back: Store = toml::from_str(&text).unwrap();
        assert_eq!(back.get("https://vecd-host:8443").unwrap().token, "vd_x");
    }
}
