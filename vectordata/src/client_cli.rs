// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Command handlers for the client-side `vecd` integration verbs —
//! `login`/`logout`/`whoami`/`ping`/`token`. Thin orchestration over
//! [`crate::endpoint`] (the HTTP API) and [`crate::credentials`] (the
//! per-origin token store). Each returns a process exit code.

use std::io::{self, Write};

use crate::credentials::{origin_of_str, Entry, Store};
use crate::endpoint;

/// `vectordata login <url>` — establish a stored bearer credential for an
/// endpoint. `--token` stores a pre-issued key directly; otherwise a
/// password grant (`POST /auth/token`) is performed for `--user`.
pub fn login(
    url: &str,
    user: Option<&str>,
    token: Option<&str>,
    password: Option<&str>,
    expires: Option<&str>,
) -> i32 {
    let Some(origin) = origin_of_str(url) else {
        eprintln!("not a valid endpoint URL: {url}");
        return 2;
    };

    let (token, user, expires_at) = if let Some(tok) = token {
        // --token is a literal token or a file (JSON record / credential store
        // / bare token); a store is keyed by origin, so pass the target url.
        let resolved = match crate::credentials::resolve_token_arg(tok, Some(url)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}");
                return 2;
            }
        };
        // The token's user: from the JSON record, else --user, else ask the
        // endpoint (so the stored credential is labeled with its user).
        let user = resolved
            .user
            .or_else(|| user.map(|s| s.to_string()))
            .or_else(|| whoami_user(url, &resolved.token));
        (resolved.token, user, resolved.expires)
    } else {
        let Some(user) = user else {
            eprintln!("login needs either --token, or --user to exchange a password");
            return 2;
        };
        let password = match password {
            Some(p) => p.to_string(),
            None => match std::env::var("VECTORDATA_PASSWORD") {
                Ok(p) if !p.is_empty() => p,
                _ => match read_password(&format!("Password for {user}@{origin}: ")) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("could not read password: {e}");
                        return 1;
                    }
                },
            },
        };
        match endpoint::login_password(url, user, &password, Some("vectordata login"), expires) {
            Ok(resp) => (
                resp.token,
                Some(user.to_string()),
                // Informational; stored as the raw epoch (the server is the
                // authority on expiry — the client never enforces it).
                resp.expires_at.map(|e| e.to_string()),
            ),
            Err(e) => {
                eprintln!("login failed: {e}");
                return 1;
            }
        }
    };

    // Record under the URL you logged in to: an origin (`https://h/`) records an
    // origin-wide credential; a catalog URL (`https://h/datasets/`) records one
    // scoped to that catalog, which then wins for reads under it.
    let key = crate::credentials::credential_key(url).unwrap_or_else(|| origin.clone());
    let mut store = Store::load();
    let user_note = user.as_deref().map(|u| format!(" as {u}")).unwrap_or_default();
    store.set(key.clone(), Entry { token, user, expires: expires_at });
    if let Err(e) = store.save() {
        eprintln!("could not write credentials: {e}");
        return 1;
    }
    println!("logged in to {key}{user_note}");
    0
}

/// After a successful login, offer to add the endpoint's namespace catalogs to
/// the local catalog list. Interactive only — skipped in scripts/pipes — and a
/// no-op when the endpoint exposes no namespaces you can target.
///
/// Called from the binary's `login` command (the interactive entry point), **not**
/// from [`login`] itself, so the pure handler never blocks on a stdin prompt
/// (which would hang non-interactive callers such as integration tests).
pub fn offer_endpoint_catalogs(url: &str) {
    use std::io::{IsTerminal, Write};
    if !std::io::stdin().is_terminal() {
        return;
    }
    let base = url.trim_end_matches('/');
    let token = crate::credentials::stored_token(base);
    let namespaces = crate::endpoint::candidate_namespaces(base, token.as_deref());
    if namespaces.is_empty() {
        return;
    }
    let urls: Vec<String> =
        namespaces.iter().map(|ns| format!("{base}/{}/", ns.trim_matches('/'))).collect();

    let plural = if urls.len() == 1 { "" } else { "s" };
    println!("\nThis endpoint exposes {} namespace{plural} you can target:", urls.len());
    for u in &urls {
        println!("  {u}");
    }
    print!("Add as catalog{plural} so `datasets list` shows them? [Y/n] ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return;
    }
    let ans = line.trim().to_lowercase();
    if !(ans.is_empty() || ans == "y" || ans == "yes") {
        return;
    }
    for u in urls {
        crate::config::add_catalog(&u);
    }
}

/// Best-effort: ask the endpoint which user a token authenticates as, so a
/// stored credential can be labeled. `None` on any error or anonymous access.
fn whoami_user(url: &str, token: &str) -> Option<String> {
    let v = endpoint::whoami(url, Some(token)).ok()?;
    if v.get("authenticated").and_then(|a| a.as_bool()) == Some(true) {
        v.get("identity").and_then(|i| i.as_str()).map(String::from)
    } else {
        None
    }
}

/// `vectordata logout <url>` — forget a stored credential.
pub fn logout(url: &str) -> i32 {
    // Forget the credential recorded under this exact key (origin or catalog).
    let Some(key) = crate::credentials::credential_key(url) else {
        eprintln!("not a valid endpoint URL: {url}");
        return 2;
    };
    let mut store = Store::load();
    if store.remove(&key) {
        if let Err(e) = store.save() {
            eprintln!("could not write credentials: {e}");
            return 1;
        }
        println!("logged out of {key}");
    } else {
        println!("no stored credential for {key}");
    }
    0
}

/// `vectordata login --list` — endpoints with stored credentials.
pub fn list_logins() -> i32 {
    let store = Store::load();
    let entries = store.list();
    if entries.is_empty() {
        println!("no stored credentials");
    }
    for (origin, user) in entries {
        match user {
            Some(u) => println!("{origin}\t{u}"),
            None => println!("{origin}"),
        }
    }
    0
}

/// `vectordata whoami [<url>]` / `vectordata ping <url>` — print the
/// caller's access at an endpoint. `ping` degrades gracefully against a
/// non-`vecd` host.
pub fn ping(url: &str, graceful: bool) -> i32 {
    let token = crate::credentials::stored_token(url);
    crate::credentials::warn_if_expiring(url);
    match endpoint::whoami(url, token.as_deref()) {
        Ok(view) => {
            print_access(url, &view);
            0
        }
        Err(e) if e == "not-a-vecd" => {
            if graceful {
                println!("endpoint: {url}");
                println!("  (not a vecd endpoint — no /-/whoami; reachable for anonymous reads only)");
                0
            } else {
                eprintln!("{url} is not a vecd endpoint");
                1
            }
        }
        Err(e) => {
            eprintln!("ping failed: {e}");
            1
        }
    }
}

fn print_access(url: &str, view: &serde_json::Value) {
    println!("endpoint:  {url}");
    let identity = view.get("identity").and_then(|v| v.as_str()).unwrap_or("(anonymous)");
    let level = view.get("level").and_then(|v| v.as_str()).unwrap_or("-");
    println!("identity:  {identity:<24} level: {level}");
    if let Some(nss) = view.get("namespaces").and_then(|v| v.as_array()) {
        println!("visible namespaces:");
        for ns in nss {
            let path = ns.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let listable = ns.get("listable").and_then(|v| v.as_str()).unwrap_or("");
            let owner = ns.get("owner").and_then(|v| v.as_bool()).unwrap_or(false);
            let actions: Vec<&str> =
                ns.get("actions").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
            let who = if owner { "(owner)".to_string() } else { listable.to_string() };
            println!("  {path:<28} {who:<10} {}", actions.join(","));
        }
    }
    if let Some(hidden) = view.get("hidden").and_then(|v| v.as_u64())
        && hidden > 0 {
            println!("hidden from you:  {hidden} namespace(s)");
        }
}

/// `vectordata token issue <url> --description … [--profile …] [--expires …]`
pub fn token_issue(
    url: &str,
    description: &str,
    profile: Option<&str>,
    expires: Option<&str>,
) -> i32 {
    let Some(session) = crate::credentials::stored_token(url) else {
        eprintln!("not logged in to {url} — run `vectordata login {url}` first");
        return 2;
    };
    crate::credentials::warn_if_expiring(url);
    match endpoint::issue_token(url, &session, description, profile, expires) {
        Ok(resp) => {
            println!("token: {}", resp.token);
            if let Some(id) = resp.id {
                println!("id:    {id}");
            }
            println!("\nHand this to the recipient; it is not recoverable.");
            0
        }
        Err(e) => {
            eprintln!("token issue failed: {e}");
            1
        }
    }
}

/// `vectordata token revoke <url> <id>`
pub fn token_revoke(url: &str, id: i64) -> i32 {
    let Some(session) = crate::credentials::stored_token(url) else {
        eprintln!("not logged in to {url}");
        return 2;
    };
    match endpoint::revoke_token(url, &session, id) {
        Ok(()) => {
            println!("revoked token {id}");
            0
        }
        Err(e) => {
            eprintln!("revoke failed: {e}");
            1
        }
    }
}

/// `vectordata backup <url> --to <dest> [--incremental]`
pub fn backup(url: &str, dest: &str, incremental: bool) -> i32 {
    let token = crate::credentials::stored_token(url);
    match crate::backup::run_backup(url, dest, incremental, token.as_deref()) {
        Ok(s) => {
            println!(
                "backed up {} namespace(s), {} version(s) ({} skipped); {} blob(s) fetched, {} already present ({} bytes)",
                s.namespaces, s.versions, s.versions_skipped, s.blobs_fetched, s.blobs_skipped, s.bytes
            );
            0
        }
        Err(e) => {
            eprintln!("backup failed: {e}");
            1
        }
    }
}

/// `vectordata restore <src> --to <url>`
pub fn restore(src: &str, url: &str) -> i32 {
    let token = crate::credentials::stored_token(url);
    match crate::backup::run_restore(src, url, token.as_deref()) {
        Ok(s) => {
            println!("restored {} namespace(s), {} object(s) to {url}", s.namespaces, s.objects);
            0
        }
        Err(e) => {
            eprintln!("restore failed: {e}");
            1
        }
    }
}

/// Read a password without echoing (raw mode). Falls back to a plain line
/// read if the terminal can't enter raw mode (e.g. piped input).
fn read_password(prompt: &str) -> io::Result<String> {
    use crossterm::event::{read, Event, KeyCode, KeyEventKind, KeyModifiers};
    use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

    eprint!("{prompt}");
    io::stderr().flush()?;

    if enable_raw_mode().is_err() {
        // Non-interactive input: read a line plainly.
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        return Ok(line.trim_end_matches(['\r', '\n']).to_string());
    }

    let mut pw = String::new();
    let result = loop {
        match read() {
            Ok(Event::Key(k)) if k.kind == KeyEventKind::Press => match k.code {
                KeyCode::Enter => break Ok(pw.clone()),
                KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                    break Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"));
                }
                KeyCode::Char(c) => pw.push(c),
                KeyCode::Backspace => {
                    pw.pop();
                }
                _ => {}
            },
            Ok(_) => {}
            Err(e) => break Err(e),
        }
    };
    let _ = disable_raw_mode();
    eprintln!();
    result
}
