// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! CLI support. Non-blocking update-availability notice, backed by the GitHub
//! Releases page.
//!
//! Design constraints (the etiquette every CLI update check owes its
//! users):
//!
//! - **Never blocks and never fails a command.** The startup hook
//!   prints from *cached* state only (zero network); the remote
//!   probe runs in a detached re-invocation of this binary (marked
//!   by `PROBE_CHILD_ENV`) that outlives the user's command —
//!   threads die with the process, and most vectordata invocations
//!   finish faster than one HTTPS round trip. The child writes the
//!   state file for the NEXT run to print.
//! - **Throttled.** Probes happen at most once per
//!   `CHECK_INTERVAL_SECS` (24 h), tracked in `update_check.yaml`
//!   under the config dir ([`crate::catalog::sources::config_dir`],
//!   so `VECTORDATA_HOME` isolation applies). The timestamp is only
//!   advanced on a successful probe.
//! - **Opt-out, loudly honored.** Disabled by the `update_check: off`
//!   settings key, the `VECTORDATA_NO_UPDATE_CHECK` env var, any
//!   `CI` env var, or a non-terminal stderr (scripts and pipelines
//!   get zero phone-home).
//! - **Notify, don't act.** One line on stderr naming the newer
//!   version and the releases page. Nothing is downloaded.
//!
//! The probe is a single request with no API quota cost: GitHub
//! serves `…/releases/latest` as a redirect to
//! `…/releases/tag/<tag>`, so reading the `Location` header of the
//! un-followed response yields the latest release tag. The client
//! sends a descriptive `User-Agent` per GitHub's terms.
//!
//! The probe goes through a transport of its own, configured from the
//! process's (`probe_config`): the same TLS trust and offline gate, no
//! redirects, a five-second ceiling and one client. It is anonymous —
//! vectordata's read credentials are for dataset servers, not GitHub
//! (`docs/design/srd-crate-boundaries.md` TR-10).

use std::path::PathBuf;
use std::time::Duration;

use crate::transport::{Request, Timeouts, Transport, TransportConfig};

/// Repository whose Releases page is probed, and the link printed
/// in the notice. Single authority: taken from the crate manifest's
/// `repository` field at compile time.
const REPO_URL: &str = env!("CARGO_PKG_REPOSITORY");

/// Minimum seconds between remote probes (24 hours).
const CHECK_INTERVAL_SECS: u64 = 24 * 60 * 60;

/// State file under the config dir. Machine-managed (rewritten
/// wholesale on each successful probe) — not user configuration, so
/// the comment-preserving editor rules for `settings.yaml` do not
/// apply to it.
const STATE_FILE: &str = "update_check.yaml";

/// Settings key controlling the check (`on` / `off`).
pub const SETTING_KEY: &str = "update_check";

/// Env var that disables the check when set to any non-empty value.
pub const OPT_OUT_ENV: &str = "VECTORDATA_NO_UPDATE_CHECK";

/// Internal env var marking the detached probe child — a
/// re-invocation of this binary whose only job is to fetch the
/// latest release tag and record it, surviving the parent command's
/// exit. Not part of the CLI surface.
const PROBE_CHILD_ENV: &str = "VECTORDATA_INTERNAL_UPDATE_PROBE";

/// Does this settings value disable the check? Unset means enabled
/// (the check is opt-out). Recognizes the usual spellings of "no".
fn setting_disables(value: &str) -> bool {
    matches!(value.trim().to_lowercase().as_str(),
        "off" | "false" | "no" | "0" | "disabled")
}

/// Pure resolution of the enabled state from its four inputs: the
/// `update_check` settings value, the [`OPT_OUT_ENV`] value, the `CI`
/// env value, and whether offline mode is on
/// ([`settings::offline`](crate::settings::offline)). Env values
/// disable when present and non-empty; offline mode disables, since the
/// probe is a request to GitHub.
pub fn enabled_from(
    setting: Option<&str>,
    opt_out_env: Option<&str>,
    ci_env: Option<&str>,
    offline: bool,
) -> bool {
    if offline { return false; }
    if opt_out_env.is_some_and(|v| !v.is_empty()) { return false; }
    if ci_env.is_some_and(|v| !v.is_empty()) { return false; }
    !setting.is_some_and(setting_disables)
}

/// Whether the configured settings value (if any) leaves the check
/// enabled — the `config get update_check` view of the world.
pub fn enabled_setting(setting: Option<&str>) -> bool {
    !setting.is_some_and(setting_disables)
}

/// Parse `"1.5.2"` / `"v1.5.2"` into a comparable version triple.
/// Strict three-numeric-component form only — pre-release or
/// otherwise decorated tags are ignored rather than misordered.
fn parse_semverish(s: &str) -> Option<(u64, u64, u64)> {
    let s = s.trim().strip_prefix('v').unwrap_or(s.trim());
    let mut it = s.split('.');
    let major = it.next()?.parse().ok()?;
    let minor = it.next()?.parse().ok()?;
    let patch = it.next()?.parse().ok()?;
    if it.next().is_some() { return None; }
    Some((major, minor, patch))
}

/// True when `remote` is a strictly newer version than `current`.
/// Unparsable versions never count as newer.
fn is_newer(remote: &str, current: &str) -> bool {
    match (parse_semverish(remote), parse_semverish(current)) {
        (Some(r), Some(c)) => r > c,
        _ => false,
    }
}

/// Extract the tag from a `releases/tag/<tag>` redirect Location.
fn tag_from_location(location: &str) -> &str {
    location.rsplit('/').next().unwrap_or(location)
}

/// Parse the state file body: `(last_check_epoch_secs, latest)`.
/// Tolerant of a missing or truncated file — `None` fields simply
/// mean "probe again".
fn parse_state(content: &str) -> (Option<u64>, Option<String>) {
    let last = crate::settings::setting_value_from(content, "last_check")
        .and_then(|v| v.parse().ok());
    let latest = crate::settings::setting_value_from(content, "latest");
    (last, latest)
}

/// Render the state file body. Mirrors [`parse_state`].
fn render_state(last_check: u64, latest: Option<&str>) -> String {
    match latest {
        Some(v) => format!("last_check: {last_check}\nlatest: {v}\n"),
        None => format!("last_check: {last_check}\n"),
    }
}

/// The one-line stderr notice.
fn notice_line(latest: &str, current: &str) -> String {
    format!("vectordata {latest} is available (current {current}) — {REPO_URL}/releases")
}

/// The probe's whole-request ceiling, connecting included.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// The probe's transport configuration, from the process's `base`
/// (pure): its offline mode and TLS trust, with redirects off — the
/// answer *is* the redirect — a [`PROBE_TIMEOUT`] ceiling on connecting
/// and on the whole request, a descriptive user agent, and one client,
/// since the probe makes one request.
fn probe_config(base: &TransportConfig) -> TransportConfig {
    TransportConfig {
        user_agent: concat!("vectordata/", env!("CARGO_PKG_VERSION"), " (+", env!("CARGO_PKG_REPOSITORY"), ")")
            .to_string(),
        timeouts: Timeouts { request: PROBE_TIMEOUT, connect: PROBE_TIMEOUT },
        redirects: 0,
        parallel_streams: 1,
        download_concurrency: 1,
        ..base.clone()
    }
}

/// The latest release tag from `releases_latest_url`, through
/// `transport`: an anonymous, un-followed GET whose redirect `Location`
/// names the tag. Every failure mode collapses to `None`.
fn latest_tag(transport: &Transport, releases_latest_url: &str) -> Option<String> {
    let resp = transport.send(Request::get(releases_latest_url).anonymous()).ok()?;
    if !resp.is_redirection() { return None; }
    Some(tag_from_location(resp.header("location")?).to_string())
}

/// Probe the Releases page once: un-followed GET of
/// `releases/latest`, latest tag read from the redirect `Location`.
fn fetch_latest_tag() -> Option<String> {
    let transport = Transport::new(probe_config(crate::transport::shared().config()));
    latest_tag(&transport, &format!("{REPO_URL}/releases/latest"))
}

/// Startup hook, called once from `main` after argument parsing
/// (completion, `--help`, and `--version` paths never reach it).
///
/// Prints the cached notice when a newer version is already known,
/// then — if the last successful probe is older than the throttle
/// interval — spawns a detached background probe that records its
/// answer for future runs. See the module docs for why nothing here
/// can block, fail, or print mid-command.
pub fn startup(current_version: &str) {
    use std::io::IsTerminal;
    if !std::io::stderr().is_terminal() { return; }
    let enabled = enabled_from(
        crate::settings::setting_value(SETTING_KEY).as_deref(),
        std::env::var(OPT_OUT_ENV).ok().as_deref(),
        std::env::var("CI").ok().as_deref(),
        crate::settings::offline(),
    );
    if !enabled { return; }

    let state_path = PathBuf::from(crate::catalog::sources::config_dir())
        .join(STATE_FILE);
    let (last_check, latest) = parse_state(
        &std::fs::read_to_string(&state_path).unwrap_or_default());

    if let Some(latest) = &latest
        && is_newer(latest, current_version) {
            eprintln!("{}", notice_line(latest, current_version));
        }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let stale = last_check
        .map(|t| now.saturating_sub(t) >= CHECK_INTERVAL_SECS)
        .unwrap_or(true);
    if !stale { return; }

    // Detached probe child: re-invoke this binary with the marker
    // env var and all stdio nulled, and never wait on it. A thread
    // would die with the process — and most commands finish faster
    // than one HTTPS round trip — while the child completes on its
    // own and records the answer for the next run. Failure to spawn
    // is ignored: the stale timestamp retries later.
    if let Ok(exe) = std::env::current_exe() {
        let mut cmd = std::process::Command::new(exe);
        cmd.env(PROBE_CHILD_ENV, "1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        // Own process group: the child must survive the terminal
        // closing right after the parent command finishes (session
        // teardown HUPs the foreground group).
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
        let _ = cmd.spawn();
    }
}

/// Entry hook for the detached probe child, called first thing in
/// `main`. Returns `true` when this process IS the probe child —
/// the probe has run and the caller must return without doing
/// anything else (no parsing, no command, no output).
pub fn run_probe_child_if_marked() -> bool {
    if std::env::var_os(PROBE_CHILD_ENV).is_none() { return false; }
    let Some(tag) = fetch_latest_tag() else { return true; };
    let version = tag.trim().trim_start_matches('v').to_string();
    // Only a parseable release version advances the throttle
    // timestamp — anything else retries on a later run.
    if parse_semverish(&version).is_none() { return true; }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let state_path = PathBuf::from(crate::catalog::sources::config_dir())
        .join(STATE_FILE);
    if let Some(parent) = state_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&state_path, render_state(now, Some(&version)));
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enabled_resolution_matrix() {
        // Default: no setting, no env → enabled.
        assert!(enabled_from(None, None, None, false));
        // Settings value disables in any usual spelling; "on" keeps it.
        for off in ["off", "false", "no", "0", "disabled", " OFF "] {
            assert!(!enabled_from(Some(off), None, None, false), "{off:?} must disable");
        }
        assert!(enabled_from(Some("on"), None, None, false));
        assert!(enabled_from(Some("true"), None, None, false));
        // Opt-out env wins over an enabling setting.
        assert!(!enabled_from(Some("on"), Some("1"), None, false));
        // CI env disables; empty env values do not.
        assert!(!enabled_from(None, None, Some("true"), false));
        assert!(enabled_from(None, Some(""), Some(""), false));
        // Offline mode disables, whatever the setting says.
        assert!(!enabled_from(None, None, None, true));
        assert!(!enabled_from(Some("on"), None, None, true));
    }

    #[test]
    fn version_parse_and_compare() {
        assert_eq!(parse_semverish("1.5.2"), Some((1, 5, 2)));
        assert_eq!(parse_semverish("v1.5.2"), Some((1, 5, 2)));
        assert_eq!(parse_semverish("v1.5.2-rc1"), None);
        assert_eq!(parse_semverish("1.5"), None);
        assert_eq!(parse_semverish("1.5.2.3"), None);
        assert_eq!(parse_semverish("latest"), None);

        assert!(is_newer("1.5.3", "1.5.2"));
        assert!(is_newer("v2.0.0", "1.9.9"));
        // Numeric, not lexicographic: 1.10.0 > 1.9.0.
        assert!(is_newer("1.10.0", "1.9.0"));
        assert!(!is_newer("1.5.2", "1.5.2"));
        assert!(!is_newer("1.5.1", "1.5.2"));
        assert!(!is_newer("garbage", "1.5.2"));
    }

    #[test]
    fn state_round_trip_and_tolerance() {
        let body = render_state(1_781_217_000, Some("1.5.3"));
        assert_eq!(parse_state(&body),
            (Some(1_781_217_000), Some("1.5.3".to_string())));
        // Missing / truncated state means "probe again".
        assert_eq!(parse_state(""), (None, None));
        assert_eq!(parse_state("last_check: not-a-number\n"), (None, None));
    }

    #[test]
    fn tag_extraction_from_redirect_location() {
        assert_eq!(tag_from_location(
            "https://github.com/nosqlbench/vectordata-rs/releases/tag/v1.5.3"),
            "v1.5.3");
        assert_eq!(tag_from_location("v1.5.3"), "v1.5.3");
    }

    /// **The probe follows no redirect and keeps its 5 s ceiling**, and
    /// otherwise carries the process's configuration (TR-10).
    #[test]
    fn the_probe_follows_no_redirect_within_five_seconds() {
        let base = TransportConfig {
            offline: true,
            tls: crate::transport::TlsTrust { trusted_ca_certs: vec!["/ca.pem".into()], trust_self_signed: vec![] },
            ..TransportConfig::default()
        };
        let c = probe_config(&base);
        assert_eq!(c.redirects, 0);
        assert_eq!(c.timeouts, Timeouts { request: PROBE_TIMEOUT, connect: PROBE_TIMEOUT });
        assert_eq!(PROBE_TIMEOUT, Duration::from_secs(5));
        assert!(c.offline && c.tls == base.tls, "the process's offline mode and trust carry over");
        assert!(c.user_agent.starts_with("vectordata/") && c.user_agent.contains(REPO_URL), "{}", c.user_agent);
    }

    /// The tag is read from the redirect itself: a server answering
    /// `302 Location: …/tag/v9.8.7` whose target would answer otherwise.
    #[test]
    fn the_tag_is_read_from_the_unfollowed_redirect() {
        use std::io::{BufRead, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let target = format!("{base}/releases/tag/v9.8.7");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut line = String::new();
                let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
                let _ = reader.read_line(&mut line);
                loop {
                    let mut h = String::new();
                    if reader.read_line(&mut h).is_err() || h.trim().is_empty() { break; }
                }
                let reply = if line.contains("/releases/latest") {
                    format!("HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                } else {
                    // A followed redirect lands here, which names no tag.
                    "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok".to_string()
                };
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        let transport = Transport::new(probe_config(&TransportConfig::default()));
        assert_eq!(latest_tag(&transport, &format!("{base}/releases/latest")).as_deref(), Some("v9.8.7"));
        let offline = Transport::new(probe_config(&TransportConfig { offline: true, ..TransportConfig::default() }));
        assert_eq!(latest_tag(&offline, &format!("{base}/releases/latest")), None);
    }

    #[test]
    fn notice_names_both_versions_and_the_releases_page() {
        let n = notice_line("1.5.3", "1.5.2");
        assert!(n.contains("1.5.3") && n.contains("1.5.2"));
        assert!(n.contains("/releases"));
    }
}
