// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! CLI support. `vectordata datasets push`: argument parsing, the
//! interactive choice of a destination, and the exit code, over the
//! library engine [`super::execute`].

use super::*;

/// When `push` has no destination (`.publish_url` absent, no `--to`) but the
/// user is logged in, offer to push to that endpoint. Returns the chosen
/// destination URL (`Some`) once confirmed, or `None` when there's no usable
/// fallback (not logged in / ambiguous, or non-interactive without `-y`) so the
/// caller emits the standard "no destination" error.
///
/// The default path is *smart*: when the user can write to exactly one
/// namespace on the endpoint (per `whoami`), the dataset lands under it
/// (`<endpoint>/<namespace>/<dataset>/`); otherwise it falls back to
/// `<endpoint>/<dataset>/`. Interactively the default is shown and editable.
fn derive_destination(root: &std::path::Path, assume_yes: bool) -> Result<Option<String>, String> {
    use std::io::{IsTerminal, Write};

    let endpoint = match crate::credentials::resolve_endpoint(None) {
        Ok(e) => e,
        Err(_) => return Ok(None), // not logged in, or several — standard error
    };
    let base = endpoint.trim_end_matches('/').to_string();
    let dataset = dataset_name(root);

    // We can only fill in a destination if we can either auto-proceed (`-y`) or
    // prompt (a terminal). Otherwise leave it to the engine's standard error —
    // and skip the `whoami` probe below entirely.
    if !assume_yes && !std::io::stdin().is_terminal() {
        return Ok(None);
    }

    let token = crate::credentials::stored_token(&base);
    let namespaces = crate::endpoint::candidate_namespaces(&base, token.as_deref());
    let mut default_url = match namespaces.as_slice() {
        // Exactly one namespace to target → place the dataset under it.
        [only] => format!("{base}/{}/{dataset}/", only.trim_matches('/')),
        // None or several — leave the namespace out of the default; the user
        // edits the prompt (and `--to` tab-completion offers the namespaces).
        _ => format!("{base}/{dataset}/"),
    };
    if !default_url.ends_with('/') {
        default_url.push('/');
    }

    // `-y`: don't prompt — announce the derived destination and proceed.
    if assume_yes {
        println!("No destination set; using your logged-in endpoint:\n  {default_url}");
        return Ok(Some(default_url));
    }

    println!();
    println!("No .publish_url here and no --to, but you're logged in to");
    println!("  {base}");
    println!();
    print!("Push \"{dataset}\" to [{default_url}]: ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    let n = std::io::stdin()
        .read_line(&mut line)
        .map_err(|e| format!("reading destination: {e}"))?;
    if n == 0 {
        // EOF (Ctrl-D) — treat as "no, don't push".
        return Err("aborted — no destination chosen".to_string());
    }
    let typed = line.trim();
    let mut url = if typed.is_empty() { default_url } else { typed.to_string() };
    if !url.ends_with('/') {
        url.push('/');
    }
    Ok(Some(url))
}

/// The dataset name used as the remote subdirectory: the canonical basename of
/// the source path (falling back to `"dataset"`).
fn dataset_name(root: &std::path::Path) -> String {
    std::fs::canonicalize(root)
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "dataset".to_string())
}

/// Resolve a `--to` value into a full destination URL.
///
/// - A URL (contains `://`) is used as-is (trailing slash ensured).
/// - Otherwise it's a namespace shorthand resolved against the **authenticated**
///   endpoint: `root` (or `/`) targets the root `/` namespace; a bare name
///   targets the namespace whose path equals it (or whose last segment does),
///   provided exactly one matches. The dataset lands under it as
///   `<endpoint>/<namespace>/<dataset>/`.
fn resolve_to(to: &str, root: &std::path::Path) -> Result<String, String> {
    if to.contains("://") {
        let mut u = to.to_string();
        if !u.ends_with('/') {
            u.push('/');
        }
        return Ok(u);
    }
    let endpoint = crate::credentials::resolve_endpoint(None)
        .map_err(|e| format!("--to '{to}' is a namespace name, but {e}"))?;
    let base = endpoint.trim_end_matches('/');
    let token = crate::credentials::stored_token(base);
    let namespaces = crate::endpoint::candidate_namespaces(base, token.as_deref());
    build_to_url(base, &dataset_name(root), to, &namespaces)
}

/// Pure core of [`resolve_to`] for a namespace shorthand: build the destination
/// URL from the resolved endpoint `base`, the local `dataset` name, the `to`
/// shorthand, and the endpoint's `namespaces`.
pub(super) fn build_to_url(base: &str, dataset: &str, to: &str, namespaces: &[String]) -> Result<String, String> {
    let base = base.trim_end_matches('/');
    if to == "root" || to == "/" {
        return Ok(format!("{base}/{dataset}/"));
    }
    let want = to.trim_matches('/');
    let matches: Vec<&str> = namespaces
        .iter()
        .map(|s| s.trim_matches('/'))
        .filter(|ns| *ns == want || ns.rsplit('/').next() == Some(want))
        .collect();
    match matches.as_slice() {
        [one] => Ok(format!("{base}/{one}/{dataset}/")),
        [] => {
            let avail = if namespaces.is_empty() {
                "no writable namespaces found — check `vecd ns list` (and that a namespace has \
                 an active backend)"
                    .to_string()
            } else {
                format!("writable namespaces: {}", namespaces.join(", "))
            };
            // `candidate_namespaces` only lists writable ones, so a miss can mean
            // "doesn't exist" OR "exists but has no backend".
            Err(format!("no writable namespace '{to}' on {base} (it may exist but lack a backend) — {avail}"))
        }
        _ => Err(format!(
            "namespace '{to}' is ambiguous — matches {}; give the full namespace path",
            matches.join(", ")
        )),
    }
}

/// Value of `--checksums`, parsed from `auto` or `keep`; maps onto
/// [`ChecksumPolicy`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ChecksumMode {
    /// Recompute a stale or missing SHA256SUMS before pushing.
    Auto,
    /// Use the existing SHA256SUMS; stale/missing stops the push.
    Keep,
}

impl std::str::FromStr for ChecksumMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(ChecksumMode::Auto),
            "keep" => Ok(ChecksumMode::Keep),
            other => Err(format!("unknown checksum mode '{other}' (expected auto|keep)")),
        }
    }
}

/// `vectordata datasets push` — push an already-good dataset or an
/// ad-hoc directory to its bound remote.
#[derive(Debug, veks_completion_derive::VeksCli)]
pub struct PushArgs {
    /// Dataset directory, catalog directory, or (with --raw) ad-hoc
    /// directory to push.
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Target endpoint. Must agree with an existing .publish_url;
    /// written into the source if none exists yet.
    #[arg(long)]
    pub to: Option<String>,

    /// Justification for overwriting remote data. Required when the
    /// push would overwrite; recorded in the remote pushlog.
    #[arg(short = 'm', long)]
    pub message: Option<String>,

    /// Push every file verbatim with no shape validation.
    #[arg(long)]
    pub raw: bool,

    /// How to treat a stale/missing SHA256SUMS.
    #[arg(long, value_parser = ["auto", "keep"], default_value = "auto")]
    pub checksums: ChecksumMode,

    /// Resolve, validate, and print the plan without writing.
    #[arg(long)]
    pub dry_run: bool,

    /// AWS profile for S3 credentials.
    #[arg(long)]
    pub profile: Option<String>,

    /// S3-compatible endpoint override.
    #[arg(long)]
    pub endpoint_url: Option<String>,

    /// Bearer token for generic https endpoints
    /// (else $VECTORDATA_PUSH_TOKEN).
    #[arg(long)]
    pub token: Option<String>,

    /// Remove remote objects under the publish root with no local
    /// counterpart. Destructive: requires -m and prompts unless -y.
    #[arg(long)]
    pub delete: bool,

    /// If an incomplete push is open on the remote and its contents
    /// differ from the current working set, abandon it (record an
    /// abort) and push fresh instead of refusing.
    #[arg(long)]
    pub abort_incomplete: bool,

    /// Parallel upload streams.
    #[arg(long, default_value = "4")]
    pub concurrency: u32,

    /// Skip known-good validation (binding/overwrite/provenance
    /// rules still apply).
    #[arg(long)]
    pub no_check: bool,

    /// Skip the interactive confirmation.
    #[arg(short = 'y', long)]
    pub yes: bool,
}

impl PushArgs {
    fn into_options(self) -> Options {
        let actor = format!(
            "{}@{}",
            std::env::var("USER").or_else(|_| std::env::var("USERNAME")).unwrap_or_else(|_| "unknown".into()),
            std::env::var("HOSTNAME").unwrap_or_else(|_| "host".into()),
        );
        let cmd = std::env::args().collect::<Vec<_>>().join(" ");
        // Token precedence: --token, then $VECTORDATA_PUSH_TOKEN, then the
        // stored login credential for the target endpoint (so `vectordata
        // login` is auto-used for push, not only for reads).
        let token = self
            .token
            .or_else(|| std::env::var("VECTORDATA_PUSH_TOKEN").ok())
            .or_else(|| {
                self.to.as_deref().and_then(|to| {
                    let t = crate::credentials::stored_token(to);
                    if t.is_some()
                        && let Some(w) = crate::credentials::expiry_warning(to)
                    {
                        eprintln!("{w}");
                    }
                    t
                })
            });
        Options {
            path: self.path,
            to: self.to,
            message: self.message,
            raw: self.raw,
            checksums: match self.checksums {
                ChecksumMode::Auto => ChecksumPolicy::Auto,
                ChecksumMode::Keep => ChecksumPolicy::Keep,
            },
            dry_run: self.dry_run,
            no_check: self.no_check,
            assume_yes: self.yes,
            delete: self.delete,
            abort_incomplete: self.abort_incomplete,
            concurrency: self.concurrency,
            files: None,
            transport: TransportOptions {
                token,
                profile: self.profile,
                endpoint_url: self.endpoint_url,
            },
            progress: ProgressSink::Stderr,
            cmd,
            actor,
        }
    }
}

/// Dispatch entry point. Returns a process exit code.
pub fn run(mut args: PushArgs) -> i32 {
    // A bare `--to <name>` is expanded before anything reads the
    // destination (token resolution, the engine, .publish_url). It may be a
    // configured catalog NAME or 1-based INDEX (→ that catalog's URL), or —
    // failing that — a vecd namespace shorthand on the logged-in endpoint
    // (`--to datasets`/`root`). Catalog match wins.
    if let Some(to) = args.to.clone()
        && !to.contains("://") {
            match crate::credentials::resolve_endpoint_spec(&to) {
                Ok(url) => args.to = Some(url),
                Err(_) => match resolve_to(&to, &args.path) {
                    Ok(url) => args.to = Some(url),
                    Err(e) => {
                        eprintln!("push: {e}");
                        return 2;
                    }
                },
            }
        }

    // No destination at all (no --to, no .publish_url)? If the user is
    // logged in, offer to push to that endpoint — prompting for the path
    // with a smart default — and fill `--to` in so the engine sees a
    // concrete destination (and persists the .publish_url on success).
    if args.to.is_none()
        && super::binding::read_binding(&args.path).ok().flatten().is_none()
    {
        match derive_destination(&args.path, args.yes) {
            Ok(Some(url)) => args.to = Some(url),
            Ok(None) => {} // no fallback — the engine emits the standard error
            Err(e) => {
                eprintln!("push: {e}");
                return 2;
            }
        }
    }

    // `--token` may be a literal token or a file (JSON token record, a
    // credential store, or a bare token); resolve it to the literal — using
    // the destination's origin to pick from a store — before the engine sees it.
    if let Some(t) = &args.token {
        match crate::credentials::resolve_token_arg(t, args.to.as_deref()) {
            Ok(r) => args.token = Some(r.token),
            Err(e) => {
                eprintln!("push: {e}");
                return 2;
            }
        }
    }
    match execute(&args.into_options()) {
        Ok(o) => {
            if o.dry_run {
                0
            } else {
                let verb = if o.resumed { "Resumed and completed" } else { "Pushed" };
                println!(
                    "{verb} version {} to {} — {} new, {} overwritten, {} deleted, {} unchanged.",
                    o.version, o.destination, o.added, o.overwritten, o.deleted, o.skipped
                );
                0
            }
        }
        Err(Failure::Usage(m)) => {
            eprintln!("push: {m}");
            2
        }
        Err(Failure::Operational(m)) => {
            eprintln!("push: {m}");
            1
        }
    }
}
