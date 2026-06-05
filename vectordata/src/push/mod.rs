// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata push` — the low-effort verb for putting an already
//! known-good dataset (or an ad-hoc directory) at a remote endpoint so
//! `vectordata` can read it later.
//!
//! This is the consumer-CLI counterpart to `veks publish`: single
//! dataset, scheme-dispatched transport, additive-with-gated-overwrite,
//! and a single-provenance event log. The full design lives in
//! `docs/design/push-command.md`; this module is its implementation.
//!
//! The orchestration ([`execute`]) is transport-agnostic and fully
//! exercised against the local `file://` transport in tests; the
//! `https://` and `s3://` transports ride the same [`transport::PushTransport`]
//! contract.

pub mod binding;
pub mod checksums;
pub mod plan;
pub mod pushlog;
pub mod transport;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use binding::Binding;
use checksums::{Freshness, CHECKSUMS_FILE};
use plan::SourceMode;
use pushlog::{Convergence, Event, Log, Overwrite};
use transport::{PushError, PushTransport, TransportOptions};

/// Whether a stale/missing `SHA256SUMS` is recomputed or treated as a
/// hard stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumPolicy {
    /// Recompute a stale or missing `SHA256SUMS` before pushing.
    Auto,
    /// Use the existing `SHA256SUMS`; a stale or missing one stops the push.
    Keep,
}

/// Resolved, clap-free options — what [`execute`] actually consumes.
/// Keeping this separate from [`PushArgs`] makes the engine testable
/// without a process/argv.
#[derive(Debug, Clone)]
pub struct Options {
    pub path: PathBuf,
    pub to: Option<String>,
    pub message: Option<String>,
    pub raw: bool,
    pub checksums: ChecksumPolicy,
    pub dry_run: bool,
    pub no_check: bool,
    pub assume_yes: bool,
    /// Remove remote objects under the publish root that have no local
    /// counterpart (orphans). Opt-in and destructive: gated behind `-m`
    /// like an overwrite, and recorded in the `begin` event.
    pub delete: bool,
    /// When an incomplete push is open on the remote and its intended
    /// contents differ from the current working set, abandon it (record
    /// an `abort`) and push fresh instead of refusing.
    pub abort_incomplete: bool,
    pub concurrency: u32,
    /// Producer-injected publish set (relative, forward-slashed paths),
    /// used when a caller like `veks publish` has already selected what
    /// to ship from a whole hierarchy. When `Some`, push skips its own
    /// scan + mode detection + known-good validation (the caller owns
    /// those) and publishes exactly this set. When `None`, push scans
    /// the source directory itself.
    pub files: Option<Vec<String>>,
    pub transport: TransportOptions,
    /// The resolved invocation, recorded verbatim on the `begin` event.
    pub cmd: String,
    /// `user@host`, recorded on the `begin` event.
    pub actor: String,
}

/// What a (non-dry-run) push accomplished, or what a dry-run would.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub mode: SourceMode,
    pub destination: String,
    pub version: u64,
    pub added: usize,
    pub overwritten: usize,
    pub skipped: usize,
    pub deleted: usize,
    /// True when this finished a previously-interrupted push (resume)
    /// rather than starting a new version.
    pub resumed: bool,
    pub dry_run: bool,
}

/// Failure classes, mapped to exit codes by [`run`].
#[derive(Debug)]
pub enum Failure {
    /// "You need to do something different" — binding conflict,
    /// overwrite without `-m`, divergent provenance, open update, stale
    /// checksums under `keep`, usage errors. Exit code 2.
    Usage(String),
    /// Operational failure — transport, auth, I/O, concurrency race.
    /// Exit code 1.
    Operational(String),
}

impl Failure {
    fn op<E: std::fmt::Display>(e: E) -> Failure {
        Failure::Operational(e.to_string())
    }
}

/// Per-file transfer decision.
enum Decision {
    Add,
    Overwrite { old_digest: String },
    Skip,
}

/// How this invocation relates to any open (incomplete) push already on
/// the remote — see [`pushlog::Log::trailing_open_begin`].
enum Resolution {
    /// No open push; a normal fresh push at the next sequence.
    Fresh,
    /// An open `begin` whose intended end-state matches ours — finish it
    /// (idempotent re-upload + `complete`) at its sequence.
    Resume { seq: u64 },
    /// An open `begin` whose intent differs; `--abort-incomplete` was
    /// given, so record an `abort` for it and push fresh.
    AbortThenFresh { aborted_seq: u64 },
}

// ─── core engine ────────────────────────────────────────────────────

/// Run a push end-to-end. Returns the [`Outcome`] (including for
/// `--dry-run`) or a classified [`Failure`].
pub fn execute(opts: &Options) -> Result<Outcome, Failure> {
    let root = &opts.path;
    if !root.is_dir() {
        return Err(Failure::Usage(format!("not a directory: {}", root.display())));
    }

    // 1. Source mode + known-good validation. When a producer injects an
    //    explicit publish set, it owns selection and validation, so we
    //    skip detection/validation and treat the source as a hierarchy.
    let mode = if opts.files.is_some() {
        SourceMode::Hierarchy
    } else {
        let m = plan::detect_mode(root, opts.raw).map_err(Failure::Usage)?;
        if !opts.no_check {
            plan::validate(m, root).map_err(Failure::Usage)?;
        }
        m
    };

    // 2. Resolve the destination binding (local .publish_url vs --to).
    let existing = binding::read_binding(root).map_err(Failure::Usage)?.map(|(_, p)| p);
    let bind = binding::reconcile(existing, opts.to.as_deref()).map_err(Failure::Usage)?;
    let endpoint = bind.url().clone();

    // 3. Determine the publish set (injected, or scanned), then bring
    //    every content directory's SHA256SUMS current per the policy.
    let scan = match &opts.files {
        Some(files) => plan::Scan::from_files(files.clone()),
        None => plan::scan(root).map_err(Failure::Usage)?,
    };
    // The exact files published from each directory — the basis for that
    // directory's SHA256SUMS, so the checksum file describes precisely
    // what ships (a filtered subset when a producer selected it).
    let mut dir_files: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for rel in &scan.files {
        let (dir_rel, name) = split_rel(rel);
        dir_files.entry(dir_rel.to_string()).or_default().push(name.to_string());
    }
    let mut local_sums: BTreeMap<String, checksums::ChecksumFile> = BTreeMap::new();
    let mut sums_digests: BTreeMap<String, String> = BTreeMap::new();
    for dir_rel in &scan.content_dirs {
        let dir = dir_join(root, dir_rel);
        let names = dir_files.get(dir_rel).cloned().unwrap_or_default();
        let fresh = checksums::freshness(&dir, &names).map_err(Failure::op)?;
        let cf = match (fresh, opts.checksums) {
            (Freshness::Current, _) => checksums::ChecksumFile::parse(
                &std::fs::read_to_string(dir.join(CHECKSUMS_FILE)).map_err(Failure::op)?,
            )
            .map_err(Failure::op)?,
            (_, ChecksumPolicy::Auto) => {
                if opts.dry_run {
                    // Don't mutate the working tree on a dry run; compute
                    // in memory so the plan is accurate.
                    compute_in_memory(&dir, &names).map_err(Failure::op)?
                } else {
                    checksums::generate(&dir, &names).map_err(Failure::op)?
                }
            }
            (Freshness::Missing, ChecksumPolicy::Keep) => {
                return Err(Failure::Usage(format!(
                    "missing {CHECKSUMS_FILE} in '{}' and --checksums keep was given",
                    display_dir(dir_rel)
                )));
            }
            (Freshness::Stale { reason }, ChecksumPolicy::Keep) => {
                return Err(Failure::Usage(format!(
                    "stale {CHECKSUMS_FILE} in '{}' ({reason}) and --checksums keep was given",
                    display_dir(dir_rel)
                )));
            }
        };
        let rendered = cf.render();
        sums_digests.insert(dir_rel.clone(), format!("sha256:{}", checksums::sha256_bytes(rendered.as_bytes())));
        local_sums.insert(dir_rel.clone(), cf);
    }

    // Snapshot (len, mtime) of every content file as of checksum time, so
    // a concurrent mutation before the bytes are uploaded (TOCTOU) is
    // caught rather than publishing bytes that disagree with the manifest.
    let mut pre_stats: BTreeMap<String, (u64, std::time::SystemTime)> = BTreeMap::new();
    for rel in &scan.files {
        let p = root.join(rel.replace('/', std::path::MAIN_SEPARATOR_STR));
        pre_stats.insert(rel.clone(), checksums::len_mtime(&p).map_err(Failure::op)?);
    }

    // 4. Open transport + auth/reachability preflight (fail fast).
    let tx = transport::open(&endpoint, &opts.transport, opts.concurrency).map_err(Failure::Usage)?;
    tx.preflight().map_err(map_transport)?;

    // 4b. The single-provenance guarantee rests on the store honoring
    //     conditional writes (the pushlog `If-Match`/`If-None-Match` CAS).
    //     Probe it before trusting it; refuse an endpoint that silently
    //     ignores the precondition. Skipped in dry-run (it would write).
    if !opts.dry_run {
        verify_conditional_writes(tx.as_ref())?;
    }

    // 5. Remote binding (identity-by-URL): the remote publish root may
    //    already be bound. Equal → ours; different → conflict.
    if let Some(bytes) = tx.get(binding::PUBLISH_FILE).map_err(map_transport)? {
        let text = String::from_utf8_lossy(&bytes);
        if let Ok(remote_bind) = binding::parse_publish_url(&text)
            && remote_bind.url != endpoint.url
        {
            return Err(Failure::Usage(format!(
                "remote conflict: {} is already bound to\n  {}\nbut this push targets\n  {}",
                tx.describe(),
                remote_bind.url,
                endpoint.url,
            )));
        }
    }

    // 6. Provenance: fetch remote + local logs. Convergence is judged
    //    against the remote's *committed* history (excluding any trailing
    //    open `begin`, which is an in-flight or crashed push reconciled in
    //    step 7).
    let (remote_log, remote_log_etag) = read_remote_log(tx.as_ref())?;
    let mut local_log = read_local_log(root)?;
    let remote_committed = remote_log.committed();

    match local_log.classify(&remote_committed) {
        Convergence::Equal | Convergence::LocalAhead { .. } => {}
        Convergence::RemoteAhead { extra } => {
            // Local is behind the remote's committed history. Three cases:
            //   (a) our content already reproduces the remote's current
            //       head (same stable `sums`) — local merely fell behind
            //       (e.g. a crash after the remote `complete` but before
            //       the local mirror). Nothing to push: fast-forward local
            //       provenance and report up-to-date.
            //   (b) the remote has no newer *committed version* than local
            //       — it is ahead only by abandoned bookkeeping (a
            //       begin+abort, or a crash between `abort` and the fresh
            //       `begin`). No stable content is being skipped: adopt the
            //       committed history as the baseline and proceed.
            //   (c) the remote has a newer committed version with different
            //       content — a genuine conflicting update. Refuse.
            if remote_committed.stable_sums() == Some(&sums_digests) {
                if !opts.dry_run {
                    atomic_write_local(
                        &root.join(pushlog::PUSHLOG_FILE),
                        remote_committed.render().as_bytes(),
                    )
                    .map_err(Failure::op)?;
                }
                return Ok(Outcome {
                    mode,
                    destination: endpoint.url.clone(),
                    version: remote_committed.stable_version().unwrap_or(0),
                    added: 0,
                    overwritten: 0,
                    skipped: scan.files.len(),
                    deleted: 0,
                    resumed: false,
                    dry_run: opts.dry_run,
                });
            }
            if remote_committed.stable_version() == local_log.stable_version() {
                // (b) — fast-forward the local baseline, then fall through.
                local_log = remote_committed.clone();
            } else {
                return Err(Failure::Usage(format!(
                    "divergent provenance: the remote has {extra} committed event(s) this source \
                     has not seen, and a newer stable version with different content.\n\
                     Re-sync the local copy (and its {}) before pushing, then reconcile.",
                    pushlog::PUSHLOG_FILE
                )));
            }
        }
        Convergence::Diverged { common } => {
            return Err(Failure::Usage(format!(
                "forked provenance: local and remote share only the first {common} event(s) \
                 and then disagree. This needs manual reconciliation."
            )));
        }
    }

    // 7. Reconcile any open (incomplete) push. A crashed push leaves a
    //    trailing `begin` whose `sums` record the intended end-state
    //    content fingerprint. If ours matches, we *resume* it — finish
    //    the upload idempotently and write `complete` at the same seq. If
    //    it differs, the source changed since the crash: refuse unless
    //    --abort-incomplete was given, in which case we record an `abort`
    //    and push fresh.
    let resolution = match remote_log.trailing_open_begin() {
        None => Resolution::Fresh,
        Some(pushlog::Event::Begin { seq: oseq, actor: oactor, ts: ots, sums: osums, .. }) => {
            if *osums == sums_digests {
                Resolution::Resume { seq: *oseq }
            } else if opts.abort_incomplete {
                Resolution::AbortThenFresh { aborted_seq: *oseq }
            } else {
                return Err(Failure::Usage(format!(
                    "an incomplete push (seq {oseq}, begun by {oactor} at {ots}) is open on the \
                     remote, and its intended contents differ from your current working set.\n  \
                     • Re-run from the original source to resume and finish it, or\n  \
                     • pass --abort-incomplete to abandon seq {oseq} and push your current state \
                     as a new version."
                )));
            }
        }
        Some(_) => unreachable!("trailing_open_begin only returns a Begin"),
    };
    let resuming = matches!(resolution, Resolution::Resume { .. });
    let seq = match &resolution {
        Resolution::Resume { seq } => *seq,
        // local_log >= remote_committed here, so this carries any
        // LocalAhead tail up when the final log is written.
        Resolution::Fresh => local_log.next_seq(),
        Resolution::AbortThenFresh { aborted_seq } => aborted_seq + 1,
    };

    // 8. Classify each content file: add / skip / overwrite.
    let mut added: Vec<String> = Vec::new();
    let mut overwrites: Vec<Overwrite> = Vec::new();
    let mut to_upload: Vec<String> = Vec::new();
    let mut skipped = 0usize;
    for rel in &scan.files {
        let (dir_rel, name) = split_rel(rel);
        let local_digest = local_sums
            .get(dir_rel)
            .and_then(|cf| cf.digest_of(name))
            .ok_or_else(|| Failure::op(format!("internal: no local digest for {rel}")))?
            .to_string();
        match classify_file(tx.as_ref(), rel, dir_rel, &local_digest, &remote_log_or_dir_sums(tx.as_ref(), dir_rel)?)? {
            Decision::Skip => skipped += 1,
            Decision::Add => {
                added.push(rel.clone());
                to_upload.push(rel.clone());
            }
            Decision::Overwrite { old_digest } => {
                overwrites.push(Overwrite {
                    key: rel.clone(),
                    old_digest,
                    new_digest: format!("sha256:{local_digest}"),
                });
                to_upload.push(rel.clone());
            }
        }
    }

    // 9. With --delete, find remote orphans: objects under the publish
    //    root with no local counterpart. The expected set is the content
    //    files plus the artifacts we manage (per-dir SHA256SUMS, the
    //    pushlog, the binding) — none of which are ever orphans.
    let mut deletes: Vec<String> = Vec::new();
    if opts.delete {
        let mut expected: std::collections::HashSet<String> = scan.files.iter().cloned().collect();
        for dir_rel in &scan.content_dirs {
            expected.insert(plan::sums_key(dir_rel));
        }
        expected.insert(pushlog::PUSHLOG_FILE.to_string());
        expected.insert(binding::PUBLISH_FILE.to_string());
        for key in tx.list("").map_err(map_transport)? {
            if !expected.contains(&key) {
                deletes.push(key);
            }
        }
        deletes.sort();
    }

    // 10. The destructive gate: overwriting or deleting existing remote
    //     data needs -m. (A SHA256SUMS manifest updating to reflect a
    //     purely additive change is a consequence, not a user overwrite,
    //     and does not trip the gate.) Resuming skips the gate — the
    //     interrupted push was already authorized when its begin was
    //     written, and we are merely finishing it.
    if !resuming && (!overwrites.is_empty() || !deletes.is_empty()) && opts.message.is_none() {
        let mut msg = String::from(
            "this push would change existing remote data; supply -m/--message to record why.",
        );
        if !overwrites.is_empty() {
            msg.push_str("\n  Would overwrite:");
            for o in &overwrites {
                msg.push_str(&format!("\n    {}", o.key));
            }
        }
        if !deletes.is_empty() {
            msg.push_str("\n  Would delete:");
            for d in &deletes {
                msg.push_str(&format!("\n    {d}"));
            }
        }
        return Err(Failure::Usage(msg));
    }

    let outcome = Outcome {
        mode,
        destination: endpoint.url.clone(),
        version: seq,
        added: added.len(),
        overwritten: overwrites.len(),
        skipped,
        deleted: deletes.len(),
        resumed: resuming,
        dry_run: opts.dry_run,
    };

    // 11. Dry run stops here, after printing the plan.
    if opts.dry_run {
        print_plan(opts, &outcome, &added, &overwrites, &deletes, &scan, seq);
        return Ok(outcome);
    }

    // 12. Confirmation.
    if !opts.assume_yes {
        confirm(&outcome, &added, &overwrites, &deletes)?;
    }

    // 13. TOCTOU guard: confirm no content file changed since it was
    //     checksummed. If one did, the manifest we are about to publish
    //     would disagree with the bytes — abort before writing anything,
    //     so a fresh re-run recomputes a consistent snapshot.
    if let Some(rel) = changed_since(root, &pre_stats).map_err(Failure::op)? {
        return Err(Failure::Operational(format!(
            "'{rel}' changed during the push (size/mtime differs from when it was \
             checksummed); re-run to publish a consistent snapshot."
        )));
    }

    // 14. Commit, in order: [abort] → begin → upload (sums last)
    //     → .publish_url → complete → delete orphans. The `begin` is
    //     skipped when resuming (it is already on the remote); the abort
    //     is written only in the AbortThenFresh case.
    let ts = httpdate::fmt_http_date(std::time::SystemTime::now());
    let first_ever = remote_log.events.is_empty();

    // The running log we extend and write back, plus the etag guarding
    // the next conditional write.
    let mut current_etag = remote_log_etag.clone();
    let mut log = match &resolution {
        // Resume: append `complete` directly onto the remote's open begin.
        Resolution::Resume { .. } => remote_log.clone(),
        // Fresh: base on local committed history (carries a LocalAhead
        // tail up when we overwrite the remote log).
        Resolution::Fresh => local_log.clone(),
        // Abort: stamp an `abort` onto the remote's open begin first, then
        // build fresh on top of it.
        Resolution::AbortThenFresh { aborted_seq } => {
            let mut aborted = remote_log.clone();
            aborted.events.push(Event::Abort {
                seq: *aborted_seq,
                ts: ts.clone(),
                actor: opts.actor.clone(),
                reason: Some("superseded by a fresh push (--abort-incomplete)".to_string()),
            });
            write_log(tx.as_ref(), &aborted, current_etag.as_deref(), false)?;
            current_etag = current_log_etag(tx.as_ref())?;
            aborted
        }
    };

    if !resuming {
        log.events.push(Event::Begin {
            seq,
            ts: ts.clone(),
            actor: opts.actor.clone(),
            cmd: opts.cmd.clone(),
            message: opts.message.clone(),
            overwrites: overwrites.clone(),
            added: added.clone(),
            deletes: deletes.clone(),
            sums: sums_digests.clone(),
            tool_version: env!("CARGO_PKG_VERSION").to_string(),
        });
        write_log(tx.as_ref(), &log, current_etag.as_deref(), first_ever)?;
        current_etag = current_log_etag(tx.as_ref())?;
    }

    // Upload changed content files first… (a resumed push re-puts only
    // what the classify step found missing or differing — idempotent).
    for rel in &to_upload {
        tx.put_file(rel, &root.join(rel.replace('/', std::path::MAIN_SEPARATOR_STR)))
            .map_err(map_transport)?;
    }
    // …then every directory's SHA256SUMS (sums last = the per-dir commit
    // signal a strict reader keys on).
    for dir_rel in &scan.content_dirs {
        let key = plan::sums_key(dir_rel);
        let bytes = local_sums.get(dir_rel).expect("sums computed").render().into_bytes();
        tx.put_bytes(&key, &bytes, None).map_err(map_transport)?;
    }

    // Refresh the remote binding (self-describing).
    tx.put_bytes(binding::PUBLISH_FILE, endpoint.url.as_bytes(), None)
        .map_err(map_transport)?;

    // Complete = the atomic instant version `seq` goes live.
    log.events.push(Event::Complete { seq, ts, sums: sums_digests.clone() });
    write_log(tx.as_ref(), &log, current_etag.as_deref(), false)?;

    // Delete orphans only after the new version is live, so a reader on
    // the prior version never sees its files vanish mid-update. The
    // removed keys are recorded on the begin event for audit.
    for key in &deletes {
        tx.delete(key).map_err(map_transport)?;
    }

    // 15. Mirror into the local pushlog; persist a staged binding. Both
    //     are written crash-atomically (temp + rename) so an interrupted
    //     mirror can't leave a torn local pushlog.
    atomic_write_local(&root.join(pushlog::PUSHLOG_FILE), log.render().as_bytes())
        .map_err(Failure::op)?;
    if matches!(bind, Binding::Staged(_)) {
        atomic_write_local(
            &root.join(binding::PUBLISH_FILE),
            format!("{}\n", endpoint.url).as_bytes(),
        )
        .map_err(Failure::op)?;
    }

    Ok(outcome)
}

// ─── helpers ────────────────────────────────────────────────────────

/// Per-directory remote SHA256SUMS, fetched and parsed (None if absent).
fn remote_log_or_dir_sums(
    tx: &dyn PushTransport,
    dir_rel: &str,
) -> Result<Option<checksums::ChecksumFile>, Failure> {
    let key = plan::sums_key(dir_rel);
    match tx.get(&key).map_err(map_transport)? {
        Some(bytes) => {
            let text = String::from_utf8_lossy(&bytes);
            Ok(Some(checksums::ChecksumFile::parse(&text).map_err(Failure::op)?))
        }
        None => Ok(None),
    }
}

/// Decide add/skip/overwrite for one file using the remote SHA256SUMS as
/// the content oracle, falling back to a HEAD existence probe.
fn classify_file(
    tx: &dyn PushTransport,
    rel: &str,
    _dir_rel: &str,
    local_digest: &str,
    remote_sums: &Option<checksums::ChecksumFile>,
) -> Result<Decision, Failure> {
    let (_dir, name) = split_rel(rel);
    if let Some(cf) = remote_sums {
        match cf.digest_of(name) {
            Some(remote_digest) if remote_digest == local_digest => return Ok(Decision::Skip),
            Some(remote_digest) => {
                return Ok(Decision::Overwrite { old_digest: format!("sha256:{remote_digest}") })
            }
            None => {} // not listed remotely — fall through to existence probe
        }
    }
    match tx.head(rel).map_err(map_transport)? {
        None => Ok(Decision::Add),
        // Present but we have no trustworthy remote digest → treat as an
        // overwrite (gated, safe) rather than risk a silent clobber.
        Some(_) => Ok(Decision::Overwrite { old_digest: "unknown".to_string() }),
    }
}

/// Read the remote `pushlog.jsonl` and its etag (for conditional writes).
fn read_remote_log(tx: &dyn PushTransport) -> Result<(Log, Option<String>), Failure> {
    let bytes = tx.get(pushlog::PUSHLOG_FILE).map_err(map_transport)?;
    let log = match &bytes {
        Some(b) => Log::parse(&String::from_utf8_lossy(b)).map_err(Failure::op)?,
        None => Log::default(),
    };
    let etag = current_log_etag(tx)?;
    Ok((log, etag))
}

fn current_log_etag(tx: &dyn PushTransport) -> Result<Option<String>, Failure> {
    Ok(tx.head(pushlog::PUSHLOG_FILE).map_err(map_transport)?.and_then(|o| o.etag))
}

fn read_local_log(root: &Path) -> Result<Log, Failure> {
    match std::fs::read_to_string(root.join(pushlog::PUSHLOG_FILE)) {
        Ok(text) => Log::parse(&text).map_err(Failure::op),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Log::default()),
        Err(e) => Err(Failure::op(e)),
    }
}

/// Write the log to the remote with an `If-Match` guard. `absent` means
/// "the remote log did not exist", which maps to a must-not-exist
/// precondition.
fn write_log(
    tx: &dyn PushTransport,
    log: &Log,
    if_match: Option<&str>,
    absent: bool,
) -> Result<(), Failure> {
    let cond = if absent { Some("") } else { if_match };
    match tx.put_bytes(pushlog::PUSHLOG_FILE, log.render().as_bytes(), cond) {
        Ok(()) => Ok(()),
        Err(PushError::PreconditionFailed) => Err(Failure::Operational(
            "the remote pushlog changed concurrently (another push raced this one); \
             re-run the push to re-converge."
                .to_string(),
        )),
        Err(e) => Err(map_transport(e)),
    }
}

/// Compute a directory's checksums in memory over exactly `names`,
/// without writing the file (used for dry-run so the working tree is
/// untouched).
fn compute_in_memory(dir: &Path, names: &[String]) -> std::io::Result<checksums::ChecksumFile> {
    let mut entries = Vec::new();
    for name in names {
        entries.push(checksums::ChecksumEntry {
            hex: checksums::sha256_file(&dir.join(name))?,
            name: name.clone(),
        });
    }
    Ok(checksums::ChecksumFile { entries })
}

fn map_transport(e: PushError) -> Failure {
    match e {
        PushError::PreconditionFailed => Failure::Operational(e.to_string()),
        PushError::Auth(m) => Failure::Operational(format!("authentication failed: {m}")),
        PushError::Other(m) => Failure::Operational(m),
    }
}

/// Probe whether the endpoint honors conditional writes. Creates a
/// throwaway key with a must-not-exist precondition, then attempts the
/// same again: a conforming store refuses the second with
/// [`PushError::PreconditionFailed`]. If the second *succeeds*, the store
/// ignores the precondition and cannot serialize concurrent pushers, so
/// the provenance log could be silently clobbered — refuse.
fn verify_conditional_writes(tx: &dyn PushTransport) -> Result<(), Failure> {
    const PROBE: &str = ".push-cas-probe";
    // Clear any probe left by a previous interrupted run.
    let _ = tx.delete(PROBE);
    match tx.put_bytes(PROBE, b"1", Some("")) {
        Ok(()) => {}
        // A leftover we couldn't delete — inconclusive; don't block.
        Err(PushError::PreconditionFailed) => {
            let _ = tx.delete(PROBE);
            return Ok(());
        }
        Err(e) => return Err(map_transport(e)),
    }
    let honored =
        matches!(tx.put_bytes(PROBE, b"2", Some("")), Err(PushError::PreconditionFailed));
    let _ = tx.delete(PROBE);
    if honored {
        Ok(())
    } else {
        Err(Failure::Operational(format!(
            "{} does not honor conditional writes (If-None-Match): concurrent pushes could \
             silently clobber the provenance log. Refusing — single provenance can't be \
             guaranteed on this endpoint.",
            tx.describe()
        )))
    }
}

/// Return the first content file (relative key) whose `(len, mtime)`
/// differs from the snapshot taken at checksum time — i.e. one that was
/// mutated during the push (TOCTOU). `None` if all are unchanged.
fn changed_since(
    root: &Path,
    pre_stats: &BTreeMap<String, (u64, std::time::SystemTime)>,
) -> std::io::Result<Option<String>> {
    for (rel, before) in pre_stats {
        let p = root.join(rel.replace('/', std::path::MAIN_SEPARATOR_STR));
        if checksums::len_mtime(&p)? != *before {
            return Ok(Some(rel.clone()));
        }
    }
    Ok(None)
}

/// Crash-atomic write of a local file (temp + rename on the same dir).
fn atomic_write_local(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.tmp.{}", std::process::id()));
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)
}

fn dir_join(root: &Path, dir_rel: &str) -> PathBuf {
    if dir_rel.is_empty() {
        root.to_path_buf()
    } else {
        root.join(dir_rel.replace('/', std::path::MAIN_SEPARATOR_STR))
    }
}

/// Split a forward-slashed relative file path into (dir, filename).
fn split_rel(rel: &str) -> (&str, &str) {
    match rel.rfind('/') {
        Some(i) => (&rel[..i], &rel[i + 1..]),
        None => ("", rel),
    }
}

fn display_dir(dir_rel: &str) -> &str {
    if dir_rel.is_empty() { "." } else { dir_rel }
}

fn confirm(
    outcome: &Outcome,
    added: &[String],
    overwrites: &[Overwrite],
    deletes: &[String],
) -> Result<(), Failure> {
    use std::io::IsTerminal;
    println!(
        "Push to {}\n  version {}  ({} new, {} overwrite, {} delete, {} unchanged)",
        outcome.destination,
        outcome.version,
        added.len(),
        overwrites.len(),
        deletes.len(),
        outcome.skipped
    );
    for d in deletes {
        println!("  - {d}");
    }
    if !std::io::stdin().is_terminal() {
        return Err(Failure::Usage(
            "refusing to push without confirmation in a non-interactive context; pass -y".to_string(),
        ));
    }
    print!("Proceed? [y/N] ");
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).map_err(Failure::op)?;
    if matches!(line.trim().to_lowercase().as_str(), "y" | "yes") {
        Ok(())
    } else {
        Err(Failure::Usage("aborted by user".to_string()))
    }
}

fn print_plan(
    opts: &Options,
    outcome: &Outcome,
    added: &[String],
    overwrites: &[Overwrite],
    deletes: &[String],
    scan: &plan::Scan,
    seq: u64,
) {
    println!("DRY RUN — nothing will be written.");
    println!("Source:      {} [{}]", opts.path.display(), outcome.mode.label());
    println!("Destination: {}", outcome.destination);
    if outcome.resumed {
        println!("Version:     {seq} (RESUMING an interrupted push at this seq)");
    } else {
        println!("Version:     {seq} (next pushlog seq)");
    }
    println!(
        "Concurrency: {} (upload streams)",
        opts.concurrency.max(1)
    );
    println!(
        "Plan:        {} new, {} overwrite, {} delete, {} unchanged across {} content dir(s)",
        added.len(),
        overwrites.len(),
        deletes.len(),
        outcome.skipped,
        scan.content_dirs.len()
    );
    for a in added {
        println!("  + {a}");
    }
    for o in overwrites {
        println!("  ~ {} (was {})", o.key, o.old_digest);
    }
    for d in deletes {
        println!("  - {d}");
    }
    if !overwrites.is_empty() || !deletes.is_empty() {
        match &opts.message {
            Some(m) => println!("Message:     {m}"),
            None => println!(
                "NOTE: overwrites/deletes present — a real push needs -m/--message."
            ),
        }
    }
}

// ─── CLI surface ────────────────────────────────────────────────────

#[cfg(feature = "cli")]
mod cli {
    use super::*;
    use clap::{Args, ValueEnum};

    #[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
    pub enum ChecksumMode {
        /// Recompute a stale or missing SHA256SUMS before pushing.
        Auto,
        /// Use the existing SHA256SUMS; stale/missing stops the push.
        Keep,
    }

    /// `vectordata datasets push` — push an already-good dataset or an
    /// ad-hoc directory to its bound remote.
    #[derive(Debug, Args)]
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
        #[arg(long, value_enum, default_value = "auto")]
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
            let token = self.token.or_else(|| std::env::var("VECTORDATA_PUSH_TOKEN").ok());
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
                cmd,
                actor,
            }
        }
    }

    /// Dispatch entry point. Returns a process exit code.
    pub fn run(args: PushArgs) -> i32 {
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
}

#[cfg(feature = "cli")]
pub use cli::{ChecksumMode, PushArgs, run};

#[cfg(test)]
mod tests;
