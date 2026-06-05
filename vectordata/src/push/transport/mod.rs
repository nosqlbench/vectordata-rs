// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Write-side transport for `vectordata push`, dispatched on the URL
//! scheme — the mirror of the read-side `Storage`/transport factoring.
//!
//! A single [`open`] call picks the implementation for a `.publish_url`
//! endpoint; callers never choose a transport by hand. Three schemes are
//! supported:
//!
//! - `file://` / local — filesystem copy ([`local::LocalTransport`]);
//! - `https://` / `http://` — REST `PUT`/`HEAD`/`GET` ([`https::HttpsTransport`]);
//! - `s3://` — the AWS CLI ([`s3::S3Transport`]), matching `veks publish`.
//!
//! See `docs/design/push-command.md` — *Transports: dispatch on the URL
//! scheme*.

mod https;
mod local;
mod s3;

use super::binding::ParsedPublishUrl;

/// Metadata about a remote object, enough to detect existence and a
/// cheap size mismatch. Content-level overwrite decisions are made
/// against the remote `SHA256SUMS`, not this — ETags are opaque across
/// stores (multipart, MD5, etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteObject {
    pub size: u64,
    pub etag: Option<String>,
}

/// Errors a transport may return, distinguished so the caller can give
/// the right message (auth vs. concurrency vs. anything else).
#[derive(Debug)]
pub enum PushError {
    /// A conditional `put` failed its `If-Match` precondition — another
    /// writer changed the object since we read it.
    PreconditionFailed,
    /// The endpoint needs credentials we don't have, or rejected them.
    Auth(String),
    /// Anything else (I/O, protocol, malformed response).
    Other(String),
}

impl std::fmt::Display for PushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PushError::PreconditionFailed => write!(f, "precondition failed (object changed concurrently)"),
            PushError::Auth(m) => write!(f, "authentication: {m}"),
            PushError::Other(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for PushError {}

/// Per-invocation transport configuration drawn from the command line /
/// environment.
#[derive(Debug, Clone, Default)]
pub struct TransportOptions {
    /// Bearer token for generic `https://` endpoints (`--token` /
    /// `VECTORDATA_PUSH_TOKEN`).
    pub token: Option<String>,
    /// AWS profile name for S3 (`--profile`).
    pub profile: Option<String>,
    /// S3-compatible endpoint override (`--endpoint-url`).
    pub endpoint_url: Option<String>,
}

/// The write-side transport contract. `rel` is a forward-slashed path
/// relative to the publish root (`""` addresses the root itself); the
/// implementation composes it onto the bound endpoint.
pub trait PushTransport {
    /// Object metadata if it exists; `None` if absent.
    fn head(&self, rel: &str) -> Result<Option<RemoteObject>, PushError>;

    /// Fetch a (small) object in full; `None` if absent. Used for the
    /// `SHA256SUMS`, `pushlog.jsonl`, and remote `.publish_url`.
    fn get(&self, rel: &str) -> Result<Option<Vec<u8>>, PushError>;

    /// Upload a local file's contents to `rel`, creating or overwriting.
    fn put_file(&self, rel: &str, src: &std::path::Path) -> Result<(), PushError>;

    /// Upload `data` to `rel`. When `if_match` is `Some(etag)`, the put
    /// must fail with [`PushError::PreconditionFailed`] unless the
    /// current object's etag equals it; `Some("")` means "must not
    /// exist yet". When `None`, the put is unconditional.
    fn put_bytes(&self, rel: &str, data: &[u8], if_match: Option<&str>) -> Result<(), PushError>;

    /// Cheap auth/reachability check against the root, run before any
    /// bytes move so failures are fast and pre-mutation.
    fn preflight(&self) -> Result<(), PushError>;

    /// List object keys (relative to the publish root, forward-slashed)
    /// under `prefix`. Used only by `--delete` to find orphans. Returns
    /// [`PushError::Other`] on transports that cannot enumerate (a bare
    /// `https://` endpoint), so `--delete` fails loudly rather than
    /// silently skipping cleanup.
    fn list(&self, prefix: &str) -> Result<Vec<String>, PushError>;

    /// Remove the object at `rel`.
    fn delete(&self, rel: &str) -> Result<(), PushError>;

    /// Human-readable endpoint description for messages.
    fn describe(&self) -> String;
}

/// Select and construct the transport for a bound endpoint. `concurrency`
/// bounds parallel chunk uploads on transports that support them (the `vecd`
/// resumable path); others ignore it.
pub fn open(
    binding: &ParsedPublishUrl,
    opts: &TransportOptions,
    concurrency: u32,
) -> Result<Box<dyn PushTransport>, String> {
    match binding.scheme.as_str() {
        "file" => Ok(Box::new(local::LocalTransport::from_url(&binding.url)?)),
        "https" | "http" => Ok(Box::new(https::HttpsTransport::with_concurrency(
            binding.url.clone(),
            opts.token.clone(),
            concurrency,
        ))),
        "s3" => Ok(Box::new(s3::S3Transport::from_url(&binding.url, opts)?)),
        other => Err(format!("no push transport for scheme '{other}'")),
    }
}

/// Join a publish-root-relative key onto a base URL/path that ends in
/// `/`. An empty `rel` addresses the root.
pub(crate) fn join_rel(base: &str, rel: &str) -> String {
    if rel.is_empty() {
        base.to_string()
    } else {
        format!("{}{}", base, rel.trim_start_matches('/'))
    }
}
