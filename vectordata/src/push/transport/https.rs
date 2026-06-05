// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Generic `https://` / `http://` push transport — REST object
//! semantics: `PUT <base>/<rel>` to create/overwrite, `HEAD` for
//! existence/size/etag, `GET` for small artifacts, conditional `PUT`
//! via `If-Match` / `If-None-Match`.
//!
//! Auth is a bearer token (`--token` / `VECTORDATA_PUSH_TOKEN`) when
//! present, anonymous otherwise. The endpoint must honor object `PUT`;
//! a `405` surfaces as a clear "not a writable object store" error.
//!
//! **Content uploads adapt to the endpoint.** Against a generic REST host a
//! content file is one streaming `PUT`. Against a **`vecd`** endpoint (probed
//! once via `/-/whoami`) content files instead use the IETF "Resumable
//! Uploads for HTTP" protocol — `POST` create → ≥2 sparse `PATCH` chunks →
//! finalize-on-full — which streams in bounded chunks, advertises a
//! resumable offset, and recovers from a dropped connection via `HEAD`. The
//! resumable path is taken for *every* content file (even small ones, split
//! into ≥2 chunks) so it gets continuous exercise. The small control
//! artifacts (`SHA256SUMS`, `.publish_url`, the conditional `pushlog`) stay
//! plain `PUT` via [`HttpsTransport::put_bytes`].

use std::io::{IsTerminal, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;

use reqwest::blocking::RequestBuilder;
use reqwest::StatusCode;

use super::{join_rel, PushError, PushTransport, RemoteObject};

/// Target size of each resumable `PATCH` chunk. A file is always split into
/// at least two chunks (so the resumable path is exercised even for small
/// files); larger files split into ~this-sized pieces.
const RESUMABLE_CHUNK_BYTES: u64 = 4 * 1024 * 1024;

/// How many times to re-`HEAD`-and-resume a single file's upload before
/// giving up on a transient failure.
const MAX_RESUME_ATTEMPTS: u32 = 3;

pub struct HttpsTransport {
    base: String,
    token: Option<String>,
    /// Memoized capability probe: is the endpoint a `vecd` server (so the
    /// resumable-upload protocol is available)?
    vecd: OnceLock<bool>,
    /// Max concurrent `PATCH` chunks per file (resumable path). vecd accepts
    /// sparse, out-of-order, disjoint chunks, so a single large object can
    /// saturate aggregate bandwidth across many connections even when each
    /// connection is individually rate-limited. 1 = sequential.
    concurrency: usize,
}

impl HttpsTransport {
    /// Construct a transport with a max concurrent-`PATCH` count for the
    /// resumable upload path (clamped to ≥1; 1 = sequential).
    pub fn with_concurrency(base: String, token: Option<String>, concurrency: u32) -> Self {
        HttpsTransport {
            base,
            token,
            vecd: OnceLock::new(),
            concurrency: (concurrency as usize).max(1),
        }
    }

    fn url(&self, rel: &str) -> String {
        join_rel(&self.base, rel)
    }

    /// Resolve an absolute server path (e.g. an upload `Location`) against
    /// the endpoint's origin — `http://host:port` + the absolute path.
    fn origin_join(&self, abs_path: &str) -> Result<String, PushError> {
        let base = reqwest::Url::parse(&self.base)
            .map_err(|e| PushError::Other(format!("invalid endpoint URL '{}': {e}", self.base)))?;
        base.join(abs_path)
            .map(|u| u.to_string())
            .map_err(|e| PushError::Other(format!("joining '{abs_path}' onto '{}': {e}", self.base)))
    }

    /// Apply bearer auth if a token is configured.
    fn auth(&self, rb: RequestBuilder) -> RequestBuilder {
        match &self.token {
            Some(t) => rb.bearer_auth(t),
            None => rb,
        }
    }

    /// Is this endpoint a `vecd` server? Probed once (anonymously-capable
    /// `/-/whoami`, which self-identifies) and memoized. A failed or
    /// ambiguous probe answers `false`, so we simply fall back to plain
    /// `PUT` — which a `vecd` endpoint also accepts.
    fn is_vecd(&self) -> bool {
        *self.vecd.get_or_init(|| {
            let Ok(url) = self.origin_join("/-/whoami") else { return false };
            let client = crate::transport::shared_client();
            match self.auth(client.get(&url)).send() {
                Ok(resp) if resp.status().is_success() => {
                    resp.text().map(|b| b.contains("\"endpoint\":\"vecd\"")).unwrap_or(false)
                }
                _ => false,
            }
        })
    }
}

impl PushTransport for HttpsTransport {
    fn head(&self, rel: &str) -> Result<Option<RemoteObject>, PushError> {
        let client = crate::transport::shared_client();
        let resp = self.auth(client.head(self.url(rel))).send().map_err(other)?;
        match resp.status() {
            s if s.is_success() => {
                let size = resp
                    .headers()
                    .get(reqwest::header::CONTENT_LENGTH)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                let etag = resp
                    .headers()
                    .get(reqwest::header::ETAG)
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.trim_matches('"').to_string());
                Ok(Some(RemoteObject { size, etag }))
            }
            StatusCode::NOT_FOUND => Ok(None),
            s @ (StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => Err(auth(rel, s)),
            s => Err(PushError::Other(format!("HEAD {} -> HTTP {s}", self.url(rel)))),
        }
    }

    fn get(&self, rel: &str) -> Result<Option<Vec<u8>>, PushError> {
        let client = crate::transport::shared_client();
        let resp = self.auth(client.get(self.url(rel))).send().map_err(other)?;
        match resp.status() {
            s if s.is_success() => {
                // Capture the advertised length before consuming the body,
                // then verify we received all of it. A truncated response
                // (connection reset mid-body) must error here — silently
                // returning a short body would let a partial pushlog parse
                // as a valid shorter log and misclassify provenance.
                let advertised = resp
                    .headers()
                    .get(reqwest::header::CONTENT_LENGTH)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok());
                let body = resp.bytes().map_err(other)?;
                if let Some(len) = advertised
                    && body.len() as u64 != len
                {
                    return Err(PushError::Other(format!(
                        "GET {} returned {} bytes but Content-Length was {len} (truncated response)",
                        self.url(rel),
                        body.len(),
                    )));
                }
                Ok(Some(body.to_vec()))
            }
            StatusCode::NOT_FOUND => Ok(None),
            s @ (StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => Err(auth(rel, s)),
            s => Err(PushError::Other(format!("GET {} -> HTTP {s}", self.url(rel)))),
        }
    }

    fn put_file(&self, rel: &str, src: &Path) -> Result<(), PushError> {
        // Against a vecd endpoint, drive the resumable protocol (chunked,
        // resumable, ≥2 splits). Against a generic REST host, one streaming
        // PUT — the body streams off disk, so large files are fine here too.
        if self.is_vecd() {
            return self.resumable_put_file(rel, src);
        }
        let file = std::fs::File::open(src).map_err(other)?;
        let client = crate::transport::shared_client();
        let resp = self
            .auth(client.put(self.url(rel)))
            .body(reqwest::blocking::Body::from(file))
            .send()
            .map_err(other)?;
        self.check_put(rel, resp)
    }

    fn put_bytes(&self, rel: &str, data: &[u8], if_match: Option<&str>) -> Result<(), PushError> {
        let client = crate::transport::shared_client();
        let mut rb = self.auth(client.put(self.url(rel))).body(data.to_vec());
        match if_match {
            Some("") => rb = rb.header(reqwest::header::IF_NONE_MATCH, "*"),
            Some(etag) => rb = rb.header(reqwest::header::IF_MATCH, format!("\"{etag}\"")),
            None => {}
        }
        let resp = rb.send().map_err(other)?;
        self.check_put(rel, resp)
    }

    fn preflight(&self) -> Result<(), PushError> {
        let client = crate::transport::shared_client();
        let resp = self.auth(client.head(&self.base)).send().map_err(other)?;
        match resp.status() {
            s @ (StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => Err(auth("", s)),
            _ => Ok(()), // 404 at the root is fine — nothing published yet
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, PushError> {
        // A `vecd` endpoint exposes a listing via `GET <prefix>?list`
        // returning `{"keys":[{"key":..,"etag":..}]}`. We attempt it and,
        // on a well-formed JSON response, enumerate the keys — enabling
        // push `--delete` over `https://vecd-host/…`. A generic REST host
        // has no such verb (404, or a non-JSON 200): there we fail loudly
        // rather than risk deleting against an endpoint we can't enumerate.
        let url = format!("{}?list", self.url(prefix));
        let client = crate::transport::shared_client();
        let resp = self.auth(client.get(&url)).send().map_err(other)?;
        match resp.status() {
            s if s.is_success() => {
                let body = resp.bytes().map_err(other)?;
                parse_list_keys(&body).ok_or_else(unsupported_list)
            }
            s @ (StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => Err(auth(prefix, s)),
            _ => Err(unsupported_list()),
        }
    }

    fn delete(&self, rel: &str) -> Result<(), PushError> {
        let client = crate::transport::shared_client();
        let resp = self.auth(client.delete(self.url(rel))).send().map_err(other)?;
        match resp.status() {
            s if s.is_success() || s == StatusCode::NOT_FOUND => Ok(()),
            s @ (StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => Err(auth(rel, s)),
            s => Err(PushError::Other(format!("DELETE {} -> HTTP {s}", self.url(rel)))),
        }
    }

    fn describe(&self) -> String {
        self.base.clone()
    }
}

// ── resumable upload driver (vecd endpoints) ────────────────────────
impl HttpsTransport {
    /// Upload a content file to a `vecd` endpoint via the resumable
    /// protocol: `POST` to create the upload, stream it as ≥2 sparse
    /// `PATCH` chunks, and let the server finalize-on-full. On a transient
    /// failure it `HEAD`s for the acked offset and resumes; on a permanent
    /// failure it abandons the upload (best-effort `DELETE`).
    fn resumable_put_file(&self, rel: &str, src: &Path) -> Result<(), PushError> {
        let total = std::fs::metadata(src).map_err(other)?.len();
        let upload_url = self.create_upload(rel, total)?;
        let result = self.drive_upload(rel, &upload_url, src, total);
        if result.is_err() {
            // Free the staging blob + state we won't be completing.
            let _ = self.delete_upload(&upload_url);
        }
        result
    }

    /// `POST <rel>` with `Upload-Length` → the absolute upload URL parsed
    /// from `Location`.
    fn create_upload(&self, rel: &str, total: u64) -> Result<String, PushError> {
        let client = crate::transport::shared_client();
        let resp = self
            .auth(client.post(self.url(rel)))
            .header("Upload-Length", total.to_string())
            .send()
            .map_err(other)?;
        match resp.status() {
            StatusCode::CREATED | StatusCode::OK => {
                let loc = resp
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| {
                        PushError::Other(format!("creating upload for '{rel}' returned no Location"))
                    })?;
                self.origin_join(loc)
            }
            s @ (StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => Err(auth(rel, s)),
            s => Err(PushError::Other(format!("POST {} -> HTTP {s}", self.url(rel)))),
        }
    }

    /// Upload all chunks, retrying only the chunks that have **not** yet been
    /// acknowledged after a transient failure.
    ///
    /// vecd stores every chunk durably and idempotently the moment its `PATCH`
    /// returns `200`, independent of order, and finalizes when the contiguous
    /// prefix reaches `Upload-Length`. So a transient drop only loses the
    /// chunks that were in flight; the already-`200`'d chunks remain staged.
    /// We track per-chunk acks locally and re-send just the missing ones —
    /// rather than re-sending everything past the server's *contiguous* offset,
    /// which (with out-of-order parallel chunks) could be far behind the bytes
    /// already stored. Once every chunk is acked the contiguous prefix equals
    /// `Upload-Length` and the server has finalized.
    fn drive_upload(&self, rel: &str, upload_url: &str, src: &Path, total: u64) -> Result<(), PushError> {
        // A zero-byte object: one empty PATCH finalizes it.
        if total == 0 {
            return self.put_chunk(upload_url, 0, Vec::new());
        }
        let plan = chunk_plan(total);
        let acked: Vec<AtomicBool> = (0..plan.len()).map(|_| AtomicBool::new(false)).collect();

        // On an interactive terminal, a background thread paints a braille
        // readout of which file fractions the server has acknowledged — lit
        // pips track the (possibly disjoint) acked chunk offsets. Off when
        // stderr isn't a TTY, so logs and tests stay clean.
        let live = std::io::stderr().is_terminal();
        let stop = AtomicBool::new(false);
        let result = std::thread::scope(|scope| {
            if live {
                scope.spawn(|| {
                    while !stop.load(Ordering::Relaxed) {
                        paint_progress(rel, &plan, &acked);
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                });
            }
            let r = self.upload_with_resume(upload_url, src, &plan, &acked);
            stop.store(true, Ordering::Relaxed);
            r
        });
        if live {
            paint_progress(rel, &plan, &acked); // final frame
            eprintln!();
        }
        result
    }

    /// The retry loop: upload all chunks, re-sending only the ones not yet
    /// acked after a transient failure (see [`Self::send_pending`]).
    fn upload_with_resume(
        &self,
        upload_url: &str,
        src: &Path,
        plan: &[(u64, u64)],
        acked: &[AtomicBool],
    ) -> Result<(), PushError> {
        let mut attempts = 0u32;
        loop {
            let pending: Vec<usize> =
                (0..plan.len()).filter(|&i| !acked[i].load(Ordering::Relaxed)).collect();
            if pending.is_empty() {
                return Ok(());
            }
            match self.send_pending(upload_url, src, plan, &pending, acked) {
                Ok(()) => return Ok(()),
                // Auth / CAS failures are terminal — no point resuming.
                Err(e @ (PushError::Auth(_) | PushError::PreconditionFailed)) => return Err(e),
                Err(e) => {
                    attempts += 1;
                    if attempts > MAX_RESUME_ATTEMPTS {
                        return Err(e);
                    }
                    // Loop: `pending` is recomputed from `acked`, so the retry
                    // re-sends only the chunks that never got a `200`.
                }
            }
        }
    }

    /// `PATCH` the `pending` chunks (indices into `plan`) concurrently,
    /// recording each chunk's offset as acked on success.
    ///
    /// With `concurrency > 1` the chunks ride a bounded pool of worker threads
    /// — vecd accepts disjoint, out-of-order, concurrent PATCHes — so one large
    /// object saturates aggregate bandwidth across many connections even when
    /// each connection is individually capped. Workers claim indices lowest-
    /// first off a shared cursor; the first error halts further dispatch and is
    /// returned (the caller retries only the still-unacked chunks).
    fn send_pending(
        &self,
        upload_url: &str,
        src: &Path,
        plan: &[(u64, u64)],
        pending: &[usize],
        acked: &[AtomicBool],
    ) -> Result<(), PushError> {
        let workers = self.concurrency.min(pending.len()).max(1);
        if workers == 1 {
            // Sequential fast path (also the single-chunk / single-gap case).
            for &i in pending {
                let (off, len) = plan[i];
                self.put_chunk(upload_url, off, read_at(src, off, len)?)?;
                acked[i].store(true, Ordering::Relaxed);
            }
            return Ok(());
        }

        // Parallel path: a scoped thread pool shares `&self`, `src`, `plan`,
        // `pending`, and `acked` by reference (no Arc). The cursor hands out
        // `pending` slots lowest-first, one chunk in flight per worker.
        let next = AtomicUsize::new(0);
        let first_err: std::sync::Mutex<Option<PushError>> = std::sync::Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    if first_err.lock().unwrap().is_some() {
                        return;
                    }
                    let slot = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&i) = pending.get(slot) else { return };
                    let (off, len) = plan[i];
                    match read_at(src, off, len).and_then(|buf| self.put_chunk(upload_url, off, buf)) {
                        Ok(()) => acked[i].store(true, Ordering::Relaxed),
                        Err(e) => {
                            let mut held = first_err.lock().unwrap();
                            if held.is_none() {
                                *held = Some(e);
                            }
                            return;
                        }
                    }
                });
            }
        });
        match first_err.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// `PATCH <upload-url>` with `Upload-Offset` and the chunk body. A `2xx`
    /// means the chunk is durably staged; the server's contiguous-offset /
    /// complete headers are advisory (the client tracks per-chunk acks
    /// itself), so they are not consumed here.
    fn put_chunk(&self, upload_url: &str, offset: u64, body: Vec<u8>) -> Result<(), PushError> {
        let client = crate::transport::shared_client();
        let resp = self
            .auth(client.patch(upload_url))
            .header("Upload-Offset", offset.to_string())
            .body(body)
            .send()
            .map_err(other)?;
        match resp.status() {
            s if s.is_success() => Ok(()),
            StatusCode::PRECONDITION_FAILED => Err(PushError::PreconditionFailed),
            s @ (StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => Err(auth(upload_url, s)),
            s => Err(PushError::Other(format!("PATCH {upload_url} @ {offset} -> HTTP {s}"))),
        }
    }

    /// `DELETE <upload-url>` — abandon an upload (best-effort cleanup).
    fn delete_upload(&self, upload_url: &str) -> Result<(), PushError> {
        let client = crate::transport::shared_client();
        self.auth(client.delete(upload_url)).send().map_err(other)?;
        Ok(())
    }
}

impl HttpsTransport {
    fn check_put(&self, rel: &str, resp: reqwest::blocking::Response) -> Result<(), PushError> {
        match resp.status() {
            s if s.is_success() => Ok(()),
            StatusCode::PRECONDITION_FAILED => Err(PushError::PreconditionFailed),
            s @ (StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => Err(auth(rel, s)),
            StatusCode::METHOD_NOT_ALLOWED => Err(PushError::Other(format!(
                "{} does not accept object PUT (HTTP 405); not a writable object store?",
                self.url(rel)
            ))),
            // Quota exceeded — surface vecd's `X-Vecd-Quota` (the limit in
            // bytes) and any hint body instead of a bare "HTTP 507", so the
            // user knows it's a quota wall and what to do about it.
            StatusCode::INSUFFICIENT_STORAGE => {
                let quota = resp
                    .headers()
                    .get("x-vecd-quota")
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string);
                let body = resp.text().unwrap_or_default();
                let hint = body.trim();
                let limit = match quota {
                    Some(q) => format!(" (namespace quota {q} bytes)"),
                    None => " (namespace quota exceeded)".to_string(),
                };
                let detail = if hint.is_empty() { String::new() } else { format!(": {hint}") };
                Err(PushError::Other(format!(
                    "PUT {} -> HTTP 507 Insufficient Storage{limit}{detail}",
                    self.url(rel)
                )))
            }
            // A redirect on a PUT (classically an S3 wrong-region 301/307)
            // can't be followed with a streaming body — surface it clearly
            // instead of as an opaque failure.
            s if s.is_redirection() => Err(PushError::Other(format!(
                "PUT {} was redirected (HTTP {s}); point at the correct regional endpoint, \
                 or use an s3:// binding so credentials and region are resolved for you",
                self.url(rel)
            ))),
            s => Err(PushError::Other(format!("PUT {} -> HTTP {s}", self.url(rel)))),
        }
    }
}

fn other<E: std::fmt::Display>(e: E) -> PushError {
    PushError::Other(e.to_string())
}

/// The fixed chunk plan for an object of `total` bytes: `(offset, len)` pairs
/// covering `[0, total)`. At least two chunks (each ≤ `RESUMABLE_CHUNK_BYTES`
/// and ≤ ⌈total/2⌉) so the resumable path is exercised even for small files.
/// `total > 0` is required (the caller handles the empty object directly).
fn chunk_plan(total: u64) -> Vec<(u64, u64)> {
    let chunk = RESUMABLE_CHUNK_BYTES.min(total.div_ceil(2)).max(1);
    let mut plan = Vec::new();
    let mut offset = 0;
    while offset < total {
        let len = chunk.min(total - offset);
        plan.push((offset, len));
        offset += len;
    }
    plan
}

/// Number of braille cells in the upload progress bar (8 pips each).
const PROGRESS_CELLS: usize = 10;
/// Total pips across the bar — one per 1/PIPS fraction of the file.
const PROGRESS_PIPS: u64 = (PROGRESS_CELLS * 8) as u64;

/// Render upload progress as a [`PROGRESS_CELLS`]-cell braille bar
/// ([`PROGRESS_PIPS`] pips). Pip `i` covers the byte fraction
/// `[i/PIPS, (i+1)/PIPS)` of `total` and is lit only when that *whole*
/// fraction lies within an acknowledged (server-buffered) byte range. Because
/// chunks are uploaded in parallel and out of order, the lit pips can form
/// several disjoint runs — e.g. the contiguous frontier plus a later chunk
/// that finished early shows as two separate filled segments.
///
/// `acked` is a list of `(start, end)` half-open byte ranges already confirmed
/// stored; it need not be sorted or merged.
fn braille_progress(total: u64, acked: &[(u64, u64)]) -> String {
    // Within a cell the 8 pips map to braille dots column-major: the left
    // column top→bottom (dots 1,2,3,7) then the right column (4,5,6,8), so a
    // filling fraction reads as the cell darkening left-to-right.
    const DOT_BITS: [u32; 8] = [0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80];

    let pip_lit = |i: u64| -> bool {
        if total == 0 {
            return true; // a zero-byte object is trivially complete
        }
        let lo = i * total / PROGRESS_PIPS;
        let hi = (i + 1) * total / PROGRESS_PIPS;
        if hi <= lo {
            return true; // empty pip span (more pips than bytes) — treat as lit
        }
        acked.iter().any(|&(s, e)| s <= lo && hi <= e)
    };

    let mut out = String::with_capacity(PROGRESS_CELLS);
    for cell in 0..PROGRESS_CELLS as u64 {
        let mut bits = 0u32;
        for pip in 0..8u64 {
            if pip_lit(cell * 8 + pip) {
                bits |= DOT_BITS[pip as usize];
            }
        }
        out.push(char::from_u32(0x2800 + bits).unwrap());
    }
    out
}

/// Paint one frame of the per-file upload readout to stderr (carriage-return
/// overwritten): `\r  <rel>  <braille bar>  <pct>%`. Derives the acked byte
/// ranges from the per-chunk ack flags + the chunk plan.
fn paint_progress(rel: &str, plan: &[(u64, u64)], acked: &[AtomicBool]) {
    let total = plan.last().map(|&(o, l)| o + l).unwrap_or(0);
    let ranges: Vec<(u64, u64)> = plan
        .iter()
        .enumerate()
        .filter(|(i, _)| acked[*i].load(Ordering::Relaxed))
        .map(|(_, &(o, l))| (o, o + l))
        .collect();
    let done: u64 = ranges.iter().map(|(s, e)| e - s).sum();
    let pct = if total > 0 { done * 100 / total } else { 100 };
    eprint!("\r  {rel}  {}  {pct:>3}%", braille_progress(total, &ranges));
    let _ = std::io::stderr().flush();
}

/// Read exactly `len` bytes at `offset` from `src`. Opens a fresh handle per
/// call so parallel chunk workers never share a file cursor (portable; no
/// positional-read platform shims).
fn read_at(src: &Path, offset: u64, len: u64) -> Result<Vec<u8>, PushError> {
    let mut file = std::fs::File::open(src).map_err(other)?;
    file.seek(SeekFrom::Start(offset)).map_err(other)?;
    let mut buf = vec![0u8; len as usize];
    file.read_exact(&mut buf).map_err(other)?;
    Ok(buf)
}

/// The error for an endpoint that can't enumerate objects (non-`vecd`
/// `https`): `--delete` can't be supported there.
fn unsupported_list() -> PushError {
    PushError::Other(
        "--delete is not supported over this https:// endpoint (no object listing); \
         it works against a vecd endpoint, or an s3:// / file:// endpoint"
            .to_string(),
    )
}

/// Parse vecd's `?list` JSON (`{"keys":[{"key":..,"etag":..}]}`) into the
/// list of keys. Returns `None` if the body is not that shape (so a
/// generic 200 response falls back to "unsupported").
fn parse_list_keys(body: &[u8]) -> Option<Vec<String>> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let arr = v.get("keys")?.as_array()?;
    Some(arr.iter().filter_map(|e| e.get("key").and_then(|k| k.as_str()).map(String::from)).collect())
}

fn auth(rel: &str, status: StatusCode) -> PushError {
    let where_ = if rel.is_empty() { "endpoint".to_string() } else { format!("'{rel}'") };
    PushError::Auth(format!(
        "{where_} rejected (HTTP {status}); authenticate with `vectordata login <endpoint>` \
         (or set --token / VECTORDATA_PUSH_TOKEN) using a credential that has write access"
    ))
}

#[cfg(test)]
mod progress_tests {
    use super::{braille_progress, chunk_plan};

    const BLANK: char = '\u{2800}'; // no dots
    const FULL: char = '\u{28FF}'; // all 8 dots

    #[test]
    fn empty_is_all_blank() {
        let bar = braille_progress(8000, &[]);
        assert_eq!(bar.chars().count(), 10);
        assert!(bar.chars().all(|c| c == BLANK), "got {bar:?}");
    }

    #[test]
    fn complete_is_all_full() {
        let bar = braille_progress(8000, &[(0, 8000)]);
        assert!(bar.chars().all(|c| c == FULL), "got {bar:?}");
    }

    #[test]
    fn first_half_fills_first_five_cells() {
        // 40 of 80 pips → the first 5 of 10 cells are full, the rest blank.
        let bar: Vec<char> = braille_progress(8000, &[(0, 4000)]).chars().collect();
        assert!(bar[..5].iter().all(|&c| c == FULL), "front: {bar:?}");
        assert!(bar[5..].iter().all(|&c| c == BLANK), "back: {bar:?}");
    }

    #[test]
    fn disjoint_ranges_show_as_separate_runs() {
        // Two acked regions with a gap — the bar must light two separate
        // segments, never the gap between them (the parallel out-of-order
        // case the readout is meant to reveal).
        let total = 8000;
        let bar: Vec<char> = braille_progress(total, &[(0, 800), (4000, 4800)]).chars().collect();
        assert_ne!(bar[0], BLANK, "first region should light: {bar:?}");
        assert_eq!(bar[2], BLANK, "the gap must stay dark: {bar:?}");
        assert_ne!(bar[5], BLANK, "second region should light: {bar:?}");
    }

    #[test]
    fn matches_a_parallel_chunk_acked_set() {
        // A 32 MiB object's chunk plan, with only the first and third chunks
        // acked — the bar shows two runs separated by the un-acked 2nd chunk.
        let total = 32 * 1024 * 1024;
        let plan = chunk_plan(total);
        assert!(plan.len() >= 4, "expected several chunks, got {}", plan.len());
        let acked = [plan[0], plan[2]]; // (offset,len) pairs are (start, len)…
        let ranges: Vec<(u64, u64)> = acked.iter().map(|&(o, l)| (o, o + l)).collect();
        let bar = braille_progress(total, &ranges);
        // Chunk 0 region lit, chunk 1 region dark, chunk 2 region lit.
        assert_ne!(bar.chars().next().unwrap(), BLANK);
        assert!(bar.chars().any(|c| c == BLANK), "the un-acked middle chunk must leave a gap: {bar:?}");
    }
}
