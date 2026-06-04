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

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::OnceLock;

use reqwest::blocking::{RequestBuilder, Response};
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
}

impl HttpsTransport {
    pub fn new(base: String, token: Option<String>) -> Self {
        HttpsTransport { base, token, vecd: OnceLock::new() }
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
        let result = self.drive_upload(&upload_url, src, total);
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

    /// Send every chunk from `start_offset` to the end, retrying the whole
    /// remainder after a `HEAD`-resync on a transient failure.
    fn drive_upload(&self, upload_url: &str, src: &Path, total: u64) -> Result<(), PushError> {
        let mut offset = 0u64;
        let mut attempts = 0u32;
        loop {
            match self.send_chunks(upload_url, src, total, offset) {
                Ok(()) => return Ok(()),
                // Auth / CAS failures are terminal — no point resuming.
                Err(e @ (PushError::Auth(_) | PushError::PreconditionFailed)) => return Err(e),
                Err(e) => {
                    attempts += 1;
                    if attempts > MAX_RESUME_ATTEMPTS {
                        return Err(e);
                    }
                    // Resume from the server's acknowledged contiguous offset.
                    offset = self.upload_offset(upload_url)?;
                }
            }
        }
    }

    /// Stream `[start_offset, total)` of `src` as ≥2 sparse `PATCH` chunks.
    fn send_chunks(&self, upload_url: &str, src: &Path, total: u64, start_offset: u64) -> Result<(), PushError> {
        // A zero-byte object: one empty PATCH finalizes it.
        if total == 0 {
            self.put_chunk(upload_url, 0, Vec::new())?;
            return Ok(());
        }
        // At least two chunks: cap each at ⌈total/2⌉ (and at the chunk size).
        let chunk = RESUMABLE_CHUNK_BYTES.min(total.div_ceil(2)).max(1);
        let mut file = std::fs::File::open(src).map_err(other)?;
        if start_offset > 0 {
            file.seek(SeekFrom::Start(start_offset)).map_err(other)?;
        }
        let mut offset = start_offset;
        while offset < total {
            let len = chunk.min(total - offset);
            let mut buf = vec![0u8; len as usize];
            file.read_exact(&mut buf).map_err(other)?;
            self.put_chunk(upload_url, offset, buf)?;
            offset += len;
        }
        Ok(())
    }

    /// `PATCH <upload-url>` with `Upload-Offset` and the chunk body.
    /// Returns `(acked_offset, complete)`.
    fn put_chunk(&self, upload_url: &str, offset: u64, body: Vec<u8>) -> Result<(u64, bool), PushError> {
        let client = crate::transport::shared_client();
        let resp = self
            .auth(client.patch(upload_url))
            .header("Upload-Offset", offset.to_string())
            .body(body)
            .send()
            .map_err(other)?;
        match resp.status() {
            s if s.is_success() => {
                let acked = header_u64(&resp, "upload-offset").unwrap_or(offset);
                let complete = resp
                    .headers()
                    .get("upload-complete")
                    .and_then(|v| v.to_str().ok())
                    .map(|v| v.trim() == "?1")
                    .unwrap_or(false);
                Ok((acked, complete))
            }
            StatusCode::PRECONDITION_FAILED => Err(PushError::PreconditionFailed),
            s @ (StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => Err(auth(upload_url, s)),
            s => Err(PushError::Other(format!("PATCH {upload_url} @ {offset} -> HTTP {s}"))),
        }
    }

    /// `HEAD <upload-url>` → the server's acknowledged contiguous offset.
    fn upload_offset(&self, upload_url: &str) -> Result<u64, PushError> {
        let client = crate::transport::shared_client();
        let resp = self.auth(client.head(upload_url)).send().map_err(other)?;
        match resp.status() {
            s if s.is_success() => Ok(header_u64(&resp, "upload-offset").unwrap_or(0)),
            s @ (StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => Err(auth(upload_url, s)),
            s => Err(PushError::Other(format!("HEAD {upload_url} -> HTTP {s}"))),
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

/// Parse an unsigned-integer response header (the resumable structured
/// fields `Upload-Offset` / `Upload-Length`).
fn header_u64(resp: &Response, name: &str) -> Option<u64> {
    resp.headers().get(name)?.to_str().ok()?.trim().parse().ok()
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
        "{where_} rejected (HTTP {status}); set --token / VECTORDATA_PUSH_TOKEN with write access"
    ))
}
