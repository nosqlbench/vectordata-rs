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

use std::path::Path;

use reqwest::blocking::RequestBuilder;
use reqwest::StatusCode;

use super::{join_rel, PushError, PushTransport, RemoteObject};

pub struct HttpsTransport {
    base: String,
    token: Option<String>,
}

impl HttpsTransport {
    pub fn new(base: String, token: Option<String>) -> Self {
        HttpsTransport { base, token }
    }

    fn url(&self, rel: &str) -> String {
        join_rel(&self.base, rel)
    }

    /// Apply bearer auth if a token is configured.
    fn auth(&self, rb: RequestBuilder) -> RequestBuilder {
        match &self.token {
            Some(t) => rb.bearer_auth(t),
            None => rb,
        }
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

    fn list(&self, _prefix: &str) -> Result<Vec<String>, PushError> {
        // A bare REST endpoint has no portable object-listing verb, so we
        // cannot enumerate orphans. Fail loudly rather than silently skip.
        Err(PushError::Other(
            "--delete is not supported over generic https:// (no object listing); \
             use an s3:// or file:// endpoint for orphan cleanup"
                .to_string(),
        ))
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

fn auth(rel: &str, status: StatusCode) -> PushError {
    let where_ = if rel.is_empty() { "endpoint".to_string() } else { format!("'{rel}'") };
    PushError::Auth(format!(
        "{where_} rejected (HTTP {status}); set --token / VECTORDATA_PUSH_TOKEN with write access"
    ))
}
