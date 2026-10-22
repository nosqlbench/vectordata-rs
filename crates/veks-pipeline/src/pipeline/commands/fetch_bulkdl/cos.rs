// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! IBM Cloud Object Storage (S3-compatible) client for bulk download.
//!
//! Implements the subset of the S3 REST protocol needed to enumerate and
//! download objects under a bucket prefix:
//!
//! - `ListObjectsV2` with paginated continuation tokens
//! - Signed `GET` request construction (path-style URLs)
//!
//! Requests are signed with AWS Signature Version 4 by veks-transport's
//! [`SigV4Signer`], just before each is sent (`docs/design/
//! srd-crate-boundaries.md` §10 Q6), and go through the process's
//! transport like every other download.
//!
//! Credentials and endpoint are read from environment variables matching the
//! AWS CLI convention so that the same `auth.sh` files used with the CLI work
//! unchanged: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
//! `AWS_SESSION_TOKEN` (optional), and `ENDPOINT` (the COS endpoint URL).
//!
//! Does NOT use the AWS SDK or any S3-specific client crate.

use std::env;
use std::sync::Arc;

use quick_xml::events::Event;
use quick_xml::reader::Reader;
use vectordata::transport::sigv4::{AwsCredentials, SigV4Signer};
use vectordata::transport::{Request, Transport};

/// Read the access key from the standard AWS environment variables.
///
/// `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are required;
/// `AWS_SESSION_TOKEN` is optional and, if present, is signed as the
/// `x-amz-security-token` header.
pub fn credentials_from_env() -> Result<AwsCredentials, String> {
    let access_key = env::var("AWS_ACCESS_KEY_ID")
        .map_err(|_| "AWS_ACCESS_KEY_ID environment variable not set".to_string())?;
    let secret_key = env::var("AWS_SECRET_ACCESS_KEY")
        .map_err(|_| "AWS_SECRET_ACCESS_KEY environment variable not set".to_string())?;
    let mut credentials = AwsCredentials::new(access_key, secret_key);
    if let Ok(token) = env::var("AWS_SESSION_TOKEN") {
        credentials = credentials.with_session_token(token);
    }
    Ok(credentials)
}

/// Endpoint + bucket + the SigV4 signer for an IBM COS workspace.
///
/// Constructed once per pipeline run and shared across worker threads via
/// `Arc`. Each request it builds is signed by the transport at the moment
/// it is sent (SigV4 signatures are valid for 15 minutes from the embedded
/// `x-amz-date`, so a request queued behind a long download is still
/// signed fresh).
#[derive(Debug, Clone)]
pub struct CosContext {
    /// Full endpoint URL, e.g. `https://s3.us-south.cloud-object-storage.appdomain.cloud`.
    pub endpoint: String,
    /// Bucket name.
    pub bucket: String,
    /// Signs every request for the endpoint's region.
    signer: Arc<SigV4Signer>,
}

impl CosContext {
    /// Build a context for `bucket` at `endpoint`, signing for `region`,
    /// validating that the endpoint is a parseable URL.
    pub fn new(
        endpoint: String,
        bucket: String,
        region: String,
        credentials: AwsCredentials,
    ) -> Result<Self, String> {
        host_from_endpoint(&endpoint)
            .ok_or_else(|| format!("invalid endpoint URL '{}'", endpoint))?;
        Ok(CosContext {
            endpoint,
            bucket,
            signer: Arc::new(SigV4Signer::s3(credentials, region)),
        })
    }

    /// A signed GET of an object key in the configured bucket.
    pub fn get_request(&self, key: &str) -> Request {
        let path = format!("/{}/{}", self.bucket, key);
        Request::get(self.url_for_path(&path, "")).signed(self.signer.clone())
    }

    /// Enumerate every object whose key begins with `prefix`, following all
    /// pagination continuation tokens. Returns `(key, size)` pairs.
    pub fn list_prefix(
        &self,
        transport: &Transport,
        prefix: &str,
    ) -> Result<Vec<(String, u64)>, String> {
        let mut all = Vec::new();
        let mut continuation: Option<String> = None;
        loop {
            let resp = transport
                .send(self.list_request(prefix, continuation.as_deref()))
                .map_err(|e| format!("ListObjectsV2 request failed: {}", e))?;
            let status = resp.status_line();
            let success = resp.is_success();
            let body = resp
                .text()
                .map_err(|e| format!("ListObjectsV2 body read failed: {}", e))?;
            if !success {
                return Err(format!("ListObjectsV2 HTTP {}: {}", status, body));
            }
            let page = parse_list_response(&body)?;
            all.extend(page.objects);
            if page.is_truncated {
                continuation = page.next_continuation_token;
                if continuation.is_none() {
                    return Err(
                        "IsTruncated=true but no NextContinuationToken in response".to_string(),
                    );
                }
            } else {
                break;
            }
        }
        Ok(all)
    }

    /// A signed ListObjectsV2 request: GET /{bucket}?list-type=2&prefix=...
    fn list_request(&self, prefix: &str, continuation: Option<&str>) -> Request {
        let path = format!("/{}", self.bucket);

        // Query parameters sorted by name, each k/v URI-encoded with the
        // strict (unreserved-only) rule — the canonical form SigV4 signs.
        let mut params: Vec<(&str, String)> = vec![
            ("list-type", "2".to_string()),
            ("prefix", prefix.to_string()),
        ];
        if let Some(token) = continuation {
            params.push(("continuation-token", token.to_string()));
        }
        params.sort_by(|a, b| a.0.cmp(b.0));
        let query = params
            .iter()
            .map(|(k, v)| format!("{}={}", encode_strict(k), encode_strict(v)))
            .collect::<Vec<_>>()
            .join("&");

        Request::get(self.url_for_path(&path, &query)).signed(self.signer.clone())
    }

    fn url_for_path(&self, path: &str, query: &str) -> String {
        let base = self.endpoint.trim_end_matches('/');
        if query.is_empty() {
            format!("{}{}", base, encode_path(path))
        } else {
            format!("{}{}?{}", base, encode_path(path), query)
        }
    }
}

fn host_from_endpoint(endpoint: &str) -> Option<String> {
    let stripped = endpoint
        .strip_prefix("https://")
        .or_else(|| endpoint.strip_prefix("http://"))?;
    let host = stripped.split('/').next()?;
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

/// Percent-encode a URI path while preserving `/` separators between segments.
fn encode_path(path: &str) -> String {
    path.split('/')
        .map(encode_strict)
        .collect::<Vec<_>>()
        .join("/")
}

/// RFC 3986 strict percent-encoding. Only the unreserved set passes through
/// unchanged: `A-Z a-z 0-9 - _ . ~`. SigV4 requires this exact rule for both
/// the canonical URI segments and the canonical query string.
fn encode_strict(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

// --- ListObjectsV2 XML parsing ---

/// One page of a `ListObjectsV2` response.
#[derive(Debug, Default)]
pub struct ListPage {
    /// `(key, size)` pairs from the `Contents` elements, in document order.
    pub objects: Vec<(String, u64)>,
    /// True if more pages remain.
    pub is_truncated: bool,
    /// Token to pass to the next request as `continuation-token`. Required
    /// when `is_truncated` is true.
    pub next_continuation_token: Option<String>,
}

/// Parse an S3 `ListObjectsV2` XML response into the elements we care about.
/// Ignores other tags (Name, Prefix, MaxKeys, KeyCount, CommonPrefixes,
/// Owner, ETag, LastModified, StorageClass).
pub fn parse_list_response(xml: &str) -> Result<ListPage, String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut page = ListPage::default();
    let mut path: Vec<String> = Vec::new();
    let mut text_buf = String::new();
    let mut current_key: Option<String> = None;
    let mut current_size: Option<u64> = None;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                let name = std::str::from_utf8(e.name().as_ref())
                    .map_err(|err| format!("non-UTF8 element name: {}", err))?
                    .to_string();
                path.push(name);
                text_buf.clear();
            }
            Ok(Event::Text(e)) => {
                let txt = e
                    .unescape()
                    .map_err(|err| format!("XML text decode error: {}", err))?;
                text_buf.push_str(&txt);
            }
            Ok(Event::End(e)) => {
                let name = std::str::from_utf8(e.name().as_ref())
                    .map_err(|err| format!("non-UTF8 element name: {}", err))?
                    .to_string();
                let parent: &str = if path.len() >= 2 {
                    path[path.len() - 2].as_str()
                } else {
                    ""
                };
                match (parent, name.as_str()) {
                    ("Contents", "Key") => {
                        current_key = Some(text_buf.clone());
                    }
                    ("Contents", "Size") => {
                        current_size = text_buf.parse().ok();
                    }
                    ("ListBucketResult", "Contents") => {
                        if let (Some(k), Some(s)) = (current_key.take(), current_size.take()) {
                            page.objects.push((k, s));
                        }
                    }
                    ("ListBucketResult", "IsTruncated") => {
                        page.is_truncated = text_buf.trim().eq_ignore_ascii_case("true");
                    }
                    ("ListBucketResult", "NextContinuationToken") => {
                        page.next_continuation_token = Some(text_buf.clone());
                    }
                    _ => {}
                }
                path.pop();
                text_buf.clear();
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parse error: {}", e)),
            _ => {}
        }
    }
    Ok(page)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectordata::transport::Authorization;

    #[test]
    fn encode_strict_only_unreserved_passes() {
        assert_eq!(encode_strict("abcXYZ-_.~09"), "abcXYZ-_.~09");
        assert_eq!(encode_strict(" "), "%20");
        assert_eq!(encode_strict("/"), "%2F");
        assert_eq!(encode_strict("a b/c"), "a%20b%2Fc");
        assert_eq!(encode_strict("résumé"), "r%C3%A9sum%C3%A9");
    }

    #[test]
    fn encode_path_preserves_slashes() {
        assert_eq!(encode_path("/bucket/key with space"), "/bucket/key%20with%20space");
        assert_eq!(encode_path("/a/b/c"), "/a/b/c");
    }

    #[test]
    fn host_from_endpoint_strips_scheme_and_path() {
        assert_eq!(
            host_from_endpoint("https://s3.us-south.cloud-object-storage.appdomain.cloud"),
            Some("s3.us-south.cloud-object-storage.appdomain.cloud".to_string()),
        );
        assert_eq!(
            host_from_endpoint("https://example.com:8080/path"),
            Some("example.com:8080".to_string()),
        );
        assert_eq!(host_from_endpoint("ftp://x"), None);
    }

    #[test]
    fn parse_list_response_basic() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>gtkbackup</Name>
  <Prefix>bulk-v0.8-1B/</Prefix>
  <KeyCount>2</KeyCount>
  <MaxKeys>1000</MaxKeys>
  <IsTruncated>false</IsTruncated>
  <Contents>
    <Key>bulk-v0.8-1B/file1.bin</Key>
    <LastModified>2024-01-01T00:00:00.000Z</LastModified>
    <ETag>"abc"</ETag>
    <Size>12345</Size>
    <StorageClass>STANDARD</StorageClass>
  </Contents>
  <Contents>
    <Key>bulk-v0.8-1B/sub/file2.bin</Key>
    <LastModified>2024-01-02T00:00:00.000Z</LastModified>
    <ETag>"def"</ETag>
    <Size>6789</Size>
    <StorageClass>STANDARD</StorageClass>
  </Contents>
</ListBucketResult>"#;
        let page = parse_list_response(xml).unwrap();
        assert_eq!(page.objects.len(), 2);
        assert_eq!(page.objects[0], ("bulk-v0.8-1B/file1.bin".to_string(), 12345));
        assert_eq!(page.objects[1], ("bulk-v0.8-1B/sub/file2.bin".to_string(), 6789));
        assert!(!page.is_truncated);
        assert!(page.next_continuation_token.is_none());
    }

    #[test]
    fn parse_list_response_truncated() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>true</IsTruncated>
  <NextContinuationToken>opaque-token-abc</NextContinuationToken>
  <Contents><Key>k1</Key><Size>100</Size></Contents>
</ListBucketResult>"#;
        let page = parse_list_response(xml).unwrap();
        assert!(page.is_truncated);
        assert_eq!(page.next_continuation_token.as_deref(), Some("opaque-token-abc"));
        assert_eq!(page.objects.len(), 1);
    }

    #[test]
    fn parse_list_response_handles_xml_entities() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult>
  <IsTruncated>false</IsTruncated>
  <Contents><Key>path/with &amp; ampersand.bin</Key><Size>1</Size></Contents>
</ListBucketResult>"#;
        let page = parse_list_response(xml).unwrap();
        assert_eq!(page.objects[0].0, "path/with & ampersand.bin");
    }

    /// The request a signer signs, as the transport would just before
    /// sending it.
    fn signed(req: &Request) -> Request {
        let mut wire = req.clone();
        let Authorization::Signed(signer) = &req.auth else { panic!("{:?} is not signed", req.auth) };
        signer.sign(&mut wire).unwrap();
        wire
    }

    #[test]
    fn signed_get_url_uses_path_style() {
        let ctx = CosContext::new(
            "https://s3.us-south.cloud-object-storage.appdomain.cloud".to_string(),
            "gtkbackup".to_string(),
            "us-south".to_string(),
            AwsCredentials::new("AKID", "SECRET"),
        ).unwrap();
        let req = ctx.get_request("bulk-v0.8-1B/file1.bin");
        assert_eq!(
            req.url,
            "https://s3.us-south.cloud-object-storage.appdomain.cloud/gtkbackup/bulk-v0.8-1B/file1.bin"
        );
        let wire = signed(&req);
        assert!(wire
            .header_value("authorization")
            .unwrap()
            .starts_with("AWS4-HMAC-SHA256 Credential=AKID/"));
        assert!(wire.header_value("x-amz-date").is_some());
        assert!(wire.header_value("x-amz-content-sha256").is_some());
    }

    #[test]
    fn signed_list_canonical_query_is_sorted() {
        let ctx = CosContext::new(
            "https://example.com".to_string(),
            "b".to_string(),
            "us-south".to_string(),
            AwsCredentials::new("AKID", "SECRET"),
        ).unwrap();
        let url = ctx.list_request("bulk-v0.8-1B/", Some("tok=en")).url;
        // Sorted by key: continuation-token, list-type, prefix
        // Each value strict-encoded: "tok=en" -> "tok%3Den", "bulk-v0.8-1B/" -> "bulk-v0.8-1B%2F"
        assert!(
            url.ends_with("?continuation-token=tok%3Den&list-type=2&prefix=bulk-v0.8-1B%2F"),
            "got url: {}", url
        );
    }
}
