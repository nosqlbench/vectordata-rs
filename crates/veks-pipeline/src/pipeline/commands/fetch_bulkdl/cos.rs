// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! IBM Cloud Object Storage (S3-compatible) client for bulk download.
//!
//! Implements the subset of the S3 REST protocol needed to enumerate and
//! download objects under a bucket prefix:
//!
//! - AWS Signature Version 4 request signing (header-based)
//! - `ListObjectsV2` with paginated continuation tokens
//! - Signed `GET` and `HEAD` request construction
//!
//! Credentials and endpoint are read from environment variables matching the
//! AWS CLI convention so that the same `auth.sh` files used with the CLI work
//! unchanged: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
//! `AWS_SESSION_TOKEN` (optional), and `ENDPOINT` (the COS endpoint URL).
//!
//! Does NOT use the AWS SDK or any S3-specific client crate. The transport is
//! plain HTTPS via reqwest with `Authorization` headers computed in this
//! module.

use std::env;

use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use sha2::{Digest, Sha256};

/// SHA-256 hex digest of the empty byte string. SigV4 requires this as the
/// payload-hash value for any request with no body (GET, HEAD, ListObjectsV2).
const SHA256_EMPTY: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Static credentials sourced from the AWS environment variables.
#[derive(Debug, Clone)]
pub struct CosCredentials {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
}

impl CosCredentials {
    /// Read credentials from the standard AWS environment variables.
    ///
    /// `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are required;
    /// `AWS_SESSION_TOKEN` is optional and, if present, is signed as the
    /// `x-amz-security-token` header.
    pub fn from_env() -> Result<Self, String> {
        let access_key = env::var("AWS_ACCESS_KEY_ID")
            .map_err(|_| "AWS_ACCESS_KEY_ID environment variable not set".to_string())?;
        let secret_key = env::var("AWS_SECRET_ACCESS_KEY")
            .map_err(|_| "AWS_SECRET_ACCESS_KEY environment variable not set".to_string())?;
        let session_token = env::var("AWS_SESSION_TOKEN").ok();
        Ok(CosCredentials { access_key, secret_key, session_token })
    }
}

/// Endpoint + bucket + region + credentials for an IBM COS workspace.
///
/// Constructed once per pipeline run and shared across worker threads via
/// `Arc`. Per-request signatures are computed inside the `signed_*` methods,
/// using the system clock at call time (SigV4 signatures are valid for 15
/// minutes from the embedded `x-amz-date`, so workers must sign at the moment
/// of the GET, not at job-enqueue time).
#[derive(Debug, Clone)]
pub struct CosContext {
    /// Full endpoint URL, e.g. `https://s3.us-south.cloud-object-storage.appdomain.cloud`.
    pub endpoint: String,
    /// Bucket name.
    pub bucket: String,
    /// SigV4 region string, e.g. `us-south`.
    pub region: String,
    /// Credentials. Cloned by value because they are tiny strings.
    pub credentials: CosCredentials,
    /// Host header derived once from `endpoint`, e.g. `s3.us-south.cloud-object-storage.appdomain.cloud`.
    host: String,
}

impl CosContext {
    /// Build a context, validating that the endpoint is a parseable URL.
    pub fn new(
        endpoint: String,
        bucket: String,
        region: String,
        credentials: CosCredentials,
    ) -> Result<Self, String> {
        let host = host_from_endpoint(&endpoint)
            .ok_or_else(|| format!("invalid endpoint URL '{}'", endpoint))?;
        Ok(CosContext {
            endpoint,
            bucket,
            region,
            credentials,
            host,
        })
    }

    /// Build a signed GET request for an object key in the configured bucket.
    pub fn signed_get(&self, key: &str) -> (String, HeaderMap) {
        let path = format!("/{}/{}", self.bucket, key);
        let url = self.url_for_path(&path, "");
        let headers = self.sign_now("GET", &path, "", SHA256_EMPTY);
        (url, headers)
    }

    /// Build a signed HEAD request for an object key in the configured bucket.
    /// Only used as a fallback when the listing did not provide a size.
    #[allow(dead_code)]
    pub fn signed_head(&self, key: &str) -> (String, HeaderMap) {
        let path = format!("/{}/{}", self.bucket, key);
        let url = self.url_for_path(&path, "");
        let headers = self.sign_now("HEAD", &path, "", SHA256_EMPTY);
        (url, headers)
    }

    /// Enumerate every object whose key begins with `prefix`, following all
    /// pagination continuation tokens. Returns `(key, size)` pairs.
    pub fn list_prefix(
        &self,
        client: &Client,
        prefix: &str,
    ) -> Result<Vec<(String, u64)>, String> {
        let mut all = Vec::new();
        let mut continuation: Option<String> = None;
        loop {
            let (url, headers) = self.signed_list(prefix, continuation.as_deref());
            let resp = client
                .get(&url)
                .headers(headers)
                .send()
                .map_err(|e| format!("ListObjectsV2 request failed: {}", e))?;
            let status = resp.status();
            let body = resp
                .text()
                .map_err(|e| format!("ListObjectsV2 body read failed: {}", e))?;
            if !status.is_success() {
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

    /// Build a signed ListObjectsV2 request: GET /{bucket}?list-type=2&prefix=...
    fn signed_list(
        &self,
        prefix: &str,
        continuation: Option<&str>,
    ) -> (String, HeaderMap) {
        let path = format!("/{}", self.bucket);

        // Canonical query string: parameters sorted by name, each k/v
        // URI-encoded with the strict (unreserved-only) rule.
        let mut params: Vec<(&str, String)> = vec![
            ("list-type", "2".to_string()),
            ("prefix", prefix.to_string()),
        ];
        if let Some(token) = continuation {
            params.push(("continuation-token", token.to_string()));
        }
        params.sort_by(|a, b| a.0.cmp(b.0));
        let canonical_query = params
            .iter()
            .map(|(k, v)| format!("{}={}", encode_strict(k), encode_strict(v)))
            .collect::<Vec<_>>()
            .join("&");

        let url = self.url_for_path(&path, &canonical_query);
        let headers = self.sign_now("GET", &path, &canonical_query, SHA256_EMPTY);
        (url, headers)
    }

    fn url_for_path(&self, path: &str, query: &str) -> String {
        let base = self.endpoint.trim_end_matches('/');
        if query.is_empty() {
            format!("{}{}", base, encode_path(path))
        } else {
            format!("{}{}?{}", base, encode_path(path), query)
        }
    }

    /// Sign a request with the current wall-clock time and return the headers
    /// reqwest needs to attach.
    fn sign_now(
        &self,
        method: &str,
        path: &str,
        canonical_query: &str,
        payload_hash: &str,
    ) -> HeaderMap {
        let now = Utc::now();
        sign(
            method,
            path,
            canonical_query,
            payload_hash,
            &self.host,
            &self.region,
            &self.credentials,
            now,
        )
    }
}

/// Pure SigV4 signing function. Separated from `CosContext` so tests can
/// exercise the math against published AWS test vectors with a fixed clock.
pub fn sign(
    method: &str,
    path: &str,
    canonical_query: &str,
    payload_hash: &str,
    host: &str,
    region: &str,
    credentials: &CosCredentials,
    now: DateTime<Utc>,
) -> HeaderMap {
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date_stamp = &amz_date[..8];

    // Headers to include in the signature. They MUST be the same set we
    // attach to the outgoing request, and `host` MUST match what reqwest
    // derives from the URL (we use path-style URLs so the host is the bare
    // endpoint host).
    let mut signed_headers: Vec<(&str, String)> = vec![
        ("host", host.to_string()),
        ("x-amz-content-sha256", payload_hash.to_string()),
        ("x-amz-date", amz_date.clone()),
    ];
    if let Some(token) = &credentials.session_token {
        signed_headers.push(("x-amz-security-token", token.clone()));
    }
    signed_headers.sort_by(|a, b| a.0.cmp(b.0));

    let canonical_headers: String = signed_headers
        .iter()
        .map(|(k, v)| format!("{}:{}\n", k, v.trim()))
        .collect();
    let signed_header_names = signed_headers
        .iter()
        .map(|(k, _)| *k)
        .collect::<Vec<_>>()
        .join(";");

    let canonical_uri = encode_path(path);
    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method,
        canonical_uri,
        canonical_query,
        canonical_headers,
        signed_header_names,
        payload_hash,
    );

    let scope = format!("{}/{}/s3/aws4_request", date_stamp, region);
    let cr_hash = hex::encode(Sha256::digest(canonical_request.as_bytes()));
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        amz_date, scope, cr_hash,
    );

    let signing_key = derive_signing_key(&credentials.secret_key, date_stamp, region, "s3");
    let signature = hex::encode(hmac_sha256(&signing_key, string_to_sign.as_bytes()));
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        credentials.access_key, scope, signed_header_names, signature,
    );

    let mut hm = HeaderMap::new();
    for (k, v) in &signed_headers {
        hm.insert(
            HeaderName::from_static(k),
            HeaderValue::from_str(v).expect("signed header value must be ASCII"),
        );
    }
    hm.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&authorization).expect("authorization is ASCII"),
    );
    hm
}

/// Compute the canonical request string used as input to the SigV4 signature.
/// Exposed for testing against AWS published vectors.
#[cfg(test)]
fn canonical_request(
    method: &str,
    path: &str,
    canonical_query: &str,
    payload_hash: &str,
    host: &str,
    amz_date: &str,
) -> String {
    let signed_headers: Vec<(&str, String)> = vec![
        ("host", host.to_string()),
        ("x-amz-date", amz_date.to_string()),
    ];
    let canonical_headers: String = signed_headers
        .iter()
        .map(|(k, v)| format!("{}:{}\n", k, v.trim()))
        .collect();
    let signed_header_names = signed_headers
        .iter()
        .map(|(k, _)| *k)
        .collect::<Vec<_>>()
        .join(";");
    let canonical_uri = encode_path(path);
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method,
        canonical_uri,
        canonical_query,
        canonical_headers,
        signed_header_names,
        payload_hash,
    )
}

// --- byte-level helpers ---

type HmacSha256 = Hmac<Sha256>;

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC key length is unbounded");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn derive_signing_key(secret: &str, date: &str, region: &str, service: &str) -> Vec<u8> {
    let kdate = hmac_sha256(format!("AWS4{}", secret).as_bytes(), date.as_bytes());
    let kregion = hmac_sha256(&kdate, region.as_bytes());
    let kservice = hmac_sha256(&kregion, service.as_bytes());
    hmac_sha256(&kservice, b"aws4_request")
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

    /// AWS published "get-vanilla" SigV4 test vector.
    /// Source: AWS SigV4 test suite (sigv4_testsuite/get-vanilla).
    #[test]
    fn sigv4_get_vanilla_signature() {
        let creds = CosCredentials {
            access_key: "AKIDEXAMPLE".to_string(),
            secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: None,
        };
        let now = DateTime::parse_from_rfc3339("2015-08-30T12:36:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let headers = sign(
            "GET",
            "/",
            "",
            SHA256_EMPTY,
            "example.amazonaws.com",
            "us-east-1",
            &creds,
            now,
        );

        // The published test vector uses service=service (not s3), so this
        // assertion uses our own end-to-end calculation. To validate against
        // AWS's vector directly, exercise canonical_request() and the signing
        // primitives below.
        let auth = headers
            .get(AUTHORIZATION)
            .expect("authorization header set")
            .to_str()
            .unwrap();
        assert!(
            auth.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/s3/aws4_request"),
            "got auth: {}", auth
        );
        assert!(auth.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date"));
    }

    /// Validate the canonical request and signing primitives against the AWS
    /// published "get-vanilla" vector. This pins the math below the public
    /// `sign()` API where the s3 service name is hardcoded.
    #[test]
    fn sigv4_canonical_request_and_signature_match_aws_vector() {
        let cr = canonical_request(
            "GET",
            "/",
            "",
            SHA256_EMPTY,
            "example.amazonaws.com",
            "20150830T123600Z",
        );
        let expected_cr = "GET\n/\n\nhost:example.amazonaws.com\nx-amz-date:20150830T123600Z\n\nhost;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(cr, expected_cr);

        let cr_hash = hex::encode(Sha256::digest(cr.as_bytes()));
        let scope = "20150830/us-east-1/service/aws4_request";
        let sts = format!("AWS4-HMAC-SHA256\n20150830T123600Z\n{}\n{}", scope, cr_hash);

        let secret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
        let key = derive_signing_key(secret, "20150830", "us-east-1", "service");
        let sig = hex::encode(hmac_sha256(&key, sts.as_bytes()));
        assert_eq!(sig, "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31");
    }

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

    #[test]
    fn signed_get_url_uses_path_style() {
        let ctx = CosContext::new(
            "https://s3.us-south.cloud-object-storage.appdomain.cloud".to_string(),
            "gtkbackup".to_string(),
            "us-south".to_string(),
            CosCredentials {
                access_key: "AKID".to_string(),
                secret_key: "SECRET".to_string(),
                session_token: None,
            },
        ).unwrap();
        let (url, headers) = ctx.signed_get("bulk-v0.8-1B/file1.bin");
        assert_eq!(
            url,
            "https://s3.us-south.cloud-object-storage.appdomain.cloud/gtkbackup/bulk-v0.8-1B/file1.bin"
        );
        assert!(headers.contains_key(AUTHORIZATION));
        assert!(headers.contains_key("x-amz-date"));
        assert!(headers.contains_key("x-amz-content-sha256"));
    }

    #[test]
    fn signed_list_canonical_query_is_sorted() {
        let ctx = CosContext::new(
            "https://example.com".to_string(),
            "b".to_string(),
            "us-south".to_string(),
            CosCredentials {
                access_key: "AKID".to_string(),
                secret_key: "SECRET".to_string(),
                session_token: None,
            },
        ).unwrap();
        let (url, _headers) = ctx.signed_list("bulk-v0.8-1B/", Some("tok=en"));
        // Sorted by key: continuation-token, list-type, prefix
        // Each value strict-encoded: "tok=en" -> "tok%3Den", "bulk-v0.8-1B/" -> "bulk-v0.8-1B%2F"
        assert!(
            url.ends_with("?continuation-token=tok%3Den&list-type=2&prefix=bulk-v0.8-1B%2F"),
            "got url: {}", url
        );
    }
}
