// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! AWS Signature Version 4 as a [`RequestSigner`]: [`SigV4Signer`]
//! signs a [`Request`] with an [`AwsCredentials`] key for one region and
//! service (`srd-crate-boundaries.md` TR-5, §10 Q6).
//!
//! The transport hands a signer the request in its wire form just before
//! sending it, so a signature is always fresh — SigV4 signatures expire
//! fifteen minutes after their `x-amz-date` — and covers exactly the host,
//! path and query that are sent.
//!
//! ```
//! use std::sync::Arc;
//! use std::time::{Duration, SystemTime};
//! use veks_transport::sigv4::{AwsCredentials, SigV4Signer};
//! use veks_transport::Request;
//!
//! let signer = SigV4Signer::s3(AwsCredentials::new("AKID", "SECRET"), "us-south");
//! let mut req = Request::get("https://cos.example/bucket/key.bin");
//! let at = SystemTime::UNIX_EPOCH + Duration::from_secs(1_440_938_160); // 2015-08-30T12:36:00Z
//! signer.sign_at(&mut req, at).unwrap();
//! assert_eq!(req.header_value("x-amz-date"), Some("20150830T123600Z"));
//! assert!(req.header_value("authorization").unwrap().starts_with(
//!     "AWS4-HMAC-SHA256 Credential=AKID/20150830/us-south/s3/aws4_request, \
//!      SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature="
//! ));
//!
//! // Handed to the transport, the request is signed as it is sent.
//! let req = Request::get("https://cos.example/bucket/key.bin").signed(Arc::new(signer));
//! # let _ = req;
//! ```

use std::fmt;
use std::time::SystemTime;

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use veks_apicore::transport::{Body, Request, RequestSigner};
use veks_apicore::{TransportError, TransportErrorKind};

/// The SHA-256 of no bytes: the payload hash of a request with no body.
const SHA256_EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// The payload hash S3 accepts for a body that is streamed unhashed.
const UNSIGNED_PAYLOAD: &str = "UNSIGNED-PAYLOAD";

/// The headers the signer itself adds, removed from a request before it
/// is signed again.
const SIGNER_HEADERS: &[&str] = &["authorization", "x-amz-date", "x-amz-content-sha256", "x-amz-security-token"];

/// An AWS access key: what SigV4 signs with. Whoever builds it decides
/// where it comes from (the AWS environment variables, a profile, a
/// vault); the transport reads none of them (TR-3).
#[derive(Clone, PartialEq, Eq)]
pub struct AwsCredentials {
    /// The access key id (`AWS_ACCESS_KEY_ID`), named in every signature.
    pub access_key: String,
    /// The secret access key (`AWS_SECRET_ACCESS_KEY`), never sent.
    pub secret_key: String,
    /// A temporary credential's session token (`AWS_SESSION_TOKEN`), sent
    /// and signed as `x-amz-security-token`.
    pub session_token: Option<String>,
}

impl AwsCredentials {
    /// A long-term key: an access key id and its secret, no session.
    pub fn new(access_key: impl Into<String>, secret_key: impl Into<String>) -> Self {
        AwsCredentials { access_key: access_key.into(), secret_key: secret_key.into(), session_token: None }
    }

    /// This key with a temporary credential's session token.
    pub fn with_session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(token.into());
        self
    }
}

impl fmt::Debug for AwsCredentials {
    /// The access key id only: the secret and the session token are
    /// never printed.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AwsCredentials")
            .field("access_key", &self.access_key)
            .field("secret_key", &"..")
            .field("session_token", &self.session_token.as_ref().map(|_| ".."))
            .finish()
    }
}

/// Signs requests with AWS Signature Version 4 (header form) for one
/// region and service.
///
/// What is signed:
/// - the method, the path and the query of the URL as it is sent — an
///   S3 path as given, any other service's path segments encoded once
///   more, as SigV4 specifies; the query's parameters decoded, encoded
///   strictly (RFC 3986 unreserved characters only) and sorted;
/// - the `host` (with a port that is not the scheme's default), the
///   request's `content-type` and `x-amz-*` headers, and the headers the
///   signer adds: `x-amz-date`, `x-amz-content-sha256` for S3, and
///   `x-amz-security-token` for a temporary credential;
/// - the payload's SHA-256: of no bytes for an empty body, of the bytes
///   for a byte body, and `UNSIGNED-PAYLOAD` for a file streamed off disk
///   (accepted by S3, so a large upload is not read twice).
#[derive(Clone)]
pub struct SigV4Signer {
    credentials: AwsCredentials,
    region: String,
    service: String,
}

impl fmt::Debug for SigV4Signer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SigV4Signer")
            .field("credentials", &self.credentials)
            .field("region", &self.region)
            .field("service", &self.service)
            .finish()
    }
}

impl SigV4Signer {
    /// A signer for `service` (`s3`, `iam`, …) in `region`.
    pub fn new(credentials: AwsCredentials, region: impl Into<String>, service: impl Into<String>) -> Self {
        SigV4Signer { credentials, region: region.into(), service: service.into() }
    }

    /// A signer for S3 and S3-compatible stores (IBM COS, MinIO) in
    /// `region`.
    pub fn s3(credentials: AwsCredentials, region: impl Into<String>) -> Self {
        Self::new(credentials, region, "s3")
    }

    /// The region signatures are scoped to.
    pub fn region(&self) -> &str {
        &self.region
    }

    /// The service signatures are scoped to.
    pub fn service(&self) -> &str {
        &self.service
    }

    /// Sign `req` as of `at`: add `x-amz-date`, the payload hash and
    /// session token where they apply, and `Authorization`. A signature
    /// the request already carries is replaced. [`RequestSigner::sign`]
    /// is this at the current time.
    pub fn sign_at(&self, req: &mut Request, at: SystemTime) -> Result<(), TransportError> {
        let canonical = self.canonical(req, at)?;
        let refuse = |message: String| refusal(req, message);
        let key = self.signing_key(&canonical.amz_date[..8]).map_err(refuse)?;
        let signature = hex(&hmac(&key, string_to_sign(&canonical).as_bytes()).map_err(refuse)?);
        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={signature}",
            self.credentials.access_key, canonical.scope, canonical.signed_headers
        );
        req.headers.retain(|(name, _)| !SIGNER_HEADERS.iter().any(|h| name.eq_ignore_ascii_case(h)));
        req.headers.extend(canonical.added);
        req.headers.push(("authorization".to_string(), authorization));
        Ok(())
    }

    /// The canonical form of `req` at `at`, and what signing it adds.
    fn canonical(&self, req: &Request, at: SystemTime) -> Result<Canonical, TransportError> {
        let refuse = |message: String| refusal(req, message);
        let url = url::Url::parse(&req.url).map_err(|e| refuse(format!("cannot sign '{}': {e}", req.url)))?;
        let host = match (url.host_str(), url.port()) {
            (Some(h), Some(p)) => format!("{h}:{p}"),
            (Some(h), None) => h.to_string(),
            (None, _) => return Err(refuse(format!("cannot sign '{}': it names no host", req.url))),
        };
        let amz_date = amz_date(at);
        let date = &amz_date[..8];
        let payload_hash = match &req.body {
            Body::Empty => SHA256_EMPTY.to_string(),
            Body::Bytes(bytes) => hex(&Sha256::digest(bytes)),
            Body::File(_) => UNSIGNED_PAYLOAD.to_string(),
        };

        // The headers the signer adds, then every header signed.
        let mut added = vec![("x-amz-date".to_string(), amz_date.clone())];
        if self.service == "s3" {
            added.push(("x-amz-content-sha256".to_string(), payload_hash.clone()));
        }
        if let Some(token) = &self.credentials.session_token {
            added.push(("x-amz-security-token".to_string(), token.clone()));
        }
        let mut signed: Vec<(String, String)> = vec![("host".to_string(), host)];
        for (name, value) in &req.headers {
            let name = name.to_ascii_lowercase();
            if (name == "content-type" || name.starts_with("x-amz-")) && !SIGNER_HEADERS.contains(&name.as_str()) {
                signed.push((name, value.clone()));
            }
        }
        signed.extend(added.iter().cloned());
        signed.sort_by(|a, b| a.0.cmp(&b.0));
        let canonical_headers: String =
            signed.iter().map(|(k, v)| format!("{k}:{}\n", v.split_whitespace().collect::<Vec<_>>().join(" "))).collect();
        let signed_headers = signed.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(";");

        let path = if url.path().is_empty() { "/" } else { url.path() };
        let canonical_uri =
            if self.service == "s3" { path.to_string() } else { path.split('/').map(encode_strict).collect::<Vec<_>>().join("/") };
        let canonical_request = format!(
            "{}\n{canonical_uri}\n{}\n{canonical_headers}\n{signed_headers}\n{payload_hash}",
            req.method.as_str(),
            canonical_query(url.query().unwrap_or("")),
        );

        let scope = format!("{date}/{}/{}/aws4_request", self.region, self.service);
        Ok(Canonical { request: canonical_request, amz_date, scope, signed_headers, added })
    }

    /// The day's signing key: the secret through HMAC-SHA256 over the
    /// date, region, service and `aws4_request`.
    fn signing_key(&self, date: &str) -> Result<Vec<u8>, String> {
        let k_date = hmac(format!("AWS4{}", self.credentials.secret_key).as_bytes(), date.as_bytes())?;
        let k_region = hmac(&k_date, self.region.as_bytes())?;
        let k_service = hmac(&k_region, self.service.as_bytes())?;
        hmac(&k_service, b"aws4_request")
    }
}

impl RequestSigner for SigV4Signer {
    /// Sign `req` at the current time ([`SigV4Signer::sign_at`]).
    fn sign(&self, req: &mut Request) -> Result<(), TransportError> {
        self.sign_at(req, SystemTime::now())
    }
}

/// A request in SigV4's canonical form, and what signing it adds.
struct Canonical {
    /// The canonical request.
    request: String,
    /// `x-amz-date`.
    amz_date: String,
    /// The credential scope: date, region, service, `aws4_request`.
    scope: String,
    /// The signed header names, `;`-separated.
    signed_headers: String,
    /// The headers the signer adds to the request, beside `Authorization`.
    added: Vec<(String, String)>,
}

/// SigV4's string to sign for a canonical request.
fn string_to_sign(c: &Canonical) -> String {
    format!("AWS4-HMAC-SHA256\n{}\n{}\n{}", c.amz_date, c.scope, hex(&Sha256::digest(c.request.as_bytes())))
}

/// A permanent error: `req` cannot be signed, so it is not sent.
fn refusal(req: &Request, message: String) -> TransportError {
    TransportError { url: Some(req.url.clone()), status: None, kind: TransportErrorKind::Permanent, message }
}

/// HMAC-SHA256 of `data` under `key`.
fn hmac(key: &[u8], data: &[u8]) -> Result<Vec<u8>, String> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).map_err(|e| format!("HMAC key: {e}"))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

/// Lower-case hexadecimal.
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(DIGITS[usize::from(b >> 4)] as char);
        out.push(DIGITS[usize::from(b & 0xf)] as char);
    }
    out
}

/// `at` as SigV4's `YYYYMMDD'T'HHMMSS'Z'`, in UTC.
fn amz_date(at: SystemTime) -> String {
    let secs = at.duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Days since 1970-01-01 to a civil date (Howard Hinnant's
    // `civil_from_days`, for days on or after the epoch).
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    format!("{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// The canonical query string: each parameter's name and value decoded
/// and encoded again strictly, sorted by name and then value.
fn canonical_query(query: &str) -> String {
    let mut params: Vec<(String, String)> = query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (encode_strict(&percent_decode(k)), encode_strict(&percent_decode(v)))
        })
        .collect();
    params.sort();
    params.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&")
}

/// `s` with its `%XX` escapes decoded (a `+` stays a `+`: SigV4 queries
/// are not form-encoded).
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex_at = |j: usize| bytes.get(j).and_then(|b| (*b as char).to_digit(16));
        if bytes[i] == b'%'
            && let (Some(hi), Some(lo)) = (hex_at(i + 1), hex_at(i + 2))
        {
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// RFC 3986 strict percent-encoding: only `A-Z a-z 0-9 - _ . ~` pass
/// unchanged, as SigV4 requires of the canonical URI and query.
fn encode_strict(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// 2015-08-30T12:36:00Z, the time of AWS's published examples.
    fn example_time() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_440_938_160)
    }

    fn example_key() -> AwsCredentials {
        AwsCredentials::new("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
    }

    /// **AWS's documented SigV4 example** (the IAM `ListUsers` request of
    /// the "Signature Version 4 signing process" guide): the canonical
    /// request, its hash, the string to sign, the signing key and the
    /// signature are each the published value.
    #[test]
    fn the_aws_documentation_example_signs_to_the_published_values() {
        let signer = SigV4Signer::new(example_key(), "us-east-1", "iam");
        let req = Request::get("https://iam.amazonaws.com/?Action=ListUsers&Version=2010-05-08")
            .header("Content-Type", "application/x-www-form-urlencoded; charset=utf-8");
        let c = signer.canonical(&req, example_time()).unwrap();
        assert_eq!(
            c.request,
            "GET\n/\nAction=ListUsers&Version=2010-05-08\n\
             content-type:application/x-www-form-urlencoded; charset=utf-8\n\
             host:iam.amazonaws.com\nx-amz-date:20150830T123600Z\n\n\
             content-type;host;x-amz-date\n\
             e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&Sha256::digest(c.request.as_bytes())),
            "f536975d06c0309214f805bb90ccff089219ecd68b2577efef23edd43b7e1a59"
        );
        assert_eq!(
            string_to_sign(&c),
            "AWS4-HMAC-SHA256\n20150830T123600Z\n20150830/us-east-1/iam/aws4_request\n\
             f536975d06c0309214f805bb90ccff089219ecd68b2577efef23edd43b7e1a59"
        );
        assert_eq!(
            hex(&signer.signing_key("20150830").unwrap()),
            "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9"
        );
        let mut req = req;
        signer.sign_at(&mut req, example_time()).unwrap();
        assert_eq!(
            req.header_value("authorization").unwrap(),
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request, \
             SignedHeaders=content-type;host;x-amz-date, \
             Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7"
        );
    }

    /// AWS's SigV4 test suite, `get-vanilla`: a bare `GET /` to service
    /// `service`.
    #[test]
    fn the_aws_test_suite_get_vanilla_signs_to_its_published_signature() {
        let signer = SigV4Signer::new(example_key(), "us-east-1", "service");
        let mut req = Request::get("https://example.amazonaws.com/");
        signer.sign_at(&mut req, example_time()).unwrap();
        assert_eq!(
            req.header_value("authorization"),
            Some(
                "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
                 SignedHeaders=host;x-amz-date, \
                 Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
            )
        );
        assert_eq!(req.header_value("x-amz-date"), Some("20150830T123600Z"));
        assert_eq!(req.header_value("x-amz-content-sha256"), None, "only S3 sends the payload hash");
    }

    /// **The COS listing and object GETs sign as before**: the signatures
    /// veks-pipeline's own SigV4 code produced for these requests (captured
    /// from it before it moved here), with and without a session token.
    #[test]
    fn s3_requests_sign_as_the_pipelines_cos_code_did() {
        let cases = [
            (
                "https://s3.us-south.cloud-object-storage.appdomain.cloud/b?list-type=2&prefix=bulk-v0.8-1B%2F&continuation-token=tok%3Den",
                None,
                "f32b24cb720818006473948ae212b41780fd7338bfa30af2944a65a0c80e8eb5",
            ),
            (
                "http://127.0.0.1:9000/b/bulk-v0.8-1B/sub/file%202.bin",
                None,
                "de6900dc9b39778a268beef5adb380e57d18ee2d7297452318550b6229c934e1",
            ),
            (
                "https://s3.us-south.cloud-object-storage.appdomain.cloud/b?continuation-token=tok%3Den&list-type=2&prefix=bulk-v0.8-1B%2F",
                Some("SESSIONTOKEN"),
                "f7d205c596edde541db0601ed78a7d029b1baf164b5efba4659312dfb37a809a",
            ),
            (
                "http://127.0.0.1:9000/b/bulk-v0.8-1B/sub/file%202.bin",
                Some("SESSIONTOKEN"),
                "6bb1345d59cb9370b3d8a884a83e9835a275a655651155aa3a0741006ac2971a",
            ),
        ];
        for (url, session, signature) in cases {
            let mut key = AwsCredentials::new("AKID", "SECRET");
            if let Some(t) = session {
                key = key.with_session_token(t);
            }
            let mut req = Request::get(url);
            SigV4Signer::s3(key, "us-south").sign_at(&mut req, example_time()).unwrap();
            let names = if session.is_some() {
                "host;x-amz-content-sha256;x-amz-date;x-amz-security-token"
            } else {
                "host;x-amz-content-sha256;x-amz-date"
            };
            assert_eq!(
                req.header_value("authorization").unwrap(),
                format!(
                    "AWS4-HMAC-SHA256 Credential=AKID/20150830/us-south/s3/aws4_request, \
                     SignedHeaders={names}, Signature={signature}"
                ),
                "{url}"
            );
            assert_eq!(req.header_value("x-amz-content-sha256"), Some(SHA256_EMPTY));
            assert_eq!(req.header_value("x-amz-security-token"), session);
        }
    }

    /// Signing again replaces the signature rather than adding a second.
    #[test]
    fn signing_twice_leaves_one_signature() {
        let signer = SigV4Signer::s3(AwsCredentials::new("AKID", "SECRET"), "us-south");
        let mut req = Request::get("https://h/b/k");
        signer.sign_at(&mut req, example_time()).unwrap();
        signer.sign_at(&mut req, example_time() + Duration::from_secs(60)).unwrap();
        for name in SIGNER_HEADERS {
            assert!(req.headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case(name)).count() <= 1, "{name}");
        }
        assert_eq!(req.header_value("x-amz-date"), Some("20150830T123700Z"));
    }

    /// A byte body is hashed; a streamed file is sent unhashed.
    #[test]
    fn the_payload_hash_follows_the_body() {
        let signer = SigV4Signer::s3(AwsCredentials::new("AKID", "SECRET"), "us-south");
        let mut bytes = Request::put("https://h/b/k").bytes(b"abc".to_vec());
        signer.sign_at(&mut bytes, example_time()).unwrap();
        assert_eq!(
            bytes.header_value("x-amz-content-sha256"),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        let mut file = Request::put("https://h/b/k").file("/nonexistent");
        signer.sign_at(&mut file, example_time()).unwrap();
        assert_eq!(file.header_value("x-amz-content-sha256"), Some(UNSIGNED_PAYLOAD));
    }

    /// The secret and the session token never reach a debug print.
    #[test]
    fn a_debug_print_names_no_secret() {
        let key = AwsCredentials::new("AKID", "SECRET").with_session_token("TOKEN");
        let shown = format!("{:?}", SigV4Signer::s3(key, "r"));
        assert!(shown.contains("AKID"), "{shown}");
        assert!(!shown.contains("SECRET") && !shown.contains("TOKEN"), "{shown}");
    }

    #[test]
    fn dates_are_utc_civil_dates() {
        assert_eq!(amz_date(SystemTime::UNIX_EPOCH), "19700101T000000Z");
        assert_eq!(amz_date(example_time()), "20150830T123600Z");
        // A leap day, and the last second of a century leap year.
        assert_eq!(amz_date(SystemTime::UNIX_EPOCH + Duration::from_secs(951_782_400)), "20000229T000000Z");
        assert_eq!(amz_date(SystemTime::UNIX_EPOCH + Duration::from_secs(978_307_199)), "20001231T235959Z");
    }

    #[test]
    fn the_canonical_query_is_decoded_encoded_strictly_and_sorted() {
        assert_eq!(canonical_query(""), "");
        assert_eq!(canonical_query("b=2&a=1"), "a=1&b=2");
        assert_eq!(canonical_query("prefix=a/b&x=a+b&y=%7e"), "prefix=a%2Fb&x=a%2Bb&y=~");
        assert_eq!(canonical_query("flag"), "flag=");
    }
}
