// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The transport's seams: how it is configured ([`TransportConfig`],
//! [`TlsTrust`]), where its read credentials come from
//! ([`Credentials`]), how a request signs itself ([`RequestSigner`]),
//! and the request a caller hands it ([`Request`]).
//!
//! `veks-transport` implements the network side and consumes these;
//! `vectordata` builds the configuration from its settings and the
//! environment and implements [`Credentials`] over its credential
//! store; a program outside the workspace can do either with this crate
//! alone (`docs/design/srd-crate-boundaries.md` TR-3..TR-6). Nothing
//! here opens a connection or reads a file: configuration crosses the
//! seam as a value, so a transport is testable with no process
//! environment.
//!
//! A [`Credentials`] written with nothing but this crate:
//!
//! ```
//! use veks_apicore::transport::{Credentials, NoCredentials};
//!
//! /// One token for one origin; anonymous everywhere else.
//! struct OneOrigin {
//!     origin: String,
//!     token: String,
//! }
//!
//! impl Credentials for OneOrigin {
//!     fn read_token(&self, url: &str) -> Option<String> {
//!         url.starts_with(&self.origin).then(|| self.token.clone())
//!     }
//! }
//!
//! let creds = OneOrigin { origin: "https://data.example".into(), token: "t0k".into() };
//! assert_eq!(creds.read_token("https://data.example/ds/base.fvecs").as_deref(), Some("t0k"));
//! assert_eq!(creds.read_token("https://elsewhere.example/x"), None);
//! assert_eq!(NoCredentials.read_token("https://data.example/x"), None);
//! ```

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::error::TransportError;

/// The client TLS-trust policy.
///
/// Verification is on for every origin. `trusted_ca_certs` adds roots
/// to it (the secure way to trust a private deployment's certificate);
/// `trust_self_signed` names the origins whose certificate is accepted
/// without verification, and only those.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TlsTrust {
    /// PEM files added to the client's trusted roots (verification stays on).
    pub trusted_ca_certs: Vec<PathBuf>,
    /// Origins (`scheme://host[:port]`) whose server cert is accepted without
    /// verification — insecure; empty by default.
    pub trust_self_signed: Vec<String>,
}

/// How long a request may take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
    /// The ceiling on a whole request, body included.
    pub request: Duration,
    /// How long connecting may take before the server counts as
    /// unreachable.
    pub connect: Duration,
}

impl Default for Timeouts {
    /// One hour for a request (a large fetch), ten seconds to connect:
    /// an unreachable server is reported in seconds, not after the
    /// operating system's TCP timeout.
    fn default() -> Self {
        Timeouts { request: Duration::from_secs(60 * 60), connect: Duration::from_secs(10) }
    }
}

/// Everything a transport is configured by, as one value
/// (`srd-crate-boundaries.md` TR-3). The transport reads no environment
/// variable and no settings file: whoever builds it decides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportConfig {
    /// Offline mode: no request is made to a dataset server, and asking
    /// for one is an error saying so. Requests marked
    /// [`Request::explicit`] are exempt.
    pub offline: bool,
    /// How the user turns offline mode off, named in its refusal
    /// (`VECTORDATA_OFFLINE / settings \`offline\``); `None` names
    /// nothing.
    pub offline_hint: Option<String>,
    /// Which certificates are trusted, and for which origins.
    pub tls: TlsTrust,
    /// The `User-Agent` every request sends.
    pub user_agent: String,
    /// Request and connect timeouts.
    pub timeouts: Timeouts,
    /// How many redirects a request follows; `0` follows none.
    pub redirects: usize,
    /// Independent HTTP clients, each with its own runtime thread, that
    /// requests are spread over, so many parallel streams decrypt on
    /// many cores.
    pub parallel_streams: usize,
    /// Workers a chunked download runs at once.
    pub download_concurrency: usize,
    /// The region an `s3://bucket/key` URL is first addressed in; a
    /// wrong one costs one redirect.
    pub s3_region: String,
}

/// The default number of parallel streams: one client runtime and one
/// download worker per stream.
pub const DEFAULT_PARALLEL_STREAMS: usize = 32;

impl Default for TransportConfig {
    /// Online, the system roots, ten redirects, the default timeouts,
    /// [`DEFAULT_PARALLEL_STREAMS`] streams and workers, `us-east-1`.
    fn default() -> Self {
        TransportConfig {
            offline: false,
            offline_hint: None,
            tls: TlsTrust::default(),
            user_agent: concat!("veks-transport/", env!("CARGO_PKG_VERSION")).to_string(),
            timeouts: Timeouts::default(),
            redirects: 10,
            parallel_streams: DEFAULT_PARALLEL_STREAMS,
            download_concurrency: DEFAULT_PARALLEL_STREAMS,
            s3_region: "us-east-1".to_string(),
        }
    }
}

/// Where a transport's read-side bearer tokens come from
/// (`srd-crate-boundaries.md` TR-4).
///
/// Asked for a request that carries no authorisation of its own
/// ([`Authorization::ReadCredentials`], the default); `None` sends the
/// request anonymously. The URL is the one the request names, before
/// any rewriting for the wire (an `s3://` URL is asked as `s3://`).
pub trait Credentials: Send + Sync {
    /// The bearer token for a read of `url`, if there is one.
    fn read_token(&self, url: &str) -> Option<String>;
}

/// No credentials: every read is anonymous.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoCredentials;

impl Credentials for NoCredentials {
    fn read_token(&self, _url: &str) -> Option<String> {
        None
    }
}

/// Signs a request just before it is sent: adds the headers a signature
/// scheme needs (SigV4's `Authorization`, `x-amz-date`, …)
/// (`srd-crate-boundaries.md` TR-5).
///
/// The signer is handed the request with its URL in the form the wire
/// takes (an `s3://` URL already rewritten to its HTTPS endpoint), and
/// what it leaves is what is sent.
///
/// ```
/// use veks_apicore::TransportError;
/// use veks_apicore::transport::{Request, RequestSigner};
///
/// /// A shared-secret header: enough to show the shape.
/// struct ApiKey(String);
///
/// impl RequestSigner for ApiKey {
///     fn sign(&self, req: &mut Request) -> Result<(), TransportError> {
///         req.headers.push(("x-api-key".into(), self.0.clone()));
///         Ok(())
///     }
/// }
///
/// let mut req = Request::get("https://api.example/v1/papers");
/// ApiKey("k".into()).sign(&mut req).unwrap();
/// assert_eq!(req.header_value("x-api-key"), Some("k"));
/// ```
pub trait RequestSigner: Send + Sync {
    /// Add the signature to `req`.
    fn sign(&self, req: &mut Request) -> Result<(), TransportError>;
}

/// The HTTP method of a [`Request`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// `GET`.
    Get,
    /// `HEAD`.
    Head,
    /// `PUT`.
    Put,
    /// `POST`.
    Post,
    /// `PATCH`.
    Patch,
    /// `DELETE`.
    Delete,
}

impl Method {
    /// The method's name on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Head => "HEAD",
            Method::Put => "PUT",
            Method::Post => "POST",
            Method::Patch => "PATCH",
            Method::Delete => "DELETE",
        }
    }
}

/// What a request sends.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Body {
    /// Nothing.
    #[default]
    Empty,
    /// These bytes.
    Bytes(Vec<u8>),
    /// The file at this path, streamed off disk.
    File(PathBuf),
}

/// Who a request says it is.
#[derive(Clone, Default)]
pub enum Authorization {
    /// The transport's read credentials ([`Credentials`]), when they
    /// have a token for the URL; anonymous otherwise. The default.
    #[default]
    ReadCredentials,
    /// Nobody: the read credentials are not asked.
    Anonymous,
    /// This bearer token, and not the read credentials.
    Bearer(String),
    /// Signed by this signer just before sending, and not the read
    /// credentials.
    Signed(Arc<dyn RequestSigner>),
}

impl fmt::Debug for Authorization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Authorization::ReadCredentials => f.write_str("ReadCredentials"),
            Authorization::Anonymous => f.write_str("Anonymous"),
            // A token is never printed.
            Authorization::Bearer(_) => f.write_str("Bearer(..)"),
            Authorization::Signed(_) => f.write_str("Signed(..)"),
        }
    }
}

/// A request for a transport to send: the transport's own type, so no
/// HTTP library's type crosses the seam (`srd-crate-boundaries.md`
/// TR-6).
///
/// Built with [`get`](Self::get), [`head`](Self::head), … and the
/// builder methods. The URL may be an `s3://` one, which the transport
/// sends to its HTTPS endpoint.
///
/// ```
/// use veks_apicore::transport::{Authorization, Method, Request};
///
/// let req = Request::put("https://store.example/ds/SHA256SUMS")
///     .bytes(b"abc  base.fvecs\n".to_vec())
///     .header("If-None-Match", "*")
///     .bearer("push-token");
/// assert_eq!(req.method, Method::Put);
/// assert_eq!(req.header_value("if-none-match"), Some("*"));
/// assert!(matches!(req.auth, Authorization::Bearer(_)));
/// ```
#[derive(Debug, Clone)]
pub struct Request {
    /// The method.
    pub method: Method,
    /// The URL, as the caller names it.
    pub url: String,
    /// Headers beyond the transport's own, in order.
    pub headers: Vec<(String, String)>,
    /// The body.
    pub body: Body,
    /// Who the request says it is.
    pub auth: Authorization,
    /// A ceiling on this request shorter than the configured one.
    pub timeout: Option<Duration>,
    /// An explicit server command (a login, a push): offline mode does
    /// not refuse it.
    pub explicit: bool,
}

impl Request {
    /// A request with `method` to `url`, no headers, no body, read
    /// credentials.
    pub fn new(method: Method, url: impl Into<String>) -> Self {
        Request {
            method,
            url: url.into(),
            headers: Vec::new(),
            body: Body::Empty,
            auth: Authorization::ReadCredentials,
            timeout: None,
            explicit: false,
        }
    }

    /// `GET url`.
    pub fn get(url: impl Into<String>) -> Self {
        Self::new(Method::Get, url)
    }

    /// `HEAD url`.
    pub fn head(url: impl Into<String>) -> Self {
        Self::new(Method::Head, url)
    }

    /// `PUT url`.
    pub fn put(url: impl Into<String>) -> Self {
        Self::new(Method::Put, url)
    }

    /// `POST url`.
    pub fn post(url: impl Into<String>) -> Self {
        Self::new(Method::Post, url)
    }

    /// `PATCH url`.
    pub fn patch(url: impl Into<String>) -> Self {
        Self::new(Method::Patch, url)
    }

    /// `DELETE url`.
    pub fn delete(url: impl Into<String>) -> Self {
        Self::new(Method::Delete, url)
    }

    /// Add a header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Send these bytes.
    pub fn bytes(mut self, body: Vec<u8>) -> Self {
        self.body = Body::Bytes(body);
        self
    }

    /// Stream the file at `path`.
    pub fn file(mut self, path: impl Into<PathBuf>) -> Self {
        self.body = Body::File(path.into());
        self
    }

    /// Authorise with this bearer token, not the read credentials.
    pub fn bearer(mut self, token: impl Into<String>) -> Self {
        self.auth = Authorization::Bearer(token.into());
        self
    }

    /// Authorise with `token` when there is one, else send anonymously:
    /// never the read credentials.
    pub fn bearer_or_anonymous(self, token: Option<&str>) -> Self {
        match token {
            Some(t) => self.bearer(t),
            None => self.anonymous(),
        }
    }

    /// Send anonymously: the read credentials are not asked.
    pub fn anonymous(mut self) -> Self {
        self.auth = Authorization::Anonymous;
        self
    }

    /// Sign with `signer` just before sending.
    pub fn signed(mut self, signer: Arc<dyn RequestSigner>) -> Self {
        self.auth = Authorization::Signed(signer);
        self
    }

    /// Give up after `timeout` rather than the configured ceiling.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Mark this an explicit server command, which offline mode does
    /// not refuse.
    pub fn explicit(mut self) -> Self {
        self.explicit = true;
        self
    }

    /// The first value of header `name`, compared without case.
    pub fn header_value(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}
