// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. The transport every remote request goes through:
//! `veks-transport`'s client API, re-exported for programs that already
//! depend on vectordata (`docs/design/srd-crate-boundaries.md` CB-3),
//! and the process's one [`Transport`], built from the user's settings
//! and the environment ([`shared`]).
//!
//! The configuration is a value, built by a pure function
//! ([`config_from`]) from the settings and environment values it reads
//! once per process: offline mode, the TLS trust, the parallel stream
//! and download worker counts, and the S3 region. Read credentials are
//! the `vectordata login` store and `$VECTORDATA_TOKEN`
//! ([`StoredCredentials`](crate::credentials::StoredCredentials)).
//!
//! A dataset's bytes are read through its facets' storage, which lands
//! them in the cache; [`shared`] is for the small files beside them and
//! for programs with requests of their own.
//!
//! ```no_run
//! use vectordata::transport::{Request, shared};
//!
//! # fn main() -> Result<(), vectordata::TransportError> {
//! let resp = shared().send(Request::get("https://data.example/catalog.json"))?;
//! if resp.is_success() {
//!     let text = resp.text()?;
//!     # let _ = text;
//! }
//! # Ok(()) }
//! ```

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

pub use veks_transport::{
    Authorization, Body, CachePlacement, Credentials, HttpTransport, Method, NoCredentials, RangeSource, RemoteSource,
    Request, RequestSigner, Response, RetryPolicy, Timeouts, TlsTrust, Transport, TransportConfig, TransportError,
    TransportErrorKind, is_remote_url, upload,
};

/// The environment values the transport's configuration is built
/// from, read once ([`TransportEnv::from_process`]) and passed to the
/// pure [`config_from`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TransportEnv {
    /// [`OFFLINE_ENV`](crate::settings::OFFLINE_ENV).
    pub offline: Option<String>,
    /// `VECTORDATA_HTTP_RUNTIMES`: independent HTTP clients (runtime
    /// threads) requests are spread over.
    pub http_runtimes: Option<String>,
    /// `VECTORDATA_DOWNLOAD_CONCURRENCY`: chunk-download workers.
    pub download_concurrency: Option<String>,
    /// `AWS_REGION`.
    pub aws_region: Option<String>,
    /// `AWS_DEFAULT_REGION`, used when `AWS_REGION` is unset.
    pub aws_default_region: Option<String>,
}

impl TransportEnv {
    /// The values this process's environment holds.
    pub fn from_process() -> Self {
        let var = |k: &str| std::env::var(k).ok();
        TransportEnv {
            offline: var(crate::settings::OFFLINE_ENV),
            http_runtimes: var("VECTORDATA_HTTP_RUNTIMES"),
            download_concurrency: var("VECTORDATA_DOWNLOAD_CONCURRENCY"),
            aws_region: var("AWS_REGION"),
            aws_default_region: var("AWS_DEFAULT_REGION"),
        }
    }
}

/// The transport configuration for the `offline` settings value, the
/// settings file's TLS trust and the environment values `env`: pure, so
/// every rule is tested with values and no process environment.
///
/// - offline: the environment decides when it says anything
///   recognizable, else the setting ([`offline_from`](crate::settings::offline_from));
/// - parallel streams and download workers: the environment's positive
///   count, else [`DEFAULT_PARALLEL_STREAMS`](veks_transport::DEFAULT_PARALLEL_STREAMS)
///   for both, so runtimes and workers stay matched 1:1 (with N
///   runtimes and M workers the per-runtime fanout is M/N, and an
///   imbalance leaves cores idle);
/// - S3 region: `AWS_REGION`, else `AWS_DEFAULT_REGION`, else
///   `us-east-1`;
/// - the user agent `vectordata/<version>`, ten redirects, a one-hour
///   request ceiling and a ten-second connect timeout.
pub fn config_from(offline_setting: Option<&str>, tls: TlsTrust, env: &TransportEnv) -> TransportConfig {
    let count = |v: &Option<String>| {
        v.as_deref().and_then(|s| s.parse::<usize>().ok()).filter(|&n| n > 0).unwrap_or(veks_transport::DEFAULT_PARALLEL_STREAMS)
    };
    TransportConfig {
        offline: crate::settings::offline_from(offline_setting, env.offline.as_deref()),
        offline_hint: Some(format!("{} / settings `{}`", crate::settings::OFFLINE_ENV, crate::settings::OFFLINE_KEY)),
        tls,
        user_agent: concat!("vectordata/", env!("CARGO_PKG_VERSION")).to_string(),
        timeouts: Timeouts::default(),
        redirects: 10,
        parallel_streams: count(&env.http_runtimes),
        download_concurrency: count(&env.download_concurrency),
        s3_region: env
            .aws_region
            .clone()
            .or_else(|| env.aws_default_region.clone())
            .unwrap_or_else(|| "us-east-1".to_string()),
    }
}

/// The process's transport: configured once, on first use, from the
/// settings file and the environment ([`config_from`]), with the stored
/// credentials. Every request vectordata makes goes through it, so they
/// share one client pool and one TLS policy.
pub fn shared() -> &'static Transport {
    static TRANSPORT: OnceLock<Transport> = OnceLock::new();
    TRANSPORT.get_or_init(|| {
        let config = config_from(
            crate::settings::setting_value(crate::settings::OFFLINE_KEY).as_deref(),
            crate::settings::tls_trust(),
            &TransportEnv::from_process(),
        );
        Transport::with_credentials(config, Arc::new(crate::credentials::StoredCredentials))
    })
}

/// Refuse a request to `url` in offline mode ([`Transport::ensure_online`]).
pub(crate) fn ensure_online(url: &str) -> std::io::Result<()> {
    shared().ensure_online(url)
}

/// `url` in its wire form: `s3://` as its HTTPS endpoint.
pub(crate) fn normalize_remote_url(url: &str) -> Cow<'_, str> {
    shared().normalize_url(url)
}

/// The byte ranges of the file at `url` (its wire form), through the
/// shared transport.
pub(crate) fn http(url: url::Url) -> HttpTransport {
    HttpTransport::new(shared(), url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> TransportEnv {
        TransportEnv::default()
    }

    /// Unset, everything is the default: online, 32 streams and
    /// workers, `us-east-1`, vectordata's user agent.
    #[test]
    fn an_empty_environment_gives_the_defaults() {
        let c = config_from(None, TlsTrust::default(), &env());
        assert!(!c.offline);
        assert_eq!((c.parallel_streams, c.download_concurrency), (32, 32));
        assert_eq!(c.s3_region, "us-east-1");
        assert!(c.user_agent.starts_with("vectordata/"), "{}", c.user_agent);
        assert_eq!(c.redirects, 10);
        assert_eq!(c.offline_hint.as_deref(), Some("VECTORDATA_OFFLINE / settings `offline`"));
    }

    /// The environment decides offline mode when it says something
    /// recognizable; otherwise the setting does.
    #[test]
    fn offline_is_the_environment_then_the_setting() {
        let on = TransportEnv { offline: Some("1".into()), ..env() };
        let off = TransportEnv { offline: Some("off".into()), ..env() };
        let noise = TransportEnv { offline: Some("maybe".into()), ..env() };
        assert!(config_from(None, TlsTrust::default(), &on).offline);
        assert!(!config_from(Some("on"), TlsTrust::default(), &off).offline);
        assert!(config_from(Some("on"), TlsTrust::default(), &noise).offline);
        assert!(config_from(Some("on"), TlsTrust::default(), &env()).offline);
    }

    /// A positive count is taken, anything else falls back, and each
    /// knob tunes its own layer.
    #[test]
    fn stream_and_worker_counts_come_from_their_own_variables() {
        let e = TransportEnv { http_runtimes: Some("8".into()), download_concurrency: Some("64".into()), ..env() };
        let c = config_from(None, TlsTrust::default(), &e);
        assert_eq!((c.parallel_streams, c.download_concurrency), (8, 64));
        let bad = TransportEnv { http_runtimes: Some("0".into()), download_concurrency: Some("x".into()), ..env() };
        let c = config_from(None, TlsTrust::default(), &bad);
        assert_eq!((c.parallel_streams, c.download_concurrency), (32, 32));
    }

    /// `AWS_REGION` wins over `AWS_DEFAULT_REGION`.
    #[test]
    fn the_s3_region_is_aws_region_then_the_default_region() {
        let both = TransportEnv { aws_region: Some("us-east-2".into()), aws_default_region: Some("eu-west-1".into()), ..env() };
        assert_eq!(config_from(None, TlsTrust::default(), &both).s3_region, "us-east-2");
        let default_only = TransportEnv { aws_default_region: Some("eu-west-1".into()), ..env() };
        assert_eq!(config_from(None, TlsTrust::default(), &default_only).s3_region, "eu-west-1");
    }

    /// The settings' TLS trust is carried as given.
    #[test]
    fn the_tls_trust_is_carried_through() {
        let tls = TlsTrust { trusted_ca_certs: vec!["/a.pem".into()], trust_self_signed: vec!["https://h:1".into()] };
        assert_eq!(config_from(None, tls.clone(), &env()).tls, tls);
    }
}
