// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **The readiness probe trusts vecd's own certificate and nothing else**
//! (`docs/design/srd-crate-boundaries.md` TR-11, §10 Q9): `vecd daemon
//! status --wait` over HTTPS reaches a server presenting the configured
//! `tls_cert`, and refuses a different self-signed server at another
//! origin. Before the probe went through the transport it accepted every
//! certificate.
//!
//! The probe connects to loopback but verifies the name the certificate
//! was issued for, so a certificate whose subject alt names leave out
//! `127.0.0.1` and `localhost` (one issued for a `--host` name, or a
//! CA-issued one for a public name) is reached too.
//!
//! Driven through the compiled binary, with a config directory of the
//! test's own (`$VECD_CONFIG` on the child only).

use std::path::{Path, PathBuf};
use std::process::Command;

/// A self-signed certificate for `127.0.0.1` and `localhost`, written as
/// PEM files under `dir` with `stem`; returns (certificate, key) paths
/// and their PEMs.
fn self_signed(dir: &Path, stem: &str) -> (PathBuf, PathBuf, Vec<u8>, Vec<u8>) {
    self_signed_for(dir, stem, &["127.0.0.1", "localhost"])
}

/// A self-signed certificate whose subject alt names are exactly
/// `names`, written as PEM files under `dir` with `stem`; returns
/// (certificate, key) paths and their PEMs.
fn self_signed_for(dir: &Path, stem: &str, names: &[&str]) -> (PathBuf, PathBuf, Vec<u8>, Vec<u8>) {
    let ck = rcgen::generate_simple_self_signed(names.iter().map(|n| n.to_string()).collect::<Vec<_>>()).unwrap();
    let (cert_pem, key_pem) = (ck.cert.pem().into_bytes(), ck.key_pair.serialize_pem().into_bytes());
    let (cert, key) = (dir.join(format!("{stem}-cert.pem")), dir.join(format!("{stem}-key.pem")));
    std::fs::write(&cert, &cert_pem).unwrap();
    std::fs::write(&key, &key_pem).unwrap();
    (cert, key, cert_pem, key_pem)
}

/// An HTTPS server on `127.0.0.1` presenting the given certificate and
/// answering `/healthz` with 200.
struct HealthServer {
    port: u16,
    handle: axum_server::Handle,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl HealthServer {
    fn start(cert_pem: Vec<u8>, key_pem: Vec<u8>) -> Self {
        let handle = axum_server::Handle::new();
        let serving = handle.clone();
        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().unwrap();
            rt.block_on(async move {
                let tls = axum_server::tls_rustls::RustlsConfig::from_pem(cert_pem, key_pem).await.unwrap();
                let app = axum::Router::new().route("/healthz", axum::routing::get(|| async { "ok" }));
                axum_server::bind_rustls("127.0.0.1:0".parse().unwrap(), tls)
                    .handle(serving)
                    .serve(app.into_make_service())
                    .await
                    .unwrap();
            });
        });
        let port = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(handle.listening())
            .expect("the TLS server listens")
            .port();
        HealthServer { port, handle, thread: Some(thread) }
    }
}

impl Drop for HealthServer {
    fn drop(&mut self) {
        self.handle.shutdown();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// `vecd daemon status --wait` with a config binding `port` over TLS
/// with `cert`/`key` as vecd's own.
fn status_wait(work: &Path, port: u16, cert: &Path, key: &Path) -> std::process::Output {
    let config = work.join(format!("config-{port}"));
    let data = work.join(format!("data-{port}"));
    std::fs::create_dir_all(&config).unwrap();
    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(
        config.join("vecd.conf"),
        format!("bind = 127.0.0.1:{port}\ntls_cert = {}\ntls_key = {}\n", cert.display(), key.display()),
    )
    .unwrap();
    Command::new(env!("CARGO_BIN_EXE_vecd"))
        .env("VECD_CONFIG", &config)
        .env_remove("PREFER_CONFIG")
        .arg("--data-dir")
        .arg(&data)
        .args(["daemon", "status", "--wait", "--timeout", "2"])
        .output()
        .expect("run vecd")
}

#[test]
fn the_readiness_probe_trusts_its_own_certificate_and_refuses_another_origin() {
    let work = tempfile::tempdir().unwrap();
    let (own_cert, own_key, own_cert_pem, own_key_pem) = self_signed(work.path(), "own");
    let (_, _, other_cert_pem, other_key_pem) = self_signed(work.path(), "other");
    let own = HealthServer::start(own_cert_pem, own_key_pem);
    let other = HealthServer::start(other_cert_pem, other_key_pem);

    let reached = status_wait(work.path(), own.port, &own_cert, &own_key);
    assert!(
        reached.status.success(),
        "the probe refused vecd's own certificate:\n{}",
        String::from_utf8_lossy(&reached.stderr)
    );

    let refused = status_wait(work.path(), other.port, &own_cert, &own_key);
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        !refused.status.success(),
        "the probe accepted a self-signed certificate that is not vecd's own, at another origin"
    );
    assert!(stderr.contains("did not answer /healthz"), "{stderr}");
}

/// **A certificate issued only for a DNS name is reached on loopback**:
/// vecd's own certificate names `vecd.test` and no IP address, the
/// server listens on `127.0.0.1`, and the probe verifies `vecd.test`
/// against vecd's own certificate while connecting to loopback.
#[test]
fn the_readiness_probe_reaches_a_certificate_issued_only_for_a_dns_name() {
    let work = tempfile::tempdir().unwrap();
    let (cert, key, cert_pem, key_pem) = self_signed_for(work.path(), "named", &["vecd.test"]);
    let server = HealthServer::start(cert_pem, key_pem);

    let reached = status_wait(work.path(), server.port, &cert, &key);
    assert!(
        reached.status.success(),
        "the probe refused vecd's own certificate issued for vecd.test:\n{}",
        String::from_utf8_lossy(&reached.stderr)
    );
}

/// **A name-only certificate that is not vecd's own is still refused**:
/// vecd's certificate and the server's both name `vecd.test`, but the
/// server's is a different self-signed certificate.
#[test]
fn the_readiness_probe_refuses_another_certificate_for_the_same_name() {
    let work = tempfile::tempdir().unwrap();
    let (own_cert, own_key, _, _) = self_signed_for(work.path(), "own", &["vecd.test"]);
    let (_, _, other_cert_pem, other_key_pem) = self_signed_for(work.path(), "other", &["vecd.test"]);
    let other = HealthServer::start(other_cert_pem, other_key_pem);

    let refused = status_wait(work.path(), other.port, &own_cert, &own_key);
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(!refused.status.success(), "the probe accepted a certificate for vecd.test that is not vecd's own");
    assert!(stderr.contains("did not answer /healthz"), "{stderr}");
}
