// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **Every request honours the per-origin TLS trust** (`settings.yaml`
//! `trust_self_signed`, `docs/design/srd-crate-boundaries.md` TR-12 and
//! acceptance test 8): a self-signed HTTPS origin the user trusts is
//! reachable by `fetch_if_modified` and `fetch_text` alike, and one the
//! user does not trust is refused.
//!
//! The trust policy is read from the client's settings, which
//! `$VECTORDATA_HOME` isolates; the reads run in a child process given
//! that variable, so this process's environment is never changed.

use std::path::{Path, PathBuf};

/// A self-signed HTTPS server for `127.0.0.1` serving `root`.
struct TlsServer {
    port: u16,
    handle: axum_server::Handle,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl TlsServer {
    fn start(root: &Path) -> Self {
        let cert = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_string(), "localhost".to_string()]).unwrap();
        let (cert_pem, key_pem) = (cert.cert.pem().into_bytes(), cert.key_pair.serialize_pem().into_bytes());
        let root = root.to_path_buf();
        let handle = axum_server::Handle::new();
        let serving = handle.clone();
        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
            rt.block_on(async move {
                let tls = axum_server::tls_rustls::RustlsConfig::from_pem(cert_pem, key_pem).await.unwrap();
                let app = axum::Router::new().fallback_service(tower_http::services::ServeDir::new(root));
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
        TlsServer { port, handle, thread: Some(thread) }
    }

    fn origin(&self) -> String {
        format!("https://127.0.0.1:{}", self.port)
    }
}

impl Drop for TlsServer {
    fn drop(&mut self) {
        self.handle.shutdown();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

const CHILD_URL: &str = "VECTORDATA_TEST_TRUST_URL";
const CHILD_COPY: &str = "VECTORDATA_TEST_TRUST_COPY";
const CHILD_EXPECT: &str = "VECTORDATA_TEST_TRUST_EXPECT";

/// Run [`fetch_in_a_child`] with `home` as the client home.
fn run_child(home: &Path, url: &str, copy: &Path, expect: &str) -> std::process::Output {
    std::process::Command::new(std::env::current_exe().unwrap())
        .args(["fetch_in_a_child", "--exact", "--nocapture", "--test-threads=1"])
        .env("VECTORDATA_HOME", home)
        .env_remove(vectordata::settings::OFFLINE_ENV)
        .env_remove("VECTORDATA_TOKEN")
        .env(CHILD_URL, url)
        .env(CHILD_COPY, copy)
        .env(CHILD_EXPECT, expect)
        .output()
        .expect("run the child")
}

/// A client home whose settings trust `origins` without verification.
fn home_trusting(dir: &Path, origins: &[String]) -> PathBuf {
    let home = dir.join("home");
    std::fs::create_dir_all(&home).unwrap();
    let mut yaml = format!("cache_dir: {}\n", dir.join("cache").display());
    if !origins.is_empty() {
        yaml.push_str("trust_self_signed:\n");
        for o in origins {
            yaml.push_str(&format!("  - {o}\n"));
        }
    }
    std::fs::write(home.join("settings.yaml"), yaml).unwrap();
    home
}

/// **A self-signed origin trusted for itself is fetched by
/// `fetch_if_modified`**, as it is by `fetch_text`. Before the transport
/// moved, `fetch_if_modified` used a client without the per-origin
/// trust and failed the certificate check here.
#[test]
fn a_trusted_self_signed_origin_is_fetched_by_fetch_if_modified() {
    let served = tempfile::tempdir().unwrap();
    std::fs::write(served.path().join("list.txt"), "one\ntwo\n").unwrap();
    let server = TlsServer::start(served.path());
    let work = tempfile::tempdir().unwrap();
    let home = home_trusting(work.path(), &[server.origin()]);
    let url = format!("{}/list.txt", server.origin());
    let out = run_child(&home, &url, &work.path().join("copy.txt"), "ok");
    assert!(
        out.status.success(),
        "a trusted self-signed origin was refused:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read_to_string(work.path().join("copy.txt")).unwrap(), "one\ntwo\n");
}

/// **An origin not listed stays verified**, so a self-signed server is
/// refused: the trust is per origin, not a global switch.
#[test]
fn an_untrusted_self_signed_origin_is_refused() {
    let served = tempfile::tempdir().unwrap();
    std::fs::write(served.path().join("list.txt"), "one\n").unwrap();
    let server = TlsServer::start(served.path());
    let work = tempfile::tempdir().unwrap();
    let home = home_trusting(work.path(), &["https://127.0.0.1:1".to_string()]);
    let url = format!("{}/list.txt", server.origin());
    let out = run_child(&home, &url, &work.path().join("copy.txt"), "refused");
    assert!(
        out.status.success(),
        "an untrusted self-signed origin was not refused:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!work.path().join("copy.txt").exists());
}

/// The child half of the tests above; does nothing unless launched by
/// one of them.
#[test]
fn fetch_in_a_child() {
    let (Ok(url), Ok(copy), Ok(expect)) =
        (std::env::var(CHILD_URL), std::env::var(CHILD_COPY), std::env::var(CHILD_EXPECT))
    else {
        return;
    };
    let copy = PathBuf::from(copy);
    let modified = vectordata::fetch::fetch_if_modified(&url, &copy);
    let text = vectordata::fetch::fetch_text(&url);
    match expect.as_str() {
        "ok" => {
            assert!(modified.expect("fetch_if_modified reaches a trusted origin"), "no copy yet: fetched");
            assert_eq!(text.expect("fetch_text reaches a trusted origin"), std::fs::read_to_string(&copy).unwrap());
            assert!(!vectordata::fetch::fetch_if_modified(&url, &copy).unwrap(), "the copy is current");
        }
        _ => {
            assert!(modified.is_err(), "fetch_if_modified accepted an untrusted certificate");
            assert!(text.is_err(), "fetch_text accepted an untrusted certificate");
        }
    }
}
