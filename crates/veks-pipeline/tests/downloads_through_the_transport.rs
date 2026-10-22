// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **The pipeline's downloaders go through the process's transport**
//! (`docs/design/srd-crate-boundaries.md` TR-9, §10 Q5, acceptance test
//! 9): `download s2ag` and `download bulk` (template and COS modes),
//! against a local server, take the transport's user agent, honour
//! offline mode and a trusted CA certificate, and send their own
//! authorisation — the S2 API key, none, a SigV4 signature — and never a
//! vectordata read token, even when `$VECTORDATA_TOKEN` would authorise
//! every URL.
//!
//! The transport is the process's (`vectordata::transport::shared`),
//! configured once from the settings `$VECTORDATA_HOME` isolates and the
//! environment, so each command runs in a child process given those
//! variables; this process's environment is never changed.

mod support;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use support::{options, run_command};

/// One request as the server saw it.
#[derive(Debug, Clone)]
struct Seen {
    method: String,
    target: String,
    headers: Vec<(String, String)>,
}

impl Seen {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// The bytes every served file has.
const FILE_BYTES: &[u8] = b"0123456789";

/// A stand-in for the three services: the S2 release file list, files
/// under `/files/`, and an S3 bucket `bkt` holding `p/obj.bin`. Records
/// every request. Plain HTTP, or HTTPS with a given certificate.
struct Server {
    base: String,
    seen: Arc<Mutex<Vec<Seen>>>,
    handle: axum_server::Handle,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Server {
    fn start(tls: Option<(Vec<u8>, Vec<u8>)>) -> Self {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let base = Arc::new(OnceLock::new());
        let served_base = Arc::clone(&base);
        let handle = axum_server::Handle::new();
        let serving = handle.clone();
        let https = tls.is_some();
        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
            rt.block_on(async move {
                let app = axum::Router::new().fallback(move |req: axum::extract::Request| {
                    let (log, base) = (Arc::clone(&log), Arc::clone(&served_base));
                    async move { answer(&log, &base, req) }
                });
                let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
                match tls {
                    Some((cert, key)) => {
                        let config = axum_server::tls_rustls::RustlsConfig::from_pem(cert, key).await.unwrap();
                        axum_server::bind_rustls(addr, config).handle(serving).serve(app.into_make_service()).await.unwrap();
                    }
                    None => axum_server::bind(addr).handle(serving).serve(app.into_make_service()).await.unwrap(),
                }
            });
        });
        let port = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(handle.listening())
            .expect("the server listens")
            .port();
        let scheme = if https { "https" } else { "http" };
        let base = base.get_or_init(|| format!("{scheme}://127.0.0.1:{port}")).clone();
        Server { base, seen, handle, thread: Some(thread) }
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.handle.shutdown();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Record `req` and answer it; `base` is the server's own URL, which the
/// S2 file list names its files under.
fn answer(log: &Mutex<Vec<Seen>>, base: &OnceLock<String>, req: axum::extract::Request) -> axum::response::Response {
    use axum::response::IntoResponse;
    let target = req.uri().path_and_query().map(|p| p.to_string()).unwrap_or_default();
    let headers = req
        .headers()
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    log.lock().unwrap().push(Seen { method: req.method().to_string(), target: target.clone(), headers });
    let body: Vec<u8> = if target.starts_with("/datasets/v1/release/") {
        let base = base.get().map(String::as_str).unwrap_or("");
        format!(r#"{{"files": ["{base}/files/part-a.gz?sig=ok"]}}"#).into_bytes()
    } else if target.starts_with("/bkt?") {
        format!(
            "<ListBucketResult><IsTruncated>false</IsTruncated><Contents><Key>p/obj.bin</Key>\
             <Size>{}</Size></Contents></ListBucketResult>",
            FILE_BYTES.len()
        )
        .into_bytes()
    } else {
        FILE_BYTES.to_vec()
    };
    ([("content-length", body.len().to_string())], body).into_response()
}

const CHILD_CMD: &str = "VEKS_TEST_DL_CMD";
const CHILD_OPTS: &str = "VEKS_TEST_DL_OPTS";
const CHILD_DIR: &str = "VEKS_TEST_DL_DIR";

/// A client home with `settings` (plus a cache directory) as its
/// `settings.yaml`.
fn home(dir: &Path, settings: &str) -> PathBuf {
    let home = dir.join("home");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(home.join("settings.yaml"), format!("cache_dir: {}\n{settings}", dir.join("cache").display())).unwrap();
    home
}

/// What a child run reported: the command's status and message.
#[derive(Debug)]
struct Outcome {
    ok: bool,
    message: String,
}

/// Run `command` with `opts` in a child process whose transport is
/// configured by `home`, with `$VECTORDATA_TOKEN` set to a token that
/// would match every URL, and the downloaders' own credentials set.
fn run_child(home: &Path, work: &Path, command: &str, opts: &[(&str, String)]) -> Outcome {
    let opts: String = opts.iter().map(|(k, v)| format!("{k}={v}\n")).collect();
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["download_in_a_child", "--exact", "--nocapture", "--test-threads=1"])
        .env("VECTORDATA_HOME", home)
        .env_remove("VECTORDATA_OFFLINE")
        .env("VECTORDATA_TOKEN", "vectordata-secret")
        .env("S2_API_KEY", "s2-key")
        .env("AWS_ACCESS_KEY_ID", "AKIDTEST")
        .env("AWS_SECRET_ACCESS_KEY", "aws-secret")
        .env_remove("AWS_SESSION_TOKEN")
        .env(CHILD_CMD, command)
        .env(CHILD_OPTS, opts)
        .env(CHILD_DIR, work)
        .output()
        .expect("run the child");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out.status.success(), "the child failed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    let field = |name: &str| {
        stdout
            .lines()
            .find_map(|l| l.split_once(name).map(|(_, v)| v))
            .unwrap_or_else(|| panic!("no {name} in the child's output:\n{stdout}"))
            .to_string()
    };
    Outcome { ok: field("DOWNLOAD-STATUS=") == "Ok", message: field("DOWNLOAD-MESSAGE=") }
}

/// The three downloads, each into its own directory under `work`.
fn downloads(base: &str, work: &Path) -> Vec<(&'static str, Vec<(&'static str, String)>)> {
    vec![
        (
            "download s2ag",
            vec![
                ("release", "2026-08-12".into()),
                ("dataset-name", "s2orc".into()),
                ("output", work.join("s2").display().to_string()),
                ("api-base", format!("{base}/datasets/v1")),
                ("concurrency", "1".into()),
                ("tries", "1".into()),
            ],
        ),
        (
            "download bulk",
            vec![
                ("mode", "template".into()),
                ("output", work.join("bulk").display().to_string()),
                ("baseurl", format!("{base}/files/part-${{n}}.bin")),
                ("tokens", "n=[0..1]".into()),
                ("concurrency", "1".into()),
                ("tries", "1".into()),
            ],
        ),
        (
            "download bulk",
            vec![
                ("mode", "cos".into()),
                ("output", work.join("cos").display().to_string()),
                ("endpoint", base.to_string()),
                ("bucket", "bkt".into()),
                ("prefix", "p/".into()),
                ("region", "us-south".into()),
                ("concurrency", "1".into()),
                ("tries", "1".into()),
            ],
        ),
    ]
}

/// **Each request carries its own authorisation and the transport's user
/// agent; none carries the vectordata token.**
#[test]
fn downloads_carry_their_own_authorisation_never_a_vectordata_token() {
    let server = Server::start(None);
    let work = tempfile::tempdir().unwrap();
    let home = home(work.path(), "");
    for (command, opts) in downloads(&server.base, work.path()) {
        let outcome = run_child(&home, work.path(), command, &opts);
        assert!(outcome.ok, "{command}: {}", outcome.message);
    }
    assert_eq!(std::fs::read(work.path().join("s2/part-a.gz")).unwrap(), FILE_BYTES);
    assert_eq!(std::fs::read(work.path().join("cos/obj.bin")).unwrap(), FILE_BYTES);

    let seen = server.seen();
    assert!(seen.len() >= 5, "{seen:#?}");
    for req in &seen {
        assert!(
            req.headers.iter().all(|(_, v)| !v.contains("vectordata-secret")),
            "a vectordata read token reached a third-party host: {req:#?}"
        );
        let agent = req.header("user-agent").unwrap_or("");
        assert!(agent.starts_with("vectordata/"), "not the transport's user agent: {req:#?}");
        let auth = req.header("authorization");
        if req.target.starts_with("/bkt") {
            assert!(
                auth.is_some_and(|a| a.starts_with("AWS4-HMAC-SHA256 Credential=AKIDTEST/")),
                "a COS request is SigV4-signed: {req:#?}"
            );
            assert!(req.header("x-amz-date").is_some() && req.header("x-amz-content-sha256").is_some());
        } else {
            assert_eq!(auth, None, "{req:#?}");
        }
        if req.target.starts_with("/datasets/v1/release/") {
            assert_eq!(req.header("x-api-key"), Some("s2-key"), "{req:#?}");
        } else {
            assert_eq!(req.header("x-api-key"), None, "the API key goes to the file list only: {req:#?}");
        }
    }
    assert!(seen.iter().any(|r| r.method == "GET" && r.target.starts_with("/bkt?")), "the COS listing: {seen:#?}");
}

/// **Offline mode stops every downloader before the network**, with the
/// offline error.
#[test]
fn offline_mode_stops_every_downloader_before_the_network() {
    let server = Server::start(None);
    let work = tempfile::tempdir().unwrap();
    let home = home(work.path(), "offline: on\n");
    for (command, opts) in downloads(&server.base, work.path()) {
        let outcome = run_child(&home, work.path(), command, &opts);
        assert!(!outcome.ok, "{command} succeeded offline: {}", outcome.message);
    }
    let seen = server.seen();
    assert!(seen.is_empty(), "offline mode let a download reach the server: {seen:#?}");
}

/// **A CA certificate the settings trust reaches a private HTTPS
/// server**, and the same server is refused without it: verification
/// stays on.
#[test]
fn a_trusted_ca_certificate_reaches_a_private_https_server() {
    let work = tempfile::tempdir().unwrap();
    let ck = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_string()]).unwrap();
    let cert = work.path().join("private-ca.pem");
    std::fs::write(&cert, ck.cert.pem()).unwrap();
    let server = Server::start(Some((ck.cert.pem().into_bytes(), ck.key_pair.serialize_pem().into_bytes())));

    let untrusting = home(&work.path().join("untrusting"), "");
    for (command, opts) in downloads(&server.base, &work.path().join("untrusting")).into_iter().take(2) {
        let outcome = run_child(&untrusting, work.path(), command, &opts);
        assert!(!outcome.ok, "{command} accepted an untrusted certificate: {}", outcome.message);
    }

    let trusting = home(&work.path().join("trusting"), &format!("trusted_ca_certs:\n  - {}\n", cert.display()));
    for (command, opts) in downloads(&server.base, &work.path().join("trusting")).into_iter().take(2) {
        let outcome = run_child(&trusting, work.path(), command, &opts);
        assert!(outcome.ok, "{command} refused a trusted CA certificate: {}", outcome.message);
    }
    assert_eq!(std::fs::read(work.path().join("trusting/s2/part-a.gz")).unwrap(), FILE_BYTES);
}

/// The child half of the tests above; does nothing unless launched by
/// one of them. Prints the command's status and message.
#[test]
fn download_in_a_child() {
    let (Ok(command), Ok(opts), Ok(dir)) = (std::env::var(CHILD_CMD), std::env::var(CHILD_OPTS), std::env::var(CHILD_DIR))
    else {
        return;
    };
    let pairs: Vec<(String, String)> = opts
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let pairs: Vec<(&str, &str)> = pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let run = run_command(&command, Path::new(&dir), &options(&pairs));
    println!("DOWNLOAD-STATUS={:?}", run.result.status);
    println!("DOWNLOAD-MESSAGE={}", run.result.message.replace('\n', " | "));
}
