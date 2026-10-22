// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A program that depends on veks-io and veks-transport alone reads a
//! remote `.fvecs` through veks-io's `XvecReader` over the transport's
//! `ByteSource`, with no vectordata (`docs/design/srd-crate-boundaries.md`
//! TR-8, acceptance test 3): through the chunk store when no `.mref` is
//! published, merkle-verified when one is, and refused with the offline
//! error in offline mode. The cache copy and its sidecar are exactly the
//! paths the caller placed (TR-7), and the read credentials the program
//! supplies are what a request sends (TR-4, acceptance test 2).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::sync::{Arc, Mutex};

use veks_apicore::transport::{Credentials, Request, TransportConfig};
use veks_apicore::{ByteSource, VectorReader};
use veks_io::XvecReader;
use veks_transport::merkle::MerkleRef;
use veks_transport::{CachePlacement, RemoteSource, Transport};

/// A minimal HTTP/1.1 file server over `files` (path → bytes): `HEAD`,
/// `GET`, and single `Range: bytes=a-b` requests. Records every
/// request's `Authorization` header (empty when there was none).
struct Server {
    base: String,
    auth: Arc<Mutex<Vec<String>>>,
}

fn serve(files: Vec<(&'static str, Vec<u8>)>) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let auth = Arc::new(Mutex::new(Vec::new()));
    let seen = auth.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request = String::new();
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                request.push_str(&line);
            }
            let mut lines = request.lines();
            let first: Vec<&str> = lines.next().unwrap_or("").split_whitespace().collect();
            let (method, path) = (first.first().copied().unwrap_or(""), first.get(1).copied().unwrap_or(""));
            let header = |name: &str| {
                request
                    .lines()
                    .find_map(|l| l.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.trim().to_string()))
            };
            seen.lock().unwrap().push(header("authorization").unwrap_or_default());
            if let Some(len) = header("content-length").and_then(|v| v.parse::<usize>().ok()) {
                let mut body = vec![0u8; len];
                let _ = reader.read_exact(&mut body);
            }
            let body = files.iter().find(|(p, _)| *p == path).map(|(_, b)| b.clone());
            let response = match (method, body) {
                (_, None) => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
                ("HEAD", Some(b)) => format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                    b.len()
                )
                .into_bytes(),
                (_, Some(b)) => match header("range").and_then(|r| {
                    let (a, z) = r.strip_prefix("bytes=")?.split_once('-')?;
                    Some((a.parse::<usize>().ok()?, z.parse::<usize>().ok()?))
                }) {
                    Some((a, z)) => {
                        let z = z.min(b.len() - 1);
                        let mut out = format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {a}-{z}/{}\r\n\
                             Accept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                            z + 1 - a,
                            b.len()
                        )
                        .into_bytes();
                        out.extend_from_slice(&b[a..=z]);
                        out
                    }
                    None => {
                        let mut out = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                            b.len()
                        )
                        .into_bytes();
                        out.extend_from_slice(&b);
                        out
                    }
                },
            };
            let _ = stream.write_all(&response);
        }
    });
    Server { base, auth }
}

/// `records` dim-`dim` f32 records, element `d` of record `i` being
/// `i * dim + d`.
fn fvecs(records: usize, dim: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for i in 0..records {
        out.extend_from_slice(&(dim as i32).to_le_bytes());
        for d in 0..dim {
            out.extend_from_slice(&((i * dim + d) as f32).to_le_bytes());
        }
    }
    out
}

/// Where the caller keeps `name`'s copy under `dir`.
fn placed(dir: &Path, name: &str) -> CachePlacement {
    CachePlacement {
        root: dir.to_path_buf(),
        data: dir.join(name),
        merkle_state: dir.join(format!("{name}.mrkl")),
        chunk_map: dir.join(format!("{name}.chunks")),
    }
}

/// Every file under `dir`, by name, sorted.
fn names(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> =
        std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    out.sort();
    out
}

fn check(reader: &XvecReader<f32, RemoteSource>, records: usize, dim: usize) {
    assert_eq!((reader.count(), reader.dim()), (records, dim));
    for i in [0, records / 2, records - 1] {
        let v = reader.get(i).unwrap();
        assert_eq!(v, (0..dim).map(|d| (i * dim + d) as f32).collect::<Vec<_>>());
    }
}

#[test]
fn a_remote_fvecs_is_read_through_the_chunk_store() {
    let (records, dim) = (3000, 16);
    let server = serve(vec![("/ds/base.fvecs", fvecs(records, dim))]);
    let cache = tempfile::tempdir().unwrap();
    let transport = Transport::new(TransportConfig::default());
    let source = RemoteSource::open(&transport, &format!("{}/ds/base.fvecs", server.base), &placed(cache.path(), "base.fvecs"))
        .unwrap();
    assert!(!source.is_complete(), "nothing fetched yet beyond what the open read");
    let reader = XvecReader::<f32, RemoteSource>::from_source(source.clone()).unwrap();
    check(&reader, records, dim);
    source.precache().unwrap();
    assert!(source.is_complete() && source.mmap_base().is_some(), "complete and mapped");
    assert_eq!(names(cache.path()), ["base.fvecs", "base.fvecs.chunks"], "exactly the placed paths");
}

#[test]
fn a_remote_fvecs_with_a_published_reference_is_verified() {
    let (records, dim) = (3000, 16);
    let bytes = fvecs(records, dim);
    let mref = MerkleRef::from_content(&bytes, 64 * 1024).to_bytes();
    let server = serve(vec![("/ds/base.fvecs", bytes), ("/ds/base.fvecs.mref", mref)]);
    let cache = tempfile::tempdir().unwrap();
    let transport = Transport::new(TransportConfig::default());
    let source = RemoteSource::open(&transport, &format!("{}/ds/base.fvecs", server.base), &placed(cache.path(), "base.fvecs"))
        .unwrap();
    assert!(matches!(source.file(), veks_transport::RemoteFile::Verified(_)));
    let reader = XvecReader::<f32, RemoteSource>::from_source(source.clone()).unwrap();
    check(&reader, records, dim);
    source.precache().unwrap();
    assert!(source.mmap_base().is_some());
    assert_eq!(names(cache.path()), ["base.fvecs", "base.fvecs.mrkl"], "exactly the placed paths");
}

#[test]
fn offline_mode_refuses_a_remote_read_with_the_offline_error() {
    let server = serve(vec![("/ds/base.fvecs", fvecs(10, 4))]);
    let cache = tempfile::tempdir().unwrap();
    let transport = Transport::new(TransportConfig { offline: true, ..TransportConfig::default() });
    let err = RemoteSource::open(&transport, &format!("{}/ds/base.fvecs", server.base), &placed(cache.path(), "base.fvecs"))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotConnected, "{err}");
    assert!(err.to_string().contains("offline mode is on"), "{err}");
    assert!(server.auth.lock().unwrap().is_empty(), "no request was made");
}

/// The program's credentials, written with veks-apicore alone.
struct Fixed(&'static str);

impl Credentials for Fixed {
    fn read_token(&self, _url: &str) -> Option<String> {
        Some(self.0.to_string())
    }
}

#[test]
fn the_read_credentials_are_sent_and_a_request_of_its_own_is_not_overridden() {
    let server = serve(vec![("/ds/base.fvecs", fvecs(10, 4))]);
    let transport = Transport::with_credentials(TransportConfig::default(), Arc::new(Fixed("t0k")));
    let url = format!("{}/ds/base.fvecs", server.base);
    assert!(transport.send(Request::head(url.as_str())).unwrap().is_success());
    assert!(transport.send(Request::head(url.as_str()).bearer("own")).unwrap().is_success());
    assert!(transport.send(Request::head(url.as_str()).anonymous()).unwrap().is_success());
    let cache = tempfile::tempdir().unwrap();
    let source = RemoteSource::open(&transport, &url, &placed(cache.path(), "base.fvecs")).unwrap();
    source.precache().unwrap();
    let seen = server.auth.lock().unwrap().clone();
    assert_eq!(&seen[..3], ["Bearer t0k", "Bearer own", ""]);
    assert!(seen[3..].iter().all(|a| a == "Bearer t0k"), "every cache fetch reads with the credentials: {seen:?}");
}

/// A signer that writes the URL it was handed into the header it adds.
struct EchoSigner;

impl veks_apicore::RequestSigner for EchoSigner {
    fn sign(&self, req: &mut Request) -> Result<(), veks_apicore::TransportError> {
        let value = format!("Signed {}", req.url);
        req.headers.push(("Authorization".into(), value));
        Ok(())
    }
}

/// **A signed request carries its signature and not the read
/// credentials** (TR-5), signed over the URL the wire sees.
#[test]
fn a_signed_request_is_sent_as_signed() {
    let server = serve(vec![("/ds/base.fvecs", fvecs(10, 4))]);
    let transport = Transport::with_credentials(TransportConfig::default(), Arc::new(Fixed("t0k")));
    let url = format!("{}/ds/base.fvecs", server.base);
    let resp = transport.send(Request::head(url.as_str()).signed(Arc::new(EchoSigner))).unwrap();
    assert!(resp.is_success());
    assert_eq!(server.auth.lock().unwrap().clone(), [format!("Signed {url}")]);
}
