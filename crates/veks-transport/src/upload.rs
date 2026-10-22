// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Upload primitives: the requests an object store's writes are made
//! of, for [`Transport::send`](crate::Transport::send).
//!
//! - an object written whole: a streaming `PUT` of a file ([`put_file`])
//!   or of bytes, optionally conditional on what is there
//!   ([`put_bytes`] with a [`Precondition`]);
//! - an object written in resumable pieces, by the IETF "Resumable
//!   Uploads for HTTP" protocol (as vecd speaks it): [`create_upload`]
//!   (`POST` with `Upload-Length`; the upload's URL comes back in
//!   `Location`), [`upload_chunk`] (`PATCH` with `Upload-Offset`; the
//!   server finalizes once the contiguous prefix reaches the length),
//!   and [`abandon_upload`] (`DELETE`).
//!
//! Each returns a [`Request`] the caller completes with its own
//! authorisation (a push token, a signer; never the read credentials
//! unless it chooses them) and sends. A write is an explicit server
//! command, so offline mode does not refuse it.
//!
//! ```
//! use veks_transport::upload::{Precondition, put_bytes};
//!
//! let req = put_bytes("https://store.example/ds/pushlog.jsonl", b"{}\n".to_vec(), &Precondition::IfMatch("e1".into()))
//!     .bearer("push-token");
//! assert_eq!(req.header_value("If-Match"), Some("\"e1\""));
//! assert!(req.explicit);
//! ```

use std::path::Path;

use veks_apicore::transport::Request;

/// What an object must be for a conditional write to go ahead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Precondition {
    /// Whatever is there.
    None,
    /// Nothing is there yet (`If-None-Match: *`).
    Absent,
    /// What is there has this entity tag (`If-Match`).
    IfMatch(String),
}

impl Precondition {
    /// The precondition a push's `if_match` argument means: `None` none,
    /// `Some("")` absent, `Some(etag)` that tag.
    pub fn from_if_match(if_match: Option<&str>) -> Self {
        match if_match {
            None => Precondition::None,
            Some("") => Precondition::Absent,
            Some(etag) => Precondition::IfMatch(etag.to_string()),
        }
    }

    fn apply(&self, req: Request) -> Request {
        match self {
            Precondition::None => req,
            Precondition::Absent => req.header("If-None-Match", "*"),
            Precondition::IfMatch(etag) => req.header("If-Match", format!("\"{etag}\"")),
        }
    }
}

/// `PUT url` streaming the file at `src` off disk.
pub fn put_file(url: &str, src: &Path) -> Request {
    Request::put(url).file(src).explicit()
}

/// `PUT url` with `data`, under `precondition`.
pub fn put_bytes(url: &str, data: Vec<u8>, precondition: &Precondition) -> Request {
    precondition.apply(Request::put(url).bytes(data).explicit())
}

/// `POST url` creating a resumable upload of `total` bytes. A `201` (or
/// `200`) answers with the upload's URL in `Location`.
pub fn create_upload(url: &str, total: u64) -> Request {
    Request::post(url).header("Upload-Length", total.to_string()).explicit()
}

/// `PATCH upload_url` with the chunk at `offset`. A `2xx` means the chunk
/// is durably staged; chunks may be sent out of order and in parallel.
pub fn upload_chunk(upload_url: &str, offset: u64, body: Vec<u8>) -> Request {
    Request::patch(upload_url).header("Upload-Offset", offset.to_string()).bytes(body).explicit()
}

/// `DELETE upload_url`: abandon an upload that will not be completed.
pub fn abandon_upload(upload_url: &str) -> Request {
    Request::delete(upload_url).explicit()
}

#[cfg(test)]
mod tests {
    use super::*;
    use veks_apicore::transport::{Body, Method};

    #[test]
    fn the_requests_carry_the_protocol_headers() {
        let c = create_upload("https://h/ns/f", 42);
        assert_eq!((c.method, c.header_value("upload-length")), (Method::Post, Some("42")));
        let p = upload_chunk("https://h/-/uploads/1", 4096, vec![1, 2]);
        assert_eq!((p.method, p.header_value("Upload-Offset")), (Method::Patch, Some("4096")));
        assert_eq!(p.body, Body::Bytes(vec![1, 2]));
        assert_eq!(abandon_upload("https://h/-/uploads/1").method, Method::Delete);
        let f = put_file("https://h/ns/f", Path::new("/data/f"));
        assert_eq!(f.body, Body::File("/data/f".into()));
        assert!([c, p, f].iter().all(|r| r.explicit), "a write is an explicit server command");
    }

    #[test]
    fn a_precondition_becomes_its_header() {
        let absent = put_bytes("https://h/x", vec![], &Precondition::from_if_match(Some("")));
        assert_eq!(absent.header_value("if-none-match"), Some("*"));
        let tagged = put_bytes("https://h/x", vec![], &Precondition::from_if_match(Some("abc")));
        assert_eq!(tagged.header_value("if-match"), Some("\"abc\""));
        let plain = put_bytes("https://h/x", vec![], &Precondition::from_if_match(None));
        assert!(plain.headers.is_empty());
    }
}
