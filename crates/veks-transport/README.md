# veks-transport

Remote access and streaming storage for vector datasets: every byte the
vectordata-rs workspace moves over a network goes through this crate.

- **One client** — `Transport`: a pool of blocking HTTP clients (one
  runtime thread each, so parallel streams decrypt on many cores), TLS
  trust (added CA certificates, and self-signed acceptance only for the
  origins named), an offline gate, user agent, timeouts and redirects,
  and read credentials.
- **Requests** — `Transport::send(Request) -> Response`, with the
  request's own authorisation (a bearer, a `RequestSigner`, anonymous)
  or the read credentials by default; `fetch_text` and
  `fetch_if_modified` for small files.
- **Range reads and whole files** — `HttpTransport` (with S3
  wrong-region correction), `fetch_chunks_parallel`, `RetryPolicy`.
- **Uploads** — `upload`: streaming and conditional `PUT`, and the
  resumable `POST`/`PATCH`/`DELETE` protocol.
- **Cache files** — `VerifiedFile` (each chunk verified against a
  published `.mref` before it is trusted; state in a `.mrkl`) and
  `ChunkedFile` (no reference; a tree recorded as chunks land, in a
  `.chunks`), each mapped for zero-copy reads once complete; the
  `merkle` formats behind them.
- **Byte sources** — `RemoteSource` (cached) and `RangeSource`
  (uncached) implement veks-apicore's `ByteSource`, so veks-io's
  readers read a remote file with no vectordata.

## What it does not do

It knows no dataset, catalog or profile. It reads no environment
variable and no settings file: a `TransportConfig` value configures it.
It computes no cache path: a `CachePlacement` says where each copy and
sidecar lives, and the caller (vectordata, by its cache layout) decides
it. It does not depend on veks-io; the two meet only through
veks-apicore's `ByteSource`.

## The seams

| Contract or value | Defined in | Implemented by | Consumed here by |
|---|---|---|---|
| `TransportConfig`, `TlsTrust` | veks-apicore | vectordata (from settings and environment), any program | `Transport::new` |
| `Credentials` | veks-apicore | vectordata (its credential store), any program; `NoCredentials` | requests without their own authorisation |
| `RequestSigner` | veks-apicore | any program | `Transport::send` |
| `Request`, `TransportError` | veks-apicore | — (values) | `Transport::send` |
| `ByteSource` | veks-apicore | `RemoteSource`, `RangeSource` | veks-io's readers, vectordata's storage |

## Reading a remote file

```rust,no_run
use veks_apicore::transport::TransportConfig;
use veks_apicore::{ByteSource, VectorReader};
use veks_io::XvecReader;
use veks_transport::{CachePlacement, RemoteSource, Transport};

let transport = Transport::new(TransportConfig::default());
let placement = CachePlacement {
    root: "/tmp/cache/ds".into(),
    data: "/tmp/cache/ds/base.fvecs".into(),
    merkle_state: "/tmp/cache/ds/base.fvecs.mrkl".into(),
    chunk_map: "/tmp/cache/ds/base.fvecs.chunks".into(),
};
let source = RemoteSource::open(&transport, "https://data.example/ds/base.fvecs", &placement)?;
let base = XvecReader::<f32, RemoteSource>::from_source(source)?;
let first = base.get(0)?; // fetches the chunk it touches
```

## Design

`docs/design/srd-crate-boundaries.md` §6 (the transport layer) and §4.1
(performance and reliability of the contracts).

License: Apache-2.0.
