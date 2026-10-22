# veks-apicore

The vectordata API: what a program codes against to read vector
datasets, and what an implementation provides.

This crate holds the contracts the vectordata-rs crates speak to one
another through, and every value, descriptor and error type their
signatures name. It holds no implementation: no file format, network,
cache or dataset resolution.

- **Datasets** — `TestDataView` and `TestDataViewExt` (one profile of a
  dataset, its facets read at the element type you ask for),
  `FacetBytes`, fetch plans and reports, verification reports, and the
  dataset descriptor model (`dataset.yaml`).
- **Catalogs** — `CatalogSource` and the catalog entry types.
- **Readers** — `VectorReader` and `VvecReader` (records by ordinal),
  `ByteSource` (where a reader's bytes come from), and the streaming
  `VecSource`, `VecSink`, `VarlenSource` and `VarlenSink`.
- **Transport** — `TransportConfig`, `TlsTrust`, `Request`,
  `Credentials` and `RequestSigner`, as values and contracts.
- **Progress and errors** — `ProgressSink`, `DownloadProgress`,
  `Error`, `IoError`, `TransportError`.

The element model (`ElementType`, `Element`, the conversion rules) is
`veks-datatypes`', re-exported here.

## Who implements it

`vectordata`, `veks-io` and `veks-transport` implement these contracts
and re-export them at the paths their users know
(`vectordata::TestDataView`, `veks_io::VectorReader`, …). Most programs
depend on `vectordata` and never name this crate. Depend on it directly
when you need only the contracts, or are writing a second
implementation of one: a test double, or your own byte source. Each
contract's documentation has an example that implements it with this
crate alone.

## Cost

A contract costs nothing on a hot path: readers are generic over their
source, so a read is monomorphized for it; trait objects appear at open
time and around whole-range calls; progress is reported per batch.

License: Apache-2.0
