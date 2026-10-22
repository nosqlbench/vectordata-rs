# Upgrading Fastervector from vectordata 4.0.2 to 4.1.0

This guide covers every change between the `v4.0.2` and `v4.1.0`
tags that a program built on vectordata can see. It was put together
from the code, not from the CHANGELOG alone: each crate's public API at
4.1.0 was compared with 4.0.2 (`cargo semver-checks`), and every path
the tool reported missing was compiled against 4.1.0 to tell a moved
item, which still resolves, from a removed one.

## First: 4.1.0 is not a drop-in update

4.1.0 removes and changes public items. A manifest line like
`vectordata = { version = "4.0.2", … }` is a caret requirement
(`^4.0.2`), so a plain `cargo update` moves to 4.1.0 and the build may
break. Until the code is migrated, pin the old release:

```toml
vectordata = { version = "=4.0.2", default-features = false }
```

To upgrade, change the requirement to `4.1.0` and work through the
sections below. vectordata's feature flags are the same as in 4.0.2.

## What changed underneath

vectordata now sits on four crates of its own, and re-exports them at
the paths it always had:

| Crate | Holds |
|---|---|
| `veks-apicore` (new) | The API: `TestDataView`, `TestDataViewExt`, `CatalogSource`, `ByteSource`, `VectorReader`, `VvecReader`, the streaming `VecSource`/`VecSink`, `FacetBytes`, `ProgressSink`, the dataset descriptor model, `Error`, `IoError`, `TransportError` |
| `veks-transport` (new) | Every HTTP request: client pools, per-origin TLS trust, the offline gate, retry, range reads, uploads, SigV4, and the merkle-verified and chunked cache files |
| `veks-io` | Every file format: readers, writers, the streaming model, npy, parquet, slab |
| `veks-datatypes` | The element model: `ElementType`, `Element`, extensions, conversion rules, `DistanceFunction` |

Most paths a program uses are unchanged: `vectordata::TestDataView`,
`vectordata::dataset::*`, `vectordata::io::*`, `vectordata::merkle`,
`vectordata::metadata_schema`, `vectordata::DownloadProgress`,
`vectordata::catalog::{CatalogSource, CatalogOrigin, Severity}`. Of the
316 items the API comparison listed as missing from vectordata,
veks-io and veks-core, 248 still resolve at their old paths.

vectordata no longer depends on reqwest. `vectordata::transport` is
public and is veks-transport's client API.

## Code changes

Each of these fails to compile against 4.1.0 if your code uses it.

### `TestDataView`: the deprecated methods are gone

They were deprecated in 4.0.0 or 2.5.0. The replacements are
`TestDataViewExt`'s, so add `use vectordata::TestDataViewExt;`.

| Removed | Use |
|---|---|
| `base_vectors()`, `query_vectors()` | `base_vectors_as::<f32>()`, `query_vectors_as::<f32>()` |
| `neighbor_indices()`, `neighbor_distances()` | `neighbor_indices_as::<i32>()`, `neighbor_distances_as::<f32>()` |
| `prefiltered_*()`, `postfiltered_*()` | `prefiltered_*_as::<T>()`, `postfiltered_*_as::<T>()` |
| `metadata_results()` | `metadata_results_as::<i32>()` |
| `facet(name)` | `facet_as::<f32>(name)` |
| `prebuffer_all()`, `prebuffer_all_with_progress(..)` | `fetch(&FetchRequest::all(), &mut sink)` |
| `prefetch(..)`, `prefetch_with_progress(..)` | `fetch(&FetchRequest::facets([f]).window(w), &mut sink)` |

`verify`, `verify_report` and `normalization` moved from
`TestDataViewExt` to `TestDataView`. Calling them is unchanged.

### `TypedReader` and its API are removed

They were deprecated since 4.0.0. `vectordata::typed_access` keeps the
element model and `ElementTypeExt`.

| Removed | Use |
|---|---|
| `open_facet_typed::<T>(&*view, name)`, `view.open_facet_typed::<T>(name)` | `view.facet_as::<T>(name)` |
| `TypedReader::<T>::open(path)`, `open_auto(path_or_url, ty)` | `vectordata::io::open_vec_as::<T>(path_or_url)` |
| `get_value(i)` | `get(i)?[0]`, or `as_slice()?[i]` on a resident column |
| `get_record(i)` | `get(i)?` |
| `get_native(i)`, `get_native_slice(i)` | `as_slice()?[i]`, `get_slice(i)` |
| `is_native()`, `native_type()` | `conversion() == ElementConversion::Native`, `native_element_type()` |
| `dispatch_typed!` | `view.open_facet_native(name)` or `io::open_vec_native(path)` (an `AnyVectorReader` to match on), or `dispatch_element!` |
| `TypedAccessError` | `IoError::ElementMismatch`, `IoError::ValueOverflow` |
| `T: TypedElement` | `T: Element` |

### Errors

- `vectordata::Error::Http(reqwest::Error)` is
  `Error::Transport(TransportError)`. A `TransportError` has the URL,
  the HTTP status when the server answered, a `kind` saying whether a
  retry could succeed, and the message. No reqwest type is in the API.
- `vectordata::io::IoError::Http` is `IoError::Transport(TransportError)`,
  the same value. `From` converts it into `Error`.
- `TransportErrorKind` has two new variants: `Unreachable` (no answer:
  refused, timed out, or an untrusted certificate) and `Offline`. An
  exhaustive `match` needs arms for them.

```rust
match err {
    vectordata::Error::Transport(t) if t.kind == TransportErrorKind::Offline => { /* offline mode */ }
    vectordata::Error::Transport(t) => eprintln!("{} ({:?})", t.message, t.status),
    other => return Err(other.into()),
}
```

### Facet storage and progress

- `TestDataView::open_facet_storage` returns `Box<dyn FacetBytes>`
  (`vectordata::FacetBytes`). Where you held a `&FacetStorage`, a
  `&dyn FacetBytes` usually serves: `total_size`, `is_complete`,
  `local_files`, `cache_stats` and `content_id` are on the contract. For
  anything that is only vectordata's, downcast with
  `as_any().downcast_ref::<vectordata::view::FacetStorage>()`.
- `content_id_with_progress` and `verify_cached` take a
  `&mut dyn ProgressSink`. A closure `|label, done, total| …` is one.
- `prebuffer_with_progress`, `prebuffer_range_with_progress` and the
  other prebuffer forms take `&mut dyn FnMut(&DownloadProgress)`.

### Dataset descriptors and catalogs

- `DatasetConfig::load_from` (a URL) and
  `DatasetConfig::reconcile_shard_declarations` belong to the
  `vectordata::dataset::DatasetConfigIo` trait. Add
  `use vectordata::dataset::DatasetConfigIo;` to call them. Loading and
  saving a local `dataset.yaml` are unchanged.
- `CatalogEntry::apply_overlay` returns the contradictions it found
  instead of logging them. vectordata still warns about them where it
  applies overlays itself.
- `vectordata::catalog::source::CatalogSource` and
  `vectordata::catalog::source::CatalogOrigin` no longer resolve; the
  types moved to veks-apicore. Use `vectordata::catalog::CatalogSource`
  and `vectordata::catalog::CatalogOrigin`, which did and still do.

### Readers and byte sources

- `ByteSource` has no constructors and no `Progress` type. A source is
  opened with veks-io's `OpenSource` (`open`, `open_path`), which the
  readers' `open` functions already call.
- veks-io's `MmapReader<T>`, `ScalarMmapReader<T>` and `MmapError` are
  deprecated aliases of `XvecReader<T>`, `ScalarReader<T>` and
  `IoError`. `open(&Path)` is `open_path(&Path)`, `get(i)` returns
  `Result`, and the bounds-checked `get_slice` is
  `VectorReader::get_slice`. The inherent `get_slice` is the
  unchecked form for inner loops.
- `veks_io::scalar::convert` (`convert_scalars`, `ConvertError`) is
  `veks_io::convert`.

### Streaming sources return `Result`

If you read files front to back with `VecSource` or `VarlenSource`, or
implement either:

| Was | Now |
|---|---|
| `fn next_record(&mut self) -> Option<Vec<u8>>` | `-> Result<Option<Vec<u8>>, IoError>` |
| `fn read_record_into(&mut self, buf: &mut Vec<u8>) -> bool` | `-> Result<bool, IoError>` |
| `VarlenSource::next_record -> Option<VarlenRecord>` | `-> Result<Option<VarlenRecord>, IoError>` |

`Ok(None)` is the end of the stream, and an `Err` is where it stops.
Use `while let Some(rec) = source.next_record()? { … }`. A source you
implement yourself wraps its records in `Ok`.

`SourceMeta` has a new `element_type: Option<ElementType>` field, so
a struct literal needs it. `VecSource::element_type()` defaults to
`None`.

### Formats

- `VecFormat` lives in veks-io (`veks_io::VecFormat`); veks-core's copy
  is gone. `from_extension("fvvecs")` now returns `Some(Fvvec)`, not
  `None`, and `is_xvec()` is true for the variable-length formats too.
  Use `is_uniform_xvec()` when you mean one dimension throughout.
  `open_format` and `create_format` refuse a vvec format.
- veks-core's format modules moved to veks-io, with no shim in
  veks-core:

  | Was | Now |
  |---|---|
  | `veks_core::formats::VecFormat` | `veks_io::VecFormat` |
  | `veks_core::formats::reader::{VecSource, SourceMeta}`, `formats::writer::VecSink` | `veks_io::{VecSource, SourceMeta, VecSink}` |
  | `veks_core::formats::writer::{open_sink, open_sharded_sink, SinkConfig}` | `veks_io::sink::*` |
  | `veks_core::formats::writer::sharded` | `veks_io::sharded` |
  | `veks_core::formats::writer::mmap_xvec::{MmapXvecWriter, SharedMmapWriter}` | `veks_io::xvec::positional::{PositionalXvecWriter, SharedPositionalWriter}` |
  | `veks_core::formats::reader::{xvec, npy, parquet, slab}`, `formats::writer::{xvec, slab}` | `veks_io::xvec::{reader, writer}`, `veks_io::npy`, `veks_io::parquet`, `veks_io::slab::{reader, writer}` (features `npy`, `parquet`, `slab`) |
  | `veks_core::formats::convert` | `veks_io::convert` |
  | `veks_core::formats::portable_io` | `veks_io::file` |
  | `veks_core::formats::element_type_of_width` | none: a type is never guessed from its width; read it from the source |
  | `veks_core::formats::reader::hdf5` | none: HDF5 is not supported |

- `SinkConfig::source_format` is replaced by
  `element_type: Option<ElementType>`, and veks-io's
  `SlabWriter::open` takes an `element_type` argument.
- `MetadataSchema::to_json_bytes` and `PredicateSchema::to_json_bytes`
  return `Result<Vec<u8>, String>`.

## Behaviour changes

These compile without changes but behave differently at run time.

**Writing**

- **A symlink at a write target is an error**, from every veks-io
  writer and from `veks_io::atomic`. The error names the link and its
  target. Before, depending on the writer, the link was removed or the
  write went through it to the file it named. Remove the link yourself
  if replacing it is what you want.
- **A byte-level writer's first I/O error is returned from `finish`.**
  A full disk used to be reported as success.

**Reading**

- **A reader never reinterprets bytes as another type.** A file opens
  at its own element type, a lossless conversion widens, a same-width
  sign change is checked value by value, and anything else is
  `IoError::ElementMismatch`. A uniform reader refuses a
  variable-length file and names `IndexedVvecReader`.
- **An empty file reads as zero records** in veks-io's readers, as it
  did in vectordata's.
- **Streaming sources stop with an error.** A corrupt file in an npy
  or parquet directory, a record whose header disagrees, a truncated
  record or an unreadable slab page is now an error at that point.
  Before, the stream ended early with no error.
- **A multi-file parquet directory reads every file.** It used to read
  only the first. Parquet also refuses null rows, rows of the wrong
  length and a later file with a different schema.
- **Scalar columns of `i8`, `u16` and `u32`** read with their own
  signedness.
- **A slab of vectors states its element type** in a `kind: "vector"`
  schema sidecar. A slab without one reads as 1-byte elements, with
  `element_type()` `None` and a warning. Converting such a slab to a
  typed format is refused instead of being read as `f32`. Slabs written
  by 4.0.2 are undeclared.

**Network**

- **Offline mode** refuses `whoami`, `ping` and namespace completion
  before any request is made, and now also gates the explorer's
  catalog read, the update check and the pipeline downloaders.
  `login`, `token issue`/`revoke`, `push` and `backup` are not
  affected.
- **Per-origin TLS trust** (`trust_self_signed` in settings) applies to
  every request, including `fetch_if_modified`, the downloaders and the
  update check.
- **The pipeline downloaders** send `vectordata/<version>` as the user
  agent, do not retry a permanent 4xx (other than 408, 425 and 429) or
  an offline refusal, and back off between retries (from 1 s, doubling
  to 30 s). `tries = 0` means one attempt.
- **A request to a server that has gone away** reports `Unreachable`,
  so a local copy can stand in.

## New in 4.1.0

These come from Fastervector's requests and from the boundary work.

- **A directory of loose files is a dataset.** A local directory with
  vector files and no `dataset.yaml` opens as the descriptor its files
  imply: canonical facet names first, then role tokens (`base`,
  `query`, `gt`, …), then shapes. Every ambiguity is refused, naming
  the candidates. `vectordata datasets init <dir>` writes the
  descriptor, and `dataset::infer::infer_dir` is the library form.
- **Attribute overlays.** `<config_dir>/overlays.yaml` (edited with
  `vectordata config overlay set | unset | show`) and program overlays
  (`CatalogSources::with_overlay`, `TestDataGroup::with_overlays`)
  state or correct a dataset's attributes. Your own keys can be
  namespaced (`fvec.ncentroids`) and read back with `attribute(key)`
  and `attributes_in("fvec")`. `attribute_source(key)` says where a
  value came from.
- **Content identity.** `FacetStorage::content_id()` is the root of a
  merkle tree over a facet's bytes, so a run can fingerprint its
  inputs exactly. Equal bytes have equal roots.
- **Verification.** `view.verify(&mut progress)` checks that a
  profile's facets belong together: they open, dimensions agree, the
  ground truth has a row per query and names real base vectors, the
  distances match, and declared counts hold. A failure is
  `Error::Verification`.
- **Normalization.** `view.normalization()` says whether the base is
  unit length, from the declaration checked against a sample, or from
  the sample alone.
- **Small remote files.** `vectordata::fetch::fetch_text(url)` and
  `fetch_if_modified(url, path)` use the same client, credentials,
  TLS trust and offline rule as dataset fetches.
- **Quiet meters.** `LogMeter::quiet_when_resident` and
  `TextMeter::quiet_when_resident` print nothing when there is nothing
  to fetch.
- **Embedded CLI completion.** `shell::completion_node()` grafts the
  vectordata CLI's completion under a host's own subcommand, and
  `shell::complete(words)` answers for a host not built on
  veks-completion.
- **Writers.** `XvecWriter<T>`, `ScalarWriter<T>` and `VvecWriter<T>`
  write typed rows at a file's own type. `XvecWriter::append`
  continues an existing file of the same dimension and refuses a cut
  one. `veks_io::xvec::record` encodes, decodes and `pread`s single
  records in your own buffer.
- **Remote reads without vectordata.** veks-transport's `RemoteSource`
  is a `ByteSource`, so veks-io's `XvecReader` reads a remote file
  directly. `vectordata::transport::sigv4::SigV4Signer` signs requests
  to S3 and S3-compatible stores.

## Corrections to earlier statements

- Request B3 assumed that cached chunks were already hashed. They were
  not: before 4.1.0 a facet fetched without a `.mref` was trusted byte
  for byte, and its chunks were recorded as present but never hashed. A facet fetched without a `.mref`
  now records each chunk's leaf as it lands. A `.chunks` sidecar
  written by 4.0.2 is still read; its leaves are hashed once, when an
  identity is first asked for.
- Integer queries are valid queries, so a `_query_` file in an
  integer format is read as the query facet when inferring a loose
  directory.
- `TestDataGroup::load` reads no configuration of its own: overlays
  come from `overlays.yaml` through the catalog, or from
  `TestDataGroup::with_overlays`.
- A normalization measurement is recorded only for remote bases, in
  the dataset's cache directory. A local base is measured each time it
  is asked.

## Checklist

- [ ] Pin `=4.0.2` until the migration is done, then require `4.1.0`.
- [ ] Replace removed `TestDataView` methods with the `*_as` forms, and
      import `TestDataViewExt`.
- [ ] Replace `TypedReader`, `open_facet_typed` and `dispatch_typed!`.
- [ ] Match `Error::Transport` / `IoError::Transport` instead of
      `Http`, and add arms for `Unreachable` and `Offline`.
- [ ] Update `open_facet_storage` callers to downcast from `FacetBytes`.
- [ ] Import `DatasetConfigIo` where you call `load_from`.
- [ ] Use `vectordata::catalog::{CatalogSource, CatalogOrigin}`, not
      `catalog::source::…`.
- [ ] Handle `Result` from `next_record` and `read_record_into`.
- [ ] Move veks-core format paths to veks-io.
- [ ] Check write targets that are symlinks.
- [ ] Re-test offline runs, and any reads of `i8`, `u16` or `u32`
      columns and of slabs.
