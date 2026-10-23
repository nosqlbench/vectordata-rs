# Changelog

## Unreleased

### Added

- **`config catalog disable <name>` and `config catalog enable <name>`**
  (`vectordata` and `veks datasets`) set the `disabled_catalogs` key of
  `settings.yaml` without the explorer, editing that one line and
  keeping every comment. A name no catalog has is refused naming the
  ones there are; disabling a disabled catalog says so and succeeds.
  The names complete. `config catalog list` marks a disabled catalog
  `(disabled)` under the number `--at` takes, which does not change.
- **`--catalog <name>`** names a configured or program catalog, as
  `--at <name>` does, and reads it beside the others; a URL or an
  existing path is a location, as before.
- **`CatalogSources::including_disabled`**: the set with its disabled
  catalogs assembled, for a program that turns catalogs on and off
  itself, as the explorer's picker does.
- **jvector `_include` is followed.** A `knn_entries.yaml` (or any
  jvector-shaped catalog) whose `_include: { url: … }` names another
  catalog lists that catalog's datasets, read as jvector reads them:
  the including file's `_defaults` apply, facets resolve against the
  included catalog's directory unless a `base_url` is stated, and the
  including file's own entry of a name wins. It was skipped, so a thin
  wrapper such as jvector's public catalog listed nothing. A remote
  include is kept for offline use like any catalog; one that cannot be
  read is a warning, not a failure.
- **A jvector catalog's `cache_dir` is the dataset's cache.** The
  entry's `cache_dir`, `_defaults.cache_dir` or `$DATASET_CACHE_DIR`
  was only where an existing copy was looked for; a fetch went to the
  configured cache. What is fetched now lands under
  `<cache_dir>/<dataset>/`, managed as any cache is (sidecars,
  `origin.json`, `measured.yaml`, the free-space check), in place of the
  configured cache; a run's `--cache-dir` or `settings::set_cache_dir`
  still outranks it. `CatalogEntry::cache_dir` and
  `EntryBuilder::cache_dir` carry it, and `TestDataView::cache_dir`
  (defaulted to `None`) says which cache a view's dataset is in.
- **The cache commands act on the cache active for them.**
  `veks datasets cache-status` takes `--cache-dir`; without it, a
  dataset's status is read from its own cache (the one its catalog
  names, else the configured one), `--tree` shows that cache's tree,
  and `--all` covers each cache a catalog names as well as the
  configured one. The explorer's picker finds, locates and purges a
  dataset in its own cache too. `vectordata::datasets::cache::
  run_cache_status_in` and `run_cache_status_all_in` take the cache
  and `tree`; `run_cache_status` and `run_cache_status_all` are
  deprecated wrappers over them.

### Breaking

- veks: `DatasetsCommand::CacheStatus` gains `cache_dir`, so a pattern
  or construction of it outside veks names the field or uses `..`.
- veks: `ConfigCatalogSubcommand` gains `Disable` and `Enable`, mirroring
  `vectordata config catalog`, and is `#[non_exhaustive]` from now on,
  so a match on it outside veks needs a wildcard arm.

### Fixes

- **`--at <name>` and `--catalog <name>` naming no catalog** were read
  as a location, so the run failed naming a file nobody asked for under
  a made-up catalog index (`no catalog file found at <name>` …
  `catalog '0', named for this run, could not be loaded`), as when the
  catalog was removed from `catalogs.yaml` while a program ran. A value
  that reads as a name (no path separator, no `.`, no `~`) and is no URL
  or existing path is now refused, listing the catalogs there are.
- **A resident facet's precache line, again.** After 4.2.1 dropped
  the zero total, the log meter's progress line for a resident facet
  read word for word as its closing line, so one facet was listed twice;
  and the size was shown only for a degraded plan, never for a whole
  facet. A resident facet is one line, with the bytes its request
  covers (`[1/3] base_vectors: already resident, 1.8 GiB`).
- **A precache with nothing to fetch printed `total 100% (0 B/0 B)`**
  on every facet's line, a total of a quantity the run does not have.
  The total is left off when nothing is to be fetched, and a resident
  facet's line says how much was found cached
  (`[1/3] base_vectors: already resident, 1.8 GiB`), in the terminal
  and log meters alike.
- **A disabled catalog is not read.** `disabled_catalogs` was honoured
  by the explorer's picker alone: every `datasets`, `cache` and
  `explore --dataset` command, every `shell::Host` program and every
  `CatalogSources::configure*` set read a disabled catalog like any
  other, listing its datasets, failing on its errors, and making a
  dataset it shares with an enabled catalog ambiguous. A catalog the
  setting names is now held but not assembled
  (`CatalogSources::iter` leaves it out), a program's catalog of that
  name included; `--at <name>` and `--catalog <name>` still read it.
  The setting is read from the `settings.yaml` beside the
  `catalogs.yaml` (`<dir>/settings.yaml` for `--configdir <dir>`).

## Unreleased

One implementation of every file format. veks-io's readers are now
the readers: vectordata's `XvecReader`, `IndexedVvecReader` and
`ScalarReader` are veks-io's, over vectordata's storage, and its own
copies are gone. The faster technique of each won, measured: random
`get` is 2.7× vectordata's before and the vvec scan 2.5×, and
veks-io's streaming readers are 2–2.5× faster. The design and the
measurements are `docs/design/srd-loose-files-overlays-integrity.md`
§6.

### Highlights

- **`veks-apicore`: the API, as a crate of its own.** The contracts the
  crates speak through — `TestDataView` and `TestDataViewExt`,
  `CatalogSource`, `ByteSource`, `VectorReader` and `VvecReader`, the
  streaming `VecSource`/`VecSink`, `FacetBytes`, `RecordContainer`,
  `FetchProgress`, `ProgressSink` — and every type they name (the
  dataset descriptor model, `Location`, fetch plans and reports,
  verification reports, `Error`, `IoError`) are defined there, with no
  implementation. vectordata and veks-io implement it and re-export
  it at every path they had, so `vectordata::TestDataView`,
  `vectordata::dataset::DatasetConfig` and `veks_io::VectorReader` are
  unchanged. A program that needs only the contracts, or a second
  implementation (a test double, its own byte source), depends on
  `veks-apicore` alone; each contract has an example that implements
  it with nothing else. Reads cost what they did: readers stay generic
  over their source, and what a hot path calls across the boundary
  inlines (`docs/design/srd-crate-boundaries.md` §4).
- **`veks-transport`: every byte that crosses a network, outside
  vectordata.** The client pools, per-origin TLS trust, the offline
  gate, retry (`RetryPolicy`), range reads and whole-file transfers,
  upload requests, and the cache files remote bytes land in
  (`VerifiedFile`, merkle-checked; `ChunkedFile`, chunk-recorded) are
  a crate of their own. It depends on veks-apicore alone and reads no
  environment variable or settings file. Its configuration
  (`TransportConfig`), read credentials (`Credentials`) and request
  signing (`RequestSigner`) are values and traits in veks-apicore, and
  vectordata builds the one transport from its settings
  (`vectordata::transport::shared()`). `RemoteSource` is a
  `ByteSource`, so a program that depends on veks-io and
  veks-transport reads a remote `.fvecs` through `XvecReader` with no
  vectordata. vectordata still places every cache file, and the cache
  layout is unchanged (`docs/design/srd-crate-boundaries.md` §6).
- **Only veks-transport talks HTTP.** vectordata, vecd, veks and
  veks-pipeline no longer depend on reqwest (vectordata and vecd keep
  it for their tests): the pipeline's Hugging Face, Semantic Scholar
  and bulk downloaders, the update check and vecd's readiness probe
  send through the transport, with its TLS trust, offline gate and
  user agent, and each request carries only its own authorisation
  (`HF_TOKEN`, the S2 API key, a SigV4 signature, or none), never a
  vectordata read token. veks-pipeline no longer depends on tokio. A
  workspace test holds the crate graph to the SRD's layers
  (`docs/design/srd-crate-boundaries.md` §7).
- **SigV4 in the transport: `vectordata::transport::sigv4`.**
  `SigV4Signer` is a `RequestSigner` for S3 and S3-compatible stores
  (and any SigV4 service), over `AwsCredentials`, whose `Debug` shows
  no secret; checked against AWS's published examples. The COS
  downloader signs with it, as before. `Response` is `io::Read`, so a
  body streams in the caller's increments, and `status_error()` gives
  the `TransportError` of an answer the caller found failed.
- **The format layer is veks-io, and only veks-io.** veks-core's
  streaming readers and writers (xvec, npy, parquet, slab), the sinks
  and the sharding sink, the positional writer, element conversion,
  `portable_io` (now `veks_io::file`) and the pipeline's atomic writer
  (`veks_io::atomic`) moved into it. No other crate reads a dimension
  header or matches an extension itself, and a workspace test keeps
  it that way.
  - `veks_io::xvec::record` encodes, decodes, walks and `pread`s single
    records in a caller's buffer.
  - `XvecWriter::append` continues an existing file.
  - `reader::probe_source` describes a partial, sparse or truncated
    file instead of refusing it.
  - `XvecWriter::write_raw` and `writer::encode_records` write packed
    batches.
  - The streaming xvec reader reads a directory or a `__NNNN` series
    as one stream, and the streaming writer releases written pages
    every 64 MiB.
- **One `VecFormat`.** veks-io's enum has the variable-length formats
  (`Fvvec` … `U64vvec`), `is_uniform_xvec`, `is_vvec`,
  `detect_from_path`, `detect_from_directory`, `record_bytes` and
  `expected_file_size`. veks-core's enum and its streaming traits,
  with the string round trip between the two models, are gone.
- **veks-io's readers take any byte source.** `XvecReader<T, S>`,
  `IndexedVvecReader<T, S>`, `ScalarReader<T, S>` and `StreamReclaim`
  are generic over a `ByteSource`, defaulting to `MappedFile`, a local
  file. `open_vec`, `open_vec_as`, `open_vec_native`, `open_vvec`,
  `ConvertingReader` and `AnyVectorReader` are veks-io's too. A program
  reading its own local files needs only veks-io.
- **`vectordata::io` is veks-io over vectordata's storage.** Every path
  in it stays: `XvecReader<T>` is `veks_io::XvecReader<T,
  StorageSource>`, and so on. vectordata depends on veks-io.
- **Typed writers: `XvecWriter<T>`, `ScalarWriter<T>`,
  `VvecWriter<T>`** in veks-io, re-exported from `vectordata::io`: rows
  of `T` into the layout the extension names, refusing another type
  or a row of the wrong length, returning every I/O error.
  `VvecWriter` writes the `IDXFOR__` index as it finishes.
- **`view.normalization()`: whether the base is unit length, and on
  whose word** — the declared `is_normalized` checked against a
  1000-vector sample, or the sample when nothing declares it. A remote
  base's sample is recorded in its cache directory by content identity
  and not re-taken; a contradicted declaration is marked, an
  overlay's warned about, and `verify()` reports it as V8.
- **Every facet has a content identity: `FacetStorage::content_id()`**,
  the root of a merkle tree over its bytes — the publisher's `.mref`
  root, the root recorded as the chunks of a file published without
  one were fetched, or a local file's, computed once per process.
  Equal bytes have equal roots, so a run can fingerprint its inputs
  exactly. A facet fetched without a `.mref` now records that tree in
  its `.chunks` sidecar as it lands (the `.mrkl` format; one leaf and
  the bitset per chunk, never the whole tree); a sidecar of the
  earlier format is read as before, and its leaves are hashed once
  when an identity is asked for. `verify_cached` and the opt-in V9
  check find cached chunks changed since they landed.
- **Attribute overlays: state or correct a dataset's attributes on a
  host.** `<config_dir>/overlays.yaml` (edited with `vectordata config
  overlay set | unset | show`, comments kept) and program overlays
  (`CatalogSources::with_overlay`) are merged over a dataset's own
  attributes, so the catalog entry, `view.distance()`, `describe`,
  `list` and `explore` all read one answer — a `knn_entries.yaml`
  dataset gets the metric its catalog never states. Consumers'
  namespaced keys (`fvec.ncentroids`) read back through
  `attribute(key)` and `attributes_in(namespace)`.
  `attribute_source(key)` and `describe` say where each value came
  from. A count an opened facet contradicts, or a metric the dataset
  declares otherwise, is a warning once per process; the overlay
  stays in force, and `verify()` reports the count as V7.
- **A directory of loose files is a dataset.** A local directory with
  three vector files and no `dataset.yaml` opens as the descriptor its
  files imply — canonical facet names, else one role token per name
  (`base`, `query`, `gt`, …), else the files' shapes — verified before
  use, every ambiguity refused naming the candidates. `datasets
  describe <dir>` says it was inferred and what each file was read as.
  `vectordata datasets init <dir>` writes that `dataset.yaml`;
  `dataset::infer::infer_dir` is the library form. A metric token in
  the names (`ip`, `angular`, `l2`, …) declares the distance function
  when they agree. The rules are in sysref 01 §1.5.
- **`view.verify(&mut progress)` proves a profile's facets belong
  together** (`TestDataViewExt`; `DatasetSelection::verify` for every
  selected profile): every facet opens, the queries have the base's
  dimension, a ground truth has one row per query, every index names a
  base vector (or is a filtered truth's `-1`), the distances match the
  truth, a declared `base_count` is the base's. A failure is
  `Error::Verification`, naming the check, the facet, the file and
  what it found. The index scan reports progress. `veks check` makes
  its ground-truth checks through it, and so now also reports queries
  of another dimension and a wrong `base_count`.
- **`vectordata::fetch::fetch_text(url)` and `fetch_if_modified(url,
  path)`**: the small files a program reads beside a dataset — a
  catalog, a list — fetched with the client, auth and offline rule the
  dataset fetches use, so a program needs no HTTP client of its own.
- **`LogMeter::quiet_when_resident` and `TextMeter::quiet_when_resident`**:
  nothing for a facet whose plan has nothing to fetch, nor for a run
  that fetches nothing; a remote facet still reports what it costs.
- **Completion for an embedded CLI**: `shell::completion_node()` grafts
  under a host's own `vectordata` subcommand, and `shell::complete(words)`
  answers for a host not built on veks-completion. `completions` under
  an embedding name says how, instead of printing a script bound to the
  `vectordata` executable.
- **A program can run the `vectordata` command line as its own.**
  `shell::run(invoked, argv, &host)` and `shell::run_with_catalogs(invoked,
  argv, catalogs)` parse and dispatch as `bin_main_with_catalogs` does,
  and return the exit status (0, 1 for a failed command, 2 for a usage
  error) instead of ending the process, so a program wrapping the argv
  carries on after the command. A `shell::Host` carries the program's
  catalogs and its own output: `after_describe` and `after_list` run
  after `datasets describe` and `datasets list` print, with what they
  described or listed, and `before_precache` expands a spec of the
  program's own (`group:<name>`) into the specs it stands for.
  `Host::owns_completion()` is for a program whose command line this is:
  `completions` prints scripts bound to the program's name and the
  completion protocol is answered for it. veks-completion's
  `handle_complete_args` answers that protocol over arguments a caller
  passes in.
- **`datasets precache a b:10m c:profile=*` takes several specs** and
  fetches them as one run: every spec resolved before anything moves,
  then one plan, one progress display and one report, each facet named
  `<dataset>:<profile>/<facet>`. `datasets::precache::run_specs` is the
  library form; `PrecacheRequest` is unchanged. Before, every spec
  after the first was silently ignored and the command exited 0.
- **`VecSource::read_record_into(&mut Vec<u8>)`** streams a file into
  one buffer instead of a new `Vec` per record. The streaming readers
  read from a mapping and release the pages behind the cursor.

### Breaking

- **`TestDataView` is defined without its deprecated methods.** Each
  has had its replacement since 4.0.0 or 2.5.0:

  | Removed | Use |
  |---|---|
  | `base_vectors()`, `query_vectors()` | `base_vectors_as::<f32>()`, `query_vectors_as::<f32>()` |
  | `neighbor_indices()`, `neighbor_distances()` | `neighbor_indices_as::<i32>()`, `neighbor_distances_as::<f32>()` |
  | `prefiltered_*()`, `postfiltered_*()` | `prefiltered_*_as::<T>()`, `postfiltered_*_as::<T>()` |
  | `metadata_results()` | `metadata_results_as::<i32>()` |
  | `facet(name)` | `facet_as::<f32>(name)` |
  | `prebuffer_all()`, `prebuffer_all_with_progress(..)` | `fetch(&FetchRequest::all(), &mut sink)` |
  | `prefetch(..)`, `prefetch_with_progress(..)` | `fetch(&FetchRequest::facets([f]).window(w), &mut sink)` |

  The `*_as` readers are `TestDataViewExt`'s: `use vectordata::TestDataViewExt`.
- `vectordata::Error::Http(reqwest::Error)` is
  `Error::Transport(TransportError)`: the URL, the HTTP status when the
  server answered, and whether a retry could succeed. No reqwest type
  is in the API.
- `TestDataView::open_facet_storage` returns `Box<dyn FacetBytes>`, the
  API's contract for a facet's bytes; vectordata's `FacetStorage`
  behind it is reached with `as_any().downcast_ref::<FacetStorage>()`.
  `content_id_with_progress` and `verify_cached` take a `&mut dyn
  ProgressSink` (`|label, done, total|`); `prebuffer_with_progress`,
  `prebuffer_range_with_progress` and the other prebuffer forms take
  `&mut dyn FnMut(&DownloadProgress)` through it.
- `verify`, `verify_report` and `normalization` are `TestDataView`'s
  (they were `TestDataViewExt`'s); a view implements `verify_report`
  and `normalization`. `vectordata::verify::VerifyProgress` is the
  API's `ProgressSink`.
- `ByteSource` has no constructors and no `Progress` type: opening a
  source is veks-io's `OpenSource` (`open`, `open_path`), which the
  readers' `open` functions take; `prebuffer_with_progress` reports a
  `DownloadProgress`.
- `DatasetConfig::load_from` (a URL) and
  `DatasetConfig::reconcile_shard_declarations` (the files on disk) are
  `vectordata::dataset::DatasetConfigIo`'s: `use` it to call them.
  Loading and saving a local `dataset.yaml` are unchanged.
- `CatalogEntry::apply_overlay` returns what it found contradicted
  instead of warning; vectordata warns as before.
- `vectordata::io::IoError::Http(reqwest::Error)` is
  `IoError::Transport(TransportError)`, the same value `Error::Transport`
  carries (`From` is implemented). The reader error is veks-apicore's,
  which has no HTTP client. `TransportErrorKind` gains `Unreachable`
  (no answer: refused, timed out or an untrusted certificate, so a
  local copy stands in) and `Offline`, so an exhaustive `match` needs
  the two arms.
- **A symlink at a write target is an error**, from every veks-io
  writer and from `veks_io::atomic`'s `AtomicWriter` and `safe_*`
  helpers (veks-pipeline's `atomic_write` is a re-export of it). The
  error names the link and what it points to. veks-core's writer
  removed the link silently, the pipeline's guard removed it with a
  warning, and veks-io's writers wrote through it to the file it
  named. A link someone made on
  purpose is not a writer's to remove. `veks prepare import`
  removes the Identity links it made itself where a new plan
  materializes the artifact, and it says so.
- **HDF5 is not supported.** The stub reader and `VecFormat::Hdf5` are
  removed. `veks prepare import` of an `.h5`/`.hdf5` source (or
  `file.h5#dataset`) stops before planning and says to convert it to
  fvecs first.
- **`TypedReader` and the rest of its API are removed** (deprecated
  since 4.0.0): `TypedReader`, `TypedElement`, `TypedAccessError`,
  `dispatch_typed!` and `open_facet_typed` (the function and the
  `TestDataView` method). `vectordata::typed_access` keeps the element
  model and `ElementTypeExt`.

  | Removed | Use |
  |---|---|
  | `open_facet_typed::<T>(&*view, name)`, `view.open_facet_typed::<T>(name)` | `view.facet_as::<T>(name)` |
  | `TypedReader::<T>::open(path)`, `open_auto(path_or_url, ty)` | `io::open_vec_as::<T>(path_or_url)` |
  | `get_value(i)` | `get(i)?[0]`, or `as_slice()?[i]` on a resident column |
  | `get_record(i)` | `get(i)?` |
  | `get_native(i)`, `get_native_slice(i)` | `as_slice()?[i]`, `get_slice(i)` |
  | `is_native()`, `native_type()` | `conversion() == Native`, `native_element_type()` |
  | `dispatch_typed!` | `view.open_facet_native(name)` or `io::open_vec_native(path)` (an `AnyVectorReader` to match on), or `dispatch_element!` |
  | `TypedAccessError` | `IoError::ElementMismatch`, `IoError::ValueOverflow` |
  | `T: TypedElement` | `T: Element` |
- **Format modules moved to veks-io**, with no shim in veks-core:

  | Was | Now |
  |---|---|
  | `veks_core::formats::VecFormat` | `veks_io::VecFormat` |
  | `veks_core::formats::reader::{VecSource, SourceMeta}`, `formats::writer::VecSink` | `veks_io::{VecSource, SourceMeta, VecSink}` (veks-apicore's) |
  | `veks_core::formats::writer::{open_sink, open_sharded_sink, SinkConfig}`, `formats::SLAB_RECORD_FRAMING` | `veks_io::sink::*` |
  | `veks_core::formats::writer::sharded` | `veks_io::sharded` |
  | `veks_core::formats::writer::mmap_xvec::{MmapXvecWriter, SharedMmapWriter}` | `veks_io::xvec::positional::{PositionalXvecWriter, SharedPositionalWriter}` |
  | `veks_core::formats::reader::{xvec, npy, parquet, slab}`, `formats::writer::{xvec, slab}` | `veks_io::xvec::{reader, writer}`, `veks_io::npy`, `veks_io::parquet`, `veks_io::slab::{reader, writer}` (features `npy`, `parquet`, `slab`) |
  | `veks_core::formats::convert`, `veks_io::scalar::convert` | `veks_io::convert` |
  | `veks_core::formats::portable_io` | `veks_io::file` |
  | `veks_pipeline::pipeline::atomic_write` | `veks_io::atomic` (re-exported at the old path) |

  These paths keep working as re-exports:
  - `vectordata::metadata_schema` (now veks-apicore's).
  - `vectordata::merkle` (veks-transport's).
  - `vectordata::DownloadProgress` and `vectordata::io::*`.
  - `vectordata::dataset::shards::{discover_shards, observe_series}`
    (`veks_io::series`).
  `vectordata::transport` is now public and re-exports
  veks-transport's client API.
- **`SourceMeta` has an `element_type` field**
  (`Option<ElementType>`), and `VecSource::element_type()` defaults to
  `None`. A struct literal needs the field. The xvec, scalar, npy and
  parquet readers fill it in.
- `VecFormat::from_extension("fvvecs")` is `Some(Fvvec)`, not `None`.
  `is_xvec()` is true for the vvec formats too, so use
  `is_uniform_xvec()` for one dimension throughout. `open_format` and
  `create_format` refuse a vvec format.
- `MetadataSchema::to_json_bytes` and `PredicateSchema::to_json_bytes`
  return `Result<Vec<u8>, String>`.
- veks-io: `MmapReader<T>` and `ScalarMmapReader<T>` are deprecated
  aliases of `XvecReader<T>` and `ScalarReader<T>`, and `MmapError` of
  `IoError`. `open(&Path)` is `open_path(&Path)`; `get(i)` returns
  `Result`; the bounds-checked `get_slice` is
  `VectorReader::get_slice`, and the inherent one is the unchecked
  inner-loop form.
- veks-io's readers read an empty file as zero records, as
  vectordata's did, where they refused it.
- A uniform reader refuses a variable-length extension (`.fvvecs`)
  and names `IndexedVvecReader`; vectordata's opened one whose records
  happened to be one length.
- The refusal of a convertible type now reads "open it through a
  dataset view or a converting reader to widen" (or "to check each
  value").
- **The downloaders' user agent and error text.** `download
  huggingface`, `download s2ag` and `download bulk` send
  `vectordata/<version>`, not `veks/0.14` (COS requests and the bulk
  HEAD probe sent none). Their HTTP failures read as the transport's
  errors ("HTTP request failed: HTTP 404: Not Found …"); the S2
  messages are unchanged, and "FAILED … after N attempts" counts the
  attempts made.
- **Offline mode refuses `whoami`, `ping` and namespace completion**
  before any request, with the offline error. `login`, `token issue`,
  `token revoke`, `push` and `backup` are not affected (`backup` lists
  its namespaces with `endpoint::access_view`, the explicit form of
  `whoami`). An offline
  `login --token` stores the credential without a user label.
- **The downloaders retry less, and later.** A permanent 4xx (other
  than 408, 425 and 429) and an offline refusal fail at once; per-file
  retries back off from 1 s, doubling to 30 s, ±10 %, where they
  retried at once. The S2 file list keeps its 2–16 s schedule, and
  also retries 408, 425 and a failed body read.
- **`tries = 0` means one attempt** in `download bulk`, which made
  none.
- **A streaming source reports a record it cannot read.**
  `VecSource::next_record` returns `Result<Option<Vec<u8>>, IoError>`,
  `read_record_into` `Result<bool, IoError>`, and
  `VarlenSource::next_record` `Result<Option<VarlenRecord>, IoError>`:
  `Ok(None)` is the end, and an error is where the stream stops. A
  source implemented outside the workspace wraps its records in `Ok`.

### Fixes

- **`vectordata::catalog::source::{CatalogSource, CatalogOrigin}`
  resolve again.** 4.1.0 moved the two to veks-apicore and re-exported
  them at `vectordata::catalog`, but not at `catalog::source`, where
  4.0.2 defined them, so code naming them there stopped compiling.
- **`datasets::at_sources` with two or more catalog names registered
  a program overlay once per name.** Each named catalog came with the
  candidates' overlays, so every name after the first added the
  program's overlay again, and that was refused as two overlays setting
  one key: an error diagnostic per name, which `Catalog::of` reported.
  The overlays are now carried once for the whole set, and a location
  in the list reads them as a name does. `CatalogSources::named`
  returns the catalog alone.
- **The attribute value type had no name in vectordata.** Attributes,
  profile tags and overlay values are `serde_yaml` values, but neither
  the type nor the crate was re-exported, so a program writing a helper
  over attributes needed its own `serde_yaml` at vectordata's exact
  version. `vectordata::AttributeValue` (and `veks_apicore::AttributeValue`)
  names the type, and `vectordata::serde_yaml` is the crate.
- **A directory of symlinked files was not inferred.** Loose-file
  inference skipped every link, so a dataset staged by linking its
  files into a scratch directory failed with "name the files for their
  roles:" and an empty list. Links are followed, for files and for
  `profiles/` directories; a link to nothing is refused, naming it; and
  a directory with no vector file says so.
- **`config catalog add .` saved the resolved directory, not the one
  the user was in.** A directory reached through a link (every temp
  directory on macOS, through `/var` → `/private/var`) was saved by its
  resolved path, so `remove` by the path the user saw could not find
  it. The shell's `$PWD` is used when it names the same directory.
- **Windows paths in catalogs.** A catalog file named by a Windows path
  gave its datasets an empty home, and `config catalog add` named a
  catalog after the whole path (`c-data-stage`); a `\` now separates a
  local path's parts there, as `/` does.
- **A directory of npy or parquet files that held a corrupt file
  ended early, quietly.** The loaders logged the file at info level
  and skipped it, so the stream stopped with fewer records than its
  `record_count` and every caller took that for the end. The stream
  now yields the records of the files before it, then an error naming
  the file; `transform convert` fails on it instead of reporting
  success. The parquet MNode reader is the same.
- **A parquet directory streamed only its first file.** The reader
  waited for that file again from its loaders, which had started at
  the second, and ended when they finished; every file is read now.
- **The other streaming sources stop with an error, not an end:** an
  xvec record whose header states another dimension, a varlen record
  the file ends inside of or whose header is negative, a scalar value
  the file can no longer deliver, a slab page that cannot be read, and
  a parquet row that is null, of another length, or in a file of
  another schema. A varlen record of dimension 0, as the varlen writer
  writes it, reads as an empty record where it ended the stream.

- **veks-io's byte-level writers reported a full disk as success.**
  `xvec::writer`, `scalar::writer` and the varlen writer discarded
  every write error, so a batch larger than their buffer, written
  straight to the file, could fail and `finish` still return `Ok`.
  They keep the first error and `finish` returns it.
- **`datasets derive` read a negative dimension as a huge one** when
  copying a window of a uniform file; it is refused.
- **`config catalog add .` saves the directory, absolute, under its
  name.** It saved `0: .`: relative, so it resolved only from that
  working directory, and named by index, so `config catalog remove`
  could not find it by name or path. A path is now saved absolute and
  named after its directory (`stage: /data/stage`), and `remove` finds
  it by either spelling. A URL is saved as written.
- **`fetch_if_modified` refused a self-signed origin the user
  trusts.** It used the default client, not the one for the URL's
  origin, so `trust_self_signed` in `settings.yaml` was ignored and
  the request failed with `UnknownIssuer`. Every request now goes
  through the client for its origin.
- **`generate vectors` appended onto a cut file.** A file with a
  partial last record took new rows after it, and every record from
  there on was misread. An append now continues only a whole file of
  the same dimension and type.
- **Scalar columns of `i8`, `u16` and `u32` were read with the wrong
  sign** by the predicate-key generator and the SQLite predicate
  verifier: an `.i8` column read as unsigned, and `.u16` and `.u32` as
  signed. Each is read at its own type.
- **A kNN result or segment cache file whose rows are not `k` wide is
  refused** when read back. `k = 0` is refused.
- **`datasets filter` sized a bf16 base as 4-byte elements**, and took
  any extension ending in `vec` for an xvec file.
- **veks-io's byte-level xvec writer wrapped a dimension over
  `i32::MAX`** into its header. It is refused at open.
- **The parquet vector reader read only `f32` list columns** and
  reported 4-byte elements whatever the column held. A
  `FixedSizeList` column was found and then decoded as a `List`, so
  every row was missing. Every numeric list column now reads at its
  own type.
- **Write errors the pipeline ignored are returned.** Writers that
  panicked on an I/O error or discarded it now return it as the step's
  error.
- **The explorer's catalog read ignored offline mode.**
- **A request to a server that has gone away is unreachable**, not a
  transient error to retry: for example, a pooled connection that gets
  no answer.
- **`vecd daemon status --wait` accepted any certificate.** Its
  readiness probe trusts vecd's own `tls_cert` and nothing else, with
  verification on.
- **The update check ignored the TLS trust and offline gate** of
  `settings.yaml`. It goes through a transport of its own built from
  the process's, still following no redirect, with its 5 s ceiling.
- **Offline mode was ignored by endpoint lookups and the downloaders.**
  `whoami`, `ping` and namespace completion, and `download
  huggingface`, `download s2ag` and `download bulk`, made requests
  with offline mode on; they are refused before the network.
- **The downloaders ignored per-origin TLS trust**: a custom CA in
  `trusted_ca_certs` or an origin in `trust_self_signed` did not reach
  them. They use the process's trust.
- **A blank `HF_TOKEN` sent `Authorization: Bearer `.** It sends no
  `Authorization` header.
- **The bulk HEAD probe retried a 404 ten times.** A permanent answer
  fails at once.

## 4.0.2 — 2026-10-20

The `datasets` commands read a dataset however it is named — by
catalog name, directory, `dataset.yaml` or URL — and a program that
embeds them can hand them its own catalogs.

### Highlights

- **`datasets describe` takes a path or URL**, as `precache` does: a
  directory, a `dataset.yaml` or a URL, each with an optional
  `:profile` selector. Catalogs are read only for a name.
  `describe::run_via_catalog` takes such a head too.
- **`--at` names a catalog.** Each value, in order, is a URL or an
  existing path; else a configured catalog's number; else the name of
  a configured or program catalog; else a location. `--at fvec --at
  /mnt/stage` reads the program's `fvec` catalog and the directory and
  nothing from `catalogs.yaml`. `describe --at` and `ping --at` repeat
  like the other commands'. `datasets::at_sources` and
  `CatalogSources::named` are the pieces.
- **`datasets::list::run_via_catalog`** lists a catalog the caller
  already assembled, in every output format, with filters and
  `--select`.

### Fixes

- **A name two catalogs hold is ambiguous**, not missing. `describe`
  and `ping` reported it as not found in any catalog; they now list
  each catalog that holds it and say to choose one with `--at`.
  `cache status` says why it prints no profile summary for one.
- **`describe` counts a series' files.** A uniform `__NNNN` series
  printed `shards 1` with its pattern as the one file; it now prints
  the expanded count, the pattern once and every file.
- **`list --select` returns its exit code** instead of ending the
  process, so a program calling it keeps running. The CLI's exit
  status is unchanged.

## 4.0.1 — 2026-10-20

### Highlights

- **`DistanceFunction`** (in `veks-datatypes`, re-exported as
  `vectordata::DistanceFunction`): L2, Cosine, DotProduct, L1,
  Hamming, Jaccard, with one table of spellings — `euclidean`, `cos`,
  `angular`, `ip`, `inner_product`, `mips`, `manhattan`, `taxicab`
  and the rest, case-insensitive, `-` or space as `_`. `distance()` on
  `TestDataView`, `DatasetConfig` and `DatasetAttributes` returns it,
  `None` when undeclared or unrecognised. `veks_simd::Metric` and
  every pipeline metric option parse through it.
- **`shell::bin_main_with_catalogs(invoked, argv, catalogs)`** runs
  the embedded `vectordata` CLI over its program's catalogs: `datasets
  list`, `describe`, `ping`, `precache`, `derive` and `explore` see
  them after the configured ones and before `--catalog`'s; `--at`
  replaces them all. `CatalogSources::extend` appends one set to
  another.
- **`TestDataGroup::name()`** returns the dataset's cache identity:
  the catalog entry's name, a directory's last segment, or
  `from_config`'s name; `None` at a server's root.

### Fixes

- **`veks prepare --metric angular`** emitted KNN steps without the
  cosine strategy they require; every spelling of cosine now gets it.
- **Publishing 4.0.0 failed**: `veks` and `veks-core` pinned
  `veks-datatypes = "=3.0.4"` as a dev-dependency, which the version
  bump did not rewrite. The dev-dependencies are path-only.
- The 4.0.0 deprecations carry `since = "4.0.0"`, and the rustdoc
  examples and sysref call the `*_as` forms; doctests in `vectordata`,
  `veks-io` and `veks-datatypes` deny deprecated items.

## 4.0.0 — 2026-10-20

One element model for the workspace, and a typed facet access API on
it. A new leaf crate, `veks-datatypes`, holds the twelve element
types (`bf16` is new), the one extension table, the `Element` trait
and the conversion table that says how a facet of one type may be
read as another: natively, widened when lossless, checked per value
across a sign, or refused — never a reinterpretation of the bytes.
`vectordata`, `veks-io`, `veks-core` and `veks-pipeline` are all on
it; their own copies of the element enum, the facet enum and the
extension tables are gone.

This changes the library API: see [Upgrading](#upgrading-from-30).
Every pinned call of 3.0 still compiles, deprecated, beside its
replacement. The design is `docs/design/srd-element-access.md`.

### Highlights

- **Any facet at any type: `view.facet_as::<T>(name)`** and the
  per-facet `base_vectors_as::<T>()`, `query_vectors_as`,
  `neighbor_indices_as`, `neighbor_distances_as`, the filtered six,
  `metadata_results_as` — on `TestDataViewExt`, implemented for every
  `TestDataView`. A `bf16`, `f16` or `u8` base reads as `f32`; `u8`
  codes read as `u8`, zero-copy; an `f32` base asked for as `u8` is
  `IoError::ElementMismatch`. `open_facet_native(name)` opens a facet
  at whatever type it holds (`AnyVectorReader`), `open_vec_as::<T>`
  and `open_vec_native` do the same for a raw path or URL.
- **`VectorReader<T>` says what it is**: `as_slice()` borrows a whole
  header-free column from the mapping, `conversion()` says whether the
  reader converts, `native_element_type()` says what the file holds.
  `ScalarReader<T>` reads a column natively; `ConvertingReader<N, T>`
  carries a widening or cross-sign read and reports the first value
  that does not fit as `IoError::ValueOverflow` at its ordinal.
- **`bf16`** everywhere: `.bf16`, `.bf16vecs`, `.bf16vvecs`, a type of
  its own in the readers, the writers, the facet spec, the pipeline's
  analysis commands and the explorer. `half::{f16, bf16}` are
  re-exported from `veks-datatypes` and `vectordata::typed_access`.
- **Float scalar columns** (`.f32`, `.f64`, `.f16`, `.bf16`) are a
  facet format, `FacetFormat::FloatScalar`.
- **Pre-quantized corpora**: three standard facets — `quant_scale`
  (`.f32`), `quant_zero_point` (`.i32`), `quant_sum` (`.i32`), one
  column each, index-aligned with `base_vectors` and windowed with it
  — and a typed `quantization` attribute (scheme, granularity, code
  type, range). `view.quantized_base::<u8>()` opens the codes and the
  three columns as one thing, checked against each other
  (`Error::FacetCardinality`) and against the attribute. `veks check`
  holds the columns, the attribute and the counts.
- **`veks check` holds every ground truth** — G, F and E: one row per
  query, distances' rows equal to the indices' rows, every index
  under the base count with the `-1` sentinel admitted only in a
  filtered truth's short row. The index scan reports progress.
- **The explorer browses `f16`, `bf16` and integer bases.**
- **`veks-io`** readers are generic over `Element`
  (`MmapReader::<f32>::open`, `ScalarMmapReader::<u8>::open`, with
  `as_slice`) and validate what they map: a negative or zero
  dimension, a truncated file and a wrong element type are refused at
  open, and an unaligned `get_slice` is `None` rather than undefined
  behaviour. `VecFormat` carries `bf16` and the float scalars.
- **`veks-pipeline`'s `dispatch_reader!`** opens an `.i8vecs` as `i8`
  and a `.u16vecs` as `u16`; it opened them as the unsigned or signed
  type of the same width.
- **Every KNN engine and verifier computes over a bf16 or 8-/16-bit
  integer base** by widening it to f32 exactly: `compute knn-metal`,
  `knn-stdarch`, `knn-blas`, `knn-faiss`, `compute prefiltered-knn`,
  `verify knn-consolidated`, `verify knn-groundtruth` and `verify
  dataset-knnutils`. The base is scanned at its own width and widened
  one vector or one streamed chunk at a time, never copied as f32;
  queries may be f32 or any type that widens to it. A 32- or 64-bit
  integer base is refused by name.
- **`transform extract` takes a vector file of every element type**,
  gathering whole records at the source's type for the types without a
  specialized extractor.

### Upgrading from 3.0

Each changed call, and what replaces it. The old spelling stays,
deprecated, unless the row says otherwise.

| 3.0 | now |
|---|---|
| `view.base_vectors()`, `query_vectors()`, `neighbor_indices()`, `neighbor_distances()`, the filtered six (f32 / i32) | `view.base_vectors_as::<f32>()` … on `TestDataViewExt` (import it); the type is yours to choose |
| `view.facet(name)` (f32) | `view.facet_as::<T>(name)` |
| `view.metadata_results()` / `metadata_indices()` | `view.metadata_results_as::<i32>()` |
| `open_facet_typed::<T>(&*view, name)`, `TypedReader<T>`, `TypedAccessError`, `dispatch_typed!` | `view.facet_as::<T>(name)`; `get_value(i)` → `get(i)?[0]`, `get_native(i)` → `as_slice()?[i]` or `get_slice(i)?[0]`, `is_native()` → `conversion() == Native`; `dispatch_element!` |
| `T: VvecElement`, `T: TypedElement` | `T: Element` (the shims remain; `TypedElement::width`/`to_i128` are gone) |
| `XvecReader::<f32>::open("x.mvecs")` — reinterpreted the bytes | `open_vec_as::<f32>("x.mvecs")` — widened; the raw open is now `ElementMismatch` |
| `match ElementType { … }` exhaustively | `ElementType` is `#[non_exhaustive]` and has `BF16`: add a `_` arm |
| `match IoError { … }` exhaustively | `#[non_exhaustive]`; new `ElementMismatch`, `ValueOverflow` |
| `match FacetFormat` / `StandardFacet` exhaustively | new `FloatScalar`; new `QuantScale`, `QuantZeroPoint`, `QuantSum` |
| `TestDataGroup::load` on a `knn_entries.yaml` directory with several entries | opens the entry named after the directory, else the only one, else `Error::AmbiguousDataset` |
| `veks_core::formats::Facet` | `vectordata::dataset::StandardFacet` (removed, no shim) |
| `veks_core::formats::preferred_format(facet, width)` | `preferred_vec_format(StandardFacet, ElementType)` |
| `veks_pipeline::pipeline::ElementType` (its own enum), `ToF64` | the same path, now `veks_datatypes::ElementType`; `Element::as_f64` |
| `veks_io::MmapReader::open_fvec(p)`, `ScalarMmapReader::open_u8(p)` | `MmapReader::<f32>::open(p)`, `ScalarMmapReader::<u8>::open(p)` |
| `veks_io::ScalarType` | `ElementType` |

### Fixes

- **A raw reader refuses a file of another type.** `XvecReader::open`
  and `open_path` compared nothing to the extension, so
  `XvecReader::<f32>::open("x.ivecs")` read integers as floats and a
  `.dvecs` opened as `f32` gave a reader of garbage with the wrong
  count. Every raw open now requires the extension's type to be the
  requested one; a conversion goes through `open_vec_as`.
- **`TypedReader` read every float as zero.** A `.f32` column opened as
  `TypedReader<i32>` passed the width check and returned `0` for every
  value. A float native type is now refused at open.
- **A `dataset.yaml` named as a catalog source** listed a dataset per
  top-level key (`profiles`, `attributes`, …). It is now one entry.
- **`transform convert` f32 → bf16 wrote f16 bits under a bf16
  extension**, and a width pair it had no converter for wrote the
  source bytes into the target unchanged (f32 → u8 a corrupt file;
  u8 → f32 broken; same-width pairs reinterpreted). The conversion is
  chosen from the two element types: widening and cross-sign as the
  conversion table allows, or any float to float; anything else is
  refused naming both.
- **`transform extract` refused `.dvec`, `.svec` and `.bvec` sources**
  its documentation listed.
- **`veks prepare import` of a base-only source never wrote the base
  facet.**
- **`XvecReader::get_slice` gave `None` for every 8-byte element type**;
  every other record of such a file is aligned and now borrows.
- **`veks-io` validated nothing it mapped.** A negative dimension was
  cast to a size, the entry size overflowed, a truncated file read
  past its end, and an 8-byte `get_slice` at an unaligned offset was
  undefined behaviour.
- **`generate dataset-json` rewrites a `dataset.json` an older veks
  rendered.** It skipped whenever `dataset.json` was newer than
  `dataset.yaml`, so the 3.0.1 re-run of the finalize steps left an
  older build's `dataset.json` in place. It now renders every time and
  writes when the result differs from the file there; an identical file
  keeps its modification time.
- **A new veks re-runs a dataset's finalize steps.** Under the default
  `--provenance config-only`, a step built by another veks counted as
  fresh. That is right for compute steps, and wrong for the finalize
  steps (`dataset.json`, the catalogs, the docs, the merkle trees):
  they project `dataset.yaml`, so a fix to the code that writes them
  never reached a dataset whose projections an older veks had written.
  Finalize steps now also count the binary under every selector; compute
  steps keep the run's selector.
- **`veks run --explain-staleness`** now reports the finalize steps too,
  after the compute steps, each judged as a run would judge it.
- **`veks run --explain-staleness` agrees with the run.** It named each
  upstream by the node a re-run would record, even for an upstream the
  run keeps, so on a dataset whose compute steps an older veks recorded
  it called their dependents and every finalize step stale while
  `veks run` correctly skipped them. A fresh upstream is now named by the
  record it keeps.
- **`veks run --explain-staleness` writes nothing.** It appended to
  `runlog.jsonl` and `.cache/run.log` and rewrote `variables.yaml`, so
  asking why a step would run changed files a publish carries.
- **Pipeline variables are saved only when one changes.** Every
  `veks run` rewrote `variables.yaml` to set `dataset_name` and
  `distance_function` to the values it already held, which gave an
  unchanged file a new modification time.
- **A publish sends only the checksum files that changed, and reports
  them.** After the content uploads it sent every directory's
  `SHA256SUMS` again, one at a time and silently, and then deleted
  orphans silently: over S3, a change touching 4 directories of 186 sat
  on one line for minutes. It now sends only the sums that differ from
  the remote's, on `--concurrency` streams, with a running count, and
  counts its deletions the same way.
- **`veks prepare cache-compress` and `cache-uncompress` stream.**
  Each read its files whole and compressed them in memory, on every
  thread at once, so a cache of large segments ran the host out of
  memory (killed at 765 GB on a 1 TB host). Each file is now compressed
  through a fixed buffer to a temporary file renamed into place, with a
  running count of files and bytes.
- **A compressed predicate-key segment cache is reused.** The
  predicate-key step looked for a `.gz` segment only when it set
  `compress-cache`, so every segment `cache-compress` compressed on a
  dataset without that option was recomputed on its next run. It now
  reads either form, decompressing by streaming rather than in memory,
  as the KNN and dedup caches already did.
- **The `veks run` hint to run `cache-compress`** now recognises a
  pipeline that sets `compress-cache`; it looked for `compress_cache`.

## 3.0.0 — 2026-10-15

Every crate in the workspace moves to 3.0.0 and is released with the
others at one version from now on. The library API changes are in
`vectordata`, and in `veks-core` and `veks-pipeline`, which re-export
its types; the other crates move for the lockstep version and their own
APIs are unchanged.

The release breaks the library API: code built against 2.x needs the
changes under [Upgrading](#upgrading-from-25). A 2.x pin
(`vectordata = "2.5"`) keeps building against 2.5.1 until you bump it.

### Highlights

- **Catalogs, entries and datasets can be built at runtime.** A program
  can describe a dataset in code (`DatasetConfig::builder`), list it in
  a catalog entry (`CatalogEntry::builder`), and register its own
  catalogs beside the configured ones (`CatalogBuilder`,
  `CatalogSources::with`). These behave as configured catalogs do: their
  datasets list, open, and report which catalog they came from. See
  `crates/vectordata/examples/register_catalog.rs`.
- **One description, one inheritance rule.** A program opening a
  dataset, the pipeline, the catalog generator, `dataset.json` and the
  docs all read a `dataset.yaml` the same way. Before, the catalog
  generator had its own rule, and published listings that disagreed
  with what an open read (see [Published data](#published-data)).
- **One catalog generator.** `veks prepare catalog generate` and the
  pipeline's `catalog generate` step now write the same files.
- **`knn_entries.yaml` states only what it can state exactly** (see
  below).

### Fixes

- **Configuration files are replaced whole, never rewritten in place.**
  `catalogs.yaml`, `settings.yaml` and `credentials.toml` were written
  by truncating the file and then writing it, so another process reading
  one in between, such as a second `vectordata` command while the
  explorer edits catalogs, could find it empty. Each is now written to a
  temporary file beside it and renamed over it, so a reader sees the old
  file or the new one.
  - A symlinked configuration file stays a symlink; its target is
    replaced, as before.
  - A configuration file keeps its permissions.
  - `credentials.toml` is owner-only (`0600`) from the moment it is
    created, before any token is written to it. Before, it was made
    owner-only after the tokens were written.
- The writes that already used a temporary file and a rename (publish's
  local files and objects, backup restores) now share one
  implementation, which also removes its temporary file when the rename
  fails.

### Upgrading from 2.5

Each removed or changed call, and what replaces it:

| 2.5 | 3.0 |
|---|---|
| `entry.name`, `.path`, `.dataset_type`, `.layout`, `.catalog_file`, `.catalog_name` (public fields of `CatalogEntry`) | `entry.name()`, `.location()`, `.dataset_type()`, `.profiles()` / `.attributes()` / `.layout()`, `.catalog_file()`, `.catalog_name()`. Use `location()` for where the entry lives; `layout()` is only the listing's summary |
| `CatalogEntry { .. }` struct literal | `CatalogEntry::builder(name)…build()` |
| `entry.dataset_home_url()` returning `String` | Returns `&str`; add `.to_string()` where you need an owned value |
| `vectordata::knn_entries` (`KnnEntries`, `KnnEntry`) | `vectordata::catalog::knn_entries`, or open the directory as a catalog |
| `CatalogSources::required()`, `optional()` | `CatalogSources::iter()`, filtering on `Requirement` |
| `CatalogSources::diagnostics()` returning a slice | Returns a `Vec` |
| `catalog::sources::catalog_file_for`, `ensure_trailing_slash` | `vectordata::location::dir_form` |
| `config::verify_catalog_source(source)` | `config::verify_catalog_source(name, source)` |
| `DataSource` | `Location` (`Location::parse`, `Location::from_path`) |
| `GenericTestDataView::new` / `with_attributes` taking a `DataSource` | Take a `Location` |
| `datasets::build_sources` returning `CatalogSources` | Returns `RunSources`; use `.sources()`, or `.open()` for the catalog |
| `datasets::{describe, ping}::run_args(.., at_extra)` | The `at_extra` argument is gone |

New public fields: `DatasetConfig.header`, `DatasetAttributes.other`,
`CatalogDiagnostic.catalog` and `PrecacheRequest.catalog`. A struct
literal of one of these types needs the field, or
`..Default::default()`.

### Behaviour changes

**Library**

- **Opening a dataset** follows the one inheritance rule. It also reads
  facets keyed by aliases (such as jvector's and HDF5's `base`, `query`,
  `gt`, `indices`, `distances`, `train`, `test`), accepts a `dataset.yaml` with no `name` (the directory
  names it), and expands compact sized profiles (`sized:`, `strata:`).
  Datasets written jvector-style, which 2.5 opened with no base, query
  or ground truth, now open fully.
- **An oracle-partition profile** no longer inherits the default
  profile's `metadata_content` and `metadata_predicates`, which describe
  other rows.
- **A profile's facets** are listed in one order everywhere: the
  facets it shares with the default profile, in the default's order;
  then its own; then what a nearer parent adds. This is the order 2.5
  published.
- **`Catalog::of`** lists catalogs in the order they were added. 2.5
  listed required catalogs first, so a set mixing
  `configure_default()` with `add_catalogs(..)` may list in a different
  order.
- **`Error::AmbiguousDataset`** names the catalog document
  (`…/catalog.json`), not only its directory.
- **A remote catalog directory** is also probed for `catalog.yaml`: one
  more request for a catalog that only has `knn_entries.yaml`.
- **Relative entries in `catalogs.yaml`** resolve against that file's
  directory, not the working directory. Names in a nested
  `catalogs.yaml` are kept.
- **A directory holding only `knn_entries.yaml`** finds local copies as
  jvector does. Each file is read from the entry's `cache_dir`, else
  `$DATASET_CACHE_DIR`, else the catalog's own directory, when it is
  there; otherwise it comes from `_defaults.base_url`. An entry key
  given twice keeps its last copy, with a warning, rather than failing
  the file.

**`knn_entries.yaml`**

jvector's format names three whole files per entry: base, query and
ground truth. The generator now lists a profile only when each of the
three is one whole file, with no window, no multi-file series and no
namespace inside a file.

- Every other profile with all three facets is left out. The generator
  reports each one (`knn_entries.yaml omits <dataset>:<profile>:
  <reason>`).
- A directory with nothing left to list has its old `knn_entries.yaml`
  removed, and that is reported too.
- `merkle create` removes a `.mref` whose file is gone.

2.5 listed a sized profile against the whole base file, beside ground
truth computed over the profile's first rows. It listed a sharded base
as its `NNNN` pattern, which jvector cannot load.

**Pipeline**

- **`veks run` leaves `dataset.yaml` alone** when what it would write
  describes the same dataset. Before, a file not in the renderer's
  exact layout was rewritten on every run. That changed its content key
  and sent every finalize step round again.

**CLI**

- **`--at` and `--catalog`**: a catalog named this way that fails to
  load makes any `datasets` command exit with status 1. A configured
  catalog that fails is reported, and the command continues.
- **`datasets describe` and `datasets ping`**: `--at` replaces the
  configured catalogs, as in every other command.
- **`veks catalog generate`** is the pipeline command (`--source`).
  `veks prepare catalog generate <INPUT>` is the manual one.
- **`veks prepare catalog generate`** writes what the pipeline writes:
  - `knn_entries.yaml` at every level;
  - template-derived views;
  - sized profiles read through `variables.yaml`.
- **`.do_not_catalog`**: nothing below a directory marked with it is
  listed in any catalog, including the catalogs above the marker. The
  manual generator used to list such datasets above it.
  `veks check catalogs` follows the same rule.
- **`dataset.yaml` writers** (`prepare bootstrap`, `datasets derive`,
  `generate example-dataset`, `gen_dataset`, `gen_derive`):
  - their output parses to the same description as before;
  - its layout is the renderer's (for example, a structured `strata:`
    block);
  - it carries the strata and tag-schema comments.

  A script that matches these files by their text layout may need
  updating.
- **The explorer:**
  - reports catalogs that fail to load, on its status line and with `✗`
    on the catalogs screen;
  - opens the catalog once per pass rather than once per action;
  - lists the catalogs screen in the order the catalogs were added;
  - finds a dataset to purge case-insensitively.
- **`vectordata::explore::run_with_sources`** opens the explorer over a
  program's own catalog set.

### Published data

Catalogs and docs generated by 2.5 could disagree with the
`dataset.yaml` they came from. Data files and `dataset.yaml` files were
never affected, and neither were programs that open a dataset.
Listings were: anything that read a profile's files from a catalog
(`catalog.json`, `catalog.yaml`, `dataset.json`, generated docs, or
`knn_entries.yaml`).

- **tessera, `*-unfiltered` profiles.** The 2.5 catalogs listed five of
  the default profile's filtered outputs under each of the 46 unfiltered
  profiles: `metadata_results`, and the pre- and post-filtered neighbour
  indices and distances. These outputs are computed over the full base.
  A filtered-search result for an unfiltered profile, scored with files
  taken from the listing, is invalid. A result from opening the profile
  is not affected.
- **`knn_entries.yaml` with sized or sharded profiles**, tessera's
  included. A result for a sized profile that read its base from
  `knn_entries.yaml` searched the whole base, while its ground truth
  covered only the profile's rows, so it is invalid.

tessera's catalogs and docs will be regenerated with this release.
Its `dataset.yaml` and data files do not change. Its `knn_entries.yaml`
will be removed, because no profile in it can be stated exactly.
