# SRD — Crate boundaries: one format layer, one transport layer

**Status:** proposed 2026-10-21; §10's questions decided 2026-10-22
including the follow-ups Q12a–Q12d. The plan in §11 follows the decisions.
**Scope:** a new API core crate (`veks-apicore`, §4), `veks-io`, a
new transport crate (`veks-transport`, §10 Q2), `vectordata`,
`veks-core`, `veks-pipeline`, `veks`, `vecd`.
**Builds on** `srd-loose-files-overlays-integrity.md` §6.1 (veks-io is
the format layer; `ByteSource`), `storage_transport_factoring.md`
(transport vs shape), `srd-element-access.md` (the element model).

**Origin.** After veks-io became the one implementation of the random
-access readers (2026-10-21), a review of the crate graph asked
whether the crates now meet on clean boundaries. They do for the read
path; they do not for streaming formats, for writers, or for the
network. This SRD makes both layers whole and puts remote access and
streaming in a crate of their own behind traits, so that nothing above
them — vectordata, the pipeline, vecd, an outside program — parses a
vector file or builds an HTTP client of its own. The API itself — the
traits the crates speak to one another through and every type they
name, dataset contracts included — lives in a crate of its own,
`veks-apicore`, which holds no implementation, so that each layer and
every program depends on the API and not on the crate that happens to
implement it.

## 1. Problem

The findings below come from a survey of the workspace at 13577a2
(veks-studies excluded). Each names files; §2 has the detail.

1. **Two streaming format layers.** veks-core `formats/` defines its
   own `VecSource`, `SourceMeta`, `VecSink` and `VecFormat`, beside
   veks-io's of the same names (veks-core re-exports veks-io's as
   `IoVecSource`, `IoVecSink`, `IoSourceMeta`). Its xvec reader and
   writer are independent implementations with their own page-cache
   policy; it wraps veks-io only for scalar formats, and that adapter
   drops `read_record_into`. The two `VecFormat`s convert through a
   string round trip that falls back to `ScalarU8` when a name does not
   parse (`reader/mod.rs:118`, `writer/mod.rs:185`).
2. **Capabilities only the duplicate has.** A directory of xvec shards
   read as one stream; a positional multi-threaded xvec writer
   (`writer/mmap_xvec.rs`); writeback control while writing
   (`sync_file_range` + `FADV_DONTNEED`); a sharding sink; the npy,
   parquet and slab readers. veks-io's `npy`, `parquet` and `slab`
   features are stubs that answer "not yet implemented".
3. **Format code written inline.** About seventy production sites in
   veks-pipeline, veks, veks-core and vectordata read a dim header,
   compute a record stride, write records with `to_le_bytes`, or keep
   an extension table of their own (§2.3). Some are lossy: an
   extension table without `u16`/`i8`/`u32`, `== Some("fvec")` that
   misses `fvecs`, a `u32` dim read. Some have no veks-io equivalent
   to move to (§2.3, G1–G5).
4. **A deprecated reader with its own parser.** `typed_access::
   TypedReader` (deprecated in 4.0.0) reads the dim with
   `i32::from_le_bytes` and does not check it: a negative dim gives a
   huge stride and a count of 0, and nothing reports the corruption.
5. **A wrong element width.** veks-core's slab and parquet readers
   report a 4-byte element whatever the data (`slab.rs:80`,
   `parquet.rs:463`; the slab probe assumes f32), while npy reports its
   dtype; `SinkConfig`'s doc says npy and parquet land as f32. One of
   these is wrong for every non-f32 source.
6. **HTTP clients outside the hub.** vectordata's `transport` module
   is crate-private, so nothing outside vectordata can use it.
   veks-pipeline builds five clients of its own (`fetch_dlhf`,
   `fetch_s2ag`, `fetch_bulkdl` download, HEAD and COS), each with no
   offline check, no custom CA and its own user agent, and hand-rolls
   SigV4 for COS. vectordata's `update_check.rs` builds its own client
   (it needs no-redirect, which the pool lacks); vecd builds one for
   its loopback health probe.
7. **Inside the hub, uneven.** `fetch_if_modified` uses
   `shared_client()` where every other call uses
   `shared_client_for(url)`, so `trust_self_signed` is ignored for it
   (`transport/mod.rs:366`). `catalog/resolver.rs:619` repeats
   `apply_read_auth` by hand. `endpoint.rs` never checks offline.
8. **A reqwest type in vectordata's public API.**
   `Error::Http(#[from] reqwest::Error)` (`lib.rs:355`), built at three
   sites in `group.rs` and matched nowhere. veks-io dropped the same
   leak (`IoError::Transport`).
9. **Dependencies declared and unused.** `veks` declares reqwest and
   uses it nowhere; `veks-pipeline` declares tokio and uses it nowhere.

## 2. Survey

### 2.1 The two streaming layers

| | veks-core `formats` | veks-io |
|---|---|---|
| `VecSource` | 4 methods, `Send` | same 4 + `read_record_into` |
| `VecSink` | `write_record`, `finish`; not `Send` | `Send`; + `write_records_fixed_dim` (one batched write) |
| `VecFormat` | veks-io's + `Hdf5` + 12 vvec variants; `ALL`, `is_vvec`, `detect_from_path`, `detect_from_directory`, `record_bytes`, `expected_file_size` | `is_signed` |
| xvec read | file or directory of shards; `.fvec`/`.ivec` via vectordata's `XvecReader` (a `Vec` per record), others via a `BufReader` with `FADV_DONTNEED` every 64 MiB | one file, mapped, released every 256 MiB, size checked |
| xvec write | 4 MiB buffer, writeback every 64 MiB, removes a symlink at the target, **panics on a write error** | 16 MiB buffer, no writeback control, first error returned from `finish` |
| threads / `max_count` | npy, parquet, parquet_mnode spawn loaders; npy honours `max_count` | none |
| sinks | `SinkConfig` (slab page size, namespace, schema sidecar, shard cap); automatic sharding | none |

Consumers of veks-core's layer: veks (`prepare/wizard.rs`,
`prepare/import.rs`, `stratify`) and veks-pipeline (`import`,
`convert`, `describe`, `variables`, `verify_alignment`, `gen_embed`).
Nothing outside veks-core calls veks-io's streaming layer directly.

### 2.2 What veks-core keeps that is not format parsing

`parquet_vector_compiler.rs`, `xvec_dir_compiler.rs`,
`parquet_compiler.rs` (Arrow rows to MNode), `passage_table.rs` (a
domain codec) and `reader/parquet_mnode.rs` (metadata through the MNode
compiler and vectordata's metadata schema) are compilation logic.
`portable_io.rs` (pread/pwrite, `SpanFile`) is local-file plumbing;
`convert.rs` (f16/f32/f64 element conversion over veks-simd) overlaps
veks-io's `scalar/convert.rs`.

### 2.3 Inline format code, by what replaces it

| Kind | Where (production) | Replacement |
|---|---|---|
| record writes (kNN results with `-1`/INF padding, vectors, dim-1 ordinal lists) | veks-pipeline: `compute_knn*`, `knn_segment`, `compute_postfiltered_knn`, `gen_vectors`, `gen_from_model`, `gen_sketch`, `gen_dataset`, `gen_extract`, `gen_shuffle*`, `compute_dedup`, `clean_ordinals`, `analyze_zeros`, `compute_sort_knnutils`, `gen_predicate_keys`, `gen_metadata`, `gen_simple_predicates`, `compute_partition_profiles`, `compute_knn_distances`, `verify_engine_parity` | `XvecWriter` / `ScalarWriter` |
| header and row reads | veks-pipeline: `knn_segment`, `gen_vectors`, `gen_shuffle*`, `verify_predicates*`, `verify_consolidated`, `gen_predicate_keys`, `cleanup_overlap`, `info_file`, `describe`, `verify_engine_parity`; veks-core `xvec_dir_compiler` | `XvecReader`, `ScalarReader`, `IndexedVvecReader`, `probe` |
| stride arithmetic | veks-pipeline: `compute_knn` (size check), `compute_knn_blas`, `compute_knn_stdarch`, `analyze_find_zeros`; vectordata `datasets/derive.rs`, `dataset/shard_sizing.rs` (its own `xvec_record_bytes`) | `xvec_record_bytes`, `entry_size`, `get_slice` |
| extension tables | veks-pipeline: `gen_vectors`, `compute_dedup`, `describe_dataset`, `gen_derive`, `verify_consolidated`; veks: `prepare/wizard.rs` | `ElementType::parse_extension` / `extension`, `VecFormat` |
| **G1** a header and stride from a partial, sparse or remote source | vectordata: `view.rs` (series lengths), `datasets/ping.rs`, `datasets/cache.rs`, `explore/dataset_picker.rs`, `datasets/filter.rs`; veks-pipeline: `cleanup_cleanfvec` (trailing-garbage repair), `bound.rs` (partial-artifact detection); veks `check/integrity.rs` | none yet |
| **G2** records written at given ordinals (pwrite / mmap) | veks-core `writer/mmap_xvec.rs`; veks-pipeline `gen_extract` (partition writes, gap compaction); veks-core `parquet_vector_compiler` | none yet |
| **G3** records packed into a buffer for one bulk write, or raw records into a typed writer | veks-pipeline `compute_knn` (buffered), `gen_vectors` (buffered), `compute_dedup` | none yet |
| **G4** an atomic-rename or shard-capped writer | veks-pipeline `AtomicWriter`, `shard_write::FacetWriter` (`gen_extract`, `transform_remove_zeros_knnutils`) | none yet |
| **G5** a vvec offset walk across a multi-file span | vectordata `datasets/derive.rs` | `load_offsets` covers one source |

`analyze_checkendian.rs` reads headers in both byte orders on purpose
and stays. Slab payloads of packed i32 (`compute_prefiltered_knn`,
`gen_predicate_keys`, `compute_dedup` run files) are not xvec and stay.

### 2.4 Network clients

| Site | Settings | Through the hub? |
|---|---|---|
| vectordata `transport` | pool of clients, UA `vectordata/<ver>`, 10 redirects, 1 h / 10 s timeouts, custom CA, a self-signed pool per trusted origin, read-side auth, offline gate, retry policy, parallel chunk pump | — |
| vectordata `endpoint.rs`, `push/transport/https.rs` | hub client; their own bearer (session / push token); streaming `PUT`, conditional headers, resumable `PATCH` | yes, no offline check |
| vectordata `update_check.rs` | own client: no redirects, 5 s, own UA | no |
| veks-pipeline `fetch_dlhf` (2), `fetch_s2ag`, `fetch_bulkdl/download` (2), `fetch_bulkdl` COS | a new client per call; UA `veks/0.14`; `HF_TOKEN` bearer, `x-api-key`, a caller's `HeaderMap`; hand-rolled SigV4; retries with backoff; bodies streamed to files | no |
| vecd `cli.rs` `wait_for_health` | own client, accepts invalid certificates, 2 s | no (a loopback self-probe) |

All client code is blocking. There is no AWS SDK in the workspace:
S3 writes shell out to the `aws` CLI (vectordata push, vecd backend and
backup); S3 reads are virtual-hosted HTTPS with a region rewrite.

## 3. Principles

1. **One layer per concern, and a layer owns its concern whole.** The
   format layer owns every byte layout of a vector file, read or
   written, local and streaming. The transport layer owns every byte
   that crosses a network, and every client that moves it.
2. **The graph points down.** `veks-datatypes` ← `veks-apicore` ←
   {`veks-io`, `veks-transport`} ← `vectordata` ← `veks-core` ←
   `veks-pipeline` ← `veks`; `vecd` → `vectordata`. veks-io and
   veks-transport are siblings: neither depends on the other, and they
   meet only through `ByteSource`, which is apicore's. Neither knows
   datasets, catalogs or profiles.
3. **Contracts in one place, values for configuration.** A trait that
   one crate implements and another consumes — where the consumer
   should not depend on the implementor — is apicore's, with the plain
   values and the error type its signatures name. Configuration
   crosses a seam as a value, so a layer can be tested with no process
   environment and no files, and a program outside the workspace can
   implement any contract without depending on the crate that holds
   the workspace's implementation.
4. **No third-party types across a seam.** A public signature names
   the layer's own error and request types, not reqwest's.
5. **The cache layout does not change.** Moving code moves no file:
   `<cache_root>/<dataset>/<files>`, the `.mrkl`, `.mref` and `.chunks`
   sidecars and the `.publish_url` binding stay exactly as they are,
   and the layer that decides where a dataset's files live stays in
   vectordata.
6. **Moved code keeps its tests and its speed.** A move carries its
   tests; a merge of two implementations keeps the faster technique of
   each, measured as §6.1 IO-4 was in the integrity SRD.
7. **A contract costs nothing and hides no failure.** No boundary this
   SRD draws may make a hot path slower or an error quieter than it is
   at 13577a2. §4.1 states it as requirements, each with a guard.

## 4. The API core (AC)

**AC-1 — `veks-apicore` is the API.** It defines what a program codes
against — the dataset, catalog, reader, source, streaming, progress
and transport contracts — together with every value, error and
descriptor type their signatures name, and default methods written
only in terms of the contract. vectordata, veks-io and veks-transport
are implementations of it; a program that needs only the contracts
depends on apicore alone, and a second implementation of any of them
(a test double, a vecd-side view, another program's source) depends on
nothing else. It holds no implementation: no file format, no network,
no cache, no dataset resolution. Its dependencies are what its
vocabulary needs — `veks-datatypes`, `thiserror`, `serde`,
`serde_yaml`, `indexmap`, `url` — and never `reqwest`, `memmap2`,
`libc` or an implementation crate. Nothing in it is optional: there is
no feature that removes part of the API.

**AC-2 — What belongs.** Every public trait a crate above an
implementation consumes, and the types it names. A trait used only
inside the crate that defines it stays there (`ChunkedTransport`,
`PushTransport`, `CommandOp`, `UiSink`). A guard (CB-2) checks that no
public trait of an implementation crate is consumed outside it unless
apicore defines it.

**AC-3 — What moves into it**, by the rule:

| Contract | Today | Implemented by | Consumed by |
|---|---|---|---|
| `ByteSource`, `SourcePart` | veks-io `source.rs` | veks-io (`MappedFile`), veks-transport (remote, cached), vectordata (`StorageSource`) | veks-io readers |
| `VectorReader<T>`, `VvecReader<T>` | veks-io `reader/mod.rs` | veks-io readers, vectordata (windowed, sharded, converting views) | vectordata, veks-core, veks-pipeline, outside programs |
| `VecSource`, `VecSink`, `SourceMeta`, `VarlenSource`, `VarlenSink` | veks-io `traits.rs` (and veks-core's copies, FL-1) | veks-io streaming formats | veks-core, veks-pipeline, veks |
| `IoError` (renamed per §10 Q11) | veks-io `reader/mod.rs` | — | every reader contract's signatures |
| `ProgressSink` (`report(label, done, total)`) and `Progress` events | `VerifyProgress` (vectordata), `ByteSource::Progress`, `DownloadProgress` (vectordata transport), per-step callbacks across veks-core and the pipeline | vectordata meters, veks-core UI, closures | veks-io, veks-transport, vectordata, veks-pipeline |
| `Credentials`, `RequestSigner`, `TransportConfig`, `TlsTrust`, `TransportError` | none (TR-3..TR-6) | vectordata (credentials), veks-transport (SigV4), outside programs | veks-transport |

`ByteSource` loses its constructors (`open`, `open_path`): opening is
an implementation's, so the contract names no path or URL scheme. The
veks-io readers' `open_*` functions take a source or a path, as now.

**AC-4 — The dataset API is apicore's** (§4.2): `TestDataView`,
`TestDataViewExt`, `CatalogSource`, the fetch and progress contracts,
and the types they name.

**AC-5 — Paths users know keep working.** veks-io, veks-transport and
vectordata re-export what they used to define (`veks_io::ByteSource`,
`veks_io::VectorReader`, `vectordata::io::VectorReader`,
`vectordata::verify::VerifyProgress` as an alias of `ProgressSink`,
`vectordata::{TestDataView, Error, Result}` and the dataset types),
so moving a contract is not itself a breaking change.

**AC-6 — Documented as the map of the seams.** Its crate docs list
each contract with who implements and who consumes it (the AC-3
table), and each trait has a runnable example that implements it with
nothing but apicore, so a program outside the workspace sees what it
takes to plug in.

### 4.1 Performance and reliability of the contracts (PR)

The release profile has no LTO and the default codegen units, so a
non-generic function in one crate is not inlined into another unless
it says so, and a trait object is a call through a vtable. The
contracts are designed around that.

**PR-1 — Hot paths dispatch statically.** Readers stay generic over
their source (`XvecReader<T, S: ByteSource>`), monomorphized in the
crate that names `S`. No contract adds a `dyn` to a per-record or
per-chunk path; the trait objects that exist today
(`Arc<dyn VectorReader<T>>` behind a view, `AnyVectorReader`) stay at
open time and around whole-range calls (`read_into`, `get_slice`),
where they are now.

**PR-2 — What crosses a crate boundary on a hot path is inlinable.**
Every non-generic function apicore defines that a hot path calls (a
stride, a bounds check, a header decode) is `#[inline]`; default trait
methods are generic and instantiated in the caller already. A guard
lists apicore's non-generic `pub fn`s and requires the attribute on
each, or an entry saying why it is cold.

**PR-3 — The mapping fast path survives.** `ByteSource::mmap_slice`
and `mmap_base` stay on the contract, and a reader still caches the
resident base (the `OnceLock` of IO-4). The hot-path guards of
`veks-io/tests/reader_hot_paths.rs` (decodes per `get`, source asks
per read) keep passing unchanged after the move.

**PR-4 — Opening is not a contract, and costs no dispatch.**
`ByteSource` loses `open`/`open_path` (AC-3); the generic open
functions (`open_vec_with::<S>`) take a constructor trait the
implementing crate defines (`OpenSource: ByteSource`), so they stay
monomorphized.

**PR-5 — Progress is reported per batch, never per record.**
`ProgressSink` is a `&mut dyn` call, so every caller reports at most
once per chunk, page batch or 1 MiB of work. A capturing-sink test per
long phase checks both that progress arrives (the mandatory-indicator
rule) and that it arrives no more often than that.

**PR-6 — Errors cost nothing until they happen.** `TransportError`
and the reader error are built only on the error path; no hot path
formats a string or boxes anything to succeed. A transport error keeps
what a caller decides on — HTTP status, permanent or transient, the
URL — so retry and offline behaviour are decided on the same facts as
today (`is_permanent`).

**PR-7 — A read token is one in-memory lookup per request, never
I/O.** Credentials are keyed by URL prefix, so two paths on one origin
can carry different tokens, and `reload_credentials` replaces them
mid-session; a token cached per origin would be the wrong one. The
store is read once per process and on reload, and each request
resolves its token by longest prefix in memory, as `apply_read_auth`
did. *(Amended 2026-10-22: it read "resolved once per origin", which
the code before the move never did; see §12.7.)*

**PR-8 — No failure becomes quieter.** Every error that reaches a
caller at 13577a2 still reaches it after the move, with the same
variant or a documented replacement: the first write error from a
writer's `finish`, a partial download (`is_complete`), the offline
refusal, a merkle verification failure, an element mismatch. A merge
picks the stricter behaviour of the two implementations (veks-io's
writer reports where veks-core's panics; veks-io's reader checks size
where veks-core's does not). No `unwrap`, `expect` or swallowed
`Result` enters production code in a moved file; clippy's
`unwrap_used` and `expect_used` are denied in veks-apicore,
veks-transport and veks-io.

**PR-9 — Retry and verification are unchanged.** The chunk store
verifies each chunk against its reference before it is marked valid,
and retries transient failures with the same policy and budgets; the
move is checked against the existing `http_storage`, `offline`,
`fetch` and transport-saturation tests, unchanged.

**PR-10 — Measured, before and after.** The reader bench
(`vectordata/benches/readers.rs`) gains the streaming readers and
writers, a remote range read against a local server, and a cached
read; it runs at 13577a2 and after each step that touches a hot path
(§11 steps 0–3, 7), and no case may be slower than before beyond the
bench's noise (the IO-4 method). The results are recorded in this
SRD.

### 4.2 The dataset API (DA)

**DA-1 — What moves.** `TestDataView`, `TestDataViewExt` and
`CatalogSource`, and the types their signatures name: `FacetConfig`,
`FacetDescriptor`, `FacetShape`, `StandardFacet`, `DistanceFunction`,
`Quantization` and `QuantizedBase`, `AttributeSource`,
`CatalogEntry` with the descriptor types it holds (`DatasetConfig`,
profiles, `DatasetAttributes`), `CatalogOrigin`, `CatalogDiagnostic`,
the fetch request, plan, window and event types (`FetchRequest`,
`Fetched`, `PrefetchPlan`, `DSWindow`, `FetchEvent`, `FetchProgress`),
the verify report types (`VerifyReport`, `VerifyFailure`, `Check`,
`VerifyOptions`, `Normalization`, `Measured`), and `AnyVectorReader` /
`AnyVvecReader` (enums over the reader contracts, with `into_typed`).
These are descriptors and values: their parsing (`serde`) moves with
them, since a descriptor's shape is part of the API. Resolving,
opening, fetching and caching do not move.

**DA-2 — The error is the API's.** `vectordata::Error` and `Result`
move into apicore as the API's error (§10 Q12c), with `IoError` (Q11)
beside it as the reader-level error it wraps. vectordata re-exports
both at their current paths.

**DA-3 — Facet bytes are a contract.** `open_facet_storage` returns
`Box<dyn FacetBytes>` (§10 Q12b): total size, prefix read, precache
and completeness, content identity, cache check. vectordata's
`FacetStorage` implements it; vectordata builds its readers from the
concrete type internally, so no reader path gains a trait object
(PR-1). `ContentId` and `ContentIdKind` move with it.

**DA-4 — Defaults use only the contract.** A provided method in
apicore is written in terms of the trait's own methods
(`facet_as` = `open_facet_as_type` then `into_typed`;
`distance` from `distance_function`). Behaviour that needs an
implementation — `verify`, `verify_report`, `normalization`,
`quantized_base`, `fetch` and `plan_fetch` — is a required method of
the contract, implemented by vectordata; its documentation, examples
and guarantees (V1–V9, the recorded measurement, the fetch progress
rules) are stated on the contract, and vectordata's tests hold every
implementation to them.

**DA-5 — Defined without its migration debt.** The 14 deprecated
methods of `TestDataView` (the pinned `f32`/`i32` readers and `facet`,
`prebuffer_all*`, `prefetch*`, the typed opens) are not part of the
API (§10 Q12d). The CHANGELOG's Breaking section carries an upgrade
table: `base_vectors()` → `base_vectors_as::<f32>()`,
`prebuffer_all()` → `fetch(&FetchRequest::all(), …)`, and the rest.

**DA-6 — Performance is unchanged.** The dataset contracts are
already object-safe and called through `Arc<dyn TestDataView>` at
open time; their readers are the `Arc<dyn VectorReader<T>>` they are
today. Moving them adds no dispatch, and `FacetBytes` serves only
cold paths (DA-3).

## 5. The format layer (FL)

**FL-1 — One streaming model.** veks-io's `VecSource`, `VecSink`,
`SourceMeta` and `VecFormat` are the only ones. veks-core's are
deleted, not aliased: their callers are inside the workspace (veks and
veks-pipeline) and move to veks-io's names. `VecFormat` gains what
veks-core's has and veks-io's lacks — the vvec variants, `ALL`,
`is_vvec`, `is_uniform_xvec`, `detect_from_path`,
`detect_from_directory`, `record_bytes`, `expected_file_size` — so no
string round trip and no silent `ScalarU8` remain. Whether `Hdf5`
stays depends on §10 Q8.

**FL-2 — veks-io's streaming xvec reader reads a directory of
shards** as one stream, in the series order vectordata reads
(`__NNNN`), with the size check per file it already makes.

**FL-3 — veks-io's streaming writer controls writeback.** It starts
writeback and releases written pages every 64 MiB, as veks-core's
does; it keeps reporting the first error from `finish` (veks-core's
panics). The buffer size is settled by measurement (16 MiB vs 4 MiB):
`stream_write/sink_write_record` (256 MiB of dim-128 f32) took 183.6 ms
at 16 MiB and 182.5 ms at 4 MiB, the same within noise, so it stays
16 MiB.
**A symlink at a write target is an error**, from every veks-io
writer, naming the link and what it points to: writing through it
would overwrite what it links to (often source data), and removing it
would discard a link someone made on purpose. veks-core's writer
removed one silently (from a bulk commit, 26c0ea2, with no stated
rule); that is not carried over. The pipeline's
`guard_against_symlink_write`, which did the same in twelve files, is
veks-io's now (`veks_io::atomic`, with `AtomicWriter` and the `safe_*`
helpers) and errors too; the atomic writer refuses a symlink final
path, since its rename would replace the link. The process that made a
link removes it itself, explicitly, when it replaces it: the import
unlinks its own Identity link where a new plan materializes that
artifact, before any step writes, and says so. Decided 2026-10-22.

**FL-4 — The gaps G1–G5 become veks-io calls.**
- G1: `probe_source(&impl ByteSource)` reads the header and stride
  from the first record and returns the record count a whole file
  would have, the bytes past the last whole record, and whether the
  source is complete — so a partial download, a sparse cache file and
  a truncated artifact are each described, not refused.
- G2: `PositionalXvecWriter` (veks-core's `mmap_xvec`, moved):
  pre-sized, records written at ordinals from many threads.
- G3: `encode_records` into a caller's buffer, and
  `XvecWriter::write_raw` for records already in the file's layout.
- G4: an atomic writer (write to a sibling, rename on `finish`) and a
  shard-capped writer that names its shards as the series does.
- G5: `load_offsets` over a series.

**FL-5 — Streaming readers of the other formats move into veks-io**
behind its existing features: npy (`npy`), parquet vectors
(`parquet`), slab (`slab`), with their loader threads and `max_count`.
Each reports its real element type (finding 5 is fixed and guarded).
`parquet_mnode`, the compilers and `passage_table` stay in veks-core
(§2.2). veks-core enables the features it needs.

**FL-6 — The sharding sink and `SinkConfig` move into veks-io**, with
the shard naming and discovery they need (§10 Q3).

**FL-7 — Element conversion is one module.** veks-core's
`convert.rs` and veks-io's `scalar/convert.rs` merge into veks-io,
keeping veks-simd's kernels.

**FL-8 — `portable_io` moves into veks-io** as `veks_io::file`
(pread/pwrite, `SpanFile`): local-file plumbing the readers, writers
and the pipeline share.

**FL-9 — Every inline site of §2.3 moves to the format layer.** The
replacements are mechanical where veks-io already has the call, and
follow FL-4 where it did not. A guard (CB-2) keeps new ones out.

**FL-10 — `TypedReader` is removed** (§10 Q4), with
`open_facet_typed`, `TypedAccessError` and `TypedElement`; their
replacements (`facet_as`, `VectorReader`) have existed since 4.0.0.

## 6. The transport layer (TR)

### 6.1 What the crate holds

**TR-1 — A crate `veks-transport`** (name per §10 Q2) holds, moved from
vectordata:
- the client pool, TLS trust (custom CA, per-origin self-signed),
  user agent, redirect and timeout settings, the offline gate, retry
  and backoff (`RetryPolicy`), Last-Modified handling, the parallel
  chunk pump and its semaphore, `DownloadProgress`;
- range reads and whole-file downloads (`HttpTransport`,
  `ChunkedTransport`, `fetch_chunks_parallel`, `fetch_text`,
  `fetch_if_modified`), S3 URL normalisation and region rewrite;
- uploads: streaming `PUT`, conditional headers, resumable `PATCH`
  (the primitives of `push/transport/https.rs`);
- per §10 Q1 (recommended), remote **streaming storage**: the chunk
  store (`chunked_http`), the merkle reference and state (`merkle`),
  the verified cache channel (`cache::CachedChannel`, `cache::reader`),
  and the remote variants of `Storage`, each implementing veks-io's
  `ByteSource`.

**TR-2 — vectordata keeps what is about datasets:** settings and
credentials files, the cache layout and the `.publish_url` binding
(`cache/layout.rs`), catalogs, `Location`, series discovery for a
dataset, `Storage`'s dispatch over a facet's sources, the push command
and its `PushTransport` trait (the `local` and `aws`-CLI `s3`
backends stay with it), and `fetch` (planning, cache-space checks,
meters).

### 6.2 The seams

The traits and values below are apicore's (AC-3); the transport
implements and consumes them.

**TR-3 — Configuration is a value.** `TransportConfig { offline,
tls: TlsTrust, user_agent, timeouts, redirects }`, built by
vectordata from settings and the environment through a pure function,
and passed in. The transport reads no environment variable and no
file of its own.

**TR-4 — Credentials are a trait the transport defines:**
`trait Credentials { fn read_token(&self, url: &str) -> Option<String>; }`
(a `&str`, so apicore needs no `url` crate).
vectordata implements it over `credentials.rs`; a program outside the
workspace implements it over whatever it keeps; `NoCredentials` is the
default.

**TR-5 — A request carries its own authorisation when the caller
chooses it**: a bearer, named headers, or a `RequestSigner`
(`trait RequestSigner { fn sign(&self, req: &mut Request) -> Result<()>; }`).
SigV4 moves from veks-pipeline's COS code into the transport as one
`RequestSigner` (§10 Q6). Read-side credentials (TR-4) apply only when
the request does not carry its own.

**TR-6 — The transport's own types at the seam.** `Request`,
`Response` (status, headers, a body read as bytes, text or a stream
to a file with progress), `TransportError` (status, a permanent or
transient kind, the URL). No reqwest type appears in a public
signature.

**TR-7 — Cache placement is a parameter.** The cached `ByteSource`
takes the cache file's path and its sidecars' paths from vectordata
(which computes them by today's layout, principle 5); the transport
never computes a cache path.

**TR-8 — The transport implements apicore's `ByteSource`** for an
uncached remote range source and for the cached sources, and does not
depend on veks-io. A program reads a remote xvec by handing
veks-transport's source to veks-io's readers, without vectordata.

### 6.3 The clients outside the hub

**TR-9 — veks-pipeline's downloaders use the transport's client**
(§10 Q5): its pool, TLS trust, offline gate and user agent; their own
authorisation per TR-5 (`HF_TOKEN`, `x-api-key`, SigV4), not
vectordata's read credentials. Their retry loops become `RetryPolicy`.

**TR-10 — `update_check` uses a transport client with redirects
off** (`TransportConfig::redirects`), keeping its 5 s timeout.

**TR-11 — vecd's health probe uses a transport client** whose TLS
trust names its own loopback origin, instead of accepting every
certificate (§10 Q9).

**TR-12 — The three defects of finding 7 are fixed in the move:**
`fetch_if_modified` honours per-origin trust; the resolver's
hand-written auth goes through TR-4; `endpoint.rs` checks offline.

### 6.4 Errors

**TR-13 — `vectordata::Error::Http(reqwest::Error)` becomes
`Error::Transport(veks_apicore::TransportError)`** (§10 Q7). The
reader error's `Transport` variant carries a `TransportError` instead
of a boxed `dyn Error`.

## 7. Crate graph and guards (CB)

**CB-1 — Only veks-transport depends on reqwest.** vectordata, veks,
veks-pipeline and vecd drop it from `[dependencies]` (tests may keep
it as a dev-dependency to stand up servers and probe them). veks drops
its unused reqwest; veks-pipeline its unused tokio.

**CB-2 — A workspace guard test** (`tests/crate_boundaries.rs` in the
root, over `cargo metadata` and the source tree):
- the dependency edges are those of principle 2, and no other;
  veks-apicore's are those AC-1 lists, and no implementation crate;
- no public trait of an implementation crate is consumed outside it
  unless apicore defines it (AC-2);
- no crate but veks-transport names `reqwest::` (vectordata's
  `NETWORK_ENTRY_POINTS` guard generalises to the workspace);
- outside veks-io, no production source reads a dim header with
  `i32::from_le_bytes` beside a stride, or matches extension literals
  (`"fvec"`, `"fvecs"`, `"ivec"`, …) — with an allow-list holding
  `analyze_checkendian.rs` and nothing else;
- no crate but veks-io defines a `VecSource`, `VecSink` or
  `VecFormat`.
Each part is shown failing on today's tree before it passes.

**CB-3 — Re-exports keep paths users know.** `vectordata::merkle`,
`vectordata::DownloadProgress` and `vectordata::io::*` keep working
as re-exports; `vectordata::transport` becomes a public re-export of
the crate's client API for programs that already depend on
vectordata.

## 8. Non-goals

- Async. Every client stays blocking; vecd's server stays axum.
- Any change to the cache layout, sidecar formats or file formats.
- An AWS SDK, or replacing the `aws` CLI behind push and vecd backup.
- Moving the compilers, `parquet_mnode` or `passage_table` out of
  veks-core.
- veks-studies.
- A release, a version bump or a publish (the maintainer's).

## 9. Acceptance tests

1. CB-2 passes, and each of its parts fails on 13577a2.
2. Each apicore trait's crate-doc example implements it with apicore
   alone; a `ByteSource` written that way is read by veks-io's
   `XvecReader`, and a `Credentials` written that way supplies the
   token a veks-transport request sends (AC-6).
3. A program depending on veks-io and veks-transport only reads a
   remote `.fvecs` through `XvecReader` over the transport's
   `ByteSource`, offline mode refusing it with the offline error.
4. `probe_source` on a half-downloaded `.fvecs`, a sparse cache file
   and a file with trailing garbage reports count, remainder and
   completeness for each (G1); `cleanup_cleanfvec`, `bound`,
   `veks check integrity`, ping, describe and the picker use it.
5. A slab and a parquet source of u8 vectors report a 1-byte element
   (fails today: 4); an npy of f16 reports 2.
6. The positional writer, the atomic writer, the shard-capped writer
   and `write_raw` round-trip through veks-io's readers over
   `ElementType::ALL`.
7. A streaming write that hits a full disk returns the error from
   `finish` through every path veks-core's writer served (fails today:
   panic).
8. `fetch_if_modified` against a self-signed server trusted for its
   origin succeeds (fails today).
9. veks-pipeline's HF, S2 and bulk downloaders, against a local test
   server, honour offline mode and a custom CA (fail today) and send
   their own authorisation, never a vectordata read token.
10. SigV4 signing reproduces the AWS documentation's test vector, and
   the COS listing signs as before.
11. The reader and writer benches show no streaming path slower than
    the faster of the two implementations it replaces (IO-4 method).
12. PR-2's inline guard, PR-5's rate checks and PR-8's lints pass;
    the PR-10 bench shows no regression, recorded in this SRD.
13. The whole workspace passes, clippy is clean on changed files and
    `RUSTDOCFLAGS=-D warnings cargo doc` is clean.

## 10. Questions and decisions

**Q1 — How far does the transport crate reach?**
(a) The client layer only (pool, TLS, offline, auth, retry, range and
whole-file transfers, uploads); the chunk store, merkle cache and
`Storage` stay in vectordata. (b) The client layer and remote
streaming storage: chunk store, merkle reference and state, the
verified cache channel, each a `ByteSource`, with cache paths passed
in (TR-7). **Recommended: (b).** Remote access and streaming are one
concern — a cached remote read is a range transfer, verified and kept
— and (b) is what lets a program read a remote dataset file through
veks-io without vectordata (TR-8). (a) is smaller and leaves
`chunked_http` and `cache` where every caller of them already is.
**Decided 2026-10-22:** (b): the HTTP layer and remote streaming storage (chunk store, merkle reference and state, verified cache channel), each a `ByteSource`, cache paths passed in.

**Q2 — Its name.** `veks-transport` (recommended; it matches
`veks-io`) or `veks-remote`.
**Decided 2026-10-22:** `veks-transport`.

**Q3 — Where shard naming and discovery live.** The series rule
(`__NNNN`, `dataset::shards::{shard_name, discover_shards}`) is used
by `Storage`, veks-core's sharding sink and vectordata's views.
**Recommended: veks-io**, since it is how one logical file spans
several on disk, and FL-2, FL-4 (G4, G5) and FL-6 need it; vectordata
re-exports it.
**Decided 2026-10-22:** veks-io owns shard naming, discovery and locate; vectordata re-exports them; the naming does not change.

**Q4 — `TypedReader`: remove, or re-implement over veks-io?**
**Recommended: remove** (FL-10). It is deprecated since 4.0.0, has no
caller outside its own wrappers, and the next release is already
breaking; re-implementing it would keep a second API alive for no
user.
**Decided 2026-10-22:** remove `TypedReader`, `open_facet_typed`, `TypedAccessError`, `TypedElement`; port the tests whose behaviour `facet_as` still reaches.

**Q5 — Third-party downloads and authorisation.** The pipeline's
downloaders talk to Hugging Face, Semantic Scholar and IBM COS.
**Recommended:** they take the transport's client, TLS trust, offline
gate and user agent, and only their own authorisation (TR-9). Applying
vectordata's read credentials to a third-party host could send a
dataset token where it does not belong.
**Decided 2026-10-22:** the transport's client, TLS trust, offline gate, user agent and `RetryPolicy`; each downloader's own authorisation; vectordata's read credentials never apply.

**Q6 — SigV4 in the transport?** **Recommended: yes**, as a
`RequestSigner` (TR-5): it is generic S3 request signing, and the
read side already speaks to S3 over HTTPS. The alternative keeps it in
veks-pipeline behind the same trait.
**Decided 2026-10-22:** SigV4 moves into veks-transport as a `RequestSigner`, checked against AWS's test vector.

**Q7 — `Error::Http` → `Error::Transport` in the same release as the
4.0.x breaking changes now under Unreleased?** **Recommended: yes**;
nothing matches on the variant, and its payload is a reqwest type.
**Decided 2026-10-22:** same release; `Error::Transport(TransportError)` under Breaking.

**Q8 — veks-core's hdf5 reader.** The hdf5 crate was removed; open,
probe and list return errors, and only `parse_hdf5_path` works (used
by the import wizard). **Recommended:** delete the reader and
`VecFormat::Hdf5`, keep the wizard's path parsing where the wizard is,
and let an hdf5 import say plainly that it is not supported.
**Decided 2026-10-22:** delete the stub reader and `VecFormat::Hdf5`; `parse_hdf5_path` stays with the wizard; an `.h5` import says hdf5 is not supported.

**Q9 — vecd's health probe.** **Recommended:** through the transport
with trust for its own origin (TR-11), so reqwest leaves vecd's
`[dependencies]`. The alternative keeps its own client and an entry in
CB-2's allow-list.
**Decided 2026-10-22:** through the transport, trusting only vecd's own origin and certificate.
**Decided 2026-10-22 (follow-up):** the probe stays cert-pinned but names the host the certificate was issued for, pinned to the loopback listener (`TransportConfig::resolve`): the connected address when the certificate has it, else `localhost`, else its first DNS name. A certificate issued for a `--host` name, or a CA-issued one for a public name, is then reached on loopback and still verified.

**Q10 — `veks-datatypes` and `veks-apicore`: two crates or one?**
The element model (`ElementType`, `Element`, the conversion rules) is
the most widely shared vocabulary of all, and apicore's contracts name
it. (a) Keep `veks-datatypes` as the leaf that apicore depends on and
re-exports; veks-simd keeps depending on it alone. (b) Fold it into
apicore, one crate of shared vocabulary. **Recommended: (a).** The
element model is values, not contracts; veks-simd needs it and none of
the seams; and every `veks_datatypes::` path keeps working.
**Decided 2026-10-22:** two crates; veks-apicore depends on and re-exports veks-datatypes.

**Q11 — The shared error's name.** Moved into apicore, the reader
error is no longer veks-io's alone. (a) Keep `IoError`, re-exported
from veks-io and vectordata as now. (b) Rename it (`AccessError`)
with `IoError` a deprecated alias. **Recommended: (a)**; the name
says what it is the error of, reading, and the release already breaks
enough.
**Decided 2026-10-22:** keep `IoError`, moved to apicore and re-exported where it is now.

**Q12 — Dataset-level contracts.** Should `TestDataView` (and
`CatalogSource`) move into apicore too, so a program can code against
a dataset view without vectordata, or a second implementation (vecd's,
a test double) can exist outside it? **Recommended: not now** (AC-4):
every implementor and consumer depends on vectordata, and
`TestDataView`'s signatures name a dozen vectordata types
(`FacetConfig`, `DistanceFunction`, `Quantization`, …) that would have
to move with it. Revisit when a second implementation appears.
**Decided 2026-10-22:** move them now: `TestDataView`, `CatalogSource` and the types their signatures name (§4.2, and the follow-ups Q12a–Q12d).

**Q12a — Where do the dataset contracts live?** (a) A second
contracts crate, `veks-dataset-api`, above apicore; (b) inside
veks-apicore.
**Decided 2026-10-22:** (b). apicore is the API (AC-1): one crate,
with no optional part.

**Q12b — `open_facet_storage`'s concrete `FacetStorage`.** (a) A
`FacetBytes` trait object; (b) off the contract.
**Decided 2026-10-22:** (a) (DA-3).

**Q12c — The API's error.** (a) `vectordata::Error` moves in; (b) a
narrower contract error, converted at the seam.
**Decided 2026-10-22:** (a) (DA-2).

**Q12d — `TestDataView`'s deprecated methods.** (a) The API is defined
without them; (b) they move, still deprecated.
**Decided 2026-10-22:** (a) (DA-5).

## 11. Plan

Each step is a themed commit when the maintainer asks; every guard is
shown failing on the code before it.

0. **`veks-apicore`** (AC-1..AC-6): the crate, the contracts of AC-3
   that exist today moved from veks-io and vectordata with re-exports
   at their old paths, the dataset API of §4.2 (DA-1..DA-6) with
   `Error`, `FacetBytes` and the deprecated methods dropped, `ProgressSink` unifying the progress traits,
   `ByteSource` without constructors; the transport contracts are
   added in step 7. Guard: apicore's dependencies, and the AC-2 rule. This step carries
   the dataset API's tests unchanged except for DA-5's removals.

1. **veks-io's model complete** (FL-1, FL-7, FL-8): the unified
   `VecFormat`, the conversion merge, `veks_io::file`; veks-core and
   its callers move to the names. Guard: no `VecSource`/`VecSink`/
   `VecFormat` outside veks-io.
2. **Streaming xvec merged** (FL-2, FL-3): directory reads, writeback
   control, the error from `finish`; the bench settles the buffer.
3. **The gaps** (FL-4, FL-6): `probe_source`, the positional, atomic
   and shard-capped writers, `encode_records`/`write_raw`, series
   offsets, the sharding sink; shard naming per Q3.
4. **Other formats** (FL-5): npy, parquet, slab readers behind
   features, element widths fixed; hdf5 per Q8.
5. **Inline sites** (FL-9) by crate — veks-pipeline, veks, veks-core,
   vectordata — and CB-2's format checks.
6. **`TypedReader`** (FL-10) per Q4.
7. **The transport crate** (TR-1..TR-8, TR-13): moved from vectordata
   with its tests, `TransportConfig`, `Credentials`, `RequestSigner`,
   the transport's types; vectordata re-exports (CB-3); the cache
   layout untouched (a test lists every path a fetch writes, before
   and after, and they are equal).
8. **Clients onto the transport** (TR-9..TR-12): the pipeline's
   downloaders and SigV4, `update_check`, vecd, the three defects.
9. **Dependencies and the guard** (CB-1, CB-2's network and graph
   checks: veks-transport no longer reaches veks-io).
10. **Docs**: sysref 02, the transport crate's README, the CHANGELOG
    under Unreleased, this SRD's implementation records.

## 12. Implementation

### 12.1 Step 0: the API core (2026-10-22)

`crates/veks-apicore` holds the contracts and the vocabulary they name;
vectordata and veks-io implement it and re-export it at every path they
had, so `vectordata::TestDataView`, `vectordata::dataset::*`,
`vectordata::io::VectorReader` and `veks_io::ByteSource` resolve as
before.

**What moved.**
- From veks-io: `ByteSource` and `SourcePart` (`source`), the reader
  contracts with `IoError`, `ConvertingReader` and the `Any…` readers
  (`reader`), the streaming traits (`stream`, the old `traits.rs`).
- From vectordata: the descriptor model (`dataset`, except `infer`,
  `layout` and shard discovery; `model`; `location`), the overlay
  values (`AttributeSource`, `OverlayKind`, `Stated`, and
  `Contradiction`, new), the fetch values and orchestration (`fetch`,
  without the meters and `fetch_text`), the verify report types
  (`verify`), the record contracts (`records`, without the slab
  container and the codecs), `TestDataView`, `TestDataViewExt`,
  `QuantizedBase` and the plan values (`view`), `CatalogSource`,
  `CatalogOrigin`, `CatalogDiagnostic` (`catalog`), `Error`,
  `InsufficientCacheSpace` (`error`), `ContentId`, `DownloadProgress`.
- New: `FacetBytes` (DA-3), `ProgressSink` and `NoProgress` (one sink
  for every labelled pass; `VerifyProgress` is it), `TransportError`
  and `TransportErrorKind` (TR-6's minimum, so `Error::Transport` could
  replace `Error::Http` now; step 7 adds the transport contracts).

**Settled in implementation.**
- **AC-1's dependency list** is the vocabulary's actual needs:
  `veks-datatypes`, `thiserror`, `serde`, `serde_json` (JSON catalogs),
  `serde_yaml`, `indexmap`, `url`, `regex` (profile selectors) and
  `log` (the descriptor model's parse warnings). The guard holds it
  to exactly that.
- **Return types that carried implementation state are contracts with
  thin API types over them**, as DA-3 set for `FacetStorage`:
  `FacetBytes` exposes what fetch execution needs (prebuffer by part
  and by shard range, revalidate, sizes, fills, record-to-byte
  mapping, content identity, cache check, records) and `as_any`, by
  which vectordata recovers its own handle on its cold paths.
  `RecordFacet` is an API type over `RecordContainer`s, opened through
  `FacetBytes::open_records`. `PrefetchHandle` is the API's (std
  threads only). Fetch planning and execution (`plan_view`,
  `plan_from`, `FetchPlan::execute`) are written against these and
  live in the API; the one implementation hook is `CapacityCheck`,
  which a view supplies through `TestDataView::cache_capacity`.
- **`FacetBytes` has defaults for bytes already here** (complete,
  local, nothing to fetch, no cache, no identity), so a second
  implementation writes three methods; vectordata overrides every one.
- **`verify`, `verify_report` and `normalization` are
  `TestDataView`'s**, `verify` provided over `verify_report`: the
  extension trait is a blanket impl, so behaviour an implementation
  supplies must be on the core trait (DA-4).
- **`DatasetConfig::validate` stays on the type**: it checks declared
  paths against the managed directories and touches no file. Only
  `load_from` (a URL) and `reconcile_shard_declarations` (the files on
  disk) moved to `DatasetConfigIo`.
- **The series rules split by what they touch**, refining Q3: the
  naming, declaration and addressing rules are descriptor vocabulary
  (a profile names a series) and are the API core's
  (`dataset::shards`); finding and measuring a series' files on disk
  (`discover_shards`, `observe_series`) is vectordata's for now and
  moves to veks-io in step 3. Nothing about the naming changed.
- **`CatalogEntry::apply_overlay` returns its contradictions**; the API
  does not log, vectordata warns once per dataset and key as before.
- **AC-2's guard lists what is not a contract**: `DatasetConfigIo` and
  `ElementTypeExt` (extension traits: one implementation, methods on an
  API type) and veks-io's `OpenSource` (its constructor protocol,
  PR-4).
- **PR-8's lint deny is on in veks-apicore**; nine `unwrap`/`expect`
  sites in the moved descriptor code were rewritten without them.
  veks-io takes the deny with its own steps (1–3).
- **apicore's doc examples on the dataset contracts run against
  vectordata**, a path-only dev-dependency (stripped on publish). Every
  contract also has an example implementing it with apicore alone
  (AC-6).

**Guards** (`crates/veks-apicore/tests/contracts.rs`), each shown
failing on 7c14023 before it passed:
- `the_api_core_depends_only_on_its_vocabulary` (AC-1): against
  7c14023's veks-io, where the contracts lived, it lists `arrow`,
  `byteorder`, `half`, `libc`, `memmap2`, `ndarray`, `ndarray-npy`,
  `parquet`, `slabtastic`.
- `contracts_are_the_api_cores` (AC-2): against 7c14023 it lists
  `veks-io::VectorReader`, `VvecReader`, `ByteSource`, `AnyElement`,
  `VecSource`, `VecSink` and vectordata's `TestDataView`,
  `TestDataViewExt` used by other crates.
- `hot_path_functions_are_inlinable_across_crates` (PR-2): with
  `DownloadProgress`'s accessors as they were (no `#[inline]`), it
  lists each of them.

**PR-10.** The reader bench gained `view` (reads through `Arc<dyn
TestDataView>`), `stream_write`, `remote_fetch` (a whole facet over
loopback into a fresh cache) and `remote_cached`, and ran at 7c14023
(saved as the `step0-before` baseline) and after, pinned to one core.
Of 39 cases, none is slower after the change once two artifacts are
set aside, each checked by running the old build (7c14023 with the
same bench) and the new one alternately:

| Case | Before | After | |
|---|---|---|---|
| `random_f32/get` (vectordata / veks-io) | 111.3 / 122.9 ms | 110.6 / 111.0 ms | the direct reader: unchanged and faster |
| `view/seq_read_into` | 118.0 ms | 117.6 ms | |
| `view/random_get` | 152.5 ms | 156.2 ms | see below |
| `view/open` | 10.48 µs | 10.51 µs | |
| `stream_write/sink_write_record` / `typed_xvec_writer` | 205.7 / 199.4 ms | 192.2 / 199.0 ms | |
| `remote_fetch/whole_facet` (64 MiB) | 167.0 ms | 165.7 ms | |
| `remote_cached/seq_read_into` / `random_get` | 5.77 / 17.20 ms | 5.71 / 17.17 ms | |
| `scalar/u8/as_slice_sum` (veks-io), first bench | 1.232 s | 1.540 s | see below |
| `scalar/u8/as_slice_sum`, fixed bench, old vs new | 206.9 ms | 206.7 ms | |

- `view/random_get` is 77 % `memmove` of a 512-byte record from a
  1 GiB mapping read at random (memory latency); the reader's `get` is
  4 % of samples in both builds, and the profiles have the same shape.
  Over the same data files, alternating old and new three times gave
  157.6/154.2/154.4 ms old and 157.6/156.9/157.2 ms new: inside the old
  build's own spread. The 152.5 ms baseline had read its own copy of
  the file. No change is attributable to the boundary.
- `scalar/u8/as_slice_sum/veks-io` was 25 % slower in every run while
  its `vectordata` twin, the same reader code, was not. The profile is
  the bench's own loop — load a byte, an indirect call through the
  `widen` `fn` pointer, branch — 94 % of samples, the reader 0 %; the
  loop's closing branch now straddled a 64-byte boundary. The case
  measured an indirect call per element, so its `widen` became an
  inlined closure; with that bench the old and new builds are equal
  (206.9 vs 206.7 ms) and every other scalar case is within 0.1 %.

The workspace passes (121 test binaries), clippy reports nothing new in
the changed files, and `RUSTDOCFLAGS="-D warnings" cargo doc` is clean
for veks-apicore, vectordata and veks-io.

### 12.2 Step 1: veks-io's model complete (2026-10-22)

`VecFormat`, the streaming traits and element conversion each have one
definition. veks-core's copies are deleted, not aliased, and veks-core,
veks-pipeline and veks use veks-io's names.

**What moved or changed.**
- FL-1: `veks_io::VecFormat` gains the twelve vvec variants (`Fvvec` …
  `U64vvec`, in `ALL`), `is_uniform_xvec` and `is_vvec` (`is_xvec` is
  now both), `detect_from_path`, `detect_from_directory` (public; npy,
  then parquet, then an xvec member, then slab), `record_bytes` and
  `expected_file_size`. `from_extension("fvvecs")` is `Fvvec` (it was
  `None`), `VecFormat::of(Container::Vvec, ty)` names a format, and a
  slab's `data_type_name` is `bytes`. `open_format` and
  `create_format` refuse a vvec format and name `open_varlen` and
  `create_varlen`.
- veks-core's `VecSource`, `VecSink`, `SourceMeta` and `VecFormat` are
  deleted, with the adapters between the two models (`IoSourceAdapter`,
  `IoSinkAdapter`) and their `from_extension(name()).unwrap_or(ScalarU8)`
  round trip.
- FL-7: veks-core's `formats/convert.rs` and veks-io's
  `scalar/convert.rs` are one module, `veks_io::convert`: float records
  on veks-simd's byte converters, and checked columns. veks-io now
  depends on veks-simd.
- FL-8: veks-core's `portable_io` is `veks_io::file`.
- Finding and measuring a series on disk (`discover_shards`,
  `observe_series`) moved from vectordata's `dataset::shards` to
  `veks_io::series`, as §12.1 said it would. vectordata re-exports
  both.
- Q8 landed here, not in step 4: veks-core's enum held `Hdf5`, so it
  went with the enum. The stub reader (`reader/hdf5.rs`) is deleted.
  `veks prepare import` refuses an `.h5` or `.hdf5` source, file or
  `file.h5#dataset`, before it plans anything: hdf5 is not read, so
  convert it first. The pipeline's `count:` variable says the same.
  The wizard no longer lists an HDF5 file's datasets and shows it as
  `hdf5 (not supported)`. The import's HDF5-with-separate-query
  strategy is gone.

**Settled in implementation.**
- **`parse_hdf5_path` did not stay with the wizard**, as Q8 assumed: it
  had no caller but its own test. It was deleted with the reader. The
  import's `is_hdf5_path` recognises the notation in order to refuse
  it.
- **veks-core's lib example** shows `preferred_vec_format` instead of
  `VecFormat` and `convert_elements`, which are veks-io's now.

**Guard** (`crates/veks-apicore/tests/contracts.rs`):
`the_streaming_model_is_defined_once` (FL-1). `VecSource`, `VecSink`
and `SourceMeta` are defined only in veks-apicore, and `VecFormat`
only in veks-io. On fa21340 it lists veks-core's `VecFormat`
(`formats/mod.rs`), `VecSource` and `SourceMeta` (`formats/reader`)
and `VecSink` (`formats/writer`).
`hdf5_sources_are_recognised_to_be_refused` (veks, `prepare/import.rs`)
holds the refusal.

### 12.3 Step 2: streaming xvec merged (2026-10-22)

veks-core's streaming xvec reader and writer (`formats/reader/xvec.rs`,
`formats/writer/xvec.rs`) are deleted. `open_source` and `open_sink`
open veks-io's.

**What moved or changed.**
- FL-2: `veks_io::xvec::reader::open` and `probe` read one file, a
  directory of files of the format in name order (the order `__NNNN`
  names sort in), or, when the path itself is absent, the series it is
  the unsharded name of. Each file is checked for a whole number of
  records and the same dimension as the files before it. As the stream
  moves on to the next file, what is left of the finished one's pages
  is released.
- FL-3: the byte-level xvec writer starts writeback every 64 MiB and
  releases what the cycle before wrote back
  (`sync_file_range`, then `FADV_DONTNEED`; Linux only). It still
  returns the first I/O error from `finish`.
- FL-3: a symlink at the target is an error from every veks-io writer:
  the xvec, scalar and varlen byte writers, and the typed writers
  (`create_for`). The error names the link and what it points to.

**Settled in implementation.**
- **The symlink rule**, by the user's ruling ("it is foolish to assume
  you can just remove a symlink"), is in FL-3 itself. veks-core's
  writer removed a link silently. That is not carried over.
- **The buffer stays 16 MiB**, by FL-3's measurement.
- **The full-disk guard no longer writes through a symlink to
  `/dev/full`**, since the writers refuse one now. It reruns itself as
  a child process with `RLIMIT_FSIZE` at 1 MiB and `SIGXFSZ` ignored,
  so the limit touches nothing else in the test process.

**Guards** (`crates/veks-io/tests/streaming.rs`, new):
- `a_directory_of_shards_reads_as_one_stream` and
  `a_series_opens_by_its_unsharded_name` (FL-2): veks-io's step-1
  reader mapped only the path it was given.
- `shards_of_another_dimension_are_refused`.
- `a_symlink_at_the_target_is_refused`: every writer, and the link and
  the file it links to left as they were. veks-io's step-1 writers
  opened the path with `File::create` and wrote through the link.
- `a_file_written_across_writeback_cycles_reads_back_whole`: 70 MiB,
  past one cycle.
- In `tests/writers.rs`, `a_full_disk_is_an_error_from_every_writer`
  runs `writes_past_the_file_size_limit_fail` under the limit.

**PR-10.** `stream_write/sink_write_record` (256 MiB of dim-128 f32):
183.6 ms at a 16 MiB buffer and 182.5 ms at 4 MiB (FL-3).

### 12.4 Step 3: the gaps (2026-10-22)

The rest of veks-core's writing side moved into veks-io: the sinks,
the sharding sink, the positional writer, the slab reader and writer,
and the pipeline's atomic writer and symlink guard. veks-core's
`formats::writer` is gone.

**What moved or changed.**
- G1: `veks_io::reader::probe_source(&impl ByteSource, ElementType)`
  returns an `XvecProbe` (`dim`, `record_bytes`, `records`,
  `trailing_bytes`, `complete`). It reads one header, so it describes
  a partial, sparse or truncated source instead of refusing it.
- G2: veks-core's `mmap_xvec` is `veks_io::xvec::positional::PositionalXvecWriter`.
- G3: `veks_io::writer::encode_records` packs rows into a caller's
  buffer. `XvecWriter::write_raw` writes records already in the file's
  layout, and checks that every header is the file's dimension and that
  the bytes are whole records.
- G4: `veks_io::atomic` (`AtomicWriter`, `temp_path_for`,
  `guard_against_symlink_write`, `safe_create_file`, `safe_write`,
  `safe_copy`), moved from veks-pipeline's `atomic_write.rs`, which now
  re-exports it. The shard-capped writer is `veks_io::sharded`
  (`ShardedSink`, `ShardedOutcome`, `ShardOpener`).
- G5: `veks_io::reader::load_series_offsets` returns a vvec series'
  offsets in the series' joined byte space.
- FL-6: `open_sink`, `open_sharded_sink`, `SinkConfig` and
  `SLAB_RECORD_FRAMING` are `veks_io::sink`. Shard naming stays the API
  core's (`dataset::shards::shard_name`, `shard_source_spec`).
- The slab reader and writer are `veks_io::slab` (feature `slab`).
  veks-core and veks-pipeline enable the feature.
- `metadata_schema` (`MetadataSchema`, `PredicateSchema`, the slab
  namespace names) moved from vectordata to veks-apicore, because
  veks-io's slab writer seals a slab with a schema sidecar and veks-io
  does not depend on vectordata. `vectordata::metadata_schema`
  re-exports it.
- veks-core's shared test samples moved to `veks-io/tests/support`.

**Settled in implementation.**
- **`guard_against_symlink_write` errors.** The pipeline's version
  printed "removing symlink" and removed it, in twelve files. Now
  every writer refuses a symlink, by the user's ruling, and the byte
  writers' `refuse_symlink` is this guard.
- **The process that made a link removes it.** Before any step writes,
  the import's `unlink_superseded_identity_links` removes the Identity
  links that it created itself and that a new plan materializes. It
  removes only symlinks and never a file, and it prints each removal.
  This is FL-3's paragraph, amended in this step.

**Guards:**
- `crates/veks-io/tests/sinks.rs` (new), over every uniform vector and
  scalar format, read back at the format's own type:
  `open_sink_writes_every_vector_and_scalar_format_that_reads_back_at_its_type`,
  `the_sharded_sink_writes_every_vector_and_scalar_format_across_shards`,
  `the_positional_writer_writes_every_uniform_vector_format`.
- `every_writer_refuses_a_symlink_target` (`veks_io::atomic`), and the
  guard's doc example. The pipeline's guard removed the link.
- `a_materialized_artifact_unlinks_the_previous_identity_link` (veks,
  `prepare/import.rs`): the import's own link is removed, and a
  regular file at another output and the linked source are left alone.
- `metadata_schema.rs` joins veks-apicore's cold-module list
  (`contracts.rs`, PR-2).

### 12.5 Step 4: other formats (2026-10-22)

The npy and parquet vector readers moved from veks-core to veks-io,
behind veks-io's `npy` and `parquet` features. `veks_io::open_format`
now opens npy, parquet and slab, where it used to return "not yet
implemented". veks-core's `formats::reader` keeps only `parquet_mnode`
and the dispatch by facet.

**What moved or changed.**
- `SourceMeta.element_type` and `VecSource::element_type()` (default
  `None`) are new in veks-apicore. The xvec, scalar, npy and parquet
  readers fill them in. A slab of opaque records and `parquet_mnode`
  state none.
- The parquet reader finds the first `List`, `LargeList` or
  `FixedSizeList` column of any numeric element type and reads it at
  that type. veks-core's took only `f32` lists and reported 4 bytes.
  Before, a `FixedSizeList` column was found and then decoded as a
  `List`, so every row was missing.
- npy states its type (`F16`, `F32`, `F64`). Its width was already
  right.
- The import and `transform convert` take the element type from the
  format, then from the source, and only then from the width
  (`element_type_of_width`), so a same-width pair (`u8`/`i8`) is no
  longer guessed.
- `scan_npy_manifest`, `scan_npy_headers`, `load_npy` and
  `NpyArrayData` are `veks_io::npy`.

**Settled in implementation.**
- **hdf5 was done in step 1** (§12.2).
- **A slab of vectors declares its element type in its schema
  sidecar** (decided 2026-10-22; acceptance test 5's slab case is
  met). A third descriptor kind, `VectorSchema` (`kind: "vector"`,
  `element_type` by `ElementType::name`), joins `MetadataSchema` and
  `PredicateSchema` in the `:schema` namespace; the file layout, the
  slab container and the cache layout are unchanged. A slab that
  declares none — no sidecar (every slab written before), a metadata
  or predicate one, or one that is not JSON — reads as bytes:
  `element_type` `None`, `element_size` 1, the dimension in bytes,
  and one `log::warn!` per opened file saying why. A vector sidecar
  that does not parse is an error, not bytes.
  - `SinkConfig::source_format` is replaced by `element_type`: the
    type of the records written, which a slab sink declares and sizes
    from (`None` = opaque, sized as bytes). It was only ever read for
    the slab width, and guessed `f32` for npy and parquet.
  - `SlabReader::probe` is the one slab probe (veks-core's
    `probe_slab`, which assumed 4 bytes, delegates to it), and
    `veks_io::probe_format` answers for a slab.
  - `metadata_schema::sidecar_kind` is the one reading of a sidecar's
    `kind` (a kindless one is metadata).
  - `transform convert` writes a declared slab from any typed source;
    from a slab that declares none it copies only into a slab (still
    undeclared) and refuses a typed target, which before was written
    at a guessed `f32`. veks-pipeline's `ImportFacetOp`, compiled but
    never registered as a command, is deleted as dead code (maintainer,
    2026-10-22). `element_type_of_width` is gone: no source
    needs a type guessed from its width.
  - `analyze describe` prints the stated element type, `undeclared`
    for such a slab; `slab explain` renders a vector sidecar.

**Guards** (`crates/veks-io/tests/containers.rs`, new; needs `npy` and
`parquet`):
- `a_parquet_u8_column_reads_at_its_type`: 1 byte, `U8`. veks-core's
  reader found no vector column in it.
- `a_parquet_fixed_size_list_column_decodes`.
- `an_npy_f16_file_states_its_type`: 2 bytes, `F16`.

**Guards** for the slab case (`crates/veks-io/tests/slab_vectors.rs`,
new; needs `slab`; and `crates/veks-pipeline/tests/slab_element_type.rs`):
- `a_slab_of_u8_vectors_reports_one_byte_and_u8`: 1 byte, `U8`.
  Failed before with `(0, 4, None)` for a 3-dim u8 slab.
- `an_f32_slab_reports_f32`; `an_undeclared_slab_reads_as_bytes` (no
  sidecar, metadata, kindless, garbage); `an_unreadable_vector_sidecar_is_an_error`;
  `every_element_type_round_trips_through_a_slab` (`ElementType::ALL`).
- `convert_into_a_slab_declares_every_element_type`,
  `an_undeclared_slab_is_not_converted_into_a_typed_file`,
  `an_undeclared_slab_copies_into_an_undeclared_slab`.

### 12.6 Steps 5 and 6: inline sites and `TypedReader` (2026-10-22)

The two steps ran together. They are recorded together, with one
tree.

**What moved or changed (step 5, FL-9).**
- `veks_io::xvec::record` (new) is the one place, outside the readers
  and writers, that writes down the record layout:
  `encode_record_into` (a row into a record-sized slot), `record_dim`,
  `decode_record_into`, `record_payload`, `records` (a strict
  zero-copy walk of mapped bytes that stops at the first bad record
  and writes no sidecar), `pread_record` (one record by `pread`, for
  reads that must not map the file), and `RecordHeader`, validated once
  (`append_record`, `append_records` for untyped payloads). veks-io's
  byte sink and veks-core's parquet compiler write through it.
- `XvecWriter::append(path, dim)` continues an existing file. It
  continues only a whole file of the same dimension and element type.
  It refuses trailing bytes, a symlink, a missing file and another
  type. An empty file takes any dimension.
- The inline sites of §2.3, by crate:
  - veks-pipeline: the kNN engines and their segment cache, dedup,
    extract, shuffle, sketch, metadata and predicate generation, the
    verifiers, `info file`, `describe`, `bound`, and `gen vectors`
    (which now appends through `XvecWriter::append`).
  - veks: `check integrity` (`probe_source`), the import and the
    wizard.
  - veks-core: the parquet and xvec-directory compilers.
  - vectordata: `datasets cache`, `derive`, `filter` and `ping`, and
    the explorer's picker.
  They read and write through veks-io's readers, writers, `probe_source`,
  `xvec::record`, `ElementType` and `VecFormat`.
- `cleanup_cleanfvec.rs` is deleted. It had been dead since 4f00785
  (not in the module tree), so acceptance test 4 has one caller fewer.

**What moved or changed (step 6, FL-10).** `TypedReader`,
`TypedElement`, `TypedAccessError`, `dispatch_typed!`,
`open_facet_typed` (the function and the `TestDataView` method) and
`Storage::open_url` are removed. `typed_access.rs` stays, holding only
the element-model re-exports and `ElementTypeExt`. `FacetShape::reader`
names `facet_as() / open_facet_native()`. The tests are ported to
`facet_as`, `open_vec_as` and `open_facet_native`, and none is
deleted: vectordata has 1004 `#[test]` functions before and after.
`deprecated_in_docs.rs` now also refuses the removed names in
documents. sysref 02 and 08 are updated.

**Settled in implementation.**
- **The format guard lives in `crates/tools/tests/crate_boundaries.rs`**,
  not in a root `tests/` as CB-2 says. `tools` is a default member with
  no dependencies, so a plain `cargo test` runs it. The root package is
  not a default member.
- **What the guard counts as matching an extension**: a literal
  compared (`==`, `!=`, `ends_with(`, `Some(`) or used as a `match`
  arm. Naming a file (`format!("{name}.ivec")`, a cache path's
  `"ivec"`) is not matching. veks-io and veks-datatypes (the one
  extension table) are exempt, and the allow-list is
  `analyze_checkendian.rs` alone. A header read is flagged when a
  stride appears within six code lines.
- **"No crate but veks-io defines a `VecSource`, `VecSink` or
  `VecFormat`" is read as defining the trait or the enum.** The traits
  are veks-apicore's, and `VecFormat` is veks-io's. Any crate may
  implement the traits. `parquet_mnode`, which §8 keeps in veks-core,
  and the compilers' in-memory sinks implement them. Confirmed by
  the maintainer, 2026-10-22.
- **Append refuses a cut file.** The old `gen vectors` appended onto a
  file with a partial last record, and every record after it was
  misread.
- **The byte sink validates its dimension** (1 to `i32::MAX`) at open,
  through `RecordHeader::new`. The old cast wrapped.
- **Behaviour that is stricter now**:
  - `i8` scalar columns read signed, and `u16`/`u32` unsigned. They
    were read with the wrong sign in `gen_predicate_keys` and
    `verify_predicates_sqlite`.
  - A kNN cache file whose rows are not `k` wide is refused
    (`open_knn_rows`).
  - `k = 0` is refused, because a record of no elements is not
    written.
  - `filter` matches an xvec by `VecFormat::is_xvec` instead of
    `ends_with("vec")`, and sizes a bf16 base, which its width table
    lacked.
  - `derive` turns a negative dimension into nothing, not a huge one.
  - Ordinal outputs must have an i32 vector extension.
  - Writes that panicked or ignored their errors are buffered and
    return them.
- **Sites left as they are**:
  - apicore's `shard_sizing` record-bytes arithmetic: apicore cannot
    depend on veks-io.
  - `derive`'s vvec record offsets over `SourceSpan`s: a span starts
    mid-file, and `load_series_offsets` writes sidecars.
  - `gen_extract`'s mvec SPLAT partition moves, which are raw byte
    moves.
  - The stdarch engine's `(1 + dim)` f32 slot layout, which is a kernel
    layout and not a file.
- **`check integrity` uses `probe_source`**, and its error still
  carries the bad dimension.

**Guards:**
- `crates/tools/tests/crate_boundaries.rs`:
  - `the_format_layer_is_veks_io`: 34 hits on the step-4 tree.
  - `the_streaming_model_and_formats_are_defined_once`: adds
    `VarlenSource`/`VarlenSink` to step 1's names.
  - `the_matchers_catch_what_they_are_for`: the matchers on the shapes
    they are for.
  Both checks fail on 13577a2 (acceptance test 1's format part).
- `veks_io::xvec::record`'s unit tests:
  `slots_round_trip_every_element_type`,
  `a_slot_of_the_wrong_size_or_header_is_refused`,
  `a_payload_is_the_bytes_its_header_counts`,
  `a_walk_yields_every_record_and_stops_at_the_first_bad_one`,
  `pread_reads_the_record_the_writer_wrote`.
- `an_append_continues_only_a_whole_file_of_its_dimension`
  (`tests/writers.rs`).

**Equivalence.** `gen vectors` output is byte-identical between the
old and the new build for six element types, including append runs.

### 12.7 Step 7: the transport crate (2026-10-22)

`crates/veks-transport` is a new crate, a member and a default member.
It depends on veks-apicore and not on veks-io or vectordata. veks-io
is only a dev-dependency.

**What moved or changed.**
- veks-transport, moved from vectordata's `transport/`,
  `chunked_http.rs`, `merkle/` and `cache/`:
  - `client`: `Transport` (the client pools, per-origin TLS trust, the
    offline gate and credentials), `Response`, `fetch_text`,
    `fetch_if_modified`, `normalize_remote_url`.
  - `http` (`HttpTransport`), `chunked` (`ChunkedTransport`,
    `fetch_chunks_parallel`), `retry` (`RetryPolicy`), and `pump`
    (internal).
  - `merkle`, `verified` (`CachedChannel`), and `chunk_store`
    (`ChunkStore`, `read_sidecar`).
  - `files`: `VerifiedFile` and `ChunkedFile`, the old remote `Storage`
    variants, and `RemoteSource` and `RangeSource`, which implement
    `ByteSource`.
  - `placement` (`CachePlacement`).
  - `upload`: the request builders for streaming, conditional and
    resumable uploads.
  The crate has a README, a seams table in its crate docs, and denies
  `unwrap` and `expect`.
- veks-apicore: the `transport` module holds `TransportConfig`,
  `TlsTrust`, `Timeouts`, `Credentials`, `NoCredentials`,
  `RequestSigner`, `Request`, `Method`, `Body` and `Authorization`.
  `TransportErrorKind` gains `Unreachable` and `Offline`, and
  `is_permanent` on the kind and the error. `IoError::Transport`
  carries a `TransportError` (TR-13). veks-transport joins
  `contracts.rs`'s implementation crates.
- vectordata: `src/transport.rs` is public and re-exports the client
  API (CB-3). `shared()` builds the one transport from the pure
  `config_from(setting, tls, &TransportEnv)`.
  `credentials::StoredCredentials` implements `Credentials`.
  `cache/layout::placement` decides the `.mrkl` and `.chunks` names
  (TR-7). `Storage` holds the transport's file types by value. The
  paths `vectordata::merkle`, `DownloadProgress` and `io::*` still
  work. `group`, the resolver, `endpoint`, the picker, `filter` and
  `push` use `Request` and `Response`. `io`'s sibling fetch is
  `veks_transport::fetch_sibling`.
- `MetadataSchema::to_json_bytes` and `PredicateSchema::to_json_bytes`
  return `Result<Vec<u8>, String>`, because veks-apicore denies
  `expect`.

**Settled in implementation.**
- **`cache/reader.rs` (listing and pruning the cache root) stays in
  vectordata**, though TR-1 names it. It reads `origin.json` and the
  layout, which TR-2 and principle 5 keep there.
- **The cache-space check stays in vectordata**, as fetch planning.
- **`Request` is apicore's**, because `RequestSigner` names it.
  **`Response` stays in veks-transport**, because it holds a live
  body.
- **`TransportConfig` gained `offline_hint`, `parallel_streams`,
  `download_concurrency` and `s3_region`**, which replace the moved
  code's environment reads (TR-3). The worker-count variable is now
  read once per process, not on every call.
- **Offline gating is in `send`**, unless a request is marked
  explicit. Push and endpoint requests are explicit. The picker's
  catalog read is now gated; it was not before.
- **The resolver takes its credentials from the transport**, and they
  resolve the same way.
- **Moved locks recover from poisoning** instead of panicking, and
  `MerkleRef::to_bytes` no longer uses `expect`.
- **A request that gets no answer is unreachable.** For example, a
  pooled connection to a stopped server fails with "error sending
  request", and it now counts as `Unreachable`. This fixed a flaky
  test.
- **PR-7 amended (decided 2026-10-22).** PR-7 said credentials were
  resolved once per origin, "as `apply_read_auth` effectively does
  now". The old `apply_read_auth` resolved a token for every request,
  as an in-memory longest-prefix lookup with no I/O, and credentials
  are keyed by URL prefix and reloadable, so a per-origin cache would
  send the wrong token. PR-7 now states the rule the code keeps: one
  in-memory lookup per request, never I/O. Guarded by
  `crates/vectordata/tests/credential_lookup.rs`
  (`a_read_token_is_looked_up_in_memory_per_prefix`): with the lookup
  made to read the store file, it fails with "a request read the
  credentials file".

**Guards:**
- `crates/vectordata/tests/cache_layout.rs`:
  `a_fetch_writes_the_paths_it_always_wrote`. These are the 25 paths a
  representative fetch writes: a merkle-published file, an unpublished
  one, a vvec with its rebuilt index, a scalar, a profile
  subdirectory, the kept catalog and `dataset.yaml`, and a bare URL.
  They were recorded on the code before the move and are identical
  after.
- `crates/vectordata/tests/per_origin_trust.rs` (TR-12, acceptance
  test 8): `a_trusted_self_signed_origin_is_fetched_by_fetch_if_modified`
  and `an_untrusted_self_signed_origin_is_refused`. Before the move,
  `fetch_if_modified` failed against a trusted self-signed origin with
  `InvalidCertificate(UnknownIssuer)`. This was re-confirmed against
  the step 5+6 tree.
- `crates/veks-transport/tests/remote_reads.rs` (TR-8, acceptance
  tests 2 and 3) reads a remote file with veks-io's reader and no
  vectordata:
  - `a_remote_fvecs_is_read_through_the_chunk_store`
  - `a_remote_fvecs_with_a_published_reference_is_verified`
  - `offline_mode_refuses_a_remote_read_with_the_offline_error`
  - `the_read_credentials_are_sent_and_a_request_of_its_own_is_not_overridden`
  - `a_signed_request_is_sent_as_signed`
- `crates/veks-transport/tests/hot_paths.rs`:
  `per_read_methods_are_inlinable_across_crates` (PR-2), which
  requires `#[inline]` on the per-read methods.
- `a_connection_closed_without_an_answer_is_unreachable`
  (`veks_transport::client`): it fails with the old rule.

**PR-10.** Pinned to core 2, against the `step7-before` baseline:

| Case | Before | After |
|---|---|---|
| `remote_fetch/whole_facet` | 169.3 ms | 167.5 ms |
| `remote_cached/seq_read_into` | 5.73 ms | 5.71 ms |
| `remote_cached/random_get` | 17.13 ms | 16.89 ms |
| `view/random_get` | 154.8 ms | 152.9 ms |
| `random_f32/get/vectordata` | 110.9 ms | 110.5 ms |

The only case that moved more than 3 % is `seq_read_into/f32/veks-io`,
at +5.6 % with a wide interval. That is veks-io's direct reader, which
this step did not touch. Alternating the old and new builds gave
118.26, 118.15, 118.19 and 118.47 ms, which is noise.

The workspace passes (131 test binaries), and
`RUSTDOCFLAGS="-D warnings" cargo doc` is clean.

### 12.8 Step 8: clients onto the transport (2026-10-22)

Every network client outside vectordata's own sites now sends through
a veks-transport `Transport`. SigV4 is the transport's (§10 Q6).

**What moved or changed.**
- veks-transport:
  - `sigv4` (new, public): `AwsCredentials` (its `Debug` hides the
    secret and the session token) and `SigV4Signer`, a
    `RequestSigner` (`new`, `s3`), with `sign_at(&mut Request,
    SystemTime)` for a fixed time. It depends on `hmac`.
  - `Response` implements `io::Read`, so a downloader streams the body
    in its own increments. `status_error()` turns an answer the caller
    found failed into a `TransportError`: the status, permanent or
    transient by `RetryPolicy`'s rule, and up to
    `STATUS_DETAIL_BYTES` (1 KiB) of the body as detail.
  - vectordata re-exports `sigv4` at `vectordata::transport::sigv4`.
- veks-pipeline (TR-9): `fetch_dlhf`, `fetch_s2ag` and
  `fetch_bulkdl/{mod,download,cos}` send through
  `vectordata::transport::shared()`: its pool, TLS trust, offline gate
  and user agent. Each request carries only its own authorisation:
  - Hugging Face: `bearer_or_anonymous(HF_TOKEN)`;
  - the S2 file list: `x-api-key`, otherwise anonymous;
  - S2 files and template downloads: anonymous;
  - COS: `.signed(SigV4Signer)`.
  The hand-rolled SigV4 and `CosCredentials` are gone, and
  `cos::credentials_from_env()` returns `AwsCredentials`.
  `download_file` and `download_signed_file` are one streaming
  `download_file(transport, Request, dest, listed_size, on_bytes)`.
  Progress is unchanged: 256 KiB reads, `on_bytes` after each.
  Retries are `RetryPolicy`:
  - the S2 file list: 2, 4, 8, 16 s, no jitter (the old schedule);
  - per-file downloads: `download_retry_policy(tries)`, with the
    transport's default backoff;
  - the HEAD probe: 0.5 s fixed, 10 attempts.
- `update_check` (TR-10) builds its `Transport` from the pure
  `probe_config(base)`: the process's TLS trust and offline mode,
  redirects 0, 5 s request and connect timeouts, one client, the
  descriptive user agent, anonymous. Its reqwest client is gone. It
  followed no redirect before either; what is new is the process's
  TLS trust and offline gate.
- vecd's readiness probe (TR-11) sends through a transport from the
  pure `health_probe_config(own_cert, host, addr)`: verification on,
  vecd's own `tls_cert` added to the trusted roots, no origin trusted
  unverified, 2 s timeouts, no redirects, anonymous and explicit. It
  accepted every certificate before. The host it names is the pure
  `probe_host(cert, ip)` over the certificate's subject alt names (§10
  Q9): a name is pinned to the listener's address through
  `TransportConfig::resolve`, so the handshake verifies the certificate
  for that name while the connection goes to loopback.
- TR-12: `fetch_if_modified` was fixed in step 7, and the resolver
  only calls `shared().send(Request::get(url))`, so no hand-written
  auth is left. `endpoint.rs` now checks offline: `whoami` and
  `candidate_namespaces` are refused offline, through `send`.
  `login_password`, `issue_token`, `revoke_token` and backup's
  `versions`, `manifest` and `get_object` go through `send_command`,
  which marks them explicit.

**Settled in implementation.**
- **One shared transport, with authorisation per request**, rather
  than a second transport with no credentials for third-party hosts.
  A request that names its own authorisation never asks the read
  credentials (TR-5).
- **The user agent is `vectordata/<version>`**, replacing `veks/0.14`.
  COS requests and the HEAD probe sent none before.
- **Retries are stricter**:
  - a permanent 4xx (other than 408, 425 and 429) and an offline
    refusal are not retried, in every downloader. The HEAD probe used
    to retry a 404 ten times.
  - per-file retries back off 1 s, doubling to 30 s, ±10 %, where they
    retried at once;
  - `tries = 0` means one attempt (bulk made none);
  - the S2 list also retries 408, 425 and a failed body read.
- **A blank `HF_TOKEN` sends no `Authorization` header.** It sent
  `Bearer ` before.
- **Downloader errors are `TransportError` text** ("HTTP request
  failed: HTTP 404: Not Found …"). The messages tests assert and the
  S2 user-facing messages are unchanged. "FAILED … after N attempts"
  now counts the attempts made.
- **Signer conventions**:
  - it signs `host`, `content-type`, the `x-amz-*` headers and
    `x-amz-date`;
  - `x-amz-content-sha256` only for `s3`; `x-amz-security-token` for a
    session token;
  - S3 paths are signed as sent, and other services' path segments
    are encoded once more, as SigV4 specifies;
  - a file body is `UNSIGNED-PAYLOAD`, so an upload is not read twice;
  - signing again replaces the signature.
- **`head_content_length` and `CosContext::signed_head` are deleted**
  as dead code.
- **The probe's trust is vecd's certificate as a root**, not a
  per-origin self-signed entry. **User decision 2026-10-22:** the probe
  is to address the name the certificate was issued for, resolved to
  loopback. A follow-up implements it.
- **The offline split in `endpoint.rs` is kept (user decision
  2026-10-22):** introspection is refused, the explicit commands are
  not (login, push and backup, per sysref 02's documented rule). One
  consequence: an offline `login --token` stores the credential
  without a user label, because the label comes from `whoami`.
- **`backup` is an explicit command throughout.** As first built,
  `run_backup` listed the readable namespaces with `whoami`, so offline
  mode stopped it before its explicit reads, against the rule. It now
  uses `endpoint::access_view`, `whoami` sent as an explicit request.
  `endpoint_offline.rs` runs a backup offline and expects exactly
  login's and backup's requests at the server; before the fix it
  failed with "backup is an explicit command: … offline mode is on".
- **hf-hub** (the optional `embed` feature) has its own HTTP stack.
  It is outside this work.

**Guards.** The integration tests below were shown failing on the
step-7 tree, with the message given. The unit tests are new with the
code they test.
- `sigv4`'s unit tests (acceptance test 10):
  - `the_aws_documentation_example_signs_to_the_published_values`:
    the IAM `ListUsers` example's canonical request, its hash
    (`f536975d…`), the string to sign, the signing key (`c4afb1cc…`)
    and the signature (`5d672d79…`);
  - `the_aws_test_suite_get_vanilla_signs_to_its_published_signature`
    (`5fa00fa3…`);
  - `s3_requests_sign_as_the_pipelines_cos_code_did`: four signatures
    captured from the old pipeline code, a listing and an object GET,
    each with and without a session token;
  - `signing_twice_leaves_one_signature`,
    `the_payload_hash_follows_the_body`, `a_debug_print_names_no_secret`,
    `dates_are_utc_civil_dates`,
    `the_canonical_query_is_decoded_encoded_strictly_and_sorted`.
- `crates/veks-pipeline/tests/downloads_through_the_transport.rs`
  (acceptance test 9), `download s2ag` and `download bulk` (template
  and COS) against a local axum server, each in a child process:
  - `offline_mode_stops_every_downloader_before_the_network`: "download
    s2ag succeeded offline: 1 downloaded…";
  - `a_trusted_ca_certificate_reaches_a_private_https_server`:
    "download s2ag refused a trusted CA certificate: file-list request
    failed: error sending request…";
  - `downloads_carry_their_own_authorisation_never_a_vectordata_token`:
    "not the transport's user agent: Seen { … }". Its "only its own
    authorisation" part cannot fail on step 7, whose downloaders had
    clients of their own. It was shown by mutation (the S2 list given
    the default read credentials): "a vectordata read token reached a
    third-party host".
- `fetch_dlhf`'s unit tests: `hub_requests_carry_only_their_own_token`,
  `offline_mode_refuses_the_hub`, `a_blank_hf_token_sends_none`.
- `crates/vectordata/tests/endpoint_offline.rs`:
  `offline_mode_refuses_endpoint_introspection_before_any_request`
  (with `login` still reaching the server): "whoami is refused
  offline: Object {"namespaces": Array []}".
- `crates/vecd/tests/health_probe_trust.rs`:
  `the_readiness_probe_trusts_its_own_certificate_and_refuses_another_origin`,
  through the binary's `daemon status --wait`: "the probe accepted a
  self-signed certificate that is not vecd's own, at another origin".
  vecd's unit test `the_health_probe_trusts_only_its_own_certificate`
  checks the configuration.
- `update_check`'s unit tests:
  `the_probe_follows_no_redirect_within_five_seconds` and
  `the_tag_is_read_from_the_unfollowed_redirect` (and, offline, no
  tag).

**PR-10.** `download bulk`, 8 × 1 GiB from a loopback axum server,
with 250 ms resolution from the progress ticker:

| Case | Before | After |
|---|---|---|
| `download bulk`, median | 4.00 s (about 2050 MiB/s) | 3.50–3.75 s (about 2340 MiB/s) |

The workspace passes (134 test binaries).

### 12.9 Step 9: dependencies and the guard (2026-10-22)

**What changed (CB-1).**
- vectordata and vecd have reqwest in `[dev-dependencies]` only; their
  tests stand up and probe servers with it. The manifests' old reqwest
  comments are kept, under a new lead-in line.
- veks drops its unused reqwest.
- veks-pipeline drops reqwest, and tokio moves to
  `[dev-dependencies]` (`rt`, `rt-multi-thread`, `net`, `macros`). Its
  axum, axum-server and rcgen dev-dependencies came in step 8, for the
  downloader tests.
- veks-transport has veks-io as a dev-dependency only, as since step
  7.

**Settled in implementation.**
- **The graph guard is a layer table** (`LAYERS`), lowest first. A
  crate's `[dependencies]` name only workspace crates on lower layers;
  skipping layers is allowed, and a crate on no layer is an error.
  veks-studies, veks-anode and tools sit on the bottom layer, and the
  root package on the top.
- **Dev-dependencies are exempt**: a test may stand up anything.
- **Manifests are read line by line** by the test, not through
  `cargo metadata`: a section header, then the `name = …` keys at the
  top level of a dependency table, `[target.….dependencies]` included.
- **hf-hub is unseen by the guard**: it is a dependency of its own
  with its own HTTP stack, and names no `reqwest::` in workspace
  source.
- CB-2's AC-2 part (no public trait of an implementation crate used
  outside it) is not in this step.

**Guards** (`crates/tools/tests/crate_boundaries.rs`, acceptance test
1's network and graph parts):
- `the_dependency_graph_is_the_srds`: every crate on a layer, edges to
  lower layers only; only veks-transport depends on reqwest;
  veks-transport and veks-io do not reach each other, and neither
  reaches vectordata; veks-apicore's dependencies are within its AC-1
  list (`APICORE_DEPENDENCIES`). On the step-7 tree it fails with
  vecd, vectordata, veks and veks-pipeline depending on reqwest. The
  transport and apicore rules were shown by mutation (veks-io added to
  veks-transport's dependencies, memmap2 to veks-apicore's):
  "veks-transport depends on veks-io, which is not on a lower layer",
  "veks-transport reaches veks-io", "veks-apicore depends on memmap2,
  outside its vocabulary".
- `only_veks_transport_names_reqwest`: no production source outside
  veks-transport names `reqwest::` or `use reqwest`. On the step-7
  tree it lists every site in `vecd/src/cli.rs`, `update_check.rs`,
  `cos.rs`, `download.rs`, `fetch_bulkdl/mod.rs` and `fetch_dlhf.rs`.
- `the_graph_matchers_catch_what_they_are_for`: the manifest reader
  (comments, multi-line tables, target and `[dependencies.<name>]`
  tables, dev-dependencies) and each graph rule, on the shapes they
  are for.

No PR-10 numbers: the step changes manifests and a test. The
workspace passes (134 test binaries; 135 with the PR-7
`credential_lookup` test).

### 12.10 Follow-ups (2026-10-22)

Decisions made while closing out steps 7–9, each implemented.

- **PR-7 amended** to one in-memory token lookup per request (§12.7).
- **vecd's probe** verifies vecd's own certificate for a name it was
  issued for: the connected loopback address when the certificate
  names it, else localhost or its first DNS name, pinned to the
  listener's address (`TransportConfig::resolve`). Guard:
  `the_readiness_probe_reaches_a_certificate_issued_only_for_a_dns_name`
  (before: "vecd did not answer /healthz within 2s"), with
  `the_readiness_probe_refuses_another_certificate_for_the_same_name`
  keeping the trust strict.
- **Slab vectors declare their element type** in a `kind: "vector"`
  schema sidecar (`VectorSchema`); see §12.5.
- **`ImportFacetOp`**, never registered as a command, is deleted.
- **Streaming sources report failures.** `VecSource::next_record`,
  `read_record_into` and `VarlenSource::next_record` return
  `Result<_, IoError>`; `Ok(None)` ends a stream and an error stops it
  for good. The npy and parquet directory readers (and veks-core's
  `ParquetMnodeReader`) skipped a file that failed to load with an
  info-level log, so a stream ended early with no error; they now
  return the file's error at its place in the order. The xvec, varlen,
  scalar and slab sources return what they turned into `None`.
  Drivers: `transform convert` (both stream paths) and veks'
  `probe_directory_vectors`. Guards (`veks-io/tests/stream_failures.rs`,
  `veks-pipeline/tests/convert_source_failures.rs`), each failing on
  the old code: "the stream ended after 4 of Some(12) records with no
  error" for npy and parquet, and `transform convert` reporting
  "converted 4 records (npy -> slab)" as success.
  - **Found by the guard:** a healthy multi-file parquet directory
    streamed only its first file (the consumer waited for the file
    `open` had preloaded). Fixed, with slot ownership made explicit;
    `a_parquet_directory_streams_every_file` failed with 4 of 12.
  - **Stricter:** parquet refuses null rows, rows of the wrong length
    and a later file of another schema (before: garbage, a panic or an
    early end); varlen refuses a negative header and a short tail; a
    varlen record of dimension 0 reads as empty instead of ending the
    stream.
  - **PR-10, deferred by the maintainer:** sequential u8 `next_record`
    is about 8 % slower (332.6 → 360–364 ms) because
    `Result<Option<Vec<u8>>, IoError>` is 64 bytes against
    `Option<Vec<u8>>`'s 24 (`IoError`'s `ValueOverflow` and `Transport`
    payloads). `read_record_into`, the hot path, is within 3 %
    (f32 −1.0 %, u8 +1.4–2.8 %, i32 +2.6 %; varlen scan +1.3 %).
    Boxing the large payloads would recover it; not done now.
- **veks-io denies `unwrap` and `expect`** (PR-8): lock poisoning is
  recovered, the slab reader's page lookup returns an error, and the
  unchecked `get_slice`'s documented panic carries a reasoned allow.
- **`backup` is explicit** under offline mode (§12.8).
