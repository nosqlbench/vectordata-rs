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

**PR-7 — Credentials are resolved once per origin.** A `Credentials`
call is not made per range request: the transport resolves a token
per origin and request kind, and keeps it for the connection pool's
life, as `apply_read_auth` effectively does now.

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
