# Storage / transport factoring for vector and typed-scalar readers

> **Status**: design proposal, not yet implemented.
> **Origin**: identified while debugging an order-of-magnitude
> performance regression in the nbrs CQL-vector workload's
> filtered-knn rampup; documented after a stop-gap fix that
> patched one cell of the underlying matrix problem.

## Summary

Vectordata's reader code is shaped as a 2D matrix of
*data-shape* × *transport*. Each cell is a separate concrete
type, with its own copy of the transport logic. Adding a new
shape or a new transport multiplies cells, and forgetting one
cell silently downgrades performance for the affected access
path — exactly what happened with `metadata_content` reads
(typed-scalar shape × cached-merkle transport) running thousands
of times slower than `base_vectors` reads (vector shape ×
cached-merkle transport) against the same remote dataset.

The fix is to make data-shape and transport orthogonal: hoist
the transport+cache behaviour into a single `Storage`
abstraction that knows about *bytes only*, and reduce the readers
to pure shape adapters parameterised on `Storage`. One transport
implementation, multiple shape adapters, no enumeration.

## The current shape

Today's reader types, by file:

| | mmap (local file) | direct HTTP | merkle cache + mmap-promote |
|---|---|---|---|
| **uniform-vector**  | `MmapVectorReader` (`io.rs`) | `HttpVectorReader` (`io.rs`) | `CachedVectorReader` (`cache/reader.rs`) |
| **typed-scalar/vector** | `TypedBackend::Mmap` (`typed_access.rs`) | `TypedBackend::Http` (`typed_access.rs`) | `TypedBackend::Cached` (`typed_access.rs`)* |

\* The third typed cell was missing entirely until very recently;
patched in as a stop-gap, but it duplicates the
`CachedVectorReader` pattern instead of sharing it.

Each cell carries a private copy of:

- The constructor logic (open path / fetch `.mref` / build
  `CachedChannel` / promote to mmap when complete).
- The `read_bytes(offset, len)` dispatch.
- The eager-init mmap check after a slow-path read.
- The `.mref` URL derivation (`format!("{}.mref", url.as_str())`).
- The cache-directory layout (via `cache_dir_for_url`).

The dispatch logic on the consumption side is also forked: every
caller has to know which reader type it has and which transports
that reader supports. `view::open_facet_typed` (used by
`metadata_value_at`, `predicate_value_at`, generic typed reads)
currently bypasses the cache because its constructor reaches into
`TypedReader::open_url`'s pre-stop-gap shape, which knew nothing
about `.mref` files.

## The cells that diverged

Two real symptoms produced by the matrix:

1. **`TypedReader::open_url` had no cache integration at all**
   until the recent stop-gap. Every `get_value(idx)` issued a
   blocking HTTP RANGE request for `native_width` bytes — for
   `metadata_content.u8`, that's a separate HTTP request *per
   single byte read*. With 100 worker fibers, throughput hit a
   wall around 13 ops/s against a 3ms p99 underlying CQL latency:
   the reader, not Cassandra, was the bottleneck. The
   uniform-vector path against the same dataset (same DNS, same
   server) ran at ~830 ops/s.

2. **`OpenableElement::open_remote` had the cache wired up** in
   the `io.rs` macro that materialises one impl per element
   type. So `vector_at`, `query_vector_at`,
   `neighbor_indices_at`, etc. *passively warmed* the cache
   through normal per-cycle reads (each `channel.read` downloads
   the chunk holding that offset and writes it to disk; once
   every chunk is verified, `try_init_mmap()` flips the reader to
   zero-copy mmap mode). Same merkle file, same cache directory,
   same chunk layout, same `OnceLock<Mmap>` promotion — but a
   different reader struct, with a copy of the same logic.

The first cell got fixed by adding a `TypedBackend::Cached`
variant that mirrors `CachedVectorReader`. That works, but it
doubles the count of places where merkle-cache + mmap-promotion
logic lives. The next reader shape someone adds will repeat the
same dance.

## The orthogonal factoring

The shape of the data (record-headed xvec, header-less scalar,
variable-length record list) is a *consumption-side* concern. The
mechanism for getting bytes from offset `o` for length `len`
(mmap a local file, blocking HTTP RANGE, merkle-verified cache
channel that auto-promotes to mmap) is a *transport-side*
concern. They share no logic and no constraints; the matrix is
the wrong factoring.

### Proposed `Storage` abstraction

```rust
// vectordata/src/storage.rs (new module)

pub enum Storage {
    /// Local file, mmap'd. Zero-copy, no fallback.
    Mmap(memmap2::Mmap),

    /// Remote URL, direct HTTP RANGE per read. Used when no
    /// `.mref` is published. No accumulation, no mmap promotion.
    Http {
        client: reqwest::blocking::Client,
        url: url::Url,
        total_size: u64,
    },

    /// Remote URL with a published `.mref`. Reads route through
    /// `CachedChannel`, which downloads + merkle-verifies chunks
    /// on demand and writes them to disk. When every chunk is
    /// verified, `try_init_mmap` switches subsequent reads to
    /// zero-copy mmap.
    Cached {
        channel: CachedChannel,
        mmap: OnceLock<memmap2::Mmap>,
    },
}

impl Storage {
    /// Open from a path-or-URL string. Local paths get `Mmap`.
    /// Remote URLs try `Cached` first (fetches `.mref`), fall
    /// back to `Http` when no `.mref` exists.
    pub fn open(source: &str) -> Result<Self>;

    /// Open a remote URL, forcing the direct-HTTP path.
    /// Internal — used by tests and by the cache-fallback
    /// branch of `open`.
    pub fn open_http_direct(url: Url) -> Result<Self>;

    /// Open a remote URL with the cache-aware path. Errors if
    /// no `.mref` exists.
    pub fn open_cached(url: Url, cache_root: &Path) -> Result<Self>;

    /// Read `len` bytes at `offset`. Always succeeds for an
    /// in-bounds offset — slow-path downloads chunks if needed.
    pub fn read_bytes(&self, offset: u64, len: u64) -> Result<Vec<u8>>;

    /// Borrow a slice if available without allocation. Returns
    /// `Some` for `Mmap` and for `Cached` once the file is
    /// fully promoted to mmap. Returns `None` otherwise — the
    /// caller falls back to `read_bytes`.
    pub fn mmap_slice(&self, offset: u64, len: u64) -> Option<&[u8]>;

    /// Total size of the underlying resource in bytes.
    pub fn total_size(&self) -> u64;

    /// Whether all bytes are locally accessible. Always `true`
    /// for `Mmap`. `true` for `Cached` once every chunk is
    /// verified. Always `false` for `Http`.
    pub fn is_complete(&self) -> bool;

    /// Force-download every chunk into the local cache. No-op
    /// for `Mmap` (already local). For `Cached`, calls
    /// `channel.prebuffer()` and promotes to mmap. For `Http`,
    /// either no-op or `Err` (no cache to populate) — see open
    /// question below.
    pub fn prebuffer(&self) -> Result<()>;
}
```

### Shape adapters reduced to dumb wrappers

```rust
// vectordata/src/io.rs (collapsed)

pub struct VectorReader<T> {
    storage: Arc<Storage>,
    dim: usize,
    count: usize,
    elem_size: usize,
    _phantom: PhantomData<T>,
}

impl<T: VvecElement> VectorReader<T> {
    pub fn open(source: &str) -> Result<Self> {
        let storage = Arc::new(Storage::open(source)?);
        // probe dim header through storage.read_bytes(0, 4)
        // count = (storage.total_size() - 4) / record_bytes
        Ok(Self { storage, dim, count, elem_size: T::ELEM_SIZE, ... })
    }

    pub fn get(&self, index: usize) -> Result<Vec<T>> {
        let offset = (index * (4 + self.dim * self.elem_size) + 4) as u64;
        let len = (self.dim * self.elem_size) as u64;
        // Try mmap first, fall back to read_bytes.
        if let Some(slice) = self.storage.mmap_slice(offset, len) {
            return Ok(slice.chunks_exact(T::ELEM_SIZE).map(T::from_le_bytes).collect());
        }
        let bytes = self.storage.read_bytes(offset, len)?;
        Ok(bytes.chunks_exact(T::ELEM_SIZE).map(T::from_le_bytes).collect())
    }

    pub fn prebuffer(&self) -> Result<()> { self.storage.prebuffer() }
    pub fn is_complete(&self) -> bool    { self.storage.is_complete() }
    // dim, count, etc.
}
```

```rust
// vectordata/src/typed_access.rs (collapsed)

pub struct TypedReader<T: TypedElement> {
    storage: Arc<Storage>,
    native_type: ElementType,
    native_width: usize,
    is_scalar: bool,
    dim: usize,
    count: usize,
    _phantom: PhantomData<T>,
}

impl<T: TypedElement> TypedReader<T> {
    pub fn open(source: &str, native_type: ElementType) -> Result<Self> {
        if T::width() < native_type.byte_width() {
            return Err(Narrowing { ... });
        }
        let storage = Arc::new(Storage::open(source)?);
        // is_scalar from URL/path extension
        // dim/count via storage.read_bytes(0, 4) for record-headed shapes
        // or (total_size / native_width) for scalar shapes
        Ok(Self { storage, native_type, ..., _phantom: PhantomData })
    }

    pub fn get_value(&self, ordinal: usize) -> Result<T> {
        let offset = ...;
        if let Some(slice) = self.storage.mmap_slice(offset, self.native_width as u64) {
            return T::from_i128(read_native_value(slice, self.native_type))
                .ok_or_else(|| ValueOverflow { ... });
        }
        let bytes = self.storage.read_bytes(offset, self.native_width as u64)?;
        T::from_i128(read_native_value(&bytes, self.native_type))
            .ok_or_else(|| ValueOverflow { ... })
    }

    pub fn prebuffer(&self) -> Result<()> { self.storage.prebuffer() }
    pub fn is_complete(&self) -> bool    { self.storage.is_complete() }
}
```

Both adapters are now ~30 lines of pure shape logic with no
transport-awareness. They share the `Arc<Storage>` so multiple
adapters can sit on the same backing data without re-opening
(useful for views that expose the same file under multiple typed
lenses).

### What gets deleted

- `MmapVectorReader` (collapse into `VectorReader<T>` over `Storage::Mmap`).
- `HttpVectorReader` (collapse into `VectorReader<T>` over `Storage::Http`).
- `CachedVectorReader` (collapse into `VectorReader<T>` over `Storage::Cached`).
- `TypedBackend` enum (replaced wholesale by `Storage`).
- `OpenableElement` trait (no longer needed — type-specific impls
  collapsed into a single `VectorReader::<T>::open`). The macro
  expansion that materialises one `OpenableElement` impl per
  element type goes away.
- The duplicated `.mref` fetch / `cache_dir_for_url` /
  `try_init_mmap` patterns scattered across the three concrete
  cached readers. One copy lives on `Storage::open_cached`.

## Then `prebuffer` finally works

`dataset_prebuffer` (the GK node in nbrs that's supposed to fully
download a dataset before the workload starts) currently *does
nothing* — it just resolves the dataset manifest. The reason it
gets away with that today is the passive-warm property of
`CachedVectorReader`: per-cycle vector access downloads chunks on
demand. So oracles workloads "feel prebuffered" even though
nothing was prebuffered.

This works for vectors only because the cache layer is wired in.
The typed-scalar metadata path bypasses the cache; per-cycle
access neither downloaded chunks nor warmed anything; throughput
collapsed.

With the orthogonal factoring, `prebuffer()` lives on `Storage`,
so a real prebuffer call works for any reader shape:

```rust
// vectordata/src/view.rs

pub trait TestDataView: Send + Sync {
    // ... existing facet getters ...

    /// Force-download every facet declared by the profile config
    /// to the local cache. Each facet's `Storage::prebuffer()`
    /// runs in turn; cached storages download + verify chunks
    /// then promote to mmap. Local-mmap and direct-HTTP storages
    /// no-op.
    fn prebuffer_all(&self) -> Result<()> {
        for (name, _desc) in self.facet_manifest() {
            // open through the typed path (covers both vector-shape and scalar
            // facets uniformly) and call prebuffer on the underlying storage
            let elem = self.facet_element_type(&name)?;
            let reader = self.open_facet_typed::<i64>(&name, elem)?;
            reader.prebuffer()?;
        }
        Ok(())
    }
}
```

And on the consumer side (nbrs):

```rust
// nbrs-variates/src/nodes/vectors.rs::DatasetPrebuffer::eval
let group = load_dataset_group(source)?;
let (_, profile) = parse_source_specifier(source);
if let Some(view) = group.profile(profile) {
    view.prebuffer_all()?;     // really does it now
}
```

After this, the workload sees a deterministic "fully resident,
zero-copy from cycle 0" mode for every facet, regardless of
shape — what users assume `dataset_prebuffer` already does.

## Migration plan

The collapse is mostly mechanical but the public API touches both
directions (the typed-access API and the vector-IO API). Suggested
sequencing:

1. **Land `Storage`** as a new module without removing anything.
   Implement all three variants. Tests are pure
   round-trips: open, read, mmap_slice, prebuffer, is_complete
   against a fixture file (local) and a fixture HTTP server (the
   existing `MemoryTransport` test harness covers this).

2. **Rewire `CachedVectorReader` internally** to delegate to
   `Storage::Cached`. Keep the public type and its methods
   (`open`, `prebuffer`, `is_complete`, `cache_path`,
   `valid_count`, etc.) so external callers don't break. This
   step is invisible from the outside.

3. **Rewire `TypedReader` internally** to use `Arc<Storage>` in
   place of `TypedBackend`. Public methods stay the same. Drop
   `TypedBackend` (private type, no API impact).

4. **Add `prebuffer_all()` to `TestDataView`** with a default impl
   that walks the facet manifest. Per-impl overrides where the
   underlying API can be more efficient (e.g. parallel
   prebuffer).

5. **Collapse `MmapVectorReader` / `HttpVectorReader` /
   `CachedVectorReader` into a single `VectorReader<T>`** parameterised on
   `Storage`. This is the breaking step — `OpenableElement`'s
   per-type impl and the three concrete reader types disappear.
   Provide type aliases for one or two release cycles
   (`pub type CachedVectorReader<T> = VectorReader<T>`) so
   downstream consumers (nbrs, anything else using these) don't
   break atomically.

6. **Drop the type aliases** in the next major release.

Steps 1–4 are non-breaking and can ship in a point release.
Steps 5–6 want a major bump.

## Test plan

The structural assertion the refactor needs to preserve, in plain
English: *opening any reader through any source, then doing N
random reads, must produce identical bytes regardless of which
storage variant backs it.* That's table-testable.

```rust
#[test]
fn storage_variants_agree_on_random_reads() {
    let fixture = "test-data/sift1m_subset/base_vectors.fvec";
    let url = test_server.serve(fixture);
    let mut local  = Storage::Mmap(mmap_of(fixture));
    let mut direct = Storage::open_http_direct(url.clone()).unwrap();
    let mut cached = Storage::open_cached(url.clone(), &tempdir).unwrap();

    for &(off, len) in &[(0, 4), (4, 128), (1<<20, 256), (last_chunk_offset, 64)] {
        let a = local.read_bytes(off, len).unwrap();
        let b = direct.read_bytes(off, len).unwrap();
        let c = cached.read_bytes(off, len).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }
}
```

Plus the same shape against a scalar `.u8` file to nail the
header-less case, and the prebuffer-then-mmap promotion test:

```rust
#[test]
fn cached_promotes_to_mmap_after_prebuffer() {
    let s = Storage::open_cached(url, cache_root).unwrap();
    assert!(s.mmap_slice(0, 16).is_none());      // pre-prebuffer: HTTP fallback
    s.prebuffer().unwrap();
    assert!(s.is_complete());
    assert!(s.mmap_slice(0, 16).is_some());      // post-prebuffer: zero-copy
}
```

## Open questions

- **`Storage::Http::prebuffer` semantics.** What does it mean to
  prebuffer a non-cacheable resource? Two options:
  (a) silent no-op so callers don't have to special-case it,
  (b) `Err(NotCacheable)` so callers see that prebuffer's intent
  was unmet. Recommendation: (b), with a sibling `try_prebuffer()`
  that returns `Ok(false)` for the no-op case. The default
  `TestDataView::prebuffer_all` calls `try_prebuffer` so a
  manifest with one Http and one Cached facet partially succeeds.

- **`Storage::open` policy for ambiguous sources.** Today,
  `io::open_vec` dispatches on `http://` / `https://` prefix and
  treats everything else as a path. `Storage::open` should match
  that exactly to preserve existing semantics. For URL schemes
  added later (`s3://`, `gs://`) the matcher extends naturally.

- **Whether `Storage` should be `Arc`-shared by default.** Both
  reader shapes can pass an `Arc<Storage>` so multiple typed views
  can share one backing storage (and one cache file) for the same
  facet. This matters when, e.g., a workload opens
  `metadata_content` once as `i32` and once as `i8` for two
  different consumers. Recommendation: yes, `Arc<Storage>`
  internally; `Storage::open` returns `Storage` and callers wrap
  in `Arc` if sharing is wanted.

- **Error type unification.** `IoError` and `TypedAccessError`
  carry overlapping Io-string variants today. With one storage
  layer, there's room to converge on a single error enum. Not
  in scope of this refactor — flag for cleanup but don't block on
  it.

- **`OpenableElement` impl macro.** The `impl_openable!` macro
  produces one impl per element type. Once `VectorReader<T>` is
  generic and uses `Storage`, the macro is no longer needed (the
  generic constraint `T: VvecElement` covers element-type
  validation). One fewer macro for new contributors to learn.

## Summary table

| | before | after |
|---|---|---|
| concrete reader types | 6 (3 vector × 2 transports + transport-promoting cache; mirrored on typed side) | 2 (`VectorReader<T>`, `TypedReader<T>`) — both pure shape |
| transport implementations | 3 in `io.rs` + 3 in `typed_access.rs` (duplicate) | 3 variants in `Storage` (one place) |
| `.mref` fetch + chunk caching + mmap promotion | duplicated in `CachedVectorReader` and `TypedBackend::Cached` | once, in `Storage::Cached` |
| `prebuffer()` works for | record-headed vectors only (and only when the consumer remembered to call it on the cached variant) | every shape, every facet, via `Storage::prebuffer` |
| adding a new shape (e.g. variable-length record) | redo all transports on the new shape | implement shape, get all transports for free |
| adding a new transport (e.g. S3 native client) | walk every reader type, plumb it in | add a variant to `Storage`, every reader gets it |

## Why now, not later

The asymmetry has been latent since the typed-access API was
introduced — it just hadn't bitten because no production
workload used the typed-scalar reader against a remote dataset at
high cycle counts. The fknn rampup is the first one. The
stop-gap landed for that particular reader fixed the symptom, but
re-anchored the matrix it should have collapsed: the next reader
shape someone adds (variable-length records, streaming, parallel
range fetch) will rediscover the same gap in a new cell. Doing
the orthogonal factoring now means *no future reader has to think
about transports*, and `dataset_prebuffer` finally honours its
contract for every facet shape automatically.
