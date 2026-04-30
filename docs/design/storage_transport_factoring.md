# Storage / transport factoring

How `vectordata` separates *byte transport* from *data shape* so
every reader inherits the right caching and mmap-promotion path
automatically, and downstream consumers can never accidentally pick
a slow transport variant.

## The factoring

Two orthogonal concerns:

- **Transport** — *how do bytes for offset `o..o+len` get into the
  process?* Local mmap, direct HTTP RANGE, or merkle-verified
  download with auto-promotion to mmap once every chunk is local.
- **Shape** — *how are the bytes interpreted?* Uniform-stride records
  (xvec), variable-length records (vvec), scalar arrays, typed
  values with widening conversions.

Transport is realised by the crate-private [`Storage`] enum; shape
is realised by [`XvecReader<T>`], [`IndexedVvecReader<T>`], and
[`TypedReader<T>`], each holding an `Arc<Storage>`. There is no way
for a downstream consumer to construct a shape adapter against a
specific transport variant — a single `open(source)` dispatch picks
the variant for them.

## Storage

```rust
// vectordata/src/storage.rs — pub(crate)

pub(crate) enum Storage {
    Mmap(memmap2::Mmap),
    Http   { transport: HttpTransport, total_size: u64 },
    Cached { channel: CachedChannel, mmap: OnceLock<memmap2::Mmap> },
}
```

| Method | Mmap | Http | Cached |
|---|---|---|---|
| `read_bytes(o, len)` | slice from mmap | HTTP RANGE | channel read (downloads chunks) |
| `mmap_slice(o, len)` | `Some(&[u8])` | `None` | `Some` once promoted, else `None` |
| `mmap_base()` | raw `*const u8` | `None` | raw `*const u8` once promoted |
| `is_complete()` | `true` | `false` | `true` after every chunk verified |
| `is_local()` | `true` | `false` | `true` once promoted |
| `local_path()` | `None` (caller knows the open path) | `None` | path to cache file |
| `prebuffer()` | no-op | no-op | downloads + verifies + promotes |
| `advise_sequential` / `advise_random` | madvise | no-op | madvise once promoted |
| `prefetch_range_bytes` / `release_range_bytes` | madvise | no-op | madvise once promoted |

`Storage::open(source)` dispatches on the source string:

- `"http://"` or `"https://"` prefix → `Storage::open_url`, which
  tries `Storage::open_url_cached` first (fetches `<url>.mref`,
  builds a `CachedChannel` rooted at
  [`crate::settings::cache_dir()`]) and silently falls back to
  `Storage::open_url_http` when no `.mref` exists.
- anything else → `Storage::open_path` (mmap).

`open_url_cached` is the default-cache-aware ctor; an explicit
cache root can be passed for tests. `open_url_http` is the
fallback-only ctor — direct HTTP RANGE per read, no caching, no
promotion.

## Shape adapters

```rust
// vectordata/src/io.rs

pub struct XvecReader<T> {
    storage: Arc<Storage>,
    dim: usize,
    count: usize,
    entry_size: usize,
    _phantom: PhantomData<T>,
}

pub struct IndexedVvecReader<T> {
    storage: Arc<Storage>,
    offsets: Vec<u64>,
    elem_size: usize,
    _phantom: PhantomData<T>,
}

// vectordata/src/typed_access.rs

pub struct TypedReader<T: TypedElement> {
    storage: Arc<Storage>,
    native_type: ElementType,
    native_width: usize,
    is_scalar: bool,
    dim: usize,
    count: usize,
    _phantom: PhantomData<T>,
}
```

Each adapter:

- Has a single `open(source)` (and a few transport-explicit
  ctors: `open_path`, `open_url`, `open_auto`) that constructs an
  `Arc<Storage>` via `Storage::open*` and reads its dim/count.
- Forwards every read through `Storage::read_bytes` /
  `Storage::mmap_slice`. The mmap-fast path is automatic — readers
  don't branch on transport.
- Re-exports the storage-level `prebuffer()`, `is_complete()`,
  `advise_sequential` / `release_range` / `prefetch_pages`
  so callers don't have to keep a separate `Storage` reference.

The hot-path `XvecReader::<T>::get_slice(index) -> &[T]` (defined for
`f32`, `f16`, `f64`, `i32`, `i16`, `u8`) panics if the storage is
not mmap-backed and elides bounds checks — this is the inner-loop
form used by KNN scans and normalisation. The trait method
`VectorReader::get_slice` returns `Option<&[T]>` for the bounds-
checked safe form.

## TestDataView prebuffer integration

```rust
trait TestDataView {
    // ... facet readers ...
    fn open_facet_storage(&self, name: &str) -> Result<FacetStorage>;
    fn prebuffer_all(&self) -> Result<()>;
    fn prebuffer_all_with_progress(
        &self,
        cb: &mut dyn FnMut(&str, &PrebufferProgress),
    ) -> Result<()>;
}
```

The default `prebuffer_all_with_progress` walks `facet_manifest()`,
opens each facet's `FacetStorage` (a public, opaque handle holding
an `Arc<Storage>`), and calls `prebuffer_with_progress` on it. The
callback is fired per facet, after that facet's underlying download
completes — local-mmap and direct-HTTP facets no-op silently.

`FacetStorage::cache_stats()` returns `Some(CacheStats)` for
cached-remote facets and `None` for local/direct-HTTP, so a UI can
render a per-facet fill bar without knowing the transport.

The trait method body is **not** generic, so `Arc<dyn TestDataView>`
remains dyn-compatible (catalog APIs return it).

## Public surface

What downstream consumers see:

```
vectordata::
  Catalog → TestDataGroup → TestDataView (trait)
      base_vectors / query_vectors / neighbor_*    → Arc<dyn VectorReader<T>>
      metadata_indices                             → Arc<dyn VvecReader<i32>>
      facet(name)
      facet_element_type(name) / facet_source(name)
      facet_manifest()
      open_facet_storage(name)                     → FacetStorage
      prebuffer_all() / prebuffer_all_with_progress(cb)
      distance_function() / base_count()

  io::
      VectorReader<T>      trait (pub)
      VvecReader<T>        trait (pub)
      XvecReader<T>        canonical uniform reader
      IndexedVvecReader<T> canonical variable-length reader
      StreamReclaim        bounded-RSS scan helper
      open_vec<T>(source)  open_vvec<T>(source)
      IoError

  typed_access::
      TypedReader<T>       single concrete struct, three open ctors
      ElementType  TypedElement  TypedAccessError

  open_facet_typed::<T>(view, name)                free fn for dyn TestDataView
  CacheStats  FacetStorage  PrebufferProgress
  settings::cache_dir() / settings_path()
```

What is hidden behind `pub(crate)`:

- `vectordata::storage` (the `Storage` enum and its variants)
- `vectordata::transport` (`HttpTransport`, `ChunkedTransport`,
  `RetryPolicy`, `DownloadProgress`, `fetch_chunks_parallel`)
- `vectordata::cache` (`CachedChannel`, the merkle-verified channel)

## Why this matters

A consumer who wants a vector reader writes:

```rust
let r = vectordata::io::open_vec::<f32>(source)?;
```

…and gets the right transport every time:

- local path → mmap
- URL with published `.mref` → merkle-cached channel that promotes
  to mmap once `prebuffer()` completes (or once every chunk has
  been read on demand)
- URL without `.mref` → direct HTTP RANGE (slow, but correct)

There is no public type or function in the crate that lets a
consumer accidentally bypass the cache or pick the slow direct-HTTP
path on a URL that has a published `.mref`. The matrix-cell
omissions that motivated this factoring (typed-scalar reads
bypassing the cache; variable-length vvec having no cached path at
all) are impossible by construction once every shape adapter routes
through `Storage`.

## Per-facet dispatch and mixed-transport datasets

Source resolution in `TestDataView` is per-facet, so a single
`dataset.yaml` can mix transports without any caller-side
branching. The `view::resolve_path_str` helper recognises absolute
HTTP URLs and passes them through unchanged regardless of where
the `dataset.yaml` itself is loaded from:

- A local `dataset.yaml` may declare some facets with relative paths
  (resolved against the dataset directory → `Storage::Mmap`) and
  others with `https://…` URLs (resolved as remote → `Storage::Cached`).
- A remote `dataset.yaml` may pull facets from a different bucket
  via absolute URLs while inheriting the catalog base for its
  relative entries.

`view.prebuffer_all_with_progress` honours this per-facet dispatch
naturally — `Storage::Mmap` and `Storage::Http` no-op silently;
`Storage::Cached` downloads. The single-source-of-truth cache layout
under `vectordata::settings::cache_dir()` is the only place
downloaded bytes ever land — there is no parallel "ad-hoc copy"
layout in any consuming crate.

## Settings.yaml resolution

The cache root is resolved via the single function
[`crate::settings::cache_dir()`], which reads `cache_dir:` from
`~/.config/vectordata/settings.yaml` (or
`$VECTORDATA_HOME/settings.yaml` for tests). If the file is missing
or does not declare `cache_dir:`, the function returns
`SettingsError::NotConfigured`. The error's `Display` impl carries
ready-to-paste setup commands; callers should surface it directly
rather than silently falling back to a default path.

`veks-pipeline`'s `configured_cache_dir()` is a thin alias for
`vectordata::settings::cache_dir()`; CLI commands use the
`configured_cache_dir_or_exit()` helper which prints the error and
exits with code 2 when the cache is unconfigured.

The cache directory layout under the resolved root is
`<host>:<port>/<url-path-prefix>/<filename>` plus
`<filename>.mrkl` for the merkle state. Including the port
isolates per-test fixtures running on ephemeral ports from each
other and from production caches.
