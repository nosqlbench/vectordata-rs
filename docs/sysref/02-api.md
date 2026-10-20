# 2. API

The `vectordata` crate provides typed, unified access to vector
datasets regardless of storage location (local file or HTTP) and
record structure (uniform or variable-length). This document is the
definitive reference for external consumers.

The transport choice — local mmap, merkle-cached HTTP with
auto-promotion to mmap, or direct HTTP RANGE — is chosen for you
based on the source string. There is no public type or function in
the crate that lets a caller bypass the cache or pick the slow
direct-HTTP path on a URL that has a published `.mref`. See
[Storage / transport factoring](../design/storage_transport_factoring.md)
for the underlying design.

---

## 2.1 Quick Start

Add the dependency:

```toml
[dependencies]
vectordata = "0.25"
```

### Find and use a dataset by name

The primary access path: catalog → dataset → profile → facet →
vectors. No URLs or paths — just names:

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;

// Load configured catalogs (~/.config/vectordata/catalogs.yaml)
let catalog = Catalog::of(&CatalogSources::new().configure_default());

use vectordata::TestDataViewExt;

// Open a dataset by name → get vectors in two calls
let group = catalog.open("my-dataset")?;
let view = group.profile("default").unwrap();
let base = view.base_vectors_as::<f32>()?;
println!("{} vectors, dim={}", base.count(), base.dim());
let v: Vec<f32> = base.get(42)?;

// Or even shorter — open a profile directly
let view = catalog.open_profile("my-dataset", "default")?;
let gt = view.neighbor_indices_as::<i32>()?;
```

The type you ask for is yours to choose: a `bf16` or `u8` base reads
as `f32` through the conversion table of §2.7, and `base_vectors_as::
<u8>()` on a quantized corpus borrows the codes as they are.

### Discover available profiles and facets

```rust
let group = catalog.open("my-dataset")?;
for name in group.profile_names() {
    let view = group.profile(&name).unwrap();
    let manifest = view.facet_manifest();
    for (facet_name, desc) in &manifest {
        println!("  {} ({})", facet_name,
            desc.source_type.as_deref().unwrap_or("?"));
    }
}
```

### Low-level file access

For direct file access without catalogs — typically only useful for
testing, debugging, or building tools on top of the library:

```rust
use vectordata::io::{open_vec, open_vvec, VectorReader, VvecReader};

// Uniform vectors — local or remote, same call
let reader = open_vec::<f32>("base_vectors.fvecs")?;
let reader = open_vec::<f32>("https://example.com/dataset/base.fvecs")?;
println!("{} vectors, dim={}", reader.count(), reader.dim());
let v: Vec<f32> = reader.get(42)?;

// Variable-length vectors (vvec)
let reader = open_vvec::<i32>("metadata_indices.ivvecs")?;
println!("{} records", reader.count());
let record: Vec<i32> = reader.get(0)?;
let dim: usize = reader.dim_at(0)?;
```

Load a dataset by URL or path:

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

let group = TestDataGroup::load("https://example.com/dataset/")?;
let view = group.profile("default").unwrap();

let base = view.base_vectors_as::<f32>()?;       // Arc<dyn VectorReader<f32>>
let gt   = view.neighbor_indices_as::<i32>()?;   // Arc<dyn VectorReader<i32>>
let mi   = view.metadata_results_as::<i32>()?;   // Arc<dyn VvecReader<i32>>
```

`TestDataGroup::load` falls back to `knn_entries.yaml`
(jvector-compatible format) when `dataset.yaml` is not present.

---

## 2.1a Layers, and the call behind each command

The library is layered from the outside in; applications use the first
four:

| Layer | Type | Holds |
|---|---|---|
| Catalog | `catalog::Catalog` | names → locations; specs (`name:selector`) → datasets |
| Group | `TestDataGroup` | one dataset: its `dataset.yaml`, its profiles, selection |
| View | `TestDataView` | one profile: its facets, and the `fetch` call |
| Reader | `VectorReader`, `VvecReader`, `TypedReader`, `records` | ordinal access to one facet |
| Storage | `FacetStorage` (low-level) | a facet's bytes: cache state, range fetches |

Transport and cache internals below storage are crate-private. Nothing
in these layers prints, prompts or exits: problems are values
(`Error`, `Catalog::diagnostics`), and progress reaches a terminal only
through a sink the caller passes (`fetch::FetchProgress`,
`push::ProgressSink`).

The `vectordata` binary sits on top. Its command implementations live
in the `datasets`, `config`, `client_cli` and `shell` modules, each
labelled *CLI support* in its first doc line, and each is a thin
adapter over a library call:

| Command | Library call |
|---|---|
| `datasets precache` | `TestDataView::fetch`, `TestDataGroup::fetch`, `Catalog::fetch` |
| `datasets list` | `Catalog::datasets`, `Catalog::match_glob` |
| `datasets describe`, `datasets ping` | `Catalog::open_spec`, `TestDataView::facet_manifest`, readers |
| `datasets push` | `push::execute` |
| `cache list`, `cache prune`, `cache prune-legacy` | `cache_admin::{list_entries, prune_by_filter, prune_legacy_layout}` |
| `config get`, `config set` | `settings::{setting_value, write_setting, cache_dir}` |
| `login`, `whoami`, `ping`, `token` | `endpoint`, `credentials` |
| `backup`, `restore` | `backup::{run_backup, run_restore}` |

The full table, one row per command, is in the crate's `AGENTS.md`, and
a test fails when a command has no row there.

## 2.2 Catalog-Based Dataset Discovery

### Catalog configuration

Catalogs are configured in `~/.config/vectordata/catalogs.yaml`:

```yaml
# Each entry is an HTTP URL or local path pointing to a directory
# containing a catalog.json (which indexes all datasets under it)
- https://vectordata-datasets.s3.amazonaws.com/production/
- https://internal-bucket.s3.us-east-1.amazonaws.com/testing/
- /mnt/data/local-datasets/
```

Manage catalogs via the CLI:

```bash
veks datasets config catalog add https://example.com/datasets/
veks datasets config catalog list
veks datasets config catalog remove --index 2
```

### Discovering datasets

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;

let sources = CatalogSources::new().configure_default();
let catalog = Catalog::of(&sources);

// List all available datasets
for entry in catalog.datasets() {
    println!("{} (profiles: {})",
        entry.name(),
        entry.profile_names().join(", "));
}

// Find by exact name (case-insensitive)
if let Some(entry) = catalog.find_exact("my-dataset") {
    println!("found: {}, at: {}", entry.name(), entry.location());
}

// Find by glob pattern
let matches = catalog.match_glob("my-vectors*");
for entry in matches {
    println!("  {}", entry.name());
}
```

### Adding catalogs programmatically

```rust
let sources = CatalogSources::new()
    .add_catalogs(&[
        "https://my-bucket.s3.amazonaws.com/datasets/".into(),
        "/local/path/to/datasets/".into(),
    ]);
let catalog = Catalog::of(&sources);
```

A directory with its own `catalogs.yaml` adds the catalogs that file
names, under its names. `add_named_catalog(name, path)` adds one
location under a name of the caller's choosing.

### Registering catalogs at runtime

A program can supply catalogs of its own: a `CatalogBuilder` catalog
from entries it built or catalog content it holds, or any type that
implements `CatalogSource`. They join the same ordered set as the
configured catalogs, through `with` (a catalog that gives nothing is
reported) or `with_optional` (it is not):

```rust
use vectordata::catalog::{Catalog, CatalogBuilder, CatalogSources};
use vectordata::Location;

let shipped = CatalogBuilder::named("shipped")
    .base(Location::parse("https://bucket.example/datasets/")?)
    .content(include_str!("catalog.yaml"))
    .build();
let sources = CatalogSources::new().with(shipped).configure_default();
let catalog = Catalog::of(&sources);
```

- **Content** is a canonical catalog list or a `knn_entries.yaml`
  document, in JSON or YAML, read by the same parser as a catalog file.
  Relative paths resolve against `base`; content with a relative path
  and no base is a diagnostic, and that entry is skipped.
- **Order** is the order catalogs were added, and is the listing order.
- **Collisions:** a dataset name in two catalogs is
  `Error::AmbiguousDataset`, whichever kinds the catalogs are. Its
  message names each catalog and where it is.
- **Provenance:** every entry carries its catalog's name
  (`catalog_name()`), and an entry read from a file carries that file
  (`catalog_file()`).
- **The explorer:** `vectordata::explore::run_with_sources(args,
  sources)` opens it over the same set. A program's catalogs are shown
  read-only on its catalogs screen, and survive an edit to
  `catalogs.yaml` there (`CatalogSources::reload_configured`).

`examples/register_catalog.rs` builds both kinds of catalog over a
dataset it writes, and opens and reads from each.

### Datasets and entries built in code

A dataset described in code needs no `dataset.yaml`:

```rust
use vectordata::dataset::{CatalogEntry, DatasetConfig, StandardFacet};
use vectordata::{Location, TestDataGroup};

let description = DatasetConfig::builder("demo-10k")
    .distance_function("COSINE")
    .profile("default", |p| p
        .facet(StandardFacet::BaseVectors, "base.fvec")
        .facet(StandardFacet::QueryVectors, "query.fvec")
        .facet(StandardFacet::NeighborIndices, "gt.ivec"))
    .build()?;
let home = Location::parse("/data/demo-10k/")?;

// Open it directly…
let group = TestDataGroup::from_config("demo-10k", home.clone(), description.clone());
// …or list it in a catalog.
let entry = CatalogEntry::builder("demo-10k").home(&home).dataset(description).build()?;
```

- **`build()` checks what a parse checks:** the version gate, stated
  parents, shard declarations and inheritance. Without
  `format_version` the description states the lowest version its
  content needs. `save(path)` writes it as `dataset.yaml`, and a parse
  of that file equals the built description.
- **The home** is where relative facet paths resolve, and what the
  cache mirrors under `<cache_root>/<name>/`.
- **An entry** holds either the whole description (`dataset`), so it
  opens with no fetch, or a summary and the `dataset.yaml` it points to
  (`descriptor`). Its `EntryDescription` says which.

### When a catalog fails to load

See [§3.1](03-catalogs.md#when-a-catalog-fails-to-load): a broken
catalog is reported and never stops a command, except one named for the
run with `--at` or `--catalog`. In the library, problems are values on
`Catalog::diagnostics`, each naming its catalog.

---

## 2.3 File Extension Scheme

An extension names an element type and a container, and nothing
else. The one table that reads them is `ElementType::parse_extension`
in `veks-datatypes`, re-exported as `vectordata::ElementType`; the full
table of types and spellings is [§1.2](01-data-model.md#12-element-types),
and the design is `docs/design/srd-element-access.md` §3.

### Summary

| Suffix | Structure | Example |
|--------|-----------|---------|
| `.<type>` | Scalar (flat packed, no header) | `.u8`, `.i32`, `.f32`, `.bf16` |
| `.<type>vecs` | Uniform vector (fixed dimension per record) | `.f32vecs`, `.u8vecs`, `.bf16vecs` |
| `.<type>vvecs` | Variable-length vector (per-record dimension) | `.i32vvecs`, `.u8vvecs` |

Legacy aliases, read everywhere and still written by the pipeline:
`.fvecs`=`.f32vecs`, `.ivecs`=`.i32vecs`, `.bvecs`=`.u8vecs`,
`.svecs`=`.i16vecs`, `.mvecs`=`.f16vecs`, `.dvecs`=`.f64vecs`, and
the same letters before `vvecs`. Singular forms (`.fvec`, `.u8vec`,
`.ivvec`) are aliases too. `bf16` is a type of its own: `.bf16`,
`.bf16vecs`, `.bf16vvecs`, never a reinterpreted `u16` or `f16`.

```rust
use vectordata::{Container, ElementType, ElementTypeExt};

assert_eq!(ElementType::parse_extension("fvecs"), Some((ElementType::F32, Container::Vec)));
assert_eq!(ElementType::F32.extension(Container::Vec), "f32vecs");          // canonical
assert_eq!(ElementType::F32.legacy_extension(Container::Vec), Some("fvecs")); // what writers emit
let ty = ElementType::from_path("profiles/base/base_vectors.bf16vecs")?;    // BF16
```

### Record layout

**Uniform (`vec`)** — `[dim:i32 | elem₀ | elem₁ | … | elem_{dim-1}]`
— all records share the same dimension. Random access by stride
arithmetic.

**Variable (`vvec`)** — same per-record layout, but each record may
have a different dimension. Random access requires an offset index
file (`IDXFOR__<name>.<i32|i64>`), built automatically on first
local open and fetched as a sibling URL on remote open.

---

## 2.4 Unified Open Functions

### `open_vec<T>(path_or_url) → Box<dyn VectorReader<T>>`

Opens a uniform vector file at its own element type. Local paths
mmap; URLs go through the merkle-cached path when a `.mref` is
published, falling back silently to direct HTTP RANGE otherwise.

```rust
use vectordata::io::{open_vec, VectorReader};

let r = open_vec::<f32>("data/base.fvecs")?;             // local mmap
let r = open_vec::<i32>("https://host/neighbors.ivecs")?; // cached or direct HTTP

println!("count={}, dim={}", r.count(), r.dim());
let vec: Vec<f32> = r.get(0)?;
```

**Supported types:** every `T: Element` — `u8 i8 u16 i16 u32 i32 u64
i64`, `half::f16`, `half::bf16`, `f32`, `f64` (the `half` types are
re-exported as `vectordata::typed_access::{f16, bf16}`).

The raw readers are **native-only**: `T` must be the type the
extension names, not merely its width. `open_vec::<f32>("x.ivecs")`,
`open_vec::<i32>("x.fvecs")` and `open_vec::<u8>("x.i8vecs")` are
each `IoError::ElementMismatch { file, requested, reason }` at open,
where they once reinterpreted the bytes. The same holds for
`XvecReader::open` / `open_path`, `IndexedVvecReader` and
`ScalarReader`.

### `open_vec_native(path_or_url) → AnyVectorReader` and `open_vec_as<T>`

To read a file whose type you do not know, or at a type the file does
not hold:

```rust
use vectordata::io::{open_vec_as, open_vec_native, AnyVectorReader};

// Whatever the file holds, at its own type.
let any: AnyVectorReader = open_vec_native("codes.u8vecs")?;
println!("{} × {} of {}", any.count(), any.dim(), any.element_type());
let codes = any.into_typed::<u8>()?;                      // ElementMismatch if not u8

// At the type you want, through the conversion table (§2.7).
let base = open_vec_as::<f32>("base.bf16vecs")?;          // widened per read
let base = open_vec_as::<f32>("base.fvecs")?;             // native: the same zero-copy reader
let err  = open_vec_as::<u8>("base.fvecs").unwrap_err();  // Refused: ElementMismatch
```

### `open_vvec<T>(path_or_url) → Box<dyn VvecReader<T>>`

Opens a variable-length vector file. Requires a companion
`IDXFOR__<name>.<i32|i64>` offset index file:

- For local files, built and persisted on first open.
- For remote files, fetched via HTTP from the same URL prefix; if
  absent, the reader walks the file via the channel to rebuild the
  index (slow first time, correct).

```rust
use vectordata::io::{open_vvec, VvecReader};

let r = open_vvec::<i32>("metadata_indices.ivvecs")?;
println!("{} records", r.count());
let record: Vec<i32> = r.get(42)?;
let dim: usize = r.dim_at(42)?;
```

---

## 2.5 Traits and concrete types

### `VectorReader<T>` — uniform vectors

```rust
pub trait VectorReader<T>: Send + Sync {
    fn dim(&self) -> usize;
    fn count(&self) -> usize;
    fn get(&self, index: usize) -> Result<Vec<T>, IoError>;

    // bounds-checked zero-copy slice; None when storage is not mmap-backed
    fn get_slice(&self, index: usize) -> Option<&[T]> { None }

    // records [start, start + buf.len() / dim) into buf, through storage
    fn read_into(&self, start: usize, buf: &mut [T]) -> Result<usize, IoError>;

    // every element of every record, contiguous, borrowed from the mapping:
    // Some only for a header-free column (or a window over one) in one
    // resident mapped file; None for xvec files, series, HTTP, conversions
    fn as_slice(&self) -> Option<&[T]> { None }

    // Native unless this reader converts (§2.7)
    fn conversion(&self) -> ElementConversion { ElementConversion::Native }

    // the element type the file holds, when the reader knows it
    fn native_element_type(&self) -> Option<ElementType> { None }

    // drive underlying storage to fully resident; no-op for local/direct-HTTP
    fn prebuffer(&self) -> std::io::Result<()> { Ok(()) }

    // true for local; true for cached once every chunk is verified; false for direct-HTTP
    fn is_complete(&self) -> bool { true }
}
```

The concrete implementations, each generic over `T: Element`:

- `XvecReader<T>` — a uniform vector file. It also exposes hot-path
  inherent methods that the `dyn VectorReader<T>` surface can't: an
  unchecked `get_slice(index) -> &[T]` (panics if the storage is not
  mmap-backed; intended for KNN inner loops), plus `advise_sequential`
  / `advise_random`, `prefetch_range`, `release_range`, and
  `prefetch_pages` (madvise hints; no-op when not mmap-backed).
- `ScalarReader<T>` — a header-free column (`.u8`, `.f32`, …): `dim()
  == 1`, `count = size / T::SIZE` (a remainder is `InvalidFormat`),
  `get`/`get_slice` at `i × SIZE`, and `as_slice()` is the whole
  column once the file is mapped and aligned.
- `ConvertingReader<N, T>` — `VectorReader<T>` over an `Arc<dyn
  VectorReader<N>>` for a `Widening` or `CrossSign` conversion: `get`
  and `read_into` convert through a bounded scratch; `get_slice` and
  `as_slice` are `None`; the first value that does not fit is
  `IoError::ValueOverflow { index, value, target }` at its ordinal.
  Never built for `Native`, so the native path costs nothing.
- `AnyVectorReader` — one variant per element type holding an
  `Arc<dyn VectorReader<T>>`: what a file opens as before a type is
  chosen. `element_type()`, `count()`, `dim()`, `into_typed::<T>()`
  (native only, else `ElementMismatch`), `as_type::<T>()` (through the
  conversion table). `AnyVvecReader` is the same over `VvecReader`.

### `VvecReader<T>` — variable-length vectors

```rust
pub trait VvecReader<T: VvecElement>: Send + Sync {
    fn count(&self) -> usize;
    fn dim_at(&self, index: usize) -> Result<usize, IoError>;
    fn get_bytes(&self, index: usize) -> Result<Vec<u8>, IoError>;
    fn get(&self, index: usize) -> Result<Vec<T>, IoError>;        // default impl
    fn prebuffer(&self) -> std::io::Result<()>;
    fn is_complete(&self) -> bool;
}
```

The canonical concrete implementation is `IndexedVvecReader<T>`.
`get_raw(index) -> Option<&[u8]>` provides a zero-copy slice when
the storage is mmap-backed.

### `Element` — the one element bound

```rust
pub trait Element: Copy + Send + Sync + 'static {
    const TYPE: ElementType;
    const SIZE: usize;
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn write_le_bytes(self, out: &mut [u8]);
    fn to_value(self) -> Value;                 // Value::{Int(i128), Float(f64)}
    fn from_value(v: Value) -> Option<Self>;    // None: does not fit
    fn as_f64(self) -> f64;
}
```

Defined in `veks-datatypes` and implemented for the twelve types of
§2.3; `vectordata::{Element, ElementType, ElementConversion,
Container}` re-export it. `VvecElement` and `TypedElement` remain as
deprecated blanket traits over `Element`, so an existing `T:
VvecElement` bound compiles with a warning; write `T: Element`.

---

## 2.6 Dataset Access via TestDataGroup

### Loading a dataset

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

// From a local directory containing dataset.yaml
let group = TestDataGroup::load("./my-dataset/")?;

// From an HTTP URL
let group = TestDataGroup::load("https://host/datasets/my-dataset/")?;

// Top-level dataset attributes
let dist = group.attribute("distance_function");  // Option<&Value>

// The dataset's name, its cache directory's (Option<&str>)
let name = group.name();
```

`group.name()` is the catalog entry's name for a catalog open, the
directory's last segment for a directory, `dataset.yaml` or URL open
(whatever `name:` the file states), the given name for `from_config`,
and `None` for a dataset at a server's root.

### Profile access

```rust
// Trait object — the universal handle. Use this for almost everything.
let view: Arc<dyn TestDataView> = group.profile("default").unwrap();

// Concrete type — rarely needed; every facet method below is on the
// trait object, through TestDataViewExt.
let gview: GenericTestDataView = group.generic_view("default").unwrap();
```

### Source resolution

A facet's source string in `dataset.yaml` may be:

- A **relative path or URL fragment** — joined onto the dataset's
  base location (the directory of `dataset.yaml` for local datasets,
  the URL prefix for remote ones).
- An **absolute HTTP URL** (`http://…` or `https://…`) — passed
  through unchanged regardless of where the `dataset.yaml` lives.
  This means a local `dataset.yaml` can declare facets hosted on
  a remote server, and a remote `dataset.yaml` can pull facets
  from a different bucket — the dispatch is per-facet.
- An **absolute local path** — used as-is on local filesystems.

For each facet, the resolved source string is then handed to
`Storage::open(source)`, which picks the transport variant
(local mmap, merkle-cached, direct HTTP) without further input from
the caller.

### Standard facet methods

The facet methods are on `TestDataViewExt`, implemented for every
`TestDataView` (so for `Arc<dyn TestDataView>` and the concrete view
alike — import the trait). Each takes the element type you want to
read at and returns a reader that works identically for local and
remote datasets:

```rust
use vectordata::TestDataViewExt;

// Uniform vector facets → Arc<dyn VectorReader<T>>
let base:  Arc<dyn VectorReader<f32>> = view.base_vectors_as::<f32>()?;
let query: Arc<dyn VectorReader<f32>> = view.query_vectors_as::<f32>()?;
let gt:    Arc<dyn VectorReader<i32>> = view.neighbor_indices_as::<i32>()?;
let dist:  Arc<dyn VectorReader<f32>> = view.neighbor_distances_as::<f32>()?;
// F facet — pre-filter ground truth (ACORN G_K).
let fki = view.prefiltered_neighbor_indices_as::<i32>()?;
let fkd = view.prefiltered_neighbor_distances_as::<f32>()?;
// E facet — post-filter ground truth (G ∩ R).
let pki = view.postfiltered_neighbor_indices_as::<i32>()?;
let pkd = view.postfiltered_neighbor_distances_as::<f32>()?;
// Any facet by key, at any type the conversion table allows (§2.7)
let codes: Arc<dyn VectorReader<u8>> = view.facet_as::<u8>("base_vectors")?;
let labels: Arc<dyn VectorReader<i64>> = view.facet_as::<i64>("metadata_content")?;

// Variable-length facet → Arc<dyn VvecReader<i32>> (native type only)
let mi: Arc<dyn VvecReader<i32>> = view.metadata_results_as::<i32>()?;

// A quantized base with its parameters (§2.7)
let q = view.quantized_base::<u8>()?;

// Metadata configuration (declared facets, no data access)
let meta_cfg: Option<&FacetConfig> = view.metadata_content();
let pred_cfg: Option<&FacetConfig> = view.metadata_predicates();

// Facet discovery
let manifest: HashMap<String, FacetDescriptor> = view.facet_manifest();

// Element type interrogation, without opening or with
let etype = view.facet_element_type("metadata_content")?;  // ElementType::U8
let any = view.open_facet_native("metadata_content")?;     // AnyVectorReader

// Resolved source path/URL for any declared facet
let src: Option<String> = view.facet_source("metadata_content");

// Fetch into the cache / inspect it
view.fetch(&FetchRequest::all(), &mut Silent)?;
let storage = view.open_facet_storage("base_vectors")?;    // FacetStorage
```

The pinned methods of 3.0 — `base_vectors()`, `query_vectors()`,
`neighbor_indices()`, `facet(name)` and the filtered six — remain on
`TestDataView`, deprecated, as `*_as::<f32>` or `<i32>`. They now go
through the same table: `facet("codes")` on a `.u8` facet widens where
it once reinterpreted the bytes, and on an `.ivecs` facet is refused.

---

## 2.7 Typed facet access

Every facet is a file of one element type (§2.3), read from its
extension at open. `facet_as::<T>(name)` opens the facet **at `T`**:
natively when the file holds `T`, through a conversion when the table
below allows one, and refused otherwise — never a reinterpretation of
the bytes, never a silent loss of precision. The three readers a
facet may come back as are the `XvecReader`/`ScalarReader` of §2.5
(native), a window or series over them, or a `ConvertingReader`.

### Native: zero-copy

```rust
use vectordata::{ElementConversion, TestDataViewExt, VectorReader};

// u8 codes of a quantized corpus, read as what they are
let codes = view.base_vectors_as::<u8>()?;
assert_eq!(codes.conversion(), ElementConversion::Native);
let row: &[u8] = codes.get_slice(42).expect("mapped and complete");

// a header-free column is one slice once the file is resident
let labels = view.facet_as::<u8>("metadata_content")?;
if let Some(all) = labels.as_slice() {
    assert_eq!(all.len(), labels.count());
}

// bulk reads into a buffer you own, through any storage
let mut buf = vec![0f32; 1024 * base.dim()];
let n = base.read_into(0, &mut buf)?;
```

`get_slice` and `as_slice` are `Some` only when the storage is mapped
and the records are resident (`fetch` first, §2.12); `get` and
`read_into` always work.

### Widened: the file's type is narrower than yours

```rust
// a bf16 base, as f32 — each read converts; the mapping is still shared
let base = view.base_vectors_as::<f32>()?;
assert_eq!(base.conversion(), ElementConversion::Widening);
assert_eq!(base.native_element_type(), Some(ElementType::BF16));
assert!(base.get_slice(0).is_none());          // nothing to borrow at f32
let v: Vec<f32> = base.get(0)?;                // exact: every bf16 is an f32
```

### Refused

```rust
use vectordata::io::IoError;
use vectordata::Error;

match view.facet_as::<u8>("base_vectors") {       // an f32 facet
    Err(Error::VectorIo(IoError::ElementMismatch { file, requested, reason })) => {
        assert_eq!((file, requested), (ElementType::F32, ElementType::U8));
    }
    _ => unreachable!(),
}
```

### The conversion table

`ElementType::conversion_to(target)`, stated once in
`veks-datatypes` and used by every open:

| file → requested | result |
|---|---|
| same type | `Native` — zero-copy |
| int → wider int, source unsigned or target signed (u8→i16, u8→u32, i16→i64) | `Widening` — never fails |
| int → int, same width, other sign (u8↔i8, i32↔u32) | `CrossSign` — every value checked |
| int → wider unsigned from signed (i8→u16) | `CrossSign` |
| int → narrower int | `Refused` |
| int → float, integer bits ≤ significand bits (u8/i8/u16/i16→f32; ≤32-bit→f64; u8/i8→f16; u8/i8→bf16) | `Widening` |
| int → float otherwise (i32→f32, i64→f64) | `Refused` — not lossless |
| f16→f32, f16→f64, bf16→f32, bf16→f64, f32→f64 | `Widening` |
| f16↔bf16 | `Refused` — neither direction is lossless, and never a reinterpretation |
| float → narrower float, float → int | `Refused` |

A `CrossSign` reader checks each value: a `u8` of 200 read as `i8` is
`IoError::ValueOverflow { index, value, target }` from `get` and from
`read_into`, at the facet ordinal. `Refused` means read natively and
convert yourself — `open_facet_native` gives the reader at the file's
type:

```rust
use vectordata::io::AnyVectorReader;

match view.open_facet_native("base_vectors")? {
    AnyVectorReader::F32(r) => { /* zero-copy f32 */ }
    AnyVectorReader::I32(r) => { /* i32 is not lossless as f32: your rounding, your call */ }
    other => { let as_f64 = other.as_type::<f64>()?; /* or widen to f64, which every ≤32-bit type is */ }
}
```

### Quantized corpora

A profile whose base is the codes of a quantized corpus carries the
parameters as three standard facets and the scheme as an attribute
([§1.5, §1.7](01-data-model.md#15-dataset-facets-bqgdmpref)).
`quantized_base::<C>()` opens the four as one thing:

```rust
let q = view.quantized_base::<u8>()?;    // QuantizedBase<u8>
let codes: &[u8] = q.codes.get_slice(i).unwrap();
let (scale, zp, sum) = (q.scale.as_slice().unwrap()[i], q.zero_point.as_slice().unwrap()[i], q.sum.as_slice().unwrap()[i]);
assert_eq!(q.scheme.scheme, vectordata::dataset::QuantScheme::Affine);
```

`C` must be the type the attribute's `codes` names (else
`ElementMismatch`); each column must hold one record per code vector
(else `Error::FacetCardinality { facet, against, expected, found }`);
the attribute must parse (`view.quantization()` is `Some(Err(why))`
when it does not).

### Migrating from `TypedReader`

`open_facet_typed`, `TypedReader<T>`, `TypedElement`,
`TypedAccessError` and `dispatch_typed!` are deprecated and will be
removed; they covered integer scalars only and read every float as
zero.

| 3.0 | now |
|---|---|
| `open_facet_typed::<T>(&*view, name)` | `view.facet_as::<T>(name)` |
| `r.get_native(i)` | `r.get_slice(i)?[0]` or `r.as_slice()?[i]` |
| `r.get_value(i)?` | `r.get(i)?[0]` |
| `r.get_native_slice(i)` | `r.get_slice(i)` |
| `r.is_native()` | `r.conversion() == ElementConversion::Native` |
| `TypedReader::<T>::open(path)` | `ScalarReader::<T>::open_path(path)` (native) or `open_vec_as::<T>(path)` |
| `TypedReader::<T>::open_url(url, native)` / `open_auto` | `open_vec_as::<T>(url)` — the native type comes from the extension |
| `dispatch_typed!(etype, T => …)` | `veks_datatypes::dispatch_element!(etype, T => …)` |

---

## 2.8 Facet Reference

One row per `StandardFacet`, in `PREFERRED_ORDER` (a test in
`vectordata` holds the keys to the code). `facet_as::<T>(key)` opens
any of the vector and scalar rows; the method column is the
`TestDataViewExt` shorthand where there is one.

| Facet code | YAML key | Method | Natural reader | Typical format |
|-----------|----------|-------------|-------------|---------------|
| B | `base_vectors` | `base_vectors_as::<T>()` | `VectorReader<f32>`, or the codes' type | `.fvecs`, `.bf16vecs`, `.u8vecs` |
| — | `quant_scale` | `quantized_base::<C>()` | `VectorReader<f32>` | `.f32` |
| — | `quant_zero_point` | `quantized_base::<C>()` | `VectorReader<i32>` | `.i32` |
| — | `quant_sum` | `quantized_base::<C>()` | `VectorReader<i32>` | `.i32` |
| Q | `query_vectors` | `query_vectors_as::<T>()` | `VectorReader<f32>` | `.fvecs` |
| G | `neighbor_indices` | `neighbor_indices_as::<T>()` | `VectorReader<i32>` | `.ivecs` |
| D | `neighbor_distances` | `neighbor_distances_as::<T>()` | `VectorReader<f32>` | `.fvecs` |
| M | `metadata_content` | `metadata_content()` (config) + `facet_as::<T>` | `VectorReader<u8>`, or a slab | `.u8`, `.slab` |
| P | `metadata_predicates` | `metadata_predicates()` (config) + `facet_as::<T>` | `VectorReader<u8>`, or a slab | `.u8`, `.slab` |
| R | `metadata_results` (alias `metadata_indices`) | `metadata_results_as::<T>()` | `VvecReader<i32>` | `.ivvecs` |
| — | `metadata_layout` | `metadata_layout()` | the schema slab | `.slab` |
| F (indices) | `prefiltered_neighbor_indices` (canonical) or `filtered_neighbor_indices` (legacy alias) | `prefiltered_neighbor_indices_as::<T>()` | `VectorReader<i32>` | `.ivecs` |
| F (distances) | `prefiltered_neighbor_distances` (canonical) or `filtered_neighbor_distances` (legacy alias) | `prefiltered_neighbor_distances_as::<T>()` | `VectorReader<f32>` | `.fvecs` |
| E (indices) | `postfiltered_neighbor_indices` | `postfiltered_neighbor_indices_as::<T>()` | `VectorReader<i32>` | `.ivecs` |
| E (distances) | `postfiltered_neighbor_distances` | `postfiltered_neighbor_distances_as::<T>()` | `VectorReader<f32>` | `.fvecs` |

The index facets (G, F, E) hold base ordinals; a filtered row shorter
than K is padded with `-1` (F when fewer than K base vectors pass the
predicate, E whenever the intersection is short), and G never is.
`veks check` holds every truth's rows to its queries, its distances'
rows to its indices, and every index to the base count.

---

## 2.9 Dataset YAML profile schema

A `dataset.yaml` profile section maps canonical facet names to file
paths. All paths are relative to the dataset directory:

```yaml
profiles:
  default:
    maxk: 100
    base_vectors: profiles/base/base_vectors.fvecs
    query_vectors: profiles/base/query_vectors.fvecs
    neighbor_indices: profiles/default/neighbor_indices.ivecs
    neighbor_distances: profiles/default/neighbor_distances.fvecs
    metadata_content: profiles/base/metadata_content.u8
    metadata_predicates: profiles/base/predicates.u8
    metadata_indices: profiles/default/metadata_indices.ivvecs
    prefiltered_neighbor_indices: profiles/default/prefiltered_neighbor_indices.ivecs    # F
    prefiltered_neighbor_distances: profiles/default/prefiltered_neighbor_distances.fvecs
    postfiltered_neighbor_indices: profiles/default/postfiltered_neighbor_indices.ivecs  # E
    postfiltered_neighbor_distances: profiles/default/postfiltered_neighbor_distances.fvecs
```

The legacy keys `filtered_neighbor_indices` /
`filtered_neighbor_distances` still parse (they resolve to **F**, the
pre-filter facet) for backwards compatibility with datasets published
before the F/E split.

The `metadata_indices` key maps to the `predicate_results` field
internally (serde alias).

### Inheritance

A non-default profile inherits every facet it does not declare from
its parent. Below `format_version` 3 an absent `inherits:` means
`default`; from 3 every profile other than `default` states its parent,
and an absent, unknown, self or cyclic parent, or a `partition: true`
profile that names one, is a load refusal naming the profiles
(`docs/design/srd-profile-layers.md`). A profile naming any parent
other than `default` is what requires version 3.

What crosses depends on whether the step changes size, and the axis of
a step is **derived from `base_count`**: a size step is one where the
child's count differs from the parent's effective one, whatever the
parent is called, so a `20m` that builds on `10m` re-cuts the windows
and takes no ground truth.

- **Across a size step** (the child's `base_count` differs from its
  parent's):
  `base_vectors` and `metadata_content` inherit cut to
  `[0..base_count)`; `query_vectors`, `metadata_predicates`, and
  `metadata_layout` inherit as they are. The neighbor facets, the
  pre- and post-filtered ground truth, and `metadata_results` do
  **not** cross: each is derived from `base_count`, so a parent's copy
  at another size is wrong for the child in the same way. A sized
  profile declares its own or has none.
- **Across a step at the same size** (no `base_count` of its own, or
  its parent's restated): every facet inherits as it is,
  `metadata_results` included.

### Layers

A **layer** is a profile that exists to be inherited from: it declares
the facets its children share and none they differ on. A **size layer**
such as `10m` holds the base window, the metadata window and the
unfiltered ground truth, and no predicate group; opening it is the
unfiltered benchmark. A **predicate set** names a size layer as its
parent and declares the predicate group — `metadata_predicates` when it
has its own slab, `metadata_results`, and the filtered ground truth —
with the tags PS-23 requires. `default` may hold a slab that is
invariant across sizes, and every layer under it inherits that slab
unchanged.

From version 3 a dataset is **layered**: a generated rung is written as
its size layer and the mixed set beside it, `10m` and `10m-mixed`, and a
per-profile pipeline template runs where the profile declares the facet
its command produces, so the unfiltered KNN runs on the layer and the
evaluation on the set, whose steps wait on the layer's ground truth. A
step that would write a predicate-group facet into a layer is refused at
plan time naming the facet and the profile. `veks check` holds the group
together by content: the results index has one row per predicate of the
slab it resolves to, each filtered ground truth was computed from the
profile's own results, and their rows agree with the query set; and it
reports a parent whose declared facet a direct child overrides, since
that declaration is a decoy.

A version-3 dataset whose only parents are `default` downgrades with
`veks prepare downgrade --to 2`, which drops those lines and the tag
schema; one with a named parent does not, since no lower version can
say what it says.

The rule for `metadata_results` was settled on 2026-09-05; before that
it crossed a size step, and a sized profile that omitted it would have
read its parent's index at the wrong size.

### Partition profiles

Partition profiles are marked with `partition: true`. They have
independent base vectors (not windowed from default) and do NOT
inherit views from the default profile:

```yaml
  label-0:
    maxk: 100
    base_count: 82993
    partition: true
    base_vectors: profiles/label-0/base_vectors.fvecs
    query_vectors: profiles/label-0/query_vectors.fvecs
    neighbor_indices: profiles/label-0/neighbor_indices.ivecs
    neighbor_distances: profiles/label-0/neighbor_distances.fvecs
```

### Selectors

Everywhere a `dataset:profile` spec is accepted, what follows the colon
is a **selector** (`docs/design/srd-profile-selectors.md`). A bare name
still names one profile; an expression names the set of profiles whose
facts match:

```text
tessera:10m                                  one profile, as before
tessera:size=10m,predicates=uniform-2        AND of two atoms
tessera:or(10m,20m)                          junctions: and(), or(), not()
tessera:selectivity=1e-3..1e-2               half-open interval
tessera:family=uniform*,not(size<10m)        glob, comparison
tessera:form='topic_l3.eq_citation_percentile.range'   quoted literal
tessera:profile=*                            every profile
```

An atom is `key op value` with `=`, `!=`, `<`, `<=`, `>`, `>=`; a
comma is AND. A value is read by its spelling: `^…` or `…$` is a
regular expression (RE2 subset, case-insensitive), `*`, `?` and `[…]`
make a glob, `lo..hi` an interval, a number may carry a count suffix
(`10m`, `128mi`, `1e-3`), `true`/`false` are booleans, anything else a
literal, and quotes force a literal. Everything is case-insensitive.
The keys `profile`, `base_count`, `maxk`, `partition` and `inherits`
are read from the profile as loaded, after inheritance; every other
key is read from its `attributes:`, which never inherit. An absent
attribute matches nothing; a list attribute matches when any element
does; a map is addressed by dotted keys.

The head of a spec is found by its shape — a URL by scheme and
authority, a path by its separators or a drive letter, otherwise a
catalog name — so `https://host:8080/ds:10m` and `C:\data\ds:10m` split
where they should. A spec with no selector means `default`. **A head
that exists on disk is a path**, before the catalogs are consulted: in
a directory that holds a `my-dataset/`, `open_spec("my-dataset:default")`
opens that directory, not the catalog entry of the same name. Run
from elsewhere, or name the catalog entry by a spelling that is not a
path, to reach the catalog.

Which surface receives a selector decides what a set means. `precache`,
`ping`, the explorer's purge and picker filter, and
`Catalog::open_profiles` act on every match. `describe`, `explore
--dataset`, `derive`, a pipeline's `--profile`, and
`Catalog::open_profile` need exactly one; more is an error listing the
matches, and zero is an error listing the dataset's profiles and their
attributes. `precache` with no selector is refused naming
`dataset:profile=*`, which is how "every profile" is spelled.

Programmatically:

```rust
let group = TestDataGroup::load("path/to/dataset")?;
let names = group.select(Some("size>=10m,predicates=mixed"))?;   // the set
let one = group.select_one(Some("10m"))?;                        // exactly one
let facts = group.profile_facts();                               // what a selector reads
let views = catalog.open_profiles("tessera", Some("profile=*"))?;
```

### Tags

From `format_version: 3` a dataset may declare a **tag schema**, and a
profile's `attributes:` are then its tags:

```yaml
profile_tags:            # naming order; `~` marks a naming tag
  size: ~                # the rung a sized profile was generated for
  predicates: ~          # `mixed` or `uniform-<n>`, wherever a predicate facet is declared
  selectivity: ~         # only single-level sets carry it
  family: stratified     # a tag with a default is carried by every profile but does not name

profiles:
  default:
    attributes: { size: 495m, predicates: mixed, family: stratified }
  10m:
    base_count: 10000000
    attributes: { size: 10m, predicates: mixed, family: stratified }
```

A tag is a plan, not a measurement: generators write what a profile was
asked to be, and verification reports what came out. The sized-profile
derivation writes `size` for every member and the default carries the
rung spelling of its own count; a predicate generator writes `family`,
`forms`, `selectivity_ladder` and the structural class `predicates`,
which `veks check` holds to a form census of the facet. A generated
profile is **named by its naming tags** in schema order joined with `-`
(`10m-uniform-2-1e-2`); names never change once declared. Tags are
written into an existing `dataset.yaml` as a textual edit of the
profile's own lines, never a serializer round trip, and every step that
writes a tag records it beside its outputs so a hand edit is reported
as stale rather than silently kept or overwritten. A deliberate change
of plan goes through `veks prepare tags --profile <selector> --set
key=value`, which edits the profiles the selector names and records the
new value in every step record that holds the tag, so nothing turns
stale except the published definition, which refreshes.

### `knn_entries.yaml` fallback

When `dataset.yaml` is not found, `TestDataGroup::load` falls back
to `knn_entries.yaml` (jvector-compatible format):

```yaml
_defaults:
  base_url: https://example.com/data

"my-dataset:default":
  base: profiles/base/base_vectors.fvecs
  query: profiles/base/query_vectors.fvecs
  gt:    profiles/base/neighbor_indices.ivecs
```

The file is read by the catalog's `knn_entries.yaml` parser, so a
directory open honours `_defaults.base_url`, `cache_dir` and `${VAR}`
exactly as a catalog listing of the same file does. The dataset opened
is the entry named after the directory; failing that, the file's only
entry. A file with several entries and none named after the directory
is `Error::AmbiguousDataset`, listing them — ambiguity is never
resolved by order. Name the entry after its directory, or open the
file as a catalog and pick an entry.

The reverse holds too: a catalog source that points at a
`dataset.yaml` lists that one dataset (named by its `name:`, else by
its directory), with its profiles embedded, rather than a dataset per
top-level key.

---

## 2.10 Offset index files

Variable-length vector files (`.ivvecs`, `.fvvecs`, etc.) require a
companion offset index for random access:

```
data/metadata_indices.ivvecs              # variable-length data
data/IDXFOR__metadata_indices.ivvecs.i32  # offset index (< 2 GB data)
data/IDXFOR__metadata_indices.ivvecs.i64  # offset index (≥ 2 GB data)
```

The index is a flat-packed array of byte offsets (one per record).
It is created automatically:

- On first local open by `IndexedVvecReader::open` /
  `IndexedVvecReader::open_path`.
- By the `generate vvec-index` pipeline step before publishing.
- By `evaluate-predicates` immediately after writing vvec output.

For remote access, the reader fetches the sidecar from the same URL
prefix. If the sidecar is absent, the reader walks the file
through the storage layer to rebuild the index (slow first time but
correct; the rebuilt index is not persisted to the remote source).

---

## 2.11 Error handling

```rust
use vectordata::io::IoError;

match open_vec::<f32>("data.fvecs") {
    Ok(reader) => { /* use reader */ }
    Err(IoError::Io(e)) => eprintln!("I/O error: {e}"),
    Err(IoError::Http(e)) => eprintln!("HTTP error: {e}"),
    Err(IoError::InvalidFormat(msg)) => eprintln!("bad format: {msg}"),
    Err(IoError::OutOfBounds(idx)) => eprintln!("index {idx} out of range"),
    Err(IoError::VariableLengthRecords(msg)) => {
        // Wrong shape — file has variable-length records; use open_vvec
        eprintln!("use open_vvec for this file: {msg}");
    }
    Err(IoError::ElementMismatch { file, requested, reason }) => {
        // The extension names `file`; `requested` is not it and the
        // conversion table has no path between them (§2.7)
        eprintln!("{file} file opened as {requested}: {reason}");
    }
    Err(IoError::ValueOverflow { index, value, target }) => {
        eprintln!("record {index} holds {value}, which is not a {target}");
    }
    Err(other) => eprintln!("{other}"),   // IoError is #[non_exhaustive]
}
```

`VariableLengthRecords` is returned by `open_vec` when the file
turns out to have variable-length records (its size is not a
multiple of the implied stride). The caller should switch to
`open_vvec`. `ElementMismatch` is an open-time error; `ValueOverflow`
is raised by `get` and `read_into` of a `CrossSign` reader at the
first value that does not fit.

A view's facet methods return `vectordata::Error`, which wraps
`IoError` as `Error::VectorIo` and adds, among others,
`Error::FacetCardinality { facet, against, expected, found }` (a
quantization column whose count disagrees with its base) and
`Error::AmbiguousDataset { name, matches }` (a name, or a
`knn_entries.yaml` directory, that resolves to more than one dataset).

---

## 2.12 Prebuffering and caching

### Cache location

Remote downloads are cached under `vectordata::settings::cache_dir()`,
the single source of truth for cache resolution shared with the
`veks-pipeline` crate. Resolution order:

1. A cache directory set for the process with
   `vectordata::settings::set_cache_dir` — or, on the command line,
   `datasets precache --cache-dir`.
2. `cache_dir:` entry in `~/.config/vectordata/settings.yaml`
   (or `$VECTORDATA_HOME/settings.yaml`).
3. `$VECTORDATA_HOME/cache`, when `$VECTORDATA_HOME` is set.
4. `$HOME/.cache/vectordata`, written to `settings.yaml`, when `$HOME`
   is on the largest writable mount.

Otherwise every API that needs the cache returns
`vectordata::settings::SettingsError::NotConfigured`. Print the
error directly — its `Display` impl includes the CLI command and the
manual `mkdir`+`cat` sequence the user can paste to fix it.

A program that embeds vectordata and keeps its own cache calls
`set_cache_dir` before its first open or fetch:

```rust
vectordata::settings::set_cache_dir(&my_cache)?;
```

Only the cache moves: the user's settings and credentials are read
where they are, and nothing is written to `settings.yaml`. The choice
holds for the rest of the process. Repeating it is a no-op; a
different path, or a call after a dataset file was opened in the
configured cache, is refused with `CacheDirConflict` rather than
splitting the process's data between two caches. Loading catalogs and
dataset definitions first is fine: the copies kept of remote ones
(`.catalogs/`, a dataset's `dataset.yaml`) are refreshed on each load,
and from the call on they are kept in the new directory.
(`override_cache_dir_for_process` is the test hook: first call wins,
silently.)

Configure via the CLI:

```bash
veks datasets config set cache /mnt/fast-storage/vectordata-cache
veks datasets config get
```

The directory layout under the resolved root is
`<dataset>/<filename>`, with a sibling `<filename>.mrkl` carrying
merkle state and `origin.json` binding the directory to its publish
URL. Kept catalog copies live in `.catalogs/`.

### Prebuffering datasets

Prebuffering downloads every facet of a profile so subsequent reads
are zero-copy mmap with no further HTTP requests.

While a precache plans, it prints one line per facet as it opens it,
`[i/N] facet (k files): opening… 4.8s`, ticking with elapsed time, and
closes the line with what the facet will cost (`3.9 GiB to fetch`,
`already resident`, or the whole facet when its window cannot be
mapped). Opening is where the wait is on a large remote dataset: one
merkle reference per shard and a slab's offset index are fetched
before a byte of data is, and until this status existed that phase
was a blank screen.

#### Strict contract

Every `prebuffer*` API in `vectordata` honours the same strict
contract:

> **When the call returns `Ok(())`, every facet covered by the call
> is fully resident on local disk and mmap-promoted. There is no
> partial-completion mode, no silent no-op variant, and no
> fallback to a slow per-read path.**

This applies to:

- `Storage::Mmap` → already resident; returns immediately.
- `Storage::Cached` → downloads + merkle-verifies every chunk;
  promotes to mmap; errors if any chunk is missing post-download.
- `Storage::Http` (no `.mref` published) → downloads the whole file
  via HTTP RANGE into the configured cache directory, atomic-renames
  into place, mmap-promotes. (No merkle — bytes are trusted from the
  server. Server-reported size mismatches surface as an error.)
- `view.fetch` / `group.fetch` → the whole run is planned and checked
  before anything moves; per-facet failure is propagated immediately;
  no facet is skipped silently.

After a prebuffer call returns `Ok(())`, **every reader against the
same source — including readers opened *before* the prebuffer call —
sees the promoted state on its next access** (lazy mmap promotion in
`mmap_slice` / `mmap_base` / `is_complete` / `is_local`).
Downstream code that opens a reader at session-init, runs prebuffer,
and then accesses per-cycle gets the zero-copy fast path on the
first cycle.

Until a cached storage is promoted, each access checks for
completion in two steps: the in-memory validity bitmap, then a `stat`
of the `.mrkl` sidecar, whose validity bitset is re-read only when
that file changed (a sibling checkpointed into it) or once per second.
The state file is never loaded on the read path. That is a hard
rule with a regression test behind it
(`reads_on_a_partial_cache_do_not_reload_the_state_file`): on a
410 GB shard the state file is 32 MB, and loading it per read once
held `vectordata explore` to about 50 vectors a second.

#### From the CLI

```bash
# One profile.
vectordata datasets precache my-dataset:default

# Every profile (a bare name is refused, naming this spelling).
# Warns on stderr above 250 MiB across the selected profiles, and continues.
vectordata datasets precache 'my-dataset:profile=*'

# Chosen facets, a record window, or just the plan.
vectordata datasets precache my-dataset:default --facet base_vectors --window 0..1m
vectordata datasets precache my-dataset:default --plan

# Other catalog flags.
vectordata datasets precache my-dataset:default --at https://host/datasets/
```

`veks datasets precache` is the same command. It is a thin adapter
over the library calls below.

#### From Rust — one profile

```rust
use vectordata::fetch::{FetchEvent, FetchRequest, Silent, TextMeter};

let view = catalog.open_profile("my-dataset", "default")?;

// Strict: returns Ok only when every planned byte is resident. The run
// is planned and checked first: an unknown facet, a window that would
// silently become a whole-facet download, or a cache directory without
// room is refused before anything moves.
view.fetch(&FetchRequest::all(), &mut Silent)?;

// Chosen facets, with the CLI's progress meter.
view.fetch(
    &FetchRequest::facets(["base_vectors", "query_vectors"]),
    &mut TextMeter::stderr("Fetch"),
)?;

// A record window. A declared or requested window the format cannot
// map is refused unless the request allows the whole facet.
let window = vectordata::dataset::source::parse_window("0..1m")?;
view.fetch(&FetchRequest::facets(["base_vectors"]).window(window), &mut Silent)?;

// Your own display: every step arrives as a FetchEvent.
view.fetch(&FetchRequest::all(), &mut |e: &FetchEvent<'_>| {
    if let FetchEvent::Progress { facet, bytes, total, .. } = e {
        eprintln!("  {facet}: {bytes}/{total} bytes");
    }
})?;
```

`TextMeter` redraws its line in place, which suits a terminal. For a
log, CI output or the `log` crate, `LogMeter` reports the same run as
whole lines: one per facet planned, started and finished, a progress
line each time the run's total crosses a step (`step`, 10% by
default, parsed from `0.1` or `10%`) but never sooner than
`min_interval` after the last (2 s by default), a 100% line at the
end, and a warning in the same form when nothing has moved for a
minute (`stall_warning`), repeated while the stall lasts.
`vectordata datasets precache` draws `TextMeter` when standard error
is a terminal and `LogMeter` otherwise.

```rust
use std::time::Duration;
use vectordata::fetch::LogMeter;

let mut meter = LogMeter::to_log("Fetch")
    .step(Some("10%".parse()?))
    .min_interval(Duration::from_secs(2));
view.fetch(&FetchRequest::all(), &mut meter)?;
```

`plan_fetch` is the first half on its own: it returns a `FetchPlan`
with the per-facet byte ranges, chunk fills and total cost, and
fetches nothing; `FetchPlan::execute` is the second half.

#### From Rust — several profiles, or straight from a spec

```rust
use vectordata::{TestDataGroup, PREBUFFER_LARGE_WARNING_BYTES};

let group = TestDataGroup::load("https://host/datasets/my-dataset/")?;
let profiles = group.select(Some("profile=*"))?;

let plan = group.plan_fetch(&profiles, &FetchRequest::all(), &mut Silent)?;
if plan.bytes_to_fetch() >= PREBUFFER_LARGE_WARNING_BYTES {
    eprintln!("about to fetch {} bytes", plan.bytes_to_fetch());
}
plan.execute(&mut TextMeter::stderr("Fetch"))?;

// The same from a spec, in one call:
catalog.fetch("my-dataset:size=10m", &FetchRequest::all(), &mut Silent)?;
```

The `prebuffer_*`, `prefetch` and `prefetch_with_progress` methods
these replace are deprecated wrappers over `fetch`;
`prefetch_in_background` remains as the background form.

### Offline opens

A remote facet whose cache copy is complete opens from disk: the merkle
reference comes from its `.mrkl` state (or, without a `.mref`, the size
from the cache file and its full chunk bitmap), so no `.mref` fetch and
no HEAD request is made. A variable-length facet's offset index is read
from the copy kept beside its cache file. A dataset's `dataset.yaml` is
fetched first — it can change upstream — with the copy kept in its
cache directory used when the server cannot be reached (connects time
out after 10 s). When there is no kept `dataset.yaml`, a kept
`knn_entries.yaml` opens the dataset instead, as the online cascade
would. A warmed cache therefore opens and reads with no network.

In offline mode a partly fetched facet opens too and serves the chunks
it holds; a read of a missing chunk is refused at once, saying offline
mode is on — the refusal is never retried.

Reads of remote chunks retry with backoff (up to 10 attempts, delays
capped at 30 s) when the failure may pass: a lost connection, a
timeout, a server error (5xx), or 408, 425 and 429. A response that
every repeat would get too — any other client error, such as 404 for a
file gone upstream or 401 and 403 for a refused credential — fails the
read on the first attempt.

Staleness is checked where the network is meant to be used: `fetch`
asks the upstream whether each complete copy is still current, and
reports `FacetFetch::upstream_checked = false` when it could not ask; a
changed upstream is an error naming the stale cache. `datasets ping`
does the same and reports an unreachable server as a failure.

A remote catalog file is fetched first too, keeping a copy under
`<cache>/.catalogs/`, named from its URL with special characters
collapsed (`https://example.com:8443/data/catalog.json` is kept as
`example.com_8443_data_catalog.json`); the copy is read when the server
cannot be reached. A cached dataset therefore opens by name with no
network.

A file URL opened with no catalog context (`XvecReader::open(url)`,
`io::open_vec(url)`, `TypedReader::open_auto(url)`) lands in the cache
directory of the dataset whose recorded origin covers it, so it shares
the copy a fetch filled instead of keying a second one.
`XvecReader::open_url` and `TypedReader::open_url` are deprecated in
favour of opening the dataset by catalog or `TestDataGroup`.

#### Offline mode

`VECTORDATA_OFFLINE=1`, or `offline: on` in `settings.yaml`
(`vectordata config set offline on`), makes no request to any dataset
server. Catalogs, definitions and offset indexes come from their kept
copies, data from the cache — a partially fetched file serves the chunks
it holds — and anything not held locally is an error saying offline mode
is on. `fetch` does not revalidate or download. The environment variable
overrides the setting in both directions. Explicit server commands
(`login`, `push`, `backup`) are not affected.

### Per-facet cache stats

`view.open_facet_storage(name)` returns a `FacetStorage` handle —
opaque from the outside but knowing whether the underlying transport
is cached.

| Method | local mmap | cached-remote | direct HTTP (no `.mref`) |
|---|---|---|---|
| `is_local()` | `true` | `true` once promoted | `false` |
| `is_complete()` | `true` | `true` once every chunk verified | `false` (always) |
| `cache_stats()` | `None` | `Some(CacheStats)` | `None` |
| `cache_path()` | the source file | the cache file | the cache file, once fetched |
| `precache()` | no-op | downloads + verifies | downloads |

`cache_path()` is for diagnostics — showing where a facet is cached —
not an access path. A cache file is pre-sized and sparse until
complete, its valid chunks are recorded separately, and a series spans
several files (`cache_path()` names the first); only vectordata's
readers account for that. Read through the facet readers, which are
zero-copy once the facet is fetched.

```rust
use vectordata::CacheStats;

let storage = view.open_facet_storage("base_vectors")?;

if let Some(cs): Option<CacheStats> = storage.cache_stats() {
    let pct = 100.0 * cs.valid_chunks as f64 / cs.total_chunks as f64;
    println!("cached: {:.0}% ({}/{} chunks, {} bytes total)",
        pct, cs.valid_chunks, cs.total_chunks, cs.content_size);
}

// Bring this one facet in, without touching the rest of the profile,
// then read it through its reader — zero-copy once resident.
view.fetch(&FetchRequest::facets(["base_vectors"]), &mut Silent)?;
assert!(storage.is_complete());
let base = view.base_vectors()?;
let first: &[f32] = base.get_slice(0).expect("mapped once fetched");
```

### How chunked verification works

1. **`.mref` file** (published) — precomputed Merkle tree with
   SHA-256 hashes for all fixed-size data chunks (1 MiB by
   default).
2. **`.mrkl` file** (local) — tracks which chunks have been
   downloaded and verified. Persists across restarts; the merkle
   reference is embedded so `.mref` is needed only on first open.
3. **Read path** — check local cache → fetch chunk if missing →
   SHA-256 hash → compare against the merkle leaf → write to
   cache → checkpoint `.mrkl` → return bytes. A checkpoint writes
   only the validity bitset into the existing file, merged under an
   exclusive file lock with whatever other channels on the same
   cache have recorded; the tree ahead of it never changes after
   the file is created and is never rewritten. It runs once per
   fetch call, so it must not scale with the tree: rewriting the
   whole 33.6 MB file per one-MiB chunk once held tessera downloads
   to 6 MB/s (guarded by
   `single_chunk_fetches_do_not_rewrite_the_state_file`).
4. **Promotion** — once every chunk is verified, the cached
   storage flips into a `Mmap` view of the cache file. Subsequent
   reads (via `read_bytes` / `mmap_slice` / `get_slice`) are
   zero-copy with no per-read overhead.
5. **Fallback** — if the URL has no `.mref`, the cache layer is
   bypassed and every read is a direct HTTP RANGE request. Slow
   but correct; `is_complete()` always returns `false` so callers
   can distinguish.

### Listing cached datasets

```bash
veks datasets cache
```

---

## 2.12a Process-wide singleton and connection caching

Two performance/correctness properties consumers can rely on:

**Per-source `Storage` singleton.** Every public reader open
ultimately routes through `Storage::open(source)`, which keys on
canonical source identity (URL or canonicalised path) and returns
the same `Arc<Storage>` for concurrent opens of the same source.
Concrete consequences:

- N threads each calling `view.base_vectors()` against the same
  dataset get readers that share one underlying `Storage` (one
  cache file handle, one `MerkleState`, one mmap promotion).
- Per-source serialization at open time means the `.mref` fetch
  and cache-file open run exactly once even if many threads race
  at session-init.
- A pre-existing `Storage` whose readers all dropped is garbage-
  collected lazily; the next open re-creates it.

This makes the "session-init prebuffer + per-cycle reads from
many threads" pattern safe by construction. The
`many_parallel_opens_of_same_url_dont_corrupt_reads` and
`parallel_opens_share_single_cache_channel` integration tests
lock this in.

**Shared `reqwest::blocking::Client`.** A single client lives
process-wide; every internal HTTP site clones from it. The clone
is cheap (the client is internally `Arc`-wrapped) and shares the
connection pool, DNS cache, and TLS session state. Without this,
each `Client::new()` would trigger
`rustls_native_certs::load_native_certs` (~1 ms on Linux walking
`/etc/ssl/certs/`). The
`many_per_record_reads_share_one_tcp_connection`,
`many_storage_opens_share_one_client_no_cert_load_storm`, and
`shared_client_is_singleton_across_storages` integration tests
lock in the connection-pool reuse property.

## 2.13 Thread Safety

All reader types are `Send + Sync`:

- `VectorReader<T>` and `VvecReader<T>` traits both require
  `Send + Sync`.
- `XvecReader<T>`, `IndexedVvecReader<T>`, and `TypedReader<T>` are
  built on `Arc<Storage>` (each variant — `Mmap`, `Http`, `Cached`
  — is internally thread-safe). Cloning the outer `Arc` shares the
  underlying storage and any promoted-mmap state, so two clones see
  the same `is_complete()` and the same zero-copy slices.
- `Arc<dyn VectorReader<T>>` and `Arc<dyn VvecReader<T>>` returned
  by `TestDataView` methods are safe to share across threads.

Typical parallel access pattern:

```rust
use std::sync::Arc;
use rayon::prelude::*;
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

let group = TestDataGroup::load("https://host/dataset/")?;
let view = group.profile("default").unwrap();
let base = view.base_vectors()?;          // Arc<dyn VectorReader<f32>>

let results: Vec<f64> = (0..base.count())
    .into_par_iter()
    .map(|i| {
        let v = base.get(i).unwrap();
        v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt()
    })
    .collect();
```

---

## 2.14 Profiles

A dataset can have multiple profiles representing different subsets
or configurations of the same data:

- **`default`** — the full dataset, always present.
- **Sized profiles** — subsets like `10K`, `100K`, `1M` with a
  `base_count` that limits the number of base vectors.

### Accessing profiles

```rust
use vectordata::TestDataGroup;
use vectordata::view::TestDataView;

let group = TestDataGroup::load("./my-dataset/")?;

if let Some(view) = group.profile("default") {
    let manifest = view.facet_manifest();
    println!("default profile has {} facets", manifest.len());
}

if let Some(view) = group.profile("100K") {
    let base = view.base_vectors()?;
    println!("100K profile: {} vectors", base.count());
}
```

### Profile YAML structure

```yaml
strata:
  mul:
    spec: "mul:100K..1M/2"
    series: ["100k", "200k", "400k", "800k"]

profiles:
  default:
    maxk: 100
    base_vectors: profiles/base/base_vectors.fvecs
    query_vectors: profiles/base/query_vectors.fvecs
    neighbor_indices: profiles/default/neighbor_indices.ivecs
    metadata_content: profiles/base/metadata_content.u8
    metadata_indices: profiles/default/metadata_indices.ivvecs

  100K:
    base_count: 100000
    maxk: 100
    base_vectors: "profiles/base/base_vectors.fvecs[0..100000)"
    query_vectors: profiles/base/query_vectors.fvecs
    neighbor_indices: profiles/100K/neighbor_indices.ivecs
```

Sized profiles share base/query data — `base_vectors` is windowed
via the `[lo..hi)` suffix rather than copied, and a quantized base's
`quant_*` columns are windowed with it — but have their own
computed KNN, filtered results, and metadata indices. The
pipeline's `per_profile` mechanism generates these automatically;
see §1.7 for the full list of generator strategies (`decade`,
`mul:`, `fib:`, `linear:`, …).

---

## 2.15 Dataset attributes and metadata

### Required attributes

Every published dataset declares these attributes in
`dataset.yaml`:

```yaml
attributes:
  distance_function: L2          # L2, COSINE, or DOT_PRODUCT
  is_zero_vector_free: true      # no zero vectors in base data
  is_duplicate_vector_free: true # no duplicate vectors in base data
```

These are set automatically by the pipeline after scanning or dedup.

### Accessing attributes

```rust
use vectordata::TestDataGroup;

let group = TestDataGroup::load("./dataset/")?;

let dist = group.attribute("distance_function")
    .and_then(|v| v.as_str())
    .unwrap_or("unknown");

let zero_free = group.attribute("is_zero_vector_free")
    .and_then(|v| v.as_bool())
    .unwrap_or(false);

// Distance function is also exposed on the view
let view = group.profile("default").unwrap();
if let Some(df) = view.distance_function() {
    println!("metric: {df}");
}
```

### Metadata facets

```rust
use vectordata::TestDataViewExt;

let view = catalog.open_profile("my-dataset", "default")?;

// Metadata content (e.g., labels as u8): a column, one slice once resident
let meta = view.facet_as::<u8>("metadata_content")?;
println!("{} metadata records", meta.count());
let labels = meta.as_slice().expect("fetched local column");
for (i, label) in labels.iter().take(5).enumerate() {
    println!("  base[{i}] label = {label}");
}

// Predicates (e.g., equality filters as u8)
let pred = view.facet_as::<u8>("metadata_predicates")?;

// Predicate results (variable-length ordinal lists)
let mi = view.metadata_results_as::<i32>()?;
for qi in 0..5 {
    let matching = mi.get(qi)?;
    println!("  query[{qi}] (field_0 == {}): {} matching base vectors",
        pred.get(qi)?[0], matching.len());
}
```

### Element type interrogation

```rust
use vectordata::io::AnyVectorReader;
use vectordata::{ElementType, TestDataViewExt};

let view = catalog.open_profile("my-dataset", "default")?;

// Without opening: from the extension
let etype: ElementType = view.facet_element_type("metadata_content")?;

// Opened at its own type, then chosen
match view.open_facet_native("metadata_content")? {
    AnyVectorReader::U8(r) => { let val: u8 = r.get(0)?[0]; }
    AnyVectorReader::I32(r) => { let val: i32 = r.get(0)?[0]; }
    other => {
        // Any integer type of 32 bits or fewer widens to i64
        let r = other.as_type::<i64>()?;
        let val: i64 = r.get(0)?[0];
    }
}
```
