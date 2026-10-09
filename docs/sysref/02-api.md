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

// Open a dataset by name → get vectors in two calls
let group = catalog.open("my-dataset")?;
let view = group.profile("default").unwrap();
let base = view.base_vectors()?;
println!("{} vectors, dim={}", base.count(), base.dim());
let v: Vec<f32> = base.get(42)?;

// Or even shorter — open a profile directly
let view = catalog.open_profile("my-dataset", "default")?;
let gt = view.neighbor_indices()?;
```

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

let base = view.base_vectors()?;           // Arc<dyn VectorReader<f32>>
let gt   = view.neighbor_indices()?;       // Arc<dyn VectorReader<i32>>
let mi   = view.metadata_indices()?;       // Arc<dyn VvecReader<i32>>
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
        entry.name,
        entry.profile_names().join(", "));
}

// Find by exact name (case-insensitive)
if let Some(entry) = catalog.find_exact("my-dataset") {
    println!("found: {}, path: {}", entry.name, entry.path);
}

// Find by glob pattern
let matches = catalog.match_glob("my-vectors*");
for entry in matches {
    println!("  {}", entry.name);
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

---

## 2.3 File Extension Scheme

See [SRD §22](22-vector-file-extensions.md) for the full
specification.

### Summary

| Suffix | Structure | Example |
|--------|-----------|---------|
| `.<type>` | Scalar (flat packed, no header) | `.u8`, `.i32`, `.f64` |
| `.<type>vec` | Uniform vector (fixed dimension per record) | `.fvecs`, `.ivecs`, `.u8vecs` |
| `.<type>vvec` | Variable-length vector (per-record dimension) | `.ivvecs`, `.fvvecs`, `.u8vvecs` |

Legacy aliases: `.fvecs`=`.f32vecs`, `.ivecs`=`.i32vecs`,
`.bvecs`=`.u8vecs`, `.svecs`=`.i16vecs`, `.mvecs`=`.f16vecs`,
`.dvecs`=`.f64vecs`.

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

Opens a uniform vector file for typed random access. Local paths
mmap; URLs go through the merkle-cached path when a `.mref` is
published, falling back silently to direct HTTP RANGE otherwise.

```rust
use vectordata::io::{open_vec, VectorReader};

let r = open_vec::<f32>("data/base.fvecs")?;             // local mmap
let r = open_vec::<i32>("https://host/neighbors.ivecs")?; // cached or direct HTTP

println!("count={}, dim={}", r.count(), r.dim());
let vec: Vec<f32> = r.get(0)?;
```

**Supported types:** `f32`, `f64`, `half::f16`, `i32`, `i16`, `i8`,
`u8`, `u16`, `u32`, `u64`, `i64`.

The type parameter `T` must match the file's element width. A
mismatch (e.g., `open_vec::<f32>("data.dvecs")` where `.dvecs` is
8-byte f64) returns an error at open time.

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

    // drive underlying storage to fully resident; no-op for local/direct-HTTP
    fn prebuffer(&self) -> std::io::Result<()> { Ok(()) }

    // true for local; true for cached once every chunk is verified; false for direct-HTTP
    fn is_complete(&self) -> bool { true }
}
```

The canonical concrete implementation is `XvecReader<T>`. It also
exposes hot-path inherent methods that the `dyn VectorReader<T>`
surface can't: an unchecked `get_slice(index) -> &[T]` (panics if
the storage is not mmap-backed; intended for KNN inner loops), plus
`advise_sequential` / `advise_random`, `prefetch_range`,
`release_range`, and `prefetch_pages` (madvise hints; no-op when
not mmap-backed).

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

### `VvecElement` — byte decoding

```rust
pub trait VvecElement: Copy + Send + Sync + 'static {
    const ELEM_SIZE: usize;
    fn from_le_bytes(bytes: &[u8]) -> Self;
}
```

Implemented for: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`,
`i64`, `f32`, `f64`, `half::f16`.

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
```

### Profile access

```rust
// Trait object — the universal handle. Use this for almost everything.
let view: Arc<dyn TestDataView> = group.profile("default").unwrap();

// Concrete type — only needed for the typed open_facet_typed::<T> method
// when you don't want to use the free function. See §2.7.
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

All methods on `TestDataView` return readers that work identically
for local and remote datasets:

```rust
// Uniform vector facets → Arc<dyn VectorReader<T>>
let base:  Arc<dyn VectorReader<f32>> = view.base_vectors()?;
let query: Arc<dyn VectorReader<f32>> = view.query_vectors()?;
let gt:    Arc<dyn VectorReader<i32>> = view.neighbor_indices()?;
let dist:  Arc<dyn VectorReader<f32>> = view.neighbor_distances()?;
// F facet — pre-filter ground truth (ACORN G_K).
let fki:   Arc<dyn VectorReader<i32>> = view.prefiltered_neighbor_indices()?;
let fkd:   Arc<dyn VectorReader<f32>> = view.prefiltered_neighbor_distances()?;
// E facet — post-filter ground truth (G ∩ R).
let pki:   Arc<dyn VectorReader<i32>> = view.postfiltered_neighbor_indices()?;
let pkd:   Arc<dyn VectorReader<f32>> = view.postfiltered_neighbor_distances()?;

// Variable-length facet → Arc<dyn VvecReader<i32>>
let mi: Arc<dyn VvecReader<i32>> = view.metadata_indices()?;

// Metadata configuration (declared facets, no data access)
let meta_cfg: Option<&FacetConfig> = view.metadata_content();
let pred_cfg: Option<&FacetConfig> = view.metadata_predicates();

// Facet discovery
let manifest: HashMap<String, FacetDescriptor> = view.facet_manifest();

// Element type interrogation
let etype = view.facet_element_type("metadata_content")?;  // ElementType::U8

// Resolved source path/URL for any declared facet
let src: Option<String> = view.facet_source("metadata_content");

// Fetch into the cache / inspect it
view.fetch(&FetchRequest::all(), &mut Silent)?;
let storage = view.open_facet_storage("base_vectors")?;    // FacetStorage
```

---

## 2.7 Typed scalar access

Metadata (M) and predicates (P) are scalar files — flat-packed
arrays with no per-record header. Read them as type-checked values
via `TypedReader<T>`, with widening conversions and runtime
overflow checks.

### From a `TestDataView` (catalog or dyn handle)

The free function `open_facet_typed::<T>` works against any
`&dyn TestDataView`:

```rust
use vectordata::{open_facet_typed, TypedReader};

let view = catalog.open_profile("my-dataset", "default")?;

// Open with native type — zero-copy when storage is mmap-backed
let r: TypedReader<u8> = open_facet_typed(&*view, "metadata_content")?;
let val: u8 = r.get_native(42);

// Open with a wider type — widening always succeeds
let r: TypedReader<i32> = open_facet_typed(&*view, "metadata_content")?;
let val: i32 = r.get_value(42)?;
```

### From a known-concrete `GenericTestDataView`

The same call exists as a method on `GenericTestDataView`:

```rust
let gview = group.generic_view("default").unwrap();
let r: TypedReader<u8> = gview.open_facet_typed::<u8>("metadata_content")?;
```

### Direct file open

```rust
use vectordata::typed_access::{ElementType, TypedReader};

// Local path, native type from extension
let r = TypedReader::<u8>::open("metadata.u8")?;

// Remote URL, native type explicit (cache-first; falls back to direct HTTP)
let r = TypedReader::<i32>::open_url(
    url::Url::parse("https://host/metadata.u8")?,
    ElementType::U8,
)?;

// Path-or-URL string, dispatched automatically
let r = TypedReader::<i64>::open_auto("https://host/metadata.i32",
    ElementType::I32)?;
```

### Width compatibility

| target T | native type | result |
|---|---|---|
| same width, same sign | exact match | zero-copy `get_native` works |
| same width, cross sign | e.g. `u8` ↔ `i8` | checked per value via `get_value`, fails on overflow |
| wider | e.g. `i32` from `u8` | widening; `get_value` always succeeds |
| narrower | e.g. `u8` from `i32` | rejected at open time (`Narrowing` error) |

---

## 2.8 Facet Reference

| Facet code | YAML key | Trait method | Reader type | Typical format |
|-----------|----------|-------------|-------------|---------------|
| B | `base_vectors` | `base_vectors()` | `VectorReader<f32>` | `.fvecs` |
| Q | `query_vectors` | `query_vectors()` | `VectorReader<f32>` | `.fvecs` |
| G | `neighbor_indices` | `neighbor_indices()` | `VectorReader<i32>` | `.ivecs` |
| D | `neighbor_distances` | `neighbor_distances()` | `VectorReader<f32>` | `.fvecs` |
| M | `metadata_content` | `metadata_content()` | config + `open_facet_typed` | `.u8`, `.slab` |
| P | `metadata_predicates` | `metadata_predicates()` | config + `open_facet_typed` | `.u8`, `.slab` |
| R | `metadata_indices` | `metadata_indices()` | `VvecReader<i32>` | `.ivvecs` |
| F (indices) | `prefiltered_neighbor_indices` (canonical) or `filtered_neighbor_indices` (legacy alias) | `prefiltered_neighbor_indices()` | `VectorReader<i32>` | `.ivecs` |
| F (distances) | `prefiltered_neighbor_distances` (canonical) or `filtered_neighbor_distances` (legacy alias) | `prefiltered_neighbor_distances()` | `VectorReader<f32>` | `.fvecs` |
| E (indices) | `postfiltered_neighbor_indices` | `postfiltered_neighbor_indices()` | `VectorReader<i32>` | `.ivecs` |
| E (distances) | `postfiltered_neighbor_distances` | `postfiltered_neighbor_distances()` | `VectorReader<f32>` | `.fvecs` |

For M and P facets, use `open_facet_typed::<T>(view, name)` for
typed data access (see §2.7).

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
where they should. A spec with no selector means `default`.

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

The `knn_entries` module can also be used directly:

```rust
use vectordata::knn_entries::KnnEntries;

let entries = KnnEntries::load("knn_entries.yaml")?;
println!("datasets: {:?}", entries.dataset_names());
let config = entries.to_config();  // → DatasetConfig
```

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
}
```

`VariableLengthRecords` is returned by `open_vec` when the file
turns out to have variable-length records (its size is not a
multiple of the implied stride). The caller should switch to
`open_vvec`.

---

## 2.12 Prebuffering and caching

### Cache location

Remote downloads are cached under `vectordata::settings::cache_dir()`,
the single source of truth for cache resolution shared with the
`veks-pipeline` crate. Resolution order:

1. A cache directory set for the process with
   `vectordata::settings::set_cache_dir`.
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
via the `[lo..hi)` suffix rather than copied — but have their own
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
use vectordata::open_facet_typed;

let view = catalog.open_profile("my-dataset", "default")?;

// Metadata content (e.g., labels as u8)
let meta = open_facet_typed::<u8>(&*view, "metadata_content")?;
println!("{} metadata records", meta.count());
for i in 0..5 {
    println!("  base[{}] label = {}", i, meta.get_native(i));
}

// Predicates (e.g., equality filters as u8)
let pred = open_facet_typed::<u8>(&*view, "metadata_predicates")?;

// Predicate results (variable-length ordinal lists)
let mi = view.metadata_indices()?;
for qi in 0..5 {
    let matching = mi.get(qi)?;
    println!("  query[{qi}] (field_0 == {}): {} matching base vectors",
        pred.get_native(qi), matching.len());
}
```

### Element type interrogation

```rust
use vectordata::typed_access::ElementType;
use vectordata::open_facet_typed;

let view = catalog.open_profile("my-dataset", "default")?;

let etype = view.facet_element_type("metadata_content")?;
match etype {
    ElementType::U8 => {
        let r = open_facet_typed::<u8>(&*view, "metadata_content")?;
        let val: u8 = r.get_native(0);
    }
    ElementType::I32 => {
        let r = open_facet_typed::<i32>(&*view, "metadata_content")?;
        let val: i32 = r.get_native(0);
    }
    _ => {
        // Widen to i64 for any integer type
        let r = open_facet_typed::<i64>(&*view, "metadata_content")?;
        let val: i64 = r.get_value(0)?;
    }
}
```
