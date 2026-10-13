# SRD — Datasets, entries and catalogs built at runtime, through one path

**Status:** proposed 2026-10-12. All open decisions in §11 were
accepted on 2026-10-12. Not implemented.
**Scope:** the `vectordata` crate's dataset, catalog-entry and catalog
types, how they are built (from files, URLs, content or code), the set
of catalogs a program resolves against, and the explorer's catalogs
screen. It covers every producer in the workspace (`veks`,
`veks-pipeline`) that builds these types today, because the point is a
single construction path.

**Origin.** An embedding program (`fastervector`) ships a vendored
catalog and cannot get it into vectordata without a file path compiled
into its binary, since every way into a catalog is a location. That is
one instance of a general gap: **a program cannot bind a dataset, a
catalog entry or a catalog at runtime.** This SRD addresses the general
requirement. The embedder is one consumer of it and does not shape the
interface.

---

## 1. Problem

A program that uses vectordata as a library can only reach data through
locations: a `catalogs.yaml` directory, a catalog file or directory, a
`dataset.yaml` path or URL. It cannot:

- describe a dataset in code, or from text it already holds, and open it;
- add a catalog entry it built itself to the set of catalogs;
- register a whole catalog (named, ordered, with provenance) that has no
  file or URL on the host;
- see such a catalog in the explorer's catalogs screen next to the
  configured ones.

Behind that public gap is an internal one. **There is no construction
API, so every internal producer builds these types its own way.** The
inventory in §2 found, for the same few types:

- 8 hand-written `TestDataGroup` literals across 4 load routes;
- 2 `DatasetConfig` types with 2 independent profile-inheritance
  implementations;
- 2 `knn_entries.yaml` parsers that disagree on what they support;
- 2 copies of catalog generation that disagree on path handling and
  exclusions;
- 5 places that write `dataset.yaml` as hand-built strings;
- 7 places that assemble a `CatalogSources`, with different severity
  rules;
- 6 trailing-slash helpers and 3 `file://` strippers.

A public builder added on top of that would be a third way to build
each thing. The requirement is the reverse: **the public construction
API becomes the only construction path**, and the file, URL and
generator routes are the same builders fed by a parser or a scan.

## 2. What exists today

Line numbers are as of 2026-10-12. Paths are under `crates/`.

### 2.1 Dataset descriptions

- **Two `DatasetConfig` types:**
  - `vectordata/src/model.rs:33` `model::DatasetConfig` is the consumer
    side, used by `TestDataGroup`: a `HashMap` of `ProfileConfig` with
    `FacetConfig` facets. Its custom `Deserialize` (model.rs:191-308)
    applies default and named-parent inheritance, validates shards, and
    silently skips unparseable profiles.
  - `vectordata/src/dataset/config.rs:117` `dataset::DatasetConfig` is
    the producer side: name, description, `DatasetAttributes`, a
    `DSProfileGroup`, `upstream`, `strata`, `variables`. Its
    `DSProfileGroup` `Deserialize` (profile.rs:2043-2342) implements
    inheritance and `sized:` expansion again, independently.
- **`dataset.yaml` writers:**
  - `DatasetConfig::save` / `save_expanded` (config.rs:646-684) render
    from the struct and keep an author's leading comment block.
  - Five generators hand-build the YAML as strings instead:
    `veks/src/prepare/import.rs:2577` (`generate_yaml`, the bootstrap
    path), `veks-pipeline/.../gen_dataset.rs:211`, `gen_derive.rs:240`,
    `gen_example_dataset.rs:353`, `vectordata/src/datasets/derive.rs:1795`.
    Four of them write comments: bootstrap explains each strata spec
    and the tag schema, and `example-dataset` and `derive` start with a
    header. The renderer kept only a file's existing leading comment
    block, so bootstrap's comments lasted until the first save (§12.3).
    *(Corrected 2026-10-12; this said none did.)*
  - In-place text edits through `yaml_edit` (tags, predicate sets,
    downgrade, runner, check fix) edit an existing file. They are edits,
    not constructions, and stay as they are (§8).
- **`derive.rs:1934` `load_rich_config`** re-implements
  `DatasetConfig::load` so it can read from a URL, and derive parses the
  same `dataset.yaml` twice, once per type.

### 2.2 Catalog entries

- **`CatalogEntry`** (`dataset/catalog.rs:77`) is public fields with no
  constructor. It is built by struct literal in `remap_entry`
  (catalog/resolver.rs:741, :779), the catalog `knn_entries` parser
  (catalog/knn_entries.rs:186), and both catalog generators.
- **Two kinds of entry share one struct and one string discriminator:**
  - A canonical entry (`dataset_type: dataset.yaml`) is a summary. Its
    full description is fetched from `path`. `Catalog::open_entry`
    discards the embedded layout and calls `TestDataGroup::load(path)`.
  - A `knn_entries` entry carries the whole description.
    `TestDataGroup::from_catalog_entry` turns it into a dataset by
    serializing the layout to YAML and parsing it back as
    `model::DatasetConfig`.
- **Two `knn_entries.yaml` parsers:**
  - `vectordata/src/knn_entries.rs` (`KnnEntries`, used by
    `TestDataGroup::load*`) stores `_defaults.base_url` but never applies
    it, requires exactly `base`/`query`/`gt`, and ignores `cache_dir` and
    `${VAR}`.
  - `vectordata/src/catalog/knn_entries.rs` (used by the catalog
    resolver) supports all of those.
  - So the same file opens differently depending on whether it was
    reached through a catalog or by path.
- **Two catalog generators:**
  - `veks/src/catalog/generate.rs` is the manual command
    `veks prepare catalog generate`. It reads each dataset with
    `DatasetConfig::load`, removes `.` path components, and avoids
    `canonicalize`.
  - `veks-pipeline/.../catalog_generate.rs` is the pipeline command
    `catalog generate`. It runs as the `gen-catalog` step of every
    dataset's finalize, so it writes the catalogs that get published. It
    reads each dataset with `load_and_resolve`, adds template-derived
    views, canonicalizes its input directory, and alone writes
    `knn_entries.yaml`.
  - Both honour `.do_not_catalog`, but differently (§11, item 3).
  - The catalog-root and publish-URL finders are copied four times.

### 2.3 Datasets (`TestDataGroup`)

- **Routes:**
  - `load` dispatches on an `http(s)://` prefix only, so `s3://` and
    `file://` fall into `load_from_path`.
  - `load_from_path` has 4 branches: explicit YAML, `dataset.yaml`,
    `knn_entries.yaml`, and an unreachable fallback whose
    `catalog_source` lacks the trailing slash the others add.
  - `load_from_url` has 3 branches.
  - Then `from_catalog_entry`.
  - That is 8 literals in total, with the "last non-empty path segment"
    extraction pasted 5 times.
- **Cache identity:**
  - `catalog_source` (the dataset's home URL) and `dataset_name` decide
    the cache directory.
  - `CatalogEntry::dataset_home_url` (catalog.rs:119) is documented as
    the single authority, but the `load*` literals derive the home
    independently. They only agree when `entry.path` is a `dataset.yaml`
    URL.
  - For a canonical entry, `open_entry` goes through `load(path)`, so
    `dataset_name` comes from the URL segment, not `entry.name`.

### 2.4 Catalogs and the set of catalogs

- **`Catalog::of(&CatalogSources)`** (resolver.rs:46) is the only
  constructor. `CatalogSources` holds `NamedCatalogSource { name,
  location }` pairs, and every method takes a location.
- **Loading is three crate-private loaders**, each repeating fetch,
  parse and provenance stamping: `load_from_explicit_catalog_file`,
  `try_load_canonical_catalog`, `try_load_knn_entries`.
  `try_load_knn_entries` checks only `http(s)`, not `s3`.
- **Provenance:**
  - `catalog_name` is stamped in one place (resolver.rs:64).
  - `catalog_file` is stamped for `knn_entries` shapes only.
- **`CatalogSources` is assembled in 7 places with different rules:**
  `datasets/mod.rs:51` `build_sources` and an inline copy at
  `datasets/cache.rs:173`, `describe.rs:44`, `ping.rs:55`,
  `dyncomp.rs:126`, `veks/src/datasets/mod.rs:566`, and four explorer
  sites. A configured catalog that fails to load is reported for
  `datasets list`, but skipped silently (debug log only) by the
  explorer, completion and `veks ping` (§11, item 4).

### 2.5 The explorer's catalogs screen

- **Where the data comes from:**
  - `run_picker` (explore/dataset_picker.rs:2345) builds its own
    `CatalogSources::new().configure_default()`.
  - The Ctrl-G screen takes its name-to-location list from
    `named_catalog_entries(config_dir())`: the raw `catalogs.yaml`,
    unresolved.
- **Gaps:**
  - A catalog the program assembled, rather than configured, cannot
    appear.
  - Names that come from a nested `catalogs.yaml` appear in the rows but
    not in the screen's location list.
- **Edits:** add/remove go through `config.rs`, which re-reads
  `catalogs.yaml` in five places, with a verbatim duplicate of
  `named_catalog_entries` at config.rs:241. After an edit the picker
  restarts and rebuilds the catalog from scratch.

## 3. Principles

1. **One construction path per type.** A file, a URL, a string of
   content, a directory scan and a program's own code all end in the same
   builder. A parser's job is to read text and drive a builder, never to
   produce the type by another route.
2. **Built and loaded are indistinguishable downstream.** A dataset's
   behaviour — lookup, selection, fetch, cache directory, `origin.json`
   — depends on its description and its home location, never on how
   the description arrived.
3. **One description type per level.** A dataset has one description
   type. Anything the consumer needs is a projection of it, not a second
   type parsed in parallel.
4. **Locations are values, not strings.** Joining, parents, trailing
   slashes, `file://` and remote schemes are handled by one type, once.
5. **The set of catalogs is the program's, not the config file's.**
   `catalogs.yaml` is one source of catalogs among others. Every surface
   that shows catalogs, the explorer included, reads the set the program
   assembled.
6. **Extension is by trait.** A program can contribute a catalog from
   anywhere — code, embedded content, a database — by implementing one
   trait or using the in-memory implementation the crate ships.

## 4. Requirements

IDs are `RC-n`.

### Locations

- **RC-1** `Location` is a public value type for where a thing lives:
  a local path, `file://`, `http(s)://` or `s3://`. It provides `parse`,
  `join(relative)`, `parent`, `as_dir` (trailing slash), `is_remote`,
  `last_segment` and `Display`. Every location operation in §2 goes
  through it: the six trailing-slash helpers, the three `file://`
  strippers, `parent_location_of`, `relative_to`, `data_source_for`,
  `resolve_path_str`/`resolve_as_string`, and the string concatenations
  in `remap_entry` and `resolve_facet_url`.
- **RC-2** Runtime facet resolution and catalog-time facet resolution
  (the picker, `datasets list` filters, completion) use the same
  `Location::join`. Today one uses `Url::join` and the other string
  concatenation, and they can disagree.

### Datasets

- **RC-3** A dataset has one description type, `dataset::DatasetConfig`.
  `model::DatasetConfig` becomes a projection computed from it by one
  function. It is no longer parsed separately and has no inheritance
  logic of its own. The two inheritance implementations become one, the
  one in `DSProfileGroup`.
- **RC-4** `DatasetConfig::builder(name)` builds a description
  fluently:
  - `description`, `attribute(key, value)` and typed shorthands such as
    `distance_function`, `model` and `license`;
  - `profile_tags`;
  - `profile(name, |p| …)`, where the profile builder sets facets by
    `StandardFacet` or custom key (`facet(StandardFacet::BaseVectors,
    "base.fvec")`, with windows), counts and a parent;
  - `strata`, `variables` and `upstream`.

  `build()` runs the same validation the deserializer runs: the version
  gate, parents and shards. A built config and a parsed one are equal
  when they say the same thing.
- **RC-5** `DatasetConfig::parse(text)` reads `dataset.yaml` content
  from memory and is what `load` uses after reading the file.
  `load_rich_config`'s URL read becomes `DatasetConfig::load_from`
  (`&Location`), replacing the copy in derive.
- **RC-6** Every producer that writes a `dataset.yaml` builds it with
  the builder and writes it with `save`. That covers the five
  string-building generators in §2.1.
- **RC-7** `TestDataGroup` has one constructor, from a resolved
  `DatasetHandle { name, home: Location, description: DatasetConfig }`.
  - `load`, `load_from_path`, `load_from_url` and `Catalog::open_entry`
    each resolve their input to a handle and call it.
  - The name and home are derived once, by `Location::last_segment` and
    `CatalogEntry::dataset_home_url`.
  - `load` dispatches on `Location`, so `s3://` and `file://` resolve
    correctly.
  - `profile()` and `generic_view()` share one view constructor.
- **RC-8** `TestDataGroup::from_config(name, home, config)` is the
  public form of RC-7, for a program that built or parsed a description
  itself. `home` is where relative facet paths resolve and what the cache
  mirrors.

### Catalog entries

- **RC-9** An entry says what it holds as a type, not a string:
  ```rust
  pub enum EntryDescription {
      /// The whole description is here (knn_entries shape, built in code).
      Embedded(DatasetConfig),
      /// A summary; the description is fetched from `descriptor` when opened.
      Referenced { descriptor: Location, summary: CatalogLayout },
  }
  ```
  `open_entry` matches on it. The YAML round-trip in
  `from_catalog_entry` goes away. `dataset_type` is still read and
  written in catalog files (§8), but is derived from the variant, not
  stored.
- **RC-10** `CatalogEntry::builder(name)` takes a `home` (`Location`),
  then either `.dataset(DatasetConfig)` (embedded) or
  `.descriptor(Location, summary)` (referenced). Provenance fields are
  private and set only by the catalog that adopts the entry (RC-14).
  `remap_entry`, the `knn_entries` parser and both catalog generators
  build entries through it.
- **RC-11** There is one `knn_entries` parser, the catalog one. Opening
  a `knn_entries.yaml` by path parses it into entries and opens the one
  matching the directory name (today's preference rule) through RC-7.
  `crate::knn_entries::KnnEntries` is removed, which fixes the ignored
  `_defaults.base_url`, `cache_dir` and `${VAR}` for path opens.
- **RC-12** There is one catalog generator, in `vectordata` (or a
  module shared by `veks` and `veks-pipeline`), used by both commands.
  The divergences in §2.2 are settled once (§11, item 3).

### Catalogs

- **RC-13** The catalog trait:
  ```rust
  pub trait CatalogSource: Send + Sync {
      /// The catalog's symbolic name; what provenance and the catalogs screen show.
      fn name(&self) -> &str;
      /// Where it came from, for display and diagnostics.
      fn origin(&self) -> CatalogOrigin;
      /// Its entries. Problems go into `diagnostics`, never stdout.
      fn entries(&self, diagnostics: &mut Vec<CatalogDiagnostic>) -> Vec<CatalogEntry>;
  }
  pub enum CatalogOrigin {
      /// A file, directory or URL — optionally bound to the catalogs.yaml line that named it.
      Located { location: Location, configured_in: Option<Location> },
      /// Built in memory; `label` is what diagnostics and the catalogs screen show.
      Registered { label: String },
  }
  ```
- **RC-14** The crate ships two implementations:
  - `LocatedCatalog` is today's loaders, collapsed into one: fetch by
    `Location`, then dispatch on content shape to the canonical or
    `knn_entries` parser.
  - `BuiltCatalog` comes from `CatalogBuilder`.

  Both parsers take content plus a base `Location` and drive
  `CatalogEntry::builder`, so a located and a built catalog with the
  same content give equal entries.
- **RC-15** `CatalogBuilder::named(name)`:
  - `.base(Location)`, where relative entry paths and facets resolve;
  - `.entry(CatalogEntry)` and `.entries(iter)`;
  - `.content(text)`, parsed by shape as in RC-14;
  - `.label(text)` for `CatalogOrigin::Registered`, default
    `registered:<name>`.

  `.build()` gives a `BuiltCatalog`. Content with a relative path and no
  base is a diagnostic at build time, not a silent mis-resolution.
- **RC-16** `CatalogSources` holds an ordered list of `(Arc<dyn
  CatalogSource>, Severity)`. The name stays: it now holds catalog
  sources in the trait sense.
  - `.with(source)` and `.with_optional(source)` add any implementation.
  - `configure*` and `add_catalogs` stay as shorthand that adds
    `LocatedCatalog`s.
  - Order is priority order. Registered and located catalogs collide and
    report provenance the same way: a name declared twice is
    `AmbiguousDataset`, never shadowed.
- **RC-17** `Catalog::of` asks each source for its entries and stamps
  `catalog_name` and the source's origin on every entry. That is the
  only place provenance is set. `catalog_file` becomes the origin, so
  canonical entries carry it too. The explorer's source view reads the
  origin instead of re-deriving a `knn_entries.yaml` path.
- **RC-18** The CLI assembles its sources in one function,
  `datasets::build_sources`, with one severity rule (§11, item 4). The
  six other assembly sites call it.

### The catalogs screen

- **RC-19** `run_picker` and the other explorer entry points take the
  `CatalogSources` the program assembled. They never call
  `configure_default` themselves. An embedder that opens the explorer
  over its own sources sees exactly those.
- **RC-20** The Ctrl-G screen lists `CatalogSources` in order, with each
  one's name, origin and dataset count. Configured catalogs that
  resolved to nothing keep showing, because they are in the set even
  when they produce no entries.
  - **Located and configured:** remove, edit and disable act on the
    `catalogs.yaml` line it is bound to, as today.
  - **Registered, or located but not configured:** shown read-only,
    marked as provided by the program.
- **RC-21** After an edit to `catalogs.yaml`, the screen rebuilds only
  the configured part of the set. Registered sources survive the reload,
  since they are not in the file.
- **RC-22** `config.rs` reads `catalogs.yaml` through one function. Its
  duplicate `load_catalog_entries` goes, and `verify_catalog_source`
  names the source with the user's chosen name rather than a synthesized
  `"0"`.

## 5. Design

### 5.1 Layering

```
Location ──────────────────────────────────────────────────────────┐
DatasetConfig  ← builder · parse(text) · load_from(&Location)      │
   └─ projection → model::DatasetConfig (consumer view)            │
CatalogEntry   ← builder(name).home(..).dataset(..)|.descriptor(..)│
CatalogSource (trait)                                              │
   ├─ LocatedCatalog   fetch(Location) → parse(content, base) ─────┤
   └─ BuiltCatalog     CatalogBuilder: entries | content+base ─────┘
CatalogSources  ordered (source, severity) · configure* · with(..)
Catalog::of     entries + provenance stamp · lookup · open_entry
TestDataGroup   from DatasetHandle{name, home, description}
```

Every arrow into a type is a builder call. The parsers (`dataset.yaml`,
`catalog.json`/`.yaml`, `knn_entries.yaml`, `catalogs.yaml`) sit beside
the builders and call them.

### 5.2 Using it

Register a catalog from content the program holds, ahead of the user's
own:

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

Build one dataset in code and register it:

```rust
let ds = DatasetConfig::builder("demo-10k")
    .distance_function("COSINE")
    .profile("default", |p| p
        .facet(StandardFacet::BaseVectors, "base.fvec")
        .facet(StandardFacet::QueryVectors, "query.fvec")
        .facet(StandardFacet::NeighborIndices, "gt.ivec"))
    .build()?;
let entry = CatalogEntry::builder("demo-10k")
    .home(Location::parse("/data/demo-10k/")?)
    .dataset(ds)
    .build();
let sources = CatalogSources::new()
    .with(CatalogBuilder::named("local").entry(entry).build());
```

Open a dataset without any catalog:

```rust
let group = TestDataGroup::from_config("demo-10k", home, ds)?;
```

### 5.3 How each existing site moves

| Today | Becomes |
|---|---|
| `TestDataGroup::load*` (8 literals) | resolve to a `DatasetHandle`, then the one constructor (RC-7) |
| `from_catalog_entry` YAML round-trip | `EntryDescription::Embedded` projected directly (RC-3, RC-9) |
| `crate::knn_entries::KnnEntries` | removed; the catalog parser plus "open the entry matching the dir" (RC-11) |
| `remap_entry`, catalog knn parser literals | `CatalogEntry::builder` (RC-10) |
| `load_from_explicit_catalog_file`, `try_load_canonical_catalog`, `try_load_knn_entries` | `LocatedCatalog` fetch and one content dispatch (RC-14) |
| two catalog generators | one shared generator over `CatalogEntry::builder` (RC-12) |
| five hand-built `dataset.yaml` writers | `DatasetConfig::builder` and `save` (RC-6) |
| `derive.rs` `load_rich_config` and the double parse | `DatasetConfig::load_from`, one parse, projected (RC-3, RC-5) |
| seven `CatalogSources` assemblies | `datasets::build_sources` (RC-18) |
| picker's own `configure_default` and raw `named_catalog_entries` | sources passed in; the screen lists the set (RC-19, RC-20) |
| trailing-slash, `file://` and join helpers | `Location` (RC-1) |

## 6. Bugs this fixes

Found by the inventory. Each gets a guard test that fails on the old
code.

- **B-1** `TestDataGroup::load("s3://…")` and `load("file://…")` are
  treated as local paths (§2.3). RC-7.
- **B-2** `try_load_knn_entries` ignores `s3://` locations
  (resolver.rs:665). RC-14.
- **B-3** A `knn_entries.yaml` opened by path ignores
  `_defaults.base_url`, `cache_dir` and `${VAR}`, which the same file
  honours when reached through a catalog (§2.2). RC-11.
- **B-4** Runtime facet resolution (`Url::join`) and catalog-time
  resolution (string concatenation) can name different URLs for the
  same facet (§2.3). RC-2.
- **B-5** The explorer's action dispatch looks a dataset up
  case-sensitively (explore/mod.rs:463), while `Catalog::lookup` is
  case-insensitive. RC-19 routes it through `lookup`.
- **B-6** The catalogs screen omits locations for names that come from
  a nested `catalogs.yaml`. RC-20.
- **B-7** `verify_catalog_source` names the catalog `"0"` instead of
  the user's name. RC-22.
- **B-8** `try_load_canonical_catalog` never stamps `catalog_file`, so
  the source view fabricates a path for canonical catalogs. RC-17.

## 7. Order of work

Each step leaves the workspace green and is its own commit (or a few).

1. **`Location`** (RC-1, RC-2), with B-1, B-2 and B-4 guards. Mechanical
   replacement of the helpers.
2. **One dataset description** (RC-3, RC-5): the projection, removal of
   the consumer inheritance and parsing, `load_from`, and derive's
   single parse. A parity test runs every `dataset.yaml` fixture in the
   repo through old and new and requires equal projections before the
   old code is deleted.
3. **`DatasetConfig::builder`** (RC-4), then the five writers (RC-6).
   Each writer's output is compared byte for byte against today's
   before the switch.
4. **`TestDataGroup` single constructor** (RC-7, RC-8).
5. **`EntryDescription`, the entry builder, one `knn_entries` parser**
   (RC-9 to RC-11), with the B-3 guard.
6. **The `CatalogSource` trait, `LocatedCatalog`, `CatalogBuilder`,
   `CatalogSources::with`, and provenance** (RC-13 to RC-17), with the
   B-8 guard.
7. **One catalog generator** (RC-12). Catalog output is compared against
   today's for the repo fixtures.
8. **`build_sources` everywhere** (RC-18), then **the explorer**
   (RC-19 to RC-22), with the B-5 to B-7 guards.
9. Docs: `AGENTS.md` common-tasks rows ("Register a catalog at
   runtime", "Build a dataset description in code"), sysref §2.2 and
   §3.1, crate rustdoc, an `examples/` program, and new
   `FETCH_BUDGET`/discoverability entries if the guards require them.

## 8. What must not change

- **The cache layout** (`<root>/<dataset>/<files>`, `.publish_url`,
  `origin.json`). The cache directory a dataset uses today is the one it
  uses after. This is checked by test (§9) and never changed by this
  work without the user's sign-off; see §11, item 1.
- **File formats:** `catalog.json`/`catalog.yaml`, `knn_entries.yaml`,
  `dataset.yaml` and `catalogs.yaml`, read and written. Generated
  catalogs and `dataset.yaml` files are byte-identical to today's for
  the same input, unless a §11 decision says otherwise.
- **In-place edits** of user files stay text edits that keep comments
  (`yaml_edit`, catalogs.yaml line edits). A builder never round-trips a
  file a user maintains.
- **Ambiguity:** a name in two catalogs is `AmbiguousDataset`, whatever
  kind each catalog is.
- **Diagnostics:** catalog problems stay values in `diagnostics()`;
  library code does not print.

## 9. Acceptance tests

- **Built equals loaded.** One catalog's content goes in once as a file
  (`LocatedCatalog`) and once through `CatalogBuilder::content`. The
  entries are equal; `lookup`, `select`, `open_entry`, the cache
  directory of every facet, and the `origin.json` written by a fetch are
  identical. The same holds for a dataset built with the builder versus
  parsed from equivalent YAML.
- **Registration order.** A registered catalog ahead of a configured
  one, both declaring the same dataset, gives `AmbiguousDataset` with
  both names and origins. A registered `CatalogOrigin` reads
  `registered:<name>` (or the label).
- **Relative paths need a base.** Content with relative paths and no
  `.base` yields a diagnostic, not entries pointing at the working
  directory.
- **Parity corpus.** Every `dataset.yaml`, `catalog.*` and
  `knn_entries.yaml` fixture in the repo projects and opens identically
  before and after steps 2 and 5.
- **Generator output.** `veks catalog generate`, the pipeline generator,
  and the five `dataset.yaml` writers produce byte-identical files for
  the repo fixtures.
- **Catalogs screen.** A picker run over sources that include a
  registered catalog lists it with its count and marks it read-only, and
  a remove of a configured catalog leaves the registered one in place.
  This is tested through the screen's model, not a terminal.
- **Guards B-1 to B-8**, each proven to fail on the old code.

## 10. Effort and risk

This is the largest single change to the catalog and dataset layers
since the storage refactor. Its risk is concentrated in step 2: the two
inheritance implementations were written separately and may differ in
edge cases. The parity corpus is the control. Any difference it finds is
a decision about which behaviour is right, recorded here, not
something to be smoothed over. Steps 1, 4 and 8 are mechanical. Steps 3,
5, 6 and 7 are new code with byte-comparison tests against today's
output.

## 11. Decisions

Each item has a proposal and the user's decision. All six proposals were
accepted on 2026-10-12. Item 1 touched the cache layout, and the decision
keeps it unchanged.

1. **Cache directory name for canonical entries.** Opening a canonical
   entry names its cache directory after the `dataset.yaml` URL's last
   segment, not `entry.name` (§2.3). One authority is needed. Using
   `entry.name` would move existing caches for any catalog where the two
   differ, which is a filesystem-layout change. The proposal: keep
   today's names (the URL segment for canonical entries, `entry.name`
   for embedded ones) and record that rule as the single authority in
   `DatasetHandle`. Changing it is a separate, explicit decision.

   **Decision:** accepted 2026-10-12. Today's cache directory names are kept: the URL segment for canonical entries, `entry.name` for embedded ones. `DatasetHandle` records that rule as the single authority. The cache layout does not change.
2. **Built entries' cache name.** A built entry has no URL segment to
   fall back on, so its cache directory is `entry.name`. The proposal is
   to accept that, since built entries have no existing caches to move.

   **Decision:** accepted 2026-10-12. A built entry's cache directory is `entry.name`.
3. **The single catalog generator's rules.** The two generators must
   become one, so each point where they differ has to be settled. This
   item says what each difference does to the files a user gets, and
   whether that is a real compatibility concern.

   **Who runs which.** The pipeline generator is the one that matters
   for published data: the `gen-catalog` step of every dataset's
   finalize runs it. `veks prepare catalog generate` is the manual
   command. So the question is less "does `veks catalog generate`
   change" and more "does the manual command start writing what
   finalize already writes".

   **Measured.** On 2026-10-12 both were run over the same tree: copies
   of the `dataset.yaml` and `variables.yaml` of six finalized datasets
   from this machine (local test, preflight and production-scale ones,
   three with sized profiles), under one `.publish_url`. Five of the six
   per-dataset catalogs and the root catalog agreed byte for byte, apart
   from the one entry below.

   | Difference | What changes in the output | Real concern? |
   |---|---|---|
   | **Template-derived views.** The pipeline adds a view to each profile for every `per_profile` pipeline step output that the profile doesn't already declare. | One of the six datasets gained three facet keys in its `default` profile (`metadata_indices`, `filtered_neighbor_indices`, `filtered_neighbor_distances`); the other five were unchanged. The files those keys name are produced by the dataset's own pipeline. | **No for compatibility.** It only adds keys. The Rust catalog reader (`DSProfile`'s deserializer) passes unknown facet keys through as custom facets, and an existing key is never removed or re-pointed. The Java toolchain's reader was not checked. Opening a dataset uses its fetched `dataset.yaml`, not the catalog's summary, so data access does not change. What changes is what listings and selectors see before fetching: they now see the facets the dataset really has. |
   | **`load` vs `load_and_resolve`.** `load_and_resolve` expands sized profiles that wait on variables (`variables.yaml`). | None on the six finalized datasets, three of them sized: finalize writes the expanded profiles into `dataset.yaml`, so nothing is left waiting. It only shows for a dataset whose pipeline has not finalized, where today's manual command lists only `default` and explicit profiles and omits every sized one. | **No.** It only adds profiles, for datasets that were missing them, and published datasets are finalized. |
   | **`.do_not_catalog`.** The manual command still descends below a marked directory: datasets under it appear in the catalogs of the directories above, and only the marked directory gets no catalog file. The pipeline stops at the marker: datasets under it appear in no catalog at all. | Depends on which rule is chosen. | **Yes, this one is real.** Either choice changes output for any tree that uses the marker below a dataset root. The manual command's code comment says its rule is deliberate ("prevents catalog placement, not descent"). The pipeline's rule is the one that has been publishing. I can't tell from the code which one people rely on. |
   | **`knn_entries.yaml`.** Only the pipeline writes it, beside each `catalog.json`. | The manual command would start writing it too. | **Low.** It's a new file beside the existing ones; no existing file changes. It is the legacy shape that jvector reads. |
   | **`canonicalize` of the input directory.** The pipeline resolves symlinks in the directory it was given before looking for the `.publish_url` above it; the manual command doesn't. | None for entries: their paths are relative to the catalog directory either way. It changes which `.publish_url` is found when the input directory is reached through a symlink into another tree. | **Low,** and only for symlinked trees. Keeping the logical path (no `canonicalize`) is what the manual command's comment asks for. |

   **Proposal:** take the pipeline generator's content (template views,
   `load_and_resolve`, `knn_entries.yaml`), since it is what has been
   publishing. Don't canonicalize. On `.do_not_catalog`, take the
   pipeline's rule, since published catalogs were made with it; changing
   it would make datasets that are hidden today appear in published
   catalogs. Separately: the top-level `veks catalog generate` shows the
   pipeline command's options in `--help`, but parsing it demands the
   manual command's `<INPUT>`. One generator removes that mix-up too.

   **Decision:** accepted 2026-10-12. The one generator writes the pipeline generator's content (`load_and_resolve`, template-derived views, `knn_entries.yaml`), does not canonicalize its input, and keeps the pipeline's `.do_not_catalog` rule: no descent below a marked directory.

4. **What happens when a configured catalog can't be loaded.** A
   catalog listed in `catalogs.yaml` may fail to load: the URL doesn't
   answer, the file is gone, or it doesn't parse. No command stops
   because of it; they carry on with the other catalogs. They differ in
   whether they **tell the user**:
   - `datasets list`, `describe`, `precache` and the other `datasets`
     commands print the problem.
   - The explorer, shell completion and `veks ping` skip it silently, to
     a debug log only. A user whose catalog broke sees fewer datasets in
     the explorer and no reason why.

   **Proposal:** every command tells the user, except shell completion.
   Completion must not print, because anything it writes lands in the
   shell's completion list. The explorer shows the problem on its status
   line and marks the catalog on the catalogs screen. No command fails
   because one catalog is broken, as today. One addition: a catalog the
   user named explicitly for this run (`--catalogs`, `--at`) that fails
   to load is an error with a non-zero exit, since the user asked for
   exactly that catalog. A program registering its own catalogs chooses
   for each one, through `with` (reported) or `with_optional` (silent).

   **Decision:** accepted 2026-10-12. Every command reports a configured catalog that fails to load, except shell completion. The explorer shows it on its status line and catalogs screen. No command fails because of one broken catalog, except one the user named for the run (`--catalogs`, `--at`), which is an error with a non-zero exit. Registered catalogs choose through `with` / `with_optional`.

5. **Where `Location` lives.** The proposal is `vectordata::Location`,
   re-exported at the crate root, since datasets, catalogs and the cache
   all use it.

   **Decision:** accepted 2026-10-12. `vectordata::Location`, re-exported at the crate root.
6. **Caching of catalog contents.** The question is where caching of
   a catalog's *contents* (its list of entries) belongs, not whether it
   is allowed.

   **What happens today.** `Catalog::of` loads every source each time
   it is called, and a `Catalog` is a snapshot. Remote catalog files are
   already fetched through the shared transport, which handles
   conditional requests and offline copies. CLI commands build one
   `Catalog` per run. The explorer builds one at start and again after
   each config edit, plus one more in each of three action paths
   (§2.5).

   **Why not cache inside `Catalog::of`:**
   - **It can't know when a source changed.** A file can be checked by
     mtime, but a registered catalog built from a program's own data, or
     a database-backed source, has no mtime. The cache would serve stale
     entries with no way to invalidate them, or need an invalidation API
     on the trait, which every source must then implement.
   - **Snapshots are what callers rely on.** A `Catalog` value already is
     a cache: hold it and reuse it, and nothing reloads. The explorer's
     extra loads are a defect of the explorer (RC-19 passes it the one
     catalog), not a reason for a hidden cache underneath.
   - **The expensive part is already cached.** For located catalogs the
     cost is the fetch, and the transport already does conditional
     requests and keeps offline copies.

   **Where caching does belong:**
   - **Inside a source, when its own data is expensive to read.** The
     implementer knows when that data changes. A database-backed source
     can memoize its query and drop the memo when its own data changes.
     The trait doesn't stop that.
   - **In the caller,** by keeping the `Catalog` rather than rebuilding
     it.

   **Implications:**
   - Two `Catalog::of` calls over the same sources each ask every
     source again.
   - A registered catalog built from static content does no I/O, so
     asking it again is cheap.
   - A source that does slow work on every call will be slow every time
     a caller rebuilds the catalog, until it caches internally.

   **Proposal:** no cache in `Catalog::of`. Say in the `CatalogSource`
   docs that `entries` may be called once per `Catalog::of`, and that a
   source doing expensive work should memoize it internally. The trait
   stays public, so a program can implement it for any backing store.

   **Decision:** accepted 2026-10-12. No cache in `Catalog::of`. The `CatalogSource` docs say `entries` may be called once per `Catalog::of` and that an expensive source memoizes internally.

## 12. Implementation

### 12.1 Step 1: `Location` (2026-10-12)

**Built:**
- `vectordata::Location` (`src/location.rs`, re-exported at the crate
  root), with two variants: `Path` (`file://` URIs are read into it) and
  `Url`. Its methods are `parse`, `from_path`, `is_remote`, `as_dir`,
  `resolve`, `join`, `parent` and `last_segment`.
- `resolve` is the one joining rule: a URL or a leading `/` is kept as
  written, `file://` becomes its path, and anything else joins onto the
  location read as a directory, by RFC 3986 for URLs and `Path::join`
  for paths. `join` is `resolve` read back as a `Location`.
- `file://` URIs go through `Url::to_file_path`, so Windows drive URIs
  (`file:///C:/x`) are read. None of the three old strippers handled
  them.
- These use it:
  - the dataset view's facet resolution, whose two copies became one;
  - `TestDataGroup::load` dispatch;
  - `CatalogEntry::resolve_facet_url`;
  - catalog entry paths (`remap_entry`);
  - catalog location handling in `catalogs.yaml`.
- `DataSource` is removed. `GenericTestDataView::new` and
  `with_attributes` take a `Location`. This is a semver-breaking change.

**Decided during the step: cache identities are never re-spelled.** A
dataset's home string is recorded in its `origin.json`, and a facet's
cache path is its resolved string minus that home. URL parsing changes
spellings (host case, percent-encoding, `..`), so parsing either string
would move or orphan cached bytes. So:
- **`resolve` keeps absolute references as written.** It only joins
  relative ones, which the runtime already joined by URL rules.
- **Three string helpers sit beside the type** and change only what
  their name says:
  - `dir_form` adds a trailing `/`;
  - `parent_form` cuts the last segment;
  - `concat_form` concatenates.

  They replaced six trailing-slash copies, `parent_location_of`, and the
  `knn_entries` parser's `join_url`.
- **`knn_entries.yaml` catalogs keep resolving facets by
  concatenation** (`concat_form`). That is their jvector rule, and the
  strings it gives are the identities their datasets were cached under.
  Both the listing and the open read those strings, so they agree.

**Bugs fixed, each with a guard that fails on the previous code:**
- **B-1:** `TestDataGroup::load` with `s3://` or `file://` URLs. An
  unsupported scheme (`gs://`) is now named in the error, not reported as
  a missing file.
  - `group.rs`: `a_file_uri_opens_the_local_dataset` and
    `an_unsupported_scheme_is_named_in_the_error`.
  - `tests/offline.rs`: `s3_locations_open_from_their_kept_copies_offline`,
    run offline against kept copies.
  - An `s3://` dataset definition is fetched through its HTTPS endpoint.
    Auth and the kept copy stay keyed by the `s3://` URL.
- **B-2:** an `s3://` catalog directory finds its `knn_entries.yaml`. It
  is guarded by the same offline test.
- **B-4:** the catalog-time and open-time resolution of a facet agree.
  The guard is `view.rs` `the_catalog_and_the_open_resolve_a_facet_alike`,
  which covers `..`, a space, `/`, `file://` and foreign URLs.
  - On the old code, a `../shared/b.fvec` facet listed as
    `…/ds/../shared/b.fvec` but opened as `…/shared/b.fvec`.
  - The picker's cache drop then derived a `../` cache path. The drop
    only deletes files it finds under the dataset's own cache directory,
    so the effect was a drop that missed files, not deletion elsewhere.

**Behaviour that changed:**
- `resolve_facet_url` gives `/x` for a `file:///x` source, as the open
  does, instead of the URI.
- A canonical catalog entry's relative `path` joins onto the catalog's
  location by URL rules instead of concatenation. This is the form the
  open already used, so the listing's home now matches the `origin.json`
  the open records.
- An entry whose `path` is already absolute is kept as written. It was
  concatenated onto the catalog's location.

**Left out, on purpose:** the trailing-slash handling in `push/`,
`vecd` and `veks check publish_url`. These normalize publishing URLs,
which are not dataset or catalog locations.

**Unrelated, seen while checking:** `cargo clippy --workspace` now
reports warnings in files this step did not touch (`veks-completion`,
`slabtastic`, `tools`, and a few vectordata files). They come from the
newer clippy (rust 1.99) lints, not from this change.

### 12.2 Step 2: one dataset description (2026-10-12)

**Found first.** RC-3 assumed the two `DatasetConfig` types were two
copies of one model. On the 105 real `dataset.yaml` files on the
development machine, the two loaders disagreed on 46. Four causes:

1. The producer required `name`, so 34 datasets (jvector-style, which
   omit it) did not load for the pipeline, catalog generation or `veks`.
2. The reader skipped compactly declared sized profiles (`sized:`,
   `strata:`), so programs never saw them.
3. A profile whose base vectors live in its own directory, an oracle
   partition, got the default's metadata cut to its count from the
   reader. That metadata describes other rows. The producer gave it
   nothing.
4. A profile naming `default` explicitly at another size got the
   default's per-profile outputs from the producer: its full-base ground
   truth and predicate results. The reader withheld them, as PL-5
   requires.

**Traced before deciding.** For cause 4, the pipeline was not misled.
Template gating runs a template only where the profile's view for that
facet lies in its own directory, and saves did not write the copied
views. The copies did reach generated catalogs: tessera's `catalog.json`
listed `default`'s ground truth under 46 of its 223 profiles, every
`*-unfiltered` one, and that catalog is published.

**Decided by the user on 2026-10-12:**
- one effective view, for readers and the pipeline alike;
- a dataset with no `name` takes its directory's name;
- readers expand compact sized profiles where the file alone says
  enough, from its own `variables:` block.

Cause 3 was settled on evidence: the affected sift1m profiles are
oracle partitions (listed in the dataset's `partition_profiles`), so the
producer's rule (a partition inherits nothing) is the right one.

**Built:**
- **One inheritance rule**, `DSProfileGroup::resolve_inheritance` with
  `crossing_view`, following PL-5:
  - parents first, along `inherits` or else `default`;
  - a facet the child declares is kept;
  - on a size step (`base_count` differs) the per-profile outputs stay
    behind;
  - `base_vectors` and `metadata_content` are re-cut to the child's
    count;
  - everything else crosses;
  - `maxk` always crosses, and `base_count` crosses only at one size;
  - a partition inherits nothing.

  It replaces the producer's default-merge with its named-parent pass,
  and the reader's `apply_default_inheritance` chain.
- **Pending rungs:** a profile whose named parent is a sized rung still
  waiting on variables waits too, and expansion resolves it.
- **One save rule:** the renderer leaves a view out exactly when
  `DSProfileGroup::inherited_view`, the same rule, would restore it. The
  old rule compared with the parent's views and dropped a partition's
  facet that named the default's file, which the next load did not
  restore.
- **The reader's config is a projection.** `model::DatasetConfig` is
  computed from `dataset::DatasetConfig` (`from_description`,
  `from_parts`), and deserializing it runs the one parse. Catalog
  layouts project the same way; `from_catalog_entry` no longer
  round-trips YAML.
- **No attribute is lost.** `DatasetAttributes::other` keeps attributes
  the type has no field for, such as `dimension`, which readers look up.
  Saves dropped them before.
- **`name` is optional.** `DatasetConfig::parse`, `load` and `load_from`
  (`&Location`) fill an empty name from the directory. `load_from`
  replaced derive's own loader.
- **Errors name where they are**, for example
  `profile 'default': base_vectors: shard_stride: …`.

**Measured on the same 105 datasets afterwards:**
- Every file loads on both sides. The producer had refused 34.
- The two sides agree on every profile and facet. The one exception is
  119 sized profiles that only `variables.yaml` can expand, which a
  reader without that file cannot see.
- Readers see 40 more profiles (the compact sized ones). The 48
  partition profiles of the sift1m copies lost the wrong metadata. The
  46 tessera `*-unfiltered` profiles lost the default's ground truth on
  the producer side.
- Reader attributes are unchanged.
- Windows now read in the parser's canonical spelling (`[0..1000]` for
  `0..1000`). Delimiters are structural, so the bounds are the same.

**Guards that fail on the previous code:**
- `naming_default_at_another_size_takes_none_of_its_ground_truth`
- `a_dataset_without_a_name_takes_its_directory_name`
- `a_save_and_load_keep_every_facet_and_attribute`, which covers both
  the partition facet and the unknown attribute
- `a_reader_sees_compactly_declared_sized_profiles`
- `a_partition_gets_none_of_the_defaults_metadata`

**Left for the user:** tessera's published catalog still lists the
default's ground truth under its `*-unfiltered` profiles until it is
regenerated and republished.

### 12.3 Step 3: the dataset builder and the writers (2026-10-12)

**Found first.** §2.1 said the five hand-built writers write no
comments. Four do. The SRD also asked for byte-identical output, which
the renderer cannot give: it writes its own canonical layout. For
example, it writes structured `strata:` where bootstrap wrote a
one-line list, and quotes step options by YAML's rules. Every save of a
bootstrapped file already rewrote it that way and dropped its comments.

**Decided by the user on 2026-10-12:**
- The renderer writes the strata and tag-schema comments on every save.
- The builder takes a header for a new file.
- Acceptance is "parses to the same description, with its comments",
  not "byte-identical".

**Built:**
- **`DatasetConfig::builder(name)`** (`dataset/builder.rs`):
  - dataset level: `description`, `attribute` (known keys go to their
    fields through `DatasetAttributes::set_known`, the rest to `other`),
    `distance_function`, `model`, `license`, `format_version`,
    `profile_tag`, `stratum`, `variable`, `upstream` and `header_line`;
  - `profile(name, |p| …)`, whose profile builder offers `facet`
    (`StandardFacet`), `custom_facet`, `view`, `base_count`, `maxk`,
    `inherits`, `partition` and `attribute`. Facet aliases are stored
    under their canonical keys, as a parse stores them.
- **`build()` runs the checks a parse runs**, factored out so both
  paths share them: `DSProfileGroup::settle` (inheritance and shard
  declarations) and `check_description` (version gate, stated version
  against content, stated parents). Without `format_version` a built
  description states the lowest version its content needs.
- **Rendering:**
  - every save writes `# strata "<spec>": <description>` above
    `strata:`, using `strata::describe_spec`, moved from bootstrap;
  - every save writes the two-line tag-schema comment above
    `profile_tags:`;
  - a new file starts with the description's `header` lines, while an
    existing file keeps its own leading block.
- **The five writers build and save:**
  - `veks prepare bootstrap`, whose steps convert to `StepDef` with each
    option read as YAML reads its old text;
  - `gen_dataset` and `gen_derive`;
  - `generate example-dataset` and `datasets derive`, each keeping its
    header.

  `derive`'s own format-version fold is gone, because the builder
  folds. The `veks prepare clean` fallback literal is now an empty build.
- **Each writer was compared before its old code was deleted.** Its old
  and new output, rendered through `save`, came out equal on
  representative inputs: three bootstrap variants (base only; base,
  queries, strata, description and asserted counts; knn_utils
  personality with synthesized metadata), plain and sharded derive
  output, and all three example-dataset facet sets.
- **Two `veks` test helpers parsed bootstrap's text layout.**
  `check_and_import.rs`'s `step_ids_from_yaml` and `after_for_step`, and
  `dag_configurations.rs`'s `parse_steps`, matched `after: [..]` and
  fixed indents. They now read the steps from the parsed description.

**Guards:**
- `a_save_explains_its_strata_and_tag_schema` checks the comments and
  that a second save changes nothing. It fails on the previous code.
- `a_built_description_equals_the_parse_of_its_save` and
  `a_builder_refuses_what_a_parse_refuses` cover the builder.

### 12.4 Step 4: one `TestDataGroup` constructor (2026-10-12)

**Found first.** A catalog entry's dataset resolved relative facets
against `entry.path`, while the catalog resolved them against
`dataset_home_url`. For a `knn_entries` entry `entry.path` is the
catalog's directory, so the open looked for a listed file one directory
too high. The parser absolutises the facets it reads, which hid this;
entries built in code (step 5) would not have been.

**Built:**
- **`DatasetHandle { name, home, origin, config }`** (crate-private,
  `group.rs`) is what every route resolves to, and
  `TestDataGroup::from_handle` is the one constructor:
  - a directory, a URL or an explicitly named descriptor file is named
    after its last segment, and its origin is the directory as written;
  - a catalog entry is named `entry.name`, and its home and origin are
    `dataset_home_url`, the home it is listed under;
  - a home with no last segment (a server's root) has no name, and its
    facets cache by URL, as before.

  These are today's cache names (§11, item 1).
- **The eight struct literals are gone.** `of_content` (dispatch by
  shape), `of_dataset_yaml` and `of_knn_entries` read a descriptor, and
  `profile()` and `generic_view()` share one view constructor.
- **`TestDataGroup::from_config(name, home, description)`** opens a
  description a program built or parsed (RC-8).
- **One reader view.** `model::DatasetConfig::opened` expands sized
  profiles from the description's own variables, then projects. The
  model's deserializer, every load route and `from_config` all go
  through it.
- **The handle holds the reader view, not the description.** A catalog
  entry's layout is not a description until `EntryDescription` (step 5).
  Step 5 changes that route, not the handle.
- **The old `knn_entries` parser builds through the builder.** Its
  `to_config`, `to_config_for` and `to_configs` literals became
  `description_for(name)`. Step 5 removes the parser itself (RC-11).
- **`DatasetConfig::parse` returns `serde_yaml::Error`,** so a malformed
  `dataset.yaml` stays `Error::ConfigParse` through the group.

**Guards:**
- `a_catalog_entry_reads_its_facets_where_the_catalog_lists_them` fails
  on the previous code.
- `a_dataset_at_a_server_root_is_not_given_an_empty_name`: the first
  draft named such datasets `""`, putting each in the cache root, where
  the HTTP suites' servers refused one another's origins.
- `a_built_description_opens_at_its_home` and
  `a_knn_entries_directory_opens_the_entry_named_after_it`.

### 12.5 Step 5: entry descriptions, the entry builder, one `knn_entries` parser (2026-10-13)

**Found first.** Three points of RC-9 and RC-10 did not fit what step 1
established or what catalogs hold:
- **RC-9's `descriptor: Location` would re-spell identities.** The
  descriptor and the home are cache identities, and §12.1's rule keeps
  them as written. The entry holds them as strings.
  `EntryBuilder::home` and `descriptor` take a `Location` for a program.
  The parsers use crate-private `home_as_written` and
  `descriptor_as_written`.
- **`dataset_type` cannot be derived from the variant alone.** Catalogs
  list `hdf5` entries, and "direct" entries default to `unknown`;
  `infer_vtype` reads the type. A referenced entry keeps its type, and
  an embedded one is `knn_entries.yaml`.
- **The old parser could not simply be swapped in for directory
  opens.** Going through `from_catalog_entry` would give a directory's
  dataset the catalog's home (`<base>/<name>/`). That moves its cache
  identity, and a facet relative to the directory would then cache flat
  by basename. A directory open uses the catalog parser for the
  description, and keeps the directory's name and origin.

**Built:**
- **`EntryDescription`:** `Embedded(Box<DatasetConfig>)`, or
  `Referenced { descriptor, dataset_type, summary }`.
- **`CatalogEntry` is private fields behind accessors.**
  - The fields are `name`, `home`, `description` and the provenance
    pair.
  - The accessors are `name`, `description`, `dataset_type`, `location`,
    `format_version`, `attributes`, `profile_tags`, `profiles`,
    `layout`, `catalog_file`, `catalog_name` and `dataset_home_url`.
  - Provenance is set only by the crate's loaders, through
    `set_catalog_file` and `set_catalog_name` (RC-14 moves this into the
    catalog in step 6).
- **`CatalogEntry::builder(name)`** (RC-10) takes `home` with `dataset`,
  or `descriptor` with a summary. It refuses an entry with no
  description, and an embedded one with no home.
- **One listing constructor.** `CatalogEntry::from_listing` is the one
  rule for a listed entry: a `knn_entries.yaml` entry is embedded and
  homed at its base and name, and any other is referenced and homed at
  its descriptor's directory. The resolver's `remap_entry` (both shapes)
  and the entry's `Deserialize` use it.
- **The file form is unchanged.** `Serialize` and `Deserialize` go
  through the wire form `{name, path, dataset_type, layout}`, in the old
  field order.
- **The builders are the only construction paths.**
  - The `knn_entries` catalog parser builds each dataset through
    `DatasetConfig::builder` and each entry through
    `CatalogEntry::builder`.
  - Both catalog generators build entries through the builder.
  - An embedded layout read from a file becomes a description through
    `DatasetConfig::from_layout`, which runs `build`'s checks.
- **`Catalog::open_entry` is `TestDataGroup::from_catalog_entry`,**
  matching on the description. An embedded entry opens with no fetch.
  A referenced one loads its descriptor, with today's cache names.
- **`crate::knn_entries` is removed** (RC-11):
  - a directory open of `knn_entries.yaml` uses the catalog parser;
  - the publish check reads raw declared paths through
    `catalog::knn_entries::declared_facet_paths`.
- **Test fixtures that built entries as struct literals now use the
  builders.** Two `filter.rs` tests had been rewritten to mutate a copy
  of the layout, which would have tested nothing. They now change the
  layout before building the entry.

**Behaviour changes:**
- A directory's `knn_entries.yaml` honours `_defaults.base_url`,
  `cache_dir` and `${VAR}` (B-3).
- Its facets are now resolved strings: the catalog's concatenation rule
  rather than the view's URL join. For a URL directory the two differ
  only for names with `..`, spaces or characters that need encoding;
  such files cache under the concatenated spelling.
- knn descriptions go through the one inheritance rule, as readers
  already did through `effective_view`. A profile lacking a facet the
  `default` profile has now carries it explicitly.
- A listing shows an embedded entry's location as its home (base and
  name) rather than its base.

**Guards:**
- `a_knn_entries_directory_honours_its_defaults_base_url` (B-3) fails on
  the previous code.
- `an_entry_writes_the_form_it_was_read_from` covers the catalog file
  text, for each kind of entry.
- Also: `an_entry_is_embedded_or_referenced_by_its_listing`,
  `an_embedded_entry_without_a_home_is_refused` and
  `a_built_entry_opens_at_its_home`.

### 12.6 Step 6: catalog sources, the catalog builder, provenance (2026-10-13)

**Found first.**
- **Severity cannot live in the source if the set decides it.** RC-13's
  `entries` returned a plain list, so "this location has no catalog"
  could not be told apart from "an empty catalog". Today's required and
  optional loads differ only there. `entries` returns `Option`: `None`
  is nothing there, and the set's `Requirement` decides whether that is
  an error.
- **`CatalogOrigin::Located` keeps the location as written,** as
  entries keep theirs (§12.5). A catalog location names its kept copy.
- **RC-17's "`catalog_file` becomes the origin" would lose the file.**
  A directory's origin is the directory, but the document read is
  `catalog.json`, `catalog.yaml` or `knn_entries.yaml`. A located
  catalog records the document on each entry it reads. `Catalog::of`
  records the catalog's name, and keeps each origin for provenance.

**Built** (`catalog/source.rs`):
- **`CatalogSource`** (`name`, `origin`, `entries`) and
  **`CatalogOrigin`**: `Located { location, configured_in }` or
  `Registered { label }`.
- **`LocatedCatalog`** replaces `load_from_explicit_catalog_file`,
  `try_load_canonical_catalog` and `try_load_knn_entries`:
  - a location naming a file is that file;
  - otherwise `catalog.json`, `catalog.yaml` and `knn_entries.yaml` are
    probed under it;
  - the first found is read by shape.
- **`CatalogBuilder::named`** has `base`, `entry`, `entries`, `content`
  and `label`. `build()` gives a `BuiltCatalog` whose content is already
  parsed. Content with a relative path and no base is a diagnostic, and
  its entry is skipped. A `knn_entries` document needs a base or a
  `_defaults.base_url`.
- **`parse_catalog_content`** is the one content parser. Located and
  built catalogs both use it, so the same content gives the same
  entries.
- **`CatalogSources`** holds `(Arc<dyn CatalogSource>, Requirement)` in
  order:
  - `with` and `with_optional` add any source, and `iter` lists them;
  - `configure*` and `add_*catalogs` add `LocatedCatalog`s, each bound
    to the `catalogs.yaml` that names it;
  - `required()`, `optional()` and the unused `catalog_file_for` are
    gone.
- **`Catalog::of`** asks each source in order and records its name on
  every entry; ambiguity provenance reads the kept origins. A required
  source that gives nothing is an error, and an optional one is a debug
  log.
- **`CatalogDiagnostic::error` and `warning` are public,** for a
  program's own sources.
- **The explorer's source view chooses by description:** a referenced
  entry shows its `dataset.yaml`, and an embedded one its catalog
  document.

**Behaviour changes:**
- **Catalogs list in the order they were added.** Required catalogs no
  longer list ahead of optional ones (RC-16).
- **A remote catalog directory is probed for `catalog.yaml` too**, as a
  local one was. A knn-only remote catalog costs one more request.
- **Canonical entries record their catalog document (B-8),** so
  ambiguity provenance names the file:
  `ds (catalog 'lab': /data/lab/catalog.json)`.

**Guards:**
- `a_located_catalogs_entries_record_their_document` (B-8) fails on the
  previous code.
- `a_built_catalog_equals_the_located_one_with_its_content` covers both
  shapes.
- Also:
  - `built_content_with_relative_paths_and_no_base_is_a_diagnostic`;
  - `a_registered_catalog_collides_like_a_located_one`;
  - `only_a_required_catalog_that_gives_nothing_is_an_error`;
  - `the_source_view_shows_a_referenced_entrys_dataset_yaml`.
- B-2 (`s3://` knn_entries) was fixed in step 1, and
  `s3_locations_open_from_their_kept_copies_offline` guards it.
