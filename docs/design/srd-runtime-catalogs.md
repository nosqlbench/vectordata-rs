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
    None of them writes YAML comments, so nothing is lost by building
    through the struct instead.
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
