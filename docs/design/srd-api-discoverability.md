# SRD — One discoverable path per task

**Status:** implemented 2026-10-06 (§11 records what was built and the
decisions taken). The project-independent method is
`blueprint-discoverable-apis.md`.
**Scope:** the public API surface of the published crates, starting with
`vectordata`: the fetch and progress API, the library/CLI boundary, crate
and module rustdoc, `examples/`, a packaged agent guide, and the CI checks
that keep all of it true.

**Origin.** The blueprint comes from the `fastervector` project, which
consumes `vectordata` 2.4.0. It rests on a miss that project made, which
is recorded in §1. The account is that project's. The facts about this
codebase were checked here when this document was written; the one
behaviour that could not be checked is marked as reported.

## 1. Problem

A downstream consumer wrote about 150 lines on top of `vectordata` to do
something `vectordata` already does. It needed to download a chosen
subset of a profile's facets with a live progress readout. It wrote its
own pieces:

- a download plan per facet;
- background prefetch handles;
- a polling loop that printed rate and time remaining;
- a clamp for byte counts that overshot the plan.

It then concluded that the library could not prebuffer a facet subset.

Two library paths already did exactly that.
`TestDataView::prefetch_with_progress` fetches a window of a facet and
reports through a callback. `vectordata::datasets::precache::run` takes a
`PrecacheRequest` with a `facets` filter and draws the same plan table
and live per-facet meter as `vectordata datasets precache --facet`. Both
are ordinary library API: `pub mod datasets` is not gated on the `cli`
feature (only `shell` is).

The consumer replaced its code with one `precache::run` call. Its labels
then matched the CLI's byte for byte, and the meter worked.

**DX-1.** The failure was one of discovery, not of capability. The API
let a careful reader conclude the feature was missing. Each of the four
ways that happened is a property of this codebase that a design can
remove:

- **No survey past the first working primitive.**
  `prefetch_in_background` worked, so the rest of the view trait went
  unread. Nothing on that function pointed to the better path.
- **Location and shape said "CLI".**
  - The `datasets` module is documented as the "canonical implementation
    of `<binary> datasets …` subcommands", and it sits next to the
    `cli`-gated `shell`.
  - `precache::run` returns an `i32` exit code and prints to stderr.
  - `PrecacheRequest` is built from CLI-shaped string fields: `configdir`,
    `at`, `extra_catalogs`.
  - Every signal said "command-line only". The `cfg` that says otherwise
    is never in view.
- **A near miss on the obvious type.** `prebuffer_all_with_progress`
  fetches every facet with no subset filter and no renderer. It reads as
  the feature with a piece missing, which invites the conclusion that
  the piece does not exist.
- **The visible behaviour was not traced.** The CLI visibly draws the
  meter. Following that command into the library finds `run` in one step,
  but nothing suggests doing it.

**DX-2.** The requirement is that a person browsing docs.rs, or an agent
reading the registry copy of the crate, finds the one recommended call
for each common task without reading `src/` beyond `lib.rs`, module
heads and a packaged guide.

## 2. What exists today

This is the state the design starts from, checked against the code.

**DX-3. Fetch entry points.** About nine public ways to bring bytes into
the cache, none marked canonical:

| Where | Entry points |
|---|---|
| `TestDataView` | `prebuffer_all`, `prebuffer_all_with_progress`, `prefetch`, `prefetch_with_progress`, `prefetch_in_background` |
| readers | `VectorReader::precache` |
| `FacetStorage` | `prebuffer_with_progress`, `prebuffer_range_with_progress`, reached through `open_facet_storage`, which is `#[doc(hidden)]` |
| `TestDataGroup` | `prebuffer_profiles_with_progress`, `prebuffer_all_profiles_with_progress` |
| `datasets::precache` | `run(PrecacheRequest)` |

**DX-4. Inconsistencies around them.**

- **Cache-capacity checks.** `ensure_cache_capacity` is `pub(crate)`.
  Only `prebuffer_all_with_progress` and the CLI precache path check
  capacity before fetching. A caller of `prefetch`,
  `prefetch_with_progress` or `prefetch_in_background` cannot check it
  at all.
- **The precache meter's total under a facet filter.** On the filtered
  path the meter is built per facet (`LiveCtx::new(1,
  plan.bytes_to_fetch())`), so its "total" covers the current facet only,
  not the run.
- **Library calls that print.** The catalog resolver writes to stderr
  from library calls with `eprintln!`, including "Did you mean…" and
  "Dataset not found" diagnostics.
- **Spec parsing.** `DatasetSpec::split_head` already handles URLs, ports
  and Windows drive letters, but nothing in its name tells a reader to
  use it instead of `split_once(':')`. `Catalog::open_profile` accepts a
  selector expression only implicitly.

**DX-5. A reported trap.** The consumer saw a 478 MB file download
silently, with no progress, when it opened facet readers
(`view.base_vectors()` and the like) on a remote file without a `.mref`
before running precache. Moving precache first fixed it. This is
reported, not reproduced here; DX-14 requires confirming it.

*Confirmed while implementing (§11.3):* a uniform facet opens lazily —
one chunk — but a remote variable-length facet with no published
offset index was walked, and so downloaded whole, before its reader
returned.

## 3. Principles

**DX-6. One canonical call per task.** Every common task has exactly one
recommended call:

- open a dataset;
- fetch facets with progress;
- read vectors;
- stream chunks;
- parse a spec;
- list datasets.

Every other public call that overlaps it says so and names it.

**DX-7. The canonical call lives on the type the caller already holds.**
A caller with a view or a catalog finds "fetch these facets with
progress" there, not in a module documented as CLI support.

**DX-8. Library first, CLI on top.** Each CLI command is a thin adapter
over a library function. That function returns `Result<Report, Error>`
and reports progress through an injected sink. The adapter chooses the
rendering and the exit code. The library never chooses an exit code and
never prints except through the sink it was given.

**DX-9. Shapes tell the truth.** A library function does not look like a
CLI:

- no `i32` exit codes;
- no bag of command-line strings as its primary input;
- no "subcommand" wording as its description.

**DX-10. Traps are designed out, or documented where they bite.** A
silent download of hundreds of megabytes is either made impossible (it
is refused, or goes through a progress sink) or stated in the first line
of the documentation of the call that triggers it.

**DX-11. Discoverability is checked.** It is tested like any other
property:

- examples compile;
- doc links resolve;
- public items are documented;
- the task index names items that exist.

**DX-12. Two readers.** The docs serve a person on docs.rs and an agent
reading `~/.cargo/registry/src/…`. An agent searches names and reads
`lib.rs` and module heads first, so the task index lives there, and the
agent guide ships inside the package.

## 4. Design

### 4.1 API shape (highest value)

**DX-13. A library-native fetch.**
- **The call.** `vectordata` gains one fetch call on the view, along the
  lines of `view.fetch(&facets, &progress) -> Result<FetchReport>`.
- **What it does.** It takes a facet subset (empty means all), optional
  windows and a progress sink, and performs the cache-capacity check
  itself.
- **The sink.** The progress sink is a type with three options: render to
  stderr as the CLI does, stay silent, or call a callback.
- **Catalog entry.** The catalog gets the matching entry point that takes
  a spec, `catalog.fetch(spec, &facets, &progress)`.
- **The CLI on top.** `datasets::precache::run` becomes the CLI adapter
  over it, keeping its exit code and stderr rendering.
- **The meter.** Its total spans every facet in the run, not only the
  current one.

**DX-14. Reader open on an uncovered remote file.**
- **First, confirm the trap.** Confirm DX-5 with a test that opens a
  reader on a served file that has no `.mref`, and measure what it
  fetches.
- **If it reproduces, remove it.** Opening must not quietly fetch the
  whole file. It either refuses with an error naming `fetch`, or fetches
  through a progress sink.
- **If it is intended, document it.** If whole-file fetch on open is
  intended behaviour, it is stated in the first doc line of every reader
  constructor that can trigger it.

**DX-15. Spec handling has an obvious name.** The catalog accepts a
dataset spec directly, through `catalog.open_spec("name:selector")` or
an equivalent. The documentation of `DatasetSpec` presents it as the
parser to use. `split_head` keeps its job but is no longer the only
place the grammar is visible.

**DX-16. The library stops printing.** Library-side diagnostics, such as
the resolver's suggestions, become values the caller can render: fields
on the error or on a report. They are no longer `eprintln!` calls. The
CLI renders them as it does today.

**DX-17. The other fetch paths say what they are.** Every entry point in
DX-3 other than the canonical one carries a first doc line naming the
canonical call (§6). A true duplicate is marked `#[deprecated(note =
"use …")]`. `#[doc(hidden)]` stays only on items that must be public for
crate-internal reasons.

### 4.2 Documentation artifacts

**DX-18. Crate-level rustdoc order.** Each published crate's `lib.rs`
opens with these sections, in order, scaled to the crate:

1. **What it is.** One paragraph: what the crate is and who calls it.
2. **Common tasks.** A table with the columns task, call and example. Each
   row names its canonical item with an intra-doc link and shows a short
   snippet or links an `examples/` file. For `vectordata` it covers at
   least:
   - opening a dataset by spec;
   - fetching chosen facets with progress;
   - fetching everything;
   - random-access reads;
   - streaming base vectors in chunks;
   - parsing a spec or selector;
   - listing catalogs and datasets;
   - cache location and capacity;
   - typed facet access.
3. **Layers.** The layer stack is catalog → group → view → reader →
   storage. This section says which layers applications use and which are
   internal or advanced.
4. **Traps.** Each surprising behaviour, linked to the item that has it.
5. **Feature flags.** What each one gates and, as explicitly, what it does
   not. For example, `datasets` is available without `cli`.

**DX-19. Module heads.** The first line of every public module's
documentation says whether the module is library API, CLI support, or
internal. A module that is both, as `datasets` is today, names which of
its items are the library API.

**DX-20. `examples/`.** Every common-task row longer than a snippet has an
example file: fetch with progress, stream base vectors, open by spec.
Examples are built in CI.

**DX-21. Crate README.** The README stays short and points to the Common
tasks section on docs.rs instead of repeating it, so the two cannot
drift.

**DX-22. A packaged agent guide.** Each crate with a non-trivial API
ships an `AGENTS.md` in its published package. The registry copy is what
agents read, so the package file list has to include it. The guide
contains:

- "to do X, call Y" lines that mirror the Common tasks table, with full
  paths;
- a "do not hand-roll" list: download planning, progress meters, spec
  parsing, cache-capacity checks, chunked streaming;
- the traps;
- a map from each CLI command to the library function that implements
  it.

The CLI-to-library map alone would have prevented the miss in §1.

**DX-23. Workspace docs.**
- **`docs/guides/`.** Task guides mirroring the Common tasks rows across
  crates.
- **`docs/sysref/`.** The layer model and the CLI-command-to-library map.
- **SRDs.** Each task's canonical-path decision is recorded in an SRD,
  so a later addition is weighed against it rather than becoming a tenth
  way to fetch.

**DX-24. Consumer guidance.** A short paragraph for downstream projects'
own agent instructions says that before writing download, progress,
spec-parsing or streaming code, read `vectordata`'s `AGENTS.md` and its
Common tasks section.

## 5. Enforcement

**DX-25. Strict docs.** `cargo doc --no-deps` runs in CI with
`-D warnings`, which already denies broken and private intra-doc links.
Every published crate carries `#![warn(missing_docs)]`, denied in CI.

**DX-26. Examples and snippets run.**
- **Where they run.** `cargo test --doc` and `cargo build --examples` both
  run in CI.
- **Network snippets.** Each Common tasks snippet that needs the network
  is a `no_run` doctest, so it still compiles. None is `ignore`.

**DX-27. The task index is real.** A test or doctest `use`s every item
the Common tasks table names. Renaming or removing a canonical call
without updating the table then fails the build.

**DX-28. CLI-to-library coverage.**
- **The check.** A test pairs each CLI subcommand with the library
  function it calls.
- **The failure.** Adding a subcommand without its library entry fails
  the test.

**DX-29. Packaging.** The existing packaging tests (readme paths, no
relative README links) extend to two more checks:
- `AGENTS.md` is in each package's file list;
- the paths it names resolve.

**DX-30. An overlap budget.**
- **The list.** Each task has a short list of its allowed public entry
  points, kept in this SRD or in a test.
- **The effect.** A new fetch-like public function must be added to the
  list, which forces the question of whether it is a tenth way to do the
  same thing.

**DX-31. Optional: an agent discoverability check.**
- **The setup.** Before a release, a fresh agent session is given only
  the packaged crate and tasks such as "fetch the base and query vectors
  of a profile with a progress meter".
- **The pass condition.** It passes if the agent finds the canonical call
  without reading `src/` beyond `lib.rs`, module heads and `AGENTS.md`.
- **The prompts.** They live in the repository.

## 6. Templates

A Common tasks row:

```text
| Fetch chosen facets with a progress meter | [`View::fetch`] with a facet list | `examples/fetch_with_progress.rs` |
```

The first doc line of a non-canonical primitive:

```text
/// Low-level: <what it does>. Most callers want [`<canonical>`], which <what it adds>.
```

A module head:

```text
//! <Library API | CLI support | Internal>. <one sentence>. Applications use [`<item>`]; the rest supports `<binary> <cmd>`.
```

`AGENTS.md`:

```text
# <crate> for agents
## Do this
- <task>: `<full::path>` — <note>
## Do not hand-roll
- <capability> — use `<path>`
## CLI ↔ library
- `<binary> <cmd>` → `<full::path>`
## Traps
- <behaviour> — <how to avoid>
```

## 7. Order of work

**DX-32.** The work is ordered by value for the effort.

1. **The API shape (DX-13 to DX-17).** Without it, the documentation would
   only be describing the problem.
2. **The Common tasks table and its examples (DX-18, DX-20), with their
   checks (DX-26, DX-27).** This fixes discovery even before the reshaping
   lands.
3. **The agent guide and its packaging check (DX-22, DX-24, DX-29).** This
   is the cheapest step, but it works only when read, so it backs up the
   first two rather than replacing them.

## 8. What must not change

**DX-33.** The CLI keeps its output and its exit codes. `vectordata
datasets precache` renders exactly what it renders today, through the
library fetch and the stderr sink.

**DX-34.** The existing fetch entry points keep working. They are marked
and pointed at the canonical call, deprecated where they are true
duplicates, and never removed in the release that introduces the
replacement.

**DX-35.** Cache layout, merkle coverage and transport behaviour do not
change. The work concerns which call a caller makes and what it is told,
not how bytes move.

## 9. Acceptance tests

| # | Case | Expect |
|---|---|---|
| 1 | `view.fetch` with a two-facet subset against a served dataset | only those facets cached; report lists both; sink saw progress for both |
| 2 | the same through `catalog.fetch(spec, …)` | identical cache contents and report |
| 3 | `vectordata datasets precache --facet …` | output byte-identical to before; implemented by `fetch` |
| 4 | precache meter over two facets | the total is the sum across both facets |
| 5 | fetch with insufficient cache space | refused with the capacity error, nothing fetched |
| 6 | reader open on a remote file without `.mref` | per DX-14: refused or reported through a sink, never silent |
| 7 | every Common tasks item | resolves (DX-27) |
| 8 | every CLI subcommand | maps to a library function (DX-28) |
| 9 | `cargo package --list` per crate | includes `AGENTS.md`; its paths resolve (DX-29) |
| 10 | library catalog resolution of an unknown name | returns the suggestions as data; prints nothing |

## 10. Open

The questions this section listed are settled in §11.1. What remains:

- **`datasets drop-cache` (veks) scans the cache itself** instead of
  calling `cache_admin::prune_by_filter`, with its own glob matcher —
  the hand-rolled-duplicate pattern this SRD exists to remove, in a
  command. Folding it onto the library call needs its interactive
  confirmation and its legacy-layout discovery reconciled with
  `prune_by_filter`'s.
- **The capacity refusal is unit-tested, not exercised end to end.**
  `cache::capacity_verdict` is pure and tested; `FetchPlan::check` calls
  it with the plan's total. Exercising a real refusal needs a
  filesystem that reports a shortfall on demand.
- **`AGENTS.md` exists for `vectordata` only.** The packaging test
  requires any other crate that gains one to ship it; `slabtastic`,
  `veks-simd` and `vecd` are the candidates.
- **The agent-discoverability check (DX-31) is manual.** Its prompts are
  in `docs/agent-checks/vectordata.md`.
- **Pre-existing clippy 1.99 lints** across the workspace are untouched;
  CI does not run clippy.

## 11. Implementation

### 11.1 Settled

- **The fetch call** is `TestDataView::fetch(&FetchRequest, &mut dyn
  FetchProgress) -> Result<FetchReport>`, a provided trait method, so
  every view has it and `Arc<dyn TestDataView>` keeps working. It is
  `plan_fetch` then `FetchPlan::execute`; both are public. The group
  form is `TestDataGroup::fetch(&profiles, …)`, the spec form
  `Catalog::fetch(spec, …)`, and `DatasetSelection::fetch` sits between
  them. Module: `vectordata::fetch`. The name is `fetch`; its docs name
  precache, prebuffer, prefetch and download so a search for any lands
  on it.
- **The sink** is a `vectordata` trait, `FetchProgress`, implemented by
  `Silent`, `TextMeter` (the CLI's renderer, writing to any `Write`),
  and every `FnMut(&FetchEvent<'_>)`. Events are cumulative per facet
  and clamped to the plan. `veks-core`'s `UiHandle` stays where it is:
  it is the pipeline's display, not a fetch sink. `push::ProgressSink`
  is kept; push's two displays that bypassed it now go through it.
- **Deprecation** happened in the release that adds `fetch` (2.5.0):
  `prebuffer_all`, `prebuffer_all_with_progress`, `prefetch`,
  `prefetch_with_progress` and the three group `prebuffer_*` forms are
  deprecated wrappers. `prefetch_in_background` stays as the background
  form; `prefetch_plan`, `precache`, `prebuffer_*` on `FacetStorage` and
  readers stay as low-level, each with a first doc line naming `fetch`.
- **`AGENTS.md`**, not `llms.txt`, rendered by rustdoc as
  `vectordata::_agents` so its links are checked.
- **The library selector default**: `open_spec` with no selector opens
  `default`, as every other library surface does. The CLI's refusal of
  a bare `precache` spec stays a CLI policy.

### 11.2 Built

- `vectordata::fetch` (DX-13): request, plan, report, events, sinks,
  `TextMeter`. The whole run is planned and checked before anything
  moves: unknown facets (`Error::UnknownFacets`), a window that would
  become a whole-facet download (`Error::WindowUnresolvable`, carrying
  the size), and the cache-capacity check (`Error::InsufficientCacheSpace`;
  the check is no longer crate-private in effect, since every fetch
  runs it, including `prefetch_in_background`). Standard facet aliases
  are accepted. A plan that degrades to the whole facet now subtracts
  what is already cached (`PrefetchPlan::resident_bytes`).
- `datasets precache` is an adapter over it, keeping its headings, plan
  table and exit codes (DX-33). The meter's total now spans the run
  under `--facet`; resident facets show `✓ already resident`.
- Catalog (DX-15, DX-16): `Catalog::open_spec`, `open_selection`,
  `fetch`, `lookup` (typed `UnknownDataset { suggestions }` /
  `AmbiguousDataset`), `suggestions`, `diagnostics`; `DatasetSelection`.
  `CatalogSources` collects diagnostics too; `resolve_catalog_value`
  returns a `Result` instead of exiting. The CLI renders all of it
  through `datasets::open_catalog` and `datasets::report_lookup_failure`.
  `Catalog::list_datasets` is gone; its rendering lives in `datasets`.
- Other library output removed: `credentials::expiry_warning` returns
  the warning; `cache_admin::PruneReport::failed` carries removal
  failures; push's plan, confirmation, hashing progress and upload
  display all go through its `ProgressSink`, and its CLI moved to
  `push/cli.rs`. One exception remains, allow-listed with its reason:
  the one-time notice when `settings` writes the user's settings file.
- Reader traps (DX-14): see §11.3.
- Docs (DX-17 to DX-23): crate docs rewritten (Common tasks, Layers,
  Traps, Feature flags); every module's first line names its layer;
  low-level and deprecated first lines; `examples/open_by_spec.rs`,
  `fetch_with_progress.rs`, `stream_base_vectors.rs`; short README;
  `AGENTS.md`; `docs/sysref/02-api.md` §2.1a (layers and CLI map) and
  §2.12 rewritten; `open_facet_storage` no longer hidden. The existing
  tutorial (`crates/vectordata/docs/access-datasets-from-rust.md`)
  serves as the task guide.
- Every published crate carries `#![warn(missing_docs)]`; about 1,400
  public items were documented to get there.
- Checks (DX-25 to DX-30): the CI doc build (`-D warnings`) now also
  denies missing docs and builds `examples/`;
  `tests/discoverability.rs` (module labels, library code never prints
  or exits, Common-tasks integrity, fetch overlap budget with
  deprecation state); `shell.rs` test diffing the real command tree
  against the `AGENTS.md` map both ways; root packaging test requiring
  `AGENTS.md` and `examples/` in the package; `tests/fetch.rs` for the
  acceptance cases.

### 11.3 DX-14 resolution

- **Uniform facets** open lazily, with or without `.mref`: pinned by a
  test that opens and reads one record of a three-chunk file and sees
  one chunk fetched.
- **Variable-length facets with no published index** are refused on
  open with `IoError::OffsetIndexUnavailable { url, bytes }`, naming
  `fetch` and the `IDXFOR__` sidecar, unless the file is fully cached
  (then the index is rebuilt from the local copy and persisted). The
  planner already refused; readers now follow the same rule, and the
  `OffsetSource` distinction between them is gone.
- **Servers without HTTP range support** can only serve whole files;
  that remains intended and is stated on `XvecReader::open` and in the
  crate's Traps.

### 11.4 Found along the way, and fixed

- A facet declared `metadata_results` appeared in `facet_manifest` as
  `predicate_results`; the manifest now uses the canonical name.
- `TypedReader::get_native` on a sharded facet read the first shard's
  bytes at the facet-wide offset — the wrong record whenever that offset
  still fell inside the first file — and turned read errors into `0`.
  It now reads through the shard that holds the ordinal and returns a
  `Result`.
- The view trait's E/F labels for pre- and post-filter ground truth were
  swapped against `StandardFacet::code`, and so was a module header.
- Doc blocks detached from their items (`AccessMode::classify`,
  `RecordFacet::record_bytes`, `expand_per_profile_steps`,
  `StreamContext`, `ProvenanceNode`, `FacetStorage`, `parse_facet_spec`,
  a stray line in `veks prepare readme`'s help) were reattached.
- `ChecksumFile` claimed to be sorted but was not when generated; it is
  now.
- A `NumberKind::Integer::signed` comment said the opposite of what the
  field holds.
