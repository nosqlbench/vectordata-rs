# Impact analysis: the runtime-catalogs arc (2.5.1 → next release)

This inventories everything that changed since release 2.5.1
(`cccc5e6`) that could affect a library user, a CLI user, a published
dataset or a cached one. It covers the `catalogs.yaml` fix (`11fcda1`)
and the nine steps of `srd-runtime-catalogs.md` (`99c59d3` to
`d04bb13`).

Each finding says how it was established: **measured** (run against
the 2.5.1 code and HEAD on real data), **verified** (read from code or
a downstream repo), or **inferred** (reasoned, not run). Section 9 lists
the proposed mitigations. **None of them is implemented yet**; each
needs a decision.

**API policy for this release.** Downstream integrations are expected
to update their code to the new API. The library carries no
compatibility shims, deprecated aliases or public fields kept only for
old callers. Mitigation is aimed at what code changes cannot fix:
data read from the wrong place, published catalogs, and version pins
that would pick up the break silently.

---

## 1. Summary

| # | Impact | Who is affected | Severity | Proposed mitigation |
|---|---|---|---|---|
| 1 | A directory holding only `knn_entries.yaml` with a remote `_defaults.base_url` now reads its facets from that URL, where 2.5.1 read the files beside it | 4 local directories (measured), and anyone opening such a directory by path | **High**: network fetches, or failure offline, for data that is on disk | M1: prefer a file present beside `knn_entries.yaml` |
| 2 | `CatalogEntry` fields became private accessors | fastervector `fvec-bench` (`catalog.rs:327`, `:352`, reading `e.name`), and any reader of `.path` or `.layout` | Medium: a compile break with a one-line fix per site | Downstream updates (§5 migration column); M8 hands fastervector the exact fix |
| 3 | Other public API changes (§5) | Downstream Rust code; only fastervector is known to use these crates at 2.x | Low to medium: compile breaks, each with a stated replacement | Downstream updates; M8 migration guide |
| 4 | A semver-breaking release as `2.6.0` would reach `^2.2` and `^2.5` pins | fastervector (`2.5.1`), fastervector_s2 (`2.2`) | **High** if published as a 2.x minor: the break would arrive on `cargo update`, not when the integrator chooses | M5: release as 3.0.0, so each integration updates when it bumps the pin |
| 5 | Tessera's published derived files (`catalog.json`, `catalog.yaml`, `dataset.json`, `docs/dataset.md`, `docs/exemplars.md` and their HTML) list the default profile's *filtered* outputs under all 46 `*-unfiltered` profiles. `dataset.yaml` is correct | Anyone listing or selecting tessera profiles from the catalog, or reading its docs | Medium: the listing is wrong; opens are not | M6: regenerate the derived files with the fixed generators; `dataset.yaml` and the profile data stay untouched (§3.8) |
| 5a | A HEAD `veks run` on tessera would rewrite its `dataset.yaml`: 851 changed lines, the same description in the renderer's layout | Tessera, and any dataset whose `dataset.yaml` is not in the renderer's exact layout | **High** for tessera, whose `dataset.yaml` must not change | M9: the closing save leaves a file alone when it parses to the same description |
| 5b | `veks run dataset.yaml` gives docs steps an empty workspace path, so `dataset.md` omits its "Source Data" table and "Total size" row; run another way, it includes them | Every dataset's generated docs | Low: docs differ by invocation | M10: one workspace path; decide what "Source Data" may publish |
| 6 | ~~2.5.1 readers give tessera's `*-uniform-*` profiles the whole `metadata_content` slab~~ *Retracted* (§3.3): a probe artifact; 2.5.1 readers apply the window | — | none | none |
| 7 | Regenerated catalogs reorder facets within profiles | Every dataset whose profiles inherit facets | Low: same JSON content, different bytes | M2: keep the 2.5.1 facet order |
| 5c | `knn_entries.yaml` lists profiles it cannot state: a sized profile against the whole base file (its window dropped), a sharded base as a literal `NNNN` pattern | jvector users of any dataset with sized or sharded profiles; **every entry** of tessera's published `knn_entries.yaml` (its base is a series) | **High** for those users: a sized entry pairs the whole base with ground truth over its first rows; a series entry cannot be loaded by jvector at all. Predates this arc | M11 (done): list only profiles stated exactly, and report the rest |
| 8 | Behaviour changes in the CLI and the explorer (§6, §7) | CLI users and scripts | Low to medium | Release notes |

---

## 2. Scope

| Commit | Change |
|---|---|
| `11fcda1` | `catalogs.yaml` entries resolve against the file's directory; nested names kept |
| `99c59d3` | Step 1: `Location` |
| `c5c53af` | Step 2: one dataset description, one inheritance rule |
| `33cf145` | Step 3: `DatasetConfig::builder`; the five writers use it |
| `9efd369` | Step 4: one `TestDataGroup` constructor |
| `1d4eb83` | Step 5: `EntryDescription`, `CatalogEntry::builder`, one knn_entries parser |
| `0c6cd86` | Step 6: the `CatalogSource` trait, located and built catalogs |
| `e07b191` | Step 7: one catalog generator |
| `d04e29c` | Step 8: one source assembly; the explorer over a program's catalogs |
| `d04bb13` | Step 9: docs and an example |

---

## 3. Dataset impacts

### 3.1 Method

- **Probe.** The same program was built against 2.5.1 and HEAD. For
  each dataset it records two views of every profile:
  - **reader:** what `TestDataGroup::load` gives a program;
  - **catalog:** what the catalog generator lists, which is
    `load_and_resolve` plus template-derived views.
- **Datasets.** It was run on copies of the `dataset.yaml` and
  `variables.yaml` of all 105 real datasets on the development machine.
- **Generator preview.** Tessera's catalogs were generated with HEAD
  into scratch (the dataset was not touched) and diffed against the
  copies that were published: the last push
  (16 Sep, 22:39) came after the catalogs were written (22:35).
- **Generator comparison.** Both versions' `catalog generate` were run
  on the same eight-dataset tree. Scratch output is under `target/`
  (`impact-*`).
- **Probe correction.** The first probe recorded each facet's source
  string but not a window declared as a field (`source:` + `window:`).
  It reported windows as missing in 2.5.1 that were present. The
  re-run records both, and §3.3 and §3.4 use it.

### 3.2 Tessera: the `*-unfiltered` profiles (measured)

**What the published catalog says.** 46 of tessera's 223 profiles are
`*-unfiltered`. Each is declared `inherits: default` at its own size,
with its own plain ground truth (for example, `300k-unfiltered` names
`profiles/300k/neighbor_indices.ivecs`). The published `catalog.json`
and `catalog.yaml` also list five of the *default* profile's
per-profile outputs under each of them:
- `metadata_results`;
- `prefiltered_neighbor_indices`, `prefiltered_neighbor_distances`;
- `postfiltered_neighbor_indices`, `postfiltered_neighbor_distances`.

All five point at `profiles/default/…`, computed over the full 496M
base. They are wrong for any sized profile.

**Finding.** The profile data is not misaligned; its *description* is.
- **On disk:** every `*-unfiltered` profile has the right files.
- **In `dataset.yaml`:** each profile declares the right files.
- **In the published derived files:** each of the 46 is also credited
  with five filtered outputs that belong to the full-size default.

An unfiltered profile has no filtered outputs by definition. Its
filtered children (`*-uniform-*`) declare their own. So the correct
state is these five facets absent, not pointed somewhere else.

**Recommendation.**
1. **Correct the derived files by regenerating them** from the
   unchanged `dataset.yaml` with the fixed generators. Do not edit
   `dataset.yaml` or any profile data. Do not hand-edit the derived
   files: they are reproducible, and the generators that write them
   are now right (§3.8).
2. **Make the republish change only those 230 entries.**
   - Land M9 first, so the run cannot rewrite `dataset.yaml`.
   - Land M2 first, so facet order does not churn.
   - Run the republish before M10, so no unrelated docs sections
     appear in the same push.
3. **Keep the rest of these profiles as they are.** They inherit
   `query_vectors` and `metadata_predicates` from the default, and both
   are correct: query vectors and the predicate slab are the same at
   every size. Their `metadata_content` and `base_vectors` windows are
   correct in both versions (§3.3).
4. **Tell consumers, in the release notes and to fastervector, which
   results are suspect** (M8). These are any filtered-search results
   for an `*-unfiltered` profile whose files were taken from the
   catalog listing rather than from an open. Those runs scored against
   the full-size default's filtered ground truth, so they are invalid.
   Opens never exposed these files, and jvector reads
   `knn_entries.yaml`, which never listed them. So the exposure is
   limited to tools that fetch by listing.

**Correction to SRD §12.2.** That section says the catalog listed the
default's *ground truth*. It listed the default's *filtered* outputs,
which are ground truth for filtered search. The plain `neighbor_indices`
and `neighbor_distances` listed are the profile's own, and correct.

**Where the defect is (verified).** Not in `dataset.yaml`. The 46
profiles declare `inherits: default` at their own `base_count` and
declare no filtered facets. Both readers, and HEAD's generator, withhold
per-profile facets across that size change. 2.5.1's *generator* resolved
a named parent by copying every facet the child lacked, which is what
put the five facets into the derived files. The same 47 copies (the
default's own, plus 46) are in:
- `catalog.json` and `catalog.yaml`;
- `dataset.json`, a published sidecar that no reader opens through;
- `docs/dataset.md` and `docs/exemplars.md`, and their `docs/html/`
  renderings.

**No `dataset.yaml` edit fixes it (measured).** Two candidate edits
were tried on a copy, under both versions:
- dropping `inherits: default` from the 46 profiles. Both versions
  reject a v3 profile that names no parent;
- declaring the five facets empty (`~`). Both versions reject it.

Any other edit would declare files these profiles do not have.
`dataset.yaml` already says the right thing, and HEAD reads it
correctly. The fix belongs in the generators, which this arc already
fixed (§3.8). The derived files are regenerated from the unchanged
`dataset.yaml` (M6).

**Who sees it:**
- **Listings and selectors** read the catalog layout: `datasets list`,
  `describe`, the explorer's facet columns and size estimates, and
  selector completion. For these 46 profiles they show filtered facets
  that do not belong to them. The explorer's size column counts the
  default's full-size filtered files against them.
- **Opening a profile** reads the dataset's own `dataset.yaml` through
  the reader. That withheld these facets in 2.5.1 and still does, so no
  program has read the wrong files through an open.
- **jvector (`knn_entries.yaml`)** lists only base, query and plain
  ground truth. HEAD's `knn_entries.yaml` (scratch) is **byte-identical**
  to the published one.

**What HEAD's generator writes (preview, in scratch).** This is the
content M6 must produce (§3.8 has every derived file).
`knn_entries.yaml` is unchanged.
`catalog.json` and `catalog.yaml` change in two ways:
1. the 230 removals above (46 profiles × 5 facets), and nothing else in
   content;
2. facet order within profiles: inherited facets such as
   `query_vectors` and `metadata_predicates` now follow the profile's
   own (§3.4). This accounts for most of the 2,750 changed lines.

**Old and new clients after M6.**
- A 2.5.1 client reads the same `dataset.yaml`, so its opens are
  unchanged. Its listings lose the wrong facets.
- The data files and `dataset.yaml` do not change. Only the
  regenerated derived files, and their `SHA256SUMS` lines, are
  re-uploaded.

### 3.3 Tessera: `*-uniform-*` profiles and `metadata_content` (measured; retracted)

128 profiles inherit from a `*-unfiltered` profile at the same size.
That parent declares `metadata_content` with a window (`[0..N]`).

An earlier version of this section said 2.5.1 readers dropped the
window. **That was wrong**; the first probe did not record field-form
windows (§3.1). 2.5.1's reader windows an inherited `metadata_content`
to the profile's `base_count` (`inherit_with_window`). The corrected
probe gives the same bounds for all 128 profiles under both versions.
HEAD writes the window as a path suffix (`…slab[0..300000]`), where
2.5.1 used a `window` field: same bounds, different text. No impact,
and no mitigation.

### 3.4 Every dataset: what changed on 105 datasets (measured)

**Reader side, what a program opening the dataset sees:**

| Change | Datasets | Profiles | What it means |
|---|---|---|---|
| Facets keyed by jvector aliases (`base`, `query`, `indices`, `distances`) are now read | 36 | 188 | 2.5.1 readers saw **no** base, query or ground truth in these datasets, which were converted from HDF5 or written jvector-style. They now open. |
| Compactly declared sized profiles (`sized:`, `strata:`) appear | 6 | +40 | Profiles a reader could not see before |
| Oracle-partition profiles lose metadata that was not theirs | 6 (sift1m copies) | 72 | 2.5.1 gave them the default's `metadata_content` and `metadata_predicates`, which describe other rows |
| Window *spelling* only: `[0..N)` becomes `[0..N]`, or a `window` field becomes a path suffix | several, tessera included | 750 facets | Same bounds (corrected probe: no facet present in both versions differs in source or bounds). Visible only to code that reads `FacetConfig::source()` / `window()` as text |

**Catalog side, what generated catalogs list:**

| Change | Datasets | What it means |
|---|---|---|
| Datasets with no `name:` now catalog | 36 | 2.5.1's generator refused them; the name comes from the directory |
| Copied filtered outputs removed | 2 (tessera copies) | §3.2 |
| Attributes the type has no field for are kept | any that use them | Catalog `attributes` gain keys such as `dimension`, `base_count`, `vector_type` (additive) |
| Facet order within profiles | any with inherited facets | Same JSON content, different bytes (§3.5) |

### 3.5 Regenerating catalogs: 2.5.1 vs HEAD on one tree (measured)

On the eight-dataset tree, with a nested group:

| Catalog | Result |
|---|---|
| 4 per-dataset catalogs | byte-identical |
| 3 catalogs | same JSON content, facets reordered |
| tessera's | content differs: §3.2 |
| one dataset's | content differs: unknown attributes now kept |
| root and group catalogs | content differs: the above, aggregated |
| 2 datasets with no `name:` | catalogs now written; 2.5.1 wrote none |
| root `knn_entries.yaml` | 2 entries added, for the two datasets above |

**Consequence.** The next `gen-catalog` in any dataset's finalize
rewrites its catalog files, even where nothing meaningful changed, and
the next push uploads them. The files are small. M2 removes the reorder
part.

### 3.6 Cache identity (measured where possible)

The cache layout (`<root>/<dataset>/<files>`, `.publish_url`,
`origin.json`) is unchanged.

| Possible effect | Measured exposure | Result |
|---|---|---|
| A canonical catalog entry's relative `path` now joins by URL rules rather than concatenation. This changes the *listing's* home only for paths with `..`, spaces or encodable characters; opens already used URL rules. | 208 entries in 119 local `catalog.json` files: no such path | none |
| A `knn_entries.yaml` opened by directory URL now resolves facets by concatenation, so a name needing encoding caches under a different file name | No local knn facet name needs encoding | none (inferred: a re-download, never an error) |
| An embedded catalog entry resolves relative facets against its home rather than the catalog base | knn parsers make facets absolute | none |
| `hdf5`-type entries | 21 local entries | open as before |

### 3.7 `knn_entries.yaml` directory opens (measured: the High item)

- **2.5.1:** `TestDataGroup::load(<dir>)` on a directory holding only
  `knn_entries.yaml` used its own parser, which ignored `_defaults` and
  read files beside the YAML.
- **HEAD:** the catalog's parser is used, so `_defaults.base_url` and
  `cache_dir` are honoured (B-3).
- **Measured:** four local directories have only `knn_entries.yaml`, an
  `s3://` base URL, the data files beside it, and no `cache_dir`. 2.5.1
  read them locally. HEAD fetches the same files from S3, or fails in
  offline mode.

M1 keeps HEAD's behaviour where the files are not there, and 2.5.1's
where they are.

**Done (M1), as jvector's rule.** jvector's loader takes an entry's
local directory as its `cache_dir`, else `$DATASET_CACHE_DIR`, else the
catalog file's own directory, and reads a file from there when it
exists, fetching from `base_url` only when it does not. The parser now
does the same; a remote catalog has no directory of its own to look in.
The environment value is a parameter of the parser's inner function, so
tests never set the process environment.

**A second regression, found while measuring.** One of the four
directories repeats an entry key (a newer copy of an entry appended
below the old one). 2.5.1 read it, keeping the last copy, as jvector's
YAML loader does; HEAD's parser rejected the whole file. It now keeps
the last copy and warns about each repeated key.

**Measured after both:** all four directories read every facet locally
(3, 399, 39 and 3 facets), with none from their remote base.

### 3.8 Tessera: regenerating the derived files (measured, in scratch)

HEAD's finalize commands were run one by one in a scratch copy of
tessera under `target/`: `generate dataset-json`,
`analyze describe-dataset --exemplars`, `generate docs-html` and
`catalog generate`. The scratch copy held copies of the root files,
with the profile data linked file by file. Afterwards, tessera's
`dataset.yaml` hash and its 2,196 profile files (size and mtime) were
unchanged.

| File | Compared with the published copy |
|---|---|
| `dataset.json` | Content differs by exactly the 230 removals (46 `*-unfiltered` profiles × 5 facets); profile order unchanged |
| `catalog.json`, `catalog.yaml` | Content differs by exactly the 230 removals |
| `knn_entries.yaml` | Byte-identical |
| `docs/dataset.md` | The 230 removed rows, plus row order within profile tables (M2). Everything else matches, once three scratch-only effects are set aside: link annotations, the Data Flow `provided:` line (it marks linked files), and the §3.9 sections |
| `docs/exemplars.md`, other `docs/html/*` | Byte-identical, except `dataset.html` (it renders `dataset.md`) |

So the fixed generators already produce the right files. What stands
between them and tessera is how they are run (§3.9) and the byte churn
from facet order (M2).

### 3.9 Running the finalize steps on tessera (measured, dry run in scratch)

- **Compute steps:** `veks run --dry-run` with HEAD, against tessera's
  own progress log, reports all 614 compute steps fresh. Nothing would
  be recomputed.
- **Finalize steps:** all 8 would run, because the binary changed
  (`tag-profiles`, `generate-dataset-json`, `-variables-json`,
  `-dataset-log-jsonl`, `-docs`, `-docs-html`, `-catalog`, `-merkle`).
- **The closing save rewrites `dataset.yaml`.** `veks run` ends with
  `update_dataset_attributes`. That re-renders the file and writes it
  when the bytes differ. A scratch replica of that save, on tessera's
  `dataset.yaml`, changes 851 lines:
  - the renderer's strata and profile-tag comments are added;
  - explicit `base_vectors` and `metadata_content` windows that equal
    the inherited view are dropped.

  Both versions resolve the original and the rewritten file to the
  same facets, so the description is the same. But the file changes,
  and with it the content key every finalize step holds. This must not
  happen to tessera (M9).
- **Docs workspace.** For `veks run dataset.yaml`,
  `crates/veks-pipeline/src/pipeline/mod.rs:514` takes the workspace
  from `Path::parent()`, which is the empty path. `describe-dataset`'s
  `read_dir("")` then fails silently, so the published `dataset.md`
  has no "Source Data" table and no "Total size on disk" row. A run
  with a non-empty workspace includes both. The "Source Data" table
  lists absolute host paths of symlink targets
  (`/mnt/…/upstream/…`). Whether published docs should carry those is
  a decision (M10).

### 3.10 `knn_entries.yaml` and what it can state (verified)

jvector's loader (`DataSetLoaderSimpleMFD`) reads `base`, `query` and
`gt` as plain file names. It has no window, no series pattern and no
namespace. The generator wrote each facet's path with its window
dropped, so:
- a **sized profile** was listed against the whole base file, beside
  ground truth computed over its first rows;
- a **sharded base** was listed as its pattern
  (`base_vectors__NNNN.fvecs`), which jvector would try to fetch as one
  file.

Tessera's published `knn_entries.yaml` lists every profile with the
series pattern, so no entry in it can be loaded by jvector as written,
and the sized ones would be wrong if they were. This predates this arc;
the earlier "byte-identical" finding (§3.2) only said the arc did not
change it.

The catalog-mirror check (M12) found it on its first run: an e2e
dataset's sized profiles disagreed with `knn_entries.yaml`.

**Done (M11):** `knn_entries.yaml` lists a profile only when its base
vectors, query vectors and neighbor indices are each one whole file (no
window, no series, no namespace). Every other profile that has all three
is left out and reported (`knn_entries.yaml omits <dataset>:<profile>:
<reason>`), once, at the dataset's own catalog. A directory whose
listing would be empty has its stale `knn_entries.yaml` removed (also
reported), since the generator owns that file as it owns `catalog.json`.
`merkle create` now also removes a `.mref` whose file is gone, so the
removed file's sidecar goes with it.

**Tessera (measured, in scratch):** all 223 profiles are omitted
(`base_vectors is a multi-file series`), and `knn_entries.yaml` is
removed. The catalogs are written as before. The real dataset has no
`.mref` without its file today, so only `knn_entries.yaml.mref` would
go.

**On publish:** removing the local files does not remove the remote
copies. `veks publish` deletes remote objects with no local counterpart
only with `--delete`, which needs a message (`-m`). Without it, jvector
users would keep fetching the old `knn_entries.yaml`.

---

## 4. Published-catalog and tooling impacts

- **The manual generator now writes what the pipeline writes.**
  `veks prepare catalog generate` adds `knn_entries.yaml` at every
  level, adds template-derived views, and reads sized profiles through
  `variables.yaml`. The pipeline's output is byte-identical to the step
  6 build (§12.7); against 2.5.1 it differs as in §3.5.
- **`.do_not_catalog` means no descent, for both generators.**
  Datasets under a marked directory are in no catalog. The manual
  command used to list them in the catalogs above. One marker exists
  locally, at the root of a test-data tree that is not published.
- **`veks check catalogs`** expects exactly what the generator writes,
  so a dataset under `.do_not_catalog` no longer fails the check.
- **The Java toolchain** reads the same catalog wire form. It now sees
  extra attribute keys, and loses the five wrong facets under tessera's
  unfiltered profiles. Inferred to be tolerant, not tested.

---

## 5. Library API changes (`vectordata`)

**Breaking.** Downstream code updates to the new form; no shims are
kept (see the API policy above). Each row: what changed, who is known
to be affected, and the replacement. This table is the source for the
M8 migration guide.

| Item | Change | Known users | Migration |
|---|---|---|---|
| `CatalogEntry` fields `name`, `path`, `dataset_type`, `layout`, `catalog_file`, `catalog_name` | Private, behind accessors | fastervector `fvec-bench` reads `e.name` (2 sites) | `name()`, `location()`, `dataset_type()`, `profiles()`/`attributes()`/`layout()`, `catalog_file()`, `catalog_name()`. `path` and `layout` changed meaning: use `location()` for where the entry lives, and `layout()` only for the listing summary |
| `CatalogEntry` struct literals | Not possible | none known | `CatalogEntry::builder` |
| `CatalogEntry::dataset_home_url` | Returns `&str`, not `String` | none known | `.to_string()` where an owned value is needed |
| `vectordata::knn_entries` (`KnnEntries`, `KnnEntry`) | Removed | none known | `catalog::knn_entries` (catalog parser), or open the directory as a catalog |
| `CatalogSources::required()`, `optional()` | Removed | none known | `iter()`, filtering on `Requirement` |
| `CatalogSources::diagnostics` | Returns `Vec`, not a slice | none (fastervector uses `Catalog::diagnostics`, which is unchanged) | Mostly source-compatible |
| `catalog::sources::catalog_file_for`, `ensure_trailing_slash` | Removed | none known | `location::dir_form` |
| `config::verify_catalog_source(source)` | Now `(name, source)` | none known | Pass the name |
| `datasets::build_sources` | Returns `RunSources`, not `CatalogSources` | CLI-support only | `.sources()` or `.open()` |
| `datasets::{describe,ping}::run_args` | `at_extra` parameter removed | CLI-support only | Drop the argument |
| `DataSource` (enum) | Removed | none known | `Location` |
| `GenericTestDataView::new`, `with_attributes` | Take `Location`, not `DataSource` | none known | `Location::parse` or `from_path` |
| `DatasetConfig.header`, `DatasetAttributes.other`, `CatalogDiagnostic.catalog`, `PrecacheRequest.catalog` | New public fields | Struct literals without `..Default::default()` | Add the field, or use the default |
| `DatasetConfig::parse` | Error type is `serde_yaml::Error` | new in this arc | — |

**Behaviour changes in the library.** Same signatures, different
results:
- `TestDataGroup::load` and `model::DatasetConfig` deserialization
  follow the one inheritance rule (§3.4), read the jvector aliases,
  accept a missing `name`, and expand compact sized profiles.
- `Catalog::of` lists catalogs in the order they were added. 2.5.1
  listed required catalogs first. This changes listing order for
  fastervector_s2 (`configure_default().add_catalogs(..)`) and
  fastervector's mixed set.
- `Error::AmbiguousDataset` messages name the catalog document (for
  example `…/catalog.json`), not just its directory.
- A remote catalog directory is also probed for `catalog.yaml`, which
  costs one more request for a knn-only remote catalog.
- `catalogs.yaml` relative entries resolve against the file's
  directory, not the working directory (`11fcda1`). Nested
  `catalogs.yaml` names are kept.

**Additive** (no impact): `Location`, `DatasetConfig::builder`,
`CatalogEntry::builder`, `EntryDescription`, `CatalogSource`,
`CatalogBuilder`, `LocatedCatalog`, `CatalogSources::with` /
`with_optional` / `iter` / `reload_configured` / `add_named_catalog`,
`TestDataGroup::from_config`, `catalog::generate`,
`explore::run_with_sources`.

---

## 6. CLI behaviour changes

- **Exit status.** A catalog named with `--at` or `--catalog` that fails
  to load exits with status 1, in every `datasets` command. Configured
  catalogs that fail are reported and the command continues.
- **`describe` and `ping`.** `--at` replaces the configured catalogs,
  as everywhere else.
- **`veks catalog generate`** is now the pipeline command its help
  shows (`--source`). `veks prepare catalog generate <INPUT>` is the
  manual command.
- **The manual generator's output** is as described in §4.
- **The explorer:**
  - reports catalogs that fail to load, on its status line and with `✗`
    on the catalogs screen;
  - opens the catalog once per pass rather than once per action;
  - lists the catalogs screen in set order;
  - finds a dataset to purge case-insensitively.
- **`dataset.yaml` writers** (`prepare bootstrap`, `datasets derive`,
  `generate example-dataset`, `gen_dataset`, `gen_derive`):
  - their output parses to the same description as before;
  - its layout is the renderer's (for example, structured `strata:`);
  - it carries the strata and tag-schema comments.

  Scripts that grep these files by text layout may need updating; two
  of the repo's own test helpers did.

---

## 7. Downstream consumers (verified from local repos)

| Consumer | Pin | Uses | Breaks? |
|---|---|---|---|
| fastervector `fvec-bench` | `vectordata = "2.5.1"` (accepts any 2.x ≥ 2.5.1) | `CatalogSources` (`new`, `configure`, `configure_default`, `add_catalogs`), `Catalog::of`/`lookup`/`datasets`/`open_spec`/`diagnostics`, `Severity`, `AmbiguousDataset`, fetch, `set_cache_dir`, `shell::bin_main_as` | **Yes:** `e.name` at `catalog.rs:327` and `:352`. Fix: `e.name()` (M8 sends this) |
| fastervector_s2 `fvec-data` | `"2.2"` (accepts any 2.x) | `CatalogSources::new().configure_default().add_catalogs()`, `Catalog::of`, `VectorReader`, errors | No compile break; listing order changes |
| polydat, nmbrs, native-vector-rust-study, nb-rs copies | `1.x` | — | Not reached by a 2.x or 3.x release |
| jvector and the Java toolchain | file formats only | `knn_entries.yaml`, `catalog.json` | No for `knn_entries.yaml` (byte-identical for tessera). Catalogs: additive keys and the tessera fix |

---

## 8. Not impacted (verified or measured)

- The cache layout and `origin.json` contents (§3.6).
- `dataset.yaml` files: no published one changes, and no data file
  changes.
- Tessera's `knn_entries.yaml` (byte-identical) and its data files.
- The 21 `hdf5` catalog entries; catalog wire field order.
- Offline opens from kept copies (guarded by `tests/offline.rs`).
- Fetch, progress and `set_cache_dir`.

---

## 9. Proposed mitigations

| # | Mitigation | Removes | Effort | Recommendation |
|---|---|---|---|---|
| **M1** (done) | For a `knn_entries.yaml` opened by directory, prefer a facet file present beside the YAML; otherwise use `_defaults.base_url` / `cache_dir` as now. This is jvector's local-copy-first rule, with the directory as an implicit cache. | §3.7 (High) | Small: a parser option, plus a guard test on the four-directory shape | **Do it** |
| **M2** (done) | Keep the 2.5.1 facet order when inheritance fills a profile: the parent's facets in its order, then the profile's own additions | §3.4/§3.5 reorder churn on every other dataset's next finalize; and tessera's regenerated files (M6) then differ from the published ones only by the removals | Small: order within `resolve_inheritance`, checked against the eight-dataset tree and tessera | **Do it** |
| ~~M3~~ | *Withdrawn.* Keeping `CatalogEntry::name` as a public field would reopen the mutable-entry hole step 5 closed, for one downstream's convenience | — | — | Not done: fastervector changes `e.name` to `e.name()` |
| ~~M4~~ | *Withdrawn.* Deprecated shims for the removed calls | — | — | Not done: the API policy is no shims; §5 gives each replacement |
| **M5** | Release as **3.0.0**, not 2.6.0, carrying M8's notes | Silent breakage of `^2.x` pins on `cargo update`; each integration takes the break when it bumps its pin | None | **Do it**, after M8's notes are written |
| **M6** | Regenerate tessera's derived files with the fixed generators, by its own finalize steps; **`dataset.yaml` and the profile data are not touched** (§3.8). Order: M2 and M9 land first; M10 after the republish, so it adds nothing to this push. Then: a dry run in scratch confirms the compute steps are still fresh; record `dataset.yaml`'s hash and a size+mtime snapshot of `profiles/`; run `veks run` on tessera; assert both unchanged; diff each regenerated file against the published copy (expected: the 230 removals, `knn_entries.yaml` and its `.mref` removed (M11), and nothing else). You then publish with `--delete -m …`, so the remote `knn_entries.yaml` and `.mref` are removed too. Running the steps through `veks run`, rather than one by one, keeps their provenance records consistent | §3.2 | Small once M2 and M9 are in; the user runs the publish | **Do it**, after M2 and M9 |
| ~~M7~~ | *Withdrawn.* Explicit `metadata_content` windows were for a 2.5.1 defect that does not exist (§3.3) | — | — | Not done |
| **M9** (done) | `veks run`'s closing save (`update_dataset_attributes`) writes `dataset.yaml` only when the description changes: if what it would write parses to the same description as the file on disk, the file is left byte-for-byte. The same rule as the writers' parse-equality acceptance (step 3). Guard: a `dataset.yaml` in a non-renderer layout (explicit inherited windows, no renderer comments) survives `veks run` byte-identical; proven to fail on HEAD. Also confirm `tag-profiles` leaves the file alone when no tag changes | §3.9's rewrite, for tessera and every hand-edited dataset | Small | **Do it**, before M6 |
| **M10** | Give the docs steps one workspace path: take `mod.rs:514`'s parent the way `:405` does (empty → `.`), so docs no longer depend on how `veks run` was invoked. Decide what "Source Data" publishes: file names and sizes only, or the host paths of symlink targets as now | §3.9's invocation-dependent docs | Small; the content choice is yours | **Do it**, with the content decision, after M6's republish |
| **M11** | `knn_entries.yaml` lists only profiles it states exactly, and reports each omitted one (§3.10) | §3.10 | Small | **Done.** Guard tests fail without the rule |
| **M12** | Tests: a test never takes catalog data over a `dataset.yaml`, and projections must mirror it. A shared check (`crates/veks/tests/support/catalog_mirror.rs`) runs after every successful e2e pipeline run, which now marks its dataset publishable. It requires `catalog.json`, `catalog.yaml`, `dataset.json` and `knn_entries.yaml` to say what the reader reads from `dataset.yaml`; the knn check works out exactness independently. Hand-written stub catalogs beside a real `dataset.yaml` are replaced by generator output (`write_catalogs` fixture) | Projections drifting from `dataset.yaml` unnoticed (no e2e test wrote catalogs before) | Small | **Done** |
| **M8** | Written **before** the release and shipped in it: release notes with a migration guide built from §5 (old call → new call), plus the behaviour changes in §5–§6. State which tessera results are suspect (§3.2, recommendation 4). Tell fastervector directly: the two `e.name` sites, the listing-order change for mixed sets, and the `--at` exit status, when the release is published | Surprise; time spent finding each replacement | Small | **Do it** first: the notes are part of M5 |

**Order:** M8's notes, then M5 (the 3.0.0 release, notes included, and the note to fastervector), then M6 (tessera regenerated by the released binary, so its files carry a released version), then M10.

With M1, M2, M5 and M9 done, M6 run, M10 after it, and M8 sent:
- **Libraries:** existing 2.x pins keep building against 2.5.1.
  Bumping to 3.0.0 is a deliberate step, with the migration guide in
  hand. fastervector's known change is two call sites; fastervector_s2
  builds unchanged and sees only the listing order.
- **Catalogs:** regenerated catalogs differ from 2.5.1's only where
  2.5.1 was wrong or lossy. Tessera's derived files are regenerated
  from its unchanged `dataset.yaml` (M6, after M9).
- **Local data:** the four knn directories keep reading local files.
