# SRD — Loose files, attribute overlays and dataset integrity

**Status:** proposed 2026-10-21. The decisions in §9 were taken on
2026-10-21. The release that
carries it, and the version it takes, are the maintainer's.
**Scope:** `vectordata` and `veks-io`. veks-io becomes the one
implementation of every file format, which vectordata reads and writes
through (§6.1). Making
a dataset of files vectordata did not produce; stating or correcting a
dataset's attributes on one host; checking a dataset's facets against
one another and measuring what a descriptor declares; a content
identity for every fetched facet; four library-surface gaps.
**Builds on** `srd-runtime-catalogs.md` (catalog sources, the program
catalog route), `srd-profile-selectors.md` (spec heads and selectors),
`srd-element-access.md` (the element model, `facet_as`, `veks check`'s
ground-truth checks), `srd-multifile-facet-shards.md` (`__NNNN` series).

**Origin.** A benchmark program (`fastervector`) keeps four pieces of
dataset logic of its own: a five-line `dataset.yaml` its users must
type beside three bare files; a table from dataset name to distance
function (`dataset-metadata.yaml`) and one of index defaults
(`index-defaults.yaml`), consulted after vectordata answers; its own
checks that the base, queries and ground truth belong together; and a
sampled unit-length test. It also keeps a direct `reqwest` dependency
for one GET, three hand-written xvec writers, a wrapper that quiets
the fetch meter, and refuses `completions` in its embedded CLI. It
asked for each of these to be vectordata's. The requester is one
consumer and does not shape the interface; each requirement is
general: any reader of a dataset should see one answer.

## 1. Problem

1. **A directory of loose files is not a dataset.** `TestDataGroup::load`
   on a directory with `base.fvecs`, `query.fvecs` and `gt.ivecs` and
   no descriptor fails ("no dataset.yaml"), and so do `open_spec`,
   `describe` and `precache`. Only `veks`, which a reader need not
   depend on, writes a descriptor.
2. **Attributes can only be stated by the publisher.** A
   `knn_entries.yaml` catalog carries no `distance_function`,
   `is_normalized` or `base_count`. A host that knows them has nowhere
   to say so that `describe`, `list`, `explore` and `view.distance()`
   read, so each program keeps a side table keyed by dataset name and
   `describe` never shows it.
3. **Nothing checks that facets belong together on open.** `veks check`
   checks a dataset being prepared. A reader opening someone else's
   dataset gets no check that the query dim equals the base dim, that
   the ground truth has a row per query, or that every index is below
   the base count. That last check is the only one that proves the
   truth was computed against this base.
4. **A declared attribute is never compared with the data.** `is_normalized:
   true` over a base whose norms are 3.7 is believed.
5. **A fetched facet has no content identity** unless its publisher
   wrote a `.mref`. A facet fetched through `ChunkStore` (no merkle
   reference) is trusted byte for byte and never hashed
   (`chunked_http.rs`, "No merkle verification"), so a consumer that
   fingerprints its inputs must hash the whole file itself or sample.
6. **`config catalog add .`** saves `0: .`: the path is stored relative,
   so it resolves only from that working directory. The name falls
   back to an index because `.` slugs to nothing, and `config catalog
   remove <dir>` cannot find it.
7. Four library gaps:
   - `transport::fetch_if_modified` is `pub` inside a `pub(crate)`
     module, so a program cannot reach it.
   - No typed xvec writer is public. veks-io's `xvec::writer::open` is
     byte-level and discards write errors.
   - `LogMeter` reports every plan, including a plan with nothing to
     fetch.
   - An embedded CLI's completions are bound to the `vectordata`
     executable.

## 2. Principles

- **One answer per question, inside the access path.** Inference,
  overlays and measurements are applied where an entry or a group is
  built. Every consumer (`describe`, `list`, `explore`, the view)
  therefore sees the same value. No reader ever consults a side table.
- **Ambiguity is refused, never settled.** A choice between candidates
  is never made by order, size, mtime or position, as for catalogs.
  The refusal names every candidate.
- **Inferred is reviewable.** What was inferred is said once, can be
  written down, and once written is declared, not inferred.
- **Provenance travels with the value.** Every attribute says whether
  it came from the dataset, an overlay (and which file), or a
  measurement.
- **Conservative warnings.** Whenever a probe or a measurement
  disagrees with an overlay value, vectordata warns once. The warning
  names the file, the key and both values. The overlay stays in force,
  because the host said so. A disagreement with the dataset's own
  declaration is an error from `verify()` (§5).
- **No new files in the cache or config layout without the
  maintainer's sign-off.** The layout changes here are the ones §9
  settles: the content of `.chunks`, `measured.yaml` in a dataset's
  cache directory, and `overlays.yaml` in the config directory.
- **One format implementation.** Every byte layout (xvec, vvec and its
  index, scalar columns) is read and written by veks-io. vectordata
  supplies where the bytes come from, never how they are laid out
  (§6.1).

## 3. Loose files (LF)

### 3.1 When inference applies

- **LF-1 (I12).** Inference applies only to a **local directory** head,
  meaning a path given to `TestDataGroup::load`, `open_spec`,
  `describe`, `precache` or `derive`, and only when the directory holds
  neither `dataset.yaml` nor `knn_entries.yaml`. A catalog name never
  infers. A URL never infers, because a server directory cannot be
  listed. Its refusal stays "no dataset.yaml or knn_entries.yaml".
- **LF-2 (I11).** An inferred dataset is opened in memory as a
  `DatasetConfig`, exactly as if the descriptor had been written. It is
  reported once per process through `log::info!` and through the
  describe output's `descriptor: inferred` line, as `inferred
  dataset.yaml for <dir>: base=<file>, query=<file>,
  neighbor_indices=<file>[, neighbor_distances=<file>]`.
  `vectordata datasets init <dir>` writes the same descriptor as
  `<dir>/dataset.yaml`. It refuses an existing file unless given
  `--force`, and prints what it wrote. After that, opens are declared.
  The library form is `dataset::infer::infer_dir(&Path) ->
  Result<Inferred, InferError>`. `Inferred` holds the `DatasetConfig`,
  the rule that classified each file, and the notes.

### 3.2 The rules, applied in order

| # | Rule |
|---|------|
| **LF-3 (I1)** | **Candidates.** Regular files in the directory (not recursive except as LF-10 says) whose extension `FacetFormat::from_extension` knows. Skipped: hidden files, `IDXFOR__*` sidecars, `*.chunks`, `*.mrkl`, `*.mref`, names containing `#`, and `*.provenance.json`. |
| **LF-4 (I2)** | **Canonical name.** `StandardFacet::classify(file)` as it stands: `base_vectors.fvec`, `neighbor_indices.ivecs`, `metadata_content.slab`, and so on, plus a `__NNNN` series. Every standard facet is reachable this way. |
| **LF-5 (I3)** | **Role token.** For a file LF-4 left unclassified, split the stem on `_` and `-`. Exactly one token, or one adjacent token pair (for `ground_truth`), must be a role alias. The aliases are `base`, `train` → base; `query`, `queries`, `test` → query; `gt`, `groundtruth`, `ground`+`truth`, `neighbors`, `indices` → neighbor_indices; `distances` → neighbor_distances. The vocabulary is `StandardFacet::from_alias`, restricted to these four facets. The format must be one that facet accepts, so `*_query_*.ivecs` is refused rather than reinterpreted. Two role tokens in one stem leave the file unclassified, and it is listed in the notes. |
| **LF-6 (I4)** | **Shape, only to finish.** If LF-4 and LF-5 leave a role of the minimum set (LF-8) unfilled, the remaining candidates are classified by their headers. An integer xvec whose record count equals one float xvec's count is the ground truth, and that float xvec is the queries. The remaining float xvec with the most records is the base. This applies only when it yields exactly one candidate per unfilled role, and the base must have more records than the queries. Otherwise the inference is refused with every leftover file's dim and count listed. A file classified by name is never reclassified by shape. |
| **LF-7 (I5)** | **One file per role.** Two candidates for one facet is a refusal naming both, and the rule that matched each. A `__NNNN` series is one candidate. It must be contiguous from `0000`, and its first gap is named. |
| **LF-8 (I6)** | **Minimum set.** base, query and neighbor_indices are required. neighbor_distances is taken when present. Every other facet (metadata, filtered, quantization) is taken only by LF-4. A missing role is a refusal naming the role and listing the unclassified files. |
| **LF-9 (I7)** | **Verified before accepting.** The §5.1 checks run on the inferred set: header dim > 0, size an exact stride multiple, query dim = base dim, gt rows = query rows, gt dim > 0, every gt index in `0..base_count`, and distances rows = query rows with distances dim = gt dim. A failure refuses the inference with §5.1's message. This is the same code as `verify()`, not a copy. |
| **LF-10 (I10)** | **veks layout without a descriptor.** A directory with a `profiles/` subdirectory and no descriptor is a veks-layout dataset whose `dataset.yaml` is missing. Each `profiles/<p>/` is a profile classified by LF-4 only. `profiles/base/` is the shared directory and lends its facets to every profile that lacks them, as veks writes them. The default profile is `profiles/default/` if present, else the only profile. Several profiles and no `default` is a refusal listing them. Windows, sized profiles and layers are not inferred, because they are not recoverable from file names. Such a profile opens with whole-file facets. |
| **LF-11 (I8)** | **Name and shape.** The name is the directory's last segment, the profile is `default` (outside LF-10), `maxk` = the gt dim and `base_count` = the base record count. |
| **LF-12 (I9)** | **Attributes from explicit tokens only.** First, `distance_function`. Each file's stem and the directory name are scanned for a metric token from `DistanceFunction::parse`'s table, restricted to `angular`, `cosine`, `cos`, `euclidean`, `l2`, `dot`, `ip` and `mips`. It is declared only when every file and name carrying one agree, after mapping through the table. Disagreement (`glove_25_angular` beside `gt_ip_100`) leaves it undeclared, with a note naming both. Second, `is_normalized: true` comes only from a `norm` or `normalized` token. It is then checked by IN-4's sample, and a contradiction refuses the inference. Nothing else is inferred. Each inferred attribute carries provenance `Inferred { file, token }`. |

Worked cases (each a test):

| Files | Result |
|-------|--------|
| `base_vectors.fvec, query_vectors.fvec, neighbor_indices.ivec` | LF-4, the same config as the five-line yaml |
| `ada_002_100k_base_99287.fvecs, …_query_10000.fvecs, …_gt_ip_100.ivecs` | LF-5, `distance_function: DOT_PRODUCT` (LF-12, `ip`) |
| `vectors.fvecs, q.fvecs, truth.ivecs` | LF-6 then LF-9 |
| `a_base.fvecs, b_base.fvecs, q_query.fvecs, gt.ivecs` | LF-7 refuses, naming both bases |
| a 10m-row gt beside a 1m-row base | LF-9 refuses with the first bad row and column |
| `glove_25_angular_base.fvecs, …_query.fvecs, …_gt_ip_100.ivecs` | opens, metric undeclared, with a note |
| `profiles/base/base_vectors.fvec, profiles/default/{query_vectors,neighbor_indices}` | LF-10 |
| `x_query.ivecs` | LF-5 refuses the format, and the file is listed unclassified |

### 3.3 Where it lives

`crates/vectordata/src/dataset/infer.rs`. Its dependencies are
`StandardFacet`, `FacetFormat`, `DistanceFunction`, the xvec header
reader in `io`, and §5's checks. It does not use veks or veks-io.
`group.rs`'s directory cascade calls it in place of the "no
dataset.yaml" error, and `DatasetHandle` records
`descriptor: Inferred`. sysref 01 gets the rule table above.

## 4. Attribute overlays (OV)

- **OV-1. File.** An overlay is a YAML file:

  ```yaml
  # comments are kept; vectordata never rewrites this file
  datasets:
    ada002-100k:                  # any dataset of this name
      distance_function: DOT_PRODUCT
      is_normalized: true
      fvec.ncentroids: 300        # a consumer's key: namespaced, opaque
  catalogs:
    bench:                        # only the entry in catalog `bench`
      ada002-100k:
        base_count: 99287
  ```

  A key is an attribute name `DatasetAttributes` knows (`distance_function`,
  `is_normalized`, `model`, `license`, …), or a stated count
  (`base_count`, `dimension`, `maxk`), or a **namespaced key** containing
  a `.`, whose value is any YAML. A namespaced key is opaque to
  vectordata and is surfaced through the same access (OV-4). Any other
  key is a warning naming the file and the key, and is ignored.
- **OV-2. Sources and precedence**, lowest to highest:
  1. the dataset's own attributes;
  2. inferred attributes (LF-12), which only fill keys the dataset
     leaves unset;
  3. program overlays, registered as `CatalogSources::with_overlay(Overlay)`
     beside program catalogs, in registration order. A key two
     program overlays both set is an error naming both, at
     registration;
  4. the user overlay, `<config_dir>/overlays.yaml`, where
     `config_dir()` is `VECTORDATA_HOME`-isolated as every client
     file is.

  Within one file, a `catalogs.<c>.<name>` block wins over
  `datasets.<name>` for the entry in catalog `c`. An overlay replaces
  a key's value. It never merges inside a value.
- **OV-3. Where it is applied.** In `Catalog::of(sources)`, to every
  entry as it is listed. `CatalogEntry::attributes()` returns the
  merged `DatasetAttributes`, so `list`, its filters, `explore` and
  `describe` see it. For a path or URL open, `open_spec` and
  `open_selection` apply the catalog's overlays to the group by its
  name (`TestDataGroup::name()`). `TestDataGroup::load` alone applies
  the user overlay. `TestDataGroup::attribute`, `view.distance()`,
  `view.distance_function()` and `TestDataView::attributes()` read the
  merged values.
- **OV-4. Access and provenance.** `CatalogEntry::attribute_source(key)
  -> Option<AttributeSource>` and the same on `TestDataGroup`.
  `AttributeSource` is `Dataset | Inferred { file, token } | Overlay {
  file, kind: User | Program(name) } | Measured { at }`. A namespaced
  key is read with `attribute(key)` like any other.
  `entry.attributes_in(namespace)` lists one consumer's keys.
  `describe` prints each attribute with its source when it is not
  `Dataset` (`distance_function  DOT_PRODUCT  (overlay
  ~/.config/vectordata/overlays.yaml)`). `describe --format json`
  carries a `source` field per attribute.
- **OV-5. Probed contradiction (the maintainer's rule).** Whenever a
  facet is opened, every overlay value that can be checked from it is
  checked:
  - `base_count` against the base's record count;
  - `dimension` against its dim;
  - `maxk` against the ground truth's dim;
  - `is_normalized` against IN-4's sample, when that sample has been
    taken;
  - `distance_function` against the dataset's own declaration, when
    the dataset has one.

  A disagreement is a warning, given once per process per (dataset,
  key) through `log::warn!` and shown in `describe`. It names the
  overlay file, the key, the overlay's value and the observed or
  declared one. The overlay stays in force. `describe` marks the line
  `(contradicted: observed 1000000)`. A caller that wants refusal
  calls `verify()`, which reports the contradiction as a failed check
  (§5).
- **OV-6. Editing.** `vectordata config overlay set <dataset> <key>
  <value> [--catalog <c>]` and `config overlay unset …`. These are
  textual line edits that keep every comment, the same mechanism as
  `config catalog add`. `config overlay show [<dataset>]` prints the
  merged view with sources.

## 5. Integrity (IN)

### 5.1 Verification

- **IN-1.** `TestDataView::verify(&self, progress: &mut dyn
  VerifyProgress) -> Result<VerifyReport, Error>`, a provided method,
  plus `DatasetSelection::verify` over every selected profile. It
  checks, for every ground-truth pair the profile has (the plain one,
  prefiltered and postfiltered):
  - V1: every facet's header dim is > 0 and its size is a whole
    stride multiple;
  - V2: query dim equals base dim;
  - V3: gt rows equal query rows (equality, not ≥);
  - V4: gt dim is > 0;
  - V5: every gt index is in `0..base_count`, in one pass over nq×k;
  - V6: distances rows equal query rows, and distances dim equals gt
    dim;
  - V7: every overlay or declared count (`base_count`, `maxk`,
    `dimension`) agrees with the facet.

  It is opt-in. Opening never scans, and V1–V4 and V6 read headers
  only.
- **IN-2. Refusal.** `Error::Verification { check, dataset, profile,
  expected, found, files }`, displayed as `V5 neighbor_indices row
  4127 col 3: index 1000412 ≥ base_count 1000000 (gt.ivecs,
  base.fvecs)`. Each check names the two quantities and the files.
- **IN-3. Progress.** V5 reports through `VerifyProgress` (rows done
  of rows, one line per 2 s by default). The `LogMeter` and
  `TextMeter` families implement it. A test asserts that a capturing
  sink sees progress for a facet over one stride. `veks check`'s
  ground-truth checks (`srd-element-access.md` §9) call this code
  rather than keeping their own copy.

### 5.2 Measured attributes

- **IN-4.** `TestDataView::measure_normalization(&self) ->
  Result<Measured<bool>>` samples 1000 base rows, or every row when
  there are fewer, at a fixed stride across the whole base. It decides
  unit length with `|‖v‖ − 1| ≤ 1e-3` for every sampled row (f32
  widened by `facet_as::<f32>`). It returns the value, the sample size
  and the worst deviation. A declared `is_normalized` that the sample
  contradicts is a failed V8 in `verify()`. An overlay value it
  contradicts is an OV-5 warning.
- **IN-5. Recorded once.** A measurement is recorded against the
  base facet's content identity (IN-6), so it is never re-taken for
  the same bytes and never applied to different ones. It is exposed
  through `attribute("is_normalized")` with `AttributeSource::Measured`
  when nothing higher in OV-2 sets it. It is stored in
  `<cache_root>/<dataset>/measured.yaml` (D2). That file is
  machine-written and keyed by content id:
  `{<content id>: {is_normalized: {value, sample, worst_deviation,
  at}}}`. drop-cache and cache-gc remove it with the dataset. A local
  dataset has a cache directory by its name as a remote one does.

### 5.3 Content identity

- **IN-6.** `FacetStorage::content_id(&self) -> Option<ContentId>`.
  `ContentId { root: [u8; 32], kind: Published | Recorded | Computed
  }` is always a merkle root at the 8 MiB chunk size, so the three
  kinds compare equal for equal bytes.
  - `Published`: the root of the publisher's `.mref`, verified chunk
    by chunk into `.mrkl`, as today.
  - `Recorded`: the root of the tree `ChunkStore` recorded on fetch
    (IN-7).
  - `Computed`: a local file's root, `MerkleRef::from_path_with_progress`
    on first request, with progress, and remembered for the process.
  - `None` while a facet is partly fetched.
- **IN-7. Fetched without a reference: a recorded tree in `.chunks`
  (D1).** `ChunkStore` keeps its state as a `MerkleState` in the
  `.mrkl` format (hashes, validity bitset, footer) under the
  `<file>.chunks` name. The name keeps saying what it is: trusted by
  TLS, with hashes recorded on fetch rather than published. `.mrkl`
  keeps meaning verified against a publisher's `.mref`, so
  `open_url_complete` and `revalidate` are unchanged for it.
  - Each chunk's leaf hash is computed as the chunk lands, in the
    worker that fetched it; leaves are independent, so order does not
    matter. The interior nodes and root are computed when the last
    leaf is valid, before promotion to mmap.
  - A later read of a cached chunk can be re-verified against its
    recorded leaf, which catches local corruption. Re-verification
    runs on `verify()`; ordinary reads do not re-hash.
  - **Legacy `.chunks`** (a bare one-byte-per-chunk bitmap) is
    recognised by its last byte: the footer's length marker is 45, and
    a bitmap byte is 0 or 1. A partial one is read as before and
    rewritten in the new format with the chunks it holds hashed from
    the cache file. A complete one is hashed once on first
    `content_id`, with progress, and rewritten.
  - `AccessMode::MerkleChunked` keeps its name and its "TLS only"
    meaning. sysref 01 documents the `.chunks` content.

## 6. Library surface (LS)

- **LS-1 (A2).** `config catalog add <path>` stores the path
  absolutized (`std::path::absolute`, without resolving symlinks), and
  names it after the directory's last segment, so `add .` in
  `/data/stage` saves `stage: /data/stage`. `config catalog remove`
  matches a path argument by its absolute form as well as by its
  spelling, so `remove .` and `remove /data/stage` both find it.
  Existing relative entries are left as written, because config is
  never rewritten.
- **LS-2 (C1).** `vectordata::fetch::fetch_if_modified(url, path) ->
  io::Result<bool>` and `vectordata::fetch::fetch_text(url) ->
  io::Result<String>`. Both use the shared client, read auth and the
  update-if-newer rule. `transport` stays crate-private, and the
  `fetch` module is where a program looks.
- **LS-3 (C2).** A typed xvec writer is added in veks-io as
  `veks_io::xvec::XvecWriter<T: Element>::create(path, dim) ->
  io::Result<Self>`, with `write(&[T])` (wrong length is an error),
  `write_all(rows)` and `finish() -> io::Result<u64>`. It refuses an
  extension whose element type is not `T`, and it propagates every
  I/O error. `ScalarWriter<T>` and `VvecWriter<T>` follow the same
  shape. vectordata re-exports them from `vectordata::io` (D3), and its
  private writers in `derive` and `shard_writer` move onto them. The
  byte-level `xvec::writer::open` stays for the pipeline, but stops
  discarding errors: they are kept and returned by `finish()`.
- **LS-4 (C3).** `LogMeter::quiet_when_resident(self, bool) -> Self`.
  When set, a plan whose files are all resident emits nothing, neither
  `PlanBegin` nor `PlanEnd`. A plan with anything to fetch reports as
  today. `TextMeter` gets the same option.
- **LS-5 (C4).** `shell::completion_node() -> veks_completion::Node`
  returns the CLI's completion tree, so a host grafts it under its own
  subcommand and answers its own completion protocol. For a host that
  does not use veks-completion, there is `shell::complete(words:
  &[&str]) -> Vec<String>`, which takes the words after the embedding
  prefix. `bin_main_with_catalogs`'s doc says so, and `completions`
  under an embedded name prints how to wire it rather than a script
  bound to `vectordata`.

### 6.1 veks-io is the format layer (IO)

vectordata's `io.rs` dates from the first commit. veks-io was
separated from veks's local readers on 2026-04-10, and the two were
never merged. Each reads xvec, vvec and scalar files. vectordata's
readers also run over its storage (remote, merkle-cached, chunked,
series), which veks-io cannot. `srd-element-access.md` §8 gave the two
the same rules but kept two implementations. This section makes them
one (D3).

- **IO-1. Byte sources.** veks-io defines `trait ByteSource: Send +
  Sync { fn len(&self) -> u64; fn read_at(&self, offset: u64, out:
  &mut [u8]) -> io::Result<()>; fn mapped(&self) -> Option<&[u8]>;
  fn advise(&self, range, Advice) {} }`. A memory-mapped local file
  implements it. vectordata's `Storage` implements it in vectordata,
  whether mmap, cached, chunked or series.
- **IO-2. One reader per format, generic over the source.** In
  veks-io: `XvecReader<T, S: ByteSource>`, `IndexedVvecReader<T, S>`
  (with its offset index and `IDXFOR__` sidecar logic),
  `ScalarReader<T, S>`, `ConvertingReader`, `AnyVectorReader`, the
  `VectorReader` and `VvecReader` traits, `IoError`, `require_native`,
  `copy_records`, the `madvise` family (`prefetch_range`,
  `prefetch_pages`, `advise_sequential`/`random`, `release`),
  `StreamReclaim`, and `open_vec`/`open_vec_as`/`open_vec_native` for
  a local path.
  - `MmapReader` and `ScalarMmapReader` become the readers over a
    mapped local file. They keep their names as aliases, deprecated
    since the release that carries this.
  - Header and stride validation is written once: the dim range,
    checked arithmetic, the remainder check and the extension's
    element type.
- **IO-3. vectordata keeps only the storage half.** `vectordata::io`
  becomes re-exports of veks-io's readers, writers and traits, plus
  the `Storage` `ByteSource` impl and the `open_*` functions that take
  a URL or catalog source. Every public path in `vectordata::io` stays
  (a re-export), so no caller breaks. The view's series and window
  readers (`ShardedReader`, `WindowedVectorReader`) wrap veks-io
  readers. The 74 files that name `vectordata::io` keep compiling
  unchanged.
- **IO-4. Speed is settled by measurement, and the faster path wins.**
  Before the move, a criterion bench in veks-io (`benches/readers.rs`)
  runs both implementations on the same local files:
  - sequential `read_into` over 1 GiB of f32, u8 and i32;
  - random `get` and `get_slice` of 1M records;
  - an indexed vvec scan;
  - a scalar column `as_slice` sum;
  - each pinned to one core with `taskset`, as the 4.0 parity runs
    were.

  For each case the faster implementation's technique (prefetch
  windows, bulk `copy_records`, page release during a scan) is what
  veks-io keeps. After the move, the same bench shows no case slower
  than the better of the two before, within noise. The results table
  goes into §11. A hot-path guard (a deterministic bytes-read bound,
  per `feedback_hot_path_guard_tests`) is added for every technique
  carried over, and shown to fail without it.
- **IO-5. The pipeline.** `veks-pipeline`'s `dispatch_reader!` and its
  `XvecReader<T>` opens resolve to the same veks-io type through
  vectordata's re-export. No pipeline reader changes beyond the import
  path.

## 7. Non-goals

- Inferring windows, sized profiles, layers or predicates from file
  names (LF-10 opens whole-file facets).
- Inference over HTTP (LF-1).
- Measuring anything but unit length (IN-4). Others can follow the
  same `Measured` shape.
- Hashing chunks as they arrive (IN-7).
- Refusing on an overlay contradiction at open (OV-5 warns, and
  `verify()` refuses).
- What a consumer does with its namespaced keys.

## 8. Acceptance tests

Every guard is shown failing on the code before its step.

- **LF.** One test per worked case in §3.2, through
  `TestDataGroup::load`, `open_spec` and the `describe` binary.
  `datasets init` writes a yaml that a second open reads as declared,
  and refuses to overwrite without `--force`. A URL head without a
  descriptor is still refused.
- **OV.** For each of these, `describe` and `view.distance()` give the
  same answer:
  - a user overlay sets `distance_function` on a knn_entries-catalog
    dataset;
  - a catalog-qualified block beats a bare one;
  - a program overlay is beaten by the user overlay;
  - two program overlays setting one key are refused;
  - a namespaced key reads back through `attribute`;
  - an unknown un-namespaced key warns;
  - a `base_count` overlay contradicting the facet warns once (via a
    captured `log` sink) and stays in force;
  - `config overlay set` keeps the file's comments.
- **IN.** For each check V1–V8, a fixture that fails it and the error
  naming it. `verify()` passes on a good fixture. A capturing sink
  sees V5's progress. A measured `is_normalized` is recorded and not
  re-sampled (a probe counts base reads). For a `ChunkStore` facet
  (local HTTP server fixture), `content_id` is `None` while partial.
  When complete, it is `Recorded` and equals
  `MerkleRef::from_content`'s root for the same bytes. A flipped byte
  in a cached chunk fails `verify()`. A legacy bitmap `.chunks`
  (partial and complete) is upgraded in place.
- **IO.** The whole vectordata and veks-io suites pass unchanged
  through the re-exports. veks-io's readers over a `Storage` source
  read a remote fixture. The IO-4 bench table shows no regression, and
  a bytes-read guard exists per carried-over technique.
- **LS.**
  - `catalog add .` saves `<dirname>: /abs`, and `remove .` finds it.
  - `fetch_text` against the local server fixture.
  - `XvecWriter::<f32>` round trip, `XvecWriter::<f32>::create("x.ivecs")`
    refused, and a full disk (an injected writer) surfacing from
    `finish`.
  - A resident plan is silent under `quiet_when_resident`, and a
    remote one reports.
  - `complete(["datasets", "desc"])` gives `describe`.

## 9. Settled decisions (2026-10-21)

- **D0. Rules.** All of I1–I12 as LF-1–LF-12, not a subset.
- **D1. Content identity of a facet fetched without `.mref`.** Use the
  merkle machinery already in place, not a separate digest. A recorded
  tree in the `.mrkl` format is kept under the `.chunks` name (IN-7).
  A recorded tree cannot use the `.mrkl` name, because the open and
  revalidate paths take `.mrkl` to mean that a published `.mref`
  exists, and the fixed footer has no room for a flag.
- **D2. Measurements** go in `<cache_root>/<dataset>/measured.yaml`,
  keyed by content id (IN-5).
- **D3. vectordata depends on veks-io**, and veks-io is the only
  implementation of the formats. vectordata's own readers move into
  it, carrying over whichever implementation is faster (§6.1).
- **D4. The user overlay** is one `<config_dir>/overlays.yaml` with
  `datasets:` and `catalogs:` sections (OV-1).
- **D5. Overlay contradictions warn conservatively** (OV-5): a
  warning whenever a probe disagrees, and the overlay stays in force.

## 10. Plan

Each step is one themed commit when the maintainer asks.

1. LS-1 (`catalog add .`), the defect.
2. IO-4's bench, run against both implementations as they stand, with
   results recorded.
3. IO-1–IO-3, IO-5: veks-io becomes the format layer, and vectordata
   depends on it and re-exports it. The bench is re-run and the guards
   are added.
4. LS-3: typed writers in veks-io. vectordata's private writers move
   onto them.
5. §5.1 verification (IN-1–IN-3). `veks check`'s ground-truth checks
   move onto it.
6. Loose files LF-1–LF-12 and `datasets init`, using step 5's checks.
7. Overlays OV-1–OV-6.
8. Content identity IN-6–IN-7: the recorded tree in `.chunks`.
9. Measured normalization IN-4–IN-5.
10. LS-2, LS-4, LS-5.
11. Docs: sysref 01 (the LF table, the overlay format), sysref 02 (the
   APIs), `AGENTS.md` (`datasets init`, `config overlay`), and the
   CHANGELOG under Unreleased. The SRD status is updated with records.
   Fastervector is told what shipped once the maintainer says so.
