# SRD — Loose files, attribute overlays and dataset integrity

**Status:** implemented 2026-10-21 (records in §11; steps 1–10 of
§10, step 11 being these records and the docs). The decisions in §9
were taken on 2026-10-21; where implementation settled something
differently, §11 says so (OV-3's `load`, IN-5's local base, IN-7's
legacy sidecar, the LF-5 format case). The release that
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
| **LF-5 (I3)** | **Role token.** For a file LF-4 left unclassified, split the stem on `_` and `-`. Exactly one token, or one adjacent token pair (for `ground_truth`), must be a role alias. The aliases are `base`, `train` → base; `query`, `queries`, `test` → query; `gt`, `groundtruth`, `ground`+`truth`, `neighbors`, `indices` → neighbor_indices; `distances` → neighbor_distances. The vocabulary is `StandardFacet::from_alias`, restricted to these four facets. The format must be one that facet accepts in the facet spec, so `*_gt_*.fvecs` is refused rather than reinterpreted (indices are integers), while `*_query_*.ivecs` is queries (the spec takes integer, quantized queries). Two role tokens in one stem leave the file unclassified, and it is listed in the notes. |
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
| `x_gt.fvecs` (with base and queries) | LF-5 refuses the format (indices are integers), the note says why, and LF-6 finds no integer file for the truth |
| `codes_query.ivecs` | queries: the facet spec takes integer (quantized) queries, so a role token in an integer vector file is not a refusal |

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
  name (`TestDataGroup::name()`). `TestDataGroup::load` reads no
  configuration (no catalogs, no overlay), so a test or a program that
  opens a path in code is not changed by a file in someone's home; it
  applies overlays with `group.with_overlays(&overlays)`.
  `TestDataGroup::attribute`, `view.distance()`,
  `view.distance_function()` and `view.dataset_attribute(key)` read the
  merged values. (Settled in implementation, 2026-10-21: the SRD first
  had `load` read the user overlay.)
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
  shape. vectordata re-exports them from `vectordata::io` (D3). The
  byte-level `xvec::writer::open` stays for the pipeline, but stops
  discarding errors: they are kept and returned by `finish()`.
  vectordata's `derive` and `shard_writer` are not encoders — one
  copies byte ranges of an existing file, the other writes the
  record bytes it is given and owns shard rollover and the atomic
  rename — so they stay copies; the one piece of format knowledge in
  them, a uniform record's size from its first header, is veks-io's
  `reader::xvec_record_bytes`, which the reader uses too.
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

### 6.2 IO-4 results before the move (2026-10-21)

Run as `taskset -c 2 cargo bench -p vectordata --bench readers` on an
Intel Xeon Platinum 8375C, warm page cache, 1 GiB uniform files (f32
dim 128, u8 dim 128, i32 dim 100), a 1M-record ragged i32 file (dims
1–64), and 1 GiB columns (4 GiB for u8). Median of criterion's
estimate; the faster column is the technique veks-io keeps.

| Case | vectordata | veks-io | Faster, and why |
|------|-----------:|--------:|-----------------|
| seq `read_into` f32 | 8.49 GiB/s | 8.46 GiB/s | tie: both one `memcpy` per record |
| seq `read_into` u8 | 10.30 GiB/s | 10.23 GiB/s | tie |
| seq `read_into` i32 | 8.44 GiB/s | 8.42 GiB/s | tie |
| seq streaming `next_record` f32 / u8 / i32 | — | 2.56 / 1.36 / 2.81 GiB/s | veks-io's streaming reader allocates a `Vec` per record through a `BufReader`: 3–7× slower than either mapped reader; it reads its records into a reused buffer instead (IO-6) |
| random `get` f32, 1M | 3.31 M/s | **7.79 M/s** | veks-io, 2.4×. Both decode element by element (`chunks_exact().map(from_le_bytes).collect()`); `perf` puts 87% of vectordata's time in that `from_iter` as an out-of-line function (~3.7× the instructions per get), while veks-io's inherent `get` inlined it into the caller. The moved `get` is one bulk copy into the new `Vec`, so it no longer depends on inlining |
| random `get_slice` f32, 1M | 53.0 M/s | **69.3 M/s** | veks-io, 1.3×: indexes its mapping directly; vectordata matches the `Storage` enum on every call. Hence readers generic over a monomorphized source (IO-2), never a `dyn` source on the hot path |
| vvec scan i32 | 1.57 GiB/s | 1.50 GiB/s (stream) | tie within 5% |
| vvec random `get` i32, 1M | 4.55 M/s | — | baseline: veks-io has no indexed vvec reader yet |
| scalar `as_slice` sum i32 / u8 | 3.15 GiB/s / 831 MiB/s | 3.15 GiB/s / 831 MiB/s | tie: the same slice; the loop bounds it |
| scalar `as_slice` sum f32 | 2.76 GiB/s | 2.52 GiB/s | the same slice and loop; codegen of the two monomorphized loops, rechecked after the move |
| scalar `read_into` i32 / f32 / u8 | 9.08 / 9.06 / 12.69 GiB/s | 9.07 / 9.12 / 12.71 GiB/s | tie |

The raw log is reproduced by the command above; the bench is
`crates/vectordata/benches/readers.rs`.

### 6.3 IO-4 results after the move (2026-10-21)

The same command and files. `vectordata` is now veks-io's reader over
`StorageSource`, `veks-io` the same reader over `MappedFile`. The
"best before" column is the faster of the two in §6.2.

| Case | vectordata | veks-io | Best before | |
|------|-----------:|--------:|------------:|--|
| seq `read_into` f32 / u8 / i32 | 8.48 / 10.36 / 8.46 GiB/s | 8.39 / 10.32 / 8.45 GiB/s | 8.49 / 10.30 / 8.44 | tie |
| seq streaming `next_record` f32 / u8 / i32 | — | 5.00 / 3.39 / 5.30 GiB/s | 2.56 / 1.36 / 2.81 | 1.9–2.5× (IO-6: read from the mapping) |
| seq streaming `read_record_into` f32 / u8 / i32 | — | 6.05 / 5.49 / 6.60 GiB/s | — | new: one buffer per scan |
| random `get` f32 | 9.09 M/s | 9.01 M/s | 7.79 | 1.17× (bulk copy from the kept base) |
| random `get_slice` f32 | 70.4 M/s | 70.6 M/s | 69.3 | tie, at veks-io's speed over either source |
| vvec scan i32 | 3.88 GiB/s | 3.51 GiB/s (stream) | 1.57 | 2.5× (bulk copy of the borrowed record) |
| vvec random `get` i32 | 7.56 M/s | — | 4.55 | 1.66× |
| scalar `as_slice` sum i32 / u8 | 3.15 GiB/s / 831 MiB/s | 3.15 GiB/s / 830 MiB/s | 3.15 / 831 | tie |
| scalar `as_slice` sum f32 | 2.14 GiB/s | 2.14 GiB/s | 2.76 | not the reader: see below |
| scalar `read_into` i32 / f32 / u8 | 9.09 / 9.12 / 12.56 GiB/s | 9.09 / 9.12 / 12.57 GiB/s | 9.08 / 9.12 / 12.71 | tie |

The random `get` row is from a re-run of that case after `get` was
changed to read from the kept base; the first after-run measured
6.45 M/s, below the best before, and that was fixed rather than
recorded.

The f32 column sum is the bench's own loop. `as_slice` returns the
same borrowed slice in O(1) either way, and `perf` puts all the time
in the bench closure: 57% in the per-element `widen` call through a
function pointer, the rest in the loop. That number changes with how
the bench binary is compiled and not with the reader: 2.76 and 2.52
for the two readers before, 2.14 for both in the after-run, and 2.55
for both in a later rebuild with no reader change. It is reported
as measured, not as a regression or a gain.

- **IO-6.** veks-io's streaming readers (`open`, `open_varlen`) stop
  allocating per record: `next_record` keeps its trait shape, and a
  `read_record_into(&mut Vec<u8>) -> bool` reuses the caller's
  buffer. Mapped files are read through the mapped reader rather than
  a `BufReader` copy. The bench's streaming cases measure both.

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

## 11. Implementation

### 11.1 Step 1: `catalog add .` (2026-10-21)

- `config::saved_location(source, cwd)`: a URL as written; a path made
  absolute against the working directory with `.` and `..` removed
  lexically (symlinks kept). The name follows from it.
- `add_catalog_ex`, `try_add_catalog` and `force_add_catalog` save
  that form; `remove` matches the spelling given, then the saved form.
- Guard: `tests/cli_config_catalog.rs`, failing on the old code with
  `0: .`.

### 11.2 Step 2: the reader bench (2026-10-21)

- `crates/vectordata/benches/readers.rs`, results in §6.2.

### 11.3 Step 3: veks-io is the format layer (2026-10-21)

- veks-io `source`: `ByteSource` (with `SourcePart`), `MappedFile`,
  and `source::mapping`, the one implementation of `madvise` use;
  vectordata's `Storage` advises through it.
- veks-io `reader`: `IoError`, `VectorReader`, `VvecReader`,
  `XvecReader<T, S = MappedFile>`, `IndexedVvecReader<T, S>` with the
  `IDXFOR__` index (`load_offsets`, one path for local and remote:
  sidecar beside the local file or copy, published sidecar via
  `ByteSource::fetch_sibling`, walk when complete, refusal otherwise),
  `ScalarReader<T, S>`, `StreamReclaim`, `ConvertingReader`,
  `AnyVectorReader`/`AnyVvecReader`/`AnyElement`, the extension
  helpers, and `open_vec*`/`open_vvec*` with `_with::<S>` forms.
- vectordata `io`: re-exports, `StorageSource` (the `ByteSource` over
  `Storage`), the aliases `XvecReader<T>` = `veks_io::XvecReader<T,
  StorageSource>` and likewise, the `*_from_storage` constructors the
  view uses, and the open functions over storage. Its format code is
  gone; vectordata depends on veks-io.
- `IoError::Http(reqwest::Error)` is `IoError::Transport(Box<dyn
  Error + Send + Sync>)`: veks-io does not depend on an HTTP client.
- Semantics settled where the two differed: an empty file is zero
  records (vectordata's rule; veks-io refused it); the dimension check
  uses checked arithmetic and says "dimension N is not positive"
  (veks-io's); a uniform reader refuses a variable-length extension
  (veks-io's; vectordata opened one whose records happened to be one
  stride).
- veks-io `MmapReader<T>`, `ScalarMmapReader<T>` and `MmapError` are
  deprecated aliases of the new types; the per-type constructors stay
  deprecated on them.
- The techniques kept for speed: `get` copies a record in bulk
  (`decode_vec`), and a uniform reader keeps its source's mapped base
  after the first answer, for `get` and both `get_slice` forms. Guards in `veks-io/tests/reader_hot_paths.rs`:
  with each technique switched off, 100 records of 64 elements decoded
  6400 elements one by one, and 3000 reads asked the source 2000
  times (1001 with only `get` reverted); both pass with them.
- The bench after the move is §6.3.
- IO-6: the streaming readers (`open`, `open_varlen`) read from a
  mapping, release the pages behind the cursor every 256 MiB, and
  `VecSource::read_record_into` reuses the caller's buffer.
- Tests moved with the code: the sidecar layouts and the atomic index
  write (to `reader/vvec.rs`), the scalar extension table (to
  `reader/mod.rs`); veks-io's element-model guards ported to the new
  types; the doc-example coverage lists in both crates follow the
  items.

### 11.4 Step 4: typed writers (2026-10-21)

- veks-io `writer`: `XvecWriter<T>` (`create(path, dim)`, `write`,
  `write_all`, `write_packed`, `finish`), `ScalarWriter<T>`,
  `VvecWriter<T>` (writes its `IDXFOR__` index on `finish`, and removes
  a stale one on `create`). Errors are `IoError`, the readers' type, so
  a refused extension is the same `ElementMismatch`. Re-exported at the
  veks-io root and from `vectordata::io`.
- The byte-level xvec, scalar and varlen sinks keep their first write
  error and return it from `finish`.
- `reader::xvec_record_bytes`, used by `XvecReader::from_source` and
  `derive`'s window copy (which accepted a negative dimension as a
  huge one).
- Guards in `veks-io/tests/writers.rs`: every element type through the
  three writers and back; the refusals; a rewritten vvec read by its
  own index; a full disk (`/dev/full`) an error from every writer —
  failing on the old byte writer with "reported success".

### 11.5 Step 5: verification (2026-10-21)

- `vectordata::verify`: `Check` (V1–V7), `VerifyFailure`,
  `VerifyReport`, `VerifyOptions { skip_absent }`, `VerifyProgress`
  (any `FnMut(&str, u64, u64)`), `NoProgress`.
  `TestDataViewExt::verify` (erring with `Error::Verification`) and
  `verify_report` (listing); `DatasetSelection::verify`, failures
  prefixed by profile.
- V8 (normalization) waits for step 9; V7 checks the dataset's own
  `base_count` until overlays (step 7) add theirs.
- `veks check`'s EA-33 is `verify_report` per non-layer profile, with
  `skip_absent` and files named relative to the dataset; its messages
  and progress label are unchanged.
- Guards: `vectordata/tests/verify.rs` (each check failing, a passing
  profile, the sentinel rule, skip-absent, progress from 0 to every
  row, a selection); `veks check` reporting queries of another
  dimension, failing with the old EA-33 body.

### 11.6 Step 6: loose files (2026-10-21)

- `dataset::infer`: `infer_dir(&Path) -> Result<Inferred, InferError>`
  (LF-3 to LF-12), `Inferred { yaml, config, classified, notes }` with
  `summary()`, `Classified { profile, facet, source, rule }`, `Rule`,
  `InferError` (each variant naming its candidates),
  `has_loose_files`.
- The descriptor is built as `dataset.yaml` text and parsed back, so
  the in-memory one and the one `datasets init` writes are the same;
  a series carries `shard_stride`, `shard_count` and `record_count`,
  and `format_version: 2`.
- `TestDataGroup::load_from_path`: `dataset.yaml`, then
  `knn_entries.yaml`, then — when the directory has candidates — the
  inference, logged once per process per directory with its notes;
  an empty directory is still `ConfigIo` (no descriptor).
  `TestDataGroup::inferred()` and `datasets describe` (`Type: inferred
  from its files`, `Read as: …`).
- `datasets init <dir> [--force]` (`datasets::init::init`): refuses an
  existing `dataset.yaml` without `--force`, and a `knn_entries.yaml`.
- LF-12's normalization claim uses `verify::measure_normalization`
  (IN-4's sample, here unrecorded; step 9 records it).
- Settled against the facet spec: integer queries are queries, so a
  `_query_` token in an `.ivecs` file classifies it; the example of a
  refused format is a `_gt_` `.fvecs`.
- Guards: `tests/loose_files.rs`, one per §3.2 case, a series and its
  gap, the normalization claim, `init`, and `datasets describe <dir>`
  through the binary, which on the old binary fails with "Failed to
  read dataset configuration: No such file or directory".

### 11.7 Step 7: overlays (2026-10-21)

- `vectordata::overlay`: `Overlay` (`parse`, `program`, `load_user`,
  `values_for`, `warnings`), `Overlays` (`add_program` refusing a key
  two programs set, `set_user`, `resolve`), `Stated`,
  `AttributeSource` (`Dataset`, `Inferred`, `Overlay { file, kind }`),
  `OverlayKind`, `Claims`, `KNOWN_KEYS`, `is_settable`, and the line
  editor `set_text` / `unset_text`.
- `CatalogSources`: `overlays` beside the catalogs; `configure*` reads
  `<dir>/overlays.yaml` as the user's; `with_overlay`, `overlays`,
  `with_overlays_from` (what `--at` keeps); `extend`, `named` and
  `reload_configured` carry them.
- `Catalog::of` applies them to every entry; `Catalog::overlays`.
  `CatalogEntry`: `attributes()` merged, `own_attributes()`,
  `attribute(key)`, `attribute_source(key)`, `attributes_in(ns)`,
  `overlaid()`. The listing written to a catalog file is the
  catalog's own; an overlay is never written into one.
- `TestDataGroup`: `with_overlays`, `attribute_source`; an entry
  opened from a catalog carries its overlay into the group's
  attributes. `open_selection` applies the catalog's overlays to a
  path it opens.
- OV-5: `GenericTestDataView::probe_overlay` at `open_any_native` (the
  one place every typed open passes), for the whole base only;
  `CatalogEntry::apply_overlay` for a distance the dataset declares
  otherwise; `warn_contradiction` once per (dataset, key). `verify`
  V7 compares the stated `base_count`, `dimension` and `maxk`
  (`TestDataView::dataset_attribute`). `describe` marks a value its
  listing contradicts.
- `vectordata config overlay set | unset | show`
  (`config::overlay_set` / `overlay_unset` / `overlay_show`, each
  taking the configuration directory).
- Guards: `overlay.rs` unit tests (the editor keeps comments, the
  precedence, the program conflict, the unknown key);
  `tests/overlays.rs` (a knn_entries dataset's metric through the
  entry and the view; precedence across overlays and catalogs;
  namespaced keys; a contradicted count warned once, in force, V7; a
  contradicted metric warned; `config overlay` and `describe` through
  the binary, with and without `--at`). The old binary describes the
  same knn dataset with "no attributes recorded".

### 11.8 Step 8: content identity (2026-10-21)

- `merkle::ContentId { root, kind }`, `ContentIdKind` (`Published`,
  `Recorded`, `Computed`); `FacetStorage::content_id`,
  `content_id_with_progress`, `verify_cached`; `Storage::content_id`
  and `verify_cached`; `CachedChannel::verify_cached`. A series has no
  content identity (`None`): it is no one tree.
- `MerkleRef::from_leaves`: the one tree builder, used by
  `from_content`, `from_path_with_progress` and the recorded tree.
- `ChunkStore`: leaves recorded as chunks land (and hashed back from
  the cache file after a whole-file transfer); `.chunks` in the
  `.mrkl` format; `persist_leaf` writes one leaf and merges the bitset
  per chunk (`MerkleState::checkpoint`), the whole tree only when the
  sidecar is created and on completion; `content_root` hashes the
  leaves a legacy sidecar lacks, once, with progress; `verify_recorded`.
- `read_sidecar` reads both formats (the footer's length marker, 45,
  against a bitmap's 0/1 bytes), and is what `complete_size_on_disk`
  and the explorer's cache survey use. A sidecar whose chunk count is
  not the file's is not read, as `ChunkStore::open` never read one;
  the picker's fixture that had a two-chunk bitmap for a one-chunk file
  was corrected.
- Changed from §5.3 as written: a legacy sidecar is not upgraded at
  open (a TB-scale base would be hashed inside an open); its leaves are
  hashed on the first `content_id`.
- V9 (`VerifyOptions::cached_chunks`, opt-in: it reads every cached
  byte).
- Guards: `a_recorded_chunk_writes_its_leaf_and_bits_not_the_tree`
  (200 chunks of a 512 GiB file's 4 MiB tree: 840 MB written with a
  whole-tree rewrite per chunk, within the 1.7 MB limit without);
  `a_legacy_complete_file_is_hashed_once_for_its_content_root`;
  `a_legacy_bitmap_sidecar_reads_and_rewrites_in_the_tree_format`;
  `tests/content_id.rs` (a served file without a `.mref`: no root while
  partial, then the root a `.mref` of the same bytes has, equal to the
  local copy's computed root; V9 passing, then naming the chunk a
  flipped byte is in). The content-identity API is new, so it has no
  old-code form to fail; the old store's sidecar held no hashes.

### 11.9 Step 9: measured normalization (2026-10-21)

- `verify::measure_normalization` (step 6), `recorded_normalization`,
  `normalization`, `Normalization { value, source, measured,
  contradicted }`, `MEASURED_FILE`; `TestDataViewExt::normalization`;
  `AttributeSource::Measured { file }`; V8 in `verify` when the base is
  resident, so verification never fetches a sample.
- `TestDataView::dataset_attribute_source` and `cache_name` (provided;
  the generic view carries the group's attribute sources); 
  `FacetStorage::known_content_id` and `Storage::known_content_id`,
  the identity the cache knows without reading the file.
- Settled in implementation against D2: a measurement is recorded by
  the identity the cache already knows (a published root, or a
  recorded one once a fetch completes). A local base is not recorded
  — its identity is a read of the whole file, a TB-scale cost to key a
  1000-row sample, and the storage layer deliberately creates no cache
  directory for a local dataset — and is measured each time, which
  costs 1000 local reads. A partly fetched base is measured and not
  recorded.
- Guards: `tests/normalization.rs` (a contradicted dataset declaration
  is V8; an overlay's warned once; a local base measured with nothing
  written; a remote base's measurement recorded by content id and
  read back after its cached bytes are corrupted — failing with the
  record lookup off; an inferred claim sourced `Inferred`).

### 11.10 Step 10: library surface (2026-10-21)

- LS-2: `fetch::fetch_text` and `fetch::fetch_if_modified` (offline:
  an error); the request lives in `transport`, the one network entry
  point, which the discoverability guard holds; both budgeted in
  `FETCH_BUDGET`.
- LS-4: `LogMeter::quiet_when_resident` (the `planning` line deferred
  to the plan's result, nothing for a resident plan or a run that
  fetches nothing, the stall watchdog unchanged) and
  `TextMeter::quiet_when_resident` (the planning ticker still moves; a
  resident plan's line is erased, `Ticker::dismiss`).
- LS-5: `shell::completion_node`, `shell::complete` (the completer's
  first word is the command's own name, which `complete` supplies),
  and `completions` under an embedding name explaining how and exiting
  1.
- Guards: the two quiet-meter unit tests (a resident plan silent, one
  with bytes to fetch reported with its planning line, a silent run;
  the terminal meter's line erased); `tests/fetch_text.rs` against the
  test server; `a_host_completes_the_embedded_cli_from_its_own_tree`
  grafting the node under a host tree.
