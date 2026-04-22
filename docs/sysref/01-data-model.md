# 1. Data Model

---

## 1.1 File Structures

Three record structures, distinguished by file extension:

| Structure | Extension pattern | Header | Random access |
|-----------|------------------|--------|---------------|
| **Scalar** | `.<type>` (`.u8`, `.i32`, `.f64`) | None | `offset = ordinal × elem_size` |
| **Uniform vector** | `.<type>vec` (`.fvec`, `.ivec`) | 4-byte dim per record | `offset = ordinal × stride` |
| **Variable vector** | `.<type>vvec` (`.ivvec`, `.fvvec`) | 4-byte dim per record | Requires offset index |

### Scalar

Flat-packed array. No header. Element N at byte offset `N × sizeof(T)`.

```
[ elem₀ | elem₁ | elem₂ | ... | elemₙ₋₁ ]
```

Record count = `file_size / elem_size`.

### Uniform vector (vec)

Each record: 4-byte LE `i32` dimension header + `dim` elements.
All records share the same dimension.

```
record = [ dim:i32 | elem₀ | elem₁ | ... | elem_{dim-1} ]
file   = record₀ record₁ ... recordₙ₋₁
```

Stride = `4 + dim × elem_size`. Record count = `file_size / stride`.

### Variable-length vector (vvec)

Same per-record layout as vec, but each record may have a different
dimension. No fixed stride.

```
record₀ = [ dim₀:i32 | ... ]   (dim₀ elements)
record₁ = [ dim₁:i32 | ... ]   (dim₁ elements)
...
```

Random access requires a companion offset index file:

```
IDXFOR__<filename>.<i32|i64>
```

The index is a scalar file of byte offsets — one per record.
`.i32` for data files up to 2 GB, `.i64` for larger.

---

## 1.2 Element Types

| Type | Size | Scalar ext | Vec ext | Vvec ext | Legacy |
|------|------|-----------|---------|----------|--------|
| f32 | 4 B | — | `.f32vec` | `.f32vvec` | `.fvec` |
| f64 | 8 B | — | `.f64vec` | `.f64vvec` | `.dvec` |
| f16 | 2 B | — | `.f16vec` | `.f16vvec` | `.mvec` |
| u8 | 1 B | `.u8` | `.u8vec` | `.u8vvec` | `.bvec` |
| i8 | 1 B | `.i8` | `.i8vec` | `.i8vvec` | — |
| u16 | 2 B | `.u16` | `.u16vec` | `.u16vvec` | — |
| i16 | 2 B | `.i16` | `.i16vec` | `.i16vvec` | `.svec` |
| u32 | 4 B | `.u32` | `.u32vec` | `.u32vvec` | — |
| i32 | 4 B | `.i32` | `.i32vec` | `.i32vvec` | `.ivec` |
| u64 | 8 B | `.u64` | `.u64vec` | `.u64vvec` | — |
| i64 | 8 B | `.i64` | `.i64vec` | `.i64vvec` | — |

All legacy extensions are fully supported as aliases. Plural forms
(`.fvecs`, `.ivvecs`) are accepted everywhere.

---

## 1.3 Container Formats

| Format | Extension | Use |
|--------|-----------|-----|
| NumPy | `.npy` | Import source for vector arrays |
| Parquet | `.parquet` | Import source for structured metadata |
| Slab | `.slab` | Variable-length binary records (metadata, predicates) |
| HDF5 | `.hdf5`, `.h5` | Import source (pre-convert to xvec; see note below) |

### HDF5 note

HDF5 is supported as an import source but the `libhdf5` C library
dependency was removed from the default build. HDF5 introduced
significant build complexity (cmake, C compiler, system library
version conflicts) and runtime linking issues across platforms.

To work with HDF5 datasets, pre-convert them to xvec format using
Python or any HDF5-capable tool:

```python
import h5py, numpy as np
with h5py.File('dataset.hdf5', 'r') as f:
    vecs = np.array(f['train'], dtype=np.float32)
    with open('base_vectors.fvec', 'wb') as out:
        for v in vecs:
            out.write(np.int32(len(v)).tobytes())
            out.write(v.tobytes())
```

Then use the resulting `.fvec` file as input to `veks bootstrap`.

### Slab internals

Page-aligned record container (provided by `slabtastic`). Records are
opaque byte slices addressed by ordinal. A **dialect leader byte** at
the start of each record identifies its type:

| Leader | Type | Codec |
|--------|------|-------|
| `0x01` | MNode | Metadata record |
| `0x02` | PNode | Predicate tree |
| `0x03` | Generic | Raw data |

This decouples storage from logic — `SlabReader`/`SlabWriter` handle
I/O while the ANode codec handles interpretation.

---

## 1.4 Wire Formats (MNode / PNode)

Binary codecs for structured metadata and predicate records, provided
by the `veks-anode` crate.

### MNode (metadata records)

Dialect leader: `0x01`. Encodes a named-field record with 28 type tags.

```
[0x01][field_count: u16 LE]
per field:
  [name_len: u16 LE][name: UTF-8]
  [type_tag: u8]
  [value: variable]
```

| Tag | Type | Encoding |
|-----|------|----------|
| 0 | text | u32 LE length + UTF-8 |
| 1 | int | i64 LE |
| 2 | float | f64 LE |
| 3 | bool | u8 (0/1) |
| 4 | bytes | u32 LE length + raw |
| 5 | null | (none) |
| 8 | list | u32 LE count + tagged values |
| 9 | map | u32 LE length + nested MNode (no leader) |
| 12 | int32 | i32 LE |
| 13 | short | i16 LE |
| 16 | float32 | f32 LE |
| 17 | half | u16 LE (IEEE 754 f16) |
| 26 | array | u8 elem_tag + u32 LE count + untagged values |

Full tag list: 0-28 (text, int, float, bool, bytes, null, enum_str,
enum_ord, list, map, text_validated, ascii, int32, short, decimal,
varint, float32, half, millis, nanos, date, time, datetime, uuid_v1,
uuid_v7, ulid, array, set, typed_map).

Framed variant: `[payload_len: u32 LE][payload...]` for stream embedding.

### PNode (predicate trees)

Dialect leader: `0x02`. Recursive pre-order tree encoding.

```
[0x02][node...]
```

Node types:

| Byte | Type | Encoding |
|------|------|----------|
| 0 | Predicate | field + op + comparands |
| 1 | AND | child_count: u8 + children |
| 2 | OR | child_count: u8 + children |

Predicate encoding (named mode):

```
[0x00][name_len: u16 LE][name: UTF-8][op: u8][count: i16 LE][comparands: i64 LE * n]
```

Predicate encoding (indexed mode):

```
[0x00][field_index: u8][op: u8][count: i16 LE][comparands: i64 LE * n]
```

Operators: 0=GT, 1=LT, 2=EQ, 3=NE, 4=GE, 5=LE, 6=IN, 7=MATCHES.

---

## 1.5 Dataset Facets (BQGDMPRF)

A dataset is a collection of facets — typed data files that together
describe a vector search benchmark:

| Code | Name | Structure | Typical format | Description |
|------|------|-----------|---------------|-------------|
| **B** | Base vectors | uniform vec | `.fvec` | The searchable vector collection |
| **Q** | Query vectors | uniform vec | `.fvec` | Vectors to search for |
| **G** | Ground truth indices | uniform vec | `.ivec` | Exact k-nearest neighbor ordinals per query |
| **D** | Ground truth distances | uniform vec | `.fvec` | Distances to the k-nearest neighbors |
| **M** | Metadata content | scalar | `.u8` | Per-base-vector attribute values |
| **P** | Metadata predicates | scalar | `.u8` | Per-query filter values |
| **R** | Predicate results | variable vec | `.ivvec` | Base ordinals matching each predicate |
| **F** | Filtered KNN | uniform vec | `.ivec` + `.fvec` | KNN results after predicate filtering |
| **O** | Oracle partitions | per-label profiles | directories | Per-label base vectors + partitioned KNN |

The **O** facet carries sub-facets controlling what is computed per
partition. Characters after `O` specify the per-partition scope
(default `bqg` when none specified):

```
BQGDMPRFO       → main=BQGDMPRF, per-partition=BQG
BQGDMPRFObqg    → same (explicit)
BQGDMPRFObqgd   → per-partition also gets distances
BQGDMPRFObqgmprf → full facets within each partition
```

### Facet inference

The pipeline infers which facets to produce from the inputs provided:

| Inputs | Inferred facets |
|--------|----------------|
| B only | B |
| B + Q | B Q G D |
| B + Q + GT | B Q G (D only if distances provided) |
| B + Q + M | B Q G D M P R F |
| B + Q + GT + synthesize | B Q G M P R F |
| above + `+O` | B Q G M P R F O (O must be explicit, partitions get BQG) |

---

## 1.6 Dataset Layout

### Standard layout

```
dataset-name/
├── dataset.yaml              # manifest: attributes, pipeline, profiles
├── profiles/
│   ├── base/                 # shared source data (symlinks or generated)
│   │   ├── base_vectors.fvec
│   │   ├── query_vectors.fvec
│   │   ├── metadata_content.u8
│   │   └── predicates.u8
│   └── default/              # per-profile computed artifacts
│       ├── neighbor_indices.ivec
│       ├── neighbor_distances.fvec
│       ├── metadata_indices.ivvec
│       ├── IDXFOR__metadata_indices.ivvec.i32
│       ├── filtered_neighbor_indices.ivec
│       └── filtered_neighbor_distances.fvec
├── dataset.json              # machine-readable metadata
├── variables.yaml            # pipeline-computed variables
├── catalog.json              # dataset index for catalog discovery
└── *.mref                    # merkle hash files (per data file)
```

### File naming conventions

- `_` prefix: source files excluded from publishing (e.g., `_source_base.fvecs`)
- `IDXFOR__` prefix: offset index companion files (auto-generated, published)
- `.mref`: merkle hash tree (one per data file, published)
- `.mrkl`: local merkle cache state (not published)

---

## 1.7 dataset.yaml

The manifest file that defines a dataset:

```yaml
name: my-dataset

attributes:
  distance_function: L2              # L2, COSINE, or DOT_PRODUCT
  is_zero_vector_free: true          # verified by pipeline
  is_duplicate_vector_free: true     # verified by pipeline

variables:
  base_count: '1000000'
  query_count: '10000'
  dim: '128'

upstream:
  defaults:
    seed: 42

  steps:
    - id: step-name
      run: command-path
      after: [dependency1, dependency2]
      per_profile: false
      phase: 0
      option-key: option-value

profiles:
  default:
    maxk: 100
    base_vectors: profiles/base/base_vectors.fvec
    query_vectors: profiles/base/query_vectors.fvec
    neighbor_indices: profiles/default/neighbor_indices.ivec
    metadata_content: profiles/base/metadata_content.u8
    metadata_indices: profiles/default/metadata_indices.ivvec

# Strata are generators that expand into per-size sized profiles at
# load/publish time. They live at the root of dataset.yaml — not under
# profiles: — so the strata templates and the resulting (already
# expanded) sized profiles can both be visible in the same file.
strata:
  - "decade"             # 100k, 200k, …, 900k, 1m, 2m, … (default start 100k)
  - "mul:1m..16m/2"      # 1m, 2m, 4m, 8m, 16m
  - "fib:1m"             # 1m, 2m, 3m, 5m, 8m, … capped at base_count
  - "linear:10m/10m"     # every 10M up to base_count
  - "step:1m..3m/1m"     # explicit-step arithmetic: 1m, 2m, 3m
  - "parts:0m..400m/10"  # 10 equal divisions: 40m, 80m, … 400m
```

### Profiles

Every profile declares `base_count` — the number of base vectors
accessible through that profile. This is part of the basic contract
for all dataset profiles and is used by consumers to size buffers,
estimate costs, and display summaries.

- **`default`** — the full dataset, always present. `base_count`
  set by the pipeline after vector preparation.
- **Sized profiles** — subsets (e.g., `100K`, `1M`) with an explicit
  `base_count` that windows into shared source data. They're produced
  either by listing them by hand under `profiles:` or, more commonly,
  by declaring one or more generator strings under the root-level
  `strata:` block. Strata-expanded profiles inherit per-vector facets
  (e.g. `base_vectors`) from `default` and clip them with the windowed
  source notation `path[0..N)` so the `vectordata` reader API serves
  exactly the first `N` rows.
- **Partition profiles** — per-label subsets (e.g., `label-0`) with
  `base_count` equal to the number of base vectors matching that
  label. Each has its own extracted base vectors and independently
  computed KNN in its own ordinal space.

### Strata generator strategies

Each entry under `strata:` is one generator string. Every multi-form
strategy uses an explicit prefix so the intent is unambiguous; only
the bare literal form is unprefixed.

| Strategy | Form | Effect |
|----------|------|--------|
| literal      | `<size>` (e.g. `100k`)         | Emit a single profile at `size`. |
| step range   | `step:<lo>..<hi>/<step>`       | Emit `lo, lo+step, lo+2·step, …` up to and including `hi`. |
| parts range  | `parts:<lo>..<hi>/<n>`         | Divide `[lo, hi]` into `n` equal segments and emit each segment endpoint. |
| geometric    | `mul:<lo>..<hi>/<factor>`      | Emit `lo, lo·factor, lo·factor², …`. The `..<hi>` upper bound is optional; if omitted, expansion stops at `base_count`. |
| Fibonacci    | `fib:<lo>`                     | Fibonacci progression starting at `lo`, capped at `base_count`. |
| arithmetic   | `linear:<lo>/<step>`           | Open-ended arithmetic — every `step` starting at `lo`, capped at `base_count`. |
| decade sweep | `decade` (or `decade:<lo>`)    | 100k, 200k, … 900k, 1m, 2m, … 9m, 10m, … — one detent per decimal click within each order of magnitude. Starts at 100k by default; meaningful only at million-plus scale. |

Legacy bare-range forms (`<lo>..<hi>/<step>` and `<lo>..<hi>/<n>`)
are still parsed for backward compatibility — the divisor's alpha
suffix decides between step- and parts-mode — but new entries
should use the explicit `step:` / `parts:` prefixes.

Profiles emitted by all strata in a single load are inserted into
`dataset.yaml` sorted by ascending count, regardless of which
strategy produced them — so e.g. `100k` from `decade` precedes `1m`
from `mul:` in the file and in TUI views.

### Required attributes

Every published dataset must have:
- `distance_function` — the metric used for KNN computation
- `is_zero_vector_free` — set automatically after zero-vector scan
- `is_duplicate_vector_free` — set automatically after dedup scan
