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
| HDF5 | `.hdf5`, `.h5` | Import source (datasets via `#path` notation) |

---

## 1.4 Dataset Facets (BQGDMPRF)

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

### Facet inference

The pipeline infers which facets to produce from the inputs provided:

| Inputs | Inferred facets |
|--------|----------------|
| B only | B |
| B + Q | B Q G D |
| B + Q + GT | B Q G (D only if distances provided) |
| B + Q + M | B Q G D M P R F |
| B + Q + GT + synthesize | B Q G M P R F |

---

## 1.5 Dataset Layout

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

- `_` prefix: source files excluded from publishing (e.g., `_sift_base.fvecs`)
- `IDXFOR__` prefix: offset index companion files (auto-generated, published)
- `.mref`: merkle hash tree (one per data file, published)
- `.mrkl`: local merkle cache state (not published)

---

## 1.6 dataset.yaml

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
```

### Profiles

- **`default`** — the full dataset, always present
- **Sized profiles** — subsets with `base_count` (e.g., `100K`, `1M`)
  that share source data but have independently computed KNN and
  filtered results

### Required attributes

Every published dataset must have:
- `distance_function` — the metric used for KNN computation
- `is_zero_vector_free` — set automatically after zero-vector scan
- `is_duplicate_vector_free` — set automatically after dedup scan
