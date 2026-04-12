# System Reference

Technical reference documentation for the vectordata-rs workspace.

Each section is self-contained and covers one functional area.
Start with **Data Model** to understand the file formats, then
**API** to access data, then explore the rest as needed.

---

## Contents

### [1. Data Model](./01-data-model.md)
File formats, extension scheme, record layouts, and the BQGDMPRF
facet system. The foundation everything else builds on.

- Vector formats: vec (uniform) and vvec (variable-length)
- Scalar formats: flat-packed integer/float arrays
- Offset index files (IDXFOR__)
- Dataset facets: B, Q, G, D, M, P, R, F
- dataset.yaml specification: profiles, attributes, variables

### [2. API](./02-api.md)
The vectordata consumer library. How to load datasets, read vectors,
and access metadata from Rust code.

- `open_vec<T>()` and `open_vvec<T>()` — unified file access
- `TestDataGroup` and `TestDataView` — catalog-driven dataset access
- `TypedReader<T>` — scalar access with width negotiation
- Traits: `VectorReader<T>`, `VvecReader<T>`, `VvecElement`
- Thread safety, error handling

### [3. Catalogs and Publishing](./03-catalogs.md)
Dataset discovery, distribution, and integrity verification.

- Catalog configuration (~/.config/vectordata/catalogs.yaml)
- Dataset publishing to S3 or static HTTP
- Prebuffering and local caching
- Merkle-verified integrity (.mref files)
- Preflight checks (veks check)

### [4. Pipeline Engine](./04-pipeline.md)
The DAG execution engine that processes dataset.yaml pipelines.

- Step resolution and topological ordering
- Variable interpolation and state management
- Per-profile expansion (sized profiles)
- Resource governance (memory, threads)
- Progress display and logging
- Pipeline DAG configurations and facet inference

### [5. Commands](./05-commands.md)
Reference for all pipeline commands and CLI operations.

- analyze: inspect, describe, explain, histogram, stats, select
- compute: knn, filtered-knn, evaluate-predicates, sort
- generate: vectors, metadata, predicates, vvec-index, merkle
- verify: knn, predicates-sqlite, filtered-knn
- transform: convert, extract, ordinals

### [6. Data Processing](./06-processing.md)
Algorithms for vector data preparation and quality assurance.

- Deduplication (sort-based, bitwise equality)
- L2 normalization with precision analysis
- Zero vector detection and filtering
- Unified sort-deduplicate-extract pipeline
- Metadata synthesis (simple-int-eq mode)
- Predicate evaluation and selectivity

### [7. Dataset Import](./07-import.md)
The bootstrap wizard and pipeline generation for new datasets.

- Import flowchart and facet inference rules
- Source file detection and role assignment
- Identity vs Materialized artifacts
- Profile layout: profiles/base/ and profiles/default/
- Sized profile generation (stratification)

### [8. Architecture](./08-architecture.md)
Internal design of the pipeline engine, CLI framework, and UI layer.

- CommandOp trait and command documentation
- Resource governance (memory/thread budgets, OOM prevention)
- UI eventing layer (sinks, handles, progress bars)
- Facet swimlane diagram (superset pipeline visualization)
- Pipeline DAG configurations and variable reference

### [9. Algorithms](./09-algorithms.md)
Detailed specifications for statistical and numerical algorithms.

- Normalization analysis (Higham bounds, precision-aware thresholds)
- Statistical vector generation (Virtdata: deterministic inverse CDF)
- Statistical model extraction (Vshapes: Pearson classification, EM clustering)
- Shared numerical utilities (log gamma, incomplete beta/gamma)

### [10. ANode Codec](./10-anode-codec.md)
Binary codecs and human-readable renderers for metadata and predicates.

- Two-stage architecture (bytes ↔ ANode ↔ text)
- 13 vernacular formats (JSON, SQL, CQL, CDDL, YAML, etc.)
- MNode and PNode usage with code examples
- Type mapping, roundtrip verification, extending with new formats

### [11. Shell Completions](./11-completions.md)
Dynamic tab-completion engine for bash and other shells.

- Two-crate architecture (veks-completion + dyncomp)
- Value providers for datasets, profiles, catalogs, metrics
- Completion algorithm and consumed-option filtering
- Bash script generation

### [12. knn_utils Verification](./12-knn-utils-verification.md)
Cross-verification against the Python knn_utils reference implementation.

- knn_utils personality (BLAS, MT19937, numpy normalization)
- Verification levels: byte-identical, set-equivalent, self-consistent
- BLAS and numerical precision considerations
- A/B testing with FAISS
