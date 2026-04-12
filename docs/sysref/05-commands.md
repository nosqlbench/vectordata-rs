# 5. Commands

All commands are available both as pipeline steps (`run: command path`)
and directly via the CLI (`veks pipeline command path --option=value`).

For working examples with real output, see the
[Command Reference](./commands/README.md) — one page per command,
each run against the synthetic-1k test fixture.

---

## 5.1 Analyze

| Command | Description |
|---------|-------------|
| `analyze describe` | File format, dimensions, record count, structure (uniform/variable) |
| `analyze stats` | Per-dimension statistics (mean, stddev, min, max) |
| `analyze display-histogram` | Value distribution histogram |
| `analyze select` | Display specific records by ordinal |
| `analyze find-zeros` | Scan for near-zero vectors (L2 norm threshold) |
| `analyze find-duplicates` | Scan for duplicate vectors (bitwise equality) |
| `analyze explain-predicates` | Trace predicate → matching metadata with selectivity |
| `analyze explain-filtered-knn` | Full query trace through all pipeline stages |
| `analyze compare-files` | Byte-level comparison of two files |
| `analyze check-endian` | Verify endianness of a vector file |
| `analyze file` | Low-level file metadata |
| `analyze survey` | Survey slab field distributions |

---

## 5.2 Compute

| Command | Description |
|---------|-------------|
| `compute knn` | Brute-force exact KNN ground truth |
| `compute filtered-knn` | KNN with predicate pre-filtering |
| `compute evaluate-predicates` | Evaluate predicates against metadata → vvec results |
| `compute sort` | Sort vectors for deduplication |

---

## 5.3 Generate

| Command | Description |
|---------|-------------|
| `generate vectors` | Random vector generation (gaussian, uniform) |
| `generate metadata` | Random integer metadata labels |
| `generate predicates` | Random equality predicates (simple-int-eq) |
| `generate vvec-index` | Build IDXFOR__ offset indices for vvec files |
| `generate shuffle` | Random permutation of vector ordinals |
| `generate dataset-json` | Produce dataset.json from dataset.yaml |
| `generate variables-json` | Produce variables.json from variables.yaml |
| `generate dataset-log-jsonl` | Convert dataset.log to JSONL |
| `generate sketch` | Dimension-reduction sketch |
| `generate derive` | Derive new facets from existing data |

---

## 5.4 Verify

| Command | Description |
|---------|-------------|
| `verify knn-consolidated` | Multi-threaded brute-force KNN verification |
| `verify predicates-sqlite` | SQLite oracle verification for predicate results |
| `verify filtered-knn-consolidated` | Verify filtered KNN with tie-break handling |
| `verify predicate-results` | Consolidated predicate verification (slab mode) |

---

## 5.5 Transform

| Command | Description |
|---------|-------------|
| `transform extract` | Extract vector subsets by ordinal range |
| `transform convert` | Convert between vector formats |
| `transform ordinals` | Apply ordinal permutation |

---

## 5.6 Other

| Command | Description |
|---------|-------------|
| `merkle create` | Generate merkle hash trees for data files |
| `merkle verify` | Verify merkle hashes |
| `catalog generate` | Generate catalog.json index |
| `state set` / `state clear` | Pipeline variable management |
| `download huggingface` | Download from Hugging Face Hub |
| `download bulk` | Parallel bulk download (see config below) |

### Bulkdl configuration

The `download bulk` command uses a YAML config file for parallel
downloads with token-expanded URLs:

```yaml
datasets:
 - name: base
   baseurl: 'https://example.com/data/emb_${number}.npy'
   tokens:
    number: [0..409]
   savedir: embeddings/
   tries: 5
   concurrency: 5
```

| Field | Description |
|-------|-------------|
| `name` | Identifier for this download set |
| `baseurl` | URL template with `${token}` placeholders |
| `tokens` | Token ranges; `[0..409]` = 0 through 409 inclusive |
| `savedir` | Local save directory (created automatically) |
| `tries` | Max retry attempts per file |
| `concurrency` | Concurrent download threads |

Existing files are skipped if remote Content-Length matches local size.
