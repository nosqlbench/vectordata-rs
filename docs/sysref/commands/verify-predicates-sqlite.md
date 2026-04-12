# verify predicates-sqlite

SQLite oracle verification for predicate evaluation results.

## Usage (pipeline step)

```yaml
- id: verify-predicates-sqlite
  run: verify predicates-sqlite
  per_profile: true
  phase: 1
  after: [evaluate-predicates]
  metadata: profiles/base/metadata_content.u8
  predicates: profiles/base/predicates.u8
  results: metadata_indices.ivvec
  fields: 1
  output: "${cache}/verify_predicates_sqlite.json"
```

## Behavior

1. Loads all metadata records into SQLite
2. Loads predicates
3. For each predicate, executes SQL query independently
4. Compares SQL results against stored R facet
5. Reports any mismatches

Provides a ground-truth oracle that shares no code with the evaluation
pipeline.
