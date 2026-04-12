# analyze find-duplicates

Scan vectors for bitwise-identical duplicates via sort+compare.

## Usage

```bash
veks pipeline analyze find-duplicates --source <file>
```

## Example

```bash
veks pipeline analyze find-duplicates --source profiles/base/base_vectors.fvec
```

```
find-duplicates: scanning ./profiles/base/base_vectors.fvec
Dedup: 1K vectors (f32, dim=128)
Phase 1: creating sorted runs
Phase 2: parallel merge + dedup
  0 duplicates in 1000 vectors (0.00%)
  No duplicates found — all vectors are unique.
```

Sets `duplicate_count` pipeline variable.
