# veks prepare cache-gc

Remove files under `.cache/` that nothing in the current pipeline can
use.

The cache holds what steps write for other steps and what engines keep
for themselves: intermediates named `${cache}/...` in `dataset.yaml`,
the KNN engines' per-segment results, the predicate-key segments of
each facet, an extract's resume partitions, and the provenance
sidecars of all of it. When a step is removed, an engine's cache
version changes, or a keying changes, what the old arrangement wrote
stays behind. This command finds and removes it — and nothing else.

## What is live

A file is live when the definition, as it stands, can still consume
it. Liveness comes from the pipeline's own knowledge, never from a list
of names kept by this command:

- **what a step names** — every input, intermediate and output under
  the cache in the projected manifest, and every option value that
  mentions the cache at all, so a step the registry does not know or
  whose variables cannot yet be resolved still protects what it names;
- **what a defined step recorded** — the outputs the progress log
  holds for a step still in the definition; records of steps the
  definition no longer has protect nothing;
- **what a command claims** — the cache each command keeps beyond its
  manifest, declared as a name prefix or a directory: a KNN engine's
  segments for its `(base, query)` pair across every `k` and metric,
  the predicate-key segments under a facet's content key, an extract's
  resume directory for its output;
- **twins** — the `.provenance.json` sidecar and `.gz` form of anything
  live;
- **the runner's own files** — the progress log and its migration
  backups, `run.log`, `.governor.log`, `meta.json`, and the
  `provenance/` tree of dataset-artifact sidecars, kept whole.

A directory that holds something live — `slab-extract/` with one
claimed output directory among stale ones — is walked, not removed:
its dead entries go and the live ones stay.

## Usage

```bash
veks prepare cache-gc --dry-run [dataset-dir | dataset.yaml]   # report only
veks prepare cache-gc [dataset-dir | dataset.yaml]             # remove
```

## Example

After a profile was removed and an engine's cache version moved from
`v2` to `v3`:

```
Orphaned under .cache/ (312 live, 3 orphaned, 41.2 GiB):
  file   38.9 GiB  knn-blas.v2.base_vectors.query_vectors.412000000_10000.range_000000000000_000001000000.k100.l2.results.bin
  file    2.3 GiB  keys.9e1c...f2.seg_0000000000_0001000000.predkeys.slab
  dir     4.1 MiB  slab-extract/topic_margin-77a0b1c2

Dry run — nothing removed. Run without --dry-run to remove.
```

The `v3` segments of the same pair, the current facet's key segments
and the current extract's resume directory are among the live entries
and are not listed.

## Options

| Option | Description |
|--------|-------------|
| `path` | Dataset directory or path to `dataset.yaml` (default: `.`) |
| `--dry-run` | Show what would be removed without changing anything |

## See also

- [veks prepare cleanup-profiles](./veks-prepare-cleanup-profiles.md) — removes profiles no stratum names and leaves the cache alone; run it first, so the cache-gc pass sees the definition that remains
