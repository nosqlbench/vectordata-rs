# generate passage-metadata

Join S2AG `papers` metadata onto a passage table. Scans papers dataset
shards (JSONL, optionally `.gz`; one record per corpusid) for the parent
documents of a `passages.parquet`, then writes `metadata.parquet` with one
row per passage **in passage row order** — the M-facet raw input for a
predicated (PVS) dataset build via `veks prepare bootstrap --metadata`.

Columns (scalars only, the shape the parquet→MNode reader consumes):
`corpusid` (i64), `section` (utf8, passage-level), `year` (i32, 0 =
unknown), `citationcount` (i64), `isopenaccess` (bool), `field` (utf8,
primary s2fieldsofstudy category), `venue` (utf8). Parents absent from the
papers shards get the documented defaults and are counted in the result
message.

## Usage (pipeline step)

```yaml
- id: generate-metadata
  run: generate passage-metadata
  after: [download-papers, generate-passages]
  source: sources/papers
  passages: upstream/passages/passages.parquet
  output: upstream/metadata/metadata.parquet
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--source` | yes | S2AG papers JSONL(.gz) shard file or directory of shards |
| `--passages` | yes | `passages.parquet` whose row order the output mirrors |
| `--output` | yes | Output `metadata.parquet` (one row per passage, row-aligned) |
| `--files` | no | Shard selection over lexically-sorted basenames: `first:N` (strict), a glob, or `all` (default) |
| `--threads` | no | Shard-scan worker threads (default 0 = all cores); does not affect output bytes |

## Notes

- Row i of the output describes row i of `passages.parquet` and therefore
  row i of the embedded vectors — assert it with `verify alignment`
  (vectors vs metadata.parquet) before the dataset build.
- Shards are scanned in parallel; a cheap `"corpusid":` prefilter means
  only records in the parent set pay a full JSON parse, so the scan is
  IO/decompress-bound.
- Parent-level fields are broadcast to every passage of the parent;
  `section` is the one passage-level column.
