# generate passages

Derive passages from an s2orc-format text corpus (JSONL or JSONL.gz shards)
with the deterministic, versioned, section-aware chunker `para-v1`. Three
record layouts are accepted: `s2orc_v2` (`body.text` / `body.annotations`,
header spans under `section_header`; the parallel `bibliography` object is
never chunked), classic nested `s2orc`
(`content.text` / `content.annotations.sectionheader`), and flat
derivatives with a top-level `text` (e.g. peS2o). Emits
`passages.parquet` plus a `parents.parquet` manifest in parent-block order:
all passages of a document are contiguous, the global passage ordinal is the
row index, and passage identity is the (corpusid, section, ordinal) triple.
Every option that changes output bytes is a provenance axis.

## Usage (pipeline step)

```yaml
- id: generate-passages
  run: generate passages
  source: s2orc
  output: upstream/passages.parquet
  doc-limit: 1000
  doc-order: corpusid
  chunker: para-v1
  seed: 42
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--source` | yes | s2orc-format JSONL(.gz) shard file or directory of shards |
| `--output` | yes | Output `passages.parquet` path |
| `--parents` | no | Parent-manifest parquet path (default: `parents.parquet` beside output) |
| `--doc-limit` | no | Select the N lowest corpusids with non-empty body text (default: all) |
| `--doc-order` | no | Parent block order: `corpusid` (default), `shuffle` (seeded), `source` (streaming, constant memory) |
| `--seed` | no | Random seed for `doc-order: shuffle` (default: 0) |
| `--chunker` | no | Chunker policy id (default: `para-v1`) |
| `--min-words` | no | Merge a trailing chunk below this many words into its predecessor (default: 40) |
| `--target-words` | no | Window size in words when splitting oversized paragraphs (default: 170) |
| `--max-words` | no | Maximum words packed into one passage (default: 230) |

## Notes

- Word budgets approximate tokens at ≈ tokens/1.3; the defaults land in the
  150–300-token passage-policy band.
- `doc-order: shuffle` is the parent-granularity shuffle that makes prefix
  windows over the output behave as parent-sampled strata.
- Downstream embedding must preserve row order (row i of the vectors
  artifact embeds parquet row i) — enforce with
  [`verify alignment`](./verify-alignment.md).
- Sets the `passage_count` and `parent_count` pipeline variables and logs
  the passages/doc fan-out distribution.
