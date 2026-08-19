# download s2ag

Download S2AG Datasets API bulk files. Negotiates the file list for a pinned
Semantic Scholar dataset release (signed, expiring URLs are re-negotiated on
every run), selects files deterministically over lexically-sorted basenames,
and downloads with retries, a worker pool, and status-file resume
(`.s2ag-status.json` in the output directory). Files are stored under their
URL basename with the signature query stripped, exactly as served (typically
gzipped JSONL) — no decompression.

The API key is sent as the `x-api-key` header on the file-list request,
resolved from `api-key-file` when set (single-line file, or a YAML map with
an `S2-API-KEY` entry), else from the `S2_API_KEY` environment variable.
Either way the key contents never land in provenance records or
`dataset.log` — at most the key-file path does.

## Usage (pipeline step)

```yaml
- id: download-s2orc
  run: download s2ag
  release: 2026-08-12
  dataset-name: s2orc
  files: first:1
  output: s2orc
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--release` | yes | Explicit release id (e.g. `2026-08-12`); `latest` is rejected so provenance identifies the data actually downloaded |
| `--dataset-name` | no | S2AG dataset name (default: `s2orc`; also `papers`, `abstracts`, `tldrs`, …) |
| `--files` | no | Selection over lexically-sorted basenames: `first:N` (default `first:1`), a glob, or `all` |
| `--output` | yes | Output directory for downloaded shard files |
| `--tries` | no | Download attempts per file (default: 3) |
| `--concurrency` | no | Parallel download workers (default: 4) |
| `--api-key-file` | no | File holding the API key (single line, or YAML with an `S2-API-KEY` entry); overrides `S2_API_KEY` |
| `--api-base` | no | Datasets API base URL (default: `https://api.semanticscholar.org/datasets/v1`) |

## Notes

- Signed URLs sign the GET method, so no HEAD probe is issued (unlike
  `download bulk` template mode, which cannot consume per-file signed URLs).
- With a `first:N` selector the step reports Complete offline once the
  status file records N completions; glob selectors always resume.
