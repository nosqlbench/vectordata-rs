<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bulkdl Configuration Reference

The `veks bulkdl` command downloads files in parallel, driven by a YAML
configuration file.

## Config format

```yaml
datasets:
 - name: base
   baseurl: 'https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_${number}.npy'
   tokens:
    number: [0..409]
   savedir: embeddings/img_emb/
   tries: 5
   concurrency: 5
```

## Fields

| Field | Description |
|-------|-------------|
| `name` | Identifier for this dataset entry |
| `baseurl` | URL template with `${token}` placeholders |
| `tokens` | Token definitions; `[0..409]` means 0 through 409 inclusive |
| `savedir` | Local directory to save downloads (created automatically) |
| `tries` | Maximum retry attempts per file |
| `concurrency` | Number of concurrent download threads |

## Behavior

- Token placeholders in `baseurl` are expanded over the values in their
  token spec. For example, `${number}` with `[0..409]` produces 410 URLs.
- Existing local files are skipped if a HEAD probe on the remote resource
  returns a content-length matching the local file size.
- Each dataset maintains a small status file that allows completely skipping
  datasets or individual resources without performing length probes.
- A progress bar is displayed on the console during downloads.
- A log file is written for troubleshooting.
