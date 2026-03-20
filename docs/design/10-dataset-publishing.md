<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 10 — Dataset Publishing

This document specifies the mechanism for publishing prepared dataset
directories to remote S3 storage, enabling datasets produced by `veks run`
to be served as static files for remote access (§5.8.6).

---

## 10.1 Scope

Dataset publishing is the final stage in the dataset lifecycle:

```
prepare (veks run) → verify → publish (veks publish) → serve (static HTTP)
```

Publishing synchronizes the publishable content of a local dataset
directory to an S3 bucket, using the same semantics as `aws s3 sync`.
The target bucket is recorded in a hidden configuration file within the
dataset directory, binding the directory to its remote location.

---

## 10.2 Publish URL Binding: `.publish_url` File

Each dataset directory that participates in publishing contains a file
named `.publish_url` at the dataset root, sibling to `dataset.yaml`.

### Format

The file contains a single line: the full publish URL of the target location,
with no trailing newline required.

```
s3://bucket-name/optional/prefix/path/
```

| Constraint | Rule |
|------------|------|
| Scheme | Must be `s3://` |
| Bucket name | Standard S3 bucket naming rules |
| Prefix path | Optional. If present, acts as the root prefix for all uploaded objects |
| Trailing slash | Recommended but not required; normalized internally |
| Whitespace | Leading/trailing whitespace is trimmed |
| Comments | Lines starting with `#` are ignored |

### Example

```
dataset-name/
├── .publish_url                      # → s3://my-datasets/vectordata/dataset-name/
├── dataset.yaml
├── .scratch/
├── .cache/
├── base_vectors.mvec
├── query_vectors.mvec
├── neighbor_indices.ivec
└── ...
```

Contents of `.publish_url`:
```
s3://my-datasets/vectordata/dataset-name/
```

### Rationale

A hidden dot-file is chosen over a field in `dataset.yaml` because:

- **Separation of concerns**: `dataset.yaml` describes the dataset's
  logical structure; the publish target is an operational binding that
  varies per deployment.
- **Git-friendliness**: `.publish_url` can be `.gitignore`d when targets
  differ across environments, or committed when the target is canonical.
- **Simplicity**: No schema changes to `dataset.yaml` are required.

---

## 10.3 `veks publish` Command

### Synopsis

```
veks publish [OPTIONS] [DIRECTORY]
```

If `DIRECTORY` is omitted, the current working directory is used.

### Behavior

1. **Locate `.publish_url`**: Read `{directory}/.publish_url`. If the file
   does not exist, exit with an error explaining how to create one.

2. **Validate**: Parse the publish URL. Verify the scheme is `s3://` and
   the bucket name is syntactically valid.

3. **Enumerate publishable files**: Walk the dataset directory, applying
   inclusion and exclusion filters (§10.4).

4. **Present summary and confirm**: Display a summary of the planned
   sync operation (§10.3.1) and wait for the user to type `YES`
   (case-insensitive) before proceeding. Skip this prompt if `-y` is
   given.

5. **Sync to S3**: For each publishable file, upload to the corresponding
   S3 key under the configured prefix. Apply sync semantics: skip files
   whose remote object already exists with matching size and modification
   time (ETag/last-modified heuristic, consistent with `aws s3 sync`).

6. **Report**: Print a summary of files uploaded, skipped, and any errors.

### 10.3.1 Confirmation Prompt

Before uploading, the command prints a summary of the planned actions
and requires explicit confirmation. This prevents accidental pushes to
the wrong bucket or unintended `--delete` operations.

**Summary contents**:

```
Publish summary:
  Source:      /data/my-dataset/
  Destination: s3://my-datasets/vectordata/my-dataset/
  Files:       47 to upload, 12 unchanged (skipped), 3.2 GiB total
  Deletes:     0 remote objects (--delete not specified)
  Excludes:    .scratch/, .cache/, .publish_url, .git/, ...

Proceed? Type YES to confirm:
```

When `--delete` is active and remote-only objects exist:

```
  Deletes:     3 remote objects to remove
    old_vectors.fvec (1.2 GiB)
    stale_index.ivec (400 MiB)
    orphan.json (2 KiB)
```

**Confirmation rules**:

| Input | Result |
|-------|--------|
| `YES`, `yes`, `Yes`, `yEs`, etc. | Proceed with upload |
| Any other input (including `y`, `Y`, empty) | Abort with message |
| `-y` flag on command line | Skip prompt entirely |
| `--dry-run` flag | Skip prompt (no destructive action) |
| Stdin is not a TTY (piped input) | Abort with error unless `-y` given |

The `-y` flag is intended for scripted/CI use where interactive
confirmation is not possible. The `--dry-run` flag naturally bypasses
confirmation since it performs no mutations.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-y` | flag | false | Skip confirmation prompt (for scripted/CI use) |
| `--dry-run` | flag | false | Show what would be uploaded without transferring |
| `--delete` | flag | false | Remove remote objects that no longer exist locally (mirror semantics) |
| `--concurrency` | integer | 4 | Number of parallel upload streams |
| `--exclude` | string[] | — | Additional glob patterns to exclude |
| `--include` | string[] | — | Additional glob patterns to force-include (overrides excludes) |
| `--size-only` | flag | false | Skip based on size only, ignoring timestamps |
| `--profile` | string | — | AWS profile name for credentials |
| `--endpoint-url` | string | — | Custom S3 endpoint (for S3-compatible stores) |

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | All files synchronized successfully |
| 1 | One or more files failed to upload |
| 2 | Configuration error (missing `.publish_url`, invalid URL, etc.) |
| 3 | User declined confirmation prompt |

---

## 10.4 File Filtering Rules

Publishing must include only the files needed for remote dataset serving
(§5.8.6) and exclude workspace artifacts that are local-only.

### Hidden files and directories

**All hidden files and directories (names starting with `.`) are
categorically excluded from publishing.** Hidden entries are treated as
local workspace state — they never appear in the remote dataset. This
includes `.scratch/`, `.cache/`, `.publish_url`, `.git/`, `.gitignore`,
`.backup/`, `.upstream.progress.yaml`, `.governor.log`, and any other
dot-prefixed name.

This rule is unconditional: `--include` cannot override it for hidden
entries. Local state must never leak into the published dataset.

### Additional default exclusions

The following non-hidden patterns are also excluded:

| Pattern | Reason |
|---------|--------|
| `*.tmp`, `*.partial` | Incomplete file artifacts |
| `__pycache__/`, `*.pyc` | Python bytecode (from tooling) |

### Default inclusions

All other files in the dataset directory are included, notably:

| Content | Examples |
|---------|---------|
| Dataset manifest | `dataset.yaml` |
| Vector data files | `*.fvec`, `*.ivec`, `*.mvec`, `*.bvec`, `*.dvec`, `*.svec` |
| Slab files | `*.slab` |
| Merkle reference files | `*.mref` |
| Merkle state files | `*.mrkl` |
| JSON metadata | `*.json` |
| Profile subdirectories | `profiles/*/...` |

### Filter precedence

1. Default exclusions apply first.
2. User `--exclude` patterns are appended to the exclusion list.
3. User `--include` patterns override all exclusions (force-include).

This matches `aws s3 sync` filter semantics where `--include` after
`--exclude` re-includes matched paths.

---

## 10.5 Sync Semantics

The publish operation follows `aws s3 sync` conventions:

### Object key mapping

Local files map to S3 object keys by joining the bucket prefix from
`.publish_url` with the file's path relative to the dataset directory.

```
local:  /data/my-dataset/profiles/1m/neighbor_indices.ivec
bucket: s3://my-datasets/vectordata/my-dataset/
key:    vectordata/my-dataset/profiles/1m/neighbor_indices.ivec
```

### Skip logic

A local file is skipped (not uploaded) when the remote object exists and:

- **Default**: Both file size and last-modified time match. The local
  file's mtime is compared against the S3 object's `LastModified`.
- **`--size-only`**: Only file size is compared, ignoring timestamps.

When in doubt (clock skew, missing metadata), the file is uploaded.

### Delete semantics (`--delete`)

When `--delete` is specified, remote objects under the configured prefix
that have no corresponding local file (after filtering) are deleted.
This provides mirror semantics for keeping the remote copy exactly in
sync with the local directory.

Without `--delete`, remote objects not present locally are left untouched.
This is the safe default for incremental publishing.

---

## 10.6 Pipeline Integration

Publishing can also be invoked as a pipeline step in `dataset.yaml`,
running after all other steps complete:

```yaml
upstream:
  steps:
    # ... data preparation steps ...

    - id: publish
      run: publish
      description: Sync dataset to S3
      after: [compute-knn, verify-knn]
```

When run as a pipeline step, the command reads `.publish_url` from the
workspace root (same as CLI mode). No additional options are required
for the common case, but pipeline step options can override defaults:

```yaml
    - id: publish
      run: publish
      after: [verify-knn]
      dry_run: false
      delete: true
      concurrency: 8
```

---

## 10.7 Implementation Strategy

### Phase 1: Shell-out to AWS CLI

The initial implementation invokes `aws s3 sync` as a subprocess,
constructing the argument list from the parsed `.publish_url` URL and
filter rules. This approach:

- Requires no new Rust S3 SDK dependency
- Leverages the battle-tested sync logic in the AWS CLI
- Supports all AWS credential resolution mechanisms (env vars, profiles,
  instance roles, SSO) without additional code
- Provides immediate feature parity with operator expectations

The command constructs an invocation equivalent to:

```bash
aws s3 sync <dataset-dir> <s3-url> \
  --exclude '.*' \
  --exclude '*/.*' \
  --exclude '*.tmp' \
  --exclude '*.partial' \
  --exclude '__pycache__/*' \
  --exclude '*.pyc' \
  [--delete] \
  [--dryrun] \
  [--only-show-errors | progress indicators]
```

The `--exclude '.*' --exclude '*/.*'` patterns categorically exclude all
hidden files and directories at every level.

Errors from the subprocess are captured and reported through the
standard `CommandResult` mechanism.

### Phase 2: Native S3 (future)

A future iteration may replace the shell-out with native S3 operations
using `aws-sdk-s3` for tighter progress integration, parallel multipart
uploads, and elimination of the external CLI dependency. This is not
required for the initial implementation.

---

## 10.8 Operational Workflow

### First-time setup

```bash
cd /data/my-dataset
echo 's3://my-datasets/vectordata/my-dataset/' > .publish_url
veks publish --dry-run    # preview what would be uploaded
veks publish              # shows summary, prompts for YES, then uploads
```

### Incremental update

After re-running a pipeline to add new profiles or recompute ground
truth:

```bash
veks run dataset.yaml --profile 10m
veks publish              # shows summary of new/changed files, prompts
```

### Scripted / CI use

```bash
veks publish -y           # skip confirmation prompt
```

### Full mirror

To ensure the remote copy exactly matches the local state (removing
stale remote files):

```bash
veks publish --delete --dry-run   # preview deletions (no prompt)
veks publish --delete             # shows summary including deletions, prompts
```

### S3-compatible stores

For MinIO, Cloudflare R2, or other S3-compatible endpoints:

```bash
echo 's3://my-bucket/datasets/' > .publish_url
veks publish --endpoint-url https://minio.internal:9000
```
