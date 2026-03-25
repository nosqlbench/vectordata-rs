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

## 10.2 Publish Sentinel (`.publish` file)

A dataset directory is included in publishing if and only if it contains
a `.publish` sentinel file (zero-length or otherwise), sibling to
`dataset.yaml`.

```
my-dataset/
├── dataset.yaml
├── .publish              ← marks this dataset for publishing
├── profiles/...
└── ...
```

To mark a dataset for publishing:
```
touch my-dataset/.publish
```

To exclude a dataset from publishing:
```
rm my-dataset/.publish
```

The `prepare publish` command MUST only include dataset directories that
contain a `.publish` file. Datasets without this sentinel are silently
excluded from the publish set. This is not an error — they are local-only
datasets that do not participate in remote distribution.

The `veks check` catalog validation also only requires catalog coverage
for publishable datasets (those with `.publish`). Non-publishable datasets
are excluded from the catalog completeness check.

The `veks check` dataset readout on publish paths shows publishable
datasets in green and local datasets in dim white.

## 10.3 Publish URL Binding: `.publish_url` File

Each publish tree contains a file named `.publish_url` at the publish
root (not necessarily at the individual dataset level).

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

### Sentinel Files

Four sentinel files control publishing and catalog behavior:

- **`.publish`** — Marks a dataset directory for publishing (see §10.2).
  A dataset is included in `prepare publish` and catalog completeness
  checks only if this file is present alongside `dataset.yaml`.

- **`.publish_url`** — Binds a publish tree to a remote URL. Placed at
  the publish root (not per-dataset). See §10.3.

- **`.catalog_root`** — Marks the top of the catalog hierarchy. When
  present, `catalog generate` uses this directory as the scan root and
  generates catalogs at every level from there down to each dataset.

- **`.do_not_catalog`** — Prevents catalog file **placement** in this
  directory. When present, `catalog generate` will not write
  `catalog.json` or `catalog.yaml` here. However, the dataset discovery
  walker still descends through directories with `.do_not_catalog` to
  find datasets below — only catalog output is suppressed, not
  discovery.

**Catalog completeness rules** (enforced by `veks check`):

1. Every directory between the publish root (inclusive) and a
   **publishable** dataset directory (one containing both `dataset.yaml`
   and `.publish`) MUST have a fresh `catalog.json` and `catalog.yaml`,
   unless that directory has `.do_not_catalog`.

2. Any directory with `.do_not_catalog` MUST NOT contain catalog files.

3. Dataset directories without `.publish` are excluded from catalog
   completeness checks entirely.

**`catalog generate` scoping**:

When `veks prepare catalog generate [DIR]` is invoked:

1. **Publish root discovery** (catalog placement boundary): Walk UP from
   `DIR` to find the enclosing `.publish_url`. Catalogs are generated at
   every directory level from each dataset up to this publish root. If no
   `.publish_url` exists at or above `DIR`, scan immediate children for
   child publish roots.

2. **Dataset discovery** (what gets cataloged): Walk DOWN from `DIR` to
   find `dataset.yaml` files. Only publishable datasets (with `.publish`
   sentinel) are included in catalog entries.

3. **Catalog placement**: Write `catalog.json` and `catalog.yaml` at
   every directory on the path from each discovered dataset up to the
   publish root. Skip directories with `.do_not_catalog`.

By default, catalogs are created at all hierarchy levels. The `--update`
flag restricts writes to directories that already have catalog files.

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

### Merkle coverage requirement

Every file within a dataset's `profiles/` directory MUST have a
companion `.mref` merkle reference file. This includes all vector
files, slab files, index files, and any other data content — no
size threshold, no exceptions within `profiles/`.

Infrastructure files at the dataset root level (`dataset.yaml`,
`catalog.json`, `catalog.yaml`, `variables.yaml`) are exempt — they
are too small to benefit from chunked merkle verification and are
frequently regenerated.

This coverage enables:
- **Change detection**: any modification to profile data is detectable
  via root hash comparison
- **Incremental download**: the cache layer verifies and resumes at
  chunk granularity for all profile data
- **Cache recovery**: pre-existing cache files without `.mrkl` state
  are verified chunk-by-chunk against the `.mref` hashes, recovering
  valid content without re-downloading

### Filter strategy

Publishing uses an **include-only** approach rather than building
exclusion patterns:

1. `enumerate_publishable_files()` determines the exact set of files
   to publish (applying all rules above).
2. The sync command excludes everything (`--exclude '*'`), then
   explicitly includes each publishable file.

This guarantees that checks, merkle coverage, and actual sync all
operate on the same file set — defined once in `enumerate_publishable_files`.

---

## 10.5 Sync Semantics

The publish operation follows `aws s3 sync` conventions:

### Publish root and path structure

The **publish root** is the directory containing the `.publish_url` file.
The `publish` command syncs the publish root (not the individual dataset
directory) to preserve the full directory hierarchy in the remote store.

```
/data/vectordata/                         ← publish root (.publish_url here)
├── laion400b/
│   ├── img-search/                       ← dataset (has dataset.yaml)
│   │   ├── dataset.yaml
│   │   └── profiles/...
│   └── import_test/                      ← dataset
│       ├── dataset.yaml
│       └── profiles/...
├── catalog.json                          ← catalog at publish root
└── laion400b/catalog.json                ← catalog at intermediate level
```

S3 key: `s3://bucket/laion400b/img-search/profiles/1m/neighbor_indices.ivec`

### Path structure inclusion rule

Only directories that are **on the path to a publishable `dataset.yaml`**
participate in publishing. Directories not on any path from the publish
root to a publishable dataset directory are excluded entirely.

**Within dataset directories** (at or below a `dataset.yaml`), all
non-excluded files are included — these are dataset content files
(vectors, profiles, metadata).

**At intermediate directories** (on-path but above a dataset directory),
only catalog infrastructure files (`catalog.json`, `catalog.yaml`) and
their companion `.mref` files are included. Arbitrary data files at
intermediate levels are NOT published, even if they happen to be on the
path to a dataset. This prevents non-dataset content (test files,
work-in-progress data, scripts) from being accidentally included.

For example:

```
publish-root/
├── .publish_url
├── group/
│   ├── catalog.json              ← published (infrastructure)
│   ├── base_test.mvec            ← NOT published (data file at intermediate level)
│   └── my-dataset/
│       ├── dataset.yaml
│       ├── .publish
│       ├── base.mvec             ← published (inside dataset)
│       └── profiles/...          ← published (inside dataset)
└── scratch/
    └── temp/data.fvec            ← NOT published (not on path to any dataset)
```

### Object key mapping

Local files map to S3 object keys by joining the bucket prefix from
`.publish_url` with the file's path relative to the **publish root**
(the directory containing `.publish_url`).

```
publish root: /data/vectordata/           (.publish_url → s3://my-datasets/)
local:        /data/vectordata/laion400b/img-search/profiles/1m/neighbor_indices.ivec
key:          laion400b/img-search/profiles/1m/neighbor_indices.ivec
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
aws s3 sync <publish-root> <s3-url> \
  --exclude '*' \
  --include 'path/to/dataset/file1.fvec' \
  --include 'path/to/dataset/file2.ivec' \
  --include 'group/catalog.json' \
  ... \
  [--delete] \
  [--dryrun] \
  [--only-show-errors | progress indicators]
```

The sync uses an **include-only** strategy: `--exclude '*'` first
excludes everything, then each publishable file (as determined by
`enumerate_publishable_files`) is explicitly included. This ensures the
sync matches the file enumeration exactly — one code path decides what's
publishable, eliminating divergence between checks and actual uploads.

Errors from the subprocess are captured and reported through the
standard `CommandResult` mechanism.

### Phase 2: Native S3 (future)

A future iteration may replace the shell-out with native S3 operations
using `aws-sdk-s3` for tighter progress integration, parallel multipart
uploads, and elimination of the external CLI dependency. This is not
required for the initial implementation.

### Transport abstraction

The `vectordata` crate provides a `ChunkedTransport` trait
(`vectordata::transport::ChunkedTransport`) that abstracts byte-range
data access across local files and HTTP endpoints. Publishing and data
access both resolve transport backends through trait-based dispatch
rather than hard-coded protocol handling. This enables S3-compatible,
HTTP, and local-file transports to be selected at runtime based on the
URL scheme in `.publish_url` or the dataset root location.

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
