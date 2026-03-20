<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 11 — Preflight Checks (`veks check`)

This document specifies the `veks check` command — an umbrella
pre-flight verification tool that inspects a dataset directory tree and
reports whether it is ready for publishing or other downstream operations.

---

## 11.1 Scope

`veks check` answers the question: *"Is this directory tree ready to
publish?"* It validates multiple independent aspects of readiness and
reports all findings in a single pass, so operators can fix everything
before attempting a `veks publish`.

The command is designed to be run as the last step before publishing, as
a CI gate, or as a quick diagnostic when something looks wrong.

---

## 11.2 Synopsis

```
veks check [OPTIONS] [DIRECTORY]
```

If `DIRECTORY` is omitted, the current working directory is used. The
command walks the directory tree rooted at `DIRECTORY`, discovering all
`dataset.yaml` files and their enclosing dataset workspaces.

---

## 11.3 Check Categories

Each check category is independently selectable. By default **all**
categories are enabled (equivalent to `--check-all`). When any explicit
`--check-*` flag is given, **only** the flagged categories run.

| Flag | Category | What it verifies |
|------|----------|-----------------|
| `--check-all` | All | Enable every check (default when no flags given) |
| `--check-pipelines` | Pipeline completeness | All pipelines fully executed and fresh |
| `--check-publish` | Publish URL binding | Valid `.publish_url` file reachable in hierarchy |
| `--check-merkle` | Merkle coverage | `.mref` files exist for all large data files |
| `--check-integrity` | File integrity | Geometry, record structure, format-specific validation |

### Flag interaction rules

```
no flags           → all checks enabled (same as --check-all)
--check-all        → all checks enabled
--check-pipelines  → only pipeline check
--check-merkle     → only merkle check
--check-pipelines --check-merkle → pipeline + merkle only
--check-all --check-pipelines    → all checks (--check-all wins)
```

---

## 11.4 Pipeline Completeness Check (`--check-pipelines`)

For each `dataset.yaml` discovered under the target directory, verify
that the pipeline has been fully executed and that all outputs are fresh.

### Algorithm

```
for each dataset.yaml found:
    1. Parse dataset.yaml → extract upstream.steps
    2. If no upstream.steps → skip (no pipeline defined)
    3. Load .cache/.upstream.progress.yaml
       - If missing → FAIL: "no progress log"
    4. Check staleness: if dataset.yaml mtime > progress log mtime
       → FAIL: "dataset.yaml modified since last run"
    5. For each step in the pipeline:
       a. Look up the step ID in the progress log
       b. If not recorded → FAIL: "step '<id>' not executed"
       c. If status ≠ Ok → FAIL: "step '<id>' status: <status>"
       d. Run freshness check (output sizes, option changes,
          input mtimes) via ProgressLog::check_step_freshness()
          - If stale → FAIL: "step '<id>': <reason>"
    6. If all steps pass → OK
```

### Profile awareness

When a dataset has multiple profiles with profile-gated steps (§5.1),
the check verifies that **at least the default profile's steps** are
complete. Additional profiles are checked only if their gated steps
appear in the progress log (i.e., they were previously run).

Future: a `--profile <name>` option could restrict the check to a
specific profile's step set.

### Output

```
check pipelines:
  ✓ my-dataset/dataset.yaml (12/12 steps ok)
  ✗ other-dataset/dataset.yaml
      step 'compute-knn': output 'neighbor_indices.ivec' missing
      step 'verify-knn': not executed
```

---

## 11.5 Bucket Binding Check (`--check-publish`)

Verify that a valid `.publish_url` file exists and is reachable from the
target directory, establishing where data would be published.

### Algorithm

```
1. Search for .publish_url starting at the target directory:
   a. Check DIRECTORY/.publish_url
   b. If not found, walk up parent directories until filesystem root
   c. Stop at the first .publish_url found (nearest ancestor wins)
2. If no .publish_url found anywhere → FAIL: "no .publish_url file"
3. Read and validate the file contents:
   a. Trim whitespace, skip comment lines (starting with #)
   b. Parse as URL
   c. Scheme must be s3:// → else FAIL: "invalid scheme '<scheme>'"
   d. Bucket name must be non-empty → else FAIL: "empty bucket name"
4. For each dataset.yaml found under the target directory:
   a. Determine which .publish_url applies (nearest ancestor)
   b. Report the effective S3 prefix for that dataset
5. All valid → OK
```

### Hierarchy semantics

The `.publish_url` file can be placed at any level in the directory tree.
A common layout for multi-dataset workspaces:

```
workspace/
├── .publish_url                    # s3://my-bucket/datasets/
├── dataset-a/
│   ├── dataset.yaml
│   └── ...
├── dataset-b/
│   ├── dataset.yaml
│   └── ...
└── dataset-c/
    ├── .publish_url                # s3://other-bucket/special/
    ├── dataset.yaml
    └── ...
```

Here `dataset-a` and `dataset-b` inherit the workspace-level bucket,
while `dataset-c` overrides with its own. The check reports the
effective binding for each dataset.

### Output

```
check bucket:
  ✓ .publish_url found: s3://my-bucket/datasets/
      my-dataset/ → s3://my-bucket/datasets/my-dataset/
      other-dataset/ → s3://my-bucket/datasets/other-dataset/
```

Or on failure:

```
check bucket:
  ✗ no .publish_url file found in directory hierarchy
```

---

## 11.6 Merkle Coverage Check (`--check-merkle`)

Verify that all publishable data files above a size threshold have
companion `.mref` merkle reference files. This ensures that remote
clients can perform integrity-verified downloads (§5.8.6).

### Algorithm

```
1. Enumerate all publishable files under the target directory
   (applying the same inclusion/exclusion filters as veks publish §10.4)
2. For each file:
   a. If file size < threshold (default 100 MB) → skip
   b. Check if companion .mref file exists at <file>.mref
   c. If .mref missing → FAIL: "no .mref for <file> (<size>)"
   d. If .mref exists but is older than the data file (mtime) →
      FAIL: ".mref stale for <file>"
3. Summarize: N files checked, M covered, K missing/stale
```

### Size threshold

The default threshold is 100 MB (`--merkle-min-size 100M`). Files below
this size can be downloaded and verified via HTTP content-length alone.
The threshold is configurable:

```
veks check --check-merkle --merkle-min-size 50M
```

Accepts the same suffix notation as window sizes (§5.3): `K`, `M`, `G`,
`KiB`, `MiB`, `GiB`, etc.

### Which files are checked

The merkle check applies the same file filtering as `veks publish`
(§10.4): `.scratch/`, `.cache/`, `.publish_url`, `.git/`, and temp files
are excluded. Only files that would actually be published are checked
for merkle coverage.

Standard publishable extensions that warrant merkle coverage:

| Extension | Typical size range |
|-----------|--------------------|
| `.fvec`, `.ivec`, `.mvec`, `.bvec`, `.dvec`, `.svec` | 100 MB – 100+ GB |
| `.slab` | 100 MB – 50+ GB |

Smaller files like `dataset.yaml`, `.json`, and `.mref` files themselves
are naturally below the threshold and skipped.

### Output

```
check merkle:
  ✓ 8 files checked, all have current .mref
```

Or with issues:

```
check merkle:
  ✗ 2 files missing merkle coverage:
      base_vectors.fvec (12.4 GiB) — no .mref
      metadata_content.slab (3.2 GiB) — .mref stale (data newer)
    6 files ok
```

---

## 11.7 File Integrity Check (`--check-integrity`)

Verify that publishable data files have correct geometry and record
structure for their format. This catches truncated files, corrupt
dimension headers, and structurally invalid slab pages.

### Algorithm

```
1. Enumerate all publishable files under the target directory
   (same filters as veks publish §10.4)
2. For each file with a recognized format extension:
   a. xvec (fvec, ivec, mvec, bvec, dvec, svec):
      - Read the dimension header (first 4 bytes LE i32)
      - Verify dimension > 0
      - Verify file size is evenly divisible by stride
        (4 + dim × element_size)
      - Spot-check: verify second record's dimension matches first
   b. slab:
      - Open via SlabReader (validates pages page structure)
      - Verify all page entry offsets fall within file bounds
   c. Other formats (npy, parquet): skipped
3. Summarize: N checked, M valid, K invalid
```

### Output

```
check integrity:
  ✓ 23 data file(s) checked, all valid
```

Or with issues:

```
check integrity:
  ✗ base_vectors.fvec: file size 12345678 is not evenly divisible
    by stride 3076 (dim=768, elem=4B): 4013 complete records +
    890 trailing bytes
  ✗ metadata.slab: slab open failed: invalid magic bytes
    22 ok, 2 invalid out of 24 checked
```

---

## 11.8 Global Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--json` | flag | false | Emit results as structured JSON instead of human-readable text |
| `--quiet` | flag | false | Suppress per-item detail; exit code only |
| `--merkle-min-size` | size | `100M` | Minimum file size for merkle coverage check |

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | All enabled checks passed |
| 1 | One or more checks failed |
| 2 | Configuration error (invalid directory, unparseable dataset.yaml, etc.) |

---

## 11.9 Structured Output (`--json`)

When `--json` is specified, the output is a single JSON object:

```json
{
  "directory": "/data/workspace",
  "checks": {
    "pipelines": {
      "status": "fail",
      "datasets": [
        {
          "path": "my-dataset/dataset.yaml",
          "status": "ok",
          "steps_total": 12,
          "steps_ok": 12
        },
        {
          "path": "other-dataset/dataset.yaml",
          "status": "fail",
          "steps_total": 8,
          "steps_ok": 6,
          "failures": [
            {"step": "compute-knn", "reason": "output 'neighbor_indices.ivec' missing"},
            {"step": "verify-knn", "reason": "not executed"}
          ]
        }
      ]
    },
    "bucket": {
      "status": "ok",
      "bucket_file": ".publish_url",
      "url": "s3://my-bucket/datasets/",
      "bindings": [
        {"dataset": "my-dataset/", "target": "s3://my-bucket/datasets/my-dataset/"},
        {"dataset": "other-dataset/", "target": "s3://my-bucket/datasets/other-dataset/"}
      ]
    },
    "merkle": {
      "status": "ok",
      "files_checked": 8,
      "files_covered": 8,
      "files_missing": 0,
      "threshold_bytes": 104857600
    }
  },
  "overall": "fail"
}
```

---

## 11.10 Integration with `veks publish`

The `veks publish` command (§10) should run the equivalent of
`veks check` as a pre-flight gate before starting any uploads. If any
check fails, publish exits with an error and directs the user to run
`veks check` for details.

This can be bypassed with `veks publish --no-check` for cases where the
operator has already verified readiness or wants to force a partial
publish.

```
$ veks publish
Error: pre-flight checks failed (1 pipeline incomplete, 2 files missing merkle)
Run 'veks check' for details, or use '--no-check' to override.
```

---

## 11.11 Future Check Categories

The `--check-*` pattern is extensible. Anticipated future additions:

| Potential flag | Description |
|---------------|-------------|
| `--check-profiles` | Verify that all declared profiles resolve to existing files |
| `--check-credentials` | Test S3 write access to the configured bucket |
| `--check-disk` | Verify sufficient local disk space for pipeline scratch |

These are listed for design context only and are not part of the initial
implementation.

---

## 11.12 CLI Registration

`veks check` is a top-level subcommand alongside `run`, `publish`,
`datasets`, etc. It is **not** a pipeline command — it does not
participate in the `CommandOp` / `CommandRegistry` / `dataset.yaml`
step machinery. It is a standalone operational tool.

```rust
enum Commands {
    // ... existing ...
    /// Pre-flight checks for dataset readiness
    Check(CheckArgs),
}
```
