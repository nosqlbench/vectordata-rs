# analyze explain-partitions

Trace how a query's predicate creates a partition and how KNN results
map between global and partition ordinal spaces.

## Usage

```bash
veks pipeline analyze explain-partitions --ordinal <n> [--label <value>] [--limit 20]
```

Auto-resolves all file paths from `dataset.yaml`.

## Example

```bash
veks pipeline analyze explain-partitions --ordinal 0 --limit 8
```

```
═══ Query 0 — Partition Trace (label=9) ══════════════════

┌─ Stage 1: Predicate ─────────────────────────────────────
│  query[0] predicate: field_0 == 9
│
├─ Stage 2: Partition Membership ──────────────────────────
│  81 of 1000 base vectors have label=9 (8.10%)
│  global ordinals: [0, 4, 24, 36, 39, 48, 53, 67, ... (73 more)]
│
├─ Stage 3: Ordinal Remapping ─────────────────────────────
│    global[     0] → partition[   0]
│    global[     4] → partition[   1]
│    ...
│
├─ Stage 4: Global KNN (default profile) ─────────────────
│  k=100 neighbors for query 0
│    [  0] global    288  ·        ← not in partition
│    [  5] global     26  ·
│    ...
│  7 of 100 global neighbors are in this partition (7.0%)
│
├─ Stage 5: Partition KNN (profile: label-9) ─────────────
│  81 real neighbors
│    [  0] partition[  57] → global[   780]  dist=69.08
│    [  1] partition[  33] → global[   521]  dist=69.76
│    ...
│
└─ Stage 6: Partition vs Filtered KNN ────────────────────
   filtered KNN:    81 neighbors (global ordinals)
   partition KNN:   81 neighbors (remapped to global)
   intersection:    81 neighbors
   overlap:         100.0%
   avg |rank shift|: 0.0
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--ordinal` | yes | Query ordinal to trace |
| `--label` | no | Label to trace (default: query's predicate value) |
| `--profile` | no | Profile for global KNN resolution (default: "default") |
| `--prefix` | no | Partition profile name prefix (default: "label") |
| `--limit` | no | Max entries per stage (default: 20) |

## Prerequisites

Partition profiles must exist — run `compute partition-profiles` first.
