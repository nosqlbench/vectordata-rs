# analyze explain-predicates

Trace a predicate through matching metadata with selectivity analysis.

## Usage

```bash
veks pipeline analyze explain-predicates --ordinal <n> [--profile default] [--limit 20]
```

Auto-resolves file paths from `dataset.yaml` when run in a dataset directory.

## Example

```bash
veks pipeline analyze explain-predicates --ordinal 0 --limit 5
```

```
── predicate [ordinal 0] ──────────────────────────────────
field_0 == 9

── matching metadata: 81 of 1000 records (selectivity 0.0810 = 8.10%) ──
  [0] metadata ordinal 0: field_0 = 9
  [1] metadata ordinal 4: field_0 = 9
  [2] metadata ordinal 24: field_0 = 9
  [3] metadata ordinal 36: field_0 = 9
  [4] metadata ordinal 39: field_0 = 9

  ... and 76 more (use limit= to show more)
```
