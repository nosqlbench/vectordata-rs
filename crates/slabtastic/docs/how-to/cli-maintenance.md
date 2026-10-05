# How to Use the CLI for File Maintenance

The `slab` binary provides subcommands for inspecting, validating,
importing, exporting, and transforming slabtastic files.

## Inspect a file

```bash
slab analyze data.slab
```

Displays: page layout, record size statistics (min/avg/max/histogram),
page size statistics, page utilization, ordinal monotonicity analysis,
and detected content type. Statistics are computed by sampling.

Override sampling with `--samples` or `--sample-percent`:

```bash
slab analyze data.slab --samples 5000
slab analyze data.slab --sample-percent 10.0
```

## Check file integrity

```bash
slab check data.slab
```

Performs three validation passes: index-driven page inspection, forward
traversal, and cross-check of index against traversal.

## Retrieve records

```bash
# Human-readable hex dump
slab get data.slab 0 42 99

# Ordinal range specifiers
slab get data.slab [0,10)
slab get data.slab 5..10
slab get data.slab [42]

# Raw binary output (e.g. pipe to another tool)
slab get data.slab 0 --raw > record0.bin

# Hex output (space-separated bytes)
slab get data.slab 0 --as-hex

# Base64 output
slab get data.slab 0 --as-base64
```

## Append records

```bash
# From stdin (newline-delimited)
echo -e "new record 1\nnew record 2" | slab append data.slab

# From a file
slab append data.slab --source records.txt

# With custom page config
slab append data.slab --source records.txt \
    --preferred-page-size 4096 \
    --min-page-size 512 \
    --page-alignment
```

The file is verified for integrity before appending (same as `check`).

## Import data from external formats

```bash
# Auto-detect format from extension
slab import data.slab source.json
slab import data.slab logs.jsonl
slab import data.slab table.csv
slab import data.slab table.tsv
slab import data.slab config.yaml

# Force a specific format
slab import data.slab myfile.dat --json
slab import data.slab myfile.dat --null-terminated-records

# From another slab file
slab import data.slab other.slab
```

See [Import and Export Data](import-export.md) for format details.

## Export data

```bash
# Text to stdout
slab export data.slab

# Text to a file
slab export data.slab --output records.txt

# Null-terminated binary
slab export data.slab --format cstrings --output records.bin

# To another slab file
slab export data.slab --output copy.slab
```

## List namespaces

```bash
slab namespaces data.slab
```

Reports the namespaces in a file: for single-namespace files, shows just
the default namespace; for multi-namespace files, lists all entries with
index, name, and pages page offset.

## Explain file layout

Display a block diagram of each page showing header, records, offsets,
and footer fields:

```bash
# Show all pages
slab explain data.slab

# Show only page 0
slab explain data.slab --pages 0

# Show pages overlapping ordinals [0,100)
slab explain data.slab --ordinals "[0,100)"
```

## Rewrite a file

Rewrite a file to new page settings, reordering records by ordinal and
eliminating logically deleted pages and padding waste:

```bash
slab rewrite input.slab output.slab \
    --preferred-page-size 65536 \
    --page-alignment
```
