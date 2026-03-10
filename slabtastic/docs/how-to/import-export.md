# How to Import and Export Data

## Importing data

### From structured formats

Use `slab import` to bring data from JSON, JSONL, CSV, TSV, or YAML
files into a slab file. The format is auto-detected from the file
extension:

```bash
# JSON — stream of objects with whitespace between them
slab import data.slab source.json

# JSONL — one JSON object per line
slab import data.slab logs.jsonl

# CSV / TSV
slab import data.slab table.csv
slab import data.slab table.tsv

# YAML — documents separated by ---
slab import data.slab config.yaml
```

Each imported record preserves its delimiter (newline, null byte, etc.)
so that exporting and concatenating records reproduces the original file.

### From text files

Newline-delimited text is the default for files without a recognized
extension:

```bash
slab import data.slab records.txt
```

Each line (including its trailing `\n`) becomes one record.

### From another slab file

```bash
slab import data.slab other.slab
```

Records are copied verbatim from the source slab.

### Overriding auto-detection

If the file extension is misleading, use an explicit format flag:

```bash
slab import data.slab myfile.dat --json
slab import data.slab myfile.dat --csv
slab import data.slab myfile.dat --null-terminated-records
```

### Import into a new or existing file

If the target slab file exists, records are appended. If it does not
exist, a new file is created. Page configuration flags work the same as
other write commands:

```bash
slab import data.slab source.json \
    --preferred-page-size 4096 \
    --page-alignment
```

## Exporting data

### To text

```bash
# To stdout
slab export data.slab

# To a file
slab export data.slab --output records.txt
```

Records that already end with `\n` are written as-is; others get a
trailing newline appended.

### To null-terminated binary

```bash
slab export data.slab --cstrings --output records.bin
```

### To another slab file

```bash
slab export data.slab --output copy.slab
```

The output format is auto-detected from the extension (`.slab` → slab
format). You can also force it with `--slab-format`.

### Exporting a subset

Use `--range` to export only a range of ordinals:

```bash
slab export data.slab --range "[0,100)" --output first100.txt
```

## Format details

### JSON

Each JSON object in the source becomes one record (compact JSON + `\n`).
Objects are parsed with `serde_json::StreamDeserializer`, so whitespace
between objects is accepted.

### JSONL

Each non-empty line must be valid JSON. Lines are stored including their
trailing `\n`.

### CSV / TSV

All rows (including the header row) become records. Each row is
reconstructed with the original delimiter and a trailing `\n`.

### YAML

Documents separated by `---` are stored individually. Each document is
validated as YAML before import. A trailing document without a final
`---` separator is also accepted.
