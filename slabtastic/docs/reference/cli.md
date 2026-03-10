# CLI Reference

The `slab` binary provides file maintenance commands.

## Synopsis

```
slab <COMMAND> [OPTIONS]
```

## Commands

### `slab analyze <FILE>`

Display file structure and statistics: page layout, record size
statistics (min/avg/max/histogram), page size statistics, page
utilization, ordinal monotonicity analysis, and detected content type.

Statistics are computed by sampling — by default 1000 records or 1%,
whichever is smaller.

| Flag | Description |
|------|-------------|
| `--samples <N>` | Number of records to sample for statistics |
| `--sample-percent <PCT>` | Percentage of records to sample (0.0–100.0) |
| `--namespace <NAME>` | Operate on a specific namespace |

### `slab check <FILE>`

Check a slabtastic file for structural errors. Performs three passes:

1. **Index-driven** — reads each data page referenced by the pages page,
   validates magic, page type, namespace index, footer length, page size
   minimum, record count, offset bounds, and ordinal monotonicity.
2. **Forward traversal** — walks from offset 0 using header page_size
   fields, validates each page structurally without relying on the index.
3. **Cross-check** — verifies every index entry appears in the forward
   traversal and vice versa.

| Flag | Description |
|------|-------------|
| `--namespace <NAME>` | Check a specific namespace |

### `slab get <FILE> <ORDINALS...>`

Retrieve records by ordinal and display as hex dump. Each ordinal
argument can be a plain integer or an ordinal range specifier.

**Ordinal range specifiers:**

| Form | Meaning | Example |
|------|---------|---------|
| `n` | First n ordinals `[0, n)` | `10` → ordinals 0–9 |
| `m..n` | Closed interval `[m, n]` | `5..10` → ordinals 5–10 |
| `[m,n)` | Half-open | `[5,10)` → ordinals 5–9 |
| `[m,n]` | Closed | `[5,10]` → ordinals 5–10 |
| `(m,n)` | Open | `(5,10)` → ordinals 6–9 |
| `(m,n]` | Half-open (left exclusive) | `(5,10]` → ordinals 6–10 |
| `[n]` | Single ordinal | `[42]` → ordinal 42 |
| `[m..n)` | Half-open with `..` | `[5..10)` → ordinals 5–9 |

| Flag | Description |
|------|-------------|
| `--raw` | Output raw bytes instead of hex dump |
| `--as-hex` | Output bytes as space-separated hex (e.g. `48 65 6c 6c 6f`) |
| `--as-base64` | Output bytes as base64 (standard alphabet, with padding) |
| `--namespace <NAME>` | Read from a specific namespace |

### `slab append <FILE>`

Append newline-delimited records to an existing slabtastic file. Reads
from stdin by default. The file is verified for structural integrity
before appending (equivalent to `slab check`).

| Flag | Description |
|------|-------------|
| `--source <PATH>` | Read records from a file instead of stdin |
| `--preferred-page-size <N>` | Preferred page size for new pages (bytes) |
| `--min-page-size <N>` | Minimum page size (bytes, >= 512) |
| `--page-alignment` | Pad new pages to multiples of min_page_size |
| `--progress` | Show progress on stderr |
| `--namespace <NAME>` | Append to a specific namespace |

### `slab import <FILE> <SOURCE>`

Import records from an external file format into a slab file. If the
target file exists, records are appended; otherwise a new file is created.

Format auto-detection uses the source file extension:

| Extension | Format |
|-----------|--------|
| `.slab` | Slabtastic slab format |
| `.json` | JSON (stream of objects) |
| `.jsonl`, `.ndjson` | JSONL (newline-delimited JSON) |
| `.csv` | CSV (comma-separated values) |
| `.tsv` | TSV (tab-separated values) |
| `.yaml`, `.yml` | YAML (documents separated by `---`) |
| other | Binary content scan (text vs null-terminated) |

Delimiters are preserved in the record data so that concatenating
exported records reproduces the original file.

| Flag | Description |
|------|-------------|
| `--newline-terminated-records` | Force newline-delimited text format |
| `--null-terminated-records` | Force null-terminated binary format |
| `--slab-format` | Force slab format |
| `--json` | Force JSON format |
| `--jsonl` | Force JSONL format |
| `--csv` | Force CSV format |
| `--tsv` | Force TSV format |
| `--yaml` | Force YAML format |
| `--skip-malformed` | Skip records that fail to parse instead of aborting |
| `--strip-newline` | Strip trailing newlines from records before storing |
| `--preferred-page-size <N>` | Preferred page size (bytes) |
| `--min-page-size <N>` | Minimum page size (bytes) |
| `--page-alignment` | Enable page alignment |
| `--progress` | Show progress on stderr |
| `--namespace <NAME>` | Import into a specific namespace |

### `slab export <FILE>`

Export records from a slab file to an external format. Output goes to
stdout by default, or to a file with `--output`.

| Flag | Description |
|------|-------------|
| `--output <PATH>` | Write output to a file (stdout if omitted) |
| `--text` | Export as newline-delimited text |
| `--cstrings` | Export as null-terminated binary |
| `--slab-format` | Export as slab format (requires `--output`) |
| `--as-is` | Output records exactly as stored, without adding missing newlines or delimiters |
| `--range <SPEC>` | Ordinal range to export (e.g. `100`, `[5,10)`) |
| `--preferred-page-size <N>` | Preferred page size (for slab output) |
| `--min-page-size <N>` | Minimum page size (for slab output) |
| `--page-alignment` | Enable page alignment (for slab output) |
| `--progress` | Show progress on stderr |
| `--namespace <NAME>` | Export from a specific namespace |

When no format flag is given, the output format is auto-detected from the
`--output` extension (`.slab` → slab, everything else → text). By default,
text mode adds a missing trailing newline to each record
(`--add-missing-newline` behavior). Use `--as-is` to disable this.

### `slab explain <FILE>`

Display a block diagram of the page layout for selected pages. When no
filtering options are given, all pages (data pages and the terminal pages
page) are shown. Each page is rendered as a box-drawing diagram showing
the header, records, offset array, and footer fields.

| Flag | Description |
|------|-------------|
| `--pages <N>...` | Show specific page indices (0-based) |
| `--namespace <NAME>` | Show only pages belonging to a namespace |
| `--ordinals <SPEC>` | Show pages overlapping an ordinal range |

### `slab namespaces <FILE>`

List all namespaces in a slab file. For single-namespace files (ending
with a pages page), reports just the default namespace. For
multi-namespace files (ending with a namespaces page), lists all
namespace entries with their index, name, and pages page offset.

### `slab completions <SHELL>`

Generate shell completion scripts for the specified shell. Supported
shells: `bash`, `zsh`, `fish`, `elvish`, `powershell`.

```bash
# Generate bash completions
slab completions bash > /etc/bash_completion.d/slab

# Generate zsh completions
slab completions zsh > ~/.zfunc/_slab
```

### `slab rewrite <INPUT> <OUTPUT>`

Rewrite a slabtastic file into a new file, reordering records by ordinal
and repacking to new page settings in a single operation. Eliminates
logically deleted pages and alignment padding waste.

| Flag | Description |
|------|-------------|
| `--preferred-page-size <N>` | Preferred page size (bytes) |
| `--min-page-size <N>` | Minimum page size (bytes) |
| `--page-alignment` | Enable page alignment |
| `--progress` | Show progress on stderr |
| `--namespace <NAME>` | Rewrite a specific namespace |
| `--range <SPEC>` | Ordinal range to rewrite (e.g. `100`, `[5,10)`) |
