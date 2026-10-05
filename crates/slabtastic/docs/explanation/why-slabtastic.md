# Why Slabtastic?

## The problem

We needed a format for organising non-uniform data by ordinal that
supports:

1. **Random access** — fetch any record by ordinal in O(1) expected time.
2. **Streaming reads** — iterate all records sequentially without loading
   the entire file.
3. **Append-only writes** — add new data without rewriting existing pages.
4. **Single-file deployment** — no sidecar index files or directories.
5. **Minimal dependencies** — ideally zero beyond the standard library.

## Alternatives considered

### Apache Arrow

Arrow's paged buffer layouts support fast access and have a modest ~15 MB
dependency footprint. However, Arrow requires rewriting the entire file
from scratch to append data. For incremental write workloads, this is
prohibitive.

### Direct I/O with offset table

Direct I/O with an external offset index is efficient but requires
managing a separate file for the index. A two-file format complicates
deployment and makes atomic operations harder.

### SQLite

SQLite is mature and supports random access, but its streaming bulk
append story requires WAL mode, which takes you back to managing a
directory of files. Single-file simplicity is lost.

## The slabtastic approach

Slabtastic keeps the index **inside** the file as the trailing pages
are added. New data pages can be appended and a fresh pages page written
without modifying any existing page. The format stays close to the metal:

- The file is memory-mapped at open time; point gets issue zero syscalls.
- Interpolation search over the page index finds the target page in
  O(1) expected probes (1–2 for uniform distributions).
- Zero-copy `get_ref()` returns a `&[u8]` directly into the mmap.
- Pages are self-describing (header + footer carry all metadata).
- Record offsets are flat arrays — no pointer chasing.
- Forward and backward traversal are both possible without the index.
- Page alignment is optional for block-store-friendly layouts.

The trade-off is that the format is append-only at the page level.
Fine-grained record updates require either in-place mutation (for
fixed-size or self-terminating records) or appending a replacement page.
