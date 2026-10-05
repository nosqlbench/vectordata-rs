# Pages Page (Index)

The pages page is the file-level index. In a single-namespace file it is
the **last page**; in a multi-namespace file each namespace has its own
pages page, and the file ends with a namespaces page that locates them.
The pages page uses the standard page layout with `page_type = Pages`.

## Entry format

Each record in the pages page is a 16-byte tuple:

```text
[start_ordinal:8][file_offset:8]   (little-endian signed i64)
```

- `start_ordinal` — the first ordinal of the referenced data page.
- `file_offset` — the byte offset of the data page within the file.

## Ordering

Entries are sorted by `start_ordinal` to enable fast lookup of any ordinal
to its containing data page. The reader uses interpolation search (O(1)
expected for uniform distributions) with a binary-search fallback.

The data pages themselves are **not** required to appear in monotonic
file-offset order. After append operations, newer pages may reference
ordinal ranges that logically precede older pages on disk.

## Single-page constraint

The pages page must fit in a single page. This puts a hard upper bound on
the number of data pages in a v1 file:

```text
max_entries = (max_page_size - header - footer) / 16
```

With default `max_page_size = 2^32`, this allows over 268 million page
entries.

## Logical deletion

Data pages not referenced by the pages page are **logically deleted** and
must not be used by readers. This happens naturally in append-only mode:
when a new pages page is written, only the pages it references are live.

## Authoritative last page

A valid slabtastic file always ends with either a pages page (single
namespace) or a namespaces page (multiple namespaces). The **last**
terminal page in the file is authoritative. Earlier pages pages (from
prior append cycles) are logically dead — they remain on disk but are
never consulted.

See also: [Namespaces Page](namespaces-page.md).

## Lookup algorithm

To find the page containing ordinal `o`:

1. **Interpolation search** the entries for the largest
   `start_ordinal <= o`. For uniform ordinal distributions (the common
   case) this finds the page in 1–2 probes. A bounded linear scan
   (±4 entries) handles minor non-uniformity; if the scan misses, a
   standard binary search fallback runs in O(log₂ n).
2. If no such entry exists, the ordinal is in a gap (sparse) — return
   `OrdinalNotFound`.
3. Look up the record via cached per-page metadata and the mmap (zero
   syscalls, zero allocation with `get_ref`).
4. Compute `local_index = o - start_ordinal`. If `local_index >=
   record_count`, the ordinal falls past the end of this page — return
   `OrdinalNotFound`.
