# Append-Only Semantics

## How appending works

A slabtastic file always ends with a pages page (the index) or a
namespaces page (for multi-namespace files). Appending new data follows
this sequence:

1. Read the existing pages page to discover existing data pages.
2. Position the write cursor after the last data page (just before the
   old pages page).
3. Write new data pages.
4. Write a **new** pages page that references both old and new data pages.

The old pages page is never modified — it is simply superseded by the new
one at the end of the file. This is a key safety property: if the append
is interrupted before the new pages page is written, the file remains
valid with only its original data.

## Logical deletion

When a new pages page is written, it references only the "live" data
pages. Any data pages not referenced become logically deleted. The bytes
remain on disk but are never consulted by readers.

This enables several patterns:

- **Append new data** — new pages are added, old pages are preserved.
- **Replace a page** — write a new data page with the same ordinal range,
  then write a pages page that references the new page instead of the old
  one.
- **Compact / rewrite** — `slab rewrite` reads all live records, sorts
  them by ordinal, and writes them to a new file, eliminating dead pages
  and wasted alignment padding.

## Ordinal continuity

When appending via `SlabWriter::append`, ordinals continue from where the
previous writer left off. If the file contained ordinals 0..99, the next
appended record will be ordinal 100.

## Writers must flush at page boundaries

Writers are required to flush buffers at page boundaries — the writer only
issues writes of complete, serialized page buffers. This does **not**
guarantee OS-level atomicity (a concurrent reader may see a
partially-written page), but it means:

- The writer never leaves a half-serialized page in its own buffers.
- Concurrent readers can use the `[magic][size]` header to determine
  whether a page is fully written before reading the body.
- The file is structurally valid up to the last complete pages page, even
  if a crash interrupts a subsequent write.

## Multiple append cycles

After multiple append cycles, the file may contain several logically-dead
pages pages interspersed between data pages. Only the **last** pages page
is authoritative. Use `slab rewrite` to eliminate the dead pages if file
size is a concern.
