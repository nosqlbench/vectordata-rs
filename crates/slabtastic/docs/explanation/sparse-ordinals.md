# Sparse Ordinals and Interior Mutation

## Sparse ordinals

Slabtastic supports **coarse chunk-level sparsity**. Ordinal ranges need
not be contiguous between pages — a file may have:

```text
Page 0: ordinals 0–99
Page 1: ordinals 200–299
         (ordinals 100–199 do not exist)
```

This is not fine-grained per-ordinal sparsity; it is a property of page
boundaries. Gaps between pages are natural when:

- Data is incrementally updated by appending new pages that cover
  non-adjacent ordinal ranges.
- Certain ordinal ranges are logically deleted by writing a new pages page
  that omits their data pages.

### API behaviour

When a reader encounters a gap:

- `SlabReader::get(ordinal)` returns `SlabError::OrdinalNotFound`.
- `SlabReader::contains(ordinal)` returns `false`.
- `SlabReader::iter()` and `batch_iter()` silently skip gaps, yielding
  only ordinals that exist.

The API does **not** provide a default-value fallback. Callers that need
default values must implement that logic themselves.

## Interior mutation

While slabtastic is primarily an append-only format, limited interior
mutation is possible:

### In-place mutation

A record can be overwritten in place if the new data fits within the
existing record's byte boundaries. This works for:

- **Fixed-size values** (e.g. 32-bit integers, 8-byte timestamps) — the
  replacement is always the same length.
- **Self-terminating formats** (e.g. null-terminated strings) — the
  replacement can be shorter than the original, with a terminator marking
  the effective end.

In-place mutation modifies the data page directly. The offset array is
unchanged since the record boundaries don't move.

### Append-based replacement

For more substantial revisions, append a new data page covering the same
ordinal range, then write a new pages page that references the replacement
page instead of the original. The old page becomes logically deleted.

This approach preserves the append-only safety model while allowing
arbitrary record changes.
