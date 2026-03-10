# Page Layout

Every page — data page, pages page, or namespaces page — uses the same
wire layout:

```text
[header][records][offsets][footer]
```

## Header (8 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | magic | ASCII `SLAB` (`0x534C4142`) |
| 4 | 4 | page_size | Total page size in bytes (u32 LE) |

The `page_size` field serves as a forward reference to the page footer and
enables forward traversal without consulting the index.

## Record data

Records are packed contiguously starting at byte 8 (immediately after the
header). Records have no internal structure from the format's perspective —
they are opaque byte sequences. The offset array defines record boundaries.

## Offset array ((record_count + 1) * 4 bytes)

The offset array sits immediately before the footer. It contains
`record_count + 1` little-endian u32 values, each measuring a byte position
from the **start of the page**.

- `offsets[0]` = start of the first record (always 8, the header size)
- `offsets[i]` = start of record `i`
- `offsets[record_count]` = end of the last record (sentinel)

Record `i` spans bytes `offsets[i]..offsets[i+1]`.

### Locating the offset array

From the end of the page:

1. Back up `footer_length` bytes to find the footer start.
2. Back up `4 * (record_count + 1)` more bytes to find `offsets[0]`.

## Footer

The footer occupies the last `footer_length` bytes of the page. See
[Footer Format](footer-format.md) for field details.

The `page_size` in the header and footer **must** be equal. This invariant
enables bidirectional traversal: forward via the header, backward via the
footer.

## Size constraints

| Limit | Value | Rationale |
|-------|-------|-----------|
| Minimum page size | 512 bytes (2^9) | Large enough for header + 1 offset + footer |
| Maximum page size | 2^32 bytes | Fits in 4-byte page_size field |

## Alignment

When page alignment is enabled in the writer configuration, pages are
padded with zero bytes to the next multiple of `min_page_size`. The offset
array and footer are relocated to the **end** of the padded page; the
padding fills the space between the last record and the offset array.
