# Error Catalogue

All fallible library functions return `Result<T, SlabError>`.

| Variant | Meaning | Common cause |
|---------|---------|--------------|
| `InvalidMagic` | First 4 bytes are not `SLAB` | Corrupt file, wrong file type |
| `InvalidNamespaceIndex(u8)` | Namespace index is invalid (0 or >= 128) | Corruption, reserved index |
| `InvalidPageType(u8)` | Page type byte is not 0, 1, 2, or 3 | Corruption, invalid file |
| `PageSizeMismatch { header, footer }` | Header and footer page_size differ | Truncation, in-place corruption |
| `PageTooSmall(u32)` | Configured page size < 512 | Invalid `WriterConfig` |
| `PageTooLarge(u64)` | Configured page size > 2^32 | Invalid `WriterConfig` |
| `RecordTooLarge { record_size, max_size }` | Single record exceeds page capacity | Record too big for `max_page_size` |
| `OrdinalNotFound(i64)` | Requested ordinal is not in the file | Sparse gap, out-of-range lookup |
| `OrdinalMismatch { expected, actual }` | Caller-specified ordinal does not match writer's next ordinal | Wrong ordinal in `add_record_at` / `add_records_at` |
| `InvalidFooter(String)` | Footer data is malformed | Corruption, bad footer_length |
| `TruncatedPage { expected, actual }` | Page data is incomplete | Truncated file, partial write |
| `Io(io::Error)` | Underlying I/O error | File not found, permission denied |
| `WithContext { source, file_offset, page_index, ordinal }` | Wraps another `SlabError` with positional context | Errors during page reads, ordinal lookups |
