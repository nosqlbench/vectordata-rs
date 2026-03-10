# Footer Format

## v1 Footer (16 bytes)

```text
Byte   Field              Width   Encoding
─────  ─────────────────  ──────  ─────────────────────────────────────────────
0–4    start_ordinal      5       signed LE, sign-extended to i64
5–7    record_count       3       unsigned LE (max 2^24 − 1)
8–11   page_size          4       unsigned LE (512 .. 2^32)
12     page_type          1       enum: 0=Invalid, 1=Pages, 2=Data, 3=Namespaces
13     namespace_index    1       unsigned (0=invalid, 1=default, 2–127=user)
14–15  footer_length      2       unsigned LE (>= 16, multiple of 16)
```

## Field details

### start_ordinal (5 bytes)

The ordinal of the first record in this page. Encoded as the low 5 bytes
of a twos-complement i64 value. On read, bit 39 is sign-extended into
bytes 5–7 to reconstruct the full i64. Range: ±2^39 (approximately
±549 billion).

### record_count (3 bytes)

The number of records in this page. Maximum: 2^24 − 1 = 16,777,215.

### page_size (4 bytes)

The total size of the page in bytes, including header, record data,
offset array, and footer. This **must** match the `page_size` field in
the header.

### page_type (1 byte)

| Value | Variant | Meaning |
|-------|---------|---------|
| 0 | Invalid | Sentinel; rejected during deserialization |
| 1 | Pages | Pages page (the file-level index) |
| 2 | Data | Data page (holds user records) |
| 3 | Namespaces | Namespaces page (maps namespace names to indices) |

Page types implicitly carry their format version — types 1, 2, and 3 are
all v1-era types. Future revisions will introduce new page type values
rather than incrementing a separate version field.

### namespace_index (1 byte)

Identifies which namespace this page belongs to. In pre-namespace files
this byte was called `version` and was always `1`. Since namespace index
`1` is the default namespace `""`, existing v1 files are backward
compatible without migration.

| Range | Meaning |
|-------|---------|
| 0 | Invalid / reserved — always rejected |
| 1 | Default namespace `""` |
| 2–127 | User-defined namespaces |
| 128–255 | Reserved (negative when interpreted as signed) — rejected |

Readers must reject namespace indices of 0 or >= 128.

### footer_length (2 bytes)

The total footer length in bytes. Must be at least 16 and a multiple of
16. This field enables future footer versions to extend the footer without
breaking readers that only understand v1.

## Backward compatibility

In the original v1 format, byte 13 was documented as `version` with a
fixed value of `1`. The reinterpretation as `namespace_index` is backward
compatible because namespace index `1` corresponds to the default
namespace, which is the only namespace in a single-namespace file.

## Future versions

Checksums are deferred to a future format version. Later versions may
extend the footer beyond 16 bytes by increasing `footer_length`.
Compatibility with previous readers should not be broken without explicit
user opt-in.
