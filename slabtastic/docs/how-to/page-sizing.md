# How to Configure Page Sizing and Alignment

## WriterConfig parameters

| Parameter            | Range          | Default    | Purpose                                          |
|----------------------|----------------|------------|--------------------------------------------------|
| `min_page_size`      | 512 .. max     | 512        | Floor for page size; alignment boundary           |
| `preferred_page_size`| min .. max     | 4,194,304  | Flush threshold — pages are written when they reach this size |
| `max_page_size`      | preferred .. 2^32 | 2^32 - 1 | Hard ceiling; records exceeding this are rejected |
| `page_alignment`     | bool           | false      | Pad pages to multiples of `min_page_size`         |

Constraints: `min_page_size <= preferred_page_size <= max_page_size`,
and `min_page_size >= 512`.

## Choosing page sizes

**Small records (< 100 bytes):**
A larger preferred page size (64 KiB–256 KiB) amortizes per-page overhead
across many records.

**Large records (> 10 KiB):**
A larger preferred page size avoids single-record pages. If individual
records approach the preferred size, increase `max_page_size` to
accommodate them.

**Block-store alignment:**
Enable `page_alignment` and set `min_page_size` to your storage block size
(e.g. 4096 for typical SSDs) so that pages align to block boundaries:

```rust
use slabtastic::WriterConfig;

let config = WriterConfig::new(
    4096,       // min_page_size — alignment boundary
    65536,      // preferred_page_size
    u32::MAX,   // max_page_size
    true,       // page_alignment enabled
).unwrap();
```

When alignment is enabled, pages are padded to the next multiple of
`min_page_size`. The padding bytes are zeroed and the offset array and
footer are relocated to the end of the padded page.

## Record-too-large errors

In v1, a single record that cannot fit in a page of `max_page_size` bytes
(accounting for the 8-byte header, offset array, and 16-byte footer) is
rejected with `SlabError::RecordTooLarge`. There is no multi-page spanning
for individual records.
