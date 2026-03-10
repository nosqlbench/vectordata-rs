# How to Append Data to an Existing File

## Using the library

Open the file in append mode with `SlabWriter::append`, add records, and
call `finish()`:

```rust
use slabtastic::{SlabWriter, WriterConfig};

fn append_records(path: &str) -> slabtastic::Result<()> {
    let config = WriterConfig::default();
    let mut writer = SlabWriter::append(path, config)?;

    writer.add_record(b"new record 1")?;
    writer.add_record(b"new record 2")?;
    writer.finish()?;
    Ok(())
}
```

`SlabWriter::append`:

1. Opens the existing file for read+write.
2. Reads the trailing pages page to discover existing data pages.
3. Positions the write cursor after the last data page (before the old
   pages page).
4. New data pages are written followed by a new pages page that references
   both old and new pages.

Ordinals continue from where the previous writer left off — if the file
had ordinals 0..99, the next record appended will be ordinal 100.

## Using the CLI

Pipe newline-delimited records from stdin:

```bash
echo -e "line one\nline two\nline three" | slab append data.slab
```

Or read from a source file:

```bash
slab append data.slab --source records.txt
```

Optional flags:

- `--preferred-page-size 4096` — override the page size for new pages
- `--min-page-size 512` — minimum page size
- `--page-alignment` — pad new pages to min_page_size multiples

## Notes

- The CLI `append` command verifies file integrity before appending
  (equivalent to running `slab check` first). If the file is malformed,
  the append is rejected.
- The original data pages are never modified. This is a strictly
  append-only operation.
- The old pages page becomes logically dead once the new one is written.
- If the append is interrupted before `finish()` writes the new pages
  page, the file remains valid with only its original data (the old pages
  page is still intact at the former end of file).
- Appending works with both single-namespace files (ending with a pages
  page) and multi-namespace files (ending with a namespaces page). In the
  latter case, the writer locates the default namespace's pages page via
  the namespaces page.
