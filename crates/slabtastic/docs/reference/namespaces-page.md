# Namespaces Page

The namespaces page maps namespace names to indices and locates each
namespace's pages page within the file. When present, it is always the
**last page** in the file and uses `page_type = Namespaces` (3).

## When present

A namespaces page is written when a file contains multiple namespaces.
Single-namespace files end with a pages page directly and do not include a
namespaces page.

## Entry format

Each record in the namespaces page is a variable-length entry:

```text
[namespace_index:1][name_length:1][name_bytes:N][pages_page_offset:8]
```

| Field | Width | Encoding |
|-------|-------|----------|
| `namespace_index` | 1 byte | unsigned; must match data page footer indices |
| `name_length` | 1 byte | unsigned; UTF-8 name length (0 for default, max 128) |
| `name_bytes` | N bytes | UTF-8 encoded namespace name |
| `pages_page_offset` | 8 bytes | signed LE (i64); file offset of the namespace's pages page |

## Namespace index values

| Range | Meaning |
|-------|---------|
| 0 | Invalid / reserved |
| 1 | Default namespace `""` (always present) |
| 2–127 | User-defined namespaces |
| 128–255 | Reserved — rejected by readers |

## Default namespace requirement

Every slabtastic file must have a default namespace (index 1, empty
name). When a reader opens a file ending with a namespaces page, it
searches the entries for `namespace_index == 1` with `name_length == 0`
and follows its `pages_page_offset` to locate the default namespace's
pages page. If no default namespace entry is found, the file is rejected.

## Reading entry point

1. Read the last 16 bytes — the namespaces page footer.
2. Verify `page_type == Namespaces`.
3. Compute `ns_page_offset = file_length - footer.page_size`.
4. Read and deserialize the namespaces page.
5. Find the default namespace entry (index 1, name length 0).
6. Read the pages page at the entry's `pages_page_offset`.
7. Parse the pages page entries to build the ordinal-to-offset index.

## Example

A file with two namespaces — `""` (default) and `"vectors"`:

```text
Entry 0: index=1, name="",        pages_page_offset=4096
Entry 1: index=2, name="vectors", pages_page_offset=8192
```

See also: [Wire Format](wire-format.md), [Pages Page](pages-page.md),
[Footer Format](footer-format.md).
