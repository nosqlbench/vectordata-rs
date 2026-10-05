# Wire Format Specification

## Overview

A slabtastic file is a sequence of **pages** followed by a trailing
**pages page** (the index) or a **namespaces page** (the multi-namespace
index). All multi-byte integers are **little-endian**. File-level offsets
are **twos-complement signed 8-byte integers** (i64).

Files may be up to **2^63 bytes**.

The conventional file extension is **`.slab`**.

## File layout

### Single-namespace file

```text
┌─────────────┐
│  Data Page 0 │  ← file offset 0
├─────────────┤
│  Data Page 1 │
├─────────────┤
│     ...      │
├─────────────┤
│  Data Page N │
├─────────────┤
│  Pages Page  │  ← always last; page_type = Pages
└─────────────┘
```

### Multi-namespace file

```text
┌──────────────────┐
│  Data Pages (ns1) │
├──────────────────┤
│  Pages Page (ns1) │
├──────────────────┤
│  Data Pages (ns2) │
├──────────────────┤
│  Pages Page (ns2) │
├──────────────────┤
│  Namespaces Page  │  ← always last; page_type = Namespaces
└──────────────────┘
```

A valid slabtastic file **must** end with either a pages page (single
namespace) or a namespaces page (multiple namespaces). A file that does
not end in one of these two page types is invalid.

## Reading entry point

1. Read the last 16 bytes of the file — this is the terminal page footer.
2. Check `page_type`:
   - **Pages (1)** — single-namespace file. The pages page is the terminal
     page. Compute `pages_page_offset = file_length - footer.page_size`
     and read the full pages page.
   - **Namespaces (3)** — multi-namespace file. The namespaces page is the
     terminal page. Read it, find the default namespace entry (index 1,
     name length 0), and follow its `pages_page_offset` to locate the
     default namespace's pages page.
3. Parse the pages page entries to build an ordinal-to-offset index.

## All-integer encoding

| Width | Encoding | Usage |
|-------|----------|-------|
| 1 byte | unsigned | page_type, namespace_index |
| 2 bytes | unsigned LE | footer_length |
| 3 bytes | unsigned LE | record_count (in footer) |
| 4 bytes | unsigned LE | page_size (header and footer), record offsets |
| 5 bytes | signed LE (sign-extended) | start_ordinal (in footer) |
| 8 bytes | signed LE | file offsets, ordinals (in pages page entries) |

See also: [Page Layout](page-layout.md), [Footer Format](footer-format.md),
[Pages Page](pages-page.md), [Namespaces Page](namespaces-page.md).
