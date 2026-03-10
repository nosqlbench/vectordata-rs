# Streamable, random-accessible, appendable data layout

_SLABTASTIC_, Version 1

---

After working with several formats and IO strategies for organizing non-uniform data by ordinal, it
seems we will have to make our own.

The top contenders considered were:

1) direct io, with offset table
2) arrow, with the dual-buffer paged approach
3) sqlite, with vfs offset support

Alas, none of these seem suitable.

Arrow comes close, with a meager ~15MB dependency overhead, and paged buffer layouts which support
fast access. But it comes with the dependency cost, however meager, and the limitations of not being
able to append to data without rewriting the entire file. In other words, you must write the entire
file from scratch every time. This is very not ideal.

Direct IO is a decent fit, however it requires managing multiple buffers in order to indirect
offsets, and this makes it inherently more complicated to manage for users. Having a separate offset
index file is a no-go, but you still need to have some form of it to efficiently deal with random
access and non-uniform sizing.

Sqlite is a good fit, however, it doesn't really support streaming bulk data as appendable and
incremental unless you're talking about WAL, but then you're back to juggling a directory.

# Our format - slabtastic

Our format will keep close to the metal, support optimal chunking from block stores, and keep flat
index data and values clustered close to each other. It will do this while being relatively simple,
and still support effective random IO, streaming out (batched) and streaming in values. The only
caveats will be that streaming interfaces will need to buffer and flush on boundaries that allow
page data to be written out with local buffering and flushing cadence, but this can easily be
absorbed in the readers and writer APIs.

## basic layout

This will be a large file format, supporting files of up to 2^63 bytes. All offset pointers which
target the whole file range will be twos-complement signed 8-byte integers, to facilitate easy
interop conventional stacks with simplified data types. All offsets will be little-endian.

## Pages

The major structure will be the page, which will contain, fundamentally, a set of values and a set
of flat offsets to those values.

## Page Magic

The first 4 bytes of every page will have the UTF-8 encoding of SLAB. This will be used to identify
the file as a slabtastic file. (and every page) The next 4 bytes will be the page length, which can
serve as a forward reference to the page footer, or when the page footer would be fully written by
streaming writes. This initial 8 bytes should always be considered when doing data layout.

## Page Alignment

Page alignment to the minimum page size is an optional feature, to be configured on slabtastic
writers. When enabled, pages will always be padded out to the minimum page size, and will always be
sized to be a multiple of the minimum page size. Larger page sizes will offer better utilization for
smaller minimum page sizes in this mode.

### page structure

page layout:
`[header][records][offsets][footer]`

header:
`[magic][size]`

records are simply packed data with no known structure. The offsets fully define the beginning and
end of each record, therefore there is one more offset than records, to make indexing math simple
for every record.

footer:
`[start_ordinal:5][record_count:3][page_size:4][page_type:1][namespace_index:1][footer_length:2]`

The page footer will contain, in this order:

* starting page ordinal (5-byte signed 2s complement integer)
* number of records (3-byte unsigned integer)
* page size (4-byte int)
* page type (1-byte enum value 0->invalid 1->pages page 2->data page 3->namespaces page)
* namespace index (1-byte signed int, 0->invalid/reserved, 1->default namespace "")
* footer length (2-byte int)

Page types implicitly carry their format version. Types 1 (pages page), 2 (data page), and
3 (namespaces page) are all v1-era types. Future format revisions will introduce new page type
values rather than incrementing a version field.

The namespace index identifies which namespace a page belongs to. Index 0 is invalid/reserved.
Index 1 is always the default namespace `""`. Indices 2 through 127 are available for user-defined
namespaces. Negative indices (-128 through -1) are reserved for future use. See the "Slab
Namespaces" section for details.

The page size in the header and footer must always be equal. Checking a slabtastic file may use
these to traverse forward and backwards to verify record sizes without necessarily reading the pages
page, so long as it is focused on structure and not data as a normal user would be.

Footers are required to always be at least 16 bytes, and will be padded out to the nearest 16 bytes
in length. Checksums are deferred to a future version.

Thus, you can always read the last 16 bytes of a page to know where to find the start of the footer,
the start of the array-structured offset data, and the start of the page. And you can always read
the last 16 bytes of the file to do the same for the pages page (described below).

The v1 page footer is 16 bytes, and supports up to 2^40 in ordinal magnitude, and 2^24 in record
count per page. The beginning of the record offsets will start before the footer. The first element
location is determined by the number of records. So, from the end of the page, backup the footer
length, then -(4*(record_count + 1)). All record offsets are encoded as array-structured int offsets
from the beginning of the page, which must take account of the page header which is 4+4 bytes.

The footer format is page-specific, since you may add to a file later with an updated format. Page
type values implicitly carry their format version, so readers must verify that the page type is
recognized before reading the page. A page type of 0 is always invalid.

## Page Data & Sizing

Records in a page will grow from the beginning to the end of the page. Page sizes limits will be
governed by a simple heuristic: Always between 2^9 (512) and 2^32 bytes. This means that the minimum
page size will be 512 bytes, and the maximum will still easily fit within a single mmap call on
older Java systems which do not have AsyncFileChannel or similar capabilities.

Users will be able to govern page layout with some configuration parameters:

- minimum page size (must be 512 or higher)
- preferred page size (governs IO buffering behavior)
- maximum page size (must be 4GB or lower)

Of course, the size and variance of record lengths will inform user choices around these parameters.

Page alignment is enforced indirectly by the page size limitations and the requirement that all
pages be a multiple of 512 bytes. Users may prefer a page size which is larger than 512 bytes, but
this is an opportunistic setting.

When a single record exceeds the limits of a page, it is an error in v1.

## The pages page

The last page will be special. It is required to be a single page. It is a page map, which uses the
native page layout to store its values. The records in the page map are tuples of the beginning
ordinal in a page and the associated file offset within the whole slabtastic file. These are
required to be sorted by ordinal to facilitate Olog2(n) offset lookup.

The record structure for these tuples is as follows:
`[start_ordinal:8][offset:8]` (little-endian)

Even though these records will be fixed size, the layout of the pages page will not diverge from the
layout of other pages. The offsets will be encoded duplicitously in format v1. (Even though, with
the uniform size of the records, array based indexing could suffice).

The pages in the pages page are not required to be monotonically structured (aligned with the
monotonic structure of their starting ordinals.) Pages may actually be out of order, should some
append-only revisions be made to existing pages which can't or shouldn't be done in place.

Pages which are not referenced in the pages page are considered logically deleted, and should not be
used. This should happen naturally since only pages referenced in the indexing structure will be
included. Any other reader behavior which is not for slabtastic file maintenance is undefined and
should be considered a bug.

The single page requirement for the pages page puts a hard limit on the number of pages in a file,
and this is acceptable for v1.

Since the pages page has the normative page layout, the footer found at the end of the file is
sufficient as an entry point to map the whole of a slabtastic file. Essentially, opening a
slabtastic file starts with reading the last page, which is then asserted to be a pages page type
(type 1) or namespaces page type (type 3) via the footer, then the ordinal offsets are read to
determine the page (offset) which to jump to next for the required ordinals (and their values).
When the last page is a namespaces page, the reader follows the namespace entries to locate each
namespace's pages page, then proceeds with ordinal lookup within the appropriate namespace.

## Append-only mode

It will be possible for pages to be logically deleted without overwriting or maintaining a delete
marker. This is done by simply by not including a page in the pages page map.

This is because a page map may "update" a previous page map with itself, and as such, it should
leave out the previous page map offsets. This can allow strictly append-only mode which does not
require overwriting a page map page, since this could be a destructive operation should it fail.

The last pages page (or namespaces page) in a slabtastic file is always authoritative. A slabtastic
file which does not end in a pages page (type 1) or namespaces page (type 3) is invalid.

## Sparse Values

The slabtastic format will support sparse chunks. This means that ordinal values may not be fully
contiguous between the minimum and maximum ordinal values in a file. This is not a form of fine
sparseness by ordinal value, but more by chunk ranges. This affords step-wise changes to data in a
slabtastic file by simply appending new pages with ordinal holes. Although this is not strictly
necessary, it may be useful for some applications making large incremental changes.

To support sparse (coarse) structure, the APIs which are used to read ordinals from slabtastic files
MUST be able to signal that a requested ordinal is not present in the file. In such cases, consumer
APIs MAY allow the user to provide a default value to be returned, but only when this can be
explicitly requested by the user. (Simply setting it to an empty buffer is not acceptable).

Further while it may be presumed that the data in a slabtastic file is conventionally monotonic with
respect to its ordinal structure, this isn't guaranteed. As such, opportunistic readers MUST follow
the header and footer structure to verify a page is written fully before reading it. Further,
reading a slabtastic file in this way must be done with caution, and the assurance that it is an
immutable stream based on the usage scenario.

## Interior Mutation

While not the strong suite of slabtastic, it will be possible to mutate interior records in a page
so long as they are either editable in place, such as a self-terminating format (null terminated
string), or a fixed-size format (e.g. a 32-bit integer). More serious revision can also be achieved
with append-only mode by simply appending a new page map, thus providing the opportunity to rewrite
an existing page with a new one referenced in the new page map.

# Slabtastic CLI `slab`

A slabtastic CLI (named simply `slab`) will be the centralized tool for maintaining a slabtastic
file. The CLI module should be part of the library (not the binary) so that subcommand logic is
directly testable without running the binary.

## CLI Commands

Any command which has a '--progress' option should show the progress on stdout, updating every second or every 1M
records, which ever comes first.

When errors are given, and the offset or page number or ordinal numbers are known, either the requested ones by the user
or the actual ones where the error was found, these details should be part of the error. All errors should be specified
here in template form with such fields tokenized and then interpolated at error time. For example
`Error: invalid page type: 2` is wholly unhelpful by itself.

When commands are creating a new slab file with import or similar, the file needs to be suffixed as `.slab.buffer` and
only renamed (linked to) the end name once it is successfully fully written and then known to be complete (flushed).

Slab commands should not assume any particular set of namespaces will be present in a slab file. Instead, if there is a
namespaces page, then each or all namespaces should be supported, or the user should be prompted to choose one if
necessary.

### `slab analyze <file>` (was slab info)

Analyze a slabtastic file and give user stats and layout details:

* detected content type (use same detection logic as for import)
* statistics will be determined by sampling. Page state will be determined by sampling random
  pages. Record stats will be determined by sampling records.
    * The number of items to be sampled in either case will be 1000 or 1%, whichever is smaller.
    * The user should be able to specify `--samples` or `--sample-percent`. In either case, this
      takes the place of the rule above. The number of samples should be presented to the user
      as part of the output.
* record statistics (min,avg,max,basic histogram)
* statistics of page sizes (min,avg,max,basic histogram)
* page utilization (active bytes vs bytes available on a page) (min, avg, max, basic histogram)
* ordinal monotonicity (strictly monotonic, sparse gaps, range, etc)
    * This requires a full walk of the pages and footers to determine the monotonicity.

### `slab check <file>`

Check a slab file for errors or inconsistencies. The check performs three validation passes:

1. **Index-driven** — iterates page entries from the pages page, reads each data page, and
   validates: magic bytes, page type, namespace index, footer length (>= 16 and multiple of
   16), header/footer page_size agreement, page_size minimum (512), record count consistency
   between footer and deserialized data, offset array bounds, ordinal monotonicity across
   pages, and index/page ordinal agreement.

2. **Forward traversal** — walks the file from offset 0 using header page_size fields, validating
   each page structurally without relying on the index. Checks that the last page is a Pages or
   Namespaces type, that the traversal exactly covers the file length, that no Invalid-type pages
   exist, and that exactly one terminal index page (Pages or Namespaces) appears at the end.

3. **Cross-check** — verifies every index entry offset appears in the forward traversal, that all
   forward-traversal data pages are accounted for in the index, and that the entry count matches
   the number of data pages found during traversal.

### `slab get <file> <ordinals...>`

Retrieve records by ordinal. Ordinals may be individual values or range specifiers (see "Ordinal
range specifiers" below). Supports output format flags: `--raw` (binary to stdout), `--as-hex`,
`--as-base64`.

### `slab append <file>`

Append more data onto the end of a slab file from stdin or a source file (`--source <path>`).
This verifies that the input file is well formed first by running the same three-pass validation
as `slab check` and throws an error if the file is not structurally sound. Supports slab layout
parameters (`--preferred-page-size`, `--min-page-size`, `--page-alignment`).

### `slab import <file> <source>`

Import more data into a new, or onto the end of a slab file from another file format.

* The file extension is trusted to determine the format if it is a well known extension.
* if the format is not specified, and can't be easily detected, then source file format is
  forced, then the file is scanned. (for up to 5 seconds if needed) to determine:
    * any non-ascii characters are assumed to be binary format
    * the lack of newlines is assumed to be text format
* In every case, every byte of the original files is included, including the delimiters.
* for record types which are presumed to be uniform, when there are records which do not follow the form of the first
  record, a user should have an option to skip these with `--skip-malformed`
* For input format which is normally newline delimited, a `--strip-newline` option should be provided.
* On export, an option `--add-missing-newline` should be assumed for these formats, but this should be disabled when the
  user specified `--as-is`.

Format may be forced with flags: `--newline-terminated-records`, `--null-terminated-records`,
`--slab-format`, `--json`, `--jsonl`, `--csv`, `--tsv`, `--yaml`. Supports slab layout
parameters.

### `slab export <file>`

Export content from a slab file. Currently supports text (newline-delimited, default), cstrings
(null-terminated), and slab format. Aspirationally should support all the formats that import
supports, with the same parsable check enabled by default, but easy to disable if the user
trusts the parsability of the data. User can specify the output file name (`--output`), or
stdout if omitted. Supports `--range` for ordinal range filtering and slab layout parameters
(for slab output). Supports `--text`, `--cstrings`, `--slab-format` flags.

### `slab rewrite <input> <output>`

Rewrite a slabtastic file into a new file, combining reordering and repacking into a single
operation:

* Reads all records from the input file
* Sorts records by ordinal to restore monotonicity (reports whether input was already monotonic)
* Writes records to a new file with the provided slab parameters
* Eliminates logically deleted pages and alignment padding waste
* Uses `.slab.buffer` convention for atomic output

Supports slab layout parameters.

### `slab namespaces <file>`

List all namespaces in a slab file. Reports namespace index, name, and pages page offset. For
single-namespace files, shows just the default namespace.

### `slab explain <file>`

Illustrate the slab layout on the console using block-diagrammatic notation. The user can
scope the output to:

- a given page number or set of page numbers
- a given namespace
- a given range of ordinals

Any page covered by the specified options is printed in diagrammatic form, showing the page
layout including header (magic + page_size), records (with sizes and ordinal labels), offset
array entries, and footer fields (start_ordinal, record_count, page_size, page_type,
namespace_index, footer_length). When no scope is specified, all pages are shown.

## File Formats Supported

For each file format, record boundaries need to be detectable. Any characters which are part of the
record boundary are kept intact within slab records as part of the original content. In effect,
concatenating the records from a slab file directly to another file should result in the same
content which was in the original non-slab file.

When a supported format is read, it is by-default required to be strictly parsable in that format as
part of imported. The only exceptions to this are when the 'text' and
'cstrings' formats are used, since these are strictly delimited by newlines or null bytes.

* text: newline-delimited, ascii or unicode text
* cstrings: data which is null byte terminated
* json: parsable stream of json objects with whitespace between them
* jsonl: parsable stream of jsonl lines delimited with newlines
    * Users can force this type to "text" to bypass parsing check
* csv: comma separated values, must be parsable, with newline record boundaries
* tsv: tab separated values, must be parsable, with newline record boundaries
* yaml: parsable yaml, with `---` document boundaries. Each document is a record.
* slab: Any slab file

## Output Formats Supported

When a command is run which prints to the console, the user should be able to override the native
format with some conversions:

* `--as-hex` output the bytes as hex, with a space between each byte, and a trailing newline for
  each record
* `--as-base64` output the bytes as base64, with a trailing newline for each record

## Ordinal range specifiers

For some commands, the user can specify the ordinals to operate on. Valid forms (all resolved to
half-open `[start, end)` internally):

* `n` — shorthand for `[0, n)` (first n ordinals)
* `m..n` — closed interval, equivalent to `[m, n+1)` (m to n inclusive)
* `[m,n)` — half-open interval (m inclusive, n exclusive)
* `[m,n]` — closed interval (m inclusive, n inclusive)
* `(m,n)` — open interval (m exclusive, n exclusive)
* `(m,n]` — half-open interval (m exclusive, n inclusive)
* `[n]` — single ordinal, equivalent to `[n, n+1)`
* Brackets and commas may also use `..` as separator: `[m..n)`

## Slab layout parameters

For some commands (those which write slab data), the user should be able to specify the standard
slab format tunables via CLI flags:

- `--preferred-page-size <bytes>` — flush threshold (default 4194304 / 4 MiB)
- `--min-page-size <bytes>` — floor / alignment boundary (default 512)
- `--page-alignment` — pad to `min_page_size` multiples (default off)

These correspond to the writer configuration parameters (see "Writer Interface" below).

## Namespace Support

All commands should be adapted to support namespaces as described in the "Slab Namespaces" section.
If no namespace is specified, the default namespace (`""`, index 1) is assumed.

# Concurrent Readers

Concurrent readers streaming in a slabtastic file may incrementally read the file by watching for
updates, but this is opportunistic at best, given that revisions may occur from subsequent pages
page writes. However, as long as the reader session can safely assume the writer to be streaming _a
version_ of data which is valid, it is valid for the reader to observe the file incrementally, as
pages are written. This is a special case where mutability is not expected, and this should be made
explicit where possible. Still readers must not assume atomic writes, and thus should ensure that
the `[magic][size]` is used to determine when a page is valid for reading based on the incremental
file size.

Further, when any writers are streaming to a slabtastic file, they are required to flush buffers at
slab boundaries. This is to ensure that the pages page is always up to date with the current state
of the file across systems which do not share state via a VFS subsystem.

# Reader Interface

The reader interface should support ordinal based get, and streaming get. Opening a file reads
the trailing page footer to locate the pages page (or namespaces page), builds the in-memory
ordinal-to-offset index, and returns a reader handle.

## Read modes

- **Point get** — fetch a single record by ordinal using O(1) expected via interpolation
  search over the pages page index (binary search fallback). The record is extracted via
  zero-copy offset lookup (reading only the footer and two offset entries) without
  deserializing the full page. The OS buffer cache
  handles repeated reads of the same file region.
- **Batched iteration** — the reader should allow the user to specify the number of records to
  read at a time (the batch size), and the reader should try to buffer that many. It should be
  possible for the reader to return less, but if the reader returns 0 then the requestor should
  assume there are no more. Each batch yields `(ordinal, data)` pairs.
- **Sink read** — the reader should be able to provide a sink for items to be read into. All
  record data is written sequentially to the sink in ordinal order, returning the total number
  of records written.
- **Async sink read** — the reader should be able to read all records to a sink on a background
  thread, with a callback to be notified when it is done. The handle returned should implement
  a pollable progress interface (see "Async Task Model" below).
- **MultiBatch read** - the reader should allow the user to submit concurrently a number of different batch requests.
  The responses should be returned to the user in the same order. Partial success should be possible and the user should
  be able to tell if any of the results was "empty". Internal benches for this mode need to test for order sensitivity.

- Check whether a record exists for a given ordinal (existence check, not full read).
- Return the number of data pages in the file.
- Return the pages page index entries (for inspection or validation).
- Iterate all records in ordinal order as `(ordinal, data)` pairs.
- Return the file size in bytes.
- Read and deserialize a page at a specific byte offset (used by forward traversal in `slab
  check`).
- Read and deserialize a data page referenced by a pages page index entry (uses a reusable
  buffer to avoid per-page allocation during sequential bulk operations).

## Opening semantics

Opening reads the last 16 bytes of the file to obtain the terminal page footer. If the terminal
page is a pages page (type 1), it reads the full pages page directly. If the terminal page is a
namespaces page (type 3), it follows the namespace entries to locate the default namespace's
pages page. If neither, the file is rejected as invalid.

# Writer Interface

The writer interface accumulates records into an in-memory page and flushes complete pages to
disk when the preferred page size threshold is reached. The caller must explicitly finish the
writer to flush any remaining records and write the trailing pages page (index).

## Construction

- **New file** — create a new slabtastic file at the given path with the given configuration.
- **Append mode** — open an existing file in append mode. Reads the trailing pages page,
  positions the write cursor after the last data page, and continues ordinals from where the
  previous writer left off.

## Write modes

- **Single** — append one record at a time. The record is added to the current in-memory page.
- **Bulk** — append a slice of records in one call, flushing pages as needed.
- **Asserted ordinal** — append a record (or records) while verifying that the ordinal matches
  the writer's next expected ordinal. If the ordinal does not match, the call fails with an
  OrdinalMismatch error and nothing is written. This is useful for pipelines that need to
  assert data integrity during ingest. Both single-record and bulk variants should be
  supported.
- **Async from iterator** — the writer should be able to consume an iterator of records on a
  background thread, with a callback when done. The handle returned should implement a pollable
  progress interface (see "Async Task Model" below). The result is the total number of records
  written.

## Writer configuration

The writer accepts a configuration object with the following parameters:

| Parameter             | Type | Default  | Constraint                                      |
|-----------------------|------|----------|-------------------------------------------------|
| `min_page_size`       | u32  | 512      | Must be >= 512                                  |
| `preferred_page_size` | u32  | 4,194,304 | Must be >= min_page_size and <= max_page_size   |
| `max_page_size`       | u32  | 2^32 - 1 | Must be <= 2^32 - 1                             |
| `page_alignment`      | bool | false    | When true, pad pages to min_page_size multiples |

Constraint: `min_page_size <= preferred_page_size <= max_page_size`. Violation of this ordering
is an error.

When alignment is enabled, the aligned page size is computed by rounding up to the next multiple
of `min_page_size`.

## Record-too-large

In v1, a single record that exceeds the configured maximum page capacity is rejected with a
RecordTooLarge error. There is no multi-page spanning for individual records.

## Flush-at-boundaries

The writer only issues writes of complete, serialized pages — it never writes a partial page
buffer. However, the underlying file write does not guarantee OS-level atomicity: a concurrent
reader may observe a partially-written page on disk.

# Async Task Model

Background work should be dispatched on a separate thread with no async runtime dependency.
Both reader and writer expose async methods that return a task handle. The task handle provides:

- **Progress view** — a read-only snapshot of progress counters:
    - `total` — total items to process (may be zero until known).
    - `completed` — items processed so far.
    - `is_done` — whether the background thread has finished.
    - `fraction` — `completed / total` (0.0 if total is zero).
- **Completion check** — quick non-blocking check whether the task is done.
- **Wait / join** — block until the background thread finishes and return the result.

Progress accessors must be thread-safe. Values are eventually consistent but always safe to
read from any thread.

# Error Conditions

All fallible library functions should return a structured error type. I/O errors from the
underlying file system are wrapped. The following error conditions must be distinguishable:

| Error                 | Context                          | Description                                                    |
|-----------------------|----------------------------------|----------------------------------------------------------------|
| InvalidMagic          | —                                | Magic bytes do not match "SLAB"                                |
| InvalidNamespaceIndex | offending byte                   | Namespace index is 0 (reserved) or negative (reserved)         |
| InvalidPageType       | offending byte                   | Page type byte is not a recognized variant                     |
| PageSizeMismatch      | header size, footer size         | Header and footer page_size values disagree                    |
| PageTooSmall          | offending size                   | Page size is below the 512-byte minimum                        |
| PageTooLarge          | offending size                   | Page size exceeds the maximum                                  |
| RecordTooLarge        | record size, max capacity        | A single record exceeds page capacity                          |
| OrdinalNotFound       | requested ordinal                | The ordinal is not present in the file                         |
| OrdinalMismatch       | expected ordinal, actual ordinal | Asserted ordinal does not match writer's next expected ordinal |
| InvalidFooter         | descriptive message              | Footer data is malformed or structurally invalid               |
| TruncatedPage         | expected bytes, actual bytes     | Page data is incomplete                                        |
| Io                    | underlying error                 | An I/O error from the file system                              |
| WithContext           | source error, file_offset, page_index, ordinal | Wraps another SlabError with positional context |

# File Naming

The filename extension for slab files shall be simply ".slab".

# Slab Namespaces

A slabtastic file supports multiple namespaces, which organize data into independent ordinal spaces.
Each namespace contains its own ordinal space, its own pages page, and its own set of data pages.
Pages are disjoint between namespaces — a page belongs to exactly one namespace, identified by the
`namespace_index` field in its footer.

## Namespace Indexing

Namespaces are identified by a 1-byte signed index stored in each page footer (byte offset 13):

| Index      | Meaning                                                                         |
|------------|---------------------------------------------------------------------------------|
| 0          | Invalid / reserved (same role as the former `VERSION_INVALID = 0`)              |
| 1          | Default namespace `""` — always present, assumed when no namespace is specified |
| 2 to 127   | User-defined namespaces                                                         |
| -128 to -1 | Reserved for future use                                                         |

This design provides backward compatibility with existing v1 files: the byte at position 13 was
previously the `version` field, which was always `1` for v1 files. Since namespace index 1 is the
default namespace, existing files read correctly without migration.

Namespace names are strictly parsable UTF-8 strings. The maximum length of a namespace identifier
is 128 bytes, regardless of the encoding. The default namespace is named `""` (empty string).

## Namespaces Page (page type 3)

When a slab file contains any non-default namespaces, a namespaces page (page type 3) is written
as the final page in the file. The namespaces page maps namespace names to their indices and
locates each namespace's pages page.

The record layout for namespaces page entries is:
`[namespace_index:1][name_length:1][name_bytes:N][namespace_pages_page_offset:8]`

Each record is variable-length:

- `namespace_index` (1 byte, signed): the index value (must match the index in data page footers)
- `name_length` (1 byte, unsigned): length of the namespace name in bytes (0 for default, max 128)
- `name_bytes` (N bytes): the UTF-8 encoded namespace name
- `namespace_pages_page_offset` (8 bytes, signed LE): file offset of this namespace's pages page

The namespaces page uses the standard page layout (`[header][records][offsets][footer]`), with the
same offset fence-post scheme as other pages. The namespaces page footer has `page_type = 3` and
`namespace_index = 1` (it belongs to the default namespace structurally).

Each referenced pages page must be valid and present in the file. Each namespace has its own
independent pages page, so that logically an entire namespace can be extracted into a separate file
or merged from another file by operating at the page level.

## File Ending Rules

A slabtastic file must end with one of:

1. A **pages page** (type 1) — when the file contains only the default namespace. This is the
   common case and is fully backward compatible with pre-namespace files. If a namespaces page is
   not the last page of a slab file, then it must be that there are no namespaces besides the
   default `""` namespace.
2. A **namespaces page** (type 3) — when the file contains any non-default namespaces. The
   namespaces page is always the last page, and every namespace's pages page must be present
   earlier in the file.

A slabtastic file that does not end in a pages page or namespaces page is invalid.

## CLI and API Namespace Support

Commands and API features must be able to specify namespaces, list them, and operate on them:

- If no namespace is specified, the default namespace (`""`, index 1) is assumed.
- If a namespace is specified that is not present for reading, the access command or call should
  fail with an error.
- If a namespace is specified that is not present for writing, the namespace should be created and
  added to the slab file.
- A `slab namespaces <file>` CLI command should list all namespaces and their metadata.

## Namespace Extraction and Merging

Because each namespace has its own pages page and its own set of data pages, namespaces can be
extracted or merged at the page level:

- **Extract**: Copy a namespace's data pages and pages page into a new single-namespace file.
- **Merge**: Copy data pages and pages page from one file into another, assigning or mapping
  namespace indices as needed, then write a new namespaces page referencing all namespaces.

# Implementation Structure

The implementation consists of a library and a thin CLI binary. The CLI module is part of the
library (not the binary) so that subcommand logic is directly testable.

## Library modules

| Module          | Purpose                                                                                                          |
|-----------------|------------------------------------------------------------------------------------------------------------------|
| config          | Writer configuration and page size validation                                                                    |
| constants       | Magic bytes, page size limits, page type enum, footer size constants                                             |
| error           | Error type and result alias                                                                                      |
| footer          | 16-byte v1 page footer serialize/deserialize                                                                     |
| page            | In-memory page representation with records, offset array, and zero-copy point extraction from serialized buffers |
| pages_page      | Pages page index serialize/deserialize                                                                           |
| namespaces_page | Namespaces page serialize/deserialize                                                                            |
| reader          | All read modes (zero-copy point get, batched iteration, sink read, async sink read, multi-batch concurrent read) |
| writer          | All write modes (single, bulk, asserted ordinal, async from iterator)                                            |
| task            | Async task handle and progress polling                                                                           |
| cli             | CLI subcommand implementations                                                                                   |

## CLI submodules

Each CLI subcommand has its own module, plus a shared ordinal range parser.

## Binary

The binary is a thin wrapper that parses CLI arguments and dispatches to the library's CLI
module.

## Public surface

The library exposes: writer configuration, page type enum, error types, footer, page,
pages page (with page entry), namespaces page (with namespace entry), reader (with batch
iterator and batch read result), async task types (task handle, progress), and writer.

## Dependencies

The core library has zero runtime dependencies beyond the standard library. The CLI adds
dependencies for argument parsing and structured format support (JSON, CSV, YAML).

# Reader and Writer implementations

Where possible, reader and writer logic should use zero-copy buffering techniques and page cache effectively.
Point reads should do precise and limited reads of buffered data. Do not validate the page beyond traversing the
necessary data to find the ordinal elements.

# Benchmarking

Benchmarks should be added, and critical sections for measurement should not include file IO setup and teardown. Benches
should focus on core read and write patterns once this setup has been done.

The pages page and namespaces page types should always be cached in memory explicitly while a slab file is open.