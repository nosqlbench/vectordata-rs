# Slabtastic Documentation

Slabtastic is a streamable, random-accessible, appendable data layout format for
non-uniform data by ordinal. Files use the `.slab` extension.

This documentation follows the [Diataxis](https://diataxis.fr/) framework:

## Tutorials

Step-by-step lessons for learning slabtastic.

- [Getting Started](tutorials/getting-started.md) — write your first slab file and read it back
- [Streaming I/O](tutorials/streaming-io.md) — batched reads, sink reads, and async write-from-iterator

## How-to Guides

Task-oriented recipes for common operations.

- [Append Data to an Existing File](how-to/append-data.md)
- [Import and Export Data](how-to/import-export.md)
- [Bulk Read and Write](how-to/bulk-read-write.md)
- [Background Tasks with Progress Polling](how-to/async-progress.md)
- [Page Sizing and Alignment](how-to/page-sizing.md)
- [CLI File Maintenance](how-to/cli-maintenance.md)

## Reference

Technical specifications and API details.

- [Wire Format Specification](reference/wire-format.md)
- [Page Layout](reference/page-layout.md)
- [Footer Format](reference/footer-format.md)
- [Pages Page (Index)](reference/pages-page.md)
- [Namespaces Page](reference/namespaces-page.md)
- [Error Catalogue](reference/errors.md)
- [CLI Reference](reference/cli.md)

## Explanation

Design decisions and architectural context.

- [Why Slabtastic?](explanation/why-slabtastic.md)
- [Append-Only Semantics](explanation/append-only.md)
- [Sparse Ordinals and Interior Mutation](explanation/sparse-ordinals.md)
- [Concurrency Model](explanation/concurrency.md)

## Benchmarks

See [critcmp.md](../critcmp.md) for throughput numbers (NVMe).
