<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# vectordata-rs Design Documentation

This directory contains the software requirements and design specification for
the vectordata-rs workspace — veks (CLI), vectordata (shared types and access
layer), and slabtastic (page-aligned record container) — used in bulk
processing of vector datasets for approximate nearest neighbor (ANN)
benchmarking.

## Documents

| Document | Description |
|----------|-------------|
| [01-system-overview.md](01-system-overview.md) | Architecture overview, crate structure, and system boundaries |
| [02-data-model.md](02-data-model.md) | Data formats, wire protocols, codec architecture, and storage |
| [03-pipeline-engine.md](03-pipeline-engine.md) | DAG execution, step lifecycle, progress tracking, and recovery |
| [04-command-catalog.md](04-command-catalog.md) | All pipeline commands, their options, and resource profiles |
| [05-dataset-specification.md](05-dataset-specification.md) | dataset.yaml schema, facets, profiles, upstream pipelines, and implementor's guide for dataset discovery and remote access |
| [06-resource-management.md](06-resource-management.md) | Resource constraints, failure modes, and requirements for active resource governance |
| [07-command-documentation.md](07-command-documentation.md) | Built-in markdown documentation for commands, completion summaries, and help rendering |
| [08-progress-display.md](08-progress-display.md) | UI-agnostic eventing layer: event algebra, sink trait, handle facade, and rendering backends |
| [09-vectordata-access-layer.md](09-vectordata-access-layer.md) | vectordata crate: VectorReader trait, mmap/HTTP backends, dataset access |
| [10-dataset-publishing.md](10-dataset-publishing.md) | S3 publishing: `.publish_url` binding, `veks publish` sync command, file filtering |
| [11-preflight-checks.md](11-preflight-checks.md) | `veks check`: pipeline completeness, publish URL binding, merkle coverage |
| [12-dataset-import-flowchart.md](12-dataset-import-flowchart.md) | `datasets import`: universal pipeline generation flowchart, input axes, stage activation rules |
| [13-data-access-layer.md](13-data-access-layer.md) | Data access layer: DatasetLoader API, MAFileChannel, merkle-verified chunk caching, windowed remote access, transport layer |

## Audience

- **Developers** extending or maintaining the veks codebase
- **Operators** running large-scale dataset preparation pipelines
- **Contributors** to the vectordata access layer

## Scope

This specification covers the vectordata-rs workspace as of `main` branch, March 2026. It captures
the current implementation faithfully and identifies gaps — particularly in
resource management — that must be addressed before running heavy workloads
reliably on constrained systems.
