# Explanation

Understanding-oriented documentation explaining the architecture, design, and core concepts of the `vectordata-rs` project.

## High-Level Architecture

- [Architecture Overview](./architecture.md): A bird's-eye view of the workspace crates, their responsibilities, and the core design principles.
- [Pipeline Engine](./pipeline-engine.md): The command-stream processing system and DAG execution engine.
- [Access Layer](./access-layer.md): Transparent access, integrity verification, and caching for local and remote vector datasets.
- [Resource Management](./resource-management.md): Dynamic resource governance and allocation strategies to prevent system saturation.

## Data and Storage

- [Data Model and Specification](./data-model.md): How vector data, metadata (MNode), and predicates (PNode) are defined and related.
- [Storage Layer (Slab Files)](./storage.md): The page-aligned record container format and record identification system.
- [Two-Stage Codec](./two-stage-codec.md): Architectural decoupling of physical storage from logical record decoding.
- [Vernacular Formats](./vernacular-formats.md): Human-readable representations (JSON, SQL, etc.) for binary records.

## System Visibility

- [Observability and Documentation](./observability.md): Real-time progress monitoring and self-documenting pipeline commands.
