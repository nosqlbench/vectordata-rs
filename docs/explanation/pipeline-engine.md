# Pipeline Engine

The **Pipeline Engine** (part of the `veks` crate) is a command-driven system for processing and transforming large-scale vector datasets. It uses a DAG-based execution engine to run modular command operations while managing system resources and progress.

## Command-Stream Processing

A pipeline is defined as a series of **Command Operations** (`CommandOp`) that process vector data. Each command is a modular unit of work that takes input (e.g., a file, a stream of records) and produces output.

### Common Pipeline Commands
- **`analyze`**: Inspect, profile, compare, and visualize vector data files.
- **`bulkdl`**: Bulk file downloader driven by YAML configuration with token expansion.
- **`convert`**: Convert vector data between various formats.
- **`import`**: Import data into the preferred internal format (e.g., Slab) based on facet type.
- **`slab`**: A suite of commands (import, export, append, rewrite, check, get, analyze, explain, namespaces, inspect) specifically for working with Slab files.
- **`compute-knn`**: Brute-force exact K-nearest-neighbor computation.

## DAG Execution Engine

Pipelines are typically defined in YAML (often within a `dataset.yaml`). The engine parses these steps and builds a **Directed Acyclic Graph (DAG)** of dependencies.
- **Dependency Resolution**: Steps are executed in an order that respects their inputs and outputs.
- **Skip-if-fresh**: The engine can skip steps if the output artifact is already present and newer than the input sources, significantly speeding up iterative processing.
- **Variable Interpolation**: Support for `${scratch}`, `${dataset}`, and other dynamic variables within pipeline definitions.

## Design and Observability

The pipeline engine is designed for high-throughput data processing:
- **Resource Governance**: Every pipeline step is monitored and throttled by a central [Resource Governor](./resource-management.md).
- **Progress Monitoring**: Real-time feedback via `indicatif` progress bars for long-running operations.
- **Integrated Documentation**: Each command carries its own documentation directly in the binary, accessible through the CLI (`veks help <command>`).

## Running Pipelines

Pipelines are executed using the `veks pipeline run` command. For simple one-off operations, individual commands can also be called directly through the `veks pipeline <group> <cmd>` interface.

```bash
veks pipeline run --steps '
  - slab import:
      from: metadata.json
      to: metadata.slab
'
```

By providing a unified, resource-aware, and observable way to process vector data, the pipeline engine simplifies the management of large-scale dataset preparation.
