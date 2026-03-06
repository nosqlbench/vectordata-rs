<!-- Copyright (c) DataStax, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Veks User Guide

Veks is a CLI tool for the full lifecycle of vector dataset preparation for ANN
benchmarking: downloading, importing, converting, computing ground truth,
generating predicated datasets, and analyzing results.

## Getting started

Run a dataset pipeline:

```sh
veks run dataset.yaml
```

Validate without executing:

```sh
veks run dataset.yaml --dry-run
```

Execute a single pipeline command directly:

```sh
veks pipeline compute knn --base base.fvec --query query.fvec --indices out.ivec --neighbors 100
```

## Resource management

Control how much of the system's resources the pipeline uses with `--resources`:

```sh
# Fixed memory ceiling
veks run dataset.yaml --resources 'mem:32GiB'

# Let the governor dynamically adjust within a range
veks run dataset.yaml --resources 'mem:25%-50%,threads:4-8'
```

The resource governor monitors system state (RSS, page faults) and dynamically
adjusts resource allocations to maximize utilization without saturating the
system. See [Resource Governance](../concepts/resource-governance.md) for the
concept and [Configuring Resource Management](../howto/configure-resource-management.md)
for practical configuration guidance.

## Built-in command help

Every pipeline command has built-in documentation:

```sh
# List all commands with summaries
veks pipeline --help

# Detailed help for a specific command
veks pipeline compute knn --help
```

See [Command Documentation](../concepts/command-documentation.md) for how the
documentation system works.

## Key concepts

- [Slab Files](../concepts/slab-files.md) — page-aligned record containers
- [MNode and PNode](../concepts/mnode-and-pnode.md) — metadata and predicate codecs
- [Predicated Datasets](../concepts/predicated-datasets.md) — filtered ANN search datasets
- [Resource Governance](../concepts/resource-governance.md) — adaptive resource management
- [Command Documentation](../concepts/command-documentation.md) — built-in help system

## Dataset sources and recipes

- [Dataset Sources](dataset_sources.md) — where to find vector datasets
- [Dataset Recipes](dataset_recipes.md) — step-by-step dataset preparation guides
