# Tutorials

Step-by-step guides for common tasks.

New to the **vecd** server? Start with the
[vecd intro & quickstart](../guides/vecd-intro.md), then the
[concepts](../guides/vecd-concepts.md) and [vecd.conf reference](../guides/vecd-config.md).

| Tutorial | What you'll learn |
|----------|------------------|
| [Getting Started](./startup-and-publish.md) | Install, browse catalogs, bootstrap a dataset, publish |
| [vecd End-to-End](./vecd-end-to-end/) | Self-host a vecd server: AAA, upload a dataset, publish a catalog, explore over HTTP |
| [vecd Rate Limits](./vecd-rate-limits/) | Per-connection vs per-client bandwidth caps; concurrent-chunk download scaling |
| [Access from Rust](./access-datasets-from-rust.md) | Load datasets, read vectors, use the vectordata API |
| [Build a Predicated Dataset](./build-predicated-dataset.md) | Metadata, predicates, filtered KNN from scratch |
| [Dataset Recipes](./dataset-recipes.md) | Common patterns: base-only, self-search, HDF5, profiles, synthetic |
| [Resource Governance](./resource-aware-pipeline.md) | Control memory and threads during large pipeline runs |
| [Inspect Slab Records](./inspect-slab-records.md) | Decode and render MNode/PNode records |
| [Explore Vector Data](./explore-vector-data.md) | Interactive TUIs: values grid, unified analytics, shell |
| [ANode Codecs](./build-anode-codec-pipeline.md) | Encode, decode, and render metadata/predicates in Rust |
| [Verify with knn_utils](./verify-with-knn-utils.md) | Cross-verify datasets against the Python knn_utils reference |
