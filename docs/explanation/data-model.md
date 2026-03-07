# Data Model and Specification

`vectordata-rs` uses a structured data model to represent large-scale vector datasets, their metadata, and their evaluation predicates.

## Vector Datasets

A **Vector Dataset** is a collection of related data files, facets, and configurations used for vector-based search and analysis. The structure and metadata of these datasets are typically described in a `dataset.yaml` file.

### Core Components
- **Facets**: Distinct parts of a dataset, such as `base_vectors`, `query_vectors`, `neighbor_indices`, or `metadata_content`. Each facet is usually stored in a dedicated file format (e.g., `.fvec`, `.ivec`, `.slab`).
- **Profiles**: Named configurations of a dataset that describe which facets are included and how they are organized. Profiles allow for different views of the same underlying data.
- **Views**: A logical projection of a dataset profile. A view provides a simplified, typed interface for accessing the data in a specific profile.
- **Attributes**: Key-value pairs of metadata that describe properties of the dataset as a whole, such as the distance function used or the dimension of the vectors.

## Record Types: MNode and PNode

Metadata and filter predicates are stored as binary records in **Slab Files**. These records serve two complementary roles:

### MNode (Metadata Node)
An MNode is a self-describing key-value record. Each field has a name (UTF-8 string), a type tag, and a value.
- **Self-describing**: Every field carries its own type tag, so the record can be decoded without external schema knowledge.
- **Rich Type System**: Supports scalars, collections, temporal types, and identifiers.
- **Nestable**: The `Map` variant can contain a nested MNode, enabling hierarchical metadata.

### PNode (Predicate Node)
A PNode is a boolean predicate tree used to filter records during a search.
- **Tree Structure**: Composed of leaf nodes (predicates like GT, LT, EQ) and interior nodes (AND/OR conjunctions).
- **Two Addressing Modes**: Fields can be referenced by name (string) or by index (u8), supporting both human-readable and compact binary encodings.
- **Pre-order Encoding**: The tree is serialized parent-first for single-pass streaming decode.

## Predicated Datasets

A **Predicated Dataset** extends a standard vector dataset with metadata and filter predicates, enabling *filtered* approximate nearest neighbor (ANN) search.

### Relationship Between Vectors, Metadata, and Predicates
In a predicated dataset:
1. Each **base vector** has an associated **MNode record** in the `metadata_content` slab. This record describes properties of the vector (e.g., category, price).
2. Each **query** has an associated **PNode tree** in the `metadata_predicates` slab. This tree expresses the filter condition (e.g., `(category = 3 AND price <= 100)`).
3. During search, only vectors whose metadata satisfies the query's predicate are considered as potential neighbors.

## Dataset Specification (`dataset.yaml`)

The `dataset.yaml` file is the source of truth for a vector dataset. It defines:
- **Facet Definitions**: Mapping names to file paths and format types.
- **Profile Definitions**: Organizing facets into logical groups.
- **View Parameters**: Defining the structure and windowing of the data views.
- **Custom Attributes**: Storing additional metadata used by the access layer or the processing pipeline.

This structured specification allows the workspace tools to automatically resolve file locations, formats, and dataset properties based on a single configuration file.
