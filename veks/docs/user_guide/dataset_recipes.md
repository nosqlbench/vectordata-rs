# Dataset Recipes

Constructing datasets from external sources can be trivial or complex. The dataflow recipes in this section are meant to
be used as a starting point and practical reference.

The goal of sourcing a dataset for the vectordata system is to get it into the internal manifest form which is
universally recognized by any vectordata client. In it simplest form, this is just a directory containing the basic
files (ivec, slab, sqlite, or any recognized encoding form) and a `dataset.yaml` descriptor which is the canonical "
dataset" ToC.

In its richest form, the dataset.yaml provides some or all of the following:

- dataset provenance and related details
- metric and dimensionality
- profiles
    - a profile is a subset of the data by range, with its own unique answer keys (ground truth, predicate indices)
- data preparation steps from upstream sources

The last item is your "easy button" for preparing datasets. By defining the upstream sources and preparation steps, the
whole pipeline becomes a declarative property of the dataset.yaml consumable form itself. This means that processes are
repeatable, idempotent, and iterable in a development sense.

It also means that you should be primarily concerned with filling out the `upstream` section of dataset.yaml as the way
you source and prepare datasets. All of of the available processing commands are also available directly as CLI options
for diagnostic, experimentation and exploration.

## Scenario: You have only base vectors

1. Scrub the base vectors to avoid zero-vectors
2. Create a total-ordering over the vectors lexicographically to eliminate duplicates
3. Create a shuffled order over the base vector ordinal in a persisted file.
4. Extract the first 10K (for example) vectors as the canonical `query_vectors`.
5. Extract the remaining vectors into a file for the canonical `base_vectors`
6. Determine the distance metrics used. If the vectors are verifiably L2-normalized, then use dot product. If not, then
   the user will be required to specify the distance function to use.
7. Using the query_vectors, base_vectors, and distance function, construct a KNN ground-truth set for some K (usually
    100)
8. Save the neighbor distances canonically as `neighbor_distances`.
9. Save the neighbor indices canonically as `neighbor_indices`.
10. Use validation tools to do a few samples of brute-force knn to verify the results.

For a knn-ground-truth dataset, this is the essential flow. The veks command line tools can assist in automating this
process.

## Scenario: Adding Profiles

## Scenario: Adding metadata

## Scenario: adding predicates

## Scenario: computing predicate answer keys

## HDF5 (ann-benchmarks) import

## raw fvec or ivec import directly


