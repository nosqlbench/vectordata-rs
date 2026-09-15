# veks prepare readme

Write the dataset's `README.md` scaffold.

`README.md` at the dataset root is how a dataset is documented
(sysref §1, "Static payload"): the narrative a reader needs before the
generated reference in `docs/dataset.md` makes sense. It ships with the
data, is checksummed and merkled like any content, and is never
regenerated or cleaned.

## Usage

```bash
veks prepare readme [PATH] [--force]
```

`PATH` is the dataset directory or its `dataset.yaml` (default `.`).
An existing README is kept; `--force` replaces it with a fresh
scaffold after taking a backup under `.backup/`.

## What it writes

The six standard sections, in order:

1. **What this dataset is** — with the embedding model, revision,
   distance, normalization and base count the definition holds.
2. **How it was made** — with the pipeline's steps and their
   descriptions as `dataset.yaml` declares them.
3. **License and attribution** — with the `license`, `vendor` and
   `notes` attributes.
4. **Profiles** — with a count of the rungs, layers, sets and
   partitions declared.
5. **Tags and selectors** — with the tag schema and the spec form
   `<name>:<selector>`.
6. **Predicates and example queries**.

Wherever prose is owed the scaffold leaves `<!-- veks: fill in -->`
with a note of what belongs there. `veks check` refuses a dataset with
no README, one that does not open with a `# <title>` heading, or one
whose markers are still there, so the scaffold cannot be mistaken for
documentation.
