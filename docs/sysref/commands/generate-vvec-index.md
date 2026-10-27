# generate vvec-index

Write the `IDXFOR__` offset index of every variable-length file.

## Usage (pipeline step)

```yaml
- id: generate-vvec-index
  run: generate vvec-index
  source: .
```

`veks prepare bootstrap` emits this step as a finalize step, ahead of
`catalog generate` and `merkle create`.

## Behavior

A variable-length file (`.ivvec`, `.fvvecs`, `.u8vvec`, …) is read by
ordinal through its `IDXFOR__<name>.<i32|i64>` sidecar: the start offset
of each record, `i32` up to 2 GiB and `i64` past it. A remote reader
without one has to download the file whole to find its records, and
vectordata refuses to do that behind an open, so a dataset ships an
index beside each such file.

For every variable-length file under `source` (a directory, searched
with the publish filters, or one file), the step writes the index when
there is none or the one there is older than the file. It is
idempotent: a current index is left as it is, and a second run writes
nothing.

- Where the file has a `.mref`, the index gets one too, at the file's
  chunk size.
- A file that is empty, or does not end at a record boundary, fails the
  step, naming it.
- The bare `.ivec` is not indexed: it is also the uniform ground-truth
  format.
- Under `--dry-run`, nothing is written; each file that would be indexed
  is named.

Pipeline steps that write variable-length files already write their
index as they go; this step covers files that arrive another way —
imported, derived, or produced by a tool outside veks. `vectordata
push`, `veks publish`, `catalog generate` and `vectordata datasets
derive` write a missing index themselves, through the same code.

## Options

| Option | Default | Description |
|---|---|---|
| `source` | `.` | Directory to search for variable-length files, or one such file |
