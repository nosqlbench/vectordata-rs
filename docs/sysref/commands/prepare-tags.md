# veks prepare tags

Edit the tags of the profiles a selector names, as a change of plan.

A tag is a plan (`docs/design/srd-profile-selectors.md`, PS-11): what a
profile is for, written so a selector can read it. A plan may be
revised, and revising it must move nothing else: no facet is
recomputed, no checksum changes, no step that produced data reads the
edit as a reason to run again. Only the published copies of the
definition — `dataset.json`, the catalog, the docs, the merkle tree —
follow the edit, because they are derived from `dataset.yaml` and must
carry the tags a selector will read.

## Usage

```bash
veks prepare tags [PATH] --profile SELECTOR --set key=value ... [--unset key ...] [--dry-run]
```

`PATH` is the dataset directory or `dataset.yaml` (default `.`).
`--profile` is a selector over the dataset's profiles: a name
(`10m`), every profile (`profile=*`), or an expression
(`predicates=mixed,selectivity_ladder=*`, `family=uniform`). The value
of `--set` is read as YAML, so `selectivity=[0.1, 0.01, 1e-6]` is a
list and `selectivity='1e-6'` keeps its spelling; a map is refused
(PS-15). `--unset` removes a tag, and the `attributes:` map with it
when it was the last one, since an empty map would read as "described
as nothing" (P-7). `--dry-run` prints the edits and writes nothing.

## What it writes

- The edit is **textual and idempotent**: each named profile's own
  `attributes:` lines change, block or flow form, comments and every
  other line kept; a second run with the same edit writes nothing.
- The edit is **recorded as the plan** (PS-13): every step record
  holding the edited tag takes the new value, or drops it on `--unset`,
  and every record naming `dataset.yaml` as its output takes the file's
  new size. A step that wrote the tag — `config tag-profiles`, a
  generator — therefore stays fresh: the edit is the plan it would
  report a hand edit against.
- A backup of `dataset.yaml` is taken under `.backup/` before the
  write.

## What follows

On the next `veks run`, the finalize steps that publish the definition
run again, because `dataset.yaml` is their input by content
(`docs/sysref/04-pipeline.md`, freshness). No compute step holds the
definition as an input, so none of them runs.

A **naming tag** edited on a generated profile — a uniform set's
`selectivity`, a rung's `size` — is noted: the profile's name stays as
it is (PS-22) and no longer spells the tag. Naming is done once, at
generation; selectors read tags, not names, so the profile is still
found by what it now says.

## Example

Back-port a `selectivity` tag onto every stratified rung, as the
ladder its slab was generated with:

```bash
veks prepare tags . --profile 'predicates=mixed,selectivity_ladder=*' \
  --set 'selectivity=[0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6, 1e-7]'
veks run dataset.yaml
```

The run reports every compute step fresh and the seven finalize steps
stale on `dataset.yaml`, and the published `dataset.json` then carries
the tag.
