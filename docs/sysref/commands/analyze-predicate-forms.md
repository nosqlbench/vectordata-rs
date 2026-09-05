# analyze predicate-forms

Enumerate the predicate forms a facet holds, with the indexes each needs.

A system under test declares indexes against *forms*, not predicates:
`topic_l3 = ? AND citation_percentile >= ?` needs the same indexes whatever
the literals are. This command decodes every predicate, replaces its
literals with `?`, canonicalises conjunct order, and reports the distinct
forms with their counts and shares, the grammar space actually used
(fields, depth, arity, junction and operator kinds), and a field × access
table — equality, inequality, range, set, pattern — that is the index
declaration needed to serve the whole facet.

With the facet's `families` namespace present (a stratified facet), each
form is attributed to the families that produced it. With
`--metadata-indices`, each form carries the min, median, and max realised
selectivity of its predicates.

## Usage

```bash
veks analyze predicate-forms --predicates <slab> [--metadata-indices <slab>] [--metadata <slab>]
```

| Option | Required | Description |
|---|---|---|
| `--predicates` | yes | Predicate slab (PNode records); a `families` namespace attributes forms |
| `--metadata-indices` | no | Per-predicate matching ordinals from `compute evaluate-predicates`; adds selectivity ranges |
| `--metadata` | no | Metadata slab; its record count turns match counts into selectivities |

## Example

tessera's 10,000 stratified predicates:

```
predicate forms — 10000 predicates, 22 distinct forms

grammar space used
  fields:      9
  max depth:   2   max leaves per predicate: 2
  junctions:   AND×6825
  operators:   <=×5319  =×4398  >=×7108

form                                                        count   share   selectivity min / median / max
(sample_bucket <= ? AND sample_bucket >= ?)                  4315   43.1%   7.46e-8 / 1.00e-5 / 1.00e-1
                                                          families: control×4315
(citation_percentile >= ? AND topic_l3 = ?)                  1417   14.2%   3.23e-8 / 1.05e-6 / 3.16e-5
                                                          families: topical×1417
year = ?                                                     1044   10.4%   9.58e-7 / 2.42e-5 / 2.94e-2
                                                          families: bibliographic×1044
…

index needs — predicates touching each field by access
  field                access        preds
  citation_percentile  equality         30
  citation_percentile  range          1432
  sample_bucket        range          4315
  topic_l3             equality       2428
  …
```

A range spelled as two leaves on one field (`year >= ? AND year <= ?`)
is one `(year, range)` need. The forms table is the grammar a system
under test must accept; the needs table is the set of indexes it must
declare.

## See also

- `analyze predicate-summary` — per-field and per-operator leaf statistics
- [analyze explain-predicates](./analyze-explain-predicates.md) — one predicate through its matching metadata
