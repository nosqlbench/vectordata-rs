# config tag-profiles

Tag every profile of an existing dataset the way its generators would
today, by textual edit of `dataset.yaml`.

A dataset written before profile tags existed has profiles a selector
can address only by name. This step gives each of them the tags
`docs/design/srd-profile-selectors.md` reads: `size` on every profile
with a count (a generated member's rung is its name, the default's is
the rung of what its base holds), and on every profile whose predicate
facet a completed `generate predicates` step produced, that facet's
`family` and `selectivity_ladder` from the step's record and its
`forms` and `predicates` class from a census of the file. Every profile
other than `default` and the partitions gains `inherits: default`.

No facet and no file moves. The edit keeps every comment and every
other line, a re-run writes nothing, and what was written is recorded
beside the step, so a later hand edit to a tag is reported as stale
rather than silently kept or overwritten.

## Usage

Declared as a finalize step, after every producing step and before
the steps that publish the dataset's definition:

```yaml
  - id: tag-profiles
    run: config tag-profiles
    description: Tag every profile the way its generators would today
    after:
    - verify-predicates
    finalize: true
```

It takes no options.

## Example

tessera, whose stratified predicate set was generated before tags
existed:

```
  tag-profiles — tags on 55 profile(s); parents named on 54 profile(s)
  tags on 'default': size=495m, family=stratified, selectivity_ladder=[0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6, 1e-7], forms=22, predicates=mixed
  tags on '10m': size=10m, family=stratified, ...
```

After which `tessera:size=10m,predicates=mixed` names the profile
`10m`, and `tessera:size>=100m` the rungs from `100m` up, the default
included.
