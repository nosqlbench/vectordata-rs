# veks prepare cleanup-profiles

Remove sized profiles that nothing references, leaving the cache alone.

A stratum under `strata:` generates sized profiles, and the runner
materialises them as ordinary entries under `profiles:` with their own
`profiles/<name>/` directories of answer keys. Removing a stratum
removes the generator, not what it generated: the entries and
directories stay, and the pipeline keeps planning steps for them. This
command finds and removes two kinds of leftovers:

- **entries no stratum names** — sized profiles under `profiles:` (a
  `base_count`, not `default`, not a partition) whose name appears in
  no stratum's `series`;
- **directories no entry names** — `profiles/<name>/` directories for
  which `profiles:` has no entry (`base` and `default` aside).

It never touches `.cache/`: predicate-key segments, provenance sidecars
and the progress log stay. Records in the progress log for the removed
profiles' steps are harmless, since the runner consults records only
for steps the definition has. Before rewriting `dataset.yaml` it backs
the file up under `.backup/`, and afterwards it refreshes the progress
log's mtime the way `stratify` does, so completed steps are not thought
stale for the definition being newer.

## Usage

```bash
veks prepare cleanup-profiles --dry-run [dataset-dir | dataset.yaml]   # report only
veks prepare cleanup-profiles [dataset-dir | dataset.yaml]             # apply
```

## Example

After removing a `linear` stratum whose profiles the runner had already
materialised:

```
Sized profiles under `profiles:` that no stratum (mul, fib, decade) names:
      110m     9.0 GiB  ./profiles/110m
      120m     9.8 GiB  ./profiles/120m
      130m     7.7 MiB  ./profiles/130m
      ...
36 entries to drop from the definition, 36 directories to remove (19.0 GiB); the cache is left alone.

Dry run — nothing changed. Run without --dry-run to apply.
```

After applying, `veks run --dry-run` no longer plans the removed
profiles' steps and `veks check` no longer lists their directories as
extraneous.

## Options

| Option | Description |
|--------|-------------|
| `path` | Dataset directory or path to `dataset.yaml` (default: `.`) |
| `--dry-run` | Show what would be removed without changing anything |
