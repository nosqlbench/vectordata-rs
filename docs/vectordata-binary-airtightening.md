# `vectordata` binary — first-experience changes

This change set audits and tightens every surface of the `vectordata`
binary, in keeping with the goal that the first command a new user
types should be self-explanatory, behave correctly under shell
scripts, and never silently swallow failure.

Nothing about the public Rust library API has changed. If you embed
the `vectordata` crate as a dependency, your code keeps compiling.

---

## What you'll notice

### `vectordata --help` actually describes the binary

Before, the top-level summary read "vectordata cache + config admin" —
inherited from when the binary really did just manage settings. The
binary has grown to cover datasets, explore, cache, config, and
completions, but the help text never caught up.

Now:

```
$ vectordata --help
vectordata — the user-facing entry point for working with published
vector-search benchmark datasets.

Common starting points:
  • vectordata datasets          — TUI browser of every reachable dataset
  • vectordata datasets list     — text catalog listing
  • vectordata explore           — interactive value/distance explorer
  • vectordata config show       — review the active configuration
  • vectordata cache list        — see what's on disk

First-time users typically start by configuring a catalog source
(vectordata config add-catalog <url-or-path>) and a cache directory
(vectordata config set-cache <dir>).
```

### `vectordata --version` is now useful in bug reports

Before:

```
$ vectordata --version
vectordata 1.1.5
```

Now:

```
$ vectordata --version
vectordata 1.1.5 (slabtastic@1.1.5-2-ga5a6616+dirty, release build, 2026-05-28)
```

The extra fields come from a new `build.rs` that captures `git
describe`, the cargo profile, and the build date at compile time. If
you build from a tarball (no git tree), the git field becomes
`unknown` and the rest still work.

### Exit codes scripts can trust

Several user-error paths used to exit `0`. They now exit `1`. If you
have a shell script that branches on `$?` from a `vectordata datasets …`
call, expect previously-silent failures to start producing visible
non-zero exits.

| Command + scenario                                        | Before | After |
|-----------------------------------------------------------|--------|-------|
| `vectordata datasets list --at file:///bad/url`           | `0`    | `1`   |
| `vectordata datasets probe zzz-unknown`                   | `1`    | `1`   |
| `vectordata datasets curlify` outside a dataset dir       | `1`    | `1`   |
| `vectordata explore` (no TTY, no `--source`)              | `0`    | `1`   |
| `vectordata explore --source /missing`                    | `1`    | `1`   |
| `vectordata` (no subcommand)                              | `2`    | `2`   |
| `vectordata bogus-subcommand`                             | `2`    | `2`   |
| `vectordata config show`                                  | `0`    | `0`   |
| `vectordata config get-cache` (configured)                | `0`    | `0`   |

Where the "before" already matched the "after", the path is now also
backed by a return-value chain instead of a deep `std::process::exit`,
so future regressions are less likely.

### `vectordata explore` is a single flat command

The previous structure exposed three sub-subcommands — `explore
explore`, `explore values`, `explore shell` — inherited verbatim from
the pre-migration `veks interact` shape. None of that nesting made
sense at the `vectordata` entry point. The `values` (raw-values grid)
and `shell` (REPL) modes have been removed; `vectordata explore` now
directly launches the unified vector-space explorer (norms,
distances, eigenvalues, PCA — all in one TUI). Pass `--source` or
`--dataset` to skip the picker.

### `vectordata explore` no longer hangs the terminal silently

If you run `vectordata explore` without a `--source` and without an
attached TTY — common in CI logs, piped scripts, or `&` background
runs — the dataset picker used to exit cleanly with no output. There
was no signal that anything had failed.

Now you get:

```
error: dataset picker requires an interactive terminal
       (stdout is not a TTY — were you redirecting output?)

Pass `--dataset <name>` or `--source <path>` to explore non-interactively,
or run `vectordata datasets list` for a text-friendly catalog view.
```

…and exit code `1`.

If you previously invoked the picker non-interactively expecting it
to no-op, switch to `vectordata datasets list` (the text catalog view)
or pass an explicit `--source`/`--dataset`.

### Better first-run onboarding

`vectordata config show` on a fresh machine — no `settings.yaml`, no
`catalogs.yaml` — now walks you through setup:

```
$ vectordata config show
Configuration: /home/you/.config/vectordata/settings.yaml
  (settings file does not exist)

First-run setup — three quick steps:
  1. Pick a cache directory:
       vectordata config set-cache <path>
       (or `vectordata config set-cache auto` to choose the largest writable mount)
  2. Subscribe to a catalog of published datasets:
       vectordata config add-catalog <URL-or-path>
  3. Browse what's available:
       vectordata datasets       # TUI
       vectordata datasets list  # text
```

When `settings.yaml` is in place, the output also lists configured
catalogs (or points at `add-catalog` if none are configured yet).

### Consistent error-message style

Across every subcommand, error and warning lines now use lowercase
`error:` / `warning:` prefixes — the same convention clap uses for its
own errors. Previously the prefix was a mix of `Error:`, `ERROR:`,
`Failed`, and `Refusing`, depending on which subcommand printed it.

If you grep stderr for `Error:` or `ERROR:` in any automation, switch
to a case-insensitive match or to lowercase `error:`.

### Stale `veks` references in help/error text

A handful of error messages still pointed users at `veks datasets
config add-catalog <URL>`. Since the explorer migrated into the
`vectordata` binary, those references were both wrong (the explorer
isn't owned by `veks` any more) and unactionable for users who only
have `vectordata` installed. They now correctly say `vectordata config
add-catalog <URL-or-path>`.

---

## What didn't change

- The Rust library API (`use vectordata::…`) — no signatures moved.
- The on-disk cache layout, settings file format, or catalog config
  shape.
- `veks` CLI behaviour. `veks datasets …` etc. continue to exist for
  veks users; the migration only affects the `vectordata` binary
  entry points.
- The catalog/transport stack — `s3://` catalog URLs work, HTTP
  catalogs work, local catalogs work; nothing about how datasets
  resolve has changed.

---

## If you script around the binary

The exit-code changes are the only externally-observable behavioural
shift. Audit any wrapper script that:

- Treats `vectordata datasets list --at <url>` as "always succeeds" —
  it now returns `1` if the URL is unreachable or resolves to no
  datasets.
- Invokes `vectordata explore …` non-interactively and ignores the
  return value — it now signals failure.
- Greps stderr for capitalised `Error:` / `ERROR:` prefixes — they're
  lowercase now.

Everything else (paths, asset names, on-disk shapes, library calls)
is unchanged.
