<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorial: Exploring Vector Data Interactively

Three TUI surfaces under `veks interact` for poking at vector data:
the **values grid** (raw cell viewer with heatmap), the **unified
explorer** (sampled analytics — norms, distances, eigenvalues, PCA),
and the **shell** (REPL for ad-hoc queries). All three open any
vector source — a local `.fvec` / `.mvec` file, a catalog dataset
specifier (`name:profile:facet`), or a profile from a remote catalog.

## Prerequisites

- A built `veks` binary (`cargo install --path veks`)
- Either a local vector file or a dataset configured via
  `veks datasets list` (or `veks datasets add-catalog`)

If you skip `--source` and `--dataset`, all three commands open the
dataset picker so you can pick interactively.

---

## 1. The values grid — `veks interact values`

A scrollable spreadsheet of vector values. One row per ordinal, one
column per dimension, with decimal-point alignment and a 24-bit-color
heatmap.

```bash
# Open by catalog name; defaults: --start 0 --digits 4
veks interact values --dataset my-dataset

# Or open a local file
veks interact values --source ./data/base.fvec

# Jump straight to the middle of a 10M dataset at 5 sig digits
veks interact values --dataset my-dataset --start 5000000 --digits 5
```

The header reports `count`, `dim`, current `row` / `dim_col`
positions, `sig`, `scope`, `palette`, `curve`, and `view` (raw vs
L2-normalized). The leftmost data column shows each row's L2 norm
(in cyan) so you can spot unnormalized vectors at a glance. Walls
appear every 8 dims to keep your eye anchored when scrolling
sideways.

### Keybindings

| Key                  | Action |
|----------------------|--------|
| `hjkl` or arrows     | Scroll one cell |
| `HJKL` or PgUp/PgDn  | Scroll one page |
| `[` `]`              | Top / bottom of dataset |
| `0` `$`              | First / last dim |
| `+` `-`              | Adjust significant digits (1–6) |
| `n`                  | Cycle heatmap normalization scope (per-vector → per-column → global → off) |
| `p`                  | Cycle palette |
| `c`                  | Cycle intensity curve (linear / sqrt / square / sigmoid) |
| `N` (capital)        | Toggle "show as L2-normalized" |
| `g`                  | Jump-to-ordinal (type a number, Enter) |
| `Esc`                | Back to picker (or exit if launched with explicit source) |
| `q`                  | Quit |

### Palettes

Cycle with `p`. Default is **blue-orange** (color-blind safe).

| Palette       | Best for |
|---------------|----------|
| `blue-orange` | Diverging (sign matters); safe for protan/deutan/tritan |
| `blue-yellow` | Diverging with maximum luminance contrast (legible in print) |
| `blue-red`    | Diverging, high saturation (avoid for color blindness) |
| `cividis`     | Sequential, designed for deuteranopia |
| `mono`        | Universal grayscale |
| `turbo`       | Perceptually-uniform multi-stop scatter palette |
| `spectrum`    | Saturated rainbow — every bin pops |

### Intensity curves

Cycle with `c`. Reshape `|t|` before palette lookup:

| Curve     | Effect |
|-----------|--------|
| `linear`  | Faithful (default) |
| `sqrt`    | Boosts the low end — surfaces near-mid outliers |
| `square`  | Suppresses the low end — only extremes stand out |
| `sigmoid` | Sharpens the mid-range transition |

### L2-normalized view

Press `N` to toggle. Each visible row is divided by its L2 norm
before formatting, so every row reads as a unit vector — handy for
comparing direction across rows whose magnitudes differ. The L2
column keeps showing the **original** norm (after normalization
every row would read 1.000 there, which is useless), and the heatmap
re-bounds against the post-scale values so the colors match what's
displayed. The underlying read cache is untouched; toggle off to
return to raw values without a re-read.

---

## 2. The unified explorer — `veks interact explore`

Loads a sample of vectors (default 50K) and runs four analytical
phases in the background: norms → pairwise distances → eigenvalues
→ PCA projection. Old data stays visible across sample-size
restarts so the screen never blanks.

```bash
veks interact explore --dataset my-dataset --sample 50000

# Streaming sample (default) — first N vectors, contiguous, fastest
# over HTTP. Other modes:
veks interact explore --dataset my-dataset --sample 50000 --sample-mode clumped
veks interact explore --dataset my-dataset --sample 5000  --sample-mode sparse
```

The reader thread groups indices into contiguous runs and uses the
batched `get_f32_range` API, so a streaming or clumped sample of
50K vectors over HTTP collapses to a handful of parallel chunk
fetches instead of 50K serialized round-trips. Sparse mode degrades
gracefully to per-index calls.

### Tabs (number keys)

| Key | View |
|-----|------|
| `1` | PCA scatter (3D / 4D / 5D — see `c/C/x/X` to permute axes) |
| `2` | Per-dim distribution |
| `3` | PC loadings |
| `4` | Eigenvalue analysis (scree / cumulative variance / log decay — `m` cycles) |
| `5` | Norm histogram |
| `6` | Sorted norm curve |
| `7` | Distance histogram |
| `8` | Sorted distance curve |
| `?` | Toggle help overlay |
| `/` | Toggle per-view theory description |
| `r` | Restart computation |
| `+` `-` | Adjust sample size |
| `q` | Quit |

### PCA scatter palette / curve (`p` and `f`)

The 4D / 5D scatter colors points by PC4 (and the 5D variant uses
PC5 for brightness). Press `p` to cycle palettes and `f` to cycle
the intensity curve — same set as the values grid. Default is
**Turbo** because the high-stop ramp keeps every PC4 bin sharply
distinct, which is the whole point of a 4D scatter.

For divergent palettes the renderer folds around the midpoint so
both sides spread out from neutral; for sequential palettes (Turbo,
Spectrum, Cividis) it uses the raw `t` so the multi-stop gradient
isn't collapsed onto one half.

---

## 3. The shell — `veks interact shell`

A REPL for one-off queries against a vector file. Lighter than the
unified explorer — no precompute, no graphics — and useful from
scripts via the batch form.

```bash
# Interactive
veks interact shell --source ./data/base.fvec

# Batch: semicolon-separated commands, one result per line, then exit
veks interact shell --source ./data/base.fvec "info; range 0 5; norm 0"
```

Available REPL commands: `info`, `get <i>`, `range <start> <end>`,
`head <n>`, `tail <n>`, `dist <i> <j>` (or `distance`), `norm <i>`
(or `norms` for stats), `stats`, `help`, `quit`. Tab-completion
covers the command names.

---

## When to use which

| You want to … | Use |
|---------------|-----|
| Stare at the actual numbers in cells | `interact values` |
| Understand the vector space (norms, PCA, etc.) | `interact explore` |
| Get one specific value or quick stat in a script | `interact shell` |

All three open the same dataset picker if you don't pass `--source`
or `--dataset`, so the natural flow is: pick a dataset, scan the
analytics with `explore`, then switch to `values` to drill into
individual rows that looked anomalous.
