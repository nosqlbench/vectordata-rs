# generate docs-html

Render every Markdown document of a dataset to a self-contained HTML
mirror.

A dataset is documented in Markdown — `README.md`, `LICENSE.md` and
their kin at the root, and everything under `docs/` — and that Markdown
is the source of truth. Hosted as plain objects on S3 or any static
store, Markdown is shown as text, so this step projects it to HTML that
needs nothing else: the stylesheet is inline, no script, nothing
fetched from anywhere. A page opens the same from the bucket, from a
local copy, or from any static host.

## Usage

Declared as a finalize step after the generated reference:

```yaml
  - id: generate-docs-html
    run: generate docs-html
    description: Render the Markdown documents to a self-contained HTML mirror under docs/html/
    after:
    - generate-docs
    finalize: true
    source: .
    output: docs/html
```

`veks prepare bootstrap` declares it for a new dataset.

| Option | Default | Description |
|---|---|---|
| `source` | `.` | The dataset directory; its root documents and `docs/` are rendered |
| `output` | `docs/html` | The folder of the rendered tree, under the dataset directory |

## The mirror

The rendered tree mirrors the Markdown tree from the dataset root:

| Source | Page |
|---|---|
| `README.md` | `docs/html/README.html` |
| `LICENSE.md` | `docs/html/LICENSE.html` |
| `docs/how-it-was-made.md` | `docs/html/docs/how-it-was-made.html` |
| `docs/dataset.md` (generated) | `docs/html/docs/dataset.html` |

Every link is rewritten relative to the page it sits in: a link to a
Markdown document points at that document's page, a link to an image
or a data file points back at the real file (nothing but text is
copied), and external links, anchors and absolute paths are left as
they are. The page's title is the document's first heading. The
rendered tree is never itself a source.

## Freshness

The step's outputs are the pages, and every Markdown document is an
input of the finalize pass by content (sysref §4.3), so editing one —
by hand, or the reference being regenerated — re-renders on the next
run. Rendered pages are documents: never merkled, accounted for as the
step's outputs, published with the data.
