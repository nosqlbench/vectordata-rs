# merkle create

Generate merkle hash trees for all publishable data files.

## Usage (pipeline step)

```yaml
- id: generate-merkle
  run: merkle create
  source: .
  min-size: 0
```

## Behavior

Walks the workspace, finds all data files, and creates `.mref`
companion files containing SHA-256 hash trees. These enable
chunk-level integrity verification during remote download.
