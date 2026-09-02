# state set

Set a pipeline variable in variables.yaml.

## Usage

```bash
veks pipeline state set --name <key> --value <value>
```

## Example

```bash
veks pipeline state set --name example_var --value 42
```

```
  example_var = 42
```

## Dataset attributes

With `--attribute true` the value is recorded as a dataset attribute
in `dataset.yaml` rather than as a pipeline variable. Only the known
attribute keys are accepted; an unknown key is an error. This is the
back-fill route for embedding provenance on a dataset whose embed step
predates recording it:

```yaml
- id: record-embed-revision
  run: state set
  name: model_revision
  value: 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
  attribute: true
```

## Special value syntax

| Syntax | Meaning |
|--------|---------|
| `42` | Literal string value |
| `count:file.fvecs` | Record count of the file |
