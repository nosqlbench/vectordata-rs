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

## Special value syntax

| Syntax | Meaning |
|--------|---------|
| `42` | Literal string value |
| `count:file.fvec` | Record count of the file |
