# analyze check-endian

Verify byte order of a vector file.

## Usage

```bash
veks pipeline analyze check-endian --source <file>
```

## Example

```bash
veks pipeline analyze check-endian --source profiles/base/base_vectors.fvecs
```

```
File: ./profiles/base/base_vectors.fvecs
  Size: 516000 bytes
  Format: fvec (element width: 4 bytes)
  Little-endian: dim=128, valid=true, vectors=1000
  Big-endian:    dim=2147483648, valid=false, vectors=0
  Result: LITTLE-ENDIAN (correct)
```
