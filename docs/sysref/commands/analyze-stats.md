# analyze stats

Per-dimension statistics across all vectors.

## Usage

```bash
veks pipeline analyze stats --source <file>
```

## Example

```bash
veks pipeline analyze stats --source profiles/base/base_vectors.fvec
```

```
  source: ./profiles/base/base_vectors.fvec (f32, dim=128, 1000 records)
Global statistics (1000 vectors, 128 dims, 128000 values):
  Mean:     0.001584
  StdDev:   0.576655
  Min:      -0.999995
  Max:      0.999986
  Skewness: -0.0020
  Kurtosis: 1.8018
```
