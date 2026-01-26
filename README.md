# Vector Data Tools

`vectordata` is a Rust library for efficiently accessing vector datasets described by a `dataset.yaml` configuration. It supports seamless access to both local file system datasets and remote datasets via HTTP/HTTPS, using efficient Range requests for on-demand data retrieval.

## Features

-   **Unified Access**: standardized interface for reading vector data regardless of location (local disk or remote URL).
-   **Dataset Configuration**: Parses `dataset.yaml` to understand dataset structure and profiles.
-   **Efficient Remote Access**: Uses HTTP Range requests to fetch only the necessary vector data without downloading entire files.
-   **Memory Mapping**: Utilizes memory mapping for high-performance reading of local files.
-   **Supported Formats**: Handles `.fvec` (float vectors) and `.ivec` (integer vectors) formats.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
vectordata = "0.1.0"
```

## Usage

### 1. Reading a Local Dataset

```rust
use vectordata::TestDataGroup;

fn main() -> anyhow::Result<()> {
    // Load from a local directory containing dataset.yaml
    let group = TestDataGroup::load("./path/to/dataset")?;

    if let Some(view) = group.profile("default") {
        let base_vectors = view.base_vectors()?;
        
        println!("Loaded {} vectors with dimension {}", 
            base_vectors.count(), 
            base_vectors.dim()
        );
        
        // Read the 100th vector
        let vector = base_vectors.get(100)?;
        println!("Vector 100: {:?}", vector);
    }
    
    Ok(())
}
```

### 2. Reading a Remote Dataset

Accessing a remote dataset is just as simple. Provide the URL to the dataset directory or the `dataset.yaml` file itself.

```rust
use vectordata::TestDataGroup;

fn main() -> anyhow::Result<()> {
    // URL to a remote dataset
    let url = "https://example.com/datasets/glove-100/dataset.yaml";
    
    // The library automatically handles HTTP requests
    let group = TestDataGroup::load(url)?;

    if let Some(view) = group.profile("small") {
        // Accessing this triggers a Range request for only the necessary bytes
        let query_vectors = view.query_vectors()?;
        
        println!("Remote query vectors: {}", query_vectors.count());
    }

    Ok(())
}
```

## Dataset Configuration (`dataset.yaml`)

This library expects a `dataset.yaml` file in the root of the dataset directory.

```yaml
attributes:
  distance_function: COSINE
  dimension: 128

profiles:
  default:
    base_vectors: base.fvec
    query_vectors: query.fvec
    neighbor_indices: ground_truth.ivec
    neighbor_distances: distances.fvec
  
  small:
    base_vectors:
      source: base.fvec
      window: 0..1000
```
