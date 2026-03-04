//! Input/Output operations for reading vector data.
//!
//! This module provides the `VectorReader` trait and implementations for reading
//! vectors from local files (memory-mapped) and remote HTTP sources.
//!
//! # Supported Formats
//!
//! - **fvec**: A binary format for floating-point vectors.
//!   - Header: 4 bytes (int32) representing the dimension.
//!   - Data: Sequence of vectors, each starting with the dimension (4 bytes) followed by `dim` floats.
//! - **ivec**: Similar to fvec but for 32-bit integers (used for ground truth indices).
//! - **hvec**: A binary format for half-precision (f16) vectors.
//!   - Header: 4 bytes (int32) representing the dimension.
//!   - Data: Sequence of vectors, each starting with the dimension (4 bytes) followed by `dim` f16 values (2 bytes each).

use std::fs::File;
use std::io::{self, Cursor};
use std::marker::PhantomData;
use std::path::Path;
use memmap2::Mmap;
use byteorder::{LittleEndian, ReadBytesExt};
use thiserror::Error;
use reqwest::blocking::Client;
use reqwest::header::{CONTENT_LENGTH, RANGE};
use url::Url;

/// Errors that can occur during vector I/O operations.
#[derive(Error, Debug)]
pub enum IoError {
    /// Wrapper for standard IO errors.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// Wrapper for HTTP errors from reqwest.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    /// Error indicating invalid file format or data.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    /// Error indicating an access out of valid bounds.
    #[error("Index out of bounds: {0}")]
    OutOfBounds(usize),
}

/// A trait for reading vectors from a data source.
///
/// Implementations handle the underlying storage details (e.g., file, network).
pub trait VectorReader<T>: Send + Sync {
    /// Returns the dimension of the vectors.
    fn dim(&self) -> usize;
    /// Returns the total number of vectors available.
    fn count(&self) -> usize;
    /// Retrieves the vector at the specified index.
    fn get(&self, index: usize) -> Result<Vec<T>, IoError>;
}

/// A `VectorReader` backed by a memory-mapped file.
///
/// This implementation is efficient for local files as it relies on the OS page cache.
#[derive(Debug)]
pub struct MmapVectorReader<T> {
    mmap: Mmap,
    dim: usize,
    count: usize,
    entry_size: usize,
    #[allow(dead_code)]
    header_size: usize, // usually 4 bytes for dim
    _phantom: PhantomData<T>,
}

impl MmapVectorReader<f32> {
    /// Opens a local `.fvec` file for reading floating-point vectors.
    pub fn open_fvec(path: &Path) -> Result<Self, IoError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        if mmap.len() < 4 {
            return Err(IoError::InvalidFormat("File too short".into()));
        }

        let mut cursor = Cursor::new(&mmap[..]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        
        if dim == 0 {
             return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let entry_size = 4 + dim * 4; // 4 bytes for dim header (repeated per vector) + vectors
        
        // Verify size alignment
        if mmap.len() % entry_size != 0 {
             // It's possible the last record is incomplete or there's trailing data, but standard fvecs align
             // We'll calculate count based on integer division
        }
        
        let count = mmap.len() / entry_size;

        Ok(Self {
            mmap,
            dim,
            count,
            entry_size,
            header_size: 4,
            _phantom: PhantomData,
        })
    }
}

impl VectorReader<f32> for MmapVectorReader<f32> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Result<Vec<f32>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        // Verify the dimension marker at start of record
        let mut cursor = Cursor::new(&self.mmap[start..start + 4]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
             return Err(IoError::InvalidFormat(format!("Record at index {} has mismatched dimension {}", index, dim)));
        }

        let vector_start = start + 4;
        let vector_end = vector_start + self.dim * 4;
        
        let mut vector = Vec::with_capacity(self.dim);
        let mut cursor = Cursor::new(&self.mmap[vector_start..vector_end]);
        
        for _ in 0..self.dim {
            vector.push(cursor.read_f32::<LittleEndian>()?);
        }

        Ok(vector)
    }
}

impl MmapVectorReader<i32> {
    /// Opens a local `.ivec` file for reading integer vectors (e.g., indices).
    pub fn open_ivec(path: &Path) -> Result<Self, IoError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        if mmap.len() < 4 {
            return Err(IoError::InvalidFormat("File too short".into()));
        }

        let mut cursor = Cursor::new(&mmap[..]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        
        if dim == 0 {
             return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let entry_size = 4 + dim * 4;
        let count = mmap.len() / entry_size;

        Ok(Self {
            mmap,
            dim,
            count,
            entry_size,
            header_size: 4,
            _phantom: PhantomData,
        })
    }
}

impl VectorReader<i32> for MmapVectorReader<i32> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Result<Vec<i32>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        let mut cursor = Cursor::new(&self.mmap[start..start + 4]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
             return Err(IoError::InvalidFormat(format!("Record at index {} has mismatched dimension {}", index, dim)));
        }

        let vector_start = start + 4;
        let vector_end = vector_start + self.dim * 4;
        
        let mut vector = Vec::with_capacity(self.dim);
        let mut cursor = Cursor::new(&self.mmap[vector_start..vector_end]);
        
        for _ in 0..self.dim {
            vector.push(cursor.read_i32::<LittleEndian>()?);
        }

        Ok(vector)
    }
}

impl MmapVectorReader<half::f16> {
    /// Opens a local `.hvec` file for reading half-precision (f16) vectors.
    pub fn open_hvec(path: &Path) -> Result<Self, IoError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 4 {
            return Err(IoError::InvalidFormat("File too short".into()));
        }

        let mut cursor = Cursor::new(&mmap[..]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;

        if dim == 0 {
            return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let entry_size = 4 + dim * 2; // 4 bytes for dim header + dim * 2 bytes per f16

        let count = mmap.len() / entry_size;

        Ok(Self {
            mmap,
            dim,
            count,
            entry_size,
            header_size: 4,
            _phantom: PhantomData,
        })
    }
}

impl VectorReader<half::f16> for MmapVectorReader<half::f16> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> Result<Vec<half::f16>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        let mut cursor = Cursor::new(&self.mmap[start..start + 4]);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
            return Err(IoError::InvalidFormat(format!(
                "Record at index {} has mismatched dimension {}",
                index, dim
            )));
        }

        let vector_start = start + 4;
        let vector_end = vector_start + self.dim * 2;

        let mut vector = Vec::with_capacity(self.dim);
        let mut cursor = Cursor::new(&self.mmap[vector_start..vector_end]);

        for _ in 0..self.dim {
            vector.push(half::f16::from_bits(cursor.read_u16::<LittleEndian>()?));
        }

        Ok(vector)
    }
}

/// A VectorReader that reads data from an HTTP(S) URL using Range requests.
///
/// This reader expects the remote file to be formatted as a standard vector file
/// (dimension header followed by vectors). It performs minimal network requests:
/// 1. Initial GET/HEAD to determine dimension and file size.
/// 2. Range GET requests for specific vector reads.
#[derive(Debug)]
pub struct HttpVectorReader<T> {
    client: Client,
    url: Url,
    dim: usize,
    count: usize,
    entry_size: usize,
    #[allow(dead_code)]
    total_size: u64,
    _phantom: PhantomData<T>,
}

impl HttpVectorReader<f32> {
    /// Opens a floating-point vector file from a URL.
    pub fn open_fvec(url: Url) -> Result<Self, IoError> {
        let client = Client::new();
        
        // Read header (dimension)
        let resp = client.get(url.clone())
            .header(RANGE, "bytes=0-3")
            .send()?
            .error_for_status()?;
            
        let bytes = resp.bytes()?;
        if bytes.len() < 4 {
             return Err(IoError::InvalidFormat("File too short".into()));
        }
        let mut cursor = Cursor::new(bytes);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim == 0 {
             return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        // Get total size via HEAD
        let resp = client.head(url.clone()).send()?.error_for_status()?;
        let total_size = resp.headers()
            .get(CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| IoError::InvalidFormat("Missing Content-Length".into()))?;

        let entry_size = 4 + dim * 4;
        let count = (total_size / entry_size as u64) as usize;

        Ok(Self {
            client,
            url,
            dim,
            count,
            entry_size,
            total_size,
            _phantom: PhantomData,
        })
    }
}

impl VectorReader<f32> for HttpVectorReader<f32> {
    fn dim(&self) -> usize { self.dim }
    fn count(&self) -> usize { self.count }

    fn get(&self, index: usize) -> Result<Vec<f32>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        let end = start + self.entry_size - 1;

        let resp = self.client.get(self.url.clone())
            .header(RANGE, format!("bytes={}-{}", start, end))
            .send()?
            .error_for_status()?;

        let bytes = resp.bytes()?;
        let mut cursor = Cursor::new(bytes);
        
        // Read and check dimension
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
             return Err(IoError::InvalidFormat(format!("Record at index {} has mismatched dimension {}", index, dim)));
        }

        let mut vector = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            vector.push(cursor.read_f32::<LittleEndian>()?);
        }
        Ok(vector)
    }
}

impl HttpVectorReader<i32> {
    /// Opens a integer vector file from a URL.
    pub fn open_ivec(url: Url) -> Result<Self, IoError> {
        let client = Client::new();
        
        let resp = client.get(url.clone())
            .header(RANGE, "bytes=0-3")
            .send()?
            .error_for_status()?;
            
        let bytes = resp.bytes()?;
        let mut cursor = Cursor::new(bytes);
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        
        if dim == 0 {
             return Err(IoError::InvalidFormat("Dimension cannot be 0".into()));
        }

        let resp = client.head(url.clone()).send()?.error_for_status()?;
        let total_size = resp.headers()
            .get(CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| IoError::InvalidFormat("Missing Content-Length".into()))?;

        let entry_size = 4 + dim * 4;
        let count = (total_size / entry_size as u64) as usize;

        Ok(Self {
            client,
            url,
            dim,
            count,
            entry_size,
            total_size,
            _phantom: PhantomData,
        })
    }
}

impl VectorReader<i32> for HttpVectorReader<i32> {
    fn dim(&self) -> usize { self.dim }
    fn count(&self) -> usize { self.count }

    fn get(&self, index: usize) -> Result<Vec<i32>, IoError> {
        if index >= self.count {
            return Err(IoError::OutOfBounds(index));
        }

        let start = index * self.entry_size;
        let end = start + self.entry_size - 1;

        let resp = self.client.get(self.url.clone())
            .header(RANGE, format!("bytes={}-{}", start, end))
            .send()?
            .error_for_status()?;

        let bytes = resp.bytes()?;
        let mut cursor = Cursor::new(bytes);
        
        let dim = cursor.read_i32::<LittleEndian>()? as usize;
        if dim != self.dim {
             return Err(IoError::InvalidFormat(format!("Record at index {} has mismatched dimension {}", index, dim)));
        }

        let mut vector = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            vector.push(cursor.read_i32::<LittleEndian>()?);
        }
        Ok(vector)
    }
}

