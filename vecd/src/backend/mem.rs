// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The `mem:` in-process backend — **ephemeral**, lost on restart. For
//! tests and short-lived scratch namespaces only. A process-global
//! registry keyed by the `mem:<id>` endpoint makes two namespaces that
//! reference the same endpoint share one store within a daemon.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::backend::Backend;
use crate::model::VecdError;

/// One endpoint's ephemeral store: committed objects plus in-progress
/// upload staging blobs (sparse byte buffers), kept apart so staging is
/// invisible to reads/listings until finalized.
#[derive(Default)]
struct MemStore {
    objects: HashMap<String, Vec<u8>>,
    staging: HashMap<String, Vec<u8>>,
}

type Store = Arc<Mutex<MemStore>>;

fn registry() -> &'static Mutex<HashMap<String, Store>> {
    static REG: OnceLock<Mutex<HashMap<String, Store>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Open (or create) the shared in-memory store for a `mem:<id>` endpoint.
pub fn open_shared(endpoint: &str) -> Arc<dyn Backend> {
    let mut reg = registry().lock().unwrap();
    let store = reg.entry(endpoint.to_string()).or_default().clone();
    Arc::new(MemBackend { endpoint: endpoint.to_string(), store })
}

pub struct MemBackend {
    endpoint: String,
    store: Store,
}

impl Backend for MemBackend {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, VecdError> {
        Ok(self.store.lock().unwrap().objects.get(key).cloned())
    }

    fn put(&self, key: &str, data: &[u8]) -> Result<(), VecdError> {
        self.store.lock().unwrap().objects.insert(key.to_string(), data.to_vec());
        Ok(())
    }

    fn put_at(&self, staging_key: &str, offset: u64, chunk: &[u8]) -> Result<(), VecdError> {
        let mut s = self.store.lock().unwrap();
        let buf = s.staging.entry(staging_key.to_string()).or_default();
        let end = offset as usize + chunk.len();
        if buf.len() < end {
            buf.resize(end, 0); // zero-fill any hole before `offset`
        }
        buf[offset as usize..end].copy_from_slice(chunk);
        Ok(())
    }

    fn finalize_staged(&self, staging_key: &str, final_key: &str) -> Result<(), VecdError> {
        let mut s = self.store.lock().unwrap();
        // Finalizing an absent staging blob is an error (matches local's
        // rename-of-missing-source) — never a silent empty object.
        let blob = s
            .staging
            .remove(staging_key)
            .ok_or_else(|| VecdError::op(format!("no staging blob '{staging_key}' to finalize")))?;
        s.objects.insert(final_key.to_string(), blob);
        Ok(())
    }

    fn discard_staged(&self, staging_key: &str) -> Result<(), VecdError> {
        self.store.lock().unwrap().staging.remove(staging_key);
        Ok(())
    }

    fn head(&self, key: &str) -> Result<Option<u64>, VecdError> {
        Ok(self.store.lock().unwrap().objects.get(key).map(|v| v.len() as u64))
    }

    fn delete(&self, key: &str) -> Result<(), VecdError> {
        self.store.lock().unwrap().objects.remove(key);
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, VecdError> {
        Ok(self
            .store
            .lock()
            .unwrap()
            .objects
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect())
    }

    fn describe(&self) -> String {
        self.endpoint.clone()
    }
}
