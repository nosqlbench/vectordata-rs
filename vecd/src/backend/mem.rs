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

type Store = Arc<Mutex<HashMap<String, Vec<u8>>>>;

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
        Ok(self.store.lock().unwrap().get(key).cloned())
    }

    fn put(&self, key: &str, data: &[u8]) -> Result<(), VecdError> {
        self.store.lock().unwrap().insert(key.to_string(), data.to_vec());
        Ok(())
    }

    fn head(&self, key: &str) -> Result<Option<u64>, VecdError> {
        Ok(self.store.lock().unwrap().get(key).map(|v| v.len() as u64))
    }

    fn delete(&self, key: &str) -> Result<(), VecdError> {
        self.store.lock().unwrap().remove(key);
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, VecdError> {
        Ok(self
            .store
            .lock()
            .unwrap()
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect())
    }

    fn describe(&self) -> String {
        self.endpoint.clone()
    }
}
