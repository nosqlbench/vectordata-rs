// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Command registry — maps command path strings to factory functions.
//!
//! Each registered command provides a factory that creates a fresh `CommandOp`
//! instance. The runner looks up commands by their `run` field in step
//! definitions.

use std::collections::HashMap;

use super::command::CommandOp;

/// Factory function type: creates a new `CommandOp` instance.
pub type CommandFactory = fn() -> Box<dyn CommandOp>;

/// Registry of available pipeline commands.
///
/// Commands are registered by their canonical path (e.g. `"import"`,
/// `"convert file"`, `"analyze describe"`). The runner resolves step `run`
/// fields against this registry.
pub struct CommandRegistry {
    factories: HashMap<String, CommandFactory>,
}

impl CommandRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        CommandRegistry {
            factories: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with all built-in commands.
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        super::commands::register_all(&mut reg);
        reg
    }

    /// Register a command factory under the given path.
    pub fn register(&mut self, command_path: &str, factory: CommandFactory) {
        self.factories.insert(command_path.to_string(), factory);
    }

    /// Look up a command factory by path.
    pub fn get(&self, command_path: &str) -> Option<&CommandFactory> {
        self.factories.get(command_path)
    }

    /// List all registered command paths.
    pub fn command_paths(&self) -> Vec<&str> {
        let mut paths: Vec<&str> = self.factories.keys().map(|s| s.as_str()).collect();
        paths.sort();
        paths
    }
}

impl Default for CommandRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_register_and_lookup() {
        let reg = CommandRegistry::new();
        assert!(reg.command_paths().is_empty());

        // Test the with_builtins path
        let reg = CommandRegistry::with_builtins();
        assert!(!reg.command_paths().is_empty());
        assert!(reg.get("transform convert").is_some());
        assert!(reg.get("compute knn").is_some());
        assert!(reg.get("analyze describe").is_some());
        assert!(reg.get("nonexistent command").is_none());
    }

    #[test]
    fn test_registry_command_paths() {
        let reg = CommandRegistry::with_builtins();
        let paths = reg.command_paths();
        assert!(paths.contains(&"transform convert"));
        assert!(paths.contains(&"compute knn"));
        assert!(paths.contains(&"analyze describe"));
    }

    #[test]
    fn test_registry_create_via_get() {
        let reg = CommandRegistry::with_builtins();
        let factory = reg.get("transform convert").expect("transform convert not found");
        let cmd = factory();
        assert_eq!(cmd.command_path(), "transform convert");
    }
}
