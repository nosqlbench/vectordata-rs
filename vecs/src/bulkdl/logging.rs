// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

use env_logger::Builder;
use log::LevelFilter;
use std::fs::OpenOptions;

/// Initializes file-based logging to `vecs-bulkdl.log`
pub fn init_logging() {
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("vecs-bulkdl.log")
        .expect("Failed to open vecs-bulkdl.log");

    Builder::new()
        .filter_level(LevelFilter::Info)
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .init();
}
