// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

pub mod args;
pub mod config;
pub mod download;
pub mod expand;
pub mod logging;

pub use args::BulkDlArgs;

use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{error, info, warn};
use tokio::sync::Semaphore;

use config::{Config, StatusFile};
use download::{download_file, head_content_length};
use expand::expand_tokens;

/// Entry point for the bulkdl subcommand
pub async fn run(args: BulkDlArgs) {
    logging::init_logging();
    info!("veks bulkdl starting with config: {}", args.config.display());

    let config_text = fs::read_to_string(&args.config).unwrap_or_else(|e| {
        eprintln!("Error reading config file {}: {}", args.config.display(), e);
        std::process::exit(1);
    });

    let config: Config = serde_yaml::from_str(&config_text).unwrap_or_else(|e| {
        eprintln!("Error parsing config: {}", e);
        std::process::exit(1);
    });

    let multi_progress = MultiProgress::new();

    for dataset in &config.datasets {
        info!("Processing dataset: {}", dataset.name);

        // Ensure save directory exists
        fs::create_dir_all(&dataset.savedir).unwrap_or_else(|e| {
            eprintln!("Failed to create directory {}: {}", dataset.savedir, e);
            std::process::exit(1);
        });

        let status_path = Path::new(&dataset.savedir).join(".down-rs-status.json");
        let status = Arc::new(Mutex::new(StatusFile::load(&status_path)));

        // Expand all URLs from the template
        let all_files = expand_tokens(&dataset.baseurl, &dataset.tokens);
        let total = all_files.len();

        // Filter out already completed files
        let pending: Vec<(String, String)> = {
            let st = status.lock().unwrap();
            all_files
                .into_iter()
                .filter(|(_, filename)| !st.completed.contains(filename))
                .collect()
        };

        let already_done = total - pending.len();
        info!(
            "Dataset '{}': {} total files, {} already completed, {} to download",
            dataset.name,
            total,
            already_done,
            pending.len()
        );

        let pb = multi_progress.add(ProgressBar::new(total as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40.cyan/blue}] {pos}/{len} ({percent}%) — {per_sec} — ETA {eta} {msg}")
                .expect("invalid progress bar template")
                .progress_chars("=>-"),
        );
        pb.set_message(dataset.name.clone());
        pb.set_position(already_done as u64);

        if pending.is_empty() {
            pb.finish_with_message(format!("{} (complete)", dataset.name));
            continue;
        }

        let semaphore = Arc::new(Semaphore::new(dataset.concurrency as usize));
        let mut handles = Vec::new();

        for (url, filename) in pending {
            let sem = Arc::clone(&semaphore);
            let status = Arc::clone(&status);
            let status_path = status_path.clone();
            let savedir = dataset.savedir.clone();
            let tries = dataset.tries;
            let pb = pb.clone();

            let handle = tokio::spawn(async move {
                // Acquire semaphore permit to limit concurrency
                let _permit = sem.acquire().await.expect("semaphore closed");

                // All curl operations are blocking, so run in spawn_blocking
                tokio::task::spawn_blocking(move || {
                    let dest = Path::new(&savedir).join(&filename);

                    // Check if file already exists with correct size via HEAD
                    if dest.exists() {
                        if let Some(remote_len) = head_content_length(&url) {
                            if let Ok(meta) = fs::metadata(&dest) {
                                if meta.len() == remote_len {
                                    info!(
                                        "Skipping {} (already complete, size matches)",
                                        filename
                                    );
                                    let mut st = status.lock().unwrap();
                                    st.completed.push(filename.clone());
                                    let _ = st.save(&status_path);
                                    pb.inc(1);
                                    return;
                                }
                            }
                        }
                    }

                    // Download with retries
                    let mut last_err = String::new();
                    for attempt in 1..=tries {
                        info!("Downloading {} (attempt {}/{})", url, attempt, tries);
                        match download_file(&url, &dest) {
                            Ok(()) => {
                                info!("Completed: {}", filename);
                                let mut st = status.lock().unwrap();
                                st.completed.push(filename.clone());
                                let _ = st.save(&status_path);
                                pb.inc(1);
                                return;
                            }
                            Err(e) => {
                                last_err = e.clone();
                                warn!(
                                    "Attempt {}/{} failed for {}: {}",
                                    attempt, tries, url, e
                                );
                            }
                        }
                    }
                    error!(
                        "All {} attempts failed for {}: {}",
                        tries, url, last_err
                    );
                    pb.inc(1);
                })
                .await
                .expect("spawn_blocking panicked");
            });

            handles.push(handle);
        }

        // Wait for all downloads in this dataset
        for handle in handles {
            if let Err(e) = handle.await {
                error!("Task join error: {}", e);
            }
        }

        pb.finish_with_message(format!("{} (done)", dataset.name));
    }

    info!("veks bulkdl finished");
}
