// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: embed a parquet text column into an npy vector
//! artifact with an in-process candle backend (Qwen3-Embedding).
//!
//! This is the plan-D3 option (a) end state: the embed stage runs inside
//! `veks run`, so model identity, revision, and every knob that can change
//! output bytes are step options and therefore provenance axes. Row i of
//! the output embeds parquet row i — the ordinal-identity contract that
//! `verify alignment` gates downstream.
//!
//! Feature-gated (`embed`); CUDA acceleration is a further feature flip
//! (`embed-cuda`) plus `device: cuda` — sized for A100/H100-class hosts at
//! full-corpus scale, while CPU covers pilot-scale runs.

#[cfg(feature = "embed-cuda-flash")]
mod kernels;
mod qwen3;

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::pipeline::atomic_write::AtomicWriter;
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, ValueCompletions, render_options_table,
};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};

/// Pipeline command: candle-backed text embedding.
pub struct GenerateEmbedOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(GenerateEmbedOp)
}

const DEFAULT_MODEL: &str = "Qwen/Qwen3-Embedding-0.6B";

impl CommandOp for GenerateEmbedOp {
    fn command_path(&self) -> &str {
        "generate embed"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_GENERATE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Embed a parquet text column into an npy vector artifact".into(),
            body: format!(
                r#"# generate embed

Embed a parquet text column into an npy vector artifact.

## Description

Reads the `column` strings of `source` in row order, embeds each with an
in-process candle backend (default model {model}, last-token pooling,
unit-normalized), and writes a C-order f32 `.npy` of shape
`[rows, hidden]` where **row i embeds parquet row i** — the ordinal
contract that `verify alignment` asserts downstream. Model weights are
fetched once into the shared HuggingFace cache and reused.

## Determinism and provenance

Model id, revision, batching, and length caps are all step options, so
embedding identity is fully recorded in provenance. Identical options on
identical input produce identical output bytes on a given device class.

## Devices

`device: cpu` runs everywhere; `device: cuda[:N]` needs a binary built
with the `embed-cuda` feature (A100/H100-class hosts; pair with
`dtype: bf16` and a larger `batch-size`). `device: auto` picks CUDA when
compiled in and available, else CPU. `dtype: auto` maps to f32 on CPU and
bf16 on CUDA.

## Options

{opts}"#,
                model = DEFAULT_MODEL,
                opts = render_options_table(&options)
            ),
        }
    }

    /// Verified against every `options.*("...")` read in `execute`.
    /// This step runs for hours and its `range` option is what bounds the
    /// work, so a silently-ignored option here is expensive in a way it is
    /// not for a command that finishes in seconds.
    fn options_declared_complete(&self) -> bool {
        true
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc {
                name: "mem".into(),
                description: "Model weights + activations".into(),
                adjustable: false,
            },
            ResourceDesc {
                name: "threads".into(),
                description: "CPU gemm parallelism".into(),
                adjustable: false,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source = match options.require("source") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let output = match options.require("output") {
            Ok(s) => resolve_path(s, &ctx.workspace),
            Err(e) => return error_result(e, start),
        };
        let column = options.get("column").unwrap_or("text").to_string();
        let model_id = options.get("model").unwrap_or(DEFAULT_MODEL).to_string();
        let revision = options.get("revision").unwrap_or("main").to_string();
        let batch_size: usize = match options.parse_or("batch-size", 128) {
            Ok(n) if n > 0 => n,
            Ok(_) => return error_result("batch-size must be > 0".into(), start),
            Err(e) => return error_result(e, start),
        };
        let max_length: usize = match options.parse_or("max-length", 1024) {
            Ok(n) if n >= 2 => n,
            Ok(_) => return error_result("max-length must be >= 2".into(), start),
            Err(e) => return error_result(e, start),
        };
        let devices = match resolve_devices(options.get("device").unwrap_or("auto")) {
            Ok(d) => d,
            Err(e) => return error_result(e, start),
        };
        let dtype = match resolve_dtype(options.get("dtype").unwrap_or("auto"), &devices[0]) {
            Ok(d) => d,
            Err(e) => return error_result(e, start),
        };

        // A row window over the source. Embedding a corpus larger than
        // memory means working through it in passes, and each pass is its
        // own step with its own output and its own freshness — so an
        // interrupted multi-day embed resumes at the pass boundary rather
        // than from the beginning.
        let (row_start, row_end) = match options.get("range") {
            None => (0u64, None),
            Some(spec) => match super::gen_extract::parse_range(spec) {
                Ok(r) => (r.start as u64, r.end.map(|e| e as u64)),
                Err(e) => return error_result(format!("invalid range '{}': {}", spec, e), start),
            },
        };
        if let Some(end) = row_end
            && end <= row_start
        {
            return error_result(
                format!("empty range: start {} is not before end {}", row_start, end),
                start,
            );
        }

        // Row count first, from footer metadata alone — the sink needs the
        // exact figure up front (an npy header encodes it) and the progress
        // bar needs a total. Text itself is streamed below rather than
        // collected: at the ~944 B/row measured on real passage tables a
        // 50M-row window is ~47 GB resident and a 532M-row one is ~500 GB,
        // so collecting made the window a memory decision and put an
        // unbounded allocation one mis-set option away.
        let n_texts = match veks_core::formats::passage_table::count_text_rows_range(
            &source, row_start, row_end,
        ) {
            Ok(n) => n as usize,
            Err(e) => return error_result(e, start),
        };
        if n_texts == 0 {
            return error_result(
                match row_end {
                    Some(end) => format!(
                        "no rows in {} for range [{},{}) — the window starts past the end",
                        source.display(),
                        row_start,
                        end
                    ),
                    None => format!("no rows in {}", source.display()),
                },
                start,
            );
        }
        ctx.ui.log(&format!(
            "embedding {} row(s){} of {}:{} with {} (rev {}, {:?}x{}/{:?}, batch {}, max-length {})",
            n_texts,
            match row_end {
                Some(end) => format!(" [{},{})", row_start, end),
                None if row_start > 0 => format!(" [{},end)", row_start),
                None => String::new(),
            },
            source.display(),
            column,
            model_id,
            revision,
            device_name(&devices[0]),
            devices.len(),
            dtype,
            batch_size,
            max_length
        ));
        // Guardrail: a large CPU embed is almost always a misconfiguration
        // (CPU-only build, `device: cpu` from a stale recipe) — the GPU
        // path is ~3 orders of magnitude faster. Warn with the arithmetic
        // up front instead of letting a silent multi-hour grind start.
        if devices.iter().all(|d| !d.is_cuda()) && n_texts > 10_000 {
            ctx.ui.log(&format!(
                "WARNING: embedding {} row(s) on CPU — expect hours (CPU runs at roughly \
                 single-digit rows/s vs ~2,500/s measured on a 2-GPU host). If this host \
                 has GPUs, build with --features embed-cuda-flash and use device: auto; \
                 otherwise consider running this step on a GPU node.",
                n_texts
            ));
        }

        // ── Model + tokenizer, via the shared HF cache ───────────────────
        let fetch = ctx.ui.spinner("fetch model");
        let files = match fetch_model_files(&model_id, &revision) {
            Ok(f) => f,
            Err(e) => {
                fetch.finish();
                return error_result(e, start);
            }
        };
        fetch.finish();

        let config: qwen3::Config = match std::fs::read_to_string(&files.config)
            .map_err(|e| format!("read config: {}", e))
            .and_then(|s| serde_json::from_str(&s).map_err(|e| format!("parse config: {}", e)))
        {
            Ok(c) => c,
            Err(e) => return error_result(e, start),
        };
        let tokenizer = match tokenizers::Tokenizer::from_file(&files.tokenizer) {
            Ok(t) => t,
            Err(e) => return error_result(format!("load tokenizer: {}", e), start),
        };
        let eos = match config
            .eos_token_id
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        {
            Some(id) => id,
            None => return error_result("cannot determine EOS token id".into(), start),
        };

        let load = ctx.ui.spinner("load weights");
        let mut models: Vec<qwen3::EmbeddingModel> = Vec::with_capacity(devices.len());
        for device in &devices {
            let vb = match unsafe {
                VarBuilder::from_mmaped_safetensors(&files.weights, dtype, device)
            } {
                Ok(vb) => vb,
                Err(e) => {
                    load.finish();
                    return error_result(format!("load weights: {}", e), start);
                }
            };
            match qwen3::EmbeddingModel::new(&config, vb) {
                Ok(m) => models.push(m),
                Err(e) => {
                    load.finish();
                    return error_result(format!("build model: {}", e), start);
                }
            }
        }
        load.finish();

        // ── Embed: tokenizer producer + one worker thread per device ─────
        // Tokenization (EOS-terminated, capped) runs on a producer thread in
        // chunks, each chunk length-sorted into batches (chunk-local sorting
        // keeps pad waste at the ~1% the global sort measured), so the GPUs
        // start embedding while later chunks still tokenize — at production
        // step sizes the up-front tokenize stretch was ~9% of the run with
        // idle GPUs. encode_batch fans out across cores via the tokenizers
        // crate's internal rayon pool. Workers pull batches from a shared
        // channel; the main thread scatters results by absolute row index
        // and drives progress. On the first error the receiver stops,
        // pending sends fail, and every thread unwinds before the scope
        // joins (the producer also checks an abort flag between chunks).
        const TOKENIZE_CHUNK: usize = 65_536;
        type Batch = (Vec<usize>, Vec<Vec<u32>>);
        let pb = ctx.ui.bar_with_unit(n_texts as u64, "embed", "psg");
        let hidden = config.hidden_size;
        // Rows stream to disk as they complete. Workers finish batches out
        // of order, so rows land in `pending` and are drained to the sink
        // whenever the next contiguous run is ready. The window stays small
        // because batches are planned within a tokenize chunk, so a row is
        // only ever reordered against its chunk-mates.
        let mut sink = match RowSink::open(&output, n_texts, hidden) {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let mut pending: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut next_row = 0usize;
        let mut done = 0u64;
        let mut first_err: Option<String> = None;
        let abort = std::sync::atomic::AtomicBool::new(false);
        // Bounded so the tokenizer cannot race arbitrarily far ahead of the
        // GPUs: unbounded, it would hold token ids for the whole input and
        // widen the reorder window it is supposed to keep narrow.
        let (btx, brx) = std::sync::mpsc::sync_channel::<Result<Batch, String>>(
            (models.len() * 4).max(4),
        );
        let brx = std::sync::Mutex::new(brx);
        let (rtx, rrx) =
            std::sync::mpsc::channel::<Result<(Vec<usize>, Vec<Vec<f32>>), String>>();
        std::thread::scope(|scope| {
            let (tokenizer, abort, brx) = (&tokenizer, &abort, &brx);
            let (src, col) = (source.clone(), column.clone());
            scope.spawn(move || {
                // Text is pulled a chunk at a time and dropped once
                // tokenized, so resident text is TOKENIZE_CHUNK rows
                // (~62 MB here) regardless of how many rows the window
                // covers. The window is a work-partitioning choice now,
                // not a memory one.
                let mut reader = match veks_core::formats::passage_table::TextColumnReader::open(
                    &src, &col, row_start, row_end,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = btx.send(Err(e));
                        return;
                    }
                };
                let mut base = 0usize;
                loop {
                    if abort.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    let chunk = match reader.next_chunk(TOKENIZE_CHUNK) {
                        Ok(c) if c.is_empty() => return,
                        Ok(c) => c,
                        Err(e) => {
                            let _ = btx.send(Err(e));
                            return;
                        }
                    };
                    let inputs: Vec<&str> = chunk.iter().map(String::as_str).collect();
                    let rows: Vec<Vec<u32>> = match tokenizer.encode_batch(inputs, false) {
                        Ok(encs) => encs
                            .into_iter()
                            .map(|e| prepare_ids(e.get_ids().to_vec(), eos, max_length))
                            .collect(),
                        Err(e) => {
                            let _ = btx.send(Err(format!("tokenize: {}", e)));
                            return;
                        }
                    };
                    for batch in batch_plan(&rows, batch_size) {
                        let abs: Vec<usize> = batch.iter().map(|&i| base + i).collect();
                        let batch_rows: Vec<Vec<u32>> =
                            batch.iter().map(|&i| rows[i].clone()).collect();
                        if btx.send(Ok((abs, batch_rows))).is_err() {
                            return;
                        }
                    }
                    base += chunk.len();
                }
            });
            for model in models.iter() {
                let rtx = rtx.clone();
                scope.spawn(move || loop {
                    let msg = brx.lock().unwrap().recv();
                    let (idx, batch_rows) = match msg {
                        Ok(Ok(b)) => b,
                        Ok(Err(e)) => {
                            let _ = rtx.send(Err(e));
                            return;
                        }
                        Err(_) => return, // producer done, channel drained
                    };
                    let res = model
                        .embed_batch(&batch_rows)
                        .map(|v| (idx, v))
                        .map_err(|e| format!("embed failed: {}", e));
                    let failed = res.is_err();
                    if rtx.send(res).is_err() || failed {
                        return;
                    }
                });
            }
            drop(rtx);
            while let Ok(msg) = rrx.recv() {
                match msg {
                    Ok((batch, embedded)) => {
                        done += batch.len() as u64;
                        for (idx, vec) in batch.into_iter().zip(embedded) {
                            pending.insert(idx, vec);
                        }
                        // Drain every row that is now in order.
                        while let Some(row) = pending.remove(&next_row) {
                            if let Err(e) = sink.write_row(&row) {
                                first_err = Some(e);
                                abort.store(true, std::sync::atomic::Ordering::Relaxed);
                                break;
                            }
                            next_row += 1;
                        }
                        if first_err.is_some() {
                            break;
                        }
                        pb.set_position(done);
                    }
                    Err(e) => {
                        first_err = Some(e);
                        abort.store(true, std::sync::atomic::Ordering::Relaxed);
                        break; // drops rrx; senders fail and threads exit
                    }
                }
            }
        });
        pb.finish();
        if let Some(e) = first_err {
            return error_result(e, start);
        }
        // Every row must have been written: the npy header states the row
        // count up front, so a gap would produce a file that reads as
        // complete and is not.
        if next_row != n_texts || !pending.is_empty() {
            return error_result(
                format!(
                    "internal: wrote {} of {} row(s), {} still buffered",
                    next_row,
                    n_texts,
                    pending.len()
                ),
                start,
            );
        }
        if let Err(e) = sink.finish() {
            return error_result(e, start);
        }
        let _ = crate::pipeline::variables::set_and_save(
            &ctx.workspace,
            "embed_dim",
            &hidden.to_string(),
        );
        ctx.defaults.insert("embed_dim".to_string(), hidden.to_string());

        let elapsed = start.elapsed();
        CommandResult {
            status: Status::Ok,
            message: format!(
                "embedded {} row(s) @ {}-d to {} ({:.1} rows/s)",
                n_texts,
                hidden,
                output.display(),
                n_texts as f64 / elapsed.as_secs_f64().max(0.001)
            ),
            produced: vec![output],
            elapsed,
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Parquet file whose text column is embedded row-by-row".to_string(),
                extended_description: None,
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Output .npy path (C-order f32, row i embeds source row i)".to_string(),
                extended_description: None,
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "column".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("text".to_string()),
                description: "Source column holding the text".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "model".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some(DEFAULT_MODEL.to_string()),
                description: "HuggingFace model id (Qwen3-Embedding family)".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "revision".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("main".to_string()),
                description: "Model revision (pin a commit for strict reproducibility)".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "range".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: None,
                description: "Half-open row window of the source to embed, e.g. \"[0,50M)\" \
                              (default: all rows). Splits an embed too large for memory into \
                              passes; row i of the output embeds source row range-start + i"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "batch-size".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("128".to_string()),
                description: "Sequences per forward pass (the GPU-tuned default; wall time is \
                              flat 128-512 on the fused path)"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "max-length".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("1024".to_string()),
                description: "Token cap per row (truncated before the EOS pooling token)".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "device".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("auto".to_string()),
                description: "auto, cpu, or cuda[:N]; a comma-separated list (cuda:0,cuda:1) \
                              shards batches across one worker per entry (cuda needs the \
                              embed-cuda build feature)"
                    .to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "dtype".to_string(),
                type_name: "String".to_string(),
                required: false,
                default: Some("auto".to_string()),
                description: "auto (f32 on cpu, bf16 on cuda), f32, bf16, or f16".to_string(),
                extended_description: None,
                role: OptionRole::Config,
            },
        ]
    }

    fn value_completions(&self) -> HashMap<String, ValueCompletions> {
        let mut map = HashMap::new();
        map.insert(
            "device".to_string(),
            ValueCompletions::enum_values(&["auto", "cpu", "cuda", "cuda:0", "cuda:0,cuda:1"]),
        );
        map.insert(
            "dtype".to_string(),
            ValueCompletions::enum_values(&["auto", "f32", "bf16", "f16"]),
        );
        map
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id,
            self.command_path(),
            options,
            &["source"],
            &["output"],
        )
    }
}

struct ModelFiles {
    config: PathBuf,
    tokenizer: PathBuf,
    weights: Vec<PathBuf>,
}

/// Fetch config/tokenizer/weights through the shared HF cache (no network
/// when already cached). Single-file weights cover the Qwen3-Embedding
/// sizes this command targets.
fn fetch_model_files(model_id: &str, revision: &str) -> Result<ModelFiles, String> {
    // ApiBuilder::from_env, not Api::new: only the former honors HF_HOME
    // (Api::new hardcodes ~/.cache/huggingface/hub), and the shared-cache
    // contract documented for this command depends on it.
    let api = ApiBuilder::from_env()
        .build()
        .map_err(|e| format!("hf api: {}", e))?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    let get = |name: &str| {
        repo.get(name)
            .map_err(|e| format!("fetch {}/{}: {}", model_id, name, e))
    };
    Ok(ModelFiles {
        config: get("config.json")?,
        tokenizer: get("tokenizer.json")?,
        weights: vec![get("model.safetensors")?],
    })
}

/// Resolve a device spec into one device per embed worker. `auto` claims
/// **every** visible CUDA device (one worker each — this command is a
/// throughput stage, and a single-GPU default silently halves a multi-GPU
/// host), else CPU. A comma-separated list (`cuda:0,cuda:1`) shards batches
/// round-robin across workers; repeating a device (`cuda:0,cuda:0`) is
/// allowed and runs two overlapping workers on one GPU.
fn resolve_devices(spec: &str) -> Result<Vec<Device>, String> {
    if spec == "auto" {
        let n = cuda_device_count();
        if n > 0 {
            return (0..n)
                .map(|i| Device::new_cuda(i).map_err(|e| format!("cuda:{} init: {}", i, e)))
                .collect();
        }
        return Ok(vec![Device::Cpu]);
    }
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(resolve_device)
        .collect::<Result<Vec<_>, _>>()
        .and_then(|v| {
            if v.is_empty() {
                Err(format!("no devices in spec '{}'", spec))
            } else {
                Ok(v)
            }
        })
}

/// Number of visible CUDA devices (0 when CUDA is unavailable or the
/// binary was built without it).
fn cuda_device_count() -> usize {
    #[cfg(feature = "embed-cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            return candle_core::cuda_backend::cudarc::driver::CudaContext::device_count()
                .map(|n| n.max(0) as usize)
                .unwrap_or(0);
        }
    }
    0
}

fn resolve_device(spec: &str) -> Result<Device, String> {
    match spec {
        "cpu" => Ok(Device::Cpu),
        "auto" => {
            if candle_core::utils::cuda_is_available() {
                Device::new_cuda(0).map_err(|e| format!("cuda init: {}", e))
            } else {
                Ok(Device::Cpu)
            }
        }
        s if s == "cuda" || s.starts_with("cuda:") => {
            let ordinal: usize = s
                .strip_prefix("cuda:")
                .map(|n| n.parse().map_err(|_| format!("invalid device '{}'", s)))
                .unwrap_or(Ok(0))?;
            Device::new_cuda(ordinal).map_err(|e| {
                format!(
                    "cuda device '{}' unavailable: {} (build with the embed-cuda feature on a GPU host)",
                    s, e
                )
            })
        }
        other => Err(format!("unknown device '{}': expected auto, cpu, or cuda[:N]", other)),
    }
}

fn resolve_dtype(spec: &str, device: &Device) -> Result<DType, String> {
    match spec {
        "auto" => Ok(if device.is_cuda() { DType::BF16 } else { DType::F32 }),
        "f32" => Ok(DType::F32),
        "bf16" => Ok(DType::BF16),
        "f16" => Ok(DType::F16),
        other => Err(format!("unknown dtype '{}': expected auto, f32, bf16, or f16", other)),
    }
}

fn device_name(device: &Device) -> &'static str {
    if device.is_cuda() { "cuda" } else { "cpu" }
}

/// Cap token ids to `max_length` including a terminal EOS, appending EOS
/// when absent — the last-token pooling position must always be EOS.
fn prepare_ids(mut ids: Vec<u32>, eos: u32, max_length: usize) -> Vec<u32> {
    if ids.len() > max_length - 1 {
        ids.truncate(max_length - 1);
    }
    if ids.last() != Some(&eos) {
        ids.push(eos);
    }
    ids
}

/// Group row indices into batches of near-equal token length (sorted
/// descending) so padding waste stays low; callers scatter results back by
/// index, so output order is unaffected.
fn batch_plan(rows: &[Vec<u32>], batch_size: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(rows[i].len()));
    order.chunks(batch_size).map(|c| c.to_vec()).collect()
}

/// Streaming row sink for the embedding output.
///
/// Rows are written as they become available rather than collected: at
/// 100M x 1024-d an in-memory buffer is ~410 GB, which no amount of host
/// RAM makes reasonable. Both variants are append-only and take rows in
/// ascending order, which is what lets the caller hold only the rows that
/// arrived out of order.
///
/// Writing a native xvec format directly is not merely a convenience.
/// `prepare bootstrap` emits a conversion step only when its base vectors
/// are *not* already native (`needs_import = !is_native_xvec_file(..)`);
/// handing it `.fvecs` collapses that step to an identity symlink,
/// removing both the conversion pass and a full second copy of the
/// vectors — 410 GB at 100M scale.
enum RowSink {
    /// `.npy` — the header carries the row count, so it is written up
    /// front from the known total and rows are appended after it.
    Npy(AtomicWriter),
    /// Any xvec format (`.fvecs` and friends): a dimension prefix per
    /// record, appended in order.
    Xvec(Box<dyn veks_core::formats::writer::VecSink>),
}

impl RowSink {
    /// Open a sink for `output`, choosing the format from its extension.
    /// `rows` is the exact number of rows that will be written — the npy
    /// header depends on it, so a short or long write corrupts the file.
    fn open(output: &Path, rows: usize, dim: usize) -> Result<Self, String> {
        let format = veks_core::formats::VecFormat::detect_from_path(output);
        match format {
            Some(f) if f.is_xvec() => {
                let sink = veks_core::formats::writer::open_sink(
                    output,
                    f,
                    &veks_core::formats::writer::SinkConfig {
                        dimension: dim as u32,
                        source_format: f,
                        slab_page_size: None,
                        slab_namespace: 0,
                        schema_sidecar: None,
                    },
                )?;
                Ok(RowSink::Xvec(sink))
            }
            // npy is the default for anything not recognized as xvec:
            // the historical output format, and what a bare `.npy` or an
            // extensionless path means here.
            _ => {
                let header_body = format!(
                    "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
                    rows, dim
                );
                let unpadded = 10 + header_body.len() + 1;
                let padding = (64 - unpadded % 64) % 64;
                let header = format!("{}{}\n", header_body, " ".repeat(padding));

                let mut writer = AtomicWriter::new(output)
                    .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;
                writer.write_all(b"\x93NUMPY\x01\x00").map_err(|e| e.to_string())?;
                writer
                    .write_all(&(header.len() as u16).to_le_bytes())
                    .map_err(|e| e.to_string())?;
                writer.write_all(header.as_bytes()).map_err(|e| e.to_string())?;
                Ok(RowSink::Npy(writer))
            }
        }
    }

    fn write_row(&mut self, row: &[f32]) -> Result<(), String> {
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        match self {
            RowSink::Npy(w) => w.write_all(&bytes).map_err(|e| e.to_string()),
            // The xvec sink appends and ignores the ordinal; ordering is
            // the caller's contract.
            RowSink::Xvec(s) => {
                s.write_record(0, &bytes);
                Ok(())
            }
        }
    }

    fn finish(self) -> Result<(), String> {
        match self {
            RowSink::Npy(w) => w.finish().map_err(|e| e.to_string()),
            RowSink::Xvec(s) => s.finish(),
        }
    }
}

/// Write a C-order f32 `.npy` of shape [rows, dim] atomically.
#[cfg(test)]
fn write_npy_f32(output: &Path, vectors: &[Vec<f32>], dim: usize) -> Result<(), String> {
    let header_body = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        vectors.len(),
        dim
    );
    let unpadded = 10 + header_body.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    let header = format!("{}{}\n", header_body, " ".repeat(padding));

    let mut writer = AtomicWriter::new(output)
        .map_err(|e| format!("failed to create {}: {}", output.display(), e))?;
    writer.write_all(b"\x93NUMPY\x01\x00").map_err(|e| e.to_string())?;
    writer
        .write_all(&(header.len() as u16).to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer.write_all(header.as_bytes()).map_err(|e| e.to_string())?;
    for row in vectors {
        if row.len() != dim {
            return Err(format!("row width {} != dim {}", row.len(), dim));
        }
        for v in row {
            writer.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    writer.finish().map_err(|e| e.to_string())
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_ids_caps_and_terminates_with_eos() {
        assert_eq!(prepare_ids(vec![1, 2, 3], 9, 8), vec![1, 2, 3, 9]);
        assert_eq!(prepare_ids(vec![1, 2, 9], 9, 8), vec![1, 2, 9]);
        // Cap 4 → 3 content tokens + EOS.
        assert_eq!(prepare_ids((1..=10).collect(), 9, 4), vec![1, 2, 3, 9]);
        assert_eq!(prepare_ids(vec![], 9, 4), vec![9]);
    }

    #[test]
    /// The sink picks its format from the output extension, and both
    /// formats write rows in the order handed to them. Writing `.fvecs`
    /// directly is what lets `prepare bootstrap` treat the artifact as
    /// native and skip its conversion step entirely.
    #[test]
    fn row_sink_writes_both_formats_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let rows: Vec<Vec<f32>> = (0..5)
            .map(|i| (0..4).map(|j| (i * 4 + j) as f32).collect())
            .collect();

        for (name, expected_len) in [
            ("out.fvecs", 5 * (4 + 4 * 4)), // dim prefix + payload per record
            ("out.npy", 128 + 5 * 4 * 4),   // padded header + payload
        ] {
            let path = tmp.path().join(name);
            let mut sink = RowSink::open(&path, rows.len(), 4).unwrap();
            for row in &rows {
                sink.write_row(row).unwrap();
            }
            sink.finish().unwrap();
            let written = std::fs::metadata(&path).unwrap().len();
            assert_eq!(written, expected_len as u64, "{name} size");

            // Read back through the format readers and confirm both the
            // values and their order survived.
            let format = veks_core::formats::VecFormat::detect(&path).unwrap();
            let mut src =
                veks_core::formats::reader::open_source(&path, format, 1, None).unwrap();
            assert_eq!(src.dimension(), 4, "{name} dimension");
            for (i, expected) in rows.iter().enumerate() {
                let bytes = src.next_record().unwrap_or_else(|| panic!("{name} row {i}"));
                let got: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                assert_eq!(&got, expected, "{name} row {i} out of order or corrupt");
            }
            assert!(src.next_record().is_none(), "{name} wrote extra rows");
        }
    }

    #[test]
    fn resolve_devices_lists_and_rejects() {
        // A list yields one device (one worker) per entry, repeats included.
        assert_eq!(resolve_devices("cpu").unwrap().len(), 1);
        assert_eq!(resolve_devices("cpu,cpu,cpu").unwrap().len(), 3);
        assert_eq!(resolve_devices(" cpu , cpu ").unwrap().len(), 2);
        // Empty and unknown specs are errors, not silent CPU fallbacks —
        // a typo'd device must never quietly cost 1000x throughput.
        assert!(resolve_devices("").is_err());
        assert!(resolve_devices("gpu").is_err());
        assert!(resolve_devices("cuda:x").is_err());
        // `auto` always resolves to at least one worker (every visible GPU
        // on a CUDA host, else CPU).
        assert!(!resolve_devices("auto").unwrap().is_empty());
    }

    #[test]
    fn batch_plan_covers_all_rows_once_longest_first() {
        let rows: Vec<Vec<u32>> = vec![vec![0; 3], vec![0; 10], vec![0; 5], vec![0; 7]];
        let plan = batch_plan(&rows, 2);
        assert_eq!(plan, vec![vec![1, 3], vec![2, 0]]);
        let mut seen: Vec<usize> = plan.into_iter().flatten().collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3]);
    }

    #[test]
    fn npy_writer_round_trips_through_veks_core_probe() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("v.npy");
        let vectors: Vec<Vec<f32>> = (0..7).map(|i| vec![i as f32; 4]).collect();
        write_npy_f32(&path, &vectors, 4).unwrap();
        let meta = veks_core::formats::reader::probe_source(
            &path,
            veks_core::formats::VecFormat::Npy,
        )
        .unwrap();
        assert_eq!(meta.record_count, Some(7));
        assert_eq!(meta.dimension, 4);
    }

    /// Full-model smoke test — downloads Qwen3-Embedding-0.6B, so ignored
    /// by default; run with `--ignored` on a networked host to verify the
    /// backend end-to-end (norms ≈ 1, batching invariant to batch-size).
    #[test]
    #[ignore = "downloads the embedding model (~1.2GB)"]
    fn real_model_embeds_unit_vectors() {
        let files = fetch_model_files(DEFAULT_MODEL, "main").unwrap();
        let config: qwen3::Config =
            serde_json::from_str(&std::fs::read_to_string(&files.config).unwrap()).unwrap();
        let tokenizer = tokenizers::Tokenizer::from_file(&files.tokenizer).unwrap();
        let eos = config
            .eos_token_id
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&files.weights, DType::F32, &Device::Cpu)
        }
        .unwrap();
        let model = qwen3::EmbeddingModel::new(&config, vb).unwrap();

        let texts = ["gravity bends light", "chunking scientific text into passages"];
        let rows: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| prepare_ids(tokenizer.encode(*t, false).unwrap().get_ids().to_vec(), eos, 64))
            .collect();
        let batched = model.embed_batch(&rows).unwrap();
        let single: Vec<Vec<f32>> = rows
            .iter()
            .map(|r| model.embed_batch(std::slice::from_ref(r)).unwrap().remove(0))
            .collect();
        for (b, s) in batched.iter().zip(&single) {
            let norm: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "norm {}", norm);
            let dot: f32 = b.iter().zip(s).map(|(x, y)| x * y).sum();
            assert!(dot > 0.999, "batched vs single cosine {}", dot);
        }
    }
}
