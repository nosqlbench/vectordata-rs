// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The `s3://` backend — S3 and S3-compatible stores (MinIO, R2, B2,
//! Wasabi, Ceph RGW, GCS's S3 API) reached through the `aws` CLI. This
//! deliberately mirrors `vectordata push`'s s3 transport so the server and
//! the producer toolkit stay in lockstep on one S3 mechanism rather than
//! introducing a second (the heavy async SDK).
//!
//! Backends are *plain byte stores* — no conditional-write support is
//! required here; `vecd`'s DB is the CAS authority.

use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::backend::Backend;
use crate::db::BackendRow;
use crate::model::VecdError;

pub struct S3Backend {
    bucket: String,
    prefix: String,
    endpoint_url: Option<String>,
    region: Option<String>,
    profile: Option<String>,
}

impl S3Backend {
    pub fn new(row: &BackendRow) -> Result<Self, VecdError> {
        let rest = row
            .endpoint
            .strip_prefix("s3://")
            .ok_or_else(|| VecdError::usage(format!("s3 endpoint must be s3://bucket/prefix, got '{}'", row.endpoint)))?;
        let (bucket, prefix) = match rest.split_once('/') {
            Some((b, p)) => (b.to_string(), p.trim_end_matches('/').to_string()),
            None => (rest.to_string(), String::new()),
        };
        if bucket.is_empty() {
            return Err(VecdError::usage("s3 endpoint is missing a bucket".to_string()));
        }
        Ok(S3Backend {
            bucket,
            prefix,
            endpoint_url: row.endpoint_url.clone(),
            region: row.region.clone(),
            // creds_ref is interpreted as an AWS named profile when set.
            profile: row.creds_ref.clone(),
        })
    }

    /// The full `s3://bucket/prefix/key` URI for a relative key.
    fn uri(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            format!("s3://{}/{}", self.bucket, key)
        } else {
            format!("s3://{}/{}/{}", self.bucket, self.prefix, key)
        }
    }

    fn full_key(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}/{}", self.prefix, key)
        }
    }

    /// Begin an `aws` invocation with the shared connection flags applied.
    fn aws(&self) -> Command {
        let mut cmd = Command::new("aws");
        if let Some(url) = &self.endpoint_url {
            cmd.arg("--endpoint-url").arg(url);
        }
        if let Some(region) = &self.region {
            cmd.arg("--region").arg(region);
        }
        if let Some(profile) = &self.profile {
            cmd.arg("--profile").arg(profile);
        }
        cmd
    }

    fn run(&self, mut cmd: Command) -> Result<std::process::Output, VecdError> {
        cmd.output().map_err(|e| {
            VecdError::op(format!(
                "failed to invoke `aws` for s3 backend (is the AWS CLI installed?): {e}"
            ))
        })
    }

    /// Local temp path holding an in-progress upload's bytes. S3 has no
    /// native sparse/append `PutObject`, so chunks are assembled on local
    /// disk (sparse via `seek`) and uploaded in one `aws s3 cp` at finalize.
    fn staging_path(&self, staging_key: &str) -> PathBuf {
        std::env::temp_dir().join("vecd-uploads").join(staging_key)
    }
}

impl Backend for S3Backend {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, VecdError> {
        let mut cmd = self.aws();
        cmd.arg("s3").arg("cp").arg(self.uri(key)).arg("-");
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        let out = self.run(cmd)?;
        if out.status.success() {
            Ok(Some(out.stdout))
        } else {
            let err = String::from_utf8_lossy(&out.stderr);
            if err.contains("Not Found") || err.contains("404") || err.contains("does not exist") {
                Ok(None)
            } else {
                Err(VecdError::op(format!("s3 get {} failed: {}", self.uri(key), err.trim())))
            }
        }
    }

    fn put(&self, key: &str, data: &[u8]) -> Result<(), VecdError> {
        let mut cmd = self.aws();
        cmd.arg("s3").arg("cp").arg("-").arg(self.uri(key));
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = cmd
            .spawn()
            .map_err(|e| VecdError::op(format!("failed to invoke `aws`: {e}")))?;
        child.stdin.take().unwrap().write_all(data)?;
        let out = child.wait_with_output()?;
        if out.status.success() {
            Ok(())
        } else {
            Err(VecdError::op(format!(
                "s3 put {} failed: {}",
                self.uri(key),
                String::from_utf8_lossy(&out.stderr).trim()
            )))
        }
    }

    fn put_at(&self, staging_key: &str, offset: u64, chunk: &[u8]) -> Result<(), VecdError> {
        let path = self.staging_path(staging_key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = std::fs::OpenOptions::new().read(true).write(true).create(true).open(&path)?;
        f.seek(SeekFrom::Start(offset))?;
        f.write_all(chunk)?;
        Ok(())
    }

    fn finalize_staged(&self, staging_key: &str, final_key: &str) -> Result<(), VecdError> {
        let path = self.staging_path(staging_key);
        // Stream the assembled file straight off disk — `aws s3 cp <file>`
        // never buffers the whole object in this process.
        let mut cmd = self.aws();
        cmd.arg("s3").arg("cp").arg(&path).arg(self.uri(final_key));
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        let out = self.run(cmd)?;
        if !out.status.success() {
            return Err(VecdError::op(format!(
                "s3 finalize {} failed: {}",
                self.uri(final_key),
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    fn discard_staged(&self, staging_key: &str) -> Result<(), VecdError> {
        match std::fs::remove_file(self.staging_path(staging_key)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    fn head(&self, key: &str) -> Result<Option<u64>, VecdError> {
        let mut cmd = self.aws();
        cmd.arg("s3api")
            .arg("head-object")
            .arg("--bucket")
            .arg(&self.bucket)
            .arg("--key")
            .arg(self.full_key(key))
            .arg("--query")
            .arg("ContentLength")
            .arg("--output")
            .arg("text");
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        let out = self.run(cmd)?;
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            Ok(s.trim().parse::<u64>().ok())
        } else {
            Ok(None)
        }
    }

    fn delete(&self, key: &str) -> Result<(), VecdError> {
        let mut cmd = self.aws();
        cmd.arg("s3").arg("rm").arg(self.uri(key));
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        let out = self.run(cmd)?;
        // `aws s3 rm` of a missing key still exits 0; treat any failure as
        // operational unless it's a not-found.
        if out.status.success() {
            Ok(())
        } else {
            let err = String::from_utf8_lossy(&out.stderr);
            if err.contains("Not Found") || err.contains("404") {
                Ok(())
            } else {
                Err(VecdError::op(format!("s3 rm {} failed: {}", self.uri(key), err.trim())))
            }
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, VecdError> {
        let listing_prefix = self.full_key(prefix);
        let mut cmd = self.aws();
        cmd.arg("s3api")
            .arg("list-objects-v2")
            .arg("--bucket")
            .arg(&self.bucket)
            .arg("--prefix")
            .arg(&listing_prefix)
            .arg("--query")
            .arg("Contents[].Key")
            .arg("--output")
            .arg("text");
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        let out = self.run(cmd)?;
        if !out.status.success() {
            return Err(VecdError::op(format!(
                "s3 list failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let strip = if self.prefix.is_empty() {
            String::new()
        } else {
            format!("{}/", self.prefix)
        };
        let mut keys = Vec::new();
        for tok in text.split_whitespace() {
            if tok == "None" {
                continue;
            }
            let rel = tok.strip_prefix(&strip).unwrap_or(tok);
            keys.push(rel.to_string());
        }
        Ok(keys)
    }

    fn describe(&self) -> String {
        self.uri("").trim_end_matches('/').to_string()
    }
}
