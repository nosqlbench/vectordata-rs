// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `s3://` push transport via the AWS CLI.
//!
//! This matches `veks publish`: rather than pull in the AWS SDK (which
//! the read side also deliberately avoids), we shell out to `aws`, so
//! credential resolution is identical to what users already have
//! working (env vars, `~/.aws/credentials`, `--profile`, SSO, IAM
//! roles). The choice is isolated behind [`super::PushTransport`], so a
//! native SDK can replace it later without touching callers.

use std::path::Path;
use std::process::Command;

use super::{PushError, PushTransport, RemoteObject, TransportOptions};

pub struct S3Transport {
    bucket: String,
    /// Key prefix (no leading slash, trailing slash preserved).
    prefix: String,
    profile: Option<String>,
    endpoint_url: Option<String>,
}

impl S3Transport {
    /// Parse `s3://bucket/prefix/` into bucket + prefix.
    pub fn from_url(url: &str, opts: &TransportOptions) -> Result<Self, String> {
        let rest = url.strip_prefix("s3://").ok_or_else(|| format!("not an s3 url: {url}"))?;
        let (bucket, prefix) = match rest.split_once('/') {
            Some((b, p)) => (b.to_string(), p.to_string()),
            None => (rest.to_string(), String::new()),
        };
        if bucket.is_empty() {
            return Err(format!("s3 url has no bucket: {url}"));
        }
        Ok(S3Transport {
            bucket,
            prefix,
            profile: opts.profile.clone(),
            endpoint_url: opts.endpoint_url.clone(),
        })
    }

    fn key(&self, rel: &str) -> String {
        format!("{}{}", self.prefix, rel.trim_start_matches('/'))
    }

    fn s3_uri(&self, rel: &str) -> String {
        format!("s3://{}/{}", self.bucket, self.key(rel))
    }

    /// Base command with global options applied.
    fn aws(&self) -> Command {
        let mut c = Command::new("aws");
        if let Some(p) = &self.profile {
            c.arg("--profile").arg(p);
        }
        if let Some(e) = &self.endpoint_url {
            c.arg("--endpoint-url").arg(e);
        }
        c
    }
}

/// Classify an `aws` failure from its stderr.
fn classify(stderr: &str, context: &str) -> PushError {
    let lc = stderr.to_lowercase();
    if lc.contains("preconditionfailed") || lc.contains("precondition failed") {
        PushError::PreconditionFailed
    } else if lc.contains("accessdenied")
        || lc.contains("forbidden")
        || lc.contains("invalidaccesskeyid")
        || lc.contains("signaturedoesnotmatch")
        || lc.contains("expiredtoken")
        || lc.contains("unable to locate credentials")
    {
        PushError::Auth(format!(
            "{context}: {}\nset AWS credentials (AWS_PROFILE / --profile, env vars, or an IAM role)",
            stderr.trim()
        ))
    } else {
        PushError::Other(format!("{context}: {}", stderr.trim()))
    }
}

/// True when the error is a plain "object/bucket not found".
fn is_not_found(stderr: &str) -> bool {
    let lc = stderr.to_lowercase();
    lc.contains("not found") || lc.contains("nosuchkey") || lc.contains("404")
}

fn spawn_err(e: std::io::Error) -> PushError {
    if e.kind() == std::io::ErrorKind::NotFound {
        PushError::Other(
            "the `aws` CLI was not found on PATH; install it to push to s3:// endpoints".to_string(),
        )
    } else {
        PushError::Other(format!("failed to run aws: {e}"))
    }
}

impl PushTransport for S3Transport {
    fn head(&self, rel: &str) -> Result<Option<RemoteObject>, PushError> {
        let out = self
            .aws()
            .args(["s3api", "head-object", "--bucket", &self.bucket, "--key"])
            .arg(self.key(rel))
            .args(["--output", "json"])
            .output()
            .map_err(spawn_err)?;
        if out.status.success() {
            let v: serde_json::Value =
                serde_json::from_slice(&out.stdout).map_err(|e| PushError::Other(e.to_string()))?;
            let size = v.get("ContentLength").and_then(|x| x.as_u64()).unwrap_or(0);
            let etag = v
                .get("ETag")
                .and_then(|x| x.as_str())
                .map(|s| s.trim_matches('"').to_string());
            Ok(Some(RemoteObject { size, etag }))
        } else {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if is_not_found(&stderr) {
                Ok(None)
            } else {
                Err(classify(&stderr, &format!("head-object {}", self.s3_uri(rel))))
            }
        }
    }

    fn get(&self, rel: &str) -> Result<Option<Vec<u8>>, PushError> {
        let out = self
            .aws()
            .args(["s3", "cp"])
            .arg(self.s3_uri(rel))
            .arg("-")
            .output()
            .map_err(spawn_err)?;
        if out.status.success() {
            Ok(Some(out.stdout))
        } else {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if is_not_found(&stderr) {
                Ok(None)
            } else {
                Err(classify(&stderr, &format!("get {}", self.s3_uri(rel))))
            }
        }
    }

    fn put_file(&self, rel: &str, src: &Path) -> Result<(), PushError> {
        let out = self
            .aws()
            .args(["s3", "cp"])
            .arg(src)
            .arg(self.s3_uri(rel))
            .output()
            .map_err(spawn_err)?;
        if out.status.success() {
            Ok(())
        } else {
            Err(classify(
                &String::from_utf8_lossy(&out.stderr),
                &format!("put {}", self.s3_uri(rel)),
            ))
        }
    }

    fn put_bytes(&self, rel: &str, data: &[u8], if_match: Option<&str>) -> Result<(), PushError> {
        // put-object needs a --body file; stage one in a temp path keyed
        // to the object so concurrent puts don't collide.
        let tmp = std::env::temp_dir().join(format!(
            "vd-push-{}-{}.tmp",
            std::process::id(),
            crate::push::checksums::sha256_bytes(self.key(rel).as_bytes())
        ));
        std::fs::write(&tmp, data).map_err(|e| PushError::Other(e.to_string()))?;

        let mut cmd = self.aws();
        cmd.args(["s3api", "put-object", "--bucket", &self.bucket, "--key"])
            .arg(self.key(rel))
            .arg("--body")
            .arg(&tmp);
        match if_match {
            Some("") => {
                cmd.args(["--if-none-match", "*"]);
            }
            Some(etag) => {
                cmd.args(["--if-match", etag]);
            }
            None => {}
        }
        let out = cmd.output().map_err(spawn_err);
        let _ = std::fs::remove_file(&tmp);
        let out = out?;
        if out.status.success() {
            Ok(())
        } else {
            Err(classify(
                &String::from_utf8_lossy(&out.stderr),
                &format!("put {}", self.s3_uri(rel)),
            ))
        }
    }

    fn preflight(&self) -> Result<(), PushError> {
        let out = self
            .aws()
            .args(["s3api", "head-bucket", "--bucket", &self.bucket])
            .output()
            .map_err(spawn_err)?;
        if out.status.success() {
            Ok(())
        } else {
            Err(classify(
                &String::from_utf8_lossy(&out.stderr),
                &format!("head-bucket {}", self.bucket),
            ))
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, PushError> {
        let full_prefix = self.key(prefix);
        let out = self
            .aws()
            .args(["s3api", "list-objects-v2", "--bucket", &self.bucket, "--prefix"])
            .arg(&full_prefix)
            .args(["--output", "json"])
            .output()
            .map_err(spawn_err)?;
        if !out.status.success() {
            return Err(classify(
                &String::from_utf8_lossy(&out.stderr),
                &format!("list {}", self.s3_uri(prefix)),
            ));
        }
        if out.stdout.is_empty() {
            return Ok(Vec::new()); // empty bucket/prefix → no Contents
        }
        let v: serde_json::Value =
            serde_json::from_slice(&out.stdout).map_err(|e| PushError::Other(e.to_string()))?;
        let mut keys = Vec::new();
        if let Some(contents) = v.get("Contents").and_then(|c| c.as_array()) {
            for obj in contents {
                if let Some(key) = obj.get("Key").and_then(|k| k.as_str()) {
                    // Strip the publish-root prefix to yield a relative key.
                    let rel = key.strip_prefix(self.prefix.as_str()).unwrap_or(key);
                    if !rel.is_empty() {
                        keys.push(rel.to_string());
                    }
                }
            }
        }
        keys.sort();
        Ok(keys)
    }

    fn delete(&self, rel: &str) -> Result<(), PushError> {
        let out = self
            .aws()
            .args(["s3api", "delete-object", "--bucket", &self.bucket, "--key"])
            .arg(self.key(rel))
            .output()
            .map_err(spawn_err)?;
        if out.status.success() {
            Ok(())
        } else {
            Err(classify(
                &String::from_utf8_lossy(&out.stderr),
                &format!("delete {}", self.s3_uri(rel)),
            ))
        }
    }

    fn describe(&self) -> String {
        format!("s3://{}/{}", self.bucket, self.prefix)
    }
}
