// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: synchronization barrier between profile groups.
//!
//! A no-op command whose sole purpose is to serve as a dependency target.
//! When the pipeline inserts a barrier step between profile groups, all
//! steps in the next profile depend on the barrier, and the barrier depends
//! on all steps in the previous profile. This guarantees sequential profile
//! execution.

use std::time::Instant;

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, Options, Status,
    StreamContext,
};

pub struct BarrierOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(BarrierOp)
}

impl CommandOp for BarrierOp {
    fn command_path(&self) -> &str {
        "barrier"
    }

    fn command_doc(&self) -> CommandDoc {
        CommandDoc {
            summary: "Synchronization barrier between profile groups".into(),
            body: r#"# barrier

No-op synchronization point between profile groups.

## Description

A barrier is a no-op command whose sole purpose is to serve as a
dependency target in the pipeline execution graph. It produces no
output and performs no computation.

## How It Works

When the pipeline engine inserts a barrier step between two profile
groups, it wires the dependency graph so that all steps in the next
profile group depend on the barrier, and the barrier depends on all
steps in the previous profile group. This creates a synchronization
fence: no step in the next group can begin until every step in the
previous group has completed. The barrier itself executes instantly,
logging a single status message.

## Data Preparation Role

Barriers enforce sequential execution of profile groups within a
pipeline. Profiles represent different dataset configurations (e.g.
a full-scale profile and a 1M-record subset profile) that share
common base data. The barrier guarantees that shared data preparation
(downloading, importing, vector assembly) completes before
profile-specific steps (index building, query set generation) begin.
Without barriers, the pipeline scheduler could overlap steps from
different profile groups, leading to race conditions on shared files."#
                .into(),
        }
    }

    fn execute(&mut self, _options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();
        ctx.ui.log("Barrier: profile group synchronization point");
        CommandResult {
            status: Status::Ok,
            message: "barrier passed".into(),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![]
    }

    fn project_artifacts(&self, step_id: &str, _options: &Options) -> ArtifactManifest {
        ArtifactManifest {
            step_id: step_id.to_string(),
            command: self.command_path().to_string(),
            inputs: vec![],
            outputs: vec![],
            intermediates: vec![],
        }
    }
}
