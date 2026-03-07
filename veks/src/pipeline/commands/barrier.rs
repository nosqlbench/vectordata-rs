// Copyright (c) DataStax, Inc.
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
    CommandDoc, CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
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
            body: "# barrier\n\nNo-op synchronization point. All steps in the \
                   next profile group depend on this barrier, ensuring sequential \
                   profile execution."
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
}
