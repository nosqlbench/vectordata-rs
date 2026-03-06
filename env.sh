#!/bin/bash
export PATH=$PATH:`pwd`/target/release
source <(veks completions --shell bash)
source <(COMPLETE=bash slab)
