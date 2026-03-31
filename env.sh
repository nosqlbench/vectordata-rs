#!/bin/bash
export PATH=$PATH:`pwd`/target/release
source <(veks completions)
source <(COMPLETE=bash slab)
