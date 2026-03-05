#!/bin/bash
export PATH=$PATH:`pwd`/target/release
source <(COMPLETE=bash veks)
source <(COMPLETE=bash slab)
