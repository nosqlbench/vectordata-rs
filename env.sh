#!/bin/bash

# Strip conda lib dirs from LD_LIBRARY_PATH — conda's libgcc_s.so is
# too old for Rust-compiled binaries and poisons build scripts.
if [ -n "$CONDA_PREFIX" ]; then
    LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "$CONDA_PREFIX" | paste -sd ':')
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH%:}"
fi

export PATH=`pwd`/target/release:$PATH

# HDF5 build and runtime support (via conda).
# Isolate just libhdf5 into a clean directory — no libgcc_s contamination.
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/lib/libhdf5.so" ]; then
    export HDF5_DIR="$CONDA_PREFIX"
    _hdf5_runlib="$HOME/.local/lib/hdf5"
    mkdir -p "$_hdf5_runlib"
    ln -sf "$CONDA_PREFIX"/lib/libhdf5.so* "$_hdf5_runlib/" 2>/dev/null
    export LD_LIBRARY_PATH="$_hdf5_runlib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    unset _hdf5_runlib
fi

source <(veks completions)
source <(COMPLETE=bash slab)
