#!/usr/bin/env bash
# This script sets LD_PRELOAD only for the mainboard process.

# Path to your GPU intercept library:
export LD_PRELOAD="/apollo/cyber/examples/bachelor_thesis/libgpu_intercept.so"

# Now launch mainboard with any/all arguments passed to this script:
exec mainboard "$@"
