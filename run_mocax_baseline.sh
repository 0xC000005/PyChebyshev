#!/bin/bash
# Wrapper script to run mocax_baseline.py with correct LD_LIBRARY_PATH

cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/mocax_lib:$LD_LIBRARY_PATH"
uv run python mocax_baseline.py "$@"
