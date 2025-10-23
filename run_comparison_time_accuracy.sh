#!/bin/bash
# Convenience script to run comprehensive method comparison
# Sets up LD_LIBRARY_PATH for MoCaX if available

export LD_LIBRARY_PATH="$PWD/mocax_lib:$LD_LIBRARY_PATH"
uv run python compare_methods_time_accuracy.py
