#!/bin/bash
# Convenience script to run MoCaX Sliding test
# Sets up LD_LIBRARY_PATH for the MoCaX shared library

export LD_LIBRARY_PATH="$PWD/mocax_lib:$LD_LIBRARY_PATH"
uv run python mocax_sliding.py
