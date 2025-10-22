#!/bin/bash
# Helper script to run MoCaX tests with proper environment setup

# Set library path for MoCaX shared library
export LD_LIBRARY_PATH="$(pwd)/mocax_lib:$LD_LIBRARY_PATH"

# Run the test
uv run python mocax_test.py
