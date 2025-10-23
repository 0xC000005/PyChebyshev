#!/bin/bash
# Convenience script to run MoCaX Extend (Tensor Train) test
# Sets up LD_LIBRARY_PATH for both MoCaX and MoCaXExtend shared libraries

export LD_LIBRARY_PATH="$PWD/mocax_lib:$PWD/mocaxextend_lib/shared_libs:$LD_LIBRARY_PATH"
uv run python mocax_tt.py
