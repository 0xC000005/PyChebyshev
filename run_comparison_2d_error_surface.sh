#!/bin/bash

# Run 2D Error Surface Visualization: MoCaX Approximation
# Tests multiple Chebyshev node configurations (4, 6, 8, 12)
# Generates 4 figures with 2 subplots each (price + theta errors)

echo "=================================================="
echo "Running 2D Error Surface Comparison"
echo "=================================================="
echo ""

# Set LD_LIBRARY_PATH to include MoCaX library
export LD_LIBRARY_PATH="$PWD/mocax_lib:$LD_LIBRARY_PATH"

# Run the comparison script
uv run python compare_2d_error_surface.py

echo ""
echo "=================================================="
echo "Complete! Check the generated plots:"
echo ""
echo "Surface plots (2 subplots each):"
echo "  • mocax_2d_error_n4.png"
echo "  • mocax_2d_error_n6.png"
echo "  • mocax_2d_error_n8.png"
echo "  • mocax_2d_error_n12.png"
echo ""
echo "Convergence plot:"
echo "  • mocax_2d_convergence.png"
echo "=================================================="
