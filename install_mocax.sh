#!/bin/bash
# Installation script for MoCaX and MoCaX Extend libraries
# Extracts and sets up library directories from MoCaXSuite-1.2.0

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "MoCaX Installation Script"
echo "=============================================="
echo ""

# Check if MoCaXSuite-1.2.0 exists
MOCAX_SUITE_DIR="MoCaXSuite-1.2.0/MoCaXSuite-1.2.0"
if [ ! -d "$MOCAX_SUITE_DIR" ]; then
    echo -e "${RED}✗ Error: MoCaXSuite-1.2.0 directory not found!${NC}"
    echo "  Expected: $MOCAX_SUITE_DIR"
    echo "  Please extract MoCaXSuite-1.2.0.zip in the repository root."
    exit 1
fi
echo -e "${GREEN}✓ Found MoCaXSuite-1.2.0 directory${NC}"

# Paths
MOCAX_LIB_DIR="$MOCAX_SUITE_DIR/MoCaX/Linux/gmake/64bit/Python/MoCaXLibrary"
MOCAX_ZIP="$MOCAX_LIB_DIR/mocaxpy-4.3.1.linux-x86_64.zip"
MOCAX_SO="$MOCAX_LIB_DIR/libmocaxc.so"
MOCAXEXTEND_LIB_DIR="$MOCAX_SUITE_DIR/MoCaXExtend/Linux/Library"

# Check if required files exist
if [ ! -f "$MOCAX_ZIP" ]; then
    echo -e "${RED}✗ Error: MoCaX Python package not found!${NC}"
    echo "  Expected: $MOCAX_ZIP"
    exit 1
fi

if [ ! -d "$MOCAXEXTEND_LIB_DIR" ]; then
    echo -e "${RED}✗ Error: MoCaX Extend library not found!${NC}"
    echo "  Expected: $MOCAXEXTEND_LIB_DIR"
    exit 1
fi

echo ""
echo "=============================================="
echo "Setting up mocax_lib/"
echo "=============================================="

# Create/clean mocax_lib directory
if [ -d "mocax_lib" ]; then
    echo -e "${YELLOW}⚠  Removing existing mocax_lib directory...${NC}"
    rm -rf mocax_lib
fi
mkdir -p mocax_lib

# Create temporary directory for extraction
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Extract mocaxpy zip
echo "Extracting mocaxpy-4.3.1.linux-x86_64.zip..."
unzip -q "$MOCAX_ZIP" -d "$TEMP_DIR"

# Copy Python module
if [ -d "$TEMP_DIR/usr/lib/python2.7/site-packages/mocaxpy" ]; then
    echo "Copying mocaxpy Python module..."
    cp -r "$TEMP_DIR/usr/lib/python2.7/site-packages/mocaxpy" mocax_lib/
    echo -e "${GREEN}✓ mocaxpy module installed${NC}"
else
    echo -e "${RED}✗ Error: mocaxpy module not found in zip!${NC}"
    exit 1
fi

# Copy shared library from MoCaXLibrary directory
if [ -f "$MOCAX_SO" ]; then
    echo "Copying libmocaxc.so..."
    cp "$MOCAX_SO" mocax_lib/
    echo -e "${GREEN}✓ libmocaxc.so installed${NC}"
else
    echo -e "${RED}✗ Error: libmocaxc.so not found!${NC}"
    echo "  Expected: $MOCAX_SO"
    exit 1
fi

echo ""
echo "=============================================="
echo "Setting up mocaxextend_lib/"
echo "=============================================="

# Create/clean mocaxextend_lib directory
if [ -d "mocaxextend_lib" ]; then
    echo -e "${YELLOW}⚠  Removing existing mocaxextend_lib directory...${NC}"
    rm -rf mocaxextend_lib
fi
mkdir -p mocaxextend_lib

# Copy mocaxextendpy module
if [ -d "$MOCAXEXTEND_LIB_DIR/mocaxextendpy" ]; then
    echo "Copying mocaxextendpy Python module..."
    cp -r "$MOCAXEXTEND_LIB_DIR/mocaxextendpy" mocaxextend_lib/
    echo -e "${GREEN}✓ mocaxextendpy module installed${NC}"
else
    echo -e "${RED}✗ Error: mocaxextendpy module not found!${NC}"
    exit 1
fi

# Copy shared libraries
if [ -d "$MOCAXEXTEND_LIB_DIR/shared_libs" ]; then
    echo "Copying shared libraries..."
    cp -r "$MOCAXEXTEND_LIB_DIR/shared_libs" mocaxextend_lib/
    echo -e "${GREEN}✓ Shared libraries installed:${NC}"
    echo "  - libhommat.so"
    echo "  - libtensorvals.so"
else
    echo -e "${RED}✗ Error: shared_libs directory not found!${NC}"
    exit 1
fi

echo ""
echo "=============================================="
echo "Installation Summary"
echo "=============================================="
echo -e "${GREEN}✓ mocax_lib/ created successfully${NC}"
echo "  - mocaxpy/ (Python module)"
echo "  - libmocaxc.so (shared library)"
echo ""
echo -e "${GREEN}✓ mocaxextend_lib/ created successfully${NC}"
echo "  - mocaxextendpy/ (Python module)"
echo "  - shared_libs/ (libhommat.so, libtensorvals.so)"
echo ""
echo "=============================================="
echo "Next Steps"
echo "=============================================="
echo "Run the test scripts:"
echo ""
echo "  # MoCaX standard (full tensor)"
echo "  ./run_mocax_baseline.sh"
echo ""
echo "  # MoCaX Sliding (dimensional decomposition)"
echo "  ./run_mocax_sliding.sh"
echo ""
echo "  # MoCaX Extend TT (Tensor Train)"
echo "  ./run_mocax_tt.sh"
echo ""
echo -e "${GREEN}Installation complete!${NC}"
