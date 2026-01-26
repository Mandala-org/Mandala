#!/bin/bash

# Minimal Overfit Study Runner
# =============================
# Runs the minimal water overfit study with verbose logging

set -e  # Exit on error

echo "============================================"
echo "  Minimal Water Overfit Study"
echo "============================================"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must run from project root (mandala/)"
    exit 1
fi

# Check if data exists
if [ ! -f "data/small/H2O/original/H2O.matrix" ]; then
    echo "Error: Water data not found at data/small/H2O/original/"
    echo "Please ensure data files are present"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "mandala-venv" ]; then
    echo "Activating virtual environment..."
    source mandala-venv/bin/activate
fi

# Check dependencies
echo "Checking dependencies..."
python -c "import torch; import e3nn; import ase" 2>/dev/null || {
    echo "Error: Missing dependencies. Please install:"
    echo "  pip install -e .[dev]"
    exit 1
}

echo "✓ Dependencies OK"
echo ""

# Run the study
echo "Starting minimal overfit study..."
echo "Output will be verbose. Consider redirecting to a file:"
echo "  ./studies/minimal_overfit_study/run.sh > overfit.log 2>&1"
echo ""

python studies/minimal_overfit_study/overfit_water_minimal.py

echo ""
echo "============================================"
echo "  Study Complete!"
echo "============================================"
