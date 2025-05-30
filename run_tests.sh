#!/bin/bash
# Script to run tests with virtual environment

# Update this path to your actual venv location
VENV_PATH="./venv"  # Change this!

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please update VENV_PATH in this script to point to your venv"
    exit 1
fi

# Activate venv and run tests
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo ""

# Restore the commented imports first
echo "Restoring monitoring imports..."
git checkout src/core/infrastructure/capabilities.py src/core/infrastructure/__init__.py 2>/dev/null || echo "Already restored"

echo ""
echo "Running integration tests..."
python test_basic_backtest_simple.py

# Run other tests
echo ""
echo "Running minimal test..."
python test_minimal_integration.py

echo ""
echo "Done!"