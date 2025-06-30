#!/bin/bash
# Clear Python bytecode cache

echo "=== Clearing Python Cache ==="

# Find and remove all __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Find and remove all .pyc files
echo "Removing .pyc files..."
find . -name "*.pyc" -delete

# Find and remove all .pyo files
echo "Removing .pyo files..."
find . -name "*.pyo" -delete

echo "✓ Python cache cleared!"
echo ""
echo "Now restart Python by:"
echo "1. If using command line: exit() and restart python"
echo "2. If using Jupyter: Kernel -> Restart"
echo "3. If using IPython: exit and restart"
echo "4. If using an IDE: restart the Python console/terminal"