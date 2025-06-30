#!/bin/bash
# Clear Python cache and test exit memory fix

echo "=== Clearing Python Cache ==="
echo "Removing all __pycache__ directories..."

# Clear all Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null

echo "✓ Python cache cleared"
echo ""
echo "=== Next Steps ==="
echo "1. Restart any Python processes/Jupyter kernels"
echo "2. Run: python main.py config/bollinger/test.yaml"
echo "3. Check the notebook for trade count and immediate re-entries"
echo ""
echo "Expected results after fix:"
echo "- More than 7 trades (exit memory not stuck on FLAT)"
echo "- Fewer immediate re-entries (exit memory working for directional signals)"
echo "- Total trades between 7 and 463"