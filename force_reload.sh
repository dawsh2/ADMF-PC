#!/bin/bash
# Force reload of Python modules

echo "=== Forcing Complete Python Reload ==="

# 1. Clear all Python cache
echo "1. Clearing all Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
rm -rf .pytest_cache 2>/dev/null

# 2. Clear pip cache
echo "2. Clearing pip cache..."
pip cache purge 2>/dev/null || echo "   (pip cache purge not available)"

# 3. Show Python process
echo "3. Current Python processes:"
ps aux | grep python | grep -v grep | head -5

echo ""
echo "=== IMPORTANT ==="
echo "You must:"
echo "1. Exit/restart any Python interpreter or Jupyter notebook"
echo "2. If using Jupyter, restart the kernel (Kernel -> Restart)"
echo "3. If using terminal Python, exit() and start fresh"
echo ""
echo "The changes won't take effect until you start a fresh Python process!"
echo ""
echo "After restarting Python, run:"
echo "  python main.py config/bollinger/test.yaml"