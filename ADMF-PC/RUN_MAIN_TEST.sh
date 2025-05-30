#!/bin/bash

# Test main.py with the configuration
echo "Testing main.py with synthetic data configuration..."
echo "="*70
echo ""

# First try with dry-run
echo "Test 1: Dry run mode"
echo "Command: python main.py --config configs/simple_synthetic_backtest.yaml --bars 100 --dry-run"
echo ""
python main.py --config configs/simple_synthetic_backtest.yaml --bars 100 --dry-run

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dry run successful!"
    echo ""
    echo "Test 2: Actual backtest"
    echo "Command: python main.py --config configs/simple_synthetic_backtest.yaml --bars 100"
    echo ""
    python main.py --config configs/simple_synthetic_backtest.yaml --bars 100
else
    echo ""
    echo "❌ Dry run failed"
fi