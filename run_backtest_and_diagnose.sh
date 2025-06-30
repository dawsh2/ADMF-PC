#!/bin/bash
# Run backtest and diagnose the results

echo "=== Running Backtest with All Fixes ==="
echo "Starting at $(date)"

# Clear Python cache first
echo "Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# Run the backtest
echo -e "\nRunning backtest..."
python main.py --config config/bollinger/test.yaml

# Check if backtest completed
if [ $? -eq 0 ]; then
    echo -e "\n✓ Backtest completed successfully!"
    
    # Wait a moment for files to be written
    sleep 2
    
    # Run diagnostics
    echo -e "\n=== Running Diagnostics ==="
    
    echo -e "\n1. Checking implementation..."
    python check_ohlc_implementation.py
    
    echo -e "\n\n2. Debugging trade count..."
    python debug_463_trades.py
    
    echo -e "\n\n3. Analyzing close vs high/low exits..."
    python analyze_close_vs_hl_exits.py
    
    echo -e "\n\n4. Comparing with notebook..."
    python compare_with_notebook_exactly.py
    
    echo -e "\n\n5. Checking bar ranges..."
    python analyze_bar_ranges.py
    
else
    echo -e "\n❌ Backtest failed!"
    echo "Please check the error messages above."
fi

echo -e "\nCompleted at $(date)"