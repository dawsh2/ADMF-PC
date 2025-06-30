#!/bin/bash
# Run the analysis on latest results

echo "Running analysis on latest results..."
cd /Users/daws/ADMF-PC

# Use the python from venv
if [ -d "venv" ]; then
    echo "Using venv Python..."
    venv/bin/python analyze_latest_results.py
else
    echo "Using system Python..."
    python3 analyze_latest_results.py
fi