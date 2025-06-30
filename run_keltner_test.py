#!/usr/bin/env python3
"""Run Keltner Bands test."""

import subprocess
import sys
import os

# Change to project directory
os.chdir('/Users/daws/ADMF-PC')

# Command to run
cmd = [
    sys.executable,  # Use current Python interpreter
    'main.py',
    '--config', 'config/indicators/volatility/test_keltner_bands_working.yaml',
    '--signal-generation',
    '--dataset', 'train'
]

print("Running command:")
print(' '.join(cmd))
print()

# Run the command
try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    print(f"\nReturn code: {result.returncode}")
    
except Exception as e:
    print(f"Error running command: {e}")