#!/usr/bin/env python3
"""Run Keltner Bands on 5-minute data."""

import subprocess
import sys

# Run the signal generation
cmd = [
    sys.executable, 
    "main.py",
    "--config", "config/indicators/volatility/test_keltner_bands_5m_optimized.yaml",
    "--signal-generation",
    "--dataset", "SPY_5m"
]

print("Running Keltner Bands on 5-minute data...")
print(f"Command: {' '.join(cmd)}")

try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("\nSTDOUT:")
    print(result.stdout)
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    # Extract workspace path from output
    import re
    workspace_match = re.search(r'workspaces/signal_generation_\w+', result.stdout)
    if workspace_match:
        workspace = workspace_match.group()
        print(f"\n✓ Workspace created: {workspace}")
        print(f"\nTo analyze results, run:")
        print(f"python analyze_keltner_5m.py {workspace}")
    
except Exception as e:
    print(f"Error: {e}")