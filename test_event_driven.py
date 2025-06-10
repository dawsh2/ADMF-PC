#!/usr/bin/env python3
"""Test event-driven execution with real bar streaming."""

import subprocess
import sys

# Run the test
result = subprocess.run([
    sys.executable, 
    "main.py",
    "--config", "config/test_bar_streaming.yaml",
    "--dry-run",
    "--verbose"
], capture_output=True, text=True)

# Print output
print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")