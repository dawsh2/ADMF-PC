#!/usr/bin/env python3
"""
Minimal test of main.py functionality.
"""

import subprocess
import sys

print("Testing main.py with --dry-run flag...")
print("=" * 70)

cmd = [
    sys.executable,
    "main.py",
    "--config", "configs/simple_synthetic_backtest.yaml",
    "--bars", "100",
    "--dry-run"
]

print(f"Running: {' '.join(cmd)}")
print("")

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    print(f"\nReturn code: {result.returncode}")
    
    if result.returncode == 0:
        print("\n✅ SUCCESS: main.py executed without errors!")
    else:
        print("\n❌ FAILED: main.py returned non-zero exit code")
        
except subprocess.TimeoutExpired:
    print("\n❌ TIMEOUT: Command took too long to execute")
except Exception as e:
    print(f"\n❌ ERROR: {e}")