#!/usr/bin/env python3
"""
Test the minimal architecture.
"""

import subprocess
import sys

print("Testing minimal main.py architecture...")
print("=" * 70)

# Test 1: Dry run
print("\nTest 1: Dry run")
cmd = [sys.executable, "main.py", "--config", "configs/simple_synthetic_backtest.yaml", "--bars", "100", "--dry-run", "--verbose"]
print(f"Command: {' '.join(cmd)}")
print()

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
    
    if result.returncode == 0:
        print("\n✅ Dry run successful!")
        
        # Test 2: Actual run
        print("\n" + "=" * 70)
        print("Test 2: Actual backtest")
        cmd2 = [sys.executable, "main.py", "--config", "configs/simple_synthetic_backtest.yaml", "--bars", "100"]
        print(f"Command: {' '.join(cmd2)}")
        print()
        
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
        print("STDOUT:")
        print(result2.stdout)
        if result2.stderr:
            print("\nSTDERR:")  
            print(result2.stderr)
        print(f"\nReturn code: {result2.returncode}")
        
        if result2.returncode == 0:
            print("\n✅ Backtest successful!")
        else:
            print("\n❌ Backtest failed")
    else:
        print("\n❌ Dry run failed")
        
except subprocess.TimeoutExpired:
    print("\n❌ Command timed out")
except Exception as e:
    print(f"\n❌ Error: {e}")