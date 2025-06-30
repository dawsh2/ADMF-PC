#!/usr/bin/env python3
"""Verify that our code changes are being applied."""

import importlib
import sys
import os

print("=== Verifying Code Changes ===")

# Force reload of modules
modules_to_check = [
    'src.risk.strategy_risk_manager',
    'src.portfolio.state'
]

print("\n1. Checking Python path:")
print(f"   Current working directory: {os.getcwd()}")
print(f"   Python executable: {sys.executable}")
print(f"   Python version: {sys.version}")

print("\n2. Force reloading modules:")
for module_name in modules_to_check:
    try:
        if module_name in sys.modules:
            print(f"   Reloading {module_name}")
            importlib.reload(sys.modules[module_name])
        else:
            print(f"   Importing {module_name}")
            importlib.import_module(module_name)
    except Exception as e:
        print(f"   Error with {module_name}: {e}")

print("\n3. Checking for our changes:")

# Check risk manager for entry signal fix
try:
    with open('src/risk/strategy_risk_manager.py', 'r') as f:
        content = f.read()
        if 'entry_signal_value = position.metadata.get' in content:
            print("   ✓ Risk manager has entry signal fix")
        else:
            print("   ❌ Risk manager missing entry signal fix")
            
        if 'Only store exit memory for directional signals' in content:
            print("   ✓ Risk manager has FLAT signal fix")
        else:
            print("   ❌ Risk manager missing FLAT signal fix")
except Exception as e:
    print(f"   Error reading risk manager: {e}")

# Check portfolio for entry signal storage
try:
    with open('src/portfolio/state.py', 'r') as f:
        content = f.read()
        if "metadata['entry_signal'] = entry_signal" in content:
            print("   ✓ Portfolio stores entry signal")
        else:
            print("   ❌ Portfolio missing entry signal storage")
except Exception as e:
    print(f"   Error reading portfolio: {e}")

print("\n4. Quick test of exit memory logic:")
print("   If signal = -1 opens position")
print("   Then signal changes to 0 (FLAT)")
print("   Then stop loss triggers")
print("   Exit memory should store -1 (not 0)")
print("   Future -1 signals blocked, but 0 and 1 allowed")

print("\n5. To ensure changes take effect:")
print("   a) Kill all Python processes")
print("   b) Clear all caches: find . -name '*.pyc' -delete")
print("   c) Start fresh Python session")
print("   d) Run backtest again")