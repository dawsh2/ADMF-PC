#!/usr/bin/env python3
"""Trace how parameters flow through the system."""

import yaml
import importlib
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.coordinator.compiler import StrategyCompiler
from src.core.components.discovery import ComponentDiscovery

print("TRACING PARAMETER FLOW")
print("=" * 50)

# 1. Load ensemble config
print("\n1. Loading ensemble config:")
with open('config/ensemble/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"   Config: {config}")
print(f"   Strategy list: {config['strategy']}")
print(f"   First strategy: {config['strategy'][0]}")

# 2. Initialize compiler
print("\n2. Initializing compiler:")
discovery = ComponentDiscovery()
compiler = StrategyCompiler(discovery)

# 3. Compile the strategy
print("\n3. Compiling strategy:")
try:
    compiled = compiler.compile(config['strategy'])
    print("   ✅ Compilation successful")
    
    # Check if it's a composite
    if hasattr(compiled['function'], '_sub_strategies'):
        print(f"   Sub-strategies: {compiled['function']._sub_strategies}")
    
    # Check metadata
    print(f"   Metadata: {compiled.get('metadata', {})}")
    
except Exception as e:
    print(f"   ❌ Compilation failed: {e}")
    import traceback
    traceback.print_exc()

# 4. Check atomic compilation
print("\n4. Testing atomic compilation directly:")
atomic_config = config['strategy'][0]  # {bollinger_bands: {period: 15, std_dev: 3.0}}
print(f"   Atomic config: {atomic_config}")

try:
    # Test _compile_atomic directly
    atomic_func = compiler._compile_atomic(atomic_config)
    print("   ✅ Atomic compilation successful")
    
    # Try to inspect the function
    if hasattr(atomic_func, '__closure__') and atomic_func.__closure__:
        print("   Function has closure variables")
        # Try to find params in closure
        for cell in atomic_func.__closure__:
            try:
                val = cell.cell_contents
                if isinstance(val, dict) and ('period' in val or 'std_dev' in val):
                    print(f"   Found params in closure: {val}")
            except:
                pass
                
except Exception as e:
    print(f"   ❌ Atomic compilation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n5. ANALYSIS:")
print("-" * 40)
print("The issue appears to be in how the MultiStrategyTracer")
print("extracts parameters from compiled strategies.")
print("\nThe parameters are compiled into the function closure")
print("but may not be accessible to the tracer.")