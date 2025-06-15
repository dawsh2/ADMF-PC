#!/usr/bin/env python3
"""
Debug why strategies aren't executing
"""
import sys
import importlib

# Test importing volume strategies
print("Testing strategy imports...")

strategies_to_test = [
    ('obv_trend', 'src.strategy.strategies.indicators.volume'),
    ('mfi_bands', 'src.strategy.strategies.indicators.volume'),
    ('vwap_deviation', 'src.strategy.strategies.indicators.volume'),
    ('pivot_points', 'src.strategy.strategies.indicators.structure'),
    ('fibonacci_retracement', 'src.strategy.strategies.indicators.structure'),
]

for strategy_name, module_path in strategies_to_test:
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, strategy_name):
            func = getattr(module, strategy_name)
            print(f"✓ {strategy_name} found in {module_path}")
            
            # Check if it's decorated
            if hasattr(func, '_strategy_metadata'):
                print(f"  Metadata: {func._strategy_metadata}")
            else:
                print(f"  WARNING: No @strategy decorator metadata!")
        else:
            print(f"✗ {strategy_name} NOT FOUND in {module_path}")
            print(f"  Available: {[name for name in dir(module) if not name.startswith('_')]}")
    except ImportError as e:
        print(f"✗ Cannot import {module_path}: {e}")

# Check if feature hub has the required features
print("\nChecking feature support...")
from src.strategy.components.features.hub import FEATURE_REGISTRY

required_features = ['obv', 'mfi', 'vwap', 'pivot', 'fibonacci', 'support_resistance']
for feature in required_features:
    if feature in FEATURE_REGISTRY:
        print(f"✓ Feature '{feature}' is registered")
    else:
        print(f"✗ Feature '{feature}' is NOT registered")
        # Check for similar names
        similar = [f for f in FEATURE_REGISTRY if feature in f or f in feature]
        if similar:
            print(f"  Similar features: {similar}")

# Test a strategy execution
print("\nTesting strategy execution...")
try:
    from src.strategy.strategies.indicators.oscillators import rsi_threshold
    
    # Mock data
    test_features = {'rsi_14': 45.5}
    test_bar = {'close': 100.0, 'timestamp': '2023-01-01', 'symbol': 'TEST', 'timeframe': '1m'}
    test_params = {'rsi_period': 14, 'threshold': 50}
    
    result = rsi_threshold(test_features, test_bar, test_params)
    print(f"✓ RSI threshold executed successfully: {result}")
except Exception as e:
    print(f"✗ RSI threshold execution failed: {e}")
    import traceback
    traceback.print_exc()