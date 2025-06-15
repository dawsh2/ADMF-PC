#!/usr/bin/env python3
"""
Test strategy execution to see how many strategies successfully run
"""
import importlib
import traceback
from collections import defaultdict

# Import feature hub
from src.strategy.components.features.hub import FEATURE_REGISTRY, compute_feature
import pandas as pd
import numpy as np

# Test data
test_bar = {
    'open': 100.0,
    'high': 101.0,
    'low': 99.0,
    'close': 100.5,
    'volume': 1000000,
    'timestamp': '2023-01-01 09:30:00',
    'symbol': 'TEST',
    'timeframe': '1m'
}

# Create test DataFrame with 100 bars
dates = pd.date_range('2023-01-01 09:30:00', periods=100, freq='1min')
test_df = pd.DataFrame({
    'open': np.random.uniform(99, 101, 100),
    'high': np.random.uniform(100, 102, 100),
    'low': np.random.uniform(98, 100, 100),
    'close': np.random.uniform(99, 101, 100),
    'volume': np.random.uniform(900000, 1100000, 100)
}, index=dates)

# Compute all features
print("Computing features...")
features = {}
for feature_name, feature_func in FEATURE_REGISTRY.items():
    try:
        result = compute_feature(feature_name, test_df)
        if isinstance(result, dict):
            for sub_name, series in result.items():
                if len(series) > 0 and not pd.isna(series.iloc[-1]):
                    features[f"{feature_name}_{sub_name}"] = float(series.iloc[-1])
        else:
            if len(result) > 0 and not pd.isna(result.iloc[-1]):
                features[feature_name] = float(result.iloc[-1])
        print(f"✓ Computed {feature_name}")
    except Exception as e:
        print(f"✗ Failed to compute {feature_name}: {e}")

print(f"\nTotal features computed: {len(features)}")

# Test all strategies
strategy_modules = [
    'src.strategy.strategies.indicators.crossovers',
    'src.strategy.strategies.indicators.oscillators',
    'src.strategy.strategies.indicators.structure',
    'src.strategy.strategies.indicators.trend',
    'src.strategy.strategies.indicators.volatility',
    'src.strategy.strategies.indicators.volume',
]

successful_strategies = []
failed_strategies = []

print("\nTesting strategies...")
for module_path in strategy_modules:
    try:
        module = importlib.import_module(module_path)
        
        # Get all strategy functions
        strategies = [name for name in dir(module) if not name.startswith('_') and callable(getattr(module, name))]
        
        for strategy_name in strategies:
            try:
                func = getattr(module, strategy_name)
                if hasattr(func, '_strategy_metadata'):
                    # Test with default parameters
                    result = func(features, test_bar, {})
                    if result is not None:
                        successful_strategies.append(f"{module_path.split('.')[-1]}.{strategy_name}")
                        print(f"✓ {strategy_name} executed successfully")
                    else:
                        failed_strategies.append((f"{module_path.split('.')[-1]}.{strategy_name}", "Returned None"))
                        print(f"✗ {strategy_name} returned None")
            except Exception as e:
                failed_strategies.append((f"{module_path.split('.')[-1]}.{strategy_name}", str(e)))
                print(f"✗ {strategy_name} failed: {e}")
                
    except Exception as e:
        print(f"✗ Could not import {module_path}: {e}")

print(f"\n=== SUMMARY ===")
print(f"Successful strategies: {len(successful_strategies)}")
print(f"Failed strategies: {len(failed_strategies)}")

if failed_strategies:
    print(f"\nFailed strategy details:")
    error_counts = defaultdict(int)
    for strategy, error in failed_strategies:
        error_counts[error] += 1
        print(f"  - {strategy}: {error}")
    
    print(f"\nError summary:")
    for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {error}: {count} strategies")

print(f"\nSuccess rate: {len(successful_strategies) / (len(successful_strategies) + len(failed_strategies)) * 100:.1f}%")