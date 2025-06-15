#!/usr/bin/env python3
"""
Test what features actually return
"""
from src.strategy.components.features.hub import compute_feature
import pandas as pd
import numpy as np

# Create test data
dates = pd.date_range('2023-01-01', periods=100, freq='1min')
test_df = pd.DataFrame({
    'open': np.random.uniform(99, 101, 100),
    'high': np.random.uniform(100, 102, 100),
    'low': np.random.uniform(98, 100, 100),
    'close': np.random.uniform(99, 101, 100),
    'volume': np.random.uniform(900000, 1100000, 100)
}, index=dates)

# Test various features that strategies need
features_to_test = [
    ('pivot_points', {'pivot_type': 'standard'}),
    ('fibonacci_retracement', {'fib_period': 50}),
    ('swing_points', {'swing_period': 10}),
    ('support_resistance', {'sr_period': 20, 'sr_threshold': 0.02}),
    ('aroon', {'aroon_period': 25}),
    ('stochastic_rsi', {'rsi_period': 14, 'stoch_period': 14}),
    ('ultimate_oscillator', {'uo_period1': 7, 'uo_period2': 14, 'uo_period3': 28}),
]

for feature_name, params in features_to_test:
    print(f"\n=== Testing {feature_name} ===")
    try:
        result = compute_feature(feature_name, test_df, **params)
        
        if isinstance(result, dict):
            print(f"Returns dict with keys: {list(result.keys())}")
            # Show sample values
            for key, series in result.items():
                if hasattr(series, 'iloc') and len(series) > 0:
                    print(f"  {key}: {series.iloc[-1]:.4f}")
        else:
            print(f"Returns single series")
            if hasattr(result, 'iloc') and len(result) > 0:
                print(f"  Last value: {result.iloc[-1]:.4f}")
                
    except Exception as e:
        print(f"ERROR: {e}")