#!/usr/bin/env python3
"""
Test if features for missing strategies can be computed
"""
from src.strategy.components.features.hub import compute_feature, FEATURE_REGISTRY
import pandas as pd
import numpy as np

# Create test data with 300 bars
dates = pd.date_range('2023-01-01', periods=300, freq='1min')
test_df = pd.DataFrame({
    'open': 100 + np.random.randn(300).cumsum() * 0.1,
    'high': 101 + np.random.randn(300).cumsum() * 0.1,
    'low': 99 + np.random.randn(300).cumsum() * 0.1,
    'close': 100 + np.random.randn(300).cumsum() * 0.1,
    'volume': 1000000 + np.random.randint(-100000, 100000, 300)
}, index=dates)

# Make sure high/low are correct
test_df['high'] = test_df[['open', 'high', 'close']].max(axis=1)
test_df['low'] = test_df[['open', 'low', 'close']].min(axis=1)

print(f"Test data shape: {test_df.shape}")
print(f"Date range: {test_df.index[0]} to {test_df.index[-1]}")

# Test features for missing strategies
test_features = [
    ('pivot_points', {'pivot_type': 'standard'}),
    ('fibonacci_retracement', {'period': 50}),
    ('swing_points', {'period': 10}),
    ('support_resistance', {'period': 20, 'threshold': 0.02}),
    ('linear_regression', {'period': 20}),
    ('aroon', {'period': 25}),
    ('psar', {'af_start': 0.02, 'af_max': 0.2}),
    ('supertrend', {'period': 10, 'multiplier': 3.0}),
    ('ultimate_oscillator', {'period1': 7, 'period2': 14, 'period3': 28}),
    ('stochastic_rsi', {'rsi_period': 14, 'stoch_period': 14}),
    ('vortex', {'period': 14}),
    ('macd', {'fast': 12, 'slow': 26, 'signal': 9}),
    ('stochastic', {'k_period': 14, 'd_period': 3}),
    ('ichimoku', {'conversion_period': 9, 'base_period': 26}),
    ('roc', {'period': 10}),
    ('bollinger_bands', {'period': 20, 'std_dev': 2.0}),
    ('donchian_channel', {'period': 20}),
    ('keltner_channel', {'period': 20, 'multiplier': 2.0}),
    ('obv', {}),
    ('vwap', {}),
    ('ad', {}),
    ('adx', {'period': 14}),
]

print("\nTesting feature computation:")
success_count = 0
failed_features = []

for feature_name, params in test_features:
    try:
        result = compute_feature(feature_name, test_df, **params)
        if result is not None:
            if isinstance(result, dict):
                last_values = {k: v.iloc[-1] if hasattr(v, 'iloc') else v for k, v in result.items()}
                print(f"✓ {feature_name}: Computed successfully, components: {list(result.keys())}")
            else:
                print(f"✓ {feature_name}: Computed successfully, last value: {result.iloc[-1]:.4f}")
            success_count += 1
        else:
            print(f"✗ {feature_name}: Returned None")
            failed_features.append(feature_name)
    except Exception as e:
        print(f"✗ {feature_name}: ERROR - {str(e)}")
        failed_features.append(feature_name)

print(f"\n=== SUMMARY ===")
print(f"Success: {success_count}/{len(test_features)}")
print(f"Failed: {len(failed_features)}")

if failed_features:
    print(f"\nFailed features: {failed_features}")
    print("\nThese features are preventing their strategies from executing!")