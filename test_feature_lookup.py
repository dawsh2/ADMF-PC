#!/usr/bin/env python3
"""Test if strategies can lookup features correctly."""

from src.strategy.strategies.indicators.volatility import bollinger_breakout

print("=== TESTING FEATURE LOOKUP ===\n")

# Test case 1: Features with exact parameter match
features1 = {
    'bollinger_bands_11_1.5_upper': 105.0,
    'bollinger_bands_11_1.5_lower': 95.0,
    'bollinger_bands_11_1.5_middle': 100.0,
    'bar_count': 100
}

bar = {
    'timestamp': 1234567890,
    'close': 106.0,  # Above upper band
    'symbol': 'SPY',
    'timeframe': '1m'
}

params1 = {'period': 11, 'std_dev': 1.5}

print("Test 1: Exact parameter match")
print(f"  Params: {params1}")
print(f"  Price: {bar['close']}")
print(f"  Upper band: {features1['bollinger_bands_11_1.5_upper']}")

result1 = bollinger_breakout(features1, bar, params1)
print(f"  Result: {result1}")
print(f"  Expected: signal_value = -1 (price above upper band)")

# Test case 2: Features with different parameters
features2 = {
    'bollinger_bands_20_2.0_upper': 105.0,
    'bollinger_bands_20_2.0_lower': 95.0,
    'bollinger_bands_20_2.0_middle': 100.0,
    'bollinger_bands_11_1.5_upper': 103.0,  # Different params
    'bollinger_bands_11_1.5_lower': 97.0,
    'bollinger_bands_11_1.5_middle': 100.0,
    'bar_count': 100
}

params2 = {'period': 20, 'std_dev': 2.0}  # Looking for 20_2.0

print("\n\nTest 2: Multiple features, different params")
print(f"  Params: {params2}")
print(f"  Features available: {list(features2.keys())}")
print(f"  Looking for: bollinger_bands_{params2['period']}_{params2['std_dev']}_upper")

result2 = bollinger_breakout(features2, bar, params2)
print(f"  Result: {result2}")

# Test case 3: Missing features
features3 = {
    'bollinger_bands_11_1.5_upper': 105.0,  # Wrong params
    'bollinger_bands_11_1.5_lower': 95.0,
    'bollinger_bands_11_1.5_middle': 100.0,
    'bar_count': 100
}

params3 = {'period': 20, 'std_dev': 2.0}  # Looking for 20_2.0 which doesn't exist

print("\n\nTest 3: Missing features")
print(f"  Params: {params3}")
print(f"  Looking for: bollinger_bands_{params3['period']}_{params3['std_dev']}_upper")
print(f"  Available: {[k for k in features3.keys() if 'bollinger' in k]}")

result3 = bollinger_breakout(features3, bar, params3)
print(f"  Result: {result3}")
print(f"  Expected: None (features not found)")

# Test with exact grid search parameters
print("\n\n=== GRID SEARCH PARAMETER TEST ===")

# From config: period: [11, 19, 27, 35], std_dev: [1.5, 2.0, 2.5]
grid_params = {'period': 11, 'std_dev': 1.5}
grid_features = {}

# Build feature name as the feature hub would
feature_id = f"bollinger_bands_{grid_params['period']}_{grid_params['std_dev']}"
grid_features[f"{feature_id}_upper"] = 105.0
grid_features[f"{feature_id}_lower"] = 95.0
grid_features[f"{feature_id}_middle"] = 100.0

print(f"Grid params: {grid_params}")
print(f"Feature ID: {feature_id}")
print(f"Features created: {list(grid_features.keys())}")

result = bollinger_breakout(grid_features, bar, grid_params)
print(f"Result: {result}")
if result:
    print("✅ SUCCESS - Strategy found features correctly!")
else:
    print("❌ FAILED - Strategy couldn't find features")