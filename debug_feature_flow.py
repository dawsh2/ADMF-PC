#!/usr/bin/env python3
"""Debug the feature computation and signal generation flow."""

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

# Test 1: Check if BollingerBands feature can be created
print("=== Test 1: Feature Creation ===")
try:
    from src.strategy.components.features.indicators.volatility import BollingerBands
    bb = BollingerBands(period=20, std_dev=2.0, name="test_bb")
    print(f"✓ Created BollingerBands feature: {bb}")
    print(f"  Name: {bb.name}")
    print(f"  Period: {bb.period}")
    print(f"  Std Dev: {bb.std_dev}")
except Exception as e:
    print(f"✗ Failed to create BollingerBands: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Check if it's in FEATURE_REGISTRY
print("\n=== Test 2: Feature Registry ===")
try:
    from src.strategy.components.features import FEATURE_REGISTRY
    if 'bollinger_bands' in FEATURE_REGISTRY:
        print(f"✓ 'bollinger_bands' found in FEATURE_REGISTRY")
        print(f"  Maps to: {FEATURE_REGISTRY['bollinger_bands']}")
    else:
        print("✗ 'bollinger_bands' NOT in FEATURE_REGISTRY")
        print(f"  Available volatility features: {[k for k in FEATURE_REGISTRY.keys() if 'boll' in k or 'bb' in k]}")
except Exception as e:
    print(f"✗ Error checking FEATURE_REGISTRY: {e}")

# Test 3: Check feature update
print("\n=== Test 3: Feature Update ===")
try:
    # Create and update feature
    bb = BollingerBands(period=2, std_dev=2.0, name="test_bb")
    
    # Need at least 2 values for period=2
    prices = [100.0, 101.0, 99.0]
    for i, price in enumerate(prices):
        result = bb.update(price)
        print(f"  Update {i+1}: price={price}, result={result}")
        if result:
            print(f"    Upper: {result['upper']}")
            print(f"    Middle: {result['middle']}")
            print(f"    Lower: {result['lower']}")
except Exception as e:
    print(f"✗ Error updating feature: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check FeatureHub
print("\n=== Test 4: FeatureHub ===")
try:
    from src.strategy.components.features import FeatureHub
    hub = FeatureHub(symbols=['SPY'])
    
    # Configure with bollinger bands
    feature_configs = {
        'bollinger_bands_20_2.0_upper': {
            'type': 'bollinger_bands',
            'period': 20,
            'std_dev': 2.0,
            'component': 'upper'
        },
        'bollinger_bands_20_2.0_middle': {
            'type': 'bollinger_bands',
            'period': 20,
            'std_dev': 2.0,
            'component': 'middle'
        },
        'bollinger_bands_20_2.0_lower': {
            'type': 'bollinger_bands',
            'period': 20,
            'std_dev': 2.0,
            'component': 'lower'
        }
    }
    
    hub.configure_features(feature_configs)
    print(f"✓ Configured FeatureHub with {len(feature_configs)} features")
    
    # Try to update with bar data
    test_bar = {
        'symbol': 'SPY',
        'open': 100.0,
        'high': 101.0,
        'low': 99.0,
        'close': 100.0,
        'volume': 1000000
    }
    
    # Update 20 times to get bollinger bands ready
    print("\n  Warming up features...")
    for i in range(25):
        test_bar['close'] = 100.0 + (i % 3 - 1)  # Vary price slightly
        hub.update_bar('SPY', test_bar)
        features = hub.get_features('SPY')
        if i == 24:  # Last update
            print(f"\n  After 25 updates, features available: {len(features)}")
            bb_features = {k: v for k, v in features.items() if 'bollinger' in k}
            print(f"  Bollinger features: {bb_features}")
            
except Exception as e:
    print(f"✗ Error with FeatureHub: {e}")
    import traceback
    traceback.print_exc()