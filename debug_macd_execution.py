#!/usr/bin/env python3
"""Debug MACD strategy execution."""

import yaml
import sys
sys.path.append('.')

from src.core.coordinator.topology import TopologyBuilder
from src.strategy.components.features.hub import FeatureHub

# Test MACD strategy execution
print("Testing MACD strategy execution...")

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Find MACD strategy
macd_strategies = [s for s in config.get('strategies', []) if s.get('type') == 'macd_crossover']
macd_config = macd_strategies[0]

# Expand parameters
builder = TopologyBuilder()
expanded = builder._expand_strategy_parameters([macd_config])

# Test first instance
first_instance = expanded[0]
print(f"Testing instance: {first_instance}")

# Infer features
inferred = builder._infer_features_from_strategies([first_instance])
print(f"Inferred features: {sorted(inferred)}")

# Create FeatureHub and configure features
feature_hub = FeatureHub(['SPY'])

# Convert inferred features to feature configs
feature_configs = {}
for feature_name in inferred:
    if feature_name.startswith('macd_'):
        parts = feature_name.split('_')
        if len(parts) == 4:  # macd_fast_slow_signal
            fast, slow, signal = int(parts[1]), int(parts[2]), int(parts[3])
            feature_configs[feature_name] = {
                'type': 'macd',
                'fast_period': fast,
                'slow_period': slow,
                'signal_period': signal
            }

print(f"Feature configs: {feature_configs}")
feature_hub.configure_features(feature_configs)

# Create test bars
import time
test_bars = []
base_price = 100.0
for i in range(50):  # 50 bars for testing
    price = base_price + (i * 0.5) + (i % 10) * 0.2  # Trending with noise
    bar = {
        'symbol': 'SPY',
        'timestamp': int(time.time()) + i,
        'open': price - 0.1,
        'high': price + 0.2,
        'low': price - 0.2,
        'close': price,
        'volume': 1000000 + i * 1000
    }
    test_bars.append(bar)

print(f"Created {len(test_bars)} test bars")

# Process bars through FeatureHub
print("Processing bars through FeatureHub...")
for i, bar in enumerate(test_bars):
    feature_hub.update_bar('SPY', bar)
    
    # Check feature readiness every 10 bars
    if i % 10 == 9:
        features = feature_hub.get_features('SPY')
        macd_feature_name = list(inferred)[0]  # First MACD feature
        macd_value = features.get(macd_feature_name)
        print(f"  Bar {i+1}: {macd_feature_name} = {macd_value}")

# Final feature check
final_features = feature_hub.get_features('SPY')
print(f"\nFinal features available: {list(final_features.keys())}")
for feature_name in inferred:
    value = final_features.get(feature_name)
    print(f"  {feature_name}: {value}")

# Test strategy execution
print(f"\nTesting strategy execution...")

# Import strategy function
from src.strategy.strategies.indicators.crossovers import macd_crossover

# Test with final features
try:
    result = macd_crossover(final_features, test_bars[-1], first_instance['params'])
    print(f"Strategy result: {result}")
    
    if result is None:
        print("\nDebugging why strategy returned None...")
        params = first_instance['params']
        fast_ema = params.get('fast_ema', 12)
        slow_ema = params.get('slow_ema', 26)
        signal_ema = params.get('signal_ema', 9)
        
        expected_macd_key = f'macd_{fast_ema}_{slow_ema}_{signal_ema}_macd'
        expected_signal_key = f'macd_{fast_ema}_{slow_ema}_{signal_ema}_signal'
        
        macd_line = final_features.get(expected_macd_key)
        signal_line = final_features.get(expected_signal_key)
        
        print(f"Expected MACD line key: {expected_macd_key} = {macd_line}")
        print(f"Expected signal line key: {expected_signal_key} = {signal_line}")
        
        # Check what MACD feature actually provides
        macd_feature_value = final_features.get(f'macd_{fast_ema}_{slow_ema}_{signal_ema}')
        print(f"Actual MACD feature value: {macd_feature_value}")
        
except Exception as e:
    print(f"Strategy execution failed: {e}")
    import traceback
    traceback.print_exc()