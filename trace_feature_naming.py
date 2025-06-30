"""Trace exact feature naming through the system to find the mismatch"""
import yaml
import pandas as pd
from src.strategy.components.features.hub import FeatureHub
from src.strategy.components.features.indicators.structure import SupportResistance

# Load the config to see what features are being created
with open('config/indicators/structure/swing_pivot_bounce.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=== Configuration Analysis ===")
print(f"Config feature names: {list(config['feature_config'].keys())}")
print(f"Full config: {config}")

# Create FeatureHub with the config
print("\n=== FeatureHub Creation ===")
feature_hub = FeatureHub(config.get('feature_config', {}))

# Print what features were actually created
print(f"\nFeatures in hub: {list(feature_hub.features.keys())}")

# Create some test data
test_data = pd.DataFrame({
    'symbol': ['SPY'] * 50,
    'open': [520] * 50,
    'high': [521] * 50,
    'low': [519] * 50,
    'close': [520] * 50,
    'volume': [1000000] * 50
})

# Update features with test data
print("\n=== Feature Updates ===")
for i in range(30):
    bar = test_data.iloc[i].to_dict()
    feature_hub.update(bar)

# Get features for a bar and see what keys are available
bar = test_data.iloc[30].to_dict()
features = feature_hub.get_features(bar)

print(f"\nFeature keys returned by hub: {list(features.keys())}")
print(f"\nFeature values:")
for key, value in features.items():
    if 'support' in key or 'resistance' in key:
        print(f"  {key}: {value}")

# Now check what the strategy expects
print("\n=== Strategy Expectations ===")
params = config['strategy_params']
sr_period = params.get('sr_period', 20)

expected_resistance_key = f'support_resistance_{sr_period}_resistance'
expected_support_key = f'support_resistance_{sr_period}_support'

print(f"Strategy expects resistance key: '{expected_resistance_key}'")
print(f"Strategy expects support key: '{expected_support_key}'")

print(f"\nActual resistance key present: '{expected_resistance_key}' in features = {expected_resistance_key in features}")
print(f"Actual support key present: '{expected_support_key}' in features = {expected_support_key in features}")

# List all keys containing 'resistance' or 'support'
print("\n=== All S/R Related Keys ===")
sr_keys = [k for k in features.keys() if 'support' in k or 'resistance' in k]
print(f"S/R related keys in features: {sr_keys}")

# Test direct feature creation
print("\n=== Direct Feature Test ===")
sr = SupportResistance(lookback=20, min_touches=2)
for i in range(30):
    bar = test_data.iloc[i].to_dict()
    sr_value = sr.update(bar['close'], bar['high'], bar['low'])
    
print(f"Direct S/R value: {sr_value}")

# Check feature decomposition logic
print("\n=== Feature Decomposition Logic ===")
feature_name = 'support_resistance_20_2'
print(f"Original feature name: {feature_name}")

# Simulate what happens in FeatureHub
if hasattr(feature_hub.features.get(feature_name), 'update'):
    feature_instance = feature_hub.features[feature_name]
    test_value = feature_instance.update(520, 521, 519)
    if isinstance(test_value, dict):
        print(f"Feature returns dict with keys: {list(test_value.keys())}")
        for sub_key, sub_value in test_value.items():
            decomposed_key = f"{feature_name}_{sub_key}"
            print(f"  Decomposed key: {decomposed_key} = {sub_value}")