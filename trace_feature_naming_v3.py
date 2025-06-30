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
print(f"Feature config details: {config['feature_config']}")

# Create FeatureHub
print("\n=== FeatureHub Creation ===")
feature_hub = FeatureHub(symbols=['SPY'])
feature_hub.configure_features(config['feature_config'])

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
    feature_hub.update_bar('SPY', bar)

# Get features and see what keys are available
features = feature_hub.get_features('SPY')

print(f"\nFeature keys returned by hub: {list(features.keys())}")
print(f"\nFeature values:")
for key, value in features.items():
    if 'support' in key or 'resistance' in key:
        print(f"  {key}: {value}")

# Now check what the strategy expects
print("\n=== Strategy Expectations ===")
params = config['strategy']['swing_pivot_bounce']['params']
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

# The mismatch issue
print("\n=== Feature Naming Mismatch ===")
print("The problem:")
print(f"1. Config defines feature as: 'support_resistance_20_2' (includes both params)")
print(f"2. FeatureHub decomposes dict values as: 'support_resistance_20_2_resistance'")
print(f"3. Strategy expects: 'support_resistance_20_resistance' (only one param)")
print("\nThis is why the strategy gets None values and generates no signals!")

# Solution
print("\n=== Solution ===")
print("Change the config feature name from 'support_resistance_20_2' to 'support_resistance_20'")
print("This will make FeatureHub create keys that match what the strategy expects")