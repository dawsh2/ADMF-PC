"""Debug why features aren't being created"""
import yaml
import pandas as pd
import logging
from src.strategy.components.features.hub import FeatureHub, FEATURE_REGISTRY

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Check feature registry
print("=== Feature Registry ===")
print(f"Available feature types: {list(FEATURE_REGISTRY.keys())}")
print(f"Is 'support_resistance' registered? {'support_resistance' in FEATURE_REGISTRY}")

# Load the config
with open('config/indicators/structure/swing_pivot_bounce.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("\n=== Configuration ===")
print(f"Feature config: {config['feature_config']}")

# Create FeatureHub
print("\n=== Creating FeatureHub ===")
feature_hub = FeatureHub(symbols=['SPY'])
feature_hub.configure_features(config['feature_config'])

# Check internal state
print(f"\nConfigured features: {feature_hub._feature_configs}")
print(f"Features dict: {feature_hub._features}")

# Create test data with varying prices to trigger S/R detection
print("\n=== Creating test data ===")
prices = [520, 521, 522, 521, 520, 519, 518, 519, 520, 521, 522, 523, 522, 521, 520,
          519, 518, 517, 518, 519, 520, 521, 522, 521, 520, 519, 520, 521, 522, 523]
          
test_data = []
for i, price in enumerate(prices):
    bar = {
        'symbol': 'SPY',
        'open': price,
        'high': price + 0.5,
        'low': price - 0.5,
        'close': price,
        'volume': 1000000
    }
    test_data.append(bar)

# Update features with test data
print("\n=== Updating features ===")
for i, bar in enumerate(test_data):
    print(f"\nBar {i}: price={bar['close']}")
    feature_hub.update_bar('SPY', bar)
    
    # Check features after each update
    features = feature_hub.get_features('SPY')
    sr_keys = [k for k in features.keys() if 'support' in k or 'resistance' in k]
    if sr_keys:
        print(f"  S/R features found: {sr_keys}")
        for key in sr_keys:
            print(f"    {key}: {features[key]}")

# Final feature state
print("\n=== Final Feature State ===")
features = feature_hub.get_features('SPY')
print(f"All feature keys: {list(features.keys())}")

# Check if features were created for the symbol
print(f"\nFeatures created for SPY: {feature_hub._features.get('SPY', {}).keys()}")

# Get feature summary
summary = feature_hub.get_feature_summary()
print(f"\nFeature summary: {summary}")