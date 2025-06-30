"""Demonstrate how feature configs get overridden"""
import yaml

print("=== FEATURE OVERRIDE DEMONSTRATION ===\n")

# Example 1: What we write in our YAML
yaml_config_we_write = """
name: test_keltner_with_rsi_filter
mode: signal_generation
symbols: ["SPY"]

# WE DEFINE THESE FEATURES FOR THE FILTER
features:
  - name: rsi
    params: {period: 14}
  - name: atr  
    params: {period: 14}

# OR IN THE DICT FORMAT:
feature_configs:
  rsi_14:
    feature: rsi
    params: {period: 14}
  atr_14:
    feature: atr
    params: {period: 14}

strategies:
  - name: keltner_filtered
    type: keltner_bands
    params:
      period: 20
      multiplier: 2.0
    filter: "signal != 0 and rsi(14) < 30"  # This needs RSI!
"""

print("1. What we write in YAML:")
print("-" * 40)
print(yaml_config_we_write)

# Example 2: What the Keltner Bands strategy declares it needs
print("\n2. What keltner_bands strategy declares via feature_discovery:")
print("-" * 40)
print("""
@strategy(
    name='keltner_bands',
    feature_discovery=lambda params: [
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, 'upper'),
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, 'middle'),
        FeatureSpec('keltner_channel', {
            'period': params.get('period', 20),
            'multiplier': params.get('multiplier', 2.0)
        }, 'lower')
    ],
    # ... rest of decorator
)
""")

# Example 3: What actually gets used
print("\n3. What topology.py does (line 967):")
print("-" * 40)
print("""
# Discovery finds only Keltner features from the strategy:
feature_specs = {
    'keltner_channel_2.0_20_upper': FeatureSpec(...),
    'keltner_channel_2.0_20_middle': FeatureSpec(...),
    'keltner_channel_2.0_20_lower': FeatureSpec(...)
}

# Creates feature_configs from ONLY discovered features:
feature_configs = {
    'keltner_channel_2.0_20_upper': {'type': 'keltner_channel', 'period': 20, 'multiplier': 2.0, 'component': 'upper'},
    'keltner_channel_2.0_20_middle': {'type': 'keltner_channel', 'period': 20, 'multiplier': 2.0, 'component': 'middle'}, 
    'keltner_channel_2.0_20_lower': {'type': 'keltner_channel', 'period': 20, 'multiplier': 2.0, 'component': 'lower'}
}

# OVERWRITES our manual config!
context['config']["feature_configs"] = feature_configs  # <-- This replaces our RSI config!
""")

print("\n4. Result:")
print("-" * 40)
print("""
- FeatureHub only computes: keltner_channel features
- RSI is NOT computed (even though we defined it)
- Filter tries to use rsi(14) but gets default value of 50
- Filter "signal != 0 and rsi(14) < 30" always returns False
- No signals pass the filter!
""")

print("\n5. What SHOULD happen:")
print("-" * 40)
print("""
# Merge manual features with discovered features:
existing_features = context['config'].get('feature_configs', {})
discovered_features = {...}  # from feature discovery

# Merge them (manual features take precedence)
merged_features = discovered_features.copy()
merged_features.update(existing_features)  # Our manual RSI config preserved!

context['config']["feature_configs"] = merged_features
""")

print("\n6. Alternative workarounds:")
print("-" * 40)
print("""
a) Create a wrapper strategy that declares ALL needed features:

@strategy(
    name='keltner_bands_with_rsi',
    feature_discovery=lambda params: [
        # Keltner features
        FeatureSpec('keltner_channel', {...}),
        # ADD RSI for filter!
        FeatureSpec('rsi', {'period': 14})
    ]
)

b) Use a different mode that doesn't override features

c) Modify the topology.py to merge instead of overwrite
""")