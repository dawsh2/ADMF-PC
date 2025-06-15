#!/usr/bin/env python3
"""Test what feature names the incremental system produces."""

from src.strategy.components.features.incremental import IncrementalFeatureHub

# Create hub and configure features
hub = IncrementalFeatureHub()

# Configure some test features
feature_configs = {
    'bollinger_bands_20_2.0': {'type': 'bollinger_bands', 'period': 20, 'std_dev': 2.0},
    'stochastic_14_3': {'type': 'stochastic', 'k_period': 14, 'd_period': 3},
    'macd_12_26_9': {'type': 'macd', 'fast': 12, 'slow': 26, 'signal': 9},
    'donchian_channel_20': {'type': 'donchian_channel', 'period': 20},
    'sma_20': {'type': 'sma', 'period': 20}
}

hub.configure_features(feature_configs)

print("=== TESTING INCREMENTAL FEATURE NAMES ===\n")

# Feed some test bars
symbol = 'SPY'
for i in range(30):
    bar = {
        'open': 100 + i * 0.1,
        'high': 101 + i * 0.1,
        'low': 99 + i * 0.1,
        'close': 100.5 + i * 0.1,
        'volume': 1000000
    }
    
    features = hub.update_bar(symbol, bar)
    
    # Show features after certain bars
    if i == 19 or i == 29:  # After 20 and 30 bars
        print(f"\nAfter {i+1} bars, features available:")
        for name, value in sorted(features.items()):
            print(f"  {name}: {value:.4f} (type: {type(value).__name__})")

print("\n\n=== FEATURE NAME PATTERNS ===")
print("\nExpected by strategies:")
print("  bollinger_breakout: bollinger_bands_<period>_<std>_upper/middle/lower")
print("  stochastic_crossover: stochastic_<k>_<d>_k and stochastic_<k>_<d>_d")
print("  macd_crossover: macd_<fast>_<slow>_<signal>_macd/signal/histogram")

print("\nActual from incremental system:")
for name in sorted(features.keys()):
    print(f"  {name}")