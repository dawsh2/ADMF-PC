#\!/usr/bin/env python3
"""Test strategies directly."""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')

# Import strategies directly
from src.strategy.strategies.indicators.crossovers import dema_crossover

# Create test data
features = {
    'dema_3': 100.0,
    'dema_15': 99.0  # Fast > Slow should give signal 1
}

bar = {
    'timestamp': '2023-01-01T09:30:00',
    'symbol': 'SPY',
    'timeframe': '1m',
    'open': 100.0,
    'high': 100.5,
    'low': 99.5,
    'close': 100.2,
    'volume': 1000000
}

params = {
    'fast_dema_period': 3,
    'slow_dema_period': 15
}

print("Testing dema_crossover directly...")
print(f"Features: {features}")
print(f"Params: {params}")

result = dema_crossover(features, bar, params)
print(f"\nResult: {result}")

if result:
    print(f"Signal value: {result.get('signal_value')}")
    print(f"Full result: {result}")
else:
    print("No signal generated (None returned)")
