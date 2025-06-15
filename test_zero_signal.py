#\!/usr/bin/env python3
"""Test what happens with zero signals."""

# If dema_3 == dema_15, the signal should be 0
features = {
    'dema_3': 100.0,
    'dema_15': 100.0  # Equal values should give signal 0
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

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')
from src.strategy.strategies.indicators.crossovers import dema_crossover

result = dema_crossover(features, bar, params)
print(f"Result with equal DEMAs: {result}")
print(f"Signal value: {result['signal_value'] if result else 'None'}")
