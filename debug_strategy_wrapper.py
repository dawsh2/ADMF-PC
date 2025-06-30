#!/usr/bin/env python3
"""Debug the strategy wrapper execution."""

import pandas as pd
import numpy as np

# First, let's see what the wrapped strategy actually does
print("=== Testing Strategy Wrapper ===")

# Create test data
test_features = {
    'bollinger_bands_20_2.0_upper': 522.0,
    'bollinger_bands_20_2.0_middle': 520.0,
    'bollinger_bands_20_2.0_lower': 518.0,
}

test_bar = {
    'close': 517.5,  # Below lower band - should generate buy signal
    'symbol': 'SPY',
    'timeframe': '5m',
    'timestamp': '2024-01-01T10:00:00Z',
    'open': 518.0,
    'high': 518.5,
    'low': 517.0,
    'volume': 1000000
}

test_params = {
    'period': 20,
    'std_dev': 2.0,
    '_strategy_type': 'strategy_0'
}

# Import and call bollinger_bands directly
from src.strategy.strategies.indicators.volatility import bollinger_bands

print(f"\nTesting direct bollinger_bands call...")
result = bollinger_bands(test_features, test_bar, {'period': 20, 'std_dev': 2.0})
print(f"Direct call result: {result}")

# Now test through the compiler
from src.core.coordinator.compiler import StrategyCompiler

# First discover strategies like main.py does
import pkgutil
import importlib
strategies = {}

# Discover strategies in the indicators module
indicators_path = 'src.strategy.strategies.indicators'
indicators_module = importlib.import_module(indicators_path)
for importer, modname, ispkg in pkgutil.iter_modules(indicators_module.__path__):
    if not ispkg:  # Skip sub-packages
        try:
            full_module_name = f"{indicators_path}.{modname}"
            module = importlib.import_module(full_module_name)
            # The module should have strategies registered via @strategy decorator
        except Exception as e:
            print(f"Could not import {modname}: {e}")

# Let's also test with a case where price touches the band
print("\n=== Testing with price at upper band ===")
test_bar2 = test_bar.copy()
test_bar2['close'] = 522.5  # Above upper band
result2 = bollinger_bands(test_features, test_bar2, {'period': 20, 'std_dev': 2.0})
print(f"Result with price above upper band: {result2}")

print("\n=== Testing with price near middle ===")
test_bar3 = test_bar.copy()
test_bar3['close'] = 520.0  # At middle band
result3 = bollinger_bands(test_features, test_bar3, {'period': 20, 'std_dev': 2.0, 'exit_threshold': 0.001})
print(f"Result with price at middle band: {result3}")