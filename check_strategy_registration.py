#!/usr/bin/env python3
"""Check strategy registration status."""

import sys
sys.path.append('.')

# Import all strategies to trigger registration
from src.core.components.discovery import get_component_registry
import importlib

# Import all indicator modules
modules = [
    'src.strategy.strategies.indicators.crossovers',
    'src.strategy.strategies.indicators.oscillators', 
    'src.strategy.strategies.indicators.volatility',
    'src.strategy.strategies.indicators.momentum',
    'src.strategy.strategies.indicators.trend',
    'src.strategy.strategies.indicators.volume',
    'src.strategy.strategies.indicators.structure'
]

print("Importing strategy modules...")
for module in modules:
    try:
        importlib.import_module(module)
        print(f"  ✓ {module}")
    except Exception as e:
        print(f"  ✗ {module}: {e}")

# Check registry
registry = get_component_registry()
all_strategies = [name for name, info in registry._components.items() if info.component_type == 'strategy']

print(f"\nTotal registered strategies: {len(all_strategies)}")

# Check our target strategies specifically
target_strategies = ['linear_regression_slope', 'fibonacci_retracement', 'price_action_swing', 'macd_crossover', 'ichimoku_cloud_position']

print(f"\nTarget strategy registration status:")
for strategy in target_strategies:
    registered = strategy in all_strategies
    status = "✓ REGISTERED" if registered else "✗ NOT REGISTERED"
    print(f"  {strategy}: {status}")

# Show top 20 registered strategies
print(f"\nFirst 20 registered strategies:")
for i, strategy in enumerate(sorted(all_strategies)[:20], 1):
    print(f"  {i:2d}. {strategy}")

# Look for any strategies containing our target keywords
print(f"\nStrategies containing target keywords:")
keywords = ['linear', 'fibonacci', 'swing', 'macd', 'ichimoku']
for keyword in keywords:
    matches = [s for s in all_strategies if keyword in s.lower()]
    print(f"  {keyword}: {matches}")