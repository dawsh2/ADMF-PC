#!/usr/bin/env python3
"""Debug why strategies aren't executing."""

import sys
sys.path.append('.')

# First, let's check what's in the stateless components that get injected
from src.core.components.discovery import get_component_registry
import importlib

# Import strategy modules
modules = [
    'src.strategy.strategies.indicators.trend',
    'src.strategy.strategies.indicators.momentum', 
    'src.strategy.strategies.indicators.oscillators',
    'src.strategy.strategies.indicators.volatility',
    'src.strategy.strategies.indicators.volume',
    'src.strategy.strategies.indicators.structure',
    'src.strategy.strategies.indicators.crossovers',
]

for module in modules:
    try:
        importlib.import_module(module)
    except ImportError as e:
        print(f"Could not import {module}: {e}")

# Get registry
registry = get_component_registry()

# Check what strategies are available
print("Checking strategies in registry:")
for name in ['parabolic_sar', 'supertrend', 'adx_trend_strength', 'aroon_crossover']:
    info = registry.get_component(name)
    if info:
        print(f"  ✓ {name} - factory: {info.factory.__name__ if hasattr(info.factory, '__name__') else info.factory}")
    else:
        print(f"  ✗ {name} - NOT FOUND")

# Let's also check the config expansion
print("\nChecking how config expands strategies:")
import yaml

with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Count strategies
strategy_count = 0
for strategy in config.get('strategies', []):
    params = strategy.get('params', {})
    # Calculate combinations
    combinations = 1
    for param_name, param_values in params.items():
        if isinstance(param_values, list):
            combinations *= len(param_values)
    strategy_count += combinations
    if strategy['type'] in ['parabolic_sar', 'supertrend']:
        print(f"\n{strategy['type']}:")
        print(f"  Base name: {strategy.get('name')}")
        print(f"  Params: {params}")
        print(f"  Combinations: {combinations}")

print(f"\nTotal strategy variations: {strategy_count}")