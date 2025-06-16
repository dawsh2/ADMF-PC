#!/usr/bin/env python3
"""Test strategy registration."""

import sys
sys.path.append('.')

from src.core.components.discovery import get_component_registry
import importlib

print("Testing strategy registration...")

# Import volume strategies module
try:
    print("Importing volume strategies module...")
    volume_module = importlib.import_module('src.strategy.strategies.indicators.volume')
    print("✓ Volume module imported successfully")
except Exception as e:
    print(f"✗ Error importing volume module: {e}")
    exit(1)

# Get registry
registry = get_component_registry()

# Check volume strategies
volume_strategies = ['obv_trend', 'mfi_bands', 'vwap_deviation', 'chaikin_money_flow', 'accumulation_distribution']

print(f"\nChecking {len(volume_strategies)} volume strategies in registry:")
for strategy_name in volume_strategies:
    info = registry.get_component(strategy_name)
    if info:
        print(f"  ✓ {strategy_name}: Found")
        print(f"    feature_config: {info.metadata.get('feature_config')}")
    else:
        print(f"  ✗ {strategy_name}: NOT FOUND")

# List all registered strategies
all_strategies = registry.get_components_by_type('strategy')
print(f"\nTotal strategies in registry: {len(all_strategies)}")
print("Registered strategies:")
for strategy in all_strategies:
    print(f"  - {strategy.name}")

# Check if volume module has the strategies defined
print(f"\nVolume module attributes:")
for attr in dir(volume_module):
    if not attr.startswith('_'):
        print(f"  - {attr}")