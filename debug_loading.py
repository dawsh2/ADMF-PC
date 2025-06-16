#!/usr/bin/env python3
"""Debug strategy loading issue."""

import yaml
import sys
sys.path.append('.')

from src.core.components.discovery import get_component_registry
import importlib

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Import strategy modules to register them
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
    except:
        pass

registry = get_component_registry()

# Check a few specific strategies
test_strategies = ['parabolic_sar', 'supertrend', 'adx_trend_strength']

print("Checking strategy registration:")
for strat in test_strategies:
    info = registry.get_component(strat)
    print(f"  {strat}: {'✓ Found' if info else '✗ Not found'}")

print("\nChecking config structure:")
# Look at first few strategies
for i, strategy in enumerate(config.get('strategies', [])[:5]):
    print(f"\nStrategy {i+1}:")
    print(f"  type: {strategy.get('type')}")
    print(f"  name: {strategy.get('name')}")
    print(f"  params: {strategy.get('params')}")
    
# Check how parabolic_sar is configured
print("\nLooking for parabolic_sar in config:")
for strategy in config.get('strategies', []):
    if strategy.get('type') == 'parabolic_sar':
        print(f"  Found: type={strategy.get('type')}, name={strategy.get('name')}")
        break

# Simulate what topology builder does
print("\nSimulating topology builder logic:")
from src.core.coordinator.topology import TopologyBuilder

builder = TopologyBuilder()

# Expand strategies like topology builder does
expanded = builder._expand_strategy_parameters(config.get('strategies', []))
print(f"\nTotal expanded strategies: {len(expanded)}")

# Check a few parabolic_sar variations
print("\nParabolic SAR variations:")
psar_count = 0
for strategy in expanded:
    if strategy.get('type') == 'parabolic_sar':
        psar_count += 1
        if psar_count <= 3:
            print(f"  name: {strategy.get('name')}, params: {strategy.get('params')}")
            
print(f"Total parabolic_sar variations: {psar_count}")