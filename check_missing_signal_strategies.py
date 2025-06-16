#!/usr/bin/env python3
"""Check which strategies from config are not generating signals."""

import yaml
import os

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get expected strategy types from config
expected_types = set()
for strategy in config.get('strategies', []):
    strategy_type = strategy.get('type')
    if strategy_type:
        expected_types.add(strategy_type)

print(f"Expected strategy types from config: {len(expected_types)}")

# Get actual signal directories
workspace = 'workspaces/indicator_grid_v3_42db06b5'
signals_dir = os.path.join(workspace, 'traces/SPY_1m/signals')
if os.path.exists(signals_dir):
    actual_dirs = os.listdir(signals_dir)
    # Map directory names back to strategy types
    actual_types = set()
    for d in actual_dirs:
        # Remove _grid suffix to get base type
        base_type = d.replace('_grid', '')
        actual_types.add(base_type)
    
    print(f"Strategy types with signals: {len(actual_types)}")
    
    # Find missing
    missing = sorted(expected_types - actual_types)
    print(f"\nMissing {len(missing)} strategy types:")
    for m in missing:
        print(f"  - {m}")
        
    # Check if these are actually registered
    import sys
    sys.path.append('.')
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
        except:
            pass
    
    registry = get_component_registry()
    
    print("\nChecking if missing strategies are registered:")
    for m in missing:
        info = registry.get_component(m)
        if info:
            print(f"  ✓ {m} is registered")
        else:
            print(f"  ✗ {m} is NOT registered")