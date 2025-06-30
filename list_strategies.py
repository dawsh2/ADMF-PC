#!/usr/bin/env python3
"""
Simple script to list all available strategies.
Use with grep to filter: python list_strategies.py | grep -i mean
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.components.discovery import discover_components_in_module, get_component_registry


def main():
    # Discover strategies from indicator modules
    modules = [
        'src.strategy.strategies.indicators.momentum',
        'src.strategy.strategies.indicators.oscillators', 
        'src.strategy.strategies.indicators.structure',
        'src.strategy.strategies.indicators.trend',
        'src.strategy.strategies.indicators.volatility',
        'src.strategy.strategies.indicators.volume',
        'src.strategy.strategies.indicators.crossovers'
    ]
    
    # Discover all modules silently
    for module in modules:
        try:
            discover_components_in_module(module)
        except:
            pass
    
    # Get registry
    registry = get_component_registry()
    strategies = []
    
    # Get all strategy components
    for name, info in registry._components.items():
        if info.component_type == 'strategy':
            # Extract category from module
            module = info.metadata.get('module', 'unknown')
            category = module.split('.')[-1] if '.' in module else 'unknown'
            strategies.append((category, name))
    
    # Sort and display
    strategies.sort()
    
    current_category = None
    for category, name in strategies:
        if category != current_category:
            current_category = category
            print(f"\n{category.upper()}:")
        print(f"  {name}")


if __name__ == '__main__':
    main()