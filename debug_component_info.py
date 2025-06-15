#!/usr/bin/env python3
"""Debug ComponentInfo structure."""

from src.core.components.discovery import get_component_registry
import importlib

# Import a strategy module
importlib.import_module('src.strategy.strategies.indicators.volatility')

registry = get_component_registry()
strategy_info = registry.get_component('bollinger_breakout')

if strategy_info:
    print(f"ComponentInfo attributes:")
    for attr in dir(strategy_info):
        if not attr.startswith('_'):
            print(f"  {attr}: {getattr(strategy_info, attr, 'N/A')}")
    
    # Try to get the function
    print(f"\nTrying to get the actual function...")
    print(f"Type: {type(strategy_info)}")
    
    # Check if it's a named tuple or similar
    if hasattr(strategy_info, '_asdict'):
        print(f"As dict: {strategy_info._asdict()}")
    
    # Try different attributes
    for attr in ['func', 'function', 'callable', 'component', 'implementation']:
        if hasattr(strategy_info, attr):
            print(f"Found {attr}: {getattr(strategy_info, attr)}")