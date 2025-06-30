#!/usr/bin/env python3
"""Debug ComponentState execution."""

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Patch ComponentState to add debugging
import src.strategy.state as state_module

# Store original method
original_execute = state_module.ComponentState._execute_components_individually

def debug_execute(self, features, bar_dict):
    """Wrapped execute method with debugging."""
    print(f"\n=== _execute_components_individually called ===")
    print(f"  Features available: {len(features)} features")
    print(f"  Components: {len(self._components)} total")
    print(f"  Bar: {bar_dict.get('close', 'N/A')}")
    
    # Show some components
    for comp_id, comp_info in list(self._components.items())[:3]:
        print(f"  Component {comp_id}: {comp_info.get('component_type')}")
    
    # Call original
    result = original_execute(self, features, bar_dict)
    
    print(f"  Results: {len(result)} outputs generated")
    for output in result[:3]:
        if output and output.get('signal_value', 0) != 0:
            print(f"    -> SIGNAL: {output}")
    
    return result

# Monkey patch
state_module.ComponentState._execute_components_individually = debug_execute

# Now run a simple test
import sys
sys.argv = ['main.py', '--config', 'config/bollinger/test_simple.yaml', '--signal-generation', '--dataset', 'train', '--bars', '60']

from main import main
main()