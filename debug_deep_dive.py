#!/usr/bin/env python3
"""Deep dive into signal generation issue."""

import logging
import sys

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Focus on key components
for logger_name in ['src.strategy.state', 'src.core.events', 'src.strategy.components.features.hub']:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

# Patch ComponentState to add detailed debugging
import src.strategy.state as state_module

# Store original methods
original_execute = state_module.ComponentState._execute_components_individually
original_check_ready = state_module.ComponentState._is_component_ready

def debug_execute(self, symbol, features, bar_dict, bar_timestamp):
    """Wrapped execute method with debugging."""
    print(f"\n=== EXECUTE COMPONENTS ===")
    print(f"  Symbol: {symbol}")
    print(f"  Bar count: {self._bar_count.get(symbol, 0)}")
    print(f"  Components: {len(self._components)}")
    print(f"  Features available: {len(features)}")
    
    # Show component details
    for comp_id, comp_info in list(self._components.items())[:3]:
        print(f"\n  Component {comp_id}:")
        print(f"    Type: {comp_info.get('component_type')}")
        print(f"    Ready: {self._is_component_ready(comp_id, comp_info, symbol, features)}")
    
    # Call original
    result = original_execute(self, symbol, features, bar_dict, bar_timestamp)
    
    print(f"\n  Execution results: {len(result)} outputs")
    for i, output in enumerate(result[:5]):
        if output:
            print(f"    Output {i}: signal_value={output.get('signal_value', 'N/A')}")
    
    return result

def debug_check_ready(self, component_id, component_info, symbol, features):
    """Debug component readiness checks."""
    result = original_check_ready(self, component_id, component_info, symbol, features)
    
    if not result and self._bar_count.get(symbol, 0) > 30:
        # After 30 bars, log why component isn't ready
        print(f"\n  Component {component_id} NOT READY after {self._bar_count.get(symbol, 0)} bars")
        print(f"    Metadata: {component_info.get('metadata', {})}")
    
    return result

# Apply patches
state_module.ComponentState._execute_components_individually = debug_execute
state_module.ComponentState._is_component_ready = debug_check_ready

# Run with simple config
sys.argv = ['main.py', '--config', 'config/bollinger/test_simple.yaml', '--signal-generation', '--dataset', 'train', '--bars', '60']

print("=== Starting execution with detailed debugging ===\n")

from main import main
main()