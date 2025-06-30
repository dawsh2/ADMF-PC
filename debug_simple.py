#!/usr/bin/env python3
"""Simple debug to find signal generation issue."""

import pandas as pd

# Patch ComponentState before imports
import sys
import src.strategy.state as state_module

# Count signals
signal_count = 0

original_publish = state_module.ComponentState._publish_output

def count_signals(self, output, symbol):
    global signal_count
    if output and output.get('signal_value', 0) != 0:
        signal_count += 1
        print(f"\n🚨 SIGNAL #{signal_count}: {output['signal_value']} for {symbol} at bar {self._bar_count.get(symbol, 0)}")
    return original_publish(self, output, symbol)

state_module.ComponentState._publish_output = count_signals

# Also check component execution
exec_count = 0
original_exec = state_module.ComponentState._execute_components_individually

def count_exec(self, symbol, features, bar_dict, bar_timestamp):
    global exec_count
    exec_count += 1
    
    if exec_count <= 3 or exec_count % 10 == 0:
        print(f"\nExecuting components (call #{exec_count}):")
        print(f"  Symbol: {symbol}")
        print(f"  Bar count: {self._bar_count.get(symbol, 0)}")
        print(f"  Components: {len(self._components)}")
        print(f"  Features: {len(features)}")
        
        # Check if required features are present
        for comp_id, comp_info in list(self._components.items())[:1]:
            metadata = comp_info.get('metadata', {})
            feature_specs = metadata.get('feature_specs', [])
            if feature_specs:
                print(f"  Required features for {comp_id}:")
                for spec in feature_specs[:3]:
                    feature_name = f"{spec.feature_type}_{spec.params['period']}_{spec.params['std_dev']}_{spec.output_component}"
                    print(f"    - {feature_name}: {'✓' if feature_name in features else '✗'}")
    
    result = original_exec(self, symbol, features, bar_dict, bar_timestamp)
    
    if result:
        print(f"  → Returned {len(result)} outputs")
    
    return result

state_module.ComponentState._execute_components_individually = count_exec

# Run with simple config
sys.argv = ['main.py', '--config', 'config/bollinger/test_simple.yaml', '--signal-generation', '--dataset', 'train', '--bars', '100']

print("=== Running with signal counting ===\n")

from main import main
main()

print(f"\n\n=== FINAL SUMMARY ===")
print(f"Total execution calls: {exec_count}")
print(f"Total signals generated: {signal_count}")