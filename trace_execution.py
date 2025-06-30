#!/usr/bin/env python3
"""Trace the exact execution flow."""

import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Add some key debug points
from src.strategy import state

# Check when components are deemed ready
original_is_ready = state.ComponentState._is_component_ready

calls_to_ready = 0
ready_results = {}

def trace_ready(self, component_id, component_info, symbol, features):
    global calls_to_ready
    calls_to_ready += 1
    
    result = original_is_ready(self, component_id, component_info, symbol, features)
    
    # Track when components become ready
    if component_id not in ready_results:
        ready_results[component_id] = []
    ready_results[component_id].append((self._bar_count.get(symbol, 0), result))
    
    # Log first time ready
    if result and len([r for _, r in ready_results[component_id] if r]) == 1:
        print(f"\n✓ Component {component_id} became READY at bar {self._bar_count.get(symbol, 0)}")
    
    return result

state.ComponentState._is_component_ready = trace_ready

# Check actual execution
executed_components = []

original_exec_individual = state.ComponentState._execute_components_individually

def trace_exec(self, symbol, features, bar_dict, bar_timestamp):
    bar_num = self._bar_count.get(symbol, 0)
    
    # Log execution attempt
    if bar_num in [1, 25, 50] or bar_num % 20 == 0:
        print(f"\nBar {bar_num}: _execute_components_individually called")
        print(f"  Components: {len(self._components)}")
        ready_count = sum(1 for comp_id, comp_info in self._components.items() 
                         if self._is_component_ready(comp_id, comp_info, symbol, features))
        print(f"  Ready: {ready_count}")
    
    result = original_exec_individual(self, symbol, features, bar_dict, bar_timestamp)
    
    if result:
        executed_components.append((bar_num, len(result)))
        print(f"  → Executed! Returned {len(result)} outputs")
        for output in result[:3]:
            if output and output.get('signal_value', 0) != 0:
                print(f"     SIGNAL: {output.get('signal_value')}")
    
    return result

state.ComponentState._execute_components_individually = trace_exec

# Run
sys.argv = ['main.py', '--config', 'config/bollinger/test_simple.yaml', '--signal-generation', '--dataset', 'train', '--bars', '60']

print("=== Tracing execution flow ===\n")

from main import main
main()

print(f"\n\n=== SUMMARY ===")
print(f"Ready checks: {calls_to_ready}")
print(f"Components that became ready:")
for comp_id, results in ready_results.items():
    ready_bars = [bar for bar, ready in results if ready]
    if ready_bars:
        print(f"  {comp_id}: first ready at bar {ready_bars[0]}")
print(f"\nExecutions that returned outputs: {len(executed_components)}")
if executed_components:
    print(f"First few: {executed_components[:5]}")