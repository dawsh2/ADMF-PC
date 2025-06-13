#!/usr/bin/env python3
"""
Test signal tracing with MultiStrategyTracer
"""

import logging
from src.core.coordinator.coordinator import Coordinator

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Filter out some noisy loggers
logging.getLogger('src.core.containers').setLevel(logging.INFO)
logging.getLogger('src.core.events.bus').setLevel(logging.INFO)

# Test configuration with signal generation
config = {
    'data': {
        'symbols': ['SPY'],
        'source': 'csv'
    },
    'symbols': ['SPY'],
    'timeframes': ['1m'],
    'max_bars': 30,  # Small for testing
    
    'strategies': [
        {
            'name': 'ma_5_20',
            'type': 'ma_crossover',
            'params': {
                'fast_period': 5,
                'slow_period': 20
            }
        }
    ],
    
    'classifiers': [],
    
    'metadata': {
        'workflow_id': 'test_signal_trace'
    },
    
    'execution': {
        'enable_event_tracing': True,
        'trace_settings': {
            'use_sparse_storage': True,
            'storage': {
                'base_dir': './workspaces'
            }
        }
    }
}

# Create coordinator and run
coordinator = Coordinator()

print("Running signal generation topology...")
result = coordinator.run_topology('signal_generation', config)

print("\nResult:", result.get('success'))
if result.get('errors'):
    print("Errors:", result['errors'])

# Check for tracer results
if 'tracer_results' in result:
    tracer_results = result['tracer_results']
    print("\nTracer Results:")
    print(f"- Total bars: {tracer_results.get('total_bars')}")
    print(f"- Total signals: {tracer_results.get('total_signals')}")
    print(f"- Stored changes: {tracer_results.get('stored_changes')}")
    print(f"- Components: {len(tracer_results.get('components', {}))}")
    
    for comp_id, comp_info in tracer_results.get('components', {}).items():
        print(f"\n  {comp_id}:")
        print(f"    - Signal changes: {comp_info.get('signal_changes')}")
        print(f"    - File: {comp_info.get('signal_file_path')}")
else:
    print("\nNo tracer results found in output")

print("\nChecking workspace directory...")
import os
workspace_path = './workspaces/test_signal_trace'
if os.path.exists(workspace_path):
    for root, dirs, files in os.walk(workspace_path):
        level = root.replace(workspace_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")