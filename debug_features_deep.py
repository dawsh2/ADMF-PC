#!/usr/bin/env python3
"""Deep debug of feature computation and strategy readiness."""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.coordinator import Coordinator

def main():
    # Create coordinator
    coord = Coordinator()
    
    # Simple test config
    config = {
        'symbols': ['SPY'],
        'timeframes': ['1m'],
        'data_source': 'file',
        'data_dir': './data',
        'start_date': '2023-01-01',
        'end_date': '2023-01-02',
        'max_bars': 100,
        'topology': 'signal_generation',
        'execution': {
            'enable_event_tracing': True,
            'trace_settings': {
                'enable_console_output': True,
                'console_filter': ['BAR', 'SIGNAL'],
                'use_sparse_storage': True
            }
        },
        # Just one simple strategy
        'strategies': [
            {
                'type': 'sma_crossover',
                'name': 'test_sma',
                'params': {
                    'fast_period': 5,
                    'slow_period': 10
                }
            }
        ],
        'metadata': {
            'experiment_id': 'debug_features'
        }
    }
    
    # Run topology
    result = coord.run_topology('signal_generation', config)
    
    # Check results
    if result.get('success'):
        print(f"\n✅ Topology completed")
        
        # Check for signals
        tracer_results = result.get('tracer_results', {})
        components = tracer_results.get('components', {})
        
        print(f"\n📊 Components with signals: {len(components)}")
        for comp_id, comp_data in components.items():
            changes = comp_data.get('signal_changes', 0)
            total_bars = comp_data.get('total_bars', 0)
            print(f"  - {comp_id}: {changes} changes in {total_bars} bars")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")

if __name__ == '__main__':
    main()