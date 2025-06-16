#!/usr/bin/env python3
"""Debug why only some strategies are generating signals."""

import sys
import os
from pathlib import Path
import logging

# Setup logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.coordinator import Coordinator

def main():
    # Create coordinator
    coord = Coordinator()
    
    # Test config with a subset of missing strategies
    config = {
        'symbols': ['SPY'],
        'timeframes': ['1m'],
        'data_source': 'file',
        'data_dir': './data',
        'start_date': '2023-01-01',
        'end_date': '2023-01-02',
        'max_bars': 200,  # More bars for proper warmup
        'topology': 'signal_generation',
        'execution': {
            'enable_event_tracing': True,
            'trace_settings': {
                'enable_console_output': False,  # Too noisy
                'use_sparse_storage': True
            }
        },
        # Test multiple missing strategies
        'strategies': [
            {
                'type': 'donchian_breakout',
                'name': 'donchian_breakout_grid',
                'params': {'period': 20}
            },
            {
                'type': 'parabolic_sar',
                'name': 'parabolic_sar_grid',
                'params': {'af_start': 0.02, 'af_max': 0.2}
            },
            {
                'type': 'vortex_crossover',
                'name': 'vortex_crossover_grid',
                'params': {'vortex_period': 14}
            }
        ],
        'metadata': {
            'experiment_id': 'test_missing_strategies'
        }
    }
    
    # Run topology
    result = coord.run_topology('signal_generation', config)
    
    # Check results
    if result.get('success'):
        print(f"\n✅ Topology completed successfully")
        
        # Check for signals
        tracer_results = result.get('tracer_results', {})
        components = tracer_results.get('components', {})
        
        print(f"\n📊 Components with signals: {len(components)}")
        for comp_id, comp_data in components.items():
            changes = comp_data.get('signal_changes', 0)
            total_bars = comp_data.get('total_bars', 0)
            if changes > 0:
                print(f"  ✓ {comp_id}: {changes} signal changes in {total_bars} bars")
            else:
                print(f"  ✗ {comp_id}: NO SIGNALS in {total_bars} bars")
                
        # Check if strategies were loaded
        print(f"\n🔍 Checking strategy loading...")
        
    else:
        print(f"❌ Topology failed: {result.get('error', 'Unknown error')}")

if __name__ == '__main__':
    main()