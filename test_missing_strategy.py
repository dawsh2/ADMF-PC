#!/usr/bin/env python3
"""Test a missing strategy to see what's happening."""

import sys
import os
from pathlib import Path
import logging

# Setup logging to see errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.coordinator import Coordinator

def main():
    # Create coordinator
    coord = Coordinator()
    
    # Test config with one of the missing strategies
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
                'console_filter': ['SIGNAL'],
                'use_sparse_storage': True
            }
        },
        # Test one missing strategy
        'strategies': [
            {
                'type': 'donchian_breakout',
                'name': 'donchian_breakout_grid',
                'params': {
                    'period': 20
                }
            }
        ],
        'metadata': {
            'experiment_id': 'test_missing_strategy'
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
            print(f"  - {comp_id}: {comp_data.get('signal_changes', 0)} signal changes")
            
        # Check workspace
        workspace_path = tracer_results.get('workspace_path')
        if workspace_path:
            workspace_dir = Path(workspace_path)
            signal_files = list(workspace_dir.glob('traces/**/signals/**/*.parquet'))
            print(f"\n📁 Signal files created: {len(signal_files)}")
            for f in signal_files:
                print(f"  - {f.relative_to(workspace_dir)}")
    else:
        print(f"❌ Topology failed: {result.get('error', 'Unknown error')}")

if __name__ == '__main__':
    main()