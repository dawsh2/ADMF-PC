#!/usr/bin/env python3
"""Test workspace creation with correct structure."""

import sys
import os
from pathlib import Path
import logging

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
        'max_bars': 50,  # Need more bars for SMA warmup
        'topology': 'signal_generation',
        'execution': {
            'enable_event_tracing': True,
            'trace_settings': {
                'enable_console_output': True,
                'console_filter': ['SIGNAL'],
                'use_sparse_storage': True  # Enable MultiStrategyTracer
            }
        },
        'strategies': [
            {
                'type': 'sma_crossover',
                'name': 'test_sma',
                'params': {
                    'fast_period': 3,
                    'slow_period': 5
                }
            }
        ],
        'metadata': {
            'experiment_id': 'test_workspace_structure'
        }
    }
    
    # Run topology
    result = coord.run_topology('signal_generation', config)
    
    # Check results
    if result.get('success'):
        print(f"✅ Topology completed successfully")
        
        # Check for analytics workspace
        if 'analytics_workspace' in result:
            print(f"📊 Analytics workspace: {result['analytics_workspace']}")
        
        # Check for tracer results at top level and in phase results
        if 'tracer_results' in result:
            tracer_results = result['tracer_results']
            print(f"📈 Tracer results found at top level: {list(tracer_results.keys())}")
            workspace_path = tracer_results.get('workspace_path')
            if workspace_path:
                print(f"📁 Workspace path: {workspace_path}")
                
                # List workspace contents
                workspace_dir = Path(workspace_path)
                if workspace_dir.exists():
                    print(f"\n📂 Workspace contents:")
                    for item in sorted(workspace_dir.rglob('*')):
                        if item.is_file():
                            rel_path = item.relative_to(workspace_dir)
                            size = item.stat().st_size
                            print(f"  📄 {rel_path} ({size} bytes)")
        else:
            # Check phase results for tracer results
            phase_results = result.get('phase_results', {})
            print(f"📋 Phase results keys: {list(phase_results.keys())}")
            
            if 'tracer_results' in phase_results:
                tracer_results = phase_results['tracer_results']
                print(f"📈 Tracer results found in phase results: {list(tracer_results.keys())}")
                workspace_path = tracer_results.get('workspace_path')
                if workspace_path:
                    print(f"📁 Workspace path: {workspace_path}")
                    
                    # List workspace contents
                    workspace_dir = Path(workspace_path)
                    if workspace_dir.exists():
                        print(f"\n📂 Workspace contents:")
                        for item in sorted(workspace_dir.rglob('*')):
                            if item.is_file():
                                rel_path = item.relative_to(workspace_dir)
                                size = item.stat().st_size
                                print(f"  📄 {rel_path} ({size} bytes)")
            else:
                print("⚠️ No tracer results found")
            
        # Check metrics
        if 'execution_result' in result:
            exec_result = result['execution_result']
            print(f"\n📊 Execution metrics:")
            print(f"  - Bars processed: {exec_result.get('bars_processed', 0)}")
            print(f"  - Signals generated: {exec_result.get('signals_generated', 0)}")
    else:
        print(f"❌ Topology failed: {result.get('error', 'Unknown error')}")

if __name__ == '__main__':
    main()