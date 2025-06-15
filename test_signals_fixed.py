#!/usr/bin/env python3
"""Test if strategies generate signals with fixed feature names."""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Reduce noise from some loggers
logging.getLogger('src.data').setLevel(logging.WARNING)
logging.getLogger('src.core.containers').setLevel(logging.WARNING)
logging.getLogger('src.core.events').setLevel(logging.WARNING)

from src.orchestration.orchestrator import Orchestrator

# Test config with multiple strategies
config = {
    'symbols': ['SPY'],
    'timeframes': ['1m'],
    'data_source': 'file',
    'data_path': 'data/1min_SPY_2022-01-03_2024-12-06.parquet',
    'start_date': '2022-01-03',
    'end_date': '2022-01-10',
    'strategies': [
        {
            'type': 'donchian_breakout',
            'name': 'donchian_test',
            'params': {'period': 20}
        },
        {
            'type': 'keltner_breakout',
            'name': 'keltner_test',
            'params': {'period': 20, 'multiplier': 2.0}
        },
        {
            'type': 'bollinger_breakout',
            'name': 'bollinger_test',
            'params': {'period': 20, 'std_dev': 2.0}
        }
    ],
    'execution': {
        'enable_event_tracing': True,
        'trace_settings': {
            'use_sparse_storage': True,
            'streaming_mode': True
        }
    }
}

print("Running signal generation with fixed strategies...")
orchestrator = Orchestrator()

try:
    # Run signal generation
    orchestrator.run(
        mode='signal_generation',
        config=config,
        max_bars=100
    )
    
    # Find the workspace
    workspaces = list(Path('./workspaces').glob('signal_generation_*'))
    if workspaces:
        latest_workspace = sorted(workspaces)[-1]
        print(f"\nWorkspace created: {latest_workspace}")
        
        # Check for signal files
        trace_files = list(latest_workspace.glob('traces/**/*.parquet'))
        print(f"\nFound {len(trace_files)} trace files:")
        for f in trace_files[:10]:  # Show first 10
            print(f"  {f.relative_to(latest_workspace)}")
            
        # Load and check one file
        if trace_files:
            import pandas as pd
            df = pd.read_parquet(trace_files[0])
            print(f"\nFirst trace file has {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            if len(df) > 0:
                print(f"\nFirst few signals:")
                print(df.head())
                
                # Count non-zero signals
                non_zero = df[df['signal_value'] != 0]
                print(f"\nNon-zero signals: {len(non_zero)} out of {len(df)} ({len(non_zero)/len(df)*100:.1f}%)")
    else:
        print("\nNo workspace created!")
        
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()