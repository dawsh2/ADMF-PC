#!/usr/bin/env python3
"""Debug why features aren't being configured in FeatureHub."""

import logging
import sys
from datetime import datetime
from src.core.coordinator.topology import TopologyBuilder

# Set up logging to see all debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Focus on specific loggers
for logger_name in ['src.core.coordinator.topology', 'src.core.containers.factory', 'src.strategy.components.features.hub']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

# Simple test config with one strategy
test_config = {
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
            'params': {
                'period': 20
            }
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

print("Building topology with test config...")
builder = TopologyBuilder()

try:
    # Build topology definition
    topology_def = {
        'mode': 'signal_generation',
        'config': test_config,
        'tracing': {'enabled': True}
    }
    
    topology = builder.build_topology(topology_def)
    
    print("\n=== TOPOLOGY BUILT ===")
    print(f"Containers: {list(topology['containers'].keys())}")
    
    # Find FeatureHub container (it has a hashed name)
    feature_hub = None
    for name, container in topology['containers'].items():
        if 'feature_hub' in name:
            feature_hub = container
            print(f"\nFound FeatureHub container: {name}")
            break
    
    if feature_hub:
        print(f"\nFeatureHub container found!")
        print(f"Config: {feature_hub.config}")
        
        # Check if it has the feature_hub component
        if hasattr(feature_hub, '_components'):
            for comp_name, comp in feature_hub._components.items():
                print(f"\nComponent {comp_name}: {type(comp)}")
                if hasattr(comp, '_feature_hub'):
                    fh = comp._feature_hub
                    print(f"  FeatureHub instance: {fh}")
                    print(f"  Configured features: {fh.feature_configs}")
                    print(f"  Symbols: {fh.symbols}")
    else:
        print("\nNo FeatureHub container found!")
        
except Exception as e:
    print(f"\nError building topology: {e}")
    import traceback
    traceback.print_exc()