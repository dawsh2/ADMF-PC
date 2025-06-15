#!/usr/bin/env python3
"""Test that all feature mapping fixes work correctly."""

import logging
import sys
from datetime import datetime
from src.core.coordinator.topology import TopologyBuilder

# Set up logging  
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('src.core.coordinator.topology').setLevel(logging.DEBUG)

# Test config with the fixed strategies
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
        },
        {
            'type': 'ultimate_oscillator',
            'name': 'ultimate_test',
            'params': {'period1': 7, 'period2': 14, 'period3': 28}
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

print("Testing all fixed strategies...")
builder = TopologyBuilder()

try:
    topology_def = {
        'mode': 'signal_generation',
        'config': test_config,
        'tracing': {'enabled': True}
    }
    
    topology = builder.build_topology(topology_def)
    
    print(f"\n=== TOPOLOGY BUILT ===")
    print(f"Containers: {list(topology['containers'].keys())}")
    
    # Find FeatureHub container
    feature_hub = None
    for name, container in topology['containers'].items():
        if 'feature_hub' in name:
            feature_hub = container
            print(f"\nFound FeatureHub container: {name}")
            break
    
    if feature_hub and hasattr(feature_hub, '_components'):
        for comp_name, comp in feature_hub._components.items():
            if hasattr(comp, '_feature_hub'):
                fh = comp._feature_hub
                print(f"\nConfigured features ({len(fh.feature_configs)}):")
                for name, config in sorted(fh.feature_configs.items()):
                    print(f"  {name}: {config}")
                
                # Check if the expected features are configured
                expected_features = [
                    'donchian_20',     # From donchian_breakout  
                    'keltner_20_2.0',  # From keltner_breakout
                    'bollinger_20_2.0', # From bollinger_breakout
                    'ultimate_7_14_28'  # From ultimate_oscillator
                ]
                
                print(f"\n=== FEATURE VALIDATION ===")
                all_good = True
                for expected in expected_features:
                    if expected in fh.feature_configs:
                        print(f"✅ {expected}: {fh.feature_configs[expected]}")
                    else:
                        print(f"❌ {expected}: MISSING")
                        all_good = False
                
                if all_good:
                    print(f"\n🎉 ALL FEATURES CORRECTLY CONFIGURED!")
                else:
                    print(f"\n💥 Some features missing!")
                        
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()