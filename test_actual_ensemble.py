#!/usr/bin/env python3
"""Test the actual two_layer_ensemble strategy with simplified config."""

import sys
sys.path.append('/Users/daws/ADMF-PC')

from pathlib import Path
import yaml

# Create a test config with just one ensemble instance
config = {
    'name': 'test_actual_ensemble',
    'description': 'Test actual two-layer ensemble',
    'symbols': ['SPY'],
    'timeframes': ['1m'],
    'data_source': 'file',
    'data_dir': './data',
    'topology': 'signal_generation',
    
    'classifiers': [{
        'type': 'market_regime_classifier',
        'name': 'market_regime_detector',
        'params': {
            'trend_threshold': 0.006,
            'vol_threshold': 0.8,
            'sma_short': 12,
            'sma_long': 20,
            'atr_period': 14,
            'rsi_period': 14
        }
    }],
    
    'strategies': [{
        'type': 'two_layer_ensemble',  # Use actual, not debug
        'name': 'test_ensemble',
        'params': {
            'classifier_name': 'market_regime_classifier',
            'baseline_allocation': 0.25,
            'baseline_aggregation': 'equal_weight',
            'booster_aggregation': 'equal_weight',
            'min_baseline_agreement': 0.2,
            'min_booster_agreement': 0.2,
            
            # Pass classifier params
            'trend_threshold': 0.006,
            'vol_threshold': 0.8,
            'sma_short': 12,
            'sma_long': 20,
            'atr_period': 14,
            'rsi_period': 14,
            
            # Just a few strategies to test
            'baseline_strategies': [
                {'name': 'sma_crossover', 'params': {'fast_period': 10, 'slow_period': 20}},
                {'name': 'rsi_threshold', 'params': {'period': 14, 'threshold': 50}}
            ],
            
            'regime_boosters': {
                'bull_ranging': [
                    {'name': 'rsi_threshold', 'params': {'period': 14, 'threshold': 50}},
                    {'name': 'roc_threshold', 'params': {'period': 5, 'threshold': 0.05}}
                ],
                'bear_ranging': [
                    {'name': 'macd_crossover', 'params': {'fast_ema': 12, 'slow_ema': 26, 'signal_ema': 9}},
                    {'name': 'rsi_threshold', 'params': {'period': 14, 'threshold': 50}}
                ],
                'neutral': [
                    {'name': 'williams_r', 'params': {'williams_period': 14, 'oversold': -80, 'overbought': -20}},
                    {'name': 'rsi_bands', 'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70}}
                ]
            }
        }
    }],
    
    'execution': {
        'enable_event_tracing': True,
        'trace_settings': {
            'events_to_trace': ['SIGNAL', 'CLASSIFICATION'],
            'storage_backend': 'parquet',
            'use_sparse_storage': True
        }
    },
    
    'metadata': {
        'experiment_id': 'test_actual_ensemble',
        'description': 'Test actual two-layer ensemble'
    }
}

# Write config
config_path = Path('config/test_actual_ensemble.yaml')
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"Created config: {config_path}")
print("\nRun with:")
print("python main.py --config config/test_actual_ensemble.yaml --bars 1000")