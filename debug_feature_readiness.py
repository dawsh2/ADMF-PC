#!/usr/bin/env python3
"""Debug feature readiness and strategy execution for failing strategies."""

import subprocess
import yaml
import re

def debug_strategy_execution(strategy_type, strategy_config):
    """Debug what happens when a strategy executes."""
    
    # Create minimal config with more bars for feature warmup
    minimal_config = {
        'name': f'debug_{strategy_type}',
        'description': f'Debug {strategy_type} strategy execution',
        'symbols': ['SPY'],
        'timeframes': ['1m'],
        'data_source': 'file',
        'data_dir': './data',
        'start_date': '2023-01-01',
        'end_date': '2023-01-10',  # More days for warmup
        'max_bars': 200,  # More bars for feature readiness
        'topology': 'signal_generation',
        'strategies': [strategy_config],
        'execution': {
            'enable_event_tracing': False,
            'trace_settings': {
                'events_to_trace': ['SIGNAL'],
                'storage_backend': 'memory',
                'use_sparse_storage': True
            }
        }
    }
    
    # Write config to temp file
    config_file = f'debug_{strategy_type}_execution.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(minimal_config, f, default_flow_style=False)
    
    print(f"\n=== DEBUGGING {strategy_type.upper()} ===")
    
    # Run with more verbose output
    result = subprocess.run([
        'python', 'main.py', 
        '--config', config_file,
        '--signal-generation', 
        '--bars', '200'
    ], capture_output=True, text=True, env={'PYTHONPATH': '.'})
    
    all_output = result.stdout + "\n" + result.stderr
    
    # Analyze output
    lines = all_output.split('\n')
    
    # Look for specific patterns
    signal_lines = [line for line in lines if '📡' in line]
    error_lines = [line for line in lines if any(keyword in line.lower() 
                   for keyword in ['error', 'exception', 'traceback'])]
    
    # Look for feature inference details
    inference_lines = [line for line in lines if 'inferred' in line.lower()]
    
    # Look for strategy container info
    strategy_lines = [line for line in lines if 'strategy' in line.lower() and 
                     any(keyword in line for keyword in ['container', 'component', 'ready'])]
    
    # Look for feature hub info
    feature_hub_lines = [line for line in lines if 'featurehub' in line.lower() or 
                        'feature_hub' in line.lower()]
    
    # Look for bar processing
    bar_lines = [line for line in lines if 'bar' in line.lower() and 
                any(keyword in line for keyword in ['received', 'processing', 'processed'])]
    
    print(f"Strategy: {strategy_type}")
    print(f"Parameters: {strategy_config.get('params', {})}")
    print(f"Signals generated: {len(signal_lines)}")
    
    if signal_lines:
        print(f"Sample signals:")
        for sig in signal_lines[:3]:
            print(f"  {sig}")
    
    if error_lines:
        print(f"Errors ({len(error_lines)}):")
        for error in error_lines[:5]:
            print(f"  {error}")
    
    print(f"Feature inference:")
    for inf in inference_lines:
        print(f"  {inf}")
    
    print(f"Strategy container info ({len(strategy_lines)}):")
    for strat in strategy_lines[:3]:
        print(f"  {strat}")
    
    print(f"Feature hub info ({len(feature_hub_lines)}):")
    for hub in feature_hub_lines[:3]:
        print(f"  {hub}")
    
    print(f"Bar processing ({len(bar_lines)}):")
    for bar in bar_lines[:3]:
        print(f"  {bar}")
    
    # Clean up
    try:
        import os
        os.remove(config_file)
    except:
        pass
    
    return len(signal_lines) > 0

# Load the complete config to get strategy configurations
with open('config/complete_grid_search.yaml', 'r') as f:
    complete_config = yaml.safe_load(f)

# Test a few key failing strategies
test_strategies = ['aroon_oscillator', 'fibonacci_retracement', 'pivot_channel_breaks']

for strategy in complete_config['strategies']:
    strategy_type = strategy.get('type')
    if strategy_type in test_strategies:
        # Test with simplified parameters
        simplified_params = {}
        for param, values in strategy.get('params', {}).items():
            if isinstance(values, list) and values:
                simplified_params[param] = values[0]  # Take first value only
            else:
                simplified_params[param] = values
        
        simplified_strategy = {
            'type': strategy_type,
            'name': f'{strategy_type}_debug',
            'params': simplified_params
        }
        
        success = debug_strategy_execution(strategy_type, simplified_strategy)
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        
        if strategy_type == 'aroon_oscillator':  # Break after first one for detailed analysis
            break