#!/usr/bin/env python3
"""Debug script to investigate the 6 remaining failing strategies."""

import subprocess
import re
import yaml
from collections import defaultdict

# The 6 failing strategies
FAILING_STRATEGIES = [
    'aroon_oscillator', 'vortex_trend', 'fibonacci_retracement', 
    'price_action_swing', 'pivot_channel_breaks', 'pivot_channel_bounces',
    'trendline_breaks', 'trendline_bounces'
]

print("=== DEBUGGING FAILING STRATEGIES ===")
print(f"Investigating: {', '.join(FAILING_STRATEGIES)}")

# Create a minimal config with just one failing strategy to debug each individually
def test_single_strategy(strategy_type, strategy_config):
    """Test a single strategy in isolation."""
    
    # Create minimal config
    minimal_config = {
        'name': f'debug_{strategy_type}',
        'description': f'Debug {strategy_type} strategy',
        'symbols': ['SPY'],
        'timeframes': ['1m'],
        'data_source': 'file',
        'data_dir': './data',
        'start_date': '2023-01-01',
        'end_date': '2023-01-03',  # Just 2 days for quick testing
        'max_bars': 50,
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
    config_file = f'debug_{strategy_type}.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(minimal_config, f, default_flow_style=False)
    
    print(f"\n--- Testing {strategy_type} ---")
    
    # Run the test
    result = subprocess.run([
        'python', 'main.py', 
        '--config', config_file,
        '--signal-generation', 
        '--bars', '50'
    ], capture_output=True, text=True)
    
    all_output = result.stdout + "\n" + result.stderr
    
    # Look for signals
    signal_lines = [line for line in all_output.split('\n') if '📡' in line]
    
    # Look for errors
    error_lines = [line for line in all_output.split('\n') 
                   if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'traceback'])]
    
    # Look for feature-related issues
    feature_lines = [line for line in all_output.split('\n') 
                     if any(keyword in line.lower() for keyword in ['feature', 'missing', 'none', 'not found'])]
    
    print(f"Signals generated: {len(signal_lines)}")
    if signal_lines:
        print(f"  Sample signal: {signal_lines[0][:100]}...")
    
    if error_lines:
        print(f"Errors found: {len(error_lines)}")
        for error in error_lines[:3]:  # Show first 3 errors
            print(f"  {error}")
    
    if feature_lines and not signal_lines:
        print(f"Feature-related messages: {len(feature_lines)}")
        for feat in feature_lines[:3]:  # Show first 3 feature messages
            print(f"  {feat}")
    
    # Clean up
    try:
        import os
        os.remove(config_file)
    except:
        pass
    
    return len(signal_lines) > 0, error_lines, feature_lines

# Load the complete config to get strategy configurations
with open('config/complete_grid_search.yaml', 'r') as f:
    complete_config = yaml.safe_load(f)

# Test each failing strategy
results = {}
for strategy in complete_config['strategies']:
    strategy_type = strategy.get('type')
    if strategy_type in FAILING_STRATEGIES:
        # Test with just the first parameter combination to simplify
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
        
        success, errors, features = test_single_strategy(strategy_type, simplified_strategy)
        results[strategy_type] = {
            'success': success,
            'errors': errors,
            'features': features,
            'params': simplified_params
        }

print(f"\n=== SUMMARY ===")
for strategy_type, result in results.items():
    status = "✅ WORKING" if result['success'] else "❌ FAILING"
    print(f"{status}: {strategy_type}")
    if result['errors']:
        print(f"  Errors: {len(result['errors'])}")
    if result['features'] and not result['success']:
        print(f"  Feature issues: {len(result['features'])}")

print(f"\nWorking: {sum(1 for r in results.values() if r['success'])}/{len(results)}")
print(f"Failing: {sum(1 for r in results.values() if not r['success'])}/{len(results)}")