#!/usr/bin/env python3
"""Debug the Bollinger Bands configuration flow."""

import yaml
import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

# Test 1: Load the config
print("=== Test 1: Loading Config ===")
with open('config/bollinger/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f"Raw config: {config}")

# Test 2: Parse with clean syntax parser
print("\n=== Test 2: Clean Syntax Parser ===")
from src.core.coordinator.config.clean_syntax_parser import parse_clean_config
parsed_config = parse_clean_config(config)
print(f"Parsed config keys: {list(parsed_config.keys())}")
if 'parameter_space' in parsed_config:
    print(f"parameter_space keys: {list(parsed_config['parameter_space'].keys())}")
    if 'strategies' in parsed_config['parameter_space']:
        print(f"Number of strategies: {len(parsed_config['parameter_space']['strategies'])}")
        print(f"First strategy: {parsed_config['parameter_space']['strategies'][0]}")

# Test 3: Check if bollinger_bands strategy exists
print("\n=== Test 3: Strategy Check ===")
# Check the actual file
import os
strategy_file = 'src/strategy/strategies/indicators/volatility.py'
if os.path.exists(strategy_file):
    print(f"✓ {strategy_file} exists")
    # Check if bollinger_bands is defined
    with open(strategy_file, 'r') as f:
        content = f.read()
        if '@strategy' in content and 'bollinger_bands' in content:
            print("✓ bollinger_bands strategy is defined in the file")
        else:
            print("✗ bollinger_bands strategy not found in file")
else:
    print(f"✗ {strategy_file} not found")

# Test 4: Minimal test - can we create a signal?
print("\n=== Test 4: Signal Generation Test ===")
# Simulate what should happen
features = {
    'bollinger_bands_20_2.0_upper': 105.0,
    'bollinger_bands_20_2.0_middle': 100.0,
    'bollinger_bands_20_2.0_lower': 95.0,
}
bar = {
    'close': 94.5,  # Below lower band
    'symbol': 'SPY',
    'timeframe': '5m',
    'timestamp': '2024-01-01T10:00:00Z'
}
params = {
    'period': 20,
    'std_dev': 2.0,
    'exit_threshold': 0.001
}

# Try to import and call the strategy directly
try:
    from src.strategy.strategies.indicators.volatility import bollinger_bands
    signal = bollinger_bands(features, bar, params)
    print(f"✓ Direct strategy call successful!")
    print(f"  Signal: {signal}")
except Exception as e:
    print(f"✗ Direct strategy call failed: {e}")
    import traceback
    traceback.print_exc()