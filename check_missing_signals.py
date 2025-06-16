#!/usr/bin/env python3
"""Check which strategies are NOT generating signals."""

import yaml
import subprocess
import sys

# Get all expected strategy types from config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

expected_strategies = set()
for strategy in config.get('strategies', []):
    strategy_type = strategy.get('type')
    if strategy_type:
        expected_strategies.add(strategy_type)

print(f"Total expected strategy types: {len(expected_strategies)}")
print(f"Expected strategies: {sorted(expected_strategies)}")

# Run the command and capture signals
print("\nRunning grid search to check signal generation...")
cmd = ['python', 'main.py', '--config', 'config/expansive_grid_search.yaml', '--signal-generation', '--bars', '100']
result = subprocess.run(cmd, capture_output=True, text=True)

# Extract strategy types that generated signals
signals_found = set()
for line in result.stdout.split('\n'):
    if '📡 SIGNAL:' in line:
        # Extract strategy type from signal name
        parts = line.split()
        if len(parts) >= 3:
            signal_name = parts[2]
            # Remove symbol prefix and grid suffix
            strategy_part = signal_name.split('_', 1)[1] if '_' in signal_name else signal_name
            strategy_type = strategy_part.replace('_grid', '').rsplit('_', 10)[0]  # Remove grid params
            
            # Special handling for some patterns
            if '_grid_' in strategy_part:
                strategy_type = strategy_part.split('_grid_')[0]
            
            signals_found.add(strategy_type)

print(f"\nStrategies that generated signals: {len(signals_found)}")
print(f"Signal-generating strategies: {sorted(signals_found)}")

# Find missing strategies
missing_strategies = expected_strategies - signals_found
print(f"\nStrategies NOT generating signals: {len(missing_strategies)}")
for strategy in sorted(missing_strategies):
    print(f"  ❌ {strategy}")

# Also check for unexpected strategies
unexpected = signals_found - expected_strategies
if unexpected:
    print(f"\nUnexpected strategies found: {sorted(unexpected)}")