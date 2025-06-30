#!/usr/bin/env python3
"""Validate the latest ensemble metadata."""

import json

# Load metadata
with open('config/ensemble/results/latest/metadata.json') as f:
    metadata = json.load(f)

print('=== METADATA VALIDATION ===')
print(f'Workspace: {metadata["workspace_path"]}')
print(f'Created: {metadata["components"]["SPY_5m_compiled_strategy_0"]["created_at"]}')
print()

# Data validation
print('=== DATA METRICS ===')
print(f'Total bars processed: {metadata["total_bars"]:,}')
print(f'Component bars: {metadata["components"]["SPY_5m_compiled_strategy_0"]["total_bars"]:,}')
print(f'Discrepancy: {metadata["total_bars"] - metadata["components"]["SPY_5m_compiled_strategy_0"]["total_bars"]} bars')
print()

# Signal validation
print('=== SIGNAL METRICS ===')
print(f'Total signals (metadata): {metadata["total_signals"]:,}')
print(f'Signal changes stored: {metadata["stored_changes"]:,}')
print(f'Compression ratio: {metadata["compression_ratio"]:.2f}x')
print(f'Signal frequency: {metadata["components"]["SPY_5m_compiled_strategy_0"]["signal_frequency"]*100:.2f}%')
print()

# Strategy validation
print('=== STRATEGY INFO ===')
strategy = metadata['strategy_metadata']['strategies']['unnamed']
print(f'Strategy type: {strategy["type"]}')
print(f'Parameters captured: {strategy["params"]}')
print(f'Is ensemble: {strategy["is_ensemble"]}')
print()

# Issues found
print('=== POTENTIAL ISSUES ===')
issues = []

# Check 1: Parameters not captured
if not strategy['params']:
    issues.append('❌ Strategy parameters are empty - not captured in metadata')

# Check 2: Total signals vs bars mismatch
expected_signals = metadata['components']['SPY_5m_compiled_strategy_0']['total_bars']
if metadata['total_signals'] != expected_signals:
    issues.append(f'❌ Total signals ({metadata["total_signals"]}) != component bars ({expected_signals})')

# Check 3: Very low signal frequency
signal_freq = metadata['components']['SPY_5m_compiled_strategy_0']['signal_frequency']
if signal_freq < 0.05:  # Less than 5%
    issues.append(f'⚠️  Low signal frequency: {signal_freq*100:.2f}%')

# Check 4: Strategy naming
if strategy['name'] == '':
    issues.append('⚠️  Strategy has no name assigned')

# Check 5: Compression ratio sanity
if metadata['compression_ratio'] < 5:
    issues.append(f'⚠️  Low compression ratio: {metadata["compression_ratio"]:.2f}x (sparse storage may be inefficient)')

# Check 6: Test data check
if metadata['total_bars'] > 17000:  # Likely test data
    issues.append('ℹ️  This appears to be test data (20,768 bars vs ~16,600 for training)')

if issues:
    for issue in issues:
        print(issue)
else:
    print('✅ No major issues found')

# Check actual parameters used
print()
print('=== INVESTIGATING PARAMETERS ===')
print('Component parameters:', metadata['components']['SPY_5m_compiled_strategy_0']['parameters'])

# Try to find config used
import os
config_path = 'config/ensemble/config.yaml'
if os.path.exists(config_path):
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print('\nCurrent config.yaml:')
    print(config.get('strategy', 'No strategy found'))