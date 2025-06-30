#!/usr/bin/env python3
"""Verify signal alignment and parameters"""

import pandas as pd
import json

# Load metadata to check parameters
with open('config/ensemble/results/latest/metadata.json', 'r') as f:
    metadata = json.load(f)

print("Strategy Configuration Check")
print("=" * 50)

# Check component metadata for actual parameters
components = metadata.get('components', {})
for comp_id, comp_data in components.items():
    print(f"\nComponent: {comp_id}")
    print(f"Strategy Type: {comp_data.get('strategy_type')}")
    print(f"Parameters from component: {comp_data.get('parameters', {})}")
    
# Also check strategy metadata
print("\nStrategy metadata:")
for strategy_name, strategy_info in metadata['strategy_metadata']['strategies'].items():
    print(f"\nStrategy: {strategy_info['type']}")
    print(f"Parameters from metadata: {strategy_info['params']}")

# Load signals
signals = pd.read_parquet('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')

print(f"\nSignal Statistics:")
print(f"Total signal changes: {len(signals)}")
print(f"Signal value counts:")
print(signals['val'].value_counts())

# Check signal timing
print(f"\nFirst signal at bar index: {signals['idx'].iloc[0]}")
print(f"Last signal at bar index: {signals['idx'].iloc[-1]}")

# Load market data
market_data = pd.read_parquet('data/SPY_5m.parquet')
print(f"\nMarket data bars: {len(market_data)}")

# Check if we're using the right data split
if 'src_file' in signals.columns:
    print(f"\nData source: {signals['src_file'].iloc[0]}")
    
# Verify Bollinger parameters match what was tested before
print("\n⚠️  IMPORTANT: Check if these parameters match your successful backtest!")
print("Your previous comment mentioned:")
print("  - period: 11, std_dev: 2.0 as best performer")
print("  - But metadata shows different parameters were used")