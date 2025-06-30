#!/usr/bin/env python3
import pandas as pd

# Load signals
signals = pd.read_parquet('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')

print('Ensemble Signal Analysis')
print('=' * 40)
print(f'Total signal changes: {len(signals)}')
print('\nSignal value distribution:')
print(signals['val'].value_counts().sort_index())
print(f'\nUnique values: {sorted(signals["val"].unique())}')
print(f'Max signal value: {signals["val"].max()}')
print(f'Min signal value: {signals["val"].min()}')

# Check for 2 or -2
if 2 in signals['val'].values or -2 in signals['val'].values:
    print("\n⚠️  WARNING: Found signal values of 2 or -2!")
    print("This indicates both strategies voting the same way")
else:
    print("\n✓ No signal values of 2 or -2 found")
    print("Signals are properly normalized to [-1, 0, 1]")

# Sample some rows
print("\nFirst 10 signal changes:")
print(signals.head(10))
print("\nLast 10 signal changes:")
print(signals.tail(10))