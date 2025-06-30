#!/usr/bin/env python3
"""Diagnose why ensemble still has poor performance."""

import pandas as pd
import json
import yaml

print("DIAGNOSING ENSEMBLE EXECUTION")
print("=" * 50)

# Load latest metadata
with open('config/ensemble/results/20250623_144646/metadata.json', 'r') as f:
    metadata = json.load(f)

print("\n1. Strategy metadata:")
strategies = metadata['strategy_metadata']['strategies']
for name, strat in strategies.items():
    print(f"   Strategy '{name}':")
    print(f"     Type: {strat['type']}")
    print(f"     Params: {strat['params']}")

# Load signals and check
signals = pd.read_parquet('config/ensemble/results/20250623_144646/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
print(f"\n2. Signal statistics:")
print(f"   Total signal changes: {len(signals)}")
print(f"   Signal values: {signals['val'].value_counts().to_dict()}")

# Calculate holding periods
gaps = signals['idx'].diff().dropna()
print(f"\n3. Holding periods:")
print(f"   Mean: {gaps.mean():.1f} bars")
print(f"   Median: {gaps.median():.0f} bars")
print(f"   1-bar exits: {(gaps == 1).sum()} ({(gaps == 1).mean()*100:.1f}%)")

# Compare with parameter sweep
print("\n4. Comparison with parameter sweep (strategy 40):")
sweep_signals = pd.read_parquet('config/bollinger/results/20250623_062931/traces/bollinger_bands/SPY_5m_compiled_strategy_40.parquet')
sweep_gaps = sweep_signals['idx'].diff().dropna()
print(f"   Sweep - Mean holding: {sweep_gaps.mean():.1f} bars")
print(f"   Sweep - 1-bar exits: {(sweep_gaps == 1).sum()} ({(sweep_gaps == 1).mean()*100:.1f}%)")

print("\n5. First 10 signals comparison:")
print("\nEnsemble:")
for i in range(min(10, len(signals))):
    row = signals.iloc[i]
    print(f"   Bar {int(row['idx'])}: signal = {row['val']}")

print("\nParameter sweep:")
for i in range(min(10, len(sweep_signals))):
    row = sweep_signals.iloc[i]
    print(f"   Bar {int(row['idx'])}: signal = {row['val']}")

print("\n6. DIAGNOSIS:")
print("-" * 40)
if strat['params'] == {}:
    print("❌ Parameters are STILL not being passed!")
    print("   Even though the fix was applied, metadata shows empty params")
    print("   This means the strategy is using defaults (period=20, std_dev=2.0)")
else:
    print("✅ Parameters are being passed correctly")
    print(f"   Using: {strat['params']}")

if (gaps == 1).mean() > 0.5:
    print("\n❌ Still seeing many 1-bar exits")
    print("   This suggests the exit logic is too tight")
else:
    print("\n✅ Holding periods look reasonable")