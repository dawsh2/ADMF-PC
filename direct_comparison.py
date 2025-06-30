#!/usr/bin/env python3
import pandas as pd

# Read sweep signals
sweep_signals = pd.read_parquet('config/bollinger/results/20250623_062931/traces/bollinger_bands/SPY_5m_compiled_strategy_40.parquet')
print("PARAMETER SWEEP SIGNALS:")
print(f"Shape: {sweep_signals.shape}")
print(f"Columns: {sweep_signals.columns.tolist()}")
print("\nFirst 10 rows:")
print(sweep_signals.head(10))

# Read ensemble signals  
ensemble_signals = pd.read_parquet('config/ensemble/results/20250623_103142/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
print("\n\nENSEMBLE SIGNALS:")
print(f"Shape: {ensemble_signals.shape}")
print(f"Columns: {ensemble_signals.columns.tolist()}")
print("\nFirst 10 rows:")
print(ensemble_signals.head(10))

# Compare signal patterns
print("\n\nSIGNAL PATTERN COMPARISON:")
print(f"Sweep signal changes: {len(sweep_signals)}")
print(f"Ensemble signal changes: {len(ensemble_signals)}")

# Look at holding periods
print("\nHOLDING PERIODS:")
sweep_gaps = sweep_signals['idx'].diff().dropna()
ensemble_gaps = ensemble_signals['idx'].diff().dropna()

print(f"Sweep - Average bars between changes: {sweep_gaps.mean():.1f}")
print(f"Ensemble - Average bars between changes: {ensemble_gaps.mean():.1f}")

print(f"\nSweep - Median bars between changes: {sweep_gaps.median():.1f}")
print(f"Ensemble - Median bars between changes: {ensemble_gaps.median():.1f}")

# Look for 1-bar holdings
print(f"\nSweep - 1-bar changes: {(sweep_gaps == 1).sum()} ({(sweep_gaps == 1).mean()*100:.1f}%)")
print(f"Ensemble - 1-bar changes: {(ensemble_gaps == 1).sum()} ({(ensemble_gaps == 1).mean()*100:.1f}%)")