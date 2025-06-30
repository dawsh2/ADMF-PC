import pandas as pd

# Read parameter sweep signals
sweep = pd.read_parquet('config/bollinger/results/20250623_062931/traces/bollinger_bands/SPY_5m_compiled_strategy_40.parquet')
print("PARAMETER SWEEP:")
print(f"Total changes: {len(sweep)}")
print(f"\nFirst 20 rows:")
print(sweep.head(20))

# Read ensemble signals
ensemble = pd.read_parquet('config/ensemble/results/20250623_103142/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
print("\n\nENSEMBLE:")
print(f"Total changes: {len(ensemble)}")
print(f"\nFirst 20 rows:")
print(ensemble.head(20))

# Check gaps
print("\n\nGAP ANALYSIS:")
sweep_gaps = sweep['idx'].diff().dropna()
ensemble_gaps = ensemble['idx'].diff().dropna()

print(f"Sweep - mean gap: {sweep_gaps.mean():.1f}, median: {sweep_gaps.median():.0f}")
print(f"Ensemble - mean gap: {ensemble_gaps.mean():.1f}, median: {ensemble_gaps.median():.0f}")

# Check for immediate exits
print(f"\nSweep - 1-bar gaps: {(sweep_gaps == 1).sum()}/{len(sweep_gaps)} ({(sweep_gaps == 1).mean()*100:.1f}%)")
print(f"Ensemble - 1-bar gaps: {(ensemble_gaps == 1).sum()}/{len(ensemble_gaps)} ({(ensemble_gaps == 1).mean()*100:.1f}%)")