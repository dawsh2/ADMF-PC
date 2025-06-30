"""Check signal file format"""
import pandas as pd

signal_file = "workspaces/signal_generation_1e32d562/traces/SPY_1m/signals/mean_reversion/SPY_keltner_baseline.parquet"
df = pd.read_parquet(signal_file)

print("Signal file structure:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)