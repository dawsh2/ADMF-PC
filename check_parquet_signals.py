#\!/usr/bin/env python3
import pandas as pd
import os
import sys

# Read a sample parquet file
parquet_file = "config/bollinger/results/20250625_185742/traces/signals/bollinger_bands/SPY_1m_strategy_8.parquet"

if not os.path.exists(parquet_file):
    print(f"File not found: {parquet_file}")
    sys.exit(1)

df = pd.read_parquet(parquet_file)

print(f"File: {os.path.basename(parquet_file)}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 10 rows:")
print(df.head(10))
print("\nSignal values distribution:")
print(df["val"].value_counts())
print(f"\nNon-zero signals: {(df['val'] \!= 0).sum()}")
print(f"\nDate range: {df['ts'].min()} to {df['ts'].max()}")

# Check multiple files
print("\n" + "="*60)
print("Checking multiple parquet files for signals...")
print("="*60)

traces_dir = "config/bollinger/results/20250625_185742/traces/signals/bollinger_bands/"
files_with_signals = 0
total_files = 0

for file in os.listdir(traces_dir):
    if file.endswith('.parquet'):
        total_files += 1
        df = pd.read_parquet(os.path.join(traces_dir, file))
        non_zero = (df['val'] \!= 0).sum()
        if non_zero > 0:
            files_with_signals += 1
            print(f"{file}: {non_zero} non-zero signals out of {len(df)} rows")

print(f"\nSummary: {files_with_signals}/{total_files} files have non-zero signals")
EOF < /dev/null