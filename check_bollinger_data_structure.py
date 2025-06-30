#!/usr/bin/env python3
"""Check the structure of Bollinger sweep data."""

import pandas as pd
from pathlib import Path

# Load one sample file to check structure
sample_file = Path("/Users/daws/ADMF-PC/config/bollinger/results/20250623_062931/traces/bollinger_bands/SPY_5m_compiled_strategy_0.parquet")

df = pd.read_parquet(sample_file)

print("DataFrame shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# Check for signal-related columns
signal_cols = [col for col in df.columns if 'signal' in col.lower() or 'position' in col.lower() or 'action' in col.lower()]
print(f"\nSignal-related columns: {signal_cols}")

# Check for unique values in potential signal columns
for col in signal_cols[:5]:  # First 5 signal columns
    if df[col].dtype in ['int64', 'float64', 'object']:
        print(f"\n{col} unique values (first 10):")
        print(df[col].value_counts().head(10))