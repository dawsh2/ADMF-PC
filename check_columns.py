#!/usr/bin/env python3
"""Check columns in the trace files."""

import pandas as pd
from pathlib import Path

# Load position data
latest_dir = Path("config/bollinger/results/latest")
positions_open = latest_dir / "traces/portfolio/positions_open/positions_open.parquet"
positions_close = latest_dir / "traces/portfolio/positions_close/positions_close.parquet"

print("=== POSITION OPEN COLUMNS ===")
if positions_open.exists():
    df_open = pd.read_parquet(positions_open)
    print(f"Columns: {list(df_open.columns)}")
    print(f"Shape: {df_open.shape}")
    print(f"\nFirst few rows:")
    print(df_open.head())

print("\n=== POSITION CLOSE COLUMNS ===")
if positions_close.exists():
    df_close = pd.read_parquet(positions_close)
    print(f"Columns: {list(df_close.columns)}")
    print(f"Shape: {df_close.shape}")
    print(f"\nFirst few rows:")
    print(df_close.head())

# Also check signals
signal_file = latest_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"
print("\n=== SIGNAL COLUMNS ===")
if signal_file.exists():
    df_signals = pd.read_parquet(signal_file)
    print(f"Columns: {list(df_signals.columns)}")
    print(f"Shape: {df_signals.shape}")
    print(f"\nFirst few rows:")
    print(df_signals.head())