#!/usr/bin/env python3
"""Quick analysis of ensemble parquet file using duckdb"""

import sys
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("DuckDB not installed. Using pandas instead.")
    import pandas as pd
    
    file_path = Path('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
    if file_path.exists():
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        print("\nFirst 10 rows:")
        print(df.head(10))
        print("\nSignal value counts:")
        if 'val' in df.columns:
            print(df['val'].value_counts())
    else:
        print(f"File not found: {file_path}")
    sys.exit(0)

# Use DuckDB
con = duckdb.connect(':memory:')

file_path = 'config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet'

# Basic structure query
print("=== FILE STRUCTURE ===")
result = con.execute(f"""
    SELECT * FROM read_parquet('{file_path}') 
    LIMIT 0
""").description

print("Columns:")
for col in result:
    print(f"  - {col[0]}")

# Row count
count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{file_path}')").fetchone()[0]
print(f"\nTotal rows: {count}")

# First 10 rows
print("\n=== FIRST 10 ROWS ===")
df = con.execute(f"SELECT * FROM read_parquet('{file_path}') LIMIT 10").df()
print(df)

# Signal analysis
print("\n=== SIGNAL ANALYSIS ===")
signal_counts = con.execute(f"""
    SELECT val, COUNT(*) as count 
    FROM read_parquet('{file_path}') 
    GROUP BY val 
    ORDER BY val
""").df()
print("Signal value distribution:")
print(signal_counts)

# Signal changes
print("\n=== SIGNAL CHANGES ===")
changes = con.execute(f"""
    WITH signal_changes AS (
        SELECT 
            idx,
            val,
            LAG(val) OVER (ORDER BY idx) as prev_val
        FROM read_parquet('{file_path}')
    )
    SELECT COUNT(*) as total_changes
    FROM signal_changes
    WHERE val != prev_val OR prev_val IS NULL
""").fetchone()[0]
print(f"Total signal changes: {changes}")

# Check for any other columns
print("\n=== COLUMN ANALYSIS ===")
all_cols = con.execute(f"SELECT * FROM read_parquet('{file_path}') LIMIT 1").df().columns.tolist()
for col in all_cols:
    if col not in ['idx', 'val', 'px']:
        unique_count = con.execute(f"SELECT COUNT(DISTINCT {col}) FROM read_parquet('{file_path}')").fetchone()[0]
        print(f"{col}: {unique_count} unique values")
        if unique_count < 20:
            values = con.execute(f"SELECT DISTINCT {col} FROM read_parquet('{file_path}') ORDER BY {col}").df()
            print(f"  Values: {values[col].tolist()}")

con.close()