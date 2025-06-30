#!/usr/bin/env python3
import subprocess
import sys

# Path to the parquet file
parquet_file = 'config/ensemble/results/20250623_084444/traces/ensemble/SPY_5m_compiled_strategy_0.parquet'

# DuckDB queries
queries = [
    # Basic statistics
    f"SELECT 'Total rows' as metric, COUNT(*) as value FROM read_parquet('{parquet_file}')",
    
    # Unique values
    f"SELECT DISTINCT val FROM read_parquet('{parquet_file}') ORDER BY val",
    
    # Distribution
    f"SELECT val, COUNT(*) as count, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage FROM read_parquet('{parquet_file}') GROUP BY val ORDER BY val",
    
    # Check for extreme values
    f"SELECT COUNT(*) as extreme_count FROM read_parquet('{parquet_file}') WHERE val < -1 OR val > 1",
    
    # Check for 2 or -2
    f"SELECT COUNT(*) as double_signal_count FROM read_parquet('{parquet_file}') WHERE val = 2 OR val = -2",
    
    # Statistics
    f"SELECT MIN(val) as min_val, MAX(val) as max_val, AVG(val) as avg_val, STDDEV(val) as std_val FROM read_parquet('{parquet_file}')",
    
    # Sample non-zero signals
    f"SELECT time, val, strategy_index FROM read_parquet('{parquet_file}') WHERE val != 0 LIMIT 10"
]

print("=== Signal Analysis Results ===\n")

for i, query in enumerate(queries):
    print(f"\n--- Query {i+1} ---")
    print(f"Query: {query[:100]}...")
    
    try:
        result = subprocess.run(
            ['duckdb', '-c', query],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.output}")
        print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Try with pandas as backup
print("\n\n=== Pandas Analysis (Backup) ===")
try:
    import pandas as pd
    df = pd.read_parquet(parquet_file)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if 'val' in df.columns:
        print(f"\nUnique values in 'val': {sorted(df['val'].unique())}")
        print(f"\nValue distribution:")
        print(df['val'].value_counts().sort_index())
        
        print(f"\nStatistics:")
        print(f"Min: {df['val'].min()}")
        print(f"Max: {df['val'].max()}")
        print(f"Mean: {df['val'].mean():.4f}")
        print(f"Std: {df['val'].std():.4f}")
        
        outside = df[(df['val'] < -1) | (df['val'] > 1)]
        print(f"\nValues outside [-1, 0, 1]: {len(outside)}")
        
        extreme = df[(df['val'] == 2) | (df['val'] == -2)]
        print(f"Values of 2 or -2: {len(extreme)}")
        
except Exception as e:
    print(f"Pandas analysis failed: {e}")