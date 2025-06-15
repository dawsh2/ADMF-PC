#!/usr/bin/env python3
"""
Check run details to understand what happened.
"""

import duckdb

# Connect to the analytics database
db_path = 'workspaces/20250614_211925_indicator_grid_v3_SPY/analytics.duckdb'
conn = duckdb.connect(db_path, read_only=True)

print("=== Run Details Analysis ===\n")

# Check run information
runs = conn.execute("SELECT * FROM runs").fetchall()
print(f"Number of runs: {len(runs)}")

if runs:
    run = runs[0]
    run_columns = [desc[0] for desc in conn.execute("DESCRIBE runs").fetchall()]
    print(f"Run columns: {run_columns}")
    
    for i, col in enumerate(run_columns):
        print(f"  {col}: {run[i]}")

# Check strategies sample
print(f"\n=== Strategy Sample ===")
strategy_sample = conn.execute("""
    SELECT strategy_type, strategy_name, parameters
    FROM strategies 
    ORDER BY strategy_type, strategy_name
    LIMIT 10
""").fetchall()

for strategy_type, strategy_name, params in strategy_sample:
    print(f"  {strategy_type}: {strategy_name}")
    print(f"    Params: {params}")

# Check for any metadata about execution
print(f"\n=== Checking _analytics_metadata ===")
try:
    metadata = conn.execute("SELECT * FROM _analytics_metadata").fetchall()
    if metadata:
        metadata_cols = [desc[0] for desc in conn.execute("DESCRIBE _analytics_metadata").fetchall()]
        print(f"Metadata columns: {metadata_cols}")
        for row in metadata:
            for i, col in enumerate(metadata_cols):
                print(f"  {col}: {row[i]}")
except Exception as e:
    print(f"Error reading metadata: {e}")

# Count by strategy type to see distribution
print(f"\n=== Strategy Distribution ===")
strategy_types = conn.execute("""
    SELECT strategy_type, COUNT(*) as count
    FROM strategies
    GROUP BY strategy_type
    ORDER BY count DESC
""").fetchall()

for strategy_type, count in strategy_types:
    print(f"  {strategy_type}: {count}")

conn.close()