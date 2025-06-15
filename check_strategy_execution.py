#!/usr/bin/env python3
"""
Check if missing strategies were executed by querying the analytics database.
"""

import duckdb
import pandas as pd

# Connect to the analytics database
db_path = 'workspaces/expansive_grid_search_8c6c181f/analytics.duckdb'
conn = duckdb.connect(db_path, read_only=True)

# List all tables
print("=== Available Tables ===")
tables = conn.execute("SHOW TABLES").fetchall()
for table in tables:
    print(f"  - {table[0]}")

# Check if there's a strategy execution or metadata table
print("\n=== Checking for Strategy Information ===")

# Try common table names
possible_tables = ['strategies', 'strategy_metadata', 'executions', 'runs', 'metadata']
for table_name in possible_tables:
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"\nTable '{table_name}' has {count} rows")
        
        # Get schema
        schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
        print(f"Schema:")
        for col in schema[:5]:  # First 5 columns
            print(f"  - {col[0]}: {col[1]}")
        if len(schema) > 5:
            print(f"  ... and {len(schema) - 5} more columns")
            
    except Exception as e:
        continue

# Check signals table for unique strategies
print("\n=== Unique Strategies in Signals ===")
try:
    strategies = conn.execute("""
        SELECT DISTINCT strat, COUNT(*) as signal_count 
        FROM signals 
        GROUP BY strat 
        ORDER BY strat
    """).fetchall()
    
    print(f"Found {len(strategies)} unique strategies with signals:")
    for strat, count in strategies[:20]:  # First 20
        print(f"  - {strat}: {count} signals")
    if len(strategies) > 20:
        print(f"  ... and {len(strategies) - 20} more")
        
except Exception as e:
    print(f"Error querying signals: {e}")

# Check for any execution logs or errors
print("\n=== Checking for Missing Strategies ===")
missing_strategies = [
    'accumulation_distribution',
    'adx_trend_strength',
    'aroon_crossover',
    'bollinger_breakout',
    'donchian_breakout',
    'ichimoku',
    'supertrend',
    'vortex_crossover',
    'parabolic_sar'
]

try:
    for strategy in missing_strategies[:5]:  # Check first 5
        result = conn.execute(f"""
            SELECT COUNT(*) 
            FROM signals 
            WHERE strat LIKE '%{strategy}%'
        """).fetchone()[0]
        
        if result > 0:
            print(f"  ✓ {strategy}: Found {result} signals")
        else:
            print(f"  ✗ {strategy}: No signals found")
            
except Exception as e:
    print(f"Error checking strategies: {e}")

conn.close()