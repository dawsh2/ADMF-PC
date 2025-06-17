#!/usr/bin/env python3
"""
Analyze cost-optimized ensemble strategy performance from DuckDB.
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Connect to the DuckDB database
db_path = '/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_cost_optimized_v1_014a539f/analytics.duckdb'
conn = duckdb.connect(db_path, read_only=True)

# First, check what tables we have
print("Available tables:")
tables = conn.execute("SHOW TABLES").fetchall()
for table in tables:
    print(f"  - {table[0]}")

# Check if we have a signals table with entry/exit prices
print("\nChecking for signal data with entry/exit prices...")

# Try to find the main data table
try:
    # Check for bars_1m table
    schema = conn.execute("DESCRIBE bars_1m").fetchall()
    print("\nbars_1m schema:")
    for col in schema:
        print(f"  {col[0]}: {col[1]}")
except:
    print("bars_1m table not found")

# Look for signals table
try:
    # Check what signal tables exist
    signal_tables = conn.execute("SHOW TABLES").fetchall()
    signal_tables = [t[0] for t in signal_tables if 'signal' in t[0].lower()]
    
    if signal_tables:
        print(f"\nSignal tables found: {signal_tables}")
        
        # Check the schema of the first signal table
        signal_table = signal_tables[0]
        schema = conn.execute(f"DESCRIBE {signal_table}").fetchall()
        print(f"\n{signal_table} schema:")
        for col in schema:
            print(f"  {col[0]}: {col[1]}")
            
        # Get a sample of the data
        sample = conn.execute(f"SELECT * FROM {signal_table} LIMIT 5").df()
        print(f"\nSample data from {signal_table}:")
        print(sample)
        
        # Try to construct full query with entry/exit prices
        print("\nAttempting to reconstruct full signal data with prices...")
        
        # Query to get signal data with prices
        query = f"""
        SELECT 
            timestamp,
            symbol,
            open,
            high,
            low,
            close,
            volume,
            signal,
            -- Assuming entry_price and exit_price might be in the signal table
            entry_price,
            exit_price
        FROM bars_1m b
        LEFT JOIN {signal_table} s ON b.timestamp = s.timestamp AND b.symbol = s.symbol
        WHERE b.symbol = 'SPY'
        ORDER BY b.timestamp
        """
        
        try:
            df = conn.execute(query).df()
            print(f"\nData shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Query failed: {e}")
            
            # Try simpler query
            query = """
            SELECT * FROM bars_1m 
            WHERE symbol = 'SPY' 
            ORDER BY timestamp
            LIMIT 10
            """
            sample = conn.execute(query).df()
            print("\nSample bars_1m data:")
            print(sample)
            
except Exception as e:
    print(f"Error accessing signal tables: {e}")

# Let's check what views are available
print("\nChecking for views...")
try:
    views = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'").fetchall()
    if views:
        print("Views found:")
        for view in views:
            print(f"  - {view[0]}")
            
        # Check schema of any relevant views
        for view in views:
            view_name = view[0]
            if 'signal' in view_name.lower() or 'strategy' in view_name.lower():
                print(f"\n{view_name} schema:")
                schema = conn.execute(f"DESCRIBE {view_name}").fetchall()
                for col in schema:
                    print(f"  {col[0]}: {col[1]}")
except Exception as e:
    print(f"Error checking views: {e}")

conn.close()