#!/usr/bin/env python3
"""
Check DuckDB structure to find market data.
"""

import duckdb
import pandas as pd

db_path = '/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_cost_optimized_v1_014a539f/analytics.duckdb'
conn = duckdb.connect(db_path, read_only=True)

print("DuckDB Structure Analysis")
print("="*60)

# List all tables
print("\nTables:")
tables = conn.execute("SHOW TABLES").fetchall()
for table in tables:
    print(f"  - {table[0]}")

# Check event_archives structure
print("\nevent_archives schema:")
try:
    schema = conn.execute("DESCRIBE event_archives").fetchall()
    for col in schema:
        print(f"  {col[0]}: {col[1]}")
    
    # Sample data
    print("\nSample event_archives data:")
    sample = conn.execute("SELECT * FROM event_archives LIMIT 5").df()
    print(sample)
    
except Exception as e:
    print(f"Error: {e}")

# Check runs table which might have bar data
print("\n" + "="*40)
print("runs table schema:")
try:
    schema = conn.execute("DESCRIBE runs").fetchall()
    for col in schema:
        print(f"  {col[0]}: {col[1]}")
    
    # Sample data
    print("\nSample runs data:")
    sample = conn.execute("SELECT * FROM runs LIMIT 2").df()
    print(sample)
    
except Exception as e:
    print(f"Error: {e}")

# Look for any column that might contain bar data
print("\n" + "="*40)
print("Searching for bar data...")

# Check if event_archives has JSON data
try:
    # Check what's in events_file_path
    print("\nChecking events_file_path in event_archives:")
    paths = conn.execute("SELECT DISTINCT events_file_path FROM event_archives").fetchall()
    for path in paths:
        print(f"  - {path[0]}")
        
    # Try to parse the parquet file paths
    print("\nTrying to read events parquet file...")
    event_file = '/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_cost_optimized_v1_014a539f/event_archives/adaptive_ensemble_cost_optimized/events.parquet'
    
    # Read directly with pandas
    import os
    if os.path.exists(event_file):
        events_df = pd.read_parquet(event_file)
        print(f"Events file shape: {events_df.shape}")
        print(f"Columns: {events_df.columns.tolist()}")
        print("\nSample events:")
        print(events_df.head())
        
        # Check for bar events
        if 'event_type' in events_df.columns:
            event_types = events_df['event_type'].value_counts()
            print("\nEvent types:")
            print(event_types)
            
            # Get sample bar event
            bar_events = events_df[events_df['event_type'] == 'bar']
            if len(bar_events) > 0:
                print(f"\nFound {len(bar_events):,} bar events!")
                print("Sample bar event:")
                print(bar_events.iloc[0])
    else:
        print(f"Event file not found: {event_file}")
        
except Exception as e:
    print(f"Error reading events: {e}")

conn.close()