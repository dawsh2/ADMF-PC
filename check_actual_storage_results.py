#!/usr/bin/env python3
"""
Check actual storage results from the grid search run
"""
import duckdb
import json

# Connect to the most recent analytics database
db_path = "workspaces/expansive_grid_search_5fe966d1/analytics.duckdb"
conn = duckdb.connect(db_path, read_only=True)

# Check what tables exist
print("Tables in database:")
tables = conn.execute("SHOW TABLES").fetchall()
for table in tables:
    print(f"  - {table[0]}")
    # Show columns for each table
    columns = conn.execute(f"DESCRIBE {table[0]}").fetchall()
    for col in columns[:5]:  # Show first 5 columns
        print(f"    * {col[0]}: {col[1]}")
    if len(columns) > 5:
        print(f"    ... and {len(columns) - 5} more columns")

# Check strategies table
print("\n=== STRATEGIES TABLE ===")
strategy_count = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
print(f"Total strategies stored: {strategy_count}")

# Sample strategies
print("\nSample strategies:")
strategies = conn.execute("""
    SELECT strategy_id, strategy_type, parameters
    FROM strategies
    LIMIT 10
""").fetchall()
for s in strategies:
    print(f"  - {s[0]}: {s[1]} with params {s[2]}")

# Check event_archives for signals
print("\n=== EVENT ARCHIVES (Signal Storage) ===")
event_count = conn.execute("SELECT COUNT(*) FROM event_archives").fetchone()[0]
print(f"Total events archived: {event_count:,}")

# Check event types
print("\nEvent types in archive:")
event_types = conn.execute("""
    SELECT 
        json_extract_string(event_data, '$.event_type') as event_type,
        COUNT(*) as count
    FROM event_archives
    GROUP BY json_extract_string(event_data, '$.event_type')
    ORDER BY COUNT(*) DESC
""").fetchall()
for et in event_types:
    print(f"  - {et[0]}: {et[1]:,}")

# Check for signal events specifically
print("\nChecking for signal events:")
signal_events = conn.execute("""
    SELECT COUNT(*) 
    FROM event_archives 
    WHERE json_extract_string(event_data, '$.event_type') = 'signal_generated'
       OR json_extract_string(event_data, '$.event_type') LIKE '%signal%'
""").fetchone()[0]
print(f"Signal-related events: {signal_events:,}")

# Sample signal events
if signal_events > 0:
    print("\nSample signal events:")
    samples = conn.execute("""
        SELECT 
            timestamp,
            json_extract_string(event_data, '$.strategy_id') as strategy_id,
            json_extract_string(event_data, '$.signal_value') as signal_value,
            event_data
        FROM event_archives 
        WHERE json_extract_string(event_data, '$.event_type') LIKE '%signal%'
        LIMIT 5
    """).fetchall()
    for s in samples:
        print(f"  - {s[0]}: {s[1]} = {s[2]}")

# Check classifiers table
print("\n=== CLASSIFIERS TABLE ===")
classifier_count = conn.execute("SELECT COUNT(*) FROM classifiers").fetchone()[0]
print(f"Total classifiers stored: {classifier_count}")

# Check traces directory
print("\n=== TRACES DIRECTORY ===")
import os
traces_dir = "workspaces/expansive_grid_search_5fe966d1/traces"
if os.path.exists(traces_dir):
    trace_files = os.listdir(traces_dir)
    print(f"Trace files: {len(trace_files)}")
    for f in trace_files[:5]:
        print(f"  - {f}")
        # Check file size
        size = os.path.getsize(os.path.join(traces_dir, f))
        print(f"    Size: {size:,} bytes")

# Check for parquet files
print("\n=== CHECKING FOR PARQUET FILES ===")
import glob
parquet_files = glob.glob("workspaces/expansive_grid_search_5fe966d1/**/*.parquet", recursive=True)
print(f"Parquet files found: {len(parquet_files)}")
for pf in parquet_files[:10]:
    print(f"  - {pf}")

conn.close()