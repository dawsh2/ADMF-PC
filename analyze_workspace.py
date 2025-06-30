#!/usr/bin/env python3
"""Analyze the workspace signal_generation_99dd4245."""

import json
import duckdb
import os

workspace_path = "workspaces/signal_generation_99dd4245"

# Read metadata
with open(os.path.join(workspace_path, "metadata.json")) as f:
    metadata = json.load(f)

print("=== WORKSPACE ANALYSIS ===")
print(f"Workflow ID: {metadata['workflow_id']}")
print(f"Total bars processed: {metadata['total_bars']:,}")
print(f"Total signals generated: {metadata['total_signals']}")
print(f"Total classifications: {metadata['total_classifications']}")
print(f"Stored changes: {metadata['stored_changes']}")

# Check DuckDB
db_path = os.path.join(workspace_path, "analytics.duckdb")
if os.path.exists(db_path):
    print("\n=== DATABASE ANALYSIS ===")
    conn = duckdb.connect(db_path, read_only=True)
    
    # List tables
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"Tables found: {len(tables)}")
    for table in tables:
        table_name = table[0]
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  - {table_name}: {count:,} rows")
        
        # Show schema
        schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
        print(f"    Schema: {[s[0] for s in schema]}")
    
    # Check for signals
    if any("signals" in t[0] for t in tables):
        print("\n=== SIGNAL ANALYSIS ===")
        signal_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        print(f"Total signals: {signal_count}")
        
        if signal_count > 0:
            # Sample signals
            print("\nSample signals (first 5):")
            signals = conn.execute("SELECT * FROM signals LIMIT 5").fetchall()
            columns = [desc[0] for desc in conn.description]
            for signal in signals:
                print(dict(zip(columns, signal)))
    
    conn.close()

print("\n=== SUMMARY ===")
if metadata['total_signals'] == 0:
    print("⚠️  No signals were generated during this run.")
    print("This could indicate:")
    print("  - The strategy conditions were not met")
    print("  - Configuration issues")
    print("  - The strategy was not properly initialized")
    print("\nCheck the configuration file and ensure strategies are properly configured.")