#!/usr/bin/env python3
"""Check data in the created workspace"""

from src.analytics.workspace import AnalyticsWorkspace

# Connect to the workspace
workspace = AnalyticsWorkspace('workspaces/20250612_213949_signal_generation_UNKNOWN')

# Check tables
print("=== TABLES ===")
tables = workspace.sql("SHOW TABLES")
print(tables)

# Check runs
print("\n=== RUNS ===")
runs = workspace.sql("SELECT * FROM runs")
print(runs)

# Check strategy schema
print("\n=== STRATEGY SCHEMA ===")
schema = workspace.sql("DESCRIBE strategies")
print(schema)

# Check strategies  
print("\n=== STRATEGIES ===")
strategies = workspace.sql("SELECT * FROM strategies")
if not strategies.empty:
    print(f"Found {len(strategies)} strategies")
    print(strategies.columns.tolist())
    print(strategies)
else:
    print("No strategies found")

# Check metadata
print("\n=== METADATA ===")
metadata = workspace.sql("SELECT * FROM _analytics_metadata")
print(metadata)

workspace.close()