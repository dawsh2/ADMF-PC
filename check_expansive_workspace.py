#!/usr/bin/env python3
"""Check expansive grid search workspace data"""

from src.analytics.workspace import AnalyticsWorkspace

# Connect to the workspace
workspace = AnalyticsWorkspace('workspaces/20250612_214409_expansive_grid_v1_SPY')

# Check runs
print("=== RUNS ===")
runs = workspace.sql("SELECT run_id, workflow_type, total_strategies, total_classifiers, status FROM runs")
print(runs)

# Check strategies  
print("\n=== STRATEGIES ===")
strategies = workspace.sql("SELECT COUNT(*) as total_strategies FROM strategies")
print(strategies)

# Sample some strategies
print("\n=== SAMPLE STRATEGIES ===")
sample = workspace.sql("""
    SELECT strategy_id, strategy_type, strategy_name, parameters 
    FROM strategies 
    LIMIT 5
""")
print(sample)

# Check strategy types
print("\n=== STRATEGY TYPES ===")
types = workspace.sql("""
    SELECT strategy_type, COUNT(*) as count 
    FROM strategies 
    GROUP BY strategy_type
""")
print(types)

workspace.close()