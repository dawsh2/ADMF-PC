#!/usr/bin/env python3
"""Check what's in the analytics database"""

import sys
sys.path.append('src')

from analytics import AnalyticsWorkspace

# Connect to the workspace
workspace = AnalyticsWorkspace('workspaces/20250612_205149_expansive_grid_v1_SPY')

# Check summary
print("=== Workspace Summary ===")
summary = workspace.summary()
for key, value in summary.items():
    print(f"{key}: {value}")

# Check strategies
print("\n=== Strategies Count ===")
strategies = workspace.sql("SELECT COUNT(*) as count FROM strategies")
print(f"Total strategies: {strategies.iloc[0]['count']}")

# Check classifiers
print("\n=== Classifiers Count ===")
classifiers = workspace.sql("SELECT COUNT(*) as count FROM classifiers")
print(f"Total classifiers: {classifiers.iloc[0]['count']}")

# Show some strategies if any
if strategies.iloc[0]['count'] > 0:
    print("\n=== Sample Strategies ===")
    sample = workspace.sql("SELECT strategy_id, strategy_type, signal_file_path FROM strategies LIMIT 5")
    print(sample)

# Show some classifiers if any
if classifiers.iloc[0]['count'] > 0:
    print("\n=== Sample Classifiers ===")
    sample = workspace.sql("SELECT classifier_id, classifier_type, states_file_path FROM classifiers LIMIT 5")
    print(sample)

# Check runs
print("\n=== Runs ===")
runs = workspace.sql("SELECT * FROM runs")
print(runs)