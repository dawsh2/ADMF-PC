#!/usr/bin/env python3
"""Test regime change detection in sparse storage."""

import json
from pathlib import Path

# Find the most recent classifier output
classifier_files = list(Path("./workspaces").rglob("*classifier*.json"))
if not classifier_files:
    print("No classifier output files found!")
    exit(1)

# Use the most recent file
latest_file = max(classifier_files, key=lambda p: p.stat().st_mtime)
print(f"Analyzing regime data from: {latest_file}")

# Load the data
with open(latest_file, 'r') as f:
    data = json.load(f)

# Analyze the changes
changes = data['changes']
print(f"\nTotal changes stored: {len(changes)}")
print(f"Total bars analyzed: {data['metadata']['total_bars']}")

# Check if these are actual regime changes
print("\nFirst 10 changes:")
for i, change in enumerate(changes[:10]):
    print(f"{i}: Bar {change['idx']}, Regime: {change['val']}")

# Count consecutive same regimes
consecutive_same = 0
prev_regime = None
for change in changes:
    if prev_regime == change['val']:
        consecutive_same += 1
    prev_regime = change['val']

print(f"\nConsecutive same regimes: {consecutive_same}")
print(f"Compression ratio: {len(changes) / data['metadata']['total_bars']:.2%}")

# Check metadata
print(f"\nMetadata regime breakdown: {data['metadata']['signal_statistics']['regime_breakdown']}")