#!/usr/bin/env python3
"""Debug why strategy parameters aren't being captured in metadata.json"""

import yaml
import json
from pathlib import Path

# Load the config
with open('config/ensemble/config.yaml') as f:
    config = yaml.safe_load(f)

print("=== CONFIG STRUCTURE ===")
print(f"Config keys: {list(config.keys())}")
print(f"Strategy value: {config.get('strategy')}")
print()

# Check what the metadata extractor sees
from src.core.events.observers.strategy_metadata_extractor import extract_all_strategies_metadata

print("=== METADATA EXTRACTION ===")
metadata = extract_all_strategies_metadata(config)
print(json.dumps(metadata, indent=2))
print()

# Load actual metadata and compare
with open('config/ensemble/results/latest/metadata.json') as f:
    actual_metadata = json.load(f)

print("=== ACTUAL METADATA ===")
print("Strategy metadata section:")
print(json.dumps(actual_metadata.get('strategy_metadata', {}), indent=2))
print()

print("=== ISSUE DIAGNOSIS ===")
# The issue appears to be that the metadata extractor is processing the config correctly,
# but the MultiStrategyTracer is not getting the full config or is processing it differently

# Check if the config was passed to the tracer
print("Components section (shows what tracer captured):")
for comp_id, comp_data in actual_metadata.get('components', {}).items():
    print(f"\nComponent: {comp_id}")
    print(f"  Strategy type: {comp_data.get('strategy_type')}")
    print(f"  Parameters: {comp_data.get('parameters')}")