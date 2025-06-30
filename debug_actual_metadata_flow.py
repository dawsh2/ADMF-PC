#!/usr/bin/env python3
"""Debug the actual metadata flow when running main.py"""

import yaml
import json
from pathlib import Path

# Load the actual generated metadata
with open('config/ensemble/results/latest/metadata.json') as f:
    metadata = json.load(f)

print("=== ACTUAL METADATA.JSON ===")
print(json.dumps(metadata, indent=2))

# Now let's simulate what should have happened
print("\n=== SIMULATING METADATA EXTRACTION ===")

# Load config
with open('config/ensemble/config.yaml') as f:
    config = yaml.safe_load(f)

# Extract metadata
from src.core.events.observers.strategy_metadata_extractor import extract_all_strategies_metadata
extracted = extract_all_strategies_metadata(config)
print("Extracted metadata:")
print(json.dumps(extracted, indent=2))

# Now simulate update_metadata_with_recursive_strategies
from src.core.events.observers.strategy_metadata_extractor import update_metadata_with_recursive_strategies

# Start with basic metadata (like the tracer creates)
basic_metadata = {
    "workflow_id": "ensemble",
    "workspace_path": "config/ensemble/results/20250623_151124",
    "total_bars": 20768,
    "total_signals": 4127,
    "components": metadata.get("components", {})
}

print("\n=== SIMULATING UPDATE_METADATA_WITH_RECURSIVE_STRATEGIES ===")
updated = update_metadata_with_recursive_strategies(basic_metadata, config)
print("Updated metadata strategy section:")
print(json.dumps(updated.get('strategy_metadata', {}), indent=2))

print("\n=== COMPARISON ===")
print("Actual params:", metadata['strategy_metadata']['strategies']['unnamed']['params'])
print("Should be:", extracted['strategies']['bollinger_bands_0']['params'])