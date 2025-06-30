#!/usr/bin/env python3
"""Debug parameter flow from config to metadata.json"""

import yaml
import json
from pathlib import Path
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Import the modules involved
from src.core.coordinator.compiler import StrategyCompiler

# Load the config
with open('config/ensemble/config.yaml') as f:
    config = yaml.safe_load(f)

print("=== CONFIG ===")
print(yaml.dump(config, default_flow_style=False))

# Compile the strategy
compiler = StrategyCompiler()
compiled_strategies = compiler.compile_strategies(config)

print("\n=== COMPILED STRATEGIES ===")
for i, compiled in enumerate(compiled_strategies):
    print(f"\nStrategy {i}:")
    print(f"  ID: {compiled['id']}")
    print(f"  Metadata keys: {list(compiled['metadata'].keys())}")
    if 'parameters' in compiled['metadata']:
        print(f"  Parameters: {compiled['metadata']['parameters']}")
    if 'composite_strategies' in compiled['metadata']:
        print(f"  Composite strategies: {compiled['metadata']['composite_strategies']}")
    
    # Check if the function has _strategy_metadata
    if hasattr(compiled['function'], '_strategy_metadata'):
        print(f"  Function metadata keys: {list(compiled['function']._strategy_metadata.keys())}")
        if 'parameters' in compiled['function']._strategy_metadata:
            print(f"  Function parameters: {compiled['function']._strategy_metadata['parameters']}")

print("\n=== PARAMETER EXTRACTION IN METADATA ===")
# Simulate what the metadata extractor does
from src.core.events.observers.strategy_metadata_extractor import extract_all_strategies_metadata
metadata = extract_all_strategies_metadata(config)
print(json.dumps(metadata, indent=2))