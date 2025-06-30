#!/usr/bin/env python3
"""Trace the metadata issue by running with extra logging"""

import logging
import sys

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Only log our specific modules
for logger_name in ['src.core.events.observers.multi_strategy_tracer',
                   'src.core.events.observers.strategy_metadata_extractor',
                   'src.core.coordinator.compiler']:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

# Suppress other loggers
logging.getLogger('src').setLevel(logging.WARNING)

print("=== RUNNING MAIN.PY WITH TRACE LOGGING ===\n")

# Import and run main
import subprocess
result = subprocess.run([
    sys.executable, 'main.py',
    '--config', 'config/ensemble/config.yaml',
    '--signal-generation',
    '--dataset', 'train',
    '--close-eod'
], capture_output=True, text=True)

print("=== STDOUT ===")
print(result.stdout)

if result.stderr:
    print("\n=== STDERR ===")
    print(result.stderr)

# Now check the metadata
import json
with open('config/ensemble/results/latest/metadata.json') as f:
    metadata = json.load(f)
    
print("\n=== RESULTING METADATA ===")
print(f"Strategy name: {list(metadata['strategy_metadata']['strategies'].keys())}")
print(f"Strategy params: {metadata['strategy_metadata']['strategies'].get('unnamed', {}).get('params', 'NOT FOUND')}")