#!/usr/bin/env python3
"""
Debug script to verify signal storage in portfolio containers.

This script tests:
1. Signal generation by MA crossover strategy
2. Signal reception by portfolio containers
3. Signal storage in hierarchical storage
"""

import logging
from pathlib import Path
import json

# Set up logging to see all relevant messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Focus on key modules
for module in ['src.strategy', 'src.portfolio', 'src.core.events', 'src.core.coordinator']:
    logger = logging.getLogger(module)
    logger.setLevel(logging.DEBUG)

# Run a simple test
import subprocess
import time

# Run the test
print("Running MA crossover signal storage test...")
result = subprocess.run([
    'python', 'main.py', 
    '--config', 'config/test_ma_crossover_storage.yaml',
    '--verbose'
], capture_output=True, text=True)

print("\n=== STDOUT ===")
print(result.stdout)
print("\n=== STDERR ===")
print(result.stderr)

# Wait a moment for files to be written
time.sleep(1)

# Check what was created
workspaces = Path('workspaces')
if workspaces.exists():
    print("\n=== WORKSPACES ===")
    # Find the most recent workspace
    workspace_dirs = sorted([d for d in workspaces.iterdir() if d.is_dir()], 
                           key=lambda x: x.stat().st_mtime, reverse=True)
    
    if workspace_dirs:
        latest = workspace_dirs[0]
        print(f"Latest workspace: {latest}")
        
        # Check all subdirectories
        for subdir in sorted(latest.iterdir()):
            print(f"\n{subdir.name}:")
            if subdir.is_dir():
                for file in sorted(subdir.iterdir()):
                    print(f"  {file.name} ({file.stat().st_size} bytes)")
                    
                    # If it's events.jsonl, show some content
                    if file.name == 'events.jsonl':
                        with open(file, 'r') as f:
                            lines = f.readlines()
                            print(f"    Total events: {len(lines)}")
                            if lines:
                                # Show first few events
                                print("    First few events:")
                                for i, line in enumerate(lines[:5]):
                                    event = json.loads(line)
                                    print(f"      {i+1}. {event.get('event_type', 'unknown')} - "
                                          f"container: {event.get('container_id', 'unknown')}")
                                    if event.get('event_type') == 'SIGNAL':
                                        payload = event.get('payload', {})
                                        print(f"         Symbol: {payload.get('symbol')}, "
                                              f"Direction: {payload.get('direction')}, "
                                              f"Strategy: {payload.get('strategy_id')}")
                    
                    # If it's portfolio_summary.json, show content
                    elif file.name == 'portfolio_summary.json':
                        with open(file, 'r') as f:
                            data = json.load(f)
                            print(f"    Portfolio count: {data.get('portfolio_count', 0)}")
                            portfolios = data.get('portfolios', {})
                            for pid, pdata in portfolios.items():
                                print(f"    Portfolio {pid}:")
                                print(f"      Signal count: {pdata.get('signal_count', 0)}")
                                print(f"      Events: {pdata.get('event_count', 0)}")