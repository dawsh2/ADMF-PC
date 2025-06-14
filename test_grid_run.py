#!/usr/bin/env python
"""Test grid search run."""

import logging
import yaml
from src.core.system_manager import SystemManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable debug for strategy state
logging.getLogger('src.strategy.state').setLevel(logging.DEBUG)

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Limit bars for testing
config['max_bars'] = 5

print("Running grid search with 5 bars...")
print(f"Number of strategies in config: {len(config.get('strategies', []))}")
print(f"Number of classifiers in config: {len(config.get('classifiers', []))}")

# Create and run system
system = SystemManager()
try:
    result = system.run_workflow(config)
    print(f"\nWorkflow completed: {result.get('status')}")
    
    # Check if signals were generated
    if 'workspace_path' in result:
        import os
        workspace = result['workspace_path']
        if os.path.exists(workspace):
            # Count signal files
            signal_files = []
            traces_dir = os.path.join(workspace, 'traces')
            if os.path.exists(traces_dir):
                for root, dirs, files in os.walk(traces_dir):
                    for file in files:
                        if file.endswith('.parquet'):
                            signal_files.append(os.path.join(root, file))
            
            print(f"Found {len(signal_files)} signal files")
            
            # Check metadata
            metadata_file = os.path.join(workspace, 'metadata.json')
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"Total signals: {metadata.get('total_signals', 0)}")
                print(f"Total classifications: {metadata.get('total_classifications', 0)}")
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()