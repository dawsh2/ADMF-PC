#!/usr/bin/env python3
"""Test workspace creation with grid search config."""

import sys
import os
from pathlib import Path
import yaml

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.coordinator import Coordinator

def main():
    # Create coordinator
    coord = Coordinator()
    
    # Load the expansive grid search config
    config_path = 'config/expansive_grid_search.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override to run just a few bars for testing
    config['max_bars'] = 50
    config['strategies'] = [
        {
            'type': 'sma_crossover',
            'name': 'test_sma',
            'params': {
                'fast_period': 5,
                'slow_period': 10
            }
        }
    ]
    config['classifiers'] = []  # No classifiers for this test
    
    # Get topology name from config
    topology_name = config.get('topology', 'signal_generation')
    
    # Run topology
    result = coord.run_topology(topology_name, config)
    
    # Check results
    if result.get('success'):
        print(f"✅ Topology completed successfully")
        
        # Check for analytics workspace
        if 'analytics_workspace' in result:
            analytics_path = Path(result['analytics_workspace'])
            workspace_dir = analytics_path.parent
            print(f"📊 Workspace created: {workspace_dir.name}")
            print(f"📁 Full path: {workspace_dir}")
            
            # List workspace contents
            if workspace_dir.exists():
                print(f"\n📂 Workspace contents:")
                for item in sorted(workspace_dir.rglob('*')):
                    if item.is_file():
                        rel_path = item.relative_to(workspace_dir)
                        size = item.stat().st_size
                        print(f"  📄 {rel_path} ({size} bytes)")
    else:
        print(f"❌ Topology failed: {result.get('error', 'Unknown error')}")

if __name__ == '__main__':
    main()