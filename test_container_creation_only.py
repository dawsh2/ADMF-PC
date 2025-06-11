#!/usr/bin/env python3
"""
Test just the container creation part of strategy name extraction.
"""

import sys
import os
import logging
from pathlib import Path

# Ensure we're in the project root and add src to path
project_root = Path(__file__).parent
os.chdir(project_root)
sys.path.insert(0, str(project_root / 'src'))

# Now import with proper path
from src.core.coordinator.topology import TopologyBuilder

def test_container_creation():
    """Test container creation with strategy name extraction."""
    
    # Set up logging to see our extraction logs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create topology builder
    builder = TopologyBuilder()
    
    # Create a simple config with multiple strategies
    config = {
        'strategies': [
            {'name': 'fast_momentum', 'type': 'momentum', 'params': {'sma_period': 5}},
            {'name': 'slow_momentum', 'type': 'momentum', 'params': {'sma_period': 20}}
        ],
        'symbols': ['SPY'],
        'timeframes': ['1m'],
        'data_source': 'file',
        'data_dir': './data',
        'start_date': '2024-01-01',
        'end_date': '2024-01-02',
        'max_bars': 5
    }
    
    # Build the signal generation topology
    topology_def = {
        'mode': 'signal_generation',
        'config': config,
        'tracing_config': {'enabled': False}  # Disable tracing for simplicity
    }
    
    print("=== Testing Strategy Name Extraction ===")
    print(f"Input strategies: {[s['name'] for s in config['strategies']]}")
    
    try:
        # Build topology
        topology = builder.build_topology(topology_def)
        
        print(f"\n=== Topology Built Successfully ===")
        containers = topology.get('containers', {})
        
        print(f"Total containers: {len(containers)}")
        
        # Check for portfolio containers by looking at actual container names
        portfolio_containers = []
        portfolio_logical_names = []
        
        for container_id, container in containers.items():
            if 'portfolio' in container_id:
                portfolio_containers.append(container_id)
                # Get the logical name from the container
                logical_name = getattr(container, 'name', container_id)
                portfolio_logical_names.append(logical_name)
                print(f"  Container ID: {container_id} -> Logical Name: {logical_name}")
        
        print(f"\nPortfolio containers by ID: {portfolio_containers}")
        print(f"Portfolio containers by logical name: {portfolio_logical_names}")
        
        # Expected portfolio containers based on strategy names
        expected = ['portfolio_fast_momentum', 'portfolio_slow_momentum']
        
        success = True
        for expected_name in expected:
            if expected_name in portfolio_logical_names:
                print(f"✅ Found: {expected_name}")
            else:
                print(f"❌ Missing: {expected_name}")
                success = False
        
        if success:
            print(f"\n🎉 SUCCESS: All expected portfolio containers created!")
            return True
        else:
            print(f"\n❌ FAILURE: Some expected containers missing")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_container_creation()
    sys.exit(0 if success else 1)