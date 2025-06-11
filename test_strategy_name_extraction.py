#!/usr/bin/env python3
"""
Test strategy name extraction in signal generation topology.
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
from src.core.coordinator.coordinator import Coordinator

def test_strategy_name_extraction():
    """Test that strategy names are properly extracted from strategies list."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize coordinator
    coordinator = Coordinator()
    
    # Load the multi-strategy config
    config_path = "config/test_multi_strategy_signal_gen.yaml"
    
    print(f"\n=== Testing Strategy Name Extraction ===")
    print(f"Config: {config_path}")
    
    try:
        # Load the YAML config
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Run the workflow
        result = coordinator.run_workflow(config)
        
        print(f"\n=== Workflow Result ===")
        print(f"Success: {result.get('success', False)}")
        
        # Check the topology for created containers
        if 'topology' in result:
            topology = result['topology']
            containers = topology.get('containers', {})
            
            print(f"\n=== Created Containers ===")
            portfolio_containers = []
            for name, container in containers.items():
                print(f"- {name}: {container.__class__.__name__}")
                if 'portfolio' in name:
                    portfolio_containers.append(name)
            
            print(f"\n=== Portfolio Containers ===")
            if portfolio_containers:
                print(f"Found {len(portfolio_containers)} portfolio containers:")
                for name in portfolio_containers:
                    print(f"  - {name}")
                
                # Should have portfolio_fast_momentum and portfolio_slow_momentum
                expected = ['portfolio_fast_momentum', 'portfolio_slow_momentum']
                found = [name for name in expected if name in portfolio_containers]
                
                if len(found) == len(expected):
                    print(f"✅ SUCCESS: Found all expected portfolio containers: {found}")
                else:
                    print(f"❌ MISSING: Expected {expected}, found {found}")
                    missing = [name for name in expected if name not in portfolio_containers]
                    print(f"   Missing: {missing}")
                    
            else:
                print("❌ NO PORTFOLIO CONTAINERS FOUND")
        
        # Check events and traces if available
        if 'events' in result:
            events = result['events']
            print(f"\n=== Events Generated ===")
            print(f"Total events: {len(events)}")
            
            # Count events by type
            event_types = {}
            for event in events:
                event_type = getattr(event, 'event_type', str(type(event)))
                event_types[str(event_type)] = event_types.get(str(event_type), 0) + 1
            
            for event_type, count in event_types.items():
                print(f"  {event_type}: {count}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strategy_name_extraction()