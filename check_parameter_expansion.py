#!/usr/bin/env python3
"""
Check how parameter expansion works with filters.
"""

import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.coordinator.config.parameter_expander import ParameterExpander

def check_expansion():
    print("=== CHECKING PARAMETER EXPANSION WITH FILTERS ===\n")
    
    # Load config
    with open('config/keltner/config_2826/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("1. Original config strategy section:")
    print(yaml.dump(config['strategy'], default_flow_style=False))
    
    # Expand parameters
    expander = ParameterExpander()
    expanded = expander.expand_config(config)
    
    print(f"\n2. Number of combinations: {len(expanded.get('parameter_combinations', []))}")
    
    if 'parameter_combinations' in expanded:
        # Check first combination
        first = expanded['parameter_combinations'][0]
        print(f"\n3. First combination:")
        print(f"   Strategy type: {first.get('strategy_type')}")
        print(f"   Parameters: {first.get('parameters')}")
        print(f"   Has filter?: {'filter' in first}")
        
        # Check what keys are in combination
        print(f"\n4. All keys in first combination:")
        for key in first.keys():
            print(f"   - {key}: {type(first[key])}")
    
    print("\n" + "="*60)
    print("INSIGHT:")
    print("="*60)
    print("The parameter expander is NOT copying the filter to combinations!")
    print("This is why filters aren't being applied.")
    print("\nThe fix needs to be in parameter_expander.py to include")
    print("filter information when generating combinations.")

if __name__ == "__main__":
    check_expansion()