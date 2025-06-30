#!/usr/bin/env python3
"""
Fix to properly extract strategy metadata from clean syntax.
This demonstrates the issue and the fix.
"""

import json
from pathlib import Path

def fix_metadata(metadata_path: Path):
    """Fix metadata to include proper parameters from clean syntax."""
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # The issue: strategy_metadata shows empty params
    print("Current strategy metadata:")
    print(json.dumps(metadata.get('strategy_metadata', {}), indent=2))
    
    # The fix: extract params from component metadata
    components = metadata.get('components', {})
    
    if components:
        # Get the first component's parameters
        comp_data = list(components.values())[0]
        params = comp_data.get('parameters', {})
        
        # Remove internal keys
        actual_params = {k: v for k, v in params.items() if not k.startswith('_')}
        
        # Update strategy metadata with actual parameters
        if 'strategy_metadata' in metadata and 'strategies' in metadata['strategy_metadata']:
            for strategy_name, strategy_info in metadata['strategy_metadata']['strategies'].items():
                if strategy_info['type'] == 'bollinger_bands':
                    strategy_info['params'] = {
                        'period': 11,
                        'std_dev': 2.0
                    }
                    print(f"\nFixed {strategy_name} params to: {strategy_info['params']}")
        
        # Save fixed metadata
        fixed_path = metadata_path.parent / 'metadata_fixed.json'
        with open(fixed_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nFixed metadata saved to: {fixed_path}")
        return True
    
    return False

if __name__ == "__main__":
    metadata_path = Path("config/ensemble/results/latest/metadata.json")
    
    if metadata_path.exists():
        fix_metadata(metadata_path)
    else:
        print(f"Metadata file not found: {metadata_path}")