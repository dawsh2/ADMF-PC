#!/usr/bin/env python3
"""Debug what features are actually configured in a real run."""

import yaml
import sys
sys.path.append('.')

from src.core.coordinator.topology import TopologyBuilder

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build topology - this is what main.py does
builder = TopologyBuilder()

# Build a test topology to see what features get configured
topology_def = {
    'mode': 'signal_generation',
    'config': config,
    'tracing_config': {},
    'metadata': {}
}

print("Building topology...")
topology = builder.build_topology(topology_def)

# Check if there's a feature hub container
if 'containers' in topology:
    for container_name, container_spec in topology['containers'].items():
        print(f"\nContainer: {container_name}")
        if 'components' in container_spec:
            for comp_type, components in container_spec['components'].items():
                if comp_type == 'feature_hubs':
                    print(f"  Found feature hub: {list(components.keys())}")
                    for hub_name, hub_instance in components.items():
                        if hasattr(hub_instance, '_feature_configs'):
                            feature_configs = hub_instance._feature_configs
                            print(f"    Configured features: {len(feature_configs)}")
                            
                            # Look for volume features
                            volume_features = {k: v for k, v in feature_configs.items() 
                                             if any(vf in k for vf in ['obv', 'ad', 'cmf', 'vwap'])}
                            print(f"    Volume features: {list(volume_features.keys())}")
                            
                            # Sample of all features
                            all_features = list(feature_configs.keys())
                            print(f"    Sample features: {all_features[:10]}")