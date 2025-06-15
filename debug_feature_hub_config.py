#!/usr/bin/env python3
"""Debug what features are actually being configured in FeatureHub during grid search."""

import logging
import yaml
from src.core.coordinator.topology import TopologyBuilder
from src.core.coordinator.config.pattern_loader import PatternLoader
from src.core.containers.factory import ContainerFactory

# Setup logging to capture feature hub initialization
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('feature_hub_debug.log')
    ]
)

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Limit to just a few strategies for debugging
config['strategies'] = config['strategies'][:5]  # Just first 5 strategy types
config['max_bars'] = 10  # Very short test

print("=== DEBUGGING FEATURE HUB CONFIGURATION ===\n")
print(f"Testing with {len(config['strategies'])} strategy types")

# Create topology builder
pattern_loader = PatternLoader()
topology_builder = TopologyBuilder(pattern_loader)
topology_builder.container_factory = ContainerFactory()

# Build topology
topology_definition = {
    'mode': 'signal_generation',
    'config': config,
    'tracing_config': {},
    'metadata': {}
}

print("\nBuilding topology...")
topology = topology_builder.build_topology(topology_definition)

# Check feature hub container
feature_hub_container = None
for name, container in topology['containers'].items():
    if 'feature_hub' in name:
        feature_hub_container = container
        break

if feature_hub_container:
    print(f"\nFound feature hub container: {feature_hub_container.name}")
    fh_component = feature_hub_container.get_component('feature_hub')
    if fh_component:
        fh = fh_component._feature_hub
        print(f"FeatureHub configured with {len(fh.feature_configs)} features")
        
        print("\nConfigured features:")
        for name, config in sorted(fh.feature_configs.items())[:20]:  # Show first 20
            print(f"  {name}: {config}")
            
        # Check if incremental hub is configured
        if fh.use_incremental and hasattr(fh, '_incremental_hub'):
            ih = fh._incremental_hub
            print(f"\nIncremental hub has {len(ih._feature_configs)} feature configs")
            
            # Try creating a test feature to verify
            print("\nTesting feature creation:")
            test_features = ['sma_20', 'bollinger_bands_20_2.0', 'stochastic_14_3', 'macd_12_26_9']
            for feat_id in test_features:
                if feat_id in ih._feature_configs:
                    try:
                        feature = ih._create_feature(feat_id, ih._feature_configs[feat_id])
                        print(f"  ✓ {feat_id} -> {type(feature).__name__}")
                    except Exception as e:
                        print(f"  ✗ {feat_id} -> ERROR: {e}")
else:
    print("\nNo feature hub container found!")

print("\nCheck feature_hub_debug.log for detailed logs")