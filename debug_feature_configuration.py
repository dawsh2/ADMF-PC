#!/usr/bin/env python3
"""Debug script to trace feature configuration during topology building."""

import logging
import yaml
from src.core.coordinator.topology import TopologyBuilder
from src.core.coordinator.config.pattern_loader import PatternLoader
from src.core.containers.factory import ContainerFactory

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create topology builder
pattern_loader = PatternLoader()
topology_builder = TopologyBuilder(pattern_loader)
topology_builder.container_factory = ContainerFactory()

# Build context
topology_definition = {
    'mode': 'signal_generation',
    'config': config,
    'tracing_config': {},
    'metadata': {}
}

# Get pattern and build context
pattern = topology_builder._get_pattern('signal_generation')
context = topology_builder._build_context(pattern, config, {}, {})

print("\n=== BEFORE FEATURE INFERENCE ===")
print(f"Config has feature_configs: {'feature_configs' in context['config']}")
print(f"Context has inferred_features: {'inferred_features' in context}")

# Infer features
topology_builder._infer_and_inject_features(context)

print("\n=== AFTER FEATURE INFERENCE ===")
print(f"Config has feature_configs: {'feature_configs' in context['config']}")
if 'feature_configs' in context['config']:
    print(f"Number of feature configs: {len(context['config']['feature_configs'])}")
    # Show first few
    feature_configs = context['config']['feature_configs']
    for i, (feature_id, config) in enumerate(list(feature_configs.items())[:5]):
        print(f"  {feature_id}: {config}")

print(f"\nContext has inferred_features: {'inferred_features' in context}")
if 'inferred_features' in context:
    print(f"Number of inferred features: {len(context['inferred_features'])}")
    print(f"First few inferred features: {sorted(context['inferred_features'])[:5]}")

# Now let's see what happens when we create the feature_hub container
print("\n=== CREATING FEATURE HUB CONTAINER ===")

# Simulate container creation with the feature hub spec
feature_hub_spec = None
for container_spec in pattern['containers']:
    if container_spec.get('name') == 'feature_hub':
        feature_hub_spec = container_spec
        break

if feature_hub_spec:
    print(f"Feature hub spec found")
    print(f"Config section: {feature_hub_spec.get('config', {})}")
    
    # Resolve the features config
    features_spec = feature_hub_spec['config'].get('features', {})
    print(f"Features spec: {features_spec}")
    
    # This should resolve to feature_configs from context
    if features_spec.get('from_config') == 'feature_configs':
        resolved_features = context['config'].get('feature_configs', {})
        print(f"Resolved features: {len(resolved_features)} feature configs")