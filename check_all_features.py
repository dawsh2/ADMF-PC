#!/usr/bin/env python3
"""Check all features that should be configured for the full grid search."""

import yaml
from src.core.coordinator.topology import TopologyBuilder
from src.core.coordinator.config.pattern_loader import PatternLoader

# Load full config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create topology builder
pattern_loader = PatternLoader()
topology_builder = TopologyBuilder(pattern_loader)

# Get pattern and build context
pattern = topology_builder._get_pattern('signal_generation')
context = topology_builder._build_context(pattern, config, {}, {})

# Infer features
topology_builder._infer_and_inject_features(context)

# Check results
if 'feature_configs' in context['config']:
    fc = context['config']['feature_configs']
    print(f"Total feature configs: {len(fc)}")
    
    # Group by feature type
    by_type = {}
    for feat_id, config in fc.items():
        feat_type = config.get('feature', 'unknown')
        if feat_type not in by_type:
            by_type[feat_type] = []
        by_type[feat_type].append(feat_id)
    
    print("\nFeatures by type:")
    for feat_type, features in sorted(by_type.items()):
        print(f"  {feat_type}: {len(features)} features")
        if len(features) <= 5:
            for f in sorted(features):
                print(f"    - {f}")
        else:
            print(f"    - {sorted(features)[0]} ... {sorted(features)[-1]}")
    
    # Check for missing feature types
    print("\nChecking for compound features:")
    compound_types = ['bollinger_bands', 'donchian_channel', 'keltner_channel', 
                      'stochastic', 'macd', 'ichimoku', 'ultimate_oscillator']
    for ct in compound_types:
        if ct in by_type:
            print(f"  ✓ {ct}: {len(by_type[ct])} instances")
            # Show a few examples
            for f in sorted(by_type[ct])[:3]:
                config = fc[f]
                print(f"    {f} -> {config}")
        else:
            print(f"  ✗ {ct}: MISSING!")
else:
    print("No feature configs found!")