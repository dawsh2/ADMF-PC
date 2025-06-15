#!/usr/bin/env python3
"""Trace the flow of feature configuration from topology to FeatureHub."""

import logging
import yaml
from src.core.coordinator.topology import TopologyBuilder
from src.core.coordinator.config.pattern_loader import PatternLoader
from src.core.containers.factory import ContainerFactory
from src.core.containers.container import Container

# Setup detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config - use a minimal config for testing
test_config = {
    'symbols': ['SPY'],
    'timeframes': ['1m'],
    'data_source': 'file',
    'data_dir': './data',
    'strategies': [
        {
            'type': 'bollinger_breakout',
            'name': 'bollinger_breakout_test',
            'params': {
                'bb_period': 20,
                'bb_std': 2.0,
                'breakout_threshold': 0.01
            }
        },
        {
            'type': 'stochastic_crossover', 
            'name': 'stochastic_crossover_test',
            'params': {
                'k_period': 14,
                'd_period': 3
            }
        }
    ],
    'classifiers': [],
    'max_bars': 100,
    'execution': {
        'mode': 'signal_generation',
        'tracing': {
            'enabled': True
        }
    }
}

print("=== TESTING FEATURE CONFIGURATION FLOW ===\n")

# Create topology builder
pattern_loader = PatternLoader()
topology_builder = TopologyBuilder(pattern_loader)
topology_builder.container_factory = ContainerFactory()

# Build topology
topology_definition = {
    'mode': 'signal_generation',
    'config': test_config,
    'tracing_config': {},
    'metadata': {}
}

print("1. Building topology...")
topology = topology_builder.build_topology(topology_definition)

print("\n2. Checking containers created:")
for name, container in topology['containers'].items():
    print(f"   - {name}: {type(container).__name__}")
    if name == 'feature_hub':
        print(f"     Has feature_hub component: {'feature_hub' in container._components}")
        if 'feature_hub' in container._components:
            fh_component = container._components['feature_hub']
            fh = fh_component._feature_hub
            print(f"     FeatureHub initialized: {fh is not None}")
            print(f"     Use incremental: {fh.use_incremental}")
            print(f"     Number of features configured: {len(fh.feature_configs)}")
            if fh.feature_configs:
                print("     First few features:")
                for i, (name, config) in enumerate(list(fh.feature_configs.items())[:5]):
                    print(f"       {name}: {config}")

print("\n3. Checking if strategies got feature requirements:")
strategy_container = topology['containers'].get('strategy')
if strategy_container and 'strategy_state' in strategy_container._components:
    strategy_state = strategy_container._components['strategy_state']
    print(f"   Strategy state has {len(strategy_state._strategies)} strategies")
    print(f"   Feature configs: {len(strategy_state.feature_configs) if hasattr(strategy_state, 'feature_configs') else 'N/A'}")

print("\n4. Tracing feature inference:")

# Get pattern and build context
pattern = topology_builder._get_pattern('signal_generation')
context = topology_builder._build_context(pattern, test_config, {}, {})

# Check context before inference
print(f"   Before inference - has feature_configs: {'feature_configs' in context['config']}")

# Run inference
topology_builder._infer_and_inject_features(context)

print(f"   After inference - has feature_configs: {'feature_configs' in context['config']}")
if 'feature_configs' in context['config']:
    fc = context['config']['feature_configs']
    print(f"   Number of inferred features: {len(fc)}")
    print("   Inferred features:")
    for name, config in fc.items():
        print(f"     {name}: {config}")

# Check how features are passed to feature_hub container
print("\n5. Checking feature_hub container spec:")
feature_hub_spec = None
for container_spec in pattern['containers']:
    if container_spec.get('name') == 'feature_hub':
        feature_hub_spec = container_spec
        break

if feature_hub_spec:
    features_config_spec = feature_hub_spec['config'].get('features', {})
    print(f"   Features config source: {features_config_spec}")
    if features_config_spec.get('from_config') == 'feature_configs':
        print("   ✓ Feature hub is configured to get features from 'feature_configs' in context")