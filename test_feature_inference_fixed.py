#!/usr/bin/env python3
"""Test that feature inference now works properly with decorated strategies."""

import logging
from src.core.coordinator.topology import TopologyBuilder
from src.core.components.discovery import get_component_registry

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the strategy module to ensure decorator runs
import src.strategy.strategies.ma_crossover

# Check if strategy is registered
registry = get_component_registry()
strategy_info = registry.get_component('ma_crossover_strategy')
if strategy_info:
    print("✅ MA crossover strategy registered in component registry")
    print(f"   Metadata: {strategy_info.metadata}")
else:
    print("❌ MA crossover strategy NOT found in registry")

# Test feature inference
strategies = [
    {
        'name': 'test_ma',
        'type': 'ma_crossover',
        'params': {
            'fast_period': 5,
            'slow_period': 10
        }
    }
]

# Create a minimal context to test inference
builder = TopologyBuilder()
required_features = builder._infer_features_from_strategies(strategies)

print(f"\n📊 Inferred features: {sorted(required_features)}")

# Test the full inference pipeline
context = {
    'config': {
        'strategies': strategies
    }
}

builder._infer_and_inject_features(context)

print(f"\n🔧 Feature configs generated: {context.get('config', {}).get('feature_configs', {})}")
print(f"\n✨ Inferred features in context: {context.get('inferred_features', [])}")