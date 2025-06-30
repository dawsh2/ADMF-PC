#!/usr/bin/env python3
"""Debug threshold flow from config to execution."""

import yaml
import logging
logging.basicConfig(level=logging.DEBUG)

# Load and compile a minimal test
from src.core.components.discovery import discover_components_in_module
discover_components_in_module('src.strategy.strategies.indicators.volatility')

from src.core.coordinator.compiler import StrategyCompiler

# Load config with threshold
with open('config/ensemble/config_with_eod.yaml') as f:
    config = yaml.safe_load(f)

print("=== CONFIG ===")
print(yaml.dump(config, default_flow_style=False))

# Try to compile
compiler = StrategyCompiler()

# Test with a simple config that has threshold at the strategy level
test_config = {
    'strategy': [
        {'bollinger_bands': {'period': 23, 'std_dev': 1.5}, 'threshold': 'bar_of_day < 78'}
    ]
}

print("\n=== COMPILING TEST CONFIG ===")
compiled = compiler.compile_strategies(test_config)
print(f"Compiled {len(compiled)} strategies")

if compiled:
    print(f"\nFirst strategy metadata: {compiled[0]['metadata']}")