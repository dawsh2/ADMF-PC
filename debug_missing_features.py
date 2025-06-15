#!/usr/bin/env python3
"""Debug which features are missing for strategies that aren't generating signals."""

import sqlite3
import yaml
from collections import defaultdict

# Load the grid search config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Connect to the most recent results database
db_path = 'workspaces/expansive_grid_search_1accaf27/results.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get strategy types that ARE generating signals
cursor.execute("""
    SELECT DISTINCT 
        substr(component_id, 5, instr(substr(component_id, 5), '_') - 1) as strategy_type,
        COUNT(DISTINCT component_id) as strategy_count
    FROM signals
    GROUP BY strategy_type
    ORDER BY strategy_count DESC
""")

working_strategies = set()
print("=== WORKING STRATEGIES (generating signals) ===")
for row in cursor.fetchall():
    strategy_type = row[0]
    count = row[1]
    working_strategies.add(strategy_type)
    print(f"  {strategy_type}: {count} instances")

# Get all strategy types from config
all_strategy_types = set()
for strategy in config['strategies']:
    strategy_type = strategy['type']
    all_strategy_types.add(strategy_type)

# Find missing strategies
missing_strategies = all_strategy_types - working_strategies
print(f"\n=== MISSING STRATEGIES (NOT generating signals) ===")
print(f"Total: {len(missing_strategies)} out of {len(all_strategy_types)}")
for strategy_type in sorted(missing_strategies):
    print(f"  {strategy_type}")

# Now let's check what features these missing strategies need
print("\n=== FEATURES NEEDED BY MISSING STRATEGIES ===")

# Import strategy modules to get feature requirements
import sys
sys.path.append('/Users/daws/ADMF-PC')

from src.core.components.discovery import get_component_registry

# Import all indicator modules
import importlib
indicator_modules = [
    'src.strategy.strategies.indicators.crossovers',
    'src.strategy.strategies.indicators.oscillators',
    'src.strategy.strategies.indicators.volatility',
    'src.strategy.strategies.indicators.volume',
    'src.strategy.strategies.indicators.trend',
    'src.strategy.strategies.indicators.structure',
]

for module_path in indicator_modules:
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        print(f"Could not import {module_path}: {e}")

registry = get_component_registry()

# Analyze each missing strategy
feature_usage = defaultdict(set)
for strategy_type in sorted(missing_strategies):
    strategy_info = registry.get_component(strategy_type)
    if strategy_info:
        feature_config = strategy_info.metadata.get('feature_config', [])
        if isinstance(feature_config, list):
            print(f"\n{strategy_type} needs features: {feature_config}")
            for feature in feature_config:
                feature_usage[feature].add(strategy_type)

print("\n=== FEATURE SUMMARY ===")
for feature, strategies in sorted(feature_usage.items()):
    print(f"\n{feature} (needed by {len(strategies)} strategies):")
    for s in sorted(strategies):
        print(f"  - {s}")

conn.close()