#!/usr/bin/env python3
"""
Test data loading issue.
"""

import yaml
import sys
sys.path.insert(0, '.')

# Load the config
with open('configs/simple_synthetic_backtest.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
print("Config loaded:")
print(f"  workflow_type: {config.get('workflow_type')}")
print(f"  data.file_path: {config.get('data', {}).get('file_path')}")
print()

# Test the coordinator directly
from src.core.coordinator.minimal_coordinator import MinimalCoordinator, MinimalWorkflowConfig

coordinator = MinimalCoordinator()
workflow_config = MinimalWorkflowConfig.from_dict(config)

print("WorkflowConfig created:")
print(f"  data_config: {workflow_config.data_config}")
print()

# Now test the engine directly
from src.execution.simple_backtest_engine import SimpleBacktestEngine

engine_config = {
    'data': workflow_config.data_config,
    'strategies': config.get('backtest', {}).get('strategies', []),
    'portfolio': config.get('backtest', {}).get('portfolio', {'initial_capital': 10000})
}

print("Engine config:")
print(f"  data: {engine_config['data']}")
print()

engine = SimpleBacktestEngine(engine_config)
print("\nEngine created, loading data...")
engine.load_data(max_bars=100)