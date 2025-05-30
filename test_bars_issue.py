#!/usr/bin/env python3
"""
Test the --bars issue.
"""

import yaml

# Load config
with open('configs/simple_synthetic_backtest.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

print("Base config:")
print(f"  data: {base_config.get('data')}")
print()

# Simulate what main.py does
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum

class WorkflowType(str, Enum):
    BACKTEST = "backtest"

@dataclass
class WorkflowConfig:
    workflow_type: WorkflowType
    parameters: Dict[str, Any]
    data_config: Dict[str, Any]
    backtest_config: Dict[str, Any]
    optimization_config: Dict[str, Any]
    
    def dict(self):
        return {
            'workflow_type': self.workflow_type.value,
            'parameters': self.parameters,
            'data_config': self.data_config,
            'backtest_config': self.backtest_config,
            'optimization_config': self.optimization_config
        }

# Build config like main.py
workflow_config = WorkflowConfig(
    workflow_type=WorkflowType.BACKTEST,
    parameters=base_config.get('parameters', {}),
    data_config=base_config.get('data', {}),
    backtest_config=base_config.get('backtest', {}),
    optimization_config=base_config.get('optimization', {})
)

# Apply --bars like main.py does
workflow_config.data_config['max_bars'] = 1000

print("After setting max_bars:")
print(f"  data_config: {workflow_config.data_config}")
print(f"  data_config['max_bars']: {workflow_config.data_config.get('max_bars')}")
print()

# Convert to dict
config_dict = workflow_config.dict()
print("After .dict():")
print(f"  data_config: {config_dict['data_config']}")
print(f"  data_config['max_bars']: {config_dict['data_config'].get('max_bars')}")

# Now test MinimalWorkflowConfig
from src.core.coordinator.minimal_coordinator import MinimalWorkflowConfig

minimal_config = MinimalWorkflowConfig.from_dict(config_dict)
print("\nMinimalWorkflowConfig:")
print(f"  data_config: {minimal_config.data_config}")
print(f"  data_config['max_bars']: {minimal_config.data_config.get('max_bars')}")