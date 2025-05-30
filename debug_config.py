#!/usr/bin/env python3
"""
Debug the config issue.
"""

import yaml
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

# Load config
with open('configs/simple_synthetic_backtest.yaml', 'r') as f:
    base_config = yaml.safe_load(f)
    
print("Base config data section:")
print(base_config.get('data'))
print()

# Create workflow config like main.py does
workflow_config = WorkflowConfig(
    workflow_type=WorkflowType.BACKTEST,
    parameters=base_config.get('parameters', {}),
    data_config=base_config.get('data', {}),
    backtest_config=base_config.get('backtest', {}),
    optimization_config=base_config.get('optimization', {})
)

print("WorkflowConfig.data_config:")
print(workflow_config.data_config)
print()

print("WorkflowConfig.dict():")
print(workflow_config.dict())