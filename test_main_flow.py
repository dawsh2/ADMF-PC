#!/usr/bin/env python3
"""
Test the exact flow from main.py
"""

import asyncio
import yaml
import sys
sys.path.insert(0, '.')

# Simulate main.py flow
with open('configs/simple_synthetic_backtest.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

print("1. Base config loaded:")
print(f"   data.file_path = {base_config.get('data', {}).get('file_path')}")

# Import like main.py does
from src.core.containers.minimal_bootstrap import MinimalBootstrap

# Create bootstrap
bootstrap = MinimalBootstrap()
bootstrap.initialize()

# Build workflow config dict like main.py
from enum import Enum
class WorkflowType(str, Enum):
    BACKTEST = "backtest"

workflow_config_dict = {
    'workflow_type': 'backtest',
    'parameters': {},
    'data_config': base_config.get('data', {}),
    'backtest_config': base_config.get('backtest', {}),
    'optimization_config': {}
}

# Add max_bars like main.py does
workflow_config_dict['data_config']['max_bars'] = 100

print("\n2. Workflow config dict:")
print(f"   data_config.file_path = {workflow_config_dict['data_config'].get('file_path')}")
print(f"   data_config.max_bars = {workflow_config_dict['data_config'].get('max_bars')}")

# Execute
async def test():
    result = await bootstrap.execute_workflow(
        workflow_config=workflow_config_dict,
        mode_override=None,
        mode_args={}
    )
    print("\n3. Result:")
    print(f"   Success: {result.get('success')}")
    print(f"   Results: {result.get('results')}")

print("\n4. Executing workflow...")
asyncio.run(test())