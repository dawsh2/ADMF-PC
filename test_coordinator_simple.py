#!/usr/bin/env python3
"""
Test the coordinator system without yaml dependency.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.coordinator.simple_types import WorkflowConfig, WorkflowType
from src.core.containers.bootstrap import ContainerBootstrap


async def test_coordinator():
    """Test basic coordinator functionality."""
    print("Testing Coordinator System...")
    print("=" * 70)
    
    # Create a simple config
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        parameters={
            'dry_run': True,
            'verbose': True
        },
        data_config={
            'type': 'csv',
            'file_path': 'data/SYNTH_1min.csv',
            'max_bars': 100
        },
        backtest_config={
            'initial_capital': 10000,
            'strategies': [{
                'name': 'threshold_strategy',
                'type': 'price_threshold',
                'parameters': {
                    'buy_threshold': 90.0,
                    'sell_threshold': 100.0
                }
            }]
        }
    )
    
    print(f"\nWorkflow Type: {config.workflow_type.value}")
    print(f"Data Source: {config.data_config['file_path']}")
    print(f"Max Bars: {config.data_config['max_bars']}")
    print(f"Initial Capital: ${config.backtest_config['initial_capital']:,}")
    
    # Test bootstrap
    print("\nTesting Bootstrap...")
    try:
        bootstrap = ContainerBootstrap()
        bootstrap.initialize()
        print("✓ Bootstrap initialized successfully")
        
        # Test workflow execution (dry-run)
        print("\nTesting Workflow Execution (dry-run)...")
        result = await bootstrap.execute_workflow(
            workflow_config=config.dict() if hasattr(config, 'dict') else config.__dict__,
            mode_override=None,
            mode_args={}
        )
        
        print(f"✓ Workflow execution returned: {result}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Test Complete")


if __name__ == '__main__':
    asyncio.run(test_coordinator())