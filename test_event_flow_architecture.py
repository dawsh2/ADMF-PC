#!/usr/bin/env python3
"""
Test the new EVENT_FLOW_ARCHITECTURE with Symbol-Timeframe containers.

This is a minimal test to verify:
1. Symbol-Timeframe containers can be created
2. Portfolio containers can be created  
3. Adapters can wire them together
4. Basic event flow works
"""

import asyncio
import logging
from datetime import datetime

from src.core.coordinator.topology import WorkflowManager
from src.core.types.workflow import WorkflowConfig, WorkflowType, ExecutionContext

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_minimal_backtest():
    """Test minimal backtest with new architecture."""
    
    # Create workflow config
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        parameters={
            'mode': 'backtest',
            'symbols': ['SPY'],
            'start_date': '2023-01-01',
            'end_date': '2023-01-31',
            'data_config': {
                'source': 'csv',
                'file_path': './data/SPY.csv'
            },
            'features': {
                'indicators': [
                    {'name': 'sma_20', 'type': 'sma', 'period': 20},
                    {'name': 'rsi', 'type': 'rsi', 'period': 14}
                ]
            },
            'strategies': [
                {'type': 'momentum', 'lookback_period': 20}
            ],
            'risk_profiles': [
                {'type': 'conservative', 'max_position_size': 0.1}
            ],
            'initial_capital': 100000
        }
    )
    
    # Create execution context
    context = ExecutionContext(
        workflow_id='test_001',
        workflow_type=WorkflowType.BACKTEST,
        metadata={'test': True}
    )
    
    # Create workflow manager
    workflow_manager = WorkflowManager()
    
    try:
        # Execute workflow
        logger.info("Starting test backtest...")
        result = await workflow_manager.execute(config, context)
        
        # Check results
        if result.success:
            logger.info("✅ Backtest completed successfully!")
            logger.info(f"Results: {result.final_results}")
        else:
            logger.error(f"❌ Backtest failed: {result.errors}")
            
    except Exception as e:
        logger.error(f"❌ Exception during backtest: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await workflow_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(test_minimal_backtest())