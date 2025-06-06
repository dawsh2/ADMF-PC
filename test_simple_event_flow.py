#!/usr/bin/env python3
"""
Simple test to verify event flow is working correctly.
"""

import asyncio
import logging

from src.core.coordinator.topology import WorkflowManager
from src.core.types.workflow import WorkflowConfig, WorkflowType, ExecutionContext

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_simple_event_flow():
    """Test simple event flow with minimal configuration."""
    
    # Create workflow config - single portfolio
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        parameters={
            'mode': 'backtest',
            'symbols': ['SPY'],
            'backtest': {
                'data': {
                    'symbols': ['SPY'],
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
                    {
                        'type': 'momentum',
                        'name': 'test_momentum',
                        'sma_period': 20,
                        'rsi_threshold_long': 30,
                        'rsi_threshold_short': 70
                    }
                ],
                'risk_profiles': [
                    {
                        'type': 'moderate',
                        'name': 'test_risk',
                        'max_position_percent': 0.10
                    }
                ],
                'portfolio': {
                    'initial_capital': 100000
                },
                'execution': {
                    'fill_probability': 1.0
                }
            }
        },
        data_config={
            'symbols': ['SPY'],
            'source': 'csv',
            'max_bars': 50  # Limit to 50 bars for testing
        }
    )
    
    # Create execution context
    context = ExecutionContext(
        workflow_id='test_simple_001',
        workflow_type=WorkflowType.BACKTEST,
        metadata={'test': True}
    )
    
    # Create workflow manager
    workflow_manager = WorkflowManager()
    
    try:
        logger.info("=" * 80)
        logger.info("Starting SIMPLE EVENT FLOW TEST")
        logger.info("=" * 80)
        
        result = await workflow_manager.execute(config, context)
        
        if result.success:
            logger.info("\n✅ Test completed successfully!")
            
            # Show results
            if result.final_results:
                logger.info("\n" + "=" * 80)
                logger.info("TEST RESULTS")
                logger.info("=" * 80)
                
                all_results = result.final_results.get('all_results', {})
                for portfolio_id, portfolio_result in all_results.items():
                    logger.info(f"\nPortfolio {portfolio_id}:")
                    logger.info(f"  Final Value: ${portfolio_result.get('final_value', 0):,.2f}")
                    logger.info(f"  Total Return: {portfolio_result.get('total_return', 0):.2%}")
                    logger.info(f"  Trades: {portfolio_result.get('metrics', {}).get('trades', 0)}")
                    
                logger.info(f"\nBars processed: {result.final_results.get('bar_count', 0)}")
                
        else:
            logger.error(f"\n❌ Test failed: {result.errors}")
            
    except Exception as e:
        logger.error(f"\n❌ Exception during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await workflow_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(test_simple_event_flow())