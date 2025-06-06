#!/usr/bin/env python3
"""
Test to demonstrate the complete event flow with tracing.
Shows: Data -> Features -> Strategy -> Signal -> Portfolio -> Order -> Execution -> Fill
"""

import asyncio
import logging
from datetime import datetime

from src.core.coordinator.topology import WorkflowManager
from src.core.types.workflow import WorkflowConfig, WorkflowType, ExecutionContext
# from src.core.tracing import get_trace_summary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_event_trace_flow():
    """Test complete event flow with tracing."""
    
    # Create workflow config - simple single portfolio
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        parameters={
            'mode': 'backtest',
            'symbols': ['SPY'],
            'start_date': '2023-01-01',
            'end_date': '2023-01-10',  # Just 10 days
            'backtest': {
                'data': {
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
                        'rsi_threshold_long': 45,    # Reasonable threshold
                        'rsi_threshold_short': 55    # Reasonable threshold
                    }
                ],
                'risk_profiles': [
                    {
                        'type': 'moderate',
                        'name': 'test_risk',
                        'max_position_percent': 0.05,
                        'max_position_value': 50000
                    }
                ],
                'portfolio': {
                    'initial_capital': 100000
                },
                'execution': {
                    'fill_probability': 1.0,
                    'random_seed': 42
                }
            }
        }
    )
    
    # Create execution context
    context = ExecutionContext(
        workflow_id='trace_test_001',
        workflow_type=WorkflowType.BACKTEST,
        metadata={'test': True}
    )
    
    # Create workflow manager
    workflow_manager = WorkflowManager()
    
    try:
        # Execute workflow
        logger.info("=" * 80)
        logger.info("Starting EVENT TRACE FLOW TEST")
        logger.info("=" * 80)
        
        result = await workflow_manager.execute(config, context)
        
        # Check results
        if result.success:
            logger.info("\n✅ Backtest completed successfully!")
            
            # Show trace summary (if available)
            logger.info("\n" + "=" * 80)
            logger.info("EVENT FLOW COMPLETED")
            logger.info("=" * 80)
            
            # Show results
            if result.final_results:
                logger.info("\n" + "=" * 80)
                logger.info("BACKTEST RESULTS")
                logger.info("=" * 80)
                
                all_results = result.final_results.get('all_results', {})
                for portfolio_id, portfolio_result in all_results.items():
                    logger.info(f"\nPortfolio {portfolio_id}:")
                    logger.info(f"  Final Value: ${portfolio_result.get('final_value', 0):,.2f}")
                    logger.info(f"  Total Return: {portfolio_result.get('total_return', 0):.2%}")
                    logger.info(f"  Trades: {portfolio_result.get('metrics', {}).get('trades', 0)}")
                
        else:
            logger.error(f"\n❌ Backtest failed: {result.errors}")
            
    except Exception as e:
        logger.error(f"\n❌ Exception during backtest: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await workflow_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(test_event_trace_flow())