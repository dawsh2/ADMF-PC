"""Test script to verify position closing at end of backtest."""

import asyncio
import logging
import yaml

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_position_closing():
    """Test that positions are closed at end of backtest."""
    
    # Load config
    with open('config/multi_strategy_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("TESTING POSITION CLOSING AT END OF BACKTEST")
    logger.info("=" * 80)
    
    # Import main components
    from src.core.coordinator.coordinator import Coordinator
    from src.core.coordinator.simple_types import WorkflowConfig, WorkflowType
    
    # Create coordinator
    coordinator = Coordinator(enable_composable_containers=True)
    
    # Build workflow config
    workflow_config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        data_config=config['data'],
        backtest_config={
            **config.get('backtest', {}),
            'strategies': config.get('strategies', [])
        },
        parameters={
            'strategies': config.get('strategies', []),
            'risk': config.get('risk', {}),
            'container_pattern': 'simple_backtest'
        }
    )
    
    # Limit to fewer bars for testing
    workflow_config.data_config['max_bars'] = 30
    
    # Add signal aggregation to parameters
    if 'signal_aggregation' in config:
        workflow_config.parameters['signal_aggregation'] = config['signal_aggregation']
    
    # Execute workflow
    try:
        from src.core.coordinator.coordinator import ExecutionMode
        logger.info("🚀 Starting workflow execution...")
        
        result = await coordinator.execute_workflow(
            workflow_config, 
            execution_mode=ExecutionMode.COMPOSABLE
        )
        
        logger.info("=" * 80)
        logger.info("WORKFLOW EXECUTION COMPLETE")
        logger.info(f"Success: {result.success}")
        
        # Check final portfolio state
        portfolio = result.data.get('portfolio', {}) if result.data else {}
        logger.info(f"\n📊 Final Portfolio State:")
        logger.info(f"   💰 Cash Balance: ${portfolio.get('cash_balance', 0):.2f}")
        logger.info(f"   📈 Position Value: ${portfolio.get('position_value', 0):.2f}")
        logger.info(f"   🏆 Total Equity: ${portfolio.get('total_equity', 0):.2f}")
        logger.info(f"   📍 Number of Positions: {portfolio.get('positions', 0)}")
        
        # Check if positions were closed
        if portfolio.get('positions', 0) == 0:
            logger.info("\n✅ SUCCESS: All positions were closed at end of backtest!")
        else:
            logger.warning(f"\n⚠️ WARNING: {portfolio.get('positions', 0)} positions still open!")
            position_details = portfolio.get('position_details', {})
            for symbol, details in position_details.items():
                logger.warning(f"   - {symbol}: {details.get('quantity', 0)} shares")
        
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_position_closing())