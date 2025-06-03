"""Debug script to trace market prices through the system."""

import asyncio
import logging
import yaml

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Patch to log market data in signals
from src.execution.containers_pipeline import RiskContainer
original_create_order = RiskContainer._create_order_from_signal

def debug_create_order(self, signal, market_data):
    """Debug order creation to see market data."""
    logger.info(f"🔍 RiskContainer._create_order_from_signal called")
    logger.info(f"   Signal: {signal}")
    logger.info(f"   Market data type: {type(market_data)}")
    logger.info(f"   Market data keys: {list(market_data.keys()) if isinstance(market_data, dict) else 'Not a dict'}")
    
    if isinstance(market_data, dict):
        for key, value in market_data.items():
            if isinstance(value, dict):
                logger.info(f"   Market data[{key}]: {list(value.keys())}")
                if 'close' in value:
                    logger.info(f"      Close price: {value['close']}")
            else:
                logger.info(f"   Market data[{key}]: {value}")
    
    # Call original
    return original_create_order(self, signal, market_data)

RiskContainer._create_order_from_signal = debug_create_order

# Also patch the execution engine to see what price it uses
from src.execution.execution_engine import DefaultExecutionEngine
original_execute = DefaultExecutionEngine.execute_order

def debug_execute_order(self, order):
    """Debug order execution."""
    logger.info(f"🎯 ExecutionEngine.execute_order called for {order.symbol}")
    
    # Call original
    result = original_execute(self, order)
    
    # Check market data
    if hasattr(self, 'event_bus'):
        logger.info(f"   Event bus exists")
    
    return result

DefaultExecutionEngine.execute_order = debug_execute_order

async def debug_prices():
    """Debug market prices."""
    
    # Load config
    with open('config/multi_strategy_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("DEBUGGING MARKET PRICES")
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
    
    # Use just 20 bars
    workflow_config.data_config['max_bars'] = 20
    
    # Execute workflow
    try:
        from src.core.coordinator.coordinator import ExecutionMode
        
        result = await coordinator.execute_workflow(
            workflow_config, 
            execution_mode=ExecutionMode.COMPOSABLE
        )
        
        logger.info("Workflow completed")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_prices())