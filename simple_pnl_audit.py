"""Simple P&L audit to understand returns."""

import asyncio
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track fills
fills = []

# Patch execution engine to log fills with more detail
from src.execution.execution_engine import DefaultExecutionEngine
original_execute = DefaultExecutionEngine.execute_order

def audit_execute_order(self, order):
    """Audit order execution."""
    # Get market data before execution
    market_data = self._market_data.get(order.symbol, {})
    market_price = market_data.get('price', 100)
    
    # Call original
    fill = original_execute(self, order)
    
    if fill:
        fill_info = {
            'symbol': order.symbol,
            'side': str(order.side),
            'quantity': float(getattr(fill, 'quantity', 0)),
            'price': float(getattr(fill, 'price', 0)),
            'market_price': market_price,
            'slippage': float(getattr(fill, 'price', 0)) - market_price
        }
        fills.append(fill_info)
        logger.info(f"FILL: {fill_info['side']} {fill_info['quantity']} {fill_info['symbol']} @ ${fill_info['price']:.2f} (market: ${fill_info['market_price']:.2f}, slip: ${fill_info['slippage']:.2f})")
    
    return fill

DefaultExecutionEngine.execute_order = audit_execute_order

async def simple_audit():
    """Run simple audit."""
    
    # Load config
    with open('config/multi_strategy_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 60)
    logger.info("SIMPLE P&L AUDIT - 50 BARS")
    logger.info("=" * 60)
    
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
    
    # Use 50 bars
    workflow_config.data_config['max_bars'] = 50
    
    # Execute workflow
    try:
        from src.core.coordinator.coordinator import ExecutionMode
        
        result = await coordinator.execute_workflow(
            workflow_config, 
            execution_mode=ExecutionMode.COMPOSABLE
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("FILL ANALYSIS")
        logger.info("=" * 60)
        
        # Analyze fills
        total_buys = sum(f['quantity'] * f['price'] for f in fills if 'BUY' in f['side'])
        total_sells = sum(f['quantity'] * f['price'] for f in fills if 'SELL' in f['side'])
        buy_count = len([f for f in fills if 'BUY' in f['side']])
        sell_count = len([f for f in fills if 'SELL' in f['side']])
        
        logger.info(f"Total fills: {len(fills)}")
        logger.info(f"Buy orders: {buy_count}, Total value: ${total_buys:,.2f}")
        logger.info(f"Sell orders: {sell_count}, Total value: ${total_sells:,.2f}")
        logger.info(f"Net cash flow: ${total_sells - total_buys:,.2f}")
        
        # Get prices
        if fills:
            prices = [f['market_price'] for f in fills]
            logger.info(f"\nPrice range: ${min(prices):.2f} - ${max(prices):.2f}")
            logger.info(f"Price change: {((max(prices) - min(prices)) / min(prices) * 100):.2f}%")
        
        # Final result
        portfolio = result.data.get('portfolio', {}) if result.data else {}
        initial_capital = 100000
        final_equity = portfolio.get('total_equity', initial_capital)
        returns = ((final_equity - initial_capital) / initial_capital) * 100
        
        logger.info(f"\nFINAL RESULTS:")
        logger.info(f"Initial: ${initial_capital:,.2f}")
        logger.info(f"Final: ${final_equity:,.2f}")
        logger.info(f"Return: {returns:.2f}% in 50 minutes")
        
        # Check position sizing
        if fills:
            avg_position_size = sum(f['quantity'] * f['price'] for f in fills) / len(fills)
            logger.info(f"\nAverage position value: ${avg_position_size:,.2f}")
            logger.info(f"As % of capital: {avg_position_size / initial_capital * 100:.1f}%")
            
            # Check configured position size
            position_size = config.get('risk', {}).get('position_sizers', [{}])[0].get('size', 0)
            logger.info(f"Configured position size: ${position_size}")
            
            if avg_position_size > position_size * 2:
                logger.warning(f"⚠️ Average position size ${avg_position_size:.0f} >> configured ${position_size}")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_audit())