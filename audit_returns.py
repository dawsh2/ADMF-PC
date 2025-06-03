"""Audit the returns to understand the unrealistic profits."""

import asyncio
import logging
import yaml
from decimal import Decimal

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Patch to trace all fills and portfolio updates
from src.execution.containers_pipeline import PortfolioContainer
original_handle_fill = PortfolioContainer._handle_fill_event

fill_audit = []
portfolio_updates = []

async def audit_handle_fill(self, event):
    """Audit fill handling."""
    fill = event.payload.get('fill')
    if fill:
        fill_data = {
            'symbol': fill.get('symbol'),
            'side': fill.get('side'),
            'quantity': fill.get('quantity'),
            'price': fill.get('price'),
            'commission': fill.get('commission', 0),
            'portfolio_before': {
                'cash': float(self.portfolio_state.cash_balance),
                'positions': len(self.portfolio_state.positions)
            }
        }
        
    # Call original
    result = await original_handle_fill(self, event)
    
    if fill:
        fill_data['portfolio_after'] = {
            'cash': float(self.portfolio_state.cash_balance),
            'positions': len(self.portfolio_state.positions)
        }
        fill_audit.append(fill_data)
        
        # Log the fill
        logger.info(f"🔍 FILL AUDIT: {fill_data['symbol']} {fill_data['side']} {fill_data['quantity']} @ ${fill_data['price']}")
        logger.info(f"   Cash: ${fill_data['portfolio_before']['cash']:.2f} -> ${fill_data['portfolio_after']['cash']:.2f}")
        
    return result

PortfolioContainer._handle_fill_event = audit_handle_fill

# Also patch portfolio closing
original_close_positions = PortfolioContainer._close_all_positions_directly

async def audit_close_positions(self, reason):
    """Audit position closing."""
    logger.info(f"🔍 CLOSING AUDIT: Starting position close for {reason}")
    
    # Log current positions
    positions = self.portfolio_state.get_all_positions()
    for symbol, position in positions.items():
        logger.info(f"   Position: {symbol} {position.quantity} shares, avg cost: ${position.average_cost}")
    
    # Call original
    result = await original_close_positions(self, reason)
    
    logger.info(f"   Final cash after closing: ${self.portfolio_state.cash_balance}")
    
    return result

PortfolioContainer._close_all_positions_directly = audit_close_positions

# Patch market data to see prices
from src.execution.containers_pipeline import DataContainer
original_stream_data = DataContainer._stream_data

async def audit_stream_data(self):
    """Audit data streaming to log prices."""
    bar_count = 0
    first_price = None
    last_price = None
    
    async for item in original_stream_data(self):
        bar_count += 1
        
        # Log first few prices
        if bar_count <= 5 or bar_count % 10 == 0:
            # Extract price from the bar data
            if hasattr(item, 'payload'):
                data = item.payload.get('data', {})
                close_price = data.get('Close', data.get('close', 0))
                if bar_count == 1:
                    first_price = close_price
                last_price = close_price
                logger.info(f"📊 BAR {bar_count}: Price = ${close_price}")
        
        yield item
    
    if first_price and last_price:
        price_change = ((last_price - first_price) / first_price) * 100
        logger.info(f"📊 PRICE AUDIT: First price: ${first_price}, Last price: ${last_price}, Change: {price_change:.2f}%")

DataContainer._stream_data = audit_stream_data

async def audit_returns():
    """Run audit of returns."""
    
    # Load config
    with open('config/multi_strategy_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("AUDITING RETURNS")
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
    
    # Use just 50 bars like the problem case
    workflow_config.data_config['max_bars'] = 50
    
    # Execute workflow
    try:
        from src.core.coordinator.coordinator import ExecutionMode
        logger.info("🚀 Starting workflow execution...")
        
        result = await coordinator.execute_workflow(
            workflow_config, 
            execution_mode=ExecutionMode.COMPOSABLE
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("AUDIT SUMMARY")
        logger.info("=" * 80)
        
        # Summary of fills
        logger.info(f"\nTotal fills: {len(fill_audit)}")
        
        total_buy_cost = 0
        total_sell_proceeds = 0
        total_commissions = 0
        
        for fill in fill_audit:
            if 'BUY' in str(fill['side']):
                total_buy_cost += fill['quantity'] * fill['price']
            else:
                total_sell_proceeds += fill['quantity'] * fill['price']
            total_commissions += fill.get('commission', 0)
        
        logger.info(f"Total buy cost: ${total_buy_cost:.2f}")
        logger.info(f"Total sell proceeds: ${total_sell_proceeds:.2f}")
        logger.info(f"Total commissions: ${total_commissions:.2f}")
        logger.info(f"Net P&L: ${total_sell_proceeds - total_buy_cost - total_commissions:.2f}")
        
        # Final result
        portfolio = result.data.get('portfolio', {}) if result.data else {}
        initial_capital = 100000
        final_equity = portfolio.get('total_equity', initial_capital)
        returns = ((final_equity - initial_capital) / initial_capital) * 100
        
        logger.info(f"\nInitial capital: ${initial_capital:.2f}")
        logger.info(f"Final equity: ${final_equity:.2f}")
        logger.info(f"Return: {returns:.2f}%")
        
        if abs(returns) > 5:
            logger.warning(f"\n⚠️ WARNING: {returns:.2f}% return in 50 minutes is unrealistic!")
            logger.warning("Possible issues:")
            logger.warning("- Position sizing not respecting limits")
            logger.warning("- Price/quantity calculation errors")
            logger.warning("- Closing positions at wrong prices")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(audit_returns())