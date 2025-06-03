"""Debug to track actual market prices at entry and exit."""

import asyncio
import logging
import yaml
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Track market prices and trades
market_prices = []
trades = []
last_bar_price = None

# Patch DataContainer to track bar prices
from src.execution.containers_pipeline import DataContainer
original_stream = DataContainer._stream_data

async def track_stream_data(self):
    """Track bar prices."""
    # Patch publish to track prices
    original_publish = self.event_bus.publish
    
    def track_publish(event):
        global last_bar_price
        if event.event_type.name == 'BAR':
            timestamp = event.payload['timestamp']
            symbol = event.payload['symbol']
            price = event.payload['data'].get('close', 0)
            last_bar_price = price
            market_prices.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'price': price
            })
        original_publish(event)
    
    self.event_bus.publish = track_publish
    return await original_stream(self)

DataContainer._stream_data = track_stream_data

# Track fills with actual market price
from src.execution.improved_backtest_broker import ImprovedBacktestBroker
original_execute = ImprovedBacktestBroker.execute_order

async def track_execute_order(self, order):
    """Track order execution with market price."""
    global last_bar_price
    
    # Get market price before execution
    market_price_at_fill = last_bar_price
    
    # Execute
    fill = await original_execute(self, order)
    
    if fill:
        trades.append({
            'timestamp': datetime.now(),
            'symbol': order.symbol,
            'side': str(order.side),
            'quantity': order.quantity,
            'fill_price': float(fill.price),
            'market_price': market_price_at_fill,
            'slippage': float(fill.price) - market_price_at_fill if market_price_at_fill else 0
        })
        
        logger.info(f"\n🔍 TRADE EXECUTION:")
        logger.info(f"   Order: {order.side} {order.quantity} {order.symbol}")
        logger.info(f"   Market Price: ${market_price_at_fill:.4f}")
        logger.info(f"   Fill Price: ${fill.price:.4f}")
        logger.info(f"   Slippage: ${float(fill.price) - market_price_at_fill:.4f}")
    
    return fill

ImprovedBacktestBroker.execute_order = track_execute_order

# Track position closing
from src.execution.containers_pipeline import PortfolioContainer
original_close = PortfolioContainer._close_all_positions_directly

async def track_close_positions(self, reason="manual"):
    """Track closing with market price."""
    global last_bar_price
    
    if self.portfolio_state:
        positions = self.portfolio_state.get_all_positions()
        if positions:
            logger.info(f"\n🏁 CLOSING POSITIONS:")
            for symbol, pos in positions.items():
                if hasattr(pos, 'quantity') and pos.quantity != 0:
                    logger.info(f"   Position: {symbol} {pos.quantity} shares")
                    logger.info(f"   Average Entry: ${pos.average_price:.4f}")
                    logger.info(f"   Current Market: ${last_bar_price:.4f}")
                    logger.info(f"   Position P&L: ${(last_bar_price - float(pos.average_price)) * float(pos.quantity):.2f}")
    
    await original_close(self, reason)

PortfolioContainer._close_all_positions_directly = track_close_positions

async def run_debug():
    """Run with price tracking."""
    with open('config/multi_strategy_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("PRICE TRACKING AUDIT")
    logger.info("=" * 80)
    
    from src.core.coordinator.coordinator import Coordinator
    from src.core.coordinator.simple_types import WorkflowConfig, WorkflowType
    
    coordinator = Coordinator(enable_composable_containers=True)
    
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
    
    try:
        from src.core.coordinator.coordinator import ExecutionMode
        
        result = await coordinator.execute_workflow(
            workflow_config, 
            execution_mode=ExecutionMode.COMPOSABLE
        )
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRADING SUMMARY")
        logger.info("=" * 80)
        
        if market_prices:
            logger.info(f"\nPrice Range: ${min(p['price'] for p in market_prices):.4f} - ${max(p['price'] for p in market_prices):.4f}")
        
        logger.info(f"\nTrades Executed: {len(trades)}")
        for i, trade in enumerate(trades, 1):
            logger.info(f"\n{i}. {trade['side']} {trade['quantity']} @ ${trade['fill_price']:.4f}")
            logger.info(f"   Market Price: ${trade['market_price']:.4f}")
            logger.info(f"   Slippage: ${trade['slippage']:.4f}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_debug())