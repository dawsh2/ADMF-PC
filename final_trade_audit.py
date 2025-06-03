"""Final comprehensive trade audit with market price tracking."""

import asyncio
import logging
import yaml
from decimal import Decimal
from datetime import datetime
from collections import defaultdict

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Global tracking
audit_trail = {
    'bars': [],
    'signals': [],
    'orders': [],
    'fills': [],
    'positions': defaultdict(dict),
    'portfolio_snapshots': [],
    'market_prices': {},  # Track last market price per symbol
}

# Track bar data
from src.execution.containers_pipeline import DataContainer
original_stream = DataContainer._stream_data

async def track_bars(self):
    """Track all bar data."""
    original_publish = self.event_bus.publish
    
    def track_publish(event):
        if event.event_type.name == 'BAR':
            bar_data = {
                'timestamp': event.payload['timestamp'],
                'symbol': event.payload['symbol'],
                'open': event.payload['data'].get('open', 0),
                'high': event.payload['data'].get('high', 0),
                'low': event.payload['data'].get('low', 0),
                'close': event.payload['data'].get('close', 0),
                'volume': event.payload['data'].get('volume', 0)
            }
            audit_trail['bars'].append(bar_data)
            audit_trail['market_prices'][bar_data['symbol']] = bar_data['close']
            
            # Log first and last few bars
            bar_count = len(audit_trail['bars'])
            if bar_count <= 5 or bar_count > 45:
                logger.info(f"BAR {bar_count:3d}: {bar_data['timestamp']} - {bar_data['symbol']} "
                          f"O:{bar_data['open']:.4f} H:{bar_data['high']:.4f} "
                          f"L:{bar_data['low']:.4f} C:{bar_data['close']:.4f}")
        
        original_publish(event)
    
    self.event_bus.publish = track_publish
    return await original_stream(self)

DataContainer._stream_data = track_bars

# Track signals
from src.execution.containers_pipeline import StrategyContainer
original_emit = StrategyContainer._emit_signals

async def track_signals(self, signals, timestamp, market_data):
    """Track all signals."""
    for signal in signals:
        market_price = None
        if hasattr(signal, 'symbol') and signal.symbol in audit_trail['market_prices']:
            market_price = audit_trail['market_prices'][signal.symbol]
            
        signal_data = {
            'timestamp': timestamp,
            'symbol': signal.symbol if hasattr(signal, 'symbol') else 'UNKNOWN',
            'side': str(signal.side) if hasattr(signal, 'side') else 'UNKNOWN',
            'strategy': signal.strategy_id if hasattr(signal, 'strategy_id') else 'UNKNOWN',
            'strength': float(signal.strength) if hasattr(signal, 'strength') else 0,
            'market_price': market_price
        }
        audit_trail['signals'].append(signal_data)
        
    return await original_emit(self, signals, timestamp, market_data)

StrategyContainer._emit_signals = track_signals

# Track orders
from src.execution.containers_pipeline import RiskContainer
original_create_order = RiskContainer._create_order_from_signal

def track_orders(self, signal, market_data):
    """Track order creation."""
    order = original_create_order(self, signal, market_data)
    
    if order:
        market_price = audit_trail['market_prices'].get(order.symbol)
        order_data = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': str(order.side),
            'quantity': order.quantity,
            'order_type': str(order.order_type),
            'market_price_at_creation': market_price,
            'timestamp': datetime.now()
        }
        audit_trail['orders'].append(order_data)
        logger.info(f"\n📋 ORDER CREATED: {order_data['side']} {order_data['quantity']} "
                   f"{order_data['symbol']} @ MARKET (current price: ${market_price:.4f})")
    
    return order

RiskContainer._create_order_from_signal = track_orders

# Track fills with detailed execution info
from src.execution.improved_backtest_broker import BacktestBrokerRefactored
original_execute = BacktestBrokerRefactored.execute_order

async def track_fills(self, order):
    """Track order execution with market price."""
    # Get market price before execution
    market_price_before = audit_trail['market_prices'].get(order.symbol)
    
    # Execute
    fill = await original_execute(self, order)
    
    if fill:
        fill_data = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': str(order.side),
            'quantity': order.quantity,
            'fill_price': float(fill.price),
            'market_price_at_order': market_price_before,
            'commission': float(getattr(fill, 'commission', 0)),
            'timestamp': fill.executed_at
        }
        audit_trail['fills'].append(fill_data)
        
        logger.info(f"✅ FILL EXECUTED:")
        logger.info(f"   Order: {fill_data['side']} {fill_data['quantity']} {fill_data['symbol']}")
        logger.info(f"   Market Price: ${market_price_before:.4f}")
        logger.info(f"   Fill Price: ${fill_data['fill_price']:.4f}")
        logger.info(f"   Slippage: ${fill_data['fill_price'] - market_price_before:.4f}")
        logger.info(f"   Commission: ${fill_data['commission']:.2f}")
    
    return fill

ImprovedBacktestBroker.execute_order = track_fills

# Track portfolio updates
from src.execution.containers_pipeline import PortfolioContainer
original_handle_fill = PortfolioContainer._handle_fill_event

async def track_portfolio(self, event):
    """Track portfolio state after fills."""
    # Get state before
    cash_before = float(self.portfolio_state.get_cash_balance()) if self.portfolio_state else 0
    
    # Process fill
    result = await original_handle_fill(self, event)
    
    # Get state after
    if self.portfolio_state:
        snapshot = {
            'timestamp': datetime.now(),
            'cash': float(self.portfolio_state.get_cash_balance()),
            'positions': {},
            'total_value': float(self.portfolio_state.get_total_value())
        }
        
        for symbol, pos in self.portfolio_state.get_all_positions().items():
            snapshot['positions'][symbol] = {
                'quantity': float(pos.quantity),
                'avg_price': float(pos.average_price),
                'current_price': float(getattr(pos, 'current_price', pos.average_price))
            }
        
        audit_trail['portfolio_snapshots'].append(snapshot)
        
        cash_change = snapshot['cash'] - cash_before
        logger.info(f"📊 PORTFOLIO UPDATE:")
        logger.info(f"   Cash: ${cash_before:.2f} → ${snapshot['cash']:.2f} (change: ${cash_change:+.2f})")
        logger.info(f"   Positions: {len(snapshot['positions'])}")
        for sym, pos in snapshot['positions'].items():
            logger.info(f"   - {sym}: {pos['quantity']} @ avg ${pos['avg_price']:.4f}")
    
    return result

PortfolioContainer._handle_fill_event = track_portfolio

# Track position closing
original_close = PortfolioContainer._close_all_positions_directly

async def track_closing(self, reason="manual"):
    """Track position closing at END_OF_DATA."""
    if self.portfolio_state:
        positions = self.portfolio_state.get_all_positions()
        if positions:
            logger.info(f"\n🏁 CLOSING ALL POSITIONS ({reason}):")
            for symbol, pos in positions.items():
                if hasattr(pos, 'quantity') and pos.quantity != 0:
                    market_price = audit_trail['market_prices'].get(symbol)
                    logger.info(f"   Position: {symbol} {pos.quantity} shares")
                    logger.info(f"   Entry Price: ${pos.average_price:.4f}")
                    logger.info(f"   Current Market: ${market_price:.4f}")
                    logger.info(f"   P&L per share: ${(market_price - float(pos.average_price)):.4f}")
                    logger.info(f"   Total P&L: ${(market_price - float(pos.average_price)) * float(pos.quantity):.2f}")
    
    await original_close(self, reason)

PortfolioContainer._close_all_positions_directly = track_closing

async def run_audit():
    """Run comprehensive trade audit."""
    with open('config/multi_strategy_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE TRADE AUDIT - 50 BARS")
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
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("AUDIT SUMMARY")
        logger.info("=" * 80)
        
        # Price range
        if audit_trail['bars']:
            prices = [b['close'] for b in audit_trail['bars']]
            logger.info(f"\nMarket Data:")
            logger.info(f"  Bars: {len(audit_trail['bars'])}")
            logger.info(f"  Price Range: ${min(prices):.4f} - ${max(prices):.4f}")
            logger.info(f"  First Bar: {audit_trail['bars'][0]['timestamp']}")
            logger.info(f"  Last Bar: {audit_trail['bars'][-1]['timestamp']}")
        
        # Signals
        logger.info(f"\nSignals: {len(audit_trail['signals'])}")
        by_strategy = defaultdict(int)
        for sig in audit_trail['signals']:
            by_strategy[sig['strategy']] += 1
        for strat, count in by_strategy.items():
            logger.info(f"  {strat}: {count}")
        
        # Orders and Fills
        logger.info(f"\nOrders Created: {len(audit_trail['orders'])}")
        logger.info(f"Fills Executed: {len(audit_trail['fills'])}")
        
        # Detailed fill analysis
        if audit_trail['fills']:
            logger.info("\nDETAILED FILL ANALYSIS:")
            for i, fill in enumerate(audit_trail['fills'], 1):
                logger.info(f"\nFill #{i}:")
                logger.info(f"  Symbol: {fill['symbol']}")
                logger.info(f"  Side: {fill['side']}")
                logger.info(f"  Quantity: {fill['quantity']}")
                logger.info(f"  Market Price at Order: ${fill['market_price_at_order']:.4f}")
                logger.info(f"  Fill Price: ${fill['fill_price']:.4f}")
                logger.info(f"  Slippage: ${fill['fill_price'] - fill['market_price_at_order']:.4f}")
                logger.info(f"  Commission: ${fill['commission']:.2f}")
                
                # Find corresponding bar
                fill_time = fill['timestamp']
                closest_bar = None
                for bar in audit_trail['bars']:
                    if bar['timestamp'] <= fill_time:
                        closest_bar = bar
                    else:
                        break
                
                if closest_bar:
                    logger.info(f"  Bar at Fill Time: {closest_bar['timestamp']}")
                    logger.info(f"    OHLC: {closest_bar['open']:.4f}/{closest_bar['high']:.4f}/"
                              f"{closest_bar['low']:.4f}/{closest_bar['close']:.4f}")
        
        # Final portfolio state
        if result and 'portfolio' in result:
            logger.info(f"\nFINAL PORTFOLIO STATE:")
            logger.info(f"  Cash: ${result['portfolio']['cash_balance']:.2f}")
            logger.info(f"  Position Value: ${result['portfolio']['position_value']:.2f}")
            logger.info(f"  Total Equity: ${result['portfolio']['total_equity']:.2f}")
            logger.info(f"  Return: {((result['portfolio']['total_equity'] - 100000) / 100000) * 100:.4f}%")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_audit())