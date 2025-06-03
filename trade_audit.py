"""Detailed trade audit for manual validation."""

import asyncio
import logging
import yaml
from decimal import Decimal
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global tracking
audit_data = {
    'signals': [],
    'orders': [],
    'fills': [],
    'positions': {},
    'portfolio_updates': [],
    'initial_capital': 100000,
    'bars_processed': 0,
    'bar_data': []
}

# Patch to track bar data
from src.execution.containers_pipeline import DataContainer
original_stream_data = DataContainer._stream_data

async def audit_stream_data(self):
    """Track bar data."""
    # Patch to count bars
    original_publish = self.event_bus.publish
    
    def audit_publish(event):
        if event.event_type.name == 'BAR':
            audit_data['bars_processed'] += 1
            bar_info = {
                'bar_num': audit_data['bars_processed'],
                'timestamp': event.payload['timestamp'],
                'symbol': event.payload['symbol'],
                'price': event.payload['data'].get('close', 0)
            }
            audit_data['bar_data'].append(bar_info)
            if audit_data['bars_processed'] <= 50:  # Log first 50 bars
                logger.info(f"BAR {audit_data['bars_processed']:3d}: {bar_info['timestamp']} - SPY @ ${bar_info['price']:.4f}")
        original_publish(event)
    
    self.event_bus.publish = audit_publish
    return await original_stream_data(self)

DataContainer._stream_data = audit_stream_data

# Patch to track signals
from src.execution.containers_pipeline import StrategyContainer
original_emit_signals = StrategyContainer._emit_signals

async def audit_emit_signals(self, signals, timestamp, market_data):
    """Track all signals."""
    for signal in signals:
        signal_info = {
            'timestamp': timestamp,
            'symbol': signal.symbol,
            'side': str(signal.side),
            'strategy': signal.strategy_id,
            'strength': float(signal.strength)
        }
        audit_data['signals'].append(signal_info)
        logger.info(f"SIGNAL: {signal_info['strategy']} - {signal_info['side']} {signal_info['symbol']} @ {timestamp}")
    
    return await original_emit_signals(self, signals, timestamp, market_data)

StrategyContainer._emit_signals = audit_emit_signals

# Patch to track orders
from src.execution.containers_pipeline import RiskContainer
original_handle_signal = RiskContainer._handle_signal_event

async def audit_handle_signal(self, event):
    """Track order creation."""
    # Store current portfolio state before processing
    portfolio_state = self._get_portfolio_state()
    cash_before = float(portfolio_state.get_cash_balance())
    positions_before = {sym: float(pos.quantity) for sym, pos in portfolio_state.get_all_positions().items()}
    
    # Call original
    result = await original_handle_signal(self, event)
    
    # Log what happened
    logger.info(f"RISK CHECK: Cash=${cash_before:.2f}, Positions={positions_before}")
    
    return result

RiskContainer._handle_signal_event = audit_handle_signal

# Track orders created
original_create_order = RiskContainer._create_order_from_signal

def audit_create_order(self, signal, market_data):
    """Track order creation."""
    order = original_create_order(self, signal, market_data)
    
    if order and order.quantity > 0:
        symbol = signal.symbol if hasattr(signal, 'symbol') else signal.get('symbol', 'UNKNOWN')
        # Get market price
        price = None
        if symbol in market_data:
            price = market_data[symbol].get('close', 0)
        
        order_info = {
            'order_id': order.order_id,
            'symbol': symbol,
            'side': str(order.side),
            'quantity': order.quantity,
            'market_price': price,
            'timestamp': datetime.now()
        }
        audit_data['orders'].append(order_info)
        logger.info(f"ORDER: {order_info['side']} {order_info['quantity']} {order_info['symbol']} @ market (price=${price:.4f})")
    
    return order

RiskContainer._create_order_from_signal = audit_create_order

# Track fills
from src.execution.containers_pipeline import PortfolioContainer
original_handle_fill = PortfolioContainer._handle_fill_event

async def audit_handle_fill(self, event):
    """Track fills and portfolio updates."""
    fill = event.payload.get('fill')
    if fill:
        fill_info = {
            'symbol': fill.symbol,
            'side': str(fill.side),
            'quantity': float(fill.quantity),
            'price': float(fill.price),
            'commission': float(getattr(fill, 'commission', 0)),
            'timestamp': fill.executed_at
        }
        audit_data['fills'].append(fill_info)
        
        # Calculate trade value
        trade_value = fill_info['quantity'] * fill_info['price']
        logger.info(f"FILL: {fill_info['side']} {fill_info['quantity']} {fill_info['symbol']} @ ${fill_info['price']:.4f} = ${trade_value:.2f} (commission=${fill_info['commission']:.2f})")
    
    # Call original
    result = await original_handle_fill(self, event)
    
    # Log portfolio state after fill
    if self.portfolio_state:
        cash = float(self.portfolio_state.get_cash_balance())
        total_value = float(self.portfolio_state.get_total_value())
        positions = self.portfolio_state.get_all_positions()
        
        portfolio_update = {
            'timestamp': datetime.now(),
            'cash': cash,
            'total_value': total_value,
            'positions': {sym: float(pos.quantity) for sym, pos in positions.items()}
        }
        audit_data['portfolio_updates'].append(portfolio_update)
        
        logger.info(f"PORTFOLIO: Cash=${cash:.2f}, Total=${total_value:.2f}, Positions={len(positions)}")
        for sym, pos in positions.items():
            logger.info(f"  POSITION: {sym} = {pos.quantity} shares")
    
    return result

PortfolioContainer._handle_fill_event = audit_handle_fill

async def run_audit():
    """Run the audit."""
    # Load config
    with open('config/multi_strategy_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("TRADE AUDIT - 50 BARS")
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
    
    # Execute workflow
    try:
        from src.core.coordinator.coordinator import ExecutionMode
        
        result = await coordinator.execute_workflow(
            workflow_config, 
            execution_mode=ExecutionMode.COMPOSABLE
        )
        
        # Final audit summary
        logger.info("\n" + "=" * 80)
        logger.info("AUDIT SUMMARY")
        logger.info("=" * 80)
        
        # Bar summary
        logger.info(f"\nBars processed: {audit_data['bars_processed']}")
        if audit_data['bar_data']:
            first_bar = audit_data['bar_data'][0]
            last_bar = audit_data['bar_data'][-1]
            logger.info(f"Period: {first_bar['timestamp']} to {last_bar['timestamp']}")
            logger.info(f"Price range: ${first_bar['price']:.4f} to ${last_bar['price']:.4f}")
        
        # Signal summary
        logger.info(f"\nSignals generated: {len(audit_data['signals'])}")
        momentum_signals = len([s for s in audit_data['signals'] if 'momentum' in s['strategy']])
        reversion_signals = len([s for s in audit_data['signals'] if 'reversion' in s['strategy']])
        logger.info(f"  Momentum: {momentum_signals}")
        logger.info(f"  Mean Reversion: {reversion_signals}")
        
        # Order summary
        logger.info(f"\nOrders created: {len(audit_data['orders'])}")
        
        # Fill summary
        logger.info(f"\nFills executed: {len(audit_data['fills'])}")
        
        # Detailed fill audit
        if audit_data['fills']:
            logger.info("\nDETAILED FILL AUDIT:")
            total_buy_value = 0
            total_sell_value = 0
            total_commission = 0
            
            for i, fill in enumerate(audit_data['fills'], 1):
                trade_value = fill['quantity'] * fill['price']
                if 'BUY' in fill['side']:
                    total_buy_value += trade_value
                else:
                    total_sell_value += trade_value
                total_commission += fill['commission']
                
                logger.info(f"\nFill #{i}:")
                logger.info(f"  Side: {fill['side']}")
                logger.info(f"  Quantity: {fill['quantity']}")
                logger.info(f"  Price: ${fill['price']:.4f}")
                logger.info(f"  Value: ${trade_value:.2f}")
                logger.info(f"  Commission: ${fill['commission']:.2f}")
            
            logger.info(f"\nTRADE SUMMARY:")
            logger.info(f"  Total Buy Value: ${total_buy_value:.2f}")
            logger.info(f"  Total Sell Value: ${total_sell_value:.2f}")
            logger.info(f"  Total Commission: ${total_commission:.2f}")
            logger.info(f"  Net Cash Flow: ${total_sell_value - total_buy_value - total_commission:.2f}")
        
        # Portfolio validation
        logger.info("\nPORTFOLIO VALIDATION:")
        initial = audit_data['initial_capital']
        logger.info(f"  Initial Capital: ${initial:.2f}")
        
        if audit_data['portfolio_updates']:
            final_update = audit_data['portfolio_updates'][-1]
            final_cash = final_update['cash']
            final_total = final_update['total_value']
            final_positions = final_update['positions']
            
            logger.info(f"  Final Cash: ${final_cash:.2f}")
            logger.info(f"  Final Total: ${final_total:.2f}")
            logger.info(f"  Final Positions: {final_positions}")
            
            # Manual calculation
            cash_change = sum(f['quantity'] * f['price'] * (-1 if 'BUY' in f['side'] else 1) for f in audit_data['fills'])
            commission_paid = sum(f['commission'] for f in audit_data['fills'])
            expected_cash = initial + cash_change - commission_paid
            
            logger.info(f"\nMANUAL VALIDATION:")
            logger.info(f"  Starting Cash: ${initial:.2f}")
            logger.info(f"  + Cash from trades: ${cash_change:.2f}")
            logger.info(f"  - Commissions: ${commission_paid:.2f}")
            logger.info(f"  = Expected Cash: ${expected_cash:.2f}")
            logger.info(f"  Actual Cash: ${final_cash:.2f}")
            logger.info(f"  Difference: ${final_cash - expected_cash:.2f}")
            
            # Return calculation
            returns = ((final_total - initial) / initial) * 100
            logger.info(f"\nRETURNS:")
            logger.info(f"  P&L: ${final_total - initial:.2f}")
            logger.info(f"  Return: {returns:.4f}%")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_audit())